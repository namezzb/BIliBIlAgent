from pathlib import Path
from typing import Any

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt

from app.agent.events import aggregate_chat_response
from app.agent.types import AgentState, IntentType, PendingAction
from app.db.repository import SQLiteRepository
from app.services.llm import OpenAICompatibleLLM


class AgentOrchestrator:
    def __init__(
        self,
        repository: SQLiteRepository,
        llm: OpenAICompatibleLLM,
        checkpoint_db_path: Path,
    ) -> None:
        self.repository = repository
        self.llm = llm
        self._checkpointer_cm = SqliteSaver.from_conn_string(str(checkpoint_db_path))
        self.checkpointer = self._checkpointer_cm.__enter__()
        self.graph = self._build_graph()

    def close(self) -> None:
        self._checkpointer_cm.__exit__(None, None, None)

    def invoke_chat(
        self,
        *,
        session_id: str,
        run_id: str,
        message: str,
        messages: list[dict[str, str]],
    ) -> dict[str, Any]:
        self._emit_event(
            run_id,
            "run_started",
            {
                "session_id": session_id,
                "message": message,
            },
        )
        self.graph.invoke(
            {
                "session_id": session_id,
                "run_id": run_id,
                "current_message": message,
                "messages": messages,
                "status": "running",
                "requires_confirmation": False,
                "pending_actions": [],
            },
            config={"configurable": {"thread_id": run_id}},
        )
        return self._build_sync_response(run_id, session_id)

    def resume_run(self, run_id: str, approved: bool) -> dict[str, Any]:
        self.graph.invoke(
            Command(resume={"approved": approved}),
            config={"configurable": {"thread_id": run_id}},
        )
        run = self.repository.get_run(run_id)
        session_id = run["session_id"] if run is not None else ""
        return self._build_sync_response(run_id, session_id)

    def _build_graph(self):
        builder = StateGraph(AgentState)
        builder.add_node("load_context", self._load_context)
        builder.add_node("classify_intent", self._classify_intent)
        builder.add_node("plan_or_answer", self._plan_or_answer)
        builder.add_node("approval_gate", self._approval_gate)
        builder.add_node("execute_placeholder", self._execute_placeholder)
        builder.add_node("finalize_run", self._finalize_run)

        builder.add_edge(START, "load_context")
        builder.add_edge("load_context", "classify_intent")
        builder.add_edge("classify_intent", "plan_or_answer")
        builder.add_conditional_edges(
            "plan_or_answer",
            self._route_after_plan,
            {
                "approval_gate": "approval_gate",
                "finalize_run": "finalize_run",
            },
        )
        builder.add_conditional_edges(
            "approval_gate",
            self._route_after_approval,
            {
                "execute_placeholder": "execute_placeholder",
                "finalize_run": "finalize_run",
            },
        )
        builder.add_edge("execute_placeholder", "finalize_run")
        builder.add_edge("finalize_run", END)

        return builder.compile(checkpointer=self.checkpointer)

    def _build_sync_response(self, run_id: str, session_id: str) -> dict[str, Any]:
        aggregated = aggregate_chat_response(run_id, self.repository.get_run_events(run_id))
        aggregated["session_id"] = session_id
        return aggregated

    def _emit_event(self, run_id: str, event_type: str, payload: dict[str, Any]) -> None:
        self.repository.append_run_event(run_id, event_type, payload)

    def _route_after_plan(self, state: AgentState) -> str:
        if state.get("requires_confirmation"):
            return "approval_gate"
        return "finalize_run"

    def _route_after_approval(self, state: AgentState) -> str:
        if state.get("approval_status") == "approved":
            return "execute_placeholder"
        return "finalize_run"

    def _load_context(self, state: AgentState) -> dict[str, Any]:
        self.repository.upsert_run_step(
            state["run_id"],
            "load_context",
            "load_context",
            "completed",
            input_summary="load session history",
            output_summary=f"loaded {len(state.get('messages', []))} messages",
        )
        self._emit_event(
            state["run_id"],
            "context_loaded",
            {"message_count": len(state.get("messages", []))},
        )
        return {}

    def _classify_intent(self, state: AgentState) -> dict[str, Any]:
        intent = self._detect_intent(state["current_message"])
        self.repository.upsert_run_step(
            state["run_id"],
            "classify_intent",
            "classify_intent",
            "completed",
            input_summary=state["current_message"],
            output_summary=intent,
        )
        self._emit_event(
            state["run_id"],
            "intent_classified",
            {
                "intent": intent,
                "message": state["current_message"],
            },
        )
        return {"intent": intent}

    def _plan_or_answer(self, state: AgentState) -> dict[str, Any]:
        intent = state["intent"]
        if intent == "tool_request":
            pending_actions = self._build_pending_actions(state["current_message"])
            response = (
                "This request needs confirmation before execution. Review the planned "
                "actions and call the confirmation endpoint to continue."
            )
            self.repository.upsert_run_step(
                state["run_id"],
                "plan_or_answer",
                "plan_or_answer",
                "completed",
                input_summary=intent,
                output_summary="prepared execution plan",
            )
            self._emit_event(
                state["run_id"],
                "response_prepared",
                {
                    "intent": intent,
                    "reply": response,
                    "pending_actions": pending_actions,
                },
            )
            return {
                "pending_actions": pending_actions,
                "requires_confirmation": True,
                "status": "awaiting_confirmation",
                "response": response,
            }

        if intent == "knowledge_query":
            response = (
                "Knowledge retrieval is recognized but not connected yet. "
                "The orchestration layer is ready for a future retrieval tool."
            )
        else:
            response = self.llm.chat(state.get("messages", []))

        self.repository.upsert_run_step(
            state["run_id"],
            "plan_or_answer",
            "plan_or_answer",
            "completed",
            input_summary=intent,
            output_summary="generated direct response",
        )
        self._emit_event(
            state["run_id"],
            "response_prepared",
            {
                "intent": intent,
                "reply": response,
                "pending_actions": [],
            },
        )
        return {
            "requires_confirmation": False,
            "status": "completed",
            "response": response,
        }

    def _approval_gate(self, state: AgentState) -> dict[str, Any]:
        payload = {
            "run_id": state["run_id"],
            "question": "Approve execution of the planned tool actions?",
            "pending_actions": state.get("pending_actions", []),
        }
        self.repository.upsert_run_step(
            state["run_id"],
            "approval_pending",
            "approval_gate",
            "awaiting_confirmation",
            input_summary="user approval required",
            output_summary="waiting for confirmation",
        )
        self._emit_event(
            state["run_id"],
            "confirmation_required",
            {
                "question": payload["question"],
                "pending_actions": payload["pending_actions"],
                "reply": state.get("response", ""),
            },
        )
        decision = interrupt(payload)
        approved = self._is_approved(decision)
        approval_status = "approved" if approved else "rejected"
        self.repository.upsert_run_step(
            state["run_id"],
            "approval_decision",
            "approval_gate",
            "completed",
            input_summary="received confirmation decision",
            output_summary=approval_status,
        )
        self._emit_event(
            state["run_id"],
            "confirmation_resolved",
            {
                "approved": approved,
                "reply": (
                    state.get("response", "")
                    if approved
                    else "Execution was cancelled. No side-effectful tool was run."
                ),
            },
        )
        return {
            "approval_status": approval_status,
            "status": "running" if approved else "cancelled",
            "response": (
                state.get("response", "")
                if approved
                else "Execution was cancelled. No side-effectful tool was run."
            ),
        }

    def _execute_placeholder(self, state: AgentState) -> dict[str, Any]:
        response = (
            "Execution was approved. The tool pipeline placeholder ran successfully, "
            "but real Bilibili import tools are not wired in yet."
        )
        self._emit_event(
            state["run_id"],
            "tool_execution_started",
            {"tool": "bilibili_import", "action": "prepare_import_plan"},
        )
        self.repository.upsert_run_step(
            state["run_id"],
            "execute_placeholder",
            "execute_placeholder",
            "completed",
            input_summary="approved tool request",
            output_summary="placeholder execution finished",
        )
        self._emit_event(
            state["run_id"],
            "tool_execution_finished",
            {
                "tool": "bilibili_import",
                "action": "prepare_import_plan",
                "reply": response,
            },
        )
        return {"status": "completed", "response": response}

    def _finalize_run(self, state: AgentState) -> dict[str, Any]:
        status = state.get("status", "completed")
        self.repository.upsert_run_step(
            state["run_id"],
            "finalize_run",
            "finalize_run",
            "completed",
            input_summary="finalize run state",
            output_summary=status,
        )
        event_type = "run_failed" if status == "failed" else "run_completed"
        self._emit_event(
            state["run_id"],
            event_type,
            {
                "status": status,
                "reply": state.get("response", ""),
            },
        )
        return {"status": status}

    def _detect_intent(self, message: str) -> IntentType:
        lowered = message.lower()
        tool_keywords = ("导入", "同步", "重试", "拉取", "import", "retry", "sync")
        knowledge_keywords = (
            "收藏夹",
            "视频",
            "分p",
            "讲了什么",
            "内容",
            "知识库",
            "字幕",
            "qa",
        )

        if any(keyword in lowered for keyword in tool_keywords):
            return "tool_request"
        if any(keyword in lowered for keyword in knowledge_keywords):
            return "knowledge_query"
        return "general_chat"

    def _build_pending_actions(self, message: str) -> list[PendingAction]:
        return [
            {
                "tool": "bilibili_import",
                "action": "prepare_import_plan",
                "target": "favorite-folder-ingestion",
                "description": f"Prepare a Bilibili import task for: {message}",
            }
        ]

    def _is_approved(self, decision: Any) -> bool:
        if isinstance(decision, bool):
            return decision
        if isinstance(decision, dict):
            return bool(decision.get("approved"))
        return False
