from copy import deepcopy
from pathlib import Path
from typing import Any

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt

from app.agent.events import aggregate_chat_response
from app.agent.tools import TOOL_REGISTRY
from app.agent.types import (
    AgentState,
    ExecutionPlan,
    ExecutionStep,
    IntentType,
    PendingAction,
    PlannedToolCall,
    RouteType,
)
from app.db.repository import SQLiteRepository
from app.services.llm import OpenAICompatibleLLM
from app.services.runtime_audit import LangSmithRuntimeAudit
from app.services.user_memory import UserMemoryManager


class AgentOrchestrator:
    def __init__(
        self,
        repository: SQLiteRepository,
        llm: OpenAICompatibleLLM,
        checkpoint_db_path: Path,
        user_memory: UserMemoryManager,
        runtime_audit: LangSmithRuntimeAudit,
    ) -> None:
        self.repository = repository
        self.llm = llm
        self.user_memory = user_memory
        self.runtime_audit = runtime_audit
        self.tools = TOOL_REGISTRY
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
        user_id: str | None = None,
        session_summary: str | None = None,
        recent_context: dict[str, object] | None = None,
        user_memory_context: str | None = None,
    ) -> dict[str, Any]:
        request_metadata = {
            "thread_id": run_id,
            "run_id": run_id,
            "session_id": session_id,
            "user_id": user_id,
            "environment": self.runtime_audit.environment,
            "app_name": self.runtime_audit.app_name,
            "operation": "chat",
        }
        request_tags = ["agent", "chat", self.runtime_audit.environment]
        with self.runtime_audit.trace_request(
            name="agent.chat",
            inputs={
                "session_id": session_id,
                "run_id": run_id,
                "message": message,
                "message_count": len(messages),
            },
            metadata=request_metadata,
            tags=request_tags,
        ) as trace_run:
            self._emit_event(
                run_id,
                "run_started",
                {
                    "session_id": session_id,
                    "user_id": user_id,
                    "message": message,
                },
            )
            self.graph.invoke(
                {
                    "session_id": session_id,
                    "user_id": user_id,
                    "run_id": run_id,
                    "current_message": message,
                    "messages": messages,
                    "session_summary": session_summary,
                    "recent_context": recent_context or {},
                    "user_memory_context": user_memory_context,
                    "status": "running",
                    "requires_confirmation": False,
                    "pending_actions": [],
                    "execution_plan": None,
                },
                config={"configurable": {"thread_id": run_id}},
            )
            result = self._build_sync_response(run_id, session_id)
            trace_run.add_metadata(
                {
                    "intent": result.get("intent"),
                    "route": result.get("route"),
                    "status": result.get("status"),
                    "approval_status": result.get("approval_status"),
                    "execution_goal": self._execution_goal(result.get("execution_plan")),
                    "planned_tools": self._planned_tool_names(result.get("execution_plan")),
                }
            )
            trace_run.end(outputs=self.runtime_audit.sanitize_payload(result))
            result.update(self.runtime_audit.build_reference(run_id=run_id, trace_run=trace_run))
            return result

    def resume_run(self, run_id: str, approved: bool) -> dict[str, Any]:
        run = self.repository.get_run(run_id)
        session_id = run["session_id"] if run is not None else ""
        user_id = None
        if run is not None:
            session = self.repository.get_session(session_id)
            user_id = session["user_id"] if session is not None else None

        with self.runtime_audit.trace_request(
            name="agent.confirm_run",
            inputs={"run_id": run_id, "approved": approved},
            metadata={
                "thread_id": run_id,
                "run_id": run_id,
                "session_id": session_id,
                "user_id": user_id,
                "environment": self.runtime_audit.environment,
                "app_name": self.runtime_audit.app_name,
                "operation": "confirm",
            },
            tags=["agent", "confirm", self.runtime_audit.environment],
        ) as trace_run:
            self.graph.invoke(
                Command(resume={"approved": approved}),
                config={"configurable": {"thread_id": run_id}},
            )
            result = self._build_sync_response(run_id, session_id)
            trace_run.add_metadata(
                {
                    "intent": result.get("intent"),
                    "route": result.get("route"),
                    "status": result.get("status"),
                    "approval_status": result.get("approval_status"),
                    "execution_goal": self._execution_goal(result.get("execution_plan")),
                    "planned_tools": self._planned_tool_names(result.get("execution_plan")),
                }
            )
            trace_run.end(outputs=self.runtime_audit.sanitize_payload(result))
            result.update(self.runtime_audit.build_reference(run_id=run_id, trace_run=trace_run))
            return result

    def _build_graph(self):
        builder = StateGraph(AgentState)
        builder.add_node("load_context", self._load_context)
        builder.add_node("classify_intent", self._classify_intent)
        builder.add_node("plan_or_answer", self._plan_or_answer)
        builder.add_node("approval_gate", self._approval_gate)
        builder.add_node("execute_tools", self._execute_tools)
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
                "execute_tools": "execute_tools",
                "finalize_run": "finalize_run",
            },
        )
        builder.add_edge("execute_tools", "finalize_run")
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
            return "execute_tools"
        return "finalize_run"

    def _load_context(self, state: AgentState) -> dict[str, Any]:
        with self.runtime_audit.trace_span(
            name="agent.load_context",
            run_type="chain",
            inputs={
                "run_id": state["run_id"],
                "message_count": len(state.get("messages", [])),
                "recent_context": state.get("recent_context", {}),
            },
            metadata=self._span_metadata(state),
            tags=["agent", "context"],
        ) as trace_run:
            output_summary = (
                f"loaded {len(state.get('messages', []))} messages; "
                f"user_memory_present={bool(state.get('user_memory_context'))}"
            )
            self.repository.upsert_run_step(
                state["run_id"],
                "load_context",
                "load_context",
                "completed",
                input_summary="load session history",
                output_summary=output_summary,
            )
            payload = {
                "message_count": len(state.get("messages", [])),
                "summary_present": bool(state.get("session_summary")),
                "recent_context_available": bool(state.get("recent_context")),
                "user_memory_present": bool(state.get("user_memory_context")),
            }
            self._emit_event(state["run_id"], "context_loaded", payload)
            result: dict[str, Any] = {}
            trace_run.end(outputs={"event": payload, "result": result})
            return result

    def _classify_intent(self, state: AgentState) -> dict[str, Any]:
        with self.runtime_audit.trace_span(
            name="agent.classify_intent",
            run_type="chain",
            inputs={
                "run_id": state["run_id"],
                "message": state["current_message"],
            },
            metadata=self._span_metadata(state),
            tags=["agent", "routing"],
        ) as trace_run:
            route = self._detect_route(state["current_message"])
            intent = self._intent_from_route(route)
            self.repository.upsert_run_step(
                state["run_id"],
                "classify_intent",
                "classify_intent",
                "completed",
                input_summary=state["current_message"],
                output_summary=f"{route} -> {intent}",
            )
            payload = {
                "intent": intent,
                "route": route,
                "message": state["current_message"],
            }
            self._emit_event(state["run_id"], "intent_classified", payload)
            result = {"intent": intent, "route": route}
            trace_run.end(outputs={"event": payload, "result": result})
            return result

    def _plan_or_answer(self, state: AgentState) -> dict[str, Any]:
        with self.runtime_audit.trace_span(
            name="agent.plan_or_answer",
            run_type="chain",
            inputs={
                "run_id": state["run_id"],
                "intent": state["intent"],
                "route": state["route"],
                "message": state["current_message"],
            },
            metadata=self._span_metadata(state),
            tags=["agent", "planning"],
        ) as trace_run:
            intent = state["intent"]
            route = state["route"]

            if route in {"import_request", "retry_request"}:
                execution_plan = self._build_execution_plan(route, state["current_message"])
                pending_actions = self._pending_actions_from_execution_plan(execution_plan)
                response = (
                    "This request needs confirmation before execution. Review the planned "
                    "actions and call the confirmation endpoint to continue."
                )
                trace_run.add_metadata(
                    {
                        "execution_goal": execution_plan["goal"],
                        "planned_tools": self._planned_tool_names(execution_plan),
                    }
                )
                self.repository.upsert_run_step(
                    state["run_id"],
                    "plan_or_answer",
                    "plan_or_answer",
                    "completed",
                    input_summary=intent,
                    output_summary="prepared execution plan",
                )
                payload = {
                    "intent": intent,
                    "route": route,
                    "reply": response,
                    "execution_plan": execution_plan,
                    "pending_actions": pending_actions,
                }
                self._emit_event(state["run_id"], "response_prepared", payload)
                result = {
                    "pending_actions": pending_actions,
                    "execution_plan": execution_plan,
                    "requires_confirmation": True,
                    "status": "awaiting_confirmation",
                    "response": response,
                }
                trace_run.end(outputs={"event": payload, "result": result})
                return result

            if self.user_memory.is_chat_command(state["current_message"]):
                user_id = state.get("user_id")
                if not user_id:
                    response = "长期记忆操作需要显式传入 user_id。"
                else:
                    response = self.user_memory.apply_chat_command(
                        user_id,
                        state["current_message"],
                        state["run_id"],
                    )
            elif route == "favorite_knowledge_query":
                response = (
                    "Favorite-folder knowledge query is recognized, but the retrieval chain "
                    "is not connected yet."
                )
            elif route == "video_knowledge_query":
                response = (
                    "Single-video knowledge query is recognized, but the retrieval chain "
                    "is not connected yet."
                )
            else:
                extra_system_messages = []
                if state.get("user_memory_context"):
                    extra_system_messages.append(str(state["user_memory_context"]))
                response = self.llm.chat(
                    state.get("messages", []),
                    extra_system_messages=extra_system_messages,
                )

            self.repository.upsert_run_step(
                state["run_id"],
                "plan_or_answer",
                "plan_or_answer",
                "completed",
                input_summary=intent,
                output_summary="generated direct response",
            )
            payload = {
                "intent": intent,
                "route": route,
                "reply": response,
                "execution_plan": None,
                "pending_actions": [],
            }
            self._emit_event(state["run_id"], "response_prepared", payload)
            result = {
                "pending_actions": [],
                "execution_plan": None,
                "requires_confirmation": False,
                "status": "completed",
                "response": response,
            }
            trace_run.end(outputs={"event": payload, "result": result})
            return result

    def _approval_gate(self, state: AgentState) -> dict[str, Any]:
        execution_plan = state.get("execution_plan")
        pending_actions = self._pending_actions_from_execution_plan(execution_plan)
        with self.runtime_audit.trace_span(
            name="agent.approval_gate",
            run_type="chain",
            inputs={
                "run_id": state["run_id"],
                "route": state.get("route"),
                "execution_plan": execution_plan,
            },
            metadata=self._span_metadata(state),
            tags=["agent", "approval"],
        ) as trace_run:
            trace_run.add_metadata(
                {
                    "execution_goal": self._execution_goal(execution_plan),
                    "planned_tools": self._planned_tool_names(execution_plan),
                }
            )
            payload = {
                "run_id": state["run_id"],
                "question": "Approve execution of the planned tool actions?",
                "route": state.get("route"),
                "execution_plan": execution_plan,
                "pending_actions": pending_actions,
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
                    "route": state.get("route"),
                    "execution_plan": execution_plan,
                    "pending_actions": payload["pending_actions"],
                    "reply": state.get("response", ""),
                },
            )
            decision = interrupt(payload)
            approved = self._is_approved(decision)
            approval_status = "approved" if approved else "rejected"
            updated_execution_plan = self._mark_execution_plan_status(
                execution_plan,
                "approved" if approved else "cancelled",
            )
            self.repository.upsert_run_step(
                state["run_id"],
                "approval_decision",
                "approval_gate",
                "completed",
                input_summary="received confirmation decision",
                output_summary=approval_status,
            )

            reply = (
                state.get("response", "")
                if approved
                else "Execution was cancelled. No side-effectful tool was run."
            )
            payload = {
                "approved": approved,
                "route": state.get("route"),
                "reply": reply,
                "execution_plan": updated_execution_plan,
                "pending_actions": pending_actions,
            }
            self._emit_event(state["run_id"], "confirmation_resolved", payload)
            result = {
                "approval_status": approval_status,
                "status": "running" if approved else "cancelled",
                "response": reply,
                "execution_plan": updated_execution_plan,
                "pending_actions": pending_actions,
            }
            trace_run.end(outputs={"event": payload, "result": result})
            return result

    def _execute_tools(self, state: AgentState) -> dict[str, Any]:
        route = state.get("route")
        execution_plan = state.get("execution_plan")
        pending_actions = self._pending_actions_from_execution_plan(execution_plan)
        tool_calls = execution_plan.get("tool_calls", []) if execution_plan else []
        tool_responses: list[str] = []
        current_plan = execution_plan

        for index, tool_call in enumerate(tool_calls, start=1):
            tool_name = str(tool_call["tool"])
            action_name = str(tool_call["action"])
            tool = self._get_registered_tool(tool_name, action_name)
            step_key = f"execute_tool_{index}_{tool_name}"
            args = dict(tool_call.get("args", {}))
            args_summary = self._summarize_tool_args(args)

            with self.runtime_audit.trace_span(
                name=f"tool.{tool_name}.{action_name}",
                run_type="tool",
                inputs={
                    "run_id": state["run_id"],
                    "route": route,
                    "tool": tool_name,
                    "action": action_name,
                    "args": args,
                },
                metadata=self._span_metadata(state),
                tags=["agent", "tool", tool_name],
            ) as trace_run:
                trace_run.add_metadata(
                    {
                        "execution_goal": self._execution_goal(current_plan),
                        "planned_tools": self._planned_tool_names(current_plan),
                        "tool_target": tool_call.get("target"),
                    }
                )
                self._emit_event(
                    state["run_id"],
                    "tool_execution_started",
                    {
                        "route": route,
                        "tool": tool_name,
                        "action": action_name,
                        "target": tool_call.get("target"),
                        "args_summary": args_summary,
                        "execution_plan": current_plan,
                        "pending_actions": pending_actions,
                    },
                )
                response = str(tool.invoke(args))
                current_plan = self._mark_execution_step_completed(
                    current_plan,
                    tool=tool_name,
                    action=action_name,
                )
                self.repository.upsert_run_step(
                    state["run_id"],
                    step_key,
                    f"{tool_name}.{action_name}",
                    "completed",
                    input_summary=args_summary,
                    output_summary=response,
                )
                payload = {
                    "route": route,
                    "tool": tool_name,
                    "action": action_name,
                    "target": tool_call.get("target"),
                    "args_summary": args_summary,
                    "reply": response,
                    "execution_plan": current_plan,
                    "pending_actions": pending_actions,
                }
                self._emit_event(state["run_id"], "tool_execution_finished", payload)
                trace_run.end(outputs={"event": payload, "result": {"response": response}})
                tool_responses.append(response)

        final_response = "\n".join(tool_responses) if tool_responses else state.get("response", "")
        result = {
            "status": "completed",
            "response": final_response,
            "execution_plan": current_plan,
            "pending_actions": pending_actions,
        }
        return result

    def _finalize_run(self, state: AgentState) -> dict[str, Any]:
        with self.runtime_audit.trace_span(
            name="agent.finalize_run",
            run_type="chain",
            inputs={
                "run_id": state["run_id"],
                "status": state.get("status", "completed"),
                "route": state.get("route"),
            },
            metadata=self._span_metadata(state),
            tags=["agent", "finalize"],
        ) as trace_run:
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
            payload = {
                "status": status,
                "route": state.get("route"),
                "reply": state.get("response", ""),
                "execution_plan": state.get("execution_plan"),
                "pending_actions": state.get("pending_actions", []),
            }
            self._emit_event(state["run_id"], event_type, payload)
            result = {
                "status": status,
                "execution_plan": state.get("execution_plan"),
                "pending_actions": state.get("pending_actions", []),
            }
            trace_run.end(outputs={"event": payload, "result": result})
            return result

    def _span_metadata(self, state: AgentState) -> dict[str, Any]:
        return {
            "thread_id": state["run_id"],
            "run_id": state["run_id"],
            "session_id": state["session_id"],
            "user_id": state.get("user_id"),
            "route": state.get("route"),
            "intent": state.get("intent"),
        }

    def _detect_route(self, message: str) -> RouteType:
        lowered = message.lower()
        retry_keywords = ("重试", "retry", "重新执行", "重跑", "失败项")
        import_keywords = ("导入", "import", "同步", "sync", "导进知识库", "拉取")
        video_keywords = (
            "bv",
            "av",
            "分p",
            "这一期",
            "这个视频",
            "这期视频",
        )
        knowledge_keywords = (
            "讲了什么",
            "内容",
            "知识库",
            "字幕",
            "qa",
            "相关视频",
            "有哪些",
        )

        if any(keyword in lowered for keyword in retry_keywords):
            return "retry_request"
        if any(keyword in lowered for keyword in import_keywords):
            return "import_request"
        if "收藏夹" in lowered and any(keyword in lowered for keyword in knowledge_keywords):
            return "favorite_knowledge_query"
        if any(keyword in lowered for keyword in video_keywords) and any(
            keyword in lowered for keyword in knowledge_keywords + ("讲了什么",)
        ):
            return "video_knowledge_query"
        if any(keyword in lowered for keyword in ("bv", "av", "分p", "这个视频", "这期视频")):
            return "video_knowledge_query"
        return "general_chat"

    def _intent_from_route(self, route: RouteType) -> IntentType:
        if route in {"favorite_knowledge_query", "video_knowledge_query"}:
            return "knowledge_query"
        if route in {"import_request", "retry_request"}:
            return "tool_request"
        return "general_chat"

    def _build_execution_plan(self, route: RouteType, message: str) -> ExecutionPlan:
        if route == "retry_request":
            tool_call: PlannedToolCall = {
                "tool": "bilibili_retry",
                "action": "prepare_retry_plan",
                "target": "failed-import-items",
                "description": f"Retry failed ingestion items related to: {message}",
                "args": {
                    "request_message": message,
                    "target": "failed-import-items",
                },
                "side_effect": True,
            }
            goal = "Retry the failed Bilibili ingestion items requested by the user."
            summary = (
                "After approval, the agent will run the retry pipeline for previously "
                "failed ingestion items related to this request."
            )
        else:
            tool_call = {
                "tool": "bilibili_import",
                "action": "prepare_import_plan",
                "target": "favorite-folder-ingestion",
                "description": f"Import Bilibili favorite-folder content for: {message}",
                "args": {
                    "request_message": message,
                    "target": "favorite-folder-ingestion",
                },
                "side_effect": True,
            }
            goal = "Import the requested Bilibili favorite-folder content into the knowledge base."
            summary = (
                "After approval, the agent will run the import pipeline for the requested "
                "favorite-folder scope."
            )

        step: ExecutionStep = {
            "id": "tool_call_1",
            "title": f"Run {tool_call['tool']}.{tool_call['action']}",
            "description": str(tool_call["description"]),
            "tool": str(tool_call["tool"]),
            "action": str(tool_call["action"]),
            "status": "pending",
        }
        return {
            "goal": goal,
            "summary": summary,
            "steps": [step],
            "tool_calls": [tool_call],
        }

    def _pending_actions_from_execution_plan(
        self,
        execution_plan: ExecutionPlan | dict[str, Any] | None,
    ) -> list[PendingAction]:
        if not execution_plan:
            return []
        return [
            {
                "tool": str(tool_call["tool"]),
                "action": str(tool_call["action"]),
                "target": str(tool_call["target"]),
                "description": str(tool_call["description"]),
            }
            for tool_call in execution_plan.get("tool_calls", [])
        ]

    def _mark_execution_plan_status(
        self,
        execution_plan: ExecutionPlan | dict[str, Any] | None,
        status: str,
    ) -> ExecutionPlan | None:
        if not execution_plan:
            return None
        updated_plan = deepcopy(execution_plan)
        for step in updated_plan.get("steps", []):
            step["status"] = status
        return updated_plan

    def _mark_execution_step_completed(
        self,
        execution_plan: ExecutionPlan | dict[str, Any] | None,
        *,
        tool: str,
        action: str,
    ) -> ExecutionPlan | None:
        if not execution_plan:
            return None
        updated_plan = deepcopy(execution_plan)
        for step in updated_plan.get("steps", []):
            if step.get("tool") == tool and step.get("action") == action:
                step["status"] = "completed"
        return updated_plan

    def _get_registered_tool(self, tool: str, action: str):
        registered_tool = self.tools.get((tool, action))
        if registered_tool is None:
            raise RuntimeError(f"No registered tool for {tool}.{action}")
        return registered_tool

    def _execution_goal(self, execution_plan: ExecutionPlan | dict[str, Any] | None) -> str | None:
        if not execution_plan:
            return None
        goal = execution_plan.get("goal")
        return str(goal) if goal is not None else None

    def _planned_tool_names(
        self,
        execution_plan: ExecutionPlan | dict[str, Any] | None,
    ) -> list[str]:
        if not execution_plan:
            return []
        return [
            f"{tool_call['tool']}.{tool_call['action']}"
            for tool_call in execution_plan.get("tool_calls", [])
        ]

    def _summarize_tool_args(self, args: dict[str, Any]) -> str:
        if not args:
            return "no args"
        return ", ".join(f"{key}={value}" for key, value in args.items())

    def _is_approved(self, decision: Any) -> bool:
        if isinstance(decision, bool):
            return decision
        if isinstance(decision, dict):
            return bool(decision.get("approved"))
        return False
