from copy import deepcopy
from typing import Any
from uuid import uuid4

from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt

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
from app.services.knowledge_qa import KnowledgeGroundedQAService
from app.services.knowledge_retrieval import KnowledgeRetrievalService
from app.services.llm import OpenAICompatibleLLM
from app.services.user_memory import UserMemoryManager


class AgentOrchestrator:
    def __init__(
        self,
        repository: SQLiteRepository,
        llm: OpenAICompatibleLLM,
        checkpointer: Any,
        user_memory: UserMemoryManager,
        tool_registry: dict[tuple[str, str], Any],
        knowledge_retrieval_service: KnowledgeRetrievalService,
        knowledge_retrieval_tool: Any,
        knowledge_qa: KnowledgeGroundedQAService,
    ) -> None:
        self.repository = repository
        self.llm = llm
        self.user_memory = user_memory
        self.tools = tool_registry
        self.knowledge_retrieval_service = knowledge_retrieval_service
        self.knowledge_retrieval_tool = knowledge_retrieval_tool
        self.knowledge_qa = knowledge_qa
        # Shared async checkpointer opened by the app lifespan.
        self.checkpointer = checkpointer

        self.graph = self._build_graph()

    def close(self) -> None:
        pass  # checkpointer lifecycle managed by lifespan async context

    async def astream_chat(
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
    ):
        """Stream chat via graph.astream(). Yields SSE-ready dicts.

        Chunk types yielded:
          {"type": "token", "content": str}  — LLM token
          {"type": "node", "node": str, "data": dict}  — node update
          {"type": "interrupt", "data": dict}  — approval required
          {"type": "done", "run_id": str, "status": str, "reply": str,
           "intent": str, "route": str, "requires_confirmation": bool,
           "approval_status": str|None, "execution_plan": dict|None,
           "pending_actions": list}  — final summary
        """
        initial_input = {
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
        }
        config = {"configurable": {"thread_id": run_id}}
        final_state: dict[str, Any] = {}
        async for chunk in self.graph.astream(
            initial_input,
            config=config,
            stream_mode=["messages", "updates"],
        ):
            chunk_type = chunk[0] if isinstance(chunk, tuple) else chunk.get("type", "")
            chunk_data = chunk[1] if isinstance(chunk, tuple) else chunk.get("data", {})

            if chunk_type == "messages":
                msg, _metadata = chunk_data if isinstance(chunk_data, tuple) else (chunk_data, {})
                token = getattr(msg, "content", None)
                if token:
                    yield {"type": "token", "content": token}

            elif chunk_type == "updates":
                data: dict[str, Any] = chunk_data if isinstance(chunk_data, dict) else {}
                # Detect interrupt
                if "__interrupt__" in data:
                    interrupt_list = data["__interrupt__"]
                    interrupt_value = interrupt_list[0].value if interrupt_list else {}
                    yield {"type": "interrupt", "data": interrupt_value}
                else:
                    for node_name, state_diff in data.items():
                        if isinstance(state_diff, dict):
                            final_state.update(state_diff)
                        yield {"type": "node", "node": node_name, "data": state_diff if isinstance(state_diff, dict) else {}}

        yield {
            "type": "done",
            "run_id": run_id,
            "status": final_state.get("status", "completed"),
            "reply": final_state.get("response", ""),
            "intent": final_state.get("intent"),
            "route": final_state.get("route"),
            "requires_confirmation": final_state.get("requires_confirmation", False),
            "approval_status": final_state.get("approval_status"),
            "execution_plan": final_state.get("execution_plan"),
            "pending_actions": final_state.get("pending_actions", []),
        }

    async def astream_resume(
        self,
        run_id: str,
        approved: bool,
    ):
        """Resume an interrupted run and stream the result. Same chunk types as astream_chat."""
        config = {"configurable": {"thread_id": run_id}}
        final_state: dict[str, Any] = {}
        async for chunk in self.graph.astream(
            Command(resume={"approved": approved}),
            config=config,
            stream_mode=["messages", "updates"],
        ):
            chunk_type = chunk[0] if isinstance(chunk, tuple) else chunk.get("type", "")
            chunk_data = chunk[1] if isinstance(chunk, tuple) else chunk.get("data", {})

            if chunk_type == "messages":
                msg, _metadata = chunk_data if isinstance(chunk_data, tuple) else (chunk_data, {})
                token = getattr(msg, "content", None)
                if token:
                    yield {"type": "token", "content": token}

            elif chunk_type == "updates":
                data = chunk_data if isinstance(chunk_data, dict) else {}
                for node_name, state_diff in data.items():
                    if isinstance(state_diff, dict):
                        final_state.update(state_diff)
                    yield {"type": "node", "node": node_name, "data": state_diff if isinstance(state_diff, dict) else {}}

        yield {
            "type": "done",
            "run_id": run_id,
            "status": final_state.get("status", "completed"),
            "reply": final_state.get("response", ""),
            "intent": final_state.get("intent"),
            "route": final_state.get("route"),
            "requires_confirmation": False,
            "approval_status": final_state.get("approval_status"),
            "execution_plan": final_state.get("execution_plan"),
            "pending_actions": final_state.get("pending_actions", []),
        }

    def _build_graph(self):
        builder = StateGraph(AgentState)
        builder.add_node("load_context", self._load_context)
        builder.add_node("classify_intent", self._classify_intent)
        builder.add_node("retrieve_knowledge", self._retrieve_knowledge)
        builder.add_node("plan_or_answer", self._plan_or_answer)
        builder.add_node("approval_gate", self._approval_gate)
        builder.add_node("execute_tools", self._execute_tools)
        builder.add_node("finalize_run", self._finalize_run)

        builder.add_edge(START, "load_context")
        builder.add_edge("load_context", "classify_intent")
        builder.add_conditional_edges(
            "classify_intent",
            self._route_after_classification,
            {
                "retrieve_knowledge": "retrieve_knowledge",
                "plan_or_answer": "plan_or_answer",
            },
        )
        builder.add_edge("retrieve_knowledge", "plan_or_answer")
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

    def _route_after_plan(self, state: AgentState) -> str:
        if state.get("requires_confirmation"):
            return "approval_gate"
        return "finalize_run"

    def _route_after_classification(self, state: AgentState) -> str:
        if state.get("intent") == "knowledge_query":
            return "retrieve_knowledge"
        return "plan_or_answer"

    def _route_after_approval(self, state: AgentState) -> str:
        if state.get("approval_status") == "approved":
            return "execute_tools"
        return "finalize_run"

    def _load_context(self, state: AgentState) -> dict[str, Any]:
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
        return {}

    def _classify_intent(self, state: AgentState) -> dict[str, Any]:
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
        return {"intent": intent, "route": route}

    def _retrieve_knowledge(self, state: AgentState) -> dict[str, Any]:
        tool_message = self.knowledge_retrieval_tool.invoke(
            {
                "type": "tool_call",
                "name": "knowledge_retrieval",
                "id": f"knowledge-retrieval-{uuid4()}",
                "args": {
                    "message": state["current_message"],
                    "route": state.get("route"),
                    "recent_context": state.get("recent_context", {}),
                    "top_k": 5,
                },
            }
        )
        artifact = tool_message.artifact if isinstance(getattr(tool_message, "artifact", None), dict) else {}
        retrieval_result = {
            "query": artifact.get("query", state["current_message"]),
            "route": artifact.get("route", state.get("route")),
            "resolved_scope": artifact.get("resolved_scope", {}),
            "total_hits": artifact.get("total_hits", 0),
            "hits": artifact.get("hits", []),
            "serialized_context": str(tool_message.content),
            "top_sources": artifact.get("top_sources", []),
        }
        output_summary = (
            f"retrieved {retrieval_result['total_hits']} hit(s); "
            f"sources={', '.join(retrieval_result.get('top_sources', [])) or 'none'}"
        )
        self.repository.upsert_run_step(
            state["run_id"],
            "knowledge_retrieval",
            "knowledge_retrieval",
            "completed",
            input_summary=state["current_message"],
            output_summary=output_summary,
        )
        return {"retrieval_result": retrieval_result}

    def _plan_or_answer(self, state: AgentState) -> dict[str, Any]:
        intent = state["intent"]
        route = state["route"]

        if route in {"import_request", "retry_request"}:
            execution_plan = self._build_execution_plan(route, state["current_message"])
            pending_actions = self._pending_actions_from_execution_plan(execution_plan)
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
            return {
                "pending_actions": pending_actions,
                "execution_plan": execution_plan,
                "requires_confirmation": True,
                "status": "awaiting_confirmation",
                "response": response,
            }

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
            retrieval_result = state.get("retrieval_result") or {}
            response = self.knowledge_qa.answer(
                question=state["current_message"],
                retrieval_result=retrieval_result,
            )
        elif route == "video_knowledge_query":
            retrieval_result = state.get("retrieval_result") or {}
            response = self.knowledge_qa.answer(
                question=state["current_message"],
                retrieval_result=retrieval_result,
            )
        else:
            extra_system_messages = []
            if state.get("user_memory_context"):
                extra_system_messages.append(str(state["user_memory_context"]))
            lc_messages = self.llm._build_lc_messages(
                state.get("messages", []),
                extra_system_messages=extra_system_messages,
            )
            try:
                llm_response = self.llm.get_langchain_llm().invoke(lc_messages)
                response = llm_response.content
            except Exception as llm_exc:
                response = f"[LLM unavailable: {llm_exc}]"

        self.repository.upsert_run_step(
            state["run_id"],
            "plan_or_answer",
            "plan_or_answer",
            "completed",
            input_summary=intent,
            output_summary="generated direct response",
        )
        return {
            "pending_actions": [],
            "execution_plan": None,
            "requires_confirmation": False,
            "status": "completed",
            "retrieval_result": state.get("retrieval_result"),
            "response": response,
        }

    async def _approval_gate(self, state: AgentState) -> dict[str, Any]:
        execution_plan = state.get("execution_plan")
        pending_actions = self._pending_actions_from_execution_plan(execution_plan)
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
        return {
            "approval_status": approval_status,
            "status": "running" if approved else "cancelled",
            "response": reply,
            "execution_plan": updated_execution_plan,
            "pending_actions": pending_actions,
        }

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
            tool_responses.append(response)

        final_response = "\n".join(tool_responses) if tool_responses else state.get("response", "")
        return {
            "status": "completed",
            "response": final_response,
            "execution_plan": current_plan,
            "pending_actions": pending_actions,
        }

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
        return {
            "status": status,
            "execution_plan": state.get("execution_plan"),
            "pending_actions": state.get("pending_actions", []),
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
                "action": "execute_import",
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
