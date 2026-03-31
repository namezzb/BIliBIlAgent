from copy import deepcopy
from typing import Any
from uuid import uuid4

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt

from app.agent.types import (
    AgentState,
    ExecutionPlan,
    ExecutionStep,
    PendingAction,
    PlannedToolCall,
    RouteDecision,
    RouteType,
    RetrievalScopeHint,
)
from app.db.repository import SQLiteRepository
from app.services.knowledge_qa import KnowledgeGroundedQAService
from app.services.knowledge_retrieval import KnowledgeRetrievalService
from app.services.llm import OpenAICompatibleLLM
from app.services.user_memory import UserMemoryManager

_ROUTER_SYSTEM_PROMPT = """\
You are the intent router for BIliBIlAgent. Classify the user message into exactly one route:

- general_chat          — casual conversation, greetings, or questions unrelated to Bilibili content
- favorite_knowledge_query — questions about what is in the user's favorite folder(s) or multiple videos
- video_knowledge_query — questions about the content of a specific video (mentions BV/AV ID, "这个视频", "这期", page number, etc.)
- import_request        — user wants to import / sync / add a favorite folder or videos into the knowledge base
- retry_request         — user wants to retry a previously failed import task

Rules:
1. If the message mentions importing AND querying, prefer import_request.
2. If unsure between favorite_knowledge_query and video_knowledge_query, prefer favorite_knowledge_query.
3. Default to general_chat when no specific route fits.
4. For mentioned_bvids: extract any BV/AV IDs found in the message (e.g. BV1xx411c7mD). Empty list if none.
5. For mentioned_video_titles: extract explicit video title strings the user refers to. Empty list if none.
6. For mentioned_folder_names: extract explicit favorite folder names the user refers to. Empty list if none.
7. Reply ONLY with the structured JSON — no extra text.
"""

_LLM_ROUTE_TO_GRAPH_ROUTE: dict[str, RouteType] = {
    "general_chat": "general_chat",
    "favorite_knowledge_query": "knowledge_query",
    "video_knowledge_query": "knowledge_query",
    "import_request": "plan_and_solve",
    "retry_request": "plan_and_solve",
}

_LLM_ROUTE_TO_SCOPE_HINT: dict[str, RetrievalScopeHint] = {
    "favorite_knowledge_query": "favorite_knowledge_query",
    "video_knowledge_query": "video_knowledge_query",
}


class AgentOrchestrator:
    def __init__(
        self,
        repository: SQLiteRepository,
        llm: OpenAICompatibleLLM,
        checkpointer: Any,
        user_memory: UserMemoryManager,
        tool_registry: dict[tuple[str, str], Any],
        knowledge_retrieval_service: KnowledgeRetrievalService,
        knowledge_qa: KnowledgeGroundedQAService,
    ) -> None:
        self.repository = repository
        self.llm = llm
        self.user_memory = user_memory
        self.tools = tool_registry
        self.knowledge_retrieval_service = knowledge_retrieval_service
        self.knowledge_qa = knowledge_qa
        self.checkpointer = checkpointer
        self._router_llm = llm.get_lc_chat_model().with_structured_output(RouteDecision)
        self.graph = self._build_graph()

    def close(self) -> None:
        pass

    # ------------------------------------------------------------------
    # Public streaming API
    # ------------------------------------------------------------------

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
        """Stream chat via graph.astream(). Yields SSE-ready dicts."""
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
            "retrieval_scope_hint": None,
            "mentioned_bvids": [],
            "mentioned_video_titles": [],
            "mentioned_folder_names": [],
        }
        config = {"configurable": {"thread_id": run_id}}
        final_state: dict[str, Any] = {}
        async for chunk in self.graph.astream(
            initial_input,
            config=config,
            stream_mode=["messages", "updates"],
            version="v2",
        ):
            # LangGraph v2 yields tuples: (stream_mode, data)
            mode, data = chunk
            if mode == "messages":
                msg, _metadata = data
                # Only stream tokens from content-generating nodes, not the router
                emitting_node = _metadata.get("langgraph_node", "") if isinstance(_metadata, dict) else ""
                token = getattr(msg, "content", None)
                if token and emitting_node not in ("router", "load_context"):
                    yield {"type": "token", "content": token}
            elif mode == "updates":
                update_data: dict[str, Any] = data
                if "__interrupt__" in update_data:
                    interrupt_list = update_data["__interrupt__"]
                    interrupt_value = interrupt_list[0].value if interrupt_list else {}
                    yield {"type": "interrupt", "data": interrupt_value}
                else:
                    for node_name, state_diff in update_data.items():
                        if isinstance(state_diff, dict):
                            final_state.update(state_diff)
                        yield {
                            "type": "node",
                            "node": node_name,
                            "data": state_diff if isinstance(state_diff, dict) else {},
                        }
        yield {
            "type": "done",
            "run_id": run_id,
            "status": final_state.get("status", "completed"),
            "reply": final_state.get("response", ""),
            "route": final_state.get("route"),
            "retrieval_result": final_state.get("retrieval_result"),
            "requires_confirmation": final_state.get("requires_confirmation", False),
            "approval_status": final_state.get("approval_status"),
            "execution_plan": final_state.get("execution_plan"),
            "pending_actions": final_state.get("pending_actions", []),
        }

    async def astream_resume(self, run_id: str, approved: bool):
        """Resume an interrupted run and stream the result."""
        config = {"configurable": {"thread_id": run_id}}
        final_state: dict[str, Any] = {}
        async for chunk in self.graph.astream(
            Command(resume={"approved": approved}),
            config=config,
            stream_mode=["messages", "updates"],
            version="v2",
        ):
            # LangGraph v2 yields tuples: (stream_mode, data)
            mode, data = chunk
            if mode == "messages":
                msg, _metadata = data
                emitting_node = _metadata.get("langgraph_node", "") if isinstance(_metadata, dict) else ""
                token = getattr(msg, "content", None)
                if token and emitting_node not in ("router", "load_context"):
                    yield {"type": "token", "content": token}
            elif mode == "updates":
                update_data: dict[str, Any] = data
                for node_name, state_diff in update_data.items():
                    if isinstance(state_diff, dict):
                        final_state.update(state_diff)
                    yield {
                        "type": "node",
                        "node": node_name,
                        "data": state_diff if isinstance(state_diff, dict) else {},
                    }
        yield {
            "type": "done",
            "run_id": run_id,
            "status": final_state.get("status", "completed"),
            "reply": final_state.get("response", ""),
            "route": final_state.get("route"),
            "requires_confirmation": False,
            "approval_status": final_state.get("approval_status"),
            "execution_plan": final_state.get("execution_plan"),
            "pending_actions": final_state.get("pending_actions", []),
        }

    # ------------------------------------------------------------------
    # Graph construction
    # ------------------------------------------------------------------

    def _build_graph(self):
        builder = StateGraph(AgentState)

        builder.add_node("load_context", self._load_context)
        builder.add_node("router", self._router)
        builder.add_node("general_chat", self._general_chat)
        builder.add_node("retrieve_knowledge", self._retrieve_knowledge)
        builder.add_node("knowledge_qa", self._knowledge_qa)
        builder.add_node("plan_and_solve", self._plan_and_solve)
        builder.add_node("approval_gate", self._approval_gate)
        builder.add_node("execute_tools", self._execute_tools)
        builder.add_node("finalize_run", self._finalize_run)

        builder.add_edge(START, "load_context")
        builder.add_edge("load_context", "router")

        builder.add_conditional_edges(
            "router",
            self._route_after_router,
            {
                "general_chat": "general_chat",
                "retrieve_knowledge": "retrieve_knowledge",
                "plan_and_solve": "plan_and_solve",
            },
        )

        builder.add_edge("retrieve_knowledge", "knowledge_qa")
        builder.add_edge("knowledge_qa", "finalize_run")
        builder.add_edge("general_chat", "finalize_run")

        builder.add_conditional_edges(
            "plan_and_solve",
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

    # ------------------------------------------------------------------
    # Routing helpers
    # ------------------------------------------------------------------

    def _route_after_router(self, state: AgentState) -> str:
        route = state.get("route", "general_chat")
        if route == "knowledge_query":
            return "retrieve_knowledge"
        if route == "plan_and_solve":
            return "plan_and_solve"
        return "general_chat"

    def _route_after_plan(self, state: AgentState) -> str:
        if state.get("requires_confirmation"):
            return "approval_gate"
        return "finalize_run"

    def _route_after_approval(self, state: AgentState) -> str:
        if state.get("approval_status") == "approved":
            return "execute_tools"
        return "finalize_run"

    # ------------------------------------------------------------------
    # Node implementations
    # ------------------------------------------------------------------

    def _load_context(self, state: AgentState) -> dict[str, Any]:
        output_summary = (
            f"loaded {len(state.get('messages', []))} messages; "
            f"user_memory_present={bool(state.get('user_memory_context'))}"
        )
        self.repository.upsert_run_step(
            state["run_id"], "load_context", "load_context", "completed",
            input_summary="load session history", output_summary=output_summary,
        )
        return {}

    def _router(self, state: AgentState) -> dict[str, Any]:
        """Classify message into 3-way RouteType + fine-grained retrieval_scope_hint."""
        decision = self._llm_detect_route_decision(state["current_message"])
        # Fallback fix: concept-style questions should prefer knowledge retrieval
        # when local knowledge already exists, even if LLM routed to general_chat.
        decision = self._coerce_route_for_knowledge_query(state["current_message"], decision)
        graph_route: RouteType = _LLM_ROUTE_TO_GRAPH_ROUTE[decision.route]
        scope_hint: RetrievalScopeHint | None = _LLM_ROUTE_TO_SCOPE_HINT.get(decision.route)
        self.repository.upsert_run_step(
            state["run_id"], "router", "router", "completed",
            input_summary=state["current_message"],
            output_summary=f"llm_route={decision.route} -> route={graph_route}",
        )
        return {
            "route": graph_route,
            "retrieval_scope_hint": scope_hint,
            # Pass LLM-identified entity references to scope resolution
            "mentioned_bvids": decision.mentioned_bvids,
            "mentioned_video_titles": decision.mentioned_video_titles,
            "mentioned_folder_names": decision.mentioned_folder_names,
        }

    def _general_chat(self, state: AgentState) -> dict[str, Any]:
        """Handle general conversation and user-memory commands."""
        if self.user_memory.is_chat_command(state["current_message"]):
            user_id = state.get("user_id")
            if not user_id:
                response = "长期记忆操作需要显式传入 user_id。"
            else:
                response = self.user_memory.apply_chat_command(
                    user_id, state["current_message"], state["run_id"],
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
            state["run_id"], "general_chat", "general_chat", "completed",
            input_summary=state["current_message"],
            output_summary="generated general response",
        )
        return {
            "response": response,
            "status": "completed",
            "requires_confirmation": False,
            "pending_actions": [],
            "execution_plan": None,
        }

    def _retrieve_knowledge(self, state: AgentState) -> dict[str, Any]:
        """Execute knowledge retrieval directly via service (no Tool wrapper)."""
        scope_hint = state.get("retrieval_scope_hint") or "general_knowledge_query"
        # Directly call the service — single source of truth, no artifact/content split
        retrieval_result = self.knowledge_retrieval_service.retrieve_for_question(
            message=state["current_message"],
            route=scope_hint,
            recent_context=state.get("recent_context", {}),
            top_k=5,
            mentioned_bvids=state.get("mentioned_bvids") or [],
            mentioned_video_titles=state.get("mentioned_video_titles") or [],
            mentioned_folder_names=state.get("mentioned_folder_names") or [],
        )
        output_summary = (
            f"retrieved {retrieval_result['total_hits']} hit(s); "
            f"sources={', '.join(retrieval_result.get('top_sources', [])) or 'none'}"
        )
        self.repository.upsert_run_step(
            state["run_id"], "knowledge_retrieval", "knowledge_retrieval", "completed",
            input_summary=state["current_message"], output_summary=output_summary,
        )
        return {"retrieval_result": retrieval_result}

    def _knowledge_qa(self, state: AgentState) -> dict[str, Any]:
        """Generate a grounded answer from retrieval results."""
        retrieval_result = state.get("retrieval_result") or {}
        response = self.knowledge_qa.answer(
            question=state["current_message"],
            retrieval_result=retrieval_result,
        )
        self.repository.upsert_run_step(
            state["run_id"], "knowledge_qa", "knowledge_qa", "completed",
            input_summary=state["current_message"],
            output_summary="generated knowledge-grounded response",
        )
        return {
            "response": response,
            "status": "completed",
            "requires_confirmation": False,
            "pending_actions": [],
            "execution_plan": None,
        }

    def _plan_and_solve(self, state: AgentState) -> dict[str, Any]:
        """Build an execution plan for import/retry requests and request user confirmation."""
        # Recover the original fine-grained llm_route from scope_hint to pick
        # the right tool; both import_request and retry_request land here.
        # We derive intent from the keyword fallback on the original message.
        raw_route = self._keyword_detect_plan_route(state["current_message"])
        execution_plan = self._build_execution_plan(raw_route, state["current_message"])
        pending_actions = self._pending_actions_from_execution_plan(execution_plan)
        response = (
            "This request needs confirmation before execution. "
            "Review the planned actions and call the confirmation endpoint to continue."
        )
        self.repository.upsert_run_step(
            state["run_id"], "plan_and_solve", "plan_and_solve", "completed",
            input_summary=state["current_message"],
            output_summary="prepared execution plan",
        )
        return {
            "pending_actions": pending_actions,
            "execution_plan": execution_plan,
            "requires_confirmation": True,
            "status": "awaiting_confirmation",
            "response": response,
        }

    async def _approval_gate(self, state: AgentState) -> dict[str, Any]:
        """Interrupt and wait for user approval before executing tools."""
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
            state["run_id"], "approval_pending", "approval_gate", "awaiting_confirmation",
            input_summary="user approval required",
            output_summary="waiting for confirmation",
        )
        decision = interrupt(payload)
        approved = self._is_approved(decision)
        approval_status = "approved" if approved else "rejected"
        updated_execution_plan = self._mark_execution_plan_status(
            execution_plan, "approved" if approved else "cancelled",
        )
        self.repository.upsert_run_step(
            state["run_id"], "approval_decision", "approval_gate", "completed",
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
                current_plan, tool=tool_name, action=action_name,
            )
            self.repository.upsert_run_step(
                state["run_id"], step_key, f"{tool_name}.{action_name}", "completed",
                input_summary=args_summary, output_summary=response,
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
            state["run_id"], "finalize_run", "finalize_run", "completed",
            input_summary="finalize run state", output_summary=status,
        )
        return {
            "status": status,
            "execution_plan": state.get("execution_plan"),
            "pending_actions": state.get("pending_actions", []),
            # Carry retrieval_result through so done_payload can pass it to session_memory
            "retrieval_result": state.get("retrieval_result"),
        }

    # ------------------------------------------------------------------
    # LLM / keyword routing helpers
    # ------------------------------------------------------------------

    def _llm_detect_route_decision(self, message: str) -> RouteDecision:
        """Use LLM structured output to classify message and extract entity references."""
        if not self.llm.api_key:
            return self._keyword_detect_route_decision(message)
        try:
            decision: RouteDecision = self._router_llm.invoke(
                [
                    SystemMessage(content=_ROUTER_SYSTEM_PROMPT),
                    HumanMessage(content=message),
                ]
            )
            return decision
        except Exception:
            return self._keyword_detect_route_decision(message)

    def _keyword_detect_route_decision(self, message: str) -> RouteDecision:
        """Keyword fallback returning a RouteDecision with empty entity lists."""
        route = self._keyword_detect_llm_route(message)
        return RouteDecision(route=route, reason="keyword fallback")

    def _keyword_detect_llm_route(self, message: str) -> str:
        """Keyword fallback returning the fine-grained 5-way route string."""
        lowered = message.lower()
        retry_kw = ("重试", "retry", "重新执行", "重跑", "失败项")
        import_kw = ("导入", "import", "同步", "sync", "导进知识库", "拉取")
        video_kw = ("bv", "av", "分p", "这一期", "这个视频", "这期视频")
        knowledge_kw = ("讲了什么", "内容", "知识库", "字幕", "qa", "相关视频", "有哪些")

        if any(k in lowered for k in retry_kw):
            return "retry_request"
        if any(k in lowered for k in import_kw):
            return "import_request"
        if "收藏夹" in lowered and any(k in lowered for k in knowledge_kw):
            return "favorite_knowledge_query"
        if any(k in lowered for k in video_kw) and any(k in lowered for k in knowledge_kw + ("讲了什么",)):
            return "video_knowledge_query"
        if any(k in lowered for k in ("bv", "av", "分p", "这个视频", "这期视频")):
            return "video_knowledge_query"
        return "general_chat"

    def _keyword_detect_plan_route(self, message: str) -> str:
        """Distinguish import_request vs retry_request for plan_and_solve node."""
        lowered = message.lower()
        if any(k in lowered for k in ("重试", "retry", "重新执行", "重跑", "失败项")):
            return "retry_request"
        return "import_request"

    def _coerce_route_for_knowledge_query(
        self,
        message: str,
        decision: RouteDecision,
    ) -> RouteDecision:
        """Force concept-style questions into knowledge retrieval when appropriate.

        This is a conservative post-router correction for cases like
        "注意力机制是什么？" where the LLM may choose general_chat even though
        the local knowledge base contains relevant indexed chunks.
        """
        if decision.route != "general_chat":
            return decision
        if not self._looks_like_concept_question(message):
            return decision
        if not self.repository.list_knowledge_videos():
            return decision
        return decision.model_copy(update={
            "route": "favorite_knowledge_query",
            "reason": f"{decision.reason}; coerced to favorite_knowledge_query by concept-question heuristic",
        })

    def _looks_like_concept_question(self, message: str) -> bool:
        lowered = message.lower().strip()
        # Do not hijack obvious general chat or task-execution intents.
        if any(k in lowered for k in ("你好", "hello", "hi", "谢谢", "导入", "import", "重试", "retry")):
            return False
        concept_tokens = (
            "是什么", "什么意思", "什么是", "原理", "作用", "区别", "怎么理解",
            "为什么", "如何理解", "概念", "机制", "用途", "怎么实现", "是什么？", "是啥"
        )
        question_mark = "?" in lowered or "？" in lowered
        return question_mark or any(token in lowered for token in concept_tokens)

    # ------------------------------------------------------------------
    # Execution plan helpers
    # ------------------------------------------------------------------

    def _build_execution_plan(self, route: str, message: str) -> ExecutionPlan:
        if route == "retry_request":
            tool_call: PlannedToolCall = {
                "tool": "bilibili_retry",
                "action": "prepare_retry_plan",
                "target": "failed-import-items",
                "description": f"Retry failed ingestion items related to: {message}",
                "args": {"request_message": message, "target": "failed-import-items"},
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
                "args": {"request_message": message, "target": "favorite-folder-ingestion"},
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
        return {"goal": goal, "summary": summary, "steps": [step], "tool_calls": [tool_call]}

    def _pending_actions_from_execution_plan(
        self,
        execution_plan: ExecutionPlan | dict[str, Any] | None,
    ) -> list[PendingAction]:
        if not execution_plan:
            return []
        return [
            {
                "tool": str(tc["tool"]),
                "action": str(tc["action"]),
                "target": str(tc["target"]),
                "description": str(tc["description"]),
            }
            for tc in execution_plan.get("tool_calls", [])
        ]

    def _mark_execution_plan_status(
        self,
        execution_plan: ExecutionPlan | dict[str, Any] | None,
        status: str,
    ) -> ExecutionPlan | None:
        if not execution_plan:
            return None
        updated = deepcopy(execution_plan)
        for step in updated.get("steps", []):
            step["status"] = status
        return updated

    def _mark_execution_step_completed(
        self,
        execution_plan: ExecutionPlan | dict[str, Any] | None,
        *,
        tool: str,
        action: str,
    ) -> ExecutionPlan | None:
        if not execution_plan:
            return None
        updated = deepcopy(execution_plan)
        for step in updated.get("steps", []):
            if step.get("tool") == tool and step.get("action") == action:
                step["status"] = "completed"
        return updated

    def _get_registered_tool(self, tool: str, action: str) -> Any:
        registered = self.tools.get((tool, action))
        if registered is None:
            raise RuntimeError(f"No registered tool for {tool}.{action}")
        return registered

    def _summarize_tool_args(self, args: dict[str, Any]) -> str:
        if not args:
            return "no args"
        return ", ".join(f"{k}={v}" for k, v in args.items())

    def _is_approved(self, decision: Any) -> bool:
        if isinstance(decision, bool):
            return decision
        if isinstance(decision, dict):
            return bool(decision.get("approved"))
        return False
