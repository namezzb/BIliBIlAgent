from copy import deepcopy
from typing import Any
from uuid import uuid4

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt

from app.agent.types import (
    ActionDecision,
    ActionRouteType,
    AgentState,
    ExecutionPlan,
    ExecutionStep,
    IntentDecision,
    KnowledgeScopeDecision,
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

_INTENT_SYSTEM_PROMPT = """\
You are the first-stage intent router for BIliBIlAgent. Classify the user message into exactly one coarse intent:

- chat       — casual conversation, greetings, thanks, or requests that do not need the local Bilibili knowledge base
- knowledge  — questions asking about concepts, explanations, comparisons, video content, favorite-folder content, or follow-up questions that should use the local knowledge base
- action     — user wants the system to execute a side-effectful operation such as import / sync / retry

Rules:
1. If the user is asking what something is, how it works, differences, principles, explanations, or "这个视频/这个收藏夹还讲了什么", prefer knowledge.
2. If the user asks to import, sync, add, retry, rerun, or execute something, prefer action.
3. Use chat only when the message is ordinary conversation or clearly unrelated to local knowledge retrieval and execution.
4. Reply ONLY with the structured JSON — no extra text.
"""

_KNOWLEDGE_SCOPE_SYSTEM_PROMPT = """\
You are the second-stage knowledge scope parser for BIliBIlAgent. The message is already known to be a knowledge query.
Classify the scope into exactly one of:

- general_knowledge_query  — concept/explanation questions without a specific video or folder target
- favorite_knowledge_query — asking about a favorite folder, multiple videos, or broad library/folder content
- video_knowledge_query    — asking about a specific video, BV/AV ID, a specific episode/page, or follow-up references like "这个视频"

Rules:
1. If the message mentions BV/AV, a specific video, page number, or "这个视频/这期", prefer video_knowledge_query.
2. If the message mentions a favorite folder or asks for multiple related videos, prefer favorite_knowledge_query.
3. If the message asks about a concept, principle, meaning, difference, or explanation without a clear target, prefer general_knowledge_query.
4. Extract mentioned_bvids, mentioned_video_titles, and mentioned_folder_names when explicit.
5. Reply ONLY with the structured JSON — no extra text.
"""

_ACTION_SYSTEM_PROMPT = """\
You are the second-stage action parser for BIliBIlAgent. The message is already known to be an action request.
Classify it into exactly one action:

- import_request — import / sync / ingest favorite-folder or video content into the local knowledge base
- retry_request  — retry previously failed import or ingestion tasks

Rules:
1. If the message explicitly mentions retrying, rerunning failed items, or trying again, prefer retry_request.
2. Otherwise prefer import_request.
3. Reply ONLY with the structured JSON — no extra text.
"""

_ACTION_ROUTE_TO_GRAPH_ROUTE: dict[str, RouteType] = {
    "import_request": "plan_and_solve",
    "retry_request": "plan_and_solve",
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
        self._intent_llm = llm.get_lc_chat_model().with_structured_output(IntentDecision)
        self._knowledge_scope_llm = llm.get_lc_chat_model().with_structured_output(KnowledgeScopeDecision)
        self._action_llm = llm.get_lc_chat_model().with_structured_output(ActionDecision)
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
        """Classify message via coarse intent, then refine into scope/action subtype."""
        message = state["current_message"]
        intent_decision = self._detect_intent(message)

        graph_route: RouteType = "general_chat"
        retrieval_scope_hint: RetrievalScopeHint | None = None
        action_route: ActionRouteType | None = None
        mentioned_bvids: list[str] = []
        mentioned_video_titles: list[str] = []
        mentioned_folder_names: list[str] = []
        output_summary = f"intent={intent_decision.intent}"

        if intent_decision.intent == "knowledge":
            scope_decision = self._detect_knowledge_scope(message)
            graph_route = "knowledge_query"
            retrieval_scope_hint = scope_decision.scope
            mentioned_bvids = scope_decision.mentioned_bvids
            mentioned_video_titles = scope_decision.mentioned_video_titles
            mentioned_folder_names = scope_decision.mentioned_folder_names
            output_summary = (
                f"intent=knowledge scope={scope_decision.scope}"
            )
        elif intent_decision.intent == "action":
            action_decision = self._detect_action_type(message)
            action_route = action_decision.action
            graph_route = _ACTION_ROUTE_TO_GRAPH_ROUTE[action_decision.action]
            output_summary = f"intent=action action={action_decision.action}"

        self.repository.upsert_run_step(
            state["run_id"], "router", "router", "completed",
            input_summary=message,
            output_summary=output_summary,
        )
        return {
            "intent": intent_decision.intent,
            "action_route": action_route,
            "route": graph_route,
            "retrieval_scope_hint": retrieval_scope_hint,
            "mentioned_bvids": mentioned_bvids,
            "mentioned_video_titles": mentioned_video_titles,
            "mentioned_folder_names": mentioned_folder_names,
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
        raw_route = state.get("action_route") or self._detect_action_type(state["current_message"]).action
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

    def _detect_intent(self, message: str) -> IntentDecision:
        if not self.llm.api_key:
            return self._keyword_detect_intent(message)
        try:
            decision: IntentDecision = self._intent_llm.invoke(
                [
                    SystemMessage(content=_INTENT_SYSTEM_PROMPT),
                    HumanMessage(content=message),
                ]
            )
            return decision
        except Exception:
            return self._keyword_detect_intent(message)

    def _detect_knowledge_scope(self, message: str) -> KnowledgeScopeDecision:
        if not self.llm.api_key:
            return self._keyword_detect_knowledge_scope(message)
        try:
            decision: KnowledgeScopeDecision = self._knowledge_scope_llm.invoke(
                [
                    SystemMessage(content=_KNOWLEDGE_SCOPE_SYSTEM_PROMPT),
                    HumanMessage(content=message),
                ]
            )
            return decision
        except Exception:
            return self._keyword_detect_knowledge_scope(message)

    def _detect_action_type(self, message: str) -> ActionDecision:
        if not self.llm.api_key:
            return self._keyword_detect_action_type(message)
        try:
            decision: ActionDecision = self._action_llm.invoke(
                [
                    SystemMessage(content=_ACTION_SYSTEM_PROMPT),
                    HumanMessage(content=message),
                ]
            )
            return decision
        except Exception:
            return self._keyword_detect_action_type(message)

    def _keyword_detect_intent(self, message: str) -> IntentDecision:
        lowered = message.lower().strip()
        if self._looks_like_action_request(lowered):
            return IntentDecision(intent="action", reason="keyword fallback: action request")
        if self._looks_like_knowledge_query(lowered):
            return IntentDecision(intent="knowledge", reason="keyword fallback: knowledge query")
        return IntentDecision(intent="chat", reason="keyword fallback: chat")

    def _keyword_detect_knowledge_scope(self, message: str) -> KnowledgeScopeDecision:
        lowered = message.lower().strip()
        mentioned_bvids = self._extract_bvids(message)
        mentioned_video_titles = self._extract_explicit_video_titles(message)
        mentioned_folder_names = self._extract_explicit_folder_names(message)

        if self._looks_like_video_scope(lowered, mentioned_bvids, mentioned_video_titles):
            return KnowledgeScopeDecision(
                scope="video_knowledge_query",
                reason="keyword fallback: video-focused knowledge query",
                mentioned_bvids=mentioned_bvids,
                mentioned_video_titles=mentioned_video_titles,
                mentioned_folder_names=mentioned_folder_names,
            )
        if self._looks_like_folder_scope(lowered, mentioned_folder_names):
            return KnowledgeScopeDecision(
                scope="favorite_knowledge_query",
                reason="keyword fallback: folder-focused knowledge query",
                mentioned_bvids=mentioned_bvids,
                mentioned_video_titles=mentioned_video_titles,
                mentioned_folder_names=mentioned_folder_names,
            )
        return KnowledgeScopeDecision(
            scope="general_knowledge_query",
            reason="keyword fallback: general knowledge query",
            mentioned_bvids=mentioned_bvids,
            mentioned_video_titles=mentioned_video_titles,
            mentioned_folder_names=mentioned_folder_names,
        )

    def _keyword_detect_action_type(self, message: str) -> ActionDecision:
        lowered = message.lower().strip()
        if any(k in lowered for k in ("重试", "retry", "重新执行", "重跑", "失败项")):
            return ActionDecision(action="retry_request", reason="keyword fallback: retry request")
        return ActionDecision(action="import_request", reason="keyword fallback: import request")

    def _looks_like_action_request(self, lowered: str) -> bool:
        return any(k in lowered for k in ("导入", "import", "同步", "sync", "导进知识库", "拉取", "重试", "retry", "重新执行", "重跑", "失败项"))

    def _looks_like_knowledge_query(self, lowered: str) -> bool:
        if any(k in lowered for k in ("你好", "hello", "hi", "谢谢", "thank", "你是谁")):
            return False
        if self._looks_like_action_request(lowered):
            return False
        knowledge_tokens = (
            "是什么", "什么意思", "什么是", "原理", "作用", "区别", "怎么理解",
            "为什么", "如何理解", "概念", "机制", "用途", "怎么实现", "讲了什么",
            "内容", "知识库", "字幕", "qa", "相关视频", "有哪些", "这个视频", "这期视频", "收藏夹"
        )
        return ("?" in lowered or "？" in lowered or any(token in lowered for token in knowledge_tokens))

    def _looks_like_video_scope(
        self,
        lowered: str,
        mentioned_bvids: list[str],
        mentioned_video_titles: list[str],
    ) -> bool:
        if mentioned_bvids or mentioned_video_titles:
            return True
        return any(k in lowered for k in ("bv", "av", "分p", "这一期", "这个视频", "这期视频", "第1p", "第2p", "p1", "p2"))

    def _looks_like_folder_scope(self, lowered: str, mentioned_folder_names: list[str]) -> bool:
        if mentioned_folder_names:
            return True
        return "收藏夹" in lowered

    def _extract_bvids(self, message: str) -> list[str]:
        return sorted({token.upper() for token in message.split() if token.upper().startswith("BV")})

    def _extract_explicit_video_titles(self, message: str) -> list[str]:
        videos = self.repository.list_knowledge_videos()
        lowered = message.lower()
        return sorted({str(video["title"]) for video in videos if str(video["title"]).lower() in lowered})

    def _extract_explicit_folder_names(self, message: str) -> list[str]:
        folders = self.repository.list_knowledge_favorite_folders()
        lowered = message.lower()
        return sorted({str(folder["title"]) for folder in folders if str(folder["title"]).lower() in lowered})


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
