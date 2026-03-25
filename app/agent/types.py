from typing import Literal, TypedDict


IntentType = Literal["general_chat", "knowledge_query", "tool_request"]

#Routing
RouteType = Literal[
    "general_chat",
    "favorite_knowledge_query",
    "video_knowledge_query",
    "import_request",
    "retry_request",
]


RunStatus = Literal["running", "completed", "awaiting_confirmation", "cancelled", "failed"]


RunEventType = Literal[
    "run_started",
    "context_loaded",
    "intent_classified",
    "response_prepared",
    "confirmation_required",
    "confirmation_resolved",
    "tool_execution_started",
    "tool_execution_finished",
    "run_completed",
    "run_failed",
]


class PendingAction(TypedDict, total=False):
    tool: str
    action: str
    target: str
    description: str


class AgentState(TypedDict, total=False):
    session_id: str
    user_id: str | None
    run_id: str
    current_message: str
    messages: list[dict[str, str]]
    session_summary: str | None
    recent_context: dict[str, object]
    user_memory_context: str | None
    intent: IntentType
    route: RouteType
    status: RunStatus
    requires_confirmation: bool
    approval_status: Literal["approved", "rejected"] | None
    pending_actions: list[PendingAction]
    response: str


class RunEvent(TypedDict):
    event_id: str
    run_id: str
    sequence: int
    type: RunEventType
    timestamp: str
    payload: dict[str, object]
