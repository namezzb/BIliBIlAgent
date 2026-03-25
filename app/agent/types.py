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
ExecutionStepStatus = Literal["pending", "approved", "cancelled", "completed"]


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


class PlannedToolCall(TypedDict, total=False):
    tool: str
    action: str
    description: str
    target: str
    args: dict[str, object]
    side_effect: bool


class ExecutionStep(TypedDict, total=False):
    id: str
    title: str
    description: str
    tool: str | None
    action: str | None
    status: ExecutionStepStatus


class ExecutionPlan(TypedDict, total=False):
    goal: str
    summary: str
    steps: list[ExecutionStep]
    tool_calls: list[PlannedToolCall]


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
    execution_plan: ExecutionPlan | None
    response: str


class RunEvent(TypedDict):
    event_id: str
    run_id: str
    sequence: int
    type: RunEventType
    timestamp: str
    payload: dict[str, object]
