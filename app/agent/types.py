from typing import Literal, TypedDict


IntentType = Literal["general_chat", "knowledge_query", "tool_request"]
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
    run_id: str
    current_message: str
    messages: list[dict[str, str]]
    intent: IntentType
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
