from typing import Literal, TypedDict


IntentType = Literal["general_chat", "knowledge_query", "tool_request"]
RunStatus = Literal["running", "completed", "awaiting_confirmation", "cancelled"]


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
