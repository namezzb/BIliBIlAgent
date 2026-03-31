from typing import Any, Literal, TypedDict, Annotated

from pydantic import BaseModel, Field


# Three top-level routes that drive the graph branching.
RouteType = Literal[
    "general_chat",
    "knowledge_query",
    "plan_and_solve",
]

# Coarse-grained first-stage intent.
IntentType = Literal[
    "chat",
    "knowledge",
    "action",
]

# Action subtype resolved only when intent=action.
ActionRouteType = Literal[
    "import_request",
    "retry_request",
]

# Fine-grained hint used only inside retrieve_knowledge for scope resolution.
# Kept separate so the graph branches stay clean (3-way only).
RetrievalScopeHint = Literal[
    "favorite_knowledge_query",
    "video_knowledge_query",
    "general_knowledge_query",
]

# Internal LLM router output — retained for compatibility during transition.
_LLMRouteType = Literal[
    "general_chat",
    "favorite_knowledge_query",
    "video_knowledge_query",
    "import_request",
    "retry_request",
]


class IntentDecision(BaseModel):
    """Structured output schema for coarse first-stage intent detection."""

    intent: IntentType = Field(
        description="Coarse user intent: chat, knowledge, or action."
    )
    reason: str = Field(
        description="One-sentence explanation of why this intent was chosen."
    )


class KnowledgeScopeDecision(BaseModel):
    """Structured output schema for second-stage knowledge scope parsing."""

    scope: RetrievalScopeHint = Field(
        description="Knowledge retrieval scope: general, favorite-folder, or video-focused."
    )
    reason: str = Field(
        description="One-sentence explanation of why this scope was chosen."
    )
    mentioned_bvids: list[str] = Field(
        default_factory=list,
        description="BV IDs explicitly mentioned in the message. Empty list if none.",
    )
    mentioned_video_titles: list[str] = Field(
        default_factory=list,
        description="Video titles explicitly mentioned by the user. Empty list if none.",
    )
    mentioned_folder_names: list[str] = Field(
        default_factory=list,
        description="Favorite folder names explicitly mentioned by the user. Empty list if none.",
    )


class ActionDecision(BaseModel):
    """Structured output schema for second-stage action subtype detection."""

    action: ActionRouteType = Field(
        description="Action subtype: import_request or retry_request."
    )
    reason: str = Field(
        description="One-sentence explanation of why this action was chosen."
    )


class RouteDecision(BaseModel):
    """Backward-compatible structured schema for the legacy fine-grained router."""

    route: _LLMRouteType = Field(
        description=(
            "The legacy fine-grained route that best matches the user's message. "
            "Use 'general_chat' when none of the specific routes apply."
        )
    )
    reason: str = Field(
        description="One-sentence explanation of why this route was chosen (for debugging)."
    )
    mentioned_bvids: list[str] = Field(
        default_factory=list,
        description="BV IDs explicitly mentioned in the message, e.g. ['BV1xx', 'BV2yy']. Empty list if none.",
    )
    mentioned_video_titles: list[str] = Field(
        default_factory=list,
        description="Video titles explicitly mentioned by the user. Empty list if none.",
    )
    mentioned_folder_names: list[str] = Field(
        default_factory=list,
        description="Favorite folder names explicitly mentioned by the user. Empty list if none.",
    )


RunStatus = Literal["running", "completed", "awaiting_confirmation", "cancelled", "failed"]
ExecutionStepStatus = Literal["pending", "approved", "cancelled", "completed"]


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


def _keep_latest_retrieval(
    old: "dict[str, Any] | None",
    new: "dict[str, Any] | None",
) -> "dict[str, Any] | None":
    """Reducer: keep the latest non-None retrieval_result across graph turns.

    LangGraph checkpointer persists AgentState between turns. With this reducer,
    the previous turn's retrieval_result is available in state for scope inheritance
    in _resolve_scope, without needing manual recent_context bookkeeping.
    """
    return new if new is not None else old


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
    action_route: ActionRouteType | None
    # Top-level route driving graph branching (3 values)
    route: RouteType
    # Fine-grained hint passed to retrieve_knowledge for scope resolution
    retrieval_scope_hint: RetrievalScopeHint | None
    # LLM-identified entity references from the router (used by _resolve_scope)
    mentioned_bvids: list[str]
    mentioned_video_titles: list[str]
    mentioned_folder_names: list[str]
    status: RunStatus
    requires_confirmation: bool
    approval_status: Literal["approved", "rejected"] | None
    pending_actions: list[PendingAction]
    execution_plan: ExecutionPlan | None
    retrieval_result: Annotated[dict[str, Any] | None, _keep_latest_retrieval]
    response: str
