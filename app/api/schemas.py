from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    session_id: str | None = None
    user_id: str | None = None
    message: str = Field(min_length=1)


class PendingActionResponse(BaseModel):
    tool: str
    action: str
    target: str
    description: str


class ChatResponse(BaseModel):
    session_id: str
    run_id: str
    intent: str | None = None
    route: str | None = None
    status: str
    reply: str
    requires_confirmation: bool = False
    approval_status: str | None = None
    pending_actions: list[PendingActionResponse] = Field(default_factory=list)


class RunConfirmationRequest(BaseModel):
    approved: bool


class RunStepResponse(BaseModel):
    step_key: str
    step_name: str
    status: str
    input_summary: str | None = None
    output_summary: str | None = None
    updated_at: str


class RunDetailResponse(ChatResponse):
    created_at: str
    updated_at: str
    event_count: int
    steps: list[RunStepResponse] = Field(default_factory=list)


class RunEventResponse(BaseModel):
    event_id: str
    run_id: str
    sequence: int
    type: str
    timestamp: str
    payload: dict[str, object]


class SessionMessageResponse(BaseModel):
    message_id: str
    run_id: str | None = None
    role: str
    content: str
    created_at: str


class SessionDetailResponse(BaseModel):
    session_id: str
    user_id: str | None = None
    summary_text: str | None = None
    recent_context: dict[str, object] = Field(default_factory=dict)
    created_at: str
    updated_at: str
    messages: list[SessionMessageResponse] = Field(default_factory=list)


class UserMemoryEntryResponse(BaseModel):
    value: str
    source_type: str
    source_run_id: str | None = None
    source_text: str | None = None
    confirmed: bool
    created_at: str
    updated_at: str


class UserMemoryProfileResponse(BaseModel):
    user_id: str
    preferences: dict[str, UserMemoryEntryResponse] = Field(default_factory=dict)
    aliases: dict[str, UserMemoryEntryResponse] = Field(default_factory=dict)
    default_scopes: dict[str, UserMemoryEntryResponse] = Field(default_factory=dict)
    created_at: str | None = None
    updated_at: str | None = None


class UserMemoryPatchRequest(BaseModel):
    preferences: dict[str, str] | None = None
    aliases: dict[str, str] | None = None
    default_scopes: dict[str, str] | None = None

    def to_updates(self) -> dict[str, dict[str, str]]:
        updates: dict[str, dict[str, str]] = {}
        for field_name in ("preferences", "aliases", "default_scopes"):
            value = getattr(self, field_name)
            if value is not None:
                updates[field_name] = value
        return updates
