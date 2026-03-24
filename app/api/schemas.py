from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    session_id: str | None = None
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
    summary_text: str | None = None
    recent_context: dict[str, object] = Field(default_factory=dict)
    created_at: str
    updated_at: str
    messages: list[SessionMessageResponse] = Field(default_factory=list)
