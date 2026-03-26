from typing import Literal

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


class PlannedToolCallResponse(BaseModel):
    tool: str
    action: str
    description: str
    target: str
    args: dict[str, object] = Field(default_factory=dict)
    side_effect: bool


class ExecutionStepResponse(BaseModel):
    id: str
    title: str
    description: str
    tool: str | None = None
    action: str | None = None
    status: str


class ExecutionPlanResponse(BaseModel):
    goal: str
    summary: str
    steps: list[ExecutionStepResponse] = Field(default_factory=list)
    tool_calls: list[PlannedToolCallResponse] = Field(default_factory=list)


class ChatResponse(BaseModel):
    session_id: str
    run_id: str
    intent: str | None = None
    route: str | None = None
    status: str
    reply: str
    requires_confirmation: bool = False
    approval_status: str | None = None
    execution_plan: ExecutionPlanResponse | None = None
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
    approval_requested_at: str | None = None
    approval_resolved_at: str | None = None
    created_at: str
    updated_at: str
    event_count: int
    steps: list[RunStepResponse] = Field(default_factory=list)



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


class BilibiliAccountResponse(BaseModel):
    mid: int
    uname: str | None = None
    is_login: bool


class BilibiliQrLoginStartResponse(BaseModel):
    qr_url: str
    qrcode_key: str
    expires_in_seconds: int


class BilibiliQrLoginPollResponse(BaseModel):
    status: Literal["pending_scan", "scanned_waiting_confirm", "expired", "success"]
    message: str
    cookie: str | None = Field(default=None, repr=False)
    refresh_token: str | None = None
    account: BilibiliAccountResponse | None = None


class BilibiliFavoriteFolderResponse(BaseModel):
    favorite_folder_id: str
    title: str
    intro: str | None = None
    cover: str | None = None
    media_count: int
    folder_attr: int | None = None
    owner_mid: int | None = None


class BilibiliFavoriteFolderListResponse(BaseModel):
    account: BilibiliAccountResponse
    total: int
    folders: list[BilibiliFavoriteFolderResponse] = Field(default_factory=list)


class BilibiliFavoriteFolderVideoResponse(BaseModel):
    item_id: str
    favorite_folder_id: str
    item_type: int
    media_type: int
    selectable: bool
    unsupported_reason: str | None = None
    video_id: str | None = None
    aid: int | None = None
    bvid: str | None = None
    title: str
    cover: str | None = None
    intro: str | None = None
    duration: int
    upper_mid: int | None = None
    upper_name: str | None = None
    fav_time: int | None = None
    pubtime: int | None = None


class BilibiliFavoriteFolderVideoListResponse(BaseModel):
    account: BilibiliAccountResponse
    folder: BilibiliFavoriteFolderResponse
    page: int
    page_size: int
    total: int
    total_pages: int
    has_more: bool
    items: list[BilibiliFavoriteFolderVideoResponse] = Field(default_factory=list)


class BilibiliImportSubmitRequest(BaseModel):
    session_id: str | None = None
    user_id: str | None = None
    favorite_folder_id: str = Field(min_length=1)
    selected_video_ids: list[str] = Field(min_length=1)


class KnowledgeFavoriteFolderInput(BaseModel):
    favorite_folder_id: str = Field(min_length=1)
    title: str = Field(min_length=1)
    intro: str | None = None


class KnowledgeTextBlockInput(BaseModel):
    text: str = Field(min_length=1)
    source_type: Literal["subtitle", "asr"]
    source_language: str | None = None
    start_ms: int | None = None
    end_ms: int | None = None


class KnowledgeVideoPageInput(BaseModel):
    page_id: str = Field(min_length=1)
    page_number: int = Field(ge=1)
    title: str = Field(min_length=1)
    text_blocks: list[KnowledgeTextBlockInput] = Field(default_factory=list)


class KnowledgeVideoInput(BaseModel):
    video_id: str = Field(min_length=1)
    bvid: str | None = None
    title: str = Field(min_length=1)
    favorite_folder_ids: list[str] = Field(default_factory=list)
    pages: list[KnowledgeVideoPageInput] = Field(default_factory=list)


class KnowledgeDebugIndexRequest(BaseModel):
    favorite_folders: list[KnowledgeFavoriteFolderInput] = Field(default_factory=list)
    videos: list[KnowledgeVideoInput] = Field(default_factory=list)


class KnowledgeDebugIndexResponse(BaseModel):
    favorite_folder_count: int
    video_count: int
    page_count: int
    chunk_count: int
    embedding_model: str
    embedding_version: str


class KnowledgeSearchRequest(BaseModel):
    query: str = Field(min_length=1)
    top_k: int = Field(default=5, ge=1, le=20)
    favorite_folder_ids: list[str] = Field(default_factory=list)
    video_ids: list[str] = Field(default_factory=list)
    page_numbers: list[int] = Field(default_factory=list)
    source_types: list[Literal["subtitle", "asr"]] = Field(default_factory=list)


class KnowledgeFavoriteFolderResponse(BaseModel):
    favorite_folder_id: str
    title: str
    intro: str | None = None


class KnowledgeVideoResponse(BaseModel):
    video_id: str
    bvid: str | None = None
    title: str


class KnowledgeVideoPageResponse(BaseModel):
    page_id: str
    page_number: int
    title: str


class KnowledgeSearchHitResponse(BaseModel):
    score: float
    chunk_id: str
    text: str
    source_type: str
    source_language: str | None = None
    start_ms: int | None = None
    end_ms: int | None = None
    favorite_folders: list[KnowledgeFavoriteFolderResponse] = Field(default_factory=list)
    pages: list[KnowledgeVideoPageResponse] = Field(default_factory=list)
    video: KnowledgeVideoResponse


class KnowledgeSearchResponse(BaseModel):
    query: str
    total_hits: int
    hits: list[KnowledgeSearchHitResponse] = Field(default_factory=list)
