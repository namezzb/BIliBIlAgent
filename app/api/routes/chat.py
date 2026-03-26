import asyncio
import json
from typing import Literal
from uuid import uuid4

from fastapi import APIRouter, BackgroundTasks, HTTPException, Query, Request, status
from fastapi.responses import StreamingResponse

from app.api.schemas import (
    BilibiliFavoriteFolderListResponse,
    BilibiliFavoriteFolderVideoListResponse,
    BilibiliImportSubmitRequest,
    BilibiliQrLoginPollResponse,
    BilibiliQrLoginStartResponse,
    ChatRequest,
    ChatResponse,
    KnowledgeDebugIndexRequest,
    KnowledgeDebugIndexResponse,
    KnowledgeSearchRequest,
    KnowledgeSearchResponse,
    RunConfirmationRequest,
    RunDetailResponse,
    SessionDetailResponse,
    UserMemoryPatchRequest,
    UserMemoryProfileResponse,
)
from app.services.bilibili_favorites import (
    BilibiliFavoriteFolderAuthError,
    BilibiliFavoriteFolderResponseError,
    BilibiliFavoriteFolderUpstreamError,
)
from app.services.knowledge_index import DuplicateKnowledgeVideoError


router = APIRouter(prefix="/api", tags=["agent"])


def _persist_run_result(
    repository,
    run_id: str,
    result: dict[str, object],
) -> None:
    repository.update_run(
        run_id,
        intent=result["intent"],
        route=result["route"],
        status=result["status"],
        requires_confirmation=result["requires_confirmation"],
        approval_status=result["approval_status"],
        latest_reply=result["reply"],
        pending_actions=result["pending_actions"],
        execution_plan=result.get("execution_plan"),
        approval_requested_at=result.get("approval_requested_at"),
        approval_resolved_at=result.get("approval_resolved_at"),
    )


def _record_run_failure(repository, run_id: str, detail: str) -> None:
    repository.upsert_run_step(
        run_id,
        "request_failed",
        "request_failed",
        "failed",
        input_summary="request lifecycle",
        output_summary=detail,
    )
    repository.append_run_event(
        run_id,
        "run_failed",
        {
            "status": "failed",
            "route": None,
            "reply": detail,
        },
    )
    repository.update_run(
        run_id,
        status="failed",
        latest_reply=detail,
    )


def _require_bilibili_cookie(request: Request) -> str:
    cookie = request.headers.get("X-Bilibili-Cookie")
    if not cookie:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing X-Bilibili-Cookie header.",
        )
    return cookie


def _prepare_session(repository, *, session_id: str | None, user_id: str | None) -> tuple[str, str | None]:
    active_session_id = session_id or str(uuid4())
    session = repository.get_session(active_session_id)
    if session is not None:
        session_user_id = session.get("user_id")
        if session_user_id and user_id and session_user_id != user_id:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="Session is already bound to a different user_id.",
            )
        if user_id and session_user_id is None:
            repository.set_session_user_id(active_session_id, user_id)
        repository.touch_session(active_session_id)
        active_user_id = user_id or session_user_id
    else:
        repository.create_session(active_session_id, user_id=user_id)
        active_user_id = user_id
    return active_session_id, active_user_id


def _execute_bilibili_import_in_background(
    app,
    *,
    run_id: str,
    session_id: str,
    user_id: str | None,
    cookie: str,
    favorite_folder_id: str,
    selected_video_ids: list[str],
) -> None:
    tool = app.state.bilibili_import_tool
    try:
        tool.invoke(
            {
                "run_id": run_id,
                "session_id": session_id,
                "user_id": user_id,
                "cookie": cookie,
                "favorite_folder_id": favorite_folder_id,
                "selected_video_ids": selected_video_ids,
            }
        )
    except Exception:
        return None


@router.get("/runs/{run_id}", response_model=RunDetailResponse)
def get_run(run_id: str, request: Request) -> RunDetailResponse:
    repository = request.app.state.repository
    existing_run = repository.get_run(run_id)
    if existing_run is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Run not found.")

    return RunDetailResponse(
        session_id=existing_run["session_id"],
        run_id=existing_run["run_id"],
        intent=existing_run["intent"],
        route=existing_run["route"],
        status=existing_run["status"],
        reply=existing_run["latest_reply"] or "",
        requires_confirmation=existing_run["requires_confirmation"],
        approval_status=existing_run["approval_status"],
        execution_plan=existing_run["execution_plan"],
        pending_actions=existing_run["pending_actions"],
        approval_requested_at=existing_run["approval_requested_at"],
        approval_resolved_at=existing_run["approval_resolved_at"],
        created_at=existing_run["created_at"],
        updated_at=existing_run["updated_at"],
        event_count=repository.get_run_event_count(run_id),
        steps=repository.get_run_steps(run_id),
    )


@router.get("/sessions/{session_id}", response_model=SessionDetailResponse)
def get_session(session_id: str, request: Request) -> SessionDetailResponse:
    session_memory = request.app.state.session_memory
    session_detail = session_memory.get_session_detail(session_id)
    if session_detail is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Session not found.")
    return SessionDetailResponse(**session_detail)


@router.get("/users/{user_id}/memory", response_model=UserMemoryProfileResponse)
def get_user_memory(user_id: str, request: Request) -> UserMemoryProfileResponse:
    user_memory = request.app.state.user_memory
    return UserMemoryProfileResponse(**user_memory.get_profile_detail(user_id))


@router.patch("/users/{user_id}/memory", response_model=UserMemoryProfileResponse)
def patch_user_memory(
    user_id: str,
    payload: UserMemoryPatchRequest,
    request: Request,
) -> UserMemoryProfileResponse:
    user_memory = request.app.state.user_memory
    updates = payload.to_updates()
    if not updates:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="At least one user-memory group must be provided.",
        )
    detail = user_memory.upsert_entries(
        user_id,
        updates,
        source_type="api",
        source_run_id=None,
        source_text=None,
    )
    return UserMemoryProfileResponse(**detail)


@router.delete("/users/{user_id}/memory/{group}/{key}", response_model=UserMemoryProfileResponse)
def delete_user_memory(
    user_id: str,
    group: Literal["preferences", "aliases", "default_scopes"],
    key: str,
    request: Request,
) -> UserMemoryProfileResponse:
    user_memory = request.app.state.user_memory
    user_memory.delete_entry(user_id, group, key)
    return UserMemoryProfileResponse(**user_memory.get_profile_detail(user_id))


@router.post("/bilibili/auth/qr/start", response_model=BilibiliQrLoginStartResponse)
def start_bilibili_qr_login(request: Request) -> BilibiliQrLoginStartResponse:
    service = request.app.state.bilibili_favorite_folder_service
    try:
        result = service.start_qr_login()
    except BilibiliFavoriteFolderUpstreamError as exc:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc)) from exc
    except BilibiliFavoriteFolderResponseError as exc:
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=str(exc)) from exc
    return BilibiliQrLoginStartResponse(**result)


@router.get("/bilibili/auth/qr/poll", response_model=BilibiliQrLoginPollResponse)
def poll_bilibili_qr_login(
    qrcode_key: str,
    request: Request,
) -> BilibiliQrLoginPollResponse:
    service = request.app.state.bilibili_favorite_folder_service
    try:
        result = service.poll_qr_login(qrcode_key)
    except BilibiliFavoriteFolderUpstreamError as exc:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc)) from exc
    except BilibiliFavoriteFolderResponseError as exc:
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=str(exc)) from exc
    return BilibiliQrLoginPollResponse(**result)


@router.get("/bilibili/favorite-folders", response_model=BilibiliFavoriteFolderListResponse)
def list_bilibili_favorite_folders(
    request: Request,
) -> BilibiliFavoriteFolderListResponse:
    service = request.app.state.bilibili_favorite_folder_service
    cookie = _require_bilibili_cookie(request)
    try:
        result = service.list_favorite_folders(cookie)
    except BilibiliFavoriteFolderAuthError as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"{exc} Cookie 失效时请重新扫码登录。",
        ) from exc
    except BilibiliFavoriteFolderUpstreamError as exc:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc)) from exc
    except BilibiliFavoriteFolderResponseError as exc:
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=str(exc)) from exc
    return BilibiliFavoriteFolderListResponse(**result)


@router.get(
    "/bilibili/favorite-folders/{favorite_folder_id}/videos",
    response_model=BilibiliFavoriteFolderVideoListResponse,
)
def list_bilibili_favorite_folder_videos(
    favorite_folder_id: str,
    request: Request,
    pn: int = Query(default=1, ge=1),
    ps: int = Query(default=20, ge=1, le=20),
    keyword: str = "",
    order: Literal["mtime", "view", "pubtime"] = "mtime",
) -> BilibiliFavoriteFolderVideoListResponse:
    service = request.app.state.bilibili_favorite_folder_service
    cookie = _require_bilibili_cookie(request)
    try:
        result = service.list_folder_items(
            cookie,
            favorite_folder_id,
            pn=pn,
            ps=ps,
            keyword=keyword,
            order=order,
        )
    except BilibiliFavoriteFolderAuthError as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"{exc} Cookie 失效时请重新扫码登录。",
        ) from exc
    except BilibiliFavoriteFolderUpstreamError as exc:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc)) from exc
    except BilibiliFavoriteFolderResponseError as exc:
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=str(exc)) from exc
    return BilibiliFavoriteFolderVideoListResponse(**result)


@router.post("/bilibili/imports", response_model=ChatResponse, status_code=status.HTTP_202_ACCEPTED)
def submit_bilibili_import(
    payload: BilibiliImportSubmitRequest,
    request: Request,
    background_tasks: BackgroundTasks,
) -> ChatResponse:
    repository = request.app.state.repository
    import_pipeline = request.app.state.bilibili_import_pipeline
    cookie = _require_bilibili_cookie(request)
    session_id, user_id = _prepare_session(
        repository,
        session_id=payload.session_id,
        user_id=payload.user_id,
    )

    try:
        validated = import_pipeline.validate_selected_items(
            cookie=cookie,
            favorite_folder_id=payload.favorite_folder_id,
            selected_video_ids=payload.selected_video_ids,
        )
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    except BilibiliFavoriteFolderAuthError as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"{exc} Cookie 失效时请重新扫码登录。",
        ) from exc
    except BilibiliFavoriteFolderUpstreamError as exc:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc)) from exc
    except BilibiliFavoriteFolderResponseError as exc:
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=str(exc)) from exc

    run_id = str(uuid4())
    execution_plan = import_pipeline.build_execution_plan(
        favorite_folder_id=payload.favorite_folder_id,
        selected_video_ids=validated["selected_video_ids"],
    )
    reply = (
        "Import accepted. The backend is fetching subtitles, preparing ASR fallback jobs, "
        f"and indexing {len(validated['selected_video_ids'])} selected video(s)."
    )

    repository.create_run(run_id, session_id, status="running")
    repository.update_run(
        run_id,
        intent="tool_request",
        route="import_request",
        status="running",
        requires_confirmation=False,
        approval_status="approved",
        latest_reply=reply,
        pending_actions=[],
        execution_plan=execution_plan,
    )
    repository.upsert_run_step(
        run_id,
        "import_submitted",
        "import_submitted",
        "completed",
        input_summary=f"favorite_folder_id={payload.favorite_folder_id}",
        output_summary=f"accepted {len(validated['selected_video_ids'])} selected video(s)",
    )
    repository.append_run_event(
        run_id,
        "run_started",
        {
            "session_id": session_id,
            "user_id": user_id,
            "favorite_folder_id": payload.favorite_folder_id,
            "selected_video_ids": list(validated["selected_video_ids"]),
            "route": "import_request",
        },
    )

    background_tasks.add_task(
        _execute_bilibili_import_in_background,
        request.app,
        run_id=run_id,
        session_id=session_id,
        user_id=user_id,
        cookie=cookie,
        favorite_folder_id=payload.favorite_folder_id,
        selected_video_ids=list(validated["selected_video_ids"]),
    )
    return ChatResponse(
        session_id=session_id,
        run_id=run_id,
        intent="tool_request",
        route="import_request",
        status="running",
        reply=reply,
        requires_confirmation=False,
        approval_status="approved",
        execution_plan=execution_plan,
        pending_actions=[],
    )


@router.post("/knowledge/debug/index", response_model=KnowledgeDebugIndexResponse)
def debug_index_knowledge(
    payload: KnowledgeDebugIndexRequest,
    request: Request,
) -> KnowledgeDebugIndexResponse:
    knowledge_index = request.app.state.knowledge_index
    try:
        result = knowledge_index.index_documents(payload.model_dump())
    except DuplicateKnowledgeVideoError as exc:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Knowledge indexing failed: {exc}",
        ) from exc
    return KnowledgeDebugIndexResponse(**result)


@router.post("/knowledge/search", response_model=KnowledgeSearchResponse)
def search_knowledge(
    payload: KnowledgeSearchRequest,
    request: Request,
) -> KnowledgeSearchResponse:
    knowledge_retrieval = request.app.state.knowledge_retrieval
    try:
        result = knowledge_retrieval.search(payload.model_dump())
    except RuntimeError as exc:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc)) from exc
    return KnowledgeSearchResponse(**result)


@router.post("/chat/stream")
async def chat_stream(request: Request, payload: ChatRequest) -> StreamingResponse:
    """SSE endpoint: streams LLM tokens + node updates from graph.astream().

    Event format (each line is: data: <json>\n\n):
      {"type": "token",     "content": str}
      {"type": "node",      "node": str, "data": dict}
      {"type": "interrupt", "data": dict}    <- approval required
      {"type": "done",      "run_id": str, "status": str, "reply": str, ...}
      "[DONE]"
    """
    repository = request.app.state.repository
    orchestrator = request.app.state.orchestrator
    session_memory = request.app.state.session_memory

    session_id, user_id = _prepare_session(
        repository,
        session_id=payload.session_id,
        user_id=payload.user_id,
    )
    run_id = str(uuid4())
    repository.create_run(run_id, session_id, status="running")
    repository.add_message(session_id, "user", payload.message, run_id=run_id)
    session_context = session_memory.load_session_context(session_id)
    user_memory_context = None
    if user_id:
        user_memory_context = request.app.state.user_memory.build_context_message(user_id)

    async def event_generator():
        done_payload: dict = {}
        try:
            async for chunk in orchestrator.astream_chat(
                session_id=session_id,
                run_id=run_id,
                message=payload.message,
                messages=session_context["messages"],
                user_id=user_id,
                session_summary=session_context["session_summary"],
                recent_context=session_context["recent_context"],
                user_memory_context=user_memory_context,
            ):
                if chunk.get("type") == "done":
                    done_payload = chunk
                yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"
        except Exception as exc:
            detail = f"Agent stream failed: {exc}"
            _record_run_failure(repository, run_id, detail)
            yield f"data: {json.dumps({'type': 'error', 'detail': detail})}\n\n"
            yield "data: [DONE]\n\n"
            return

        yield "data: [DONE]\n\n"

        # Persist run result and update session memory after stream completes
        if done_payload:
            result = {
                "session_id": session_id,
                "run_id": run_id,
                "intent": done_payload.get("intent"),
                "route": done_payload.get("route"),
                "status": done_payload.get("status", "completed"),
                "reply": done_payload.get("reply", ""),
                "requires_confirmation": done_payload.get("requires_confirmation", False),
                "approval_status": done_payload.get("approval_status"),
                "execution_plan": done_payload.get("execution_plan"),
                "pending_actions": done_payload.get("pending_actions", []),
                "approval_requested_at": None,
                "approval_resolved_at": None,
            }
            _persist_run_result(repository, run_id, result)
            if not result["requires_confirmation"]:
                repository.add_message(session_id, "assistant", result["reply"], run_id=run_id)
            session_memory.refresh_session_memory(
                session_id,
                run_id=run_id,
                intent=result["intent"],
                route=result["route"],
                status=result["status"],
                reply=result["reply"],
                pending_actions=result["pending_actions"],
                retrieval_result=None,
            )

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Run-Id": run_id,
            "Access-Control-Expose-Headers": "X-Run-Id",
        },
    )


@router.post("/runs/{run_id}/confirm/stream")
async def confirm_run_stream(
    run_id: str,
    payload: RunConfirmationRequest,
    request: Request,
) -> StreamingResponse:
    """SSE endpoint: resume an interrupted run and stream the continuation.

    Same event format as /chat/stream.
    """
    repository = request.app.state.repository
    orchestrator = request.app.state.orchestrator
    session_memory = request.app.state.session_memory

    existing_run = repository.get_run(run_id)
    if existing_run is None:
        raise HTTPException(status_code=404, detail="Run not found.")
    if existing_run["status"] != "awaiting_confirmation":
        raise HTTPException(status_code=409, detail="Run is not waiting for confirmation.")

    session_id = existing_run["session_id"]

    async def event_generator():
        done_payload: dict = {}
        try:
            async for chunk in orchestrator.astream_resume(
                run_id,
                payload.approved,
            ):
                if chunk.get("type") == "done":
                    done_payload = chunk
                yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"
        except Exception as exc:
            detail = f"Agent confirm stream failed: {exc}"
            _record_run_failure(repository, run_id, detail)
            yield f"data: {json.dumps({'type': 'error', 'detail': detail})}\n\n"
            yield "data: [DONE]\n\n"
            return

        yield "data: [DONE]\n\n"

        if done_payload:
            result = {
                "session_id": session_id,
                "run_id": run_id,
                "intent": done_payload.get("intent"),
                "route": done_payload.get("route"),
                "status": done_payload.get("status", "completed"),
                "reply": done_payload.get("reply", ""),
                "requires_confirmation": False,
                "approval_status": done_payload.get("approval_status"),
                "execution_plan": done_payload.get("execution_plan"),
                "pending_actions": done_payload.get("pending_actions", []),
                "approval_requested_at": None,
                "approval_resolved_at": None,
            }
            _persist_run_result(repository, run_id, result)
            repository.add_message(session_id, "assistant", result["reply"], run_id=run_id)
            session_memory.refresh_session_memory(
                session_id,
                run_id=run_id,
                intent=result["intent"],
                route=result["route"],
                status=result["status"],
                reply=result["reply"],
                pending_actions=result["pending_actions"],
                retrieval_result=None,
            )

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        },
    )
