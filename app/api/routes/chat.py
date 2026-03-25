import asyncio
from typing import Literal
from uuid import uuid4

from fastapi import APIRouter, HTTPException, Query, Request, status
from fastapi.responses import StreamingResponse

from app.api.schemas import (
    ChatRequest,
    ChatResponse,
    KnowledgeDebugIndexRequest,
    KnowledgeDebugIndexResponse,
    KnowledgeSearchRequest,
    KnowledgeSearchResponse,
    RunEventResponse,
    RunConfirmationRequest,
    RunDetailResponse,
    SessionDetailResponse,
    UserMemoryPatchRequest,
    UserMemoryProfileResponse,
)
from app.services.knowledge_index import DuplicateKnowledgeVideoError


router = APIRouter(prefix="/api", tags=["agent"])
TERMINAL_STATUSES = {"completed", "cancelled", "failed"}


def _persist_run_result(
    repository,
    run_id: str,
    result: dict[str, object],
) -> None:
    repository.update_run(
        run_id,
        intent=result["intent"],
        route=result["route"],
        langsmith_thread_id=result.get("langsmith_thread_id"),
        langsmith_thread_url=result.get("langsmith_thread_url"),
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


@router.post("/chat", response_model=ChatResponse)
def chat(request: Request, payload: ChatRequest) -> ChatResponse:
    repository = request.app.state.repository
    orchestrator = request.app.state.orchestrator
    session_memory = request.app.state.session_memory

    session_id = payload.session_id or str(uuid4())
    session = repository.get_session(session_id)
    if session is not None:
        session_user_id = session.get("user_id")
        if session_user_id and payload.user_id and session_user_id != payload.user_id:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="Session is already bound to a different user_id.",
            )
        if payload.user_id and session_user_id is None:
            repository.set_session_user_id(session_id, payload.user_id)
        repository.touch_session(session_id)
        user_id = payload.user_id or session_user_id
    else:
        repository.create_session(session_id, user_id=payload.user_id)
        user_id = payload.user_id

    run_id = str(uuid4())
    repository.create_run(run_id, session_id, status="running")
    repository.add_message(session_id, "user", payload.message, run_id=run_id)
    session_context = session_memory.load_session_context(session_id)
    user_memory_context = None
    if user_id:
        user_memory_context = request.app.state.user_memory.build_context_message(user_id)

    try:
        result = orchestrator.invoke_chat(
            session_id=session_id,
            run_id=run_id,
            message=payload.message,
            messages=session_context["messages"],
            user_id=user_id,
            session_summary=session_context["session_summary"],
            recent_context=session_context["recent_context"],
            user_memory_context=user_memory_context,
        )
    except Exception as exc:
        detail = f"Agent run failed: {exc}"
        _record_run_failure(repository, run_id, detail)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=detail,
        ) from exc

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
    )
    return ChatResponse(**result)


@router.post("/runs/{run_id}/confirm", response_model=ChatResponse)
def confirm_run(
    run_id: str,
    payload: RunConfirmationRequest,
    request: Request,
) -> ChatResponse:
    repository = request.app.state.repository
    orchestrator = request.app.state.orchestrator
    session_memory = request.app.state.session_memory

    existing_run = repository.get_run(run_id)
    if existing_run is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Run not found.")
    if existing_run["status"] != "awaiting_confirmation":
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Run is not waiting for confirmation."
        )

    try:
        result = orchestrator.resume_run(run_id, payload.approved)
    except Exception as exc:
        detail = f"Agent run failed during confirmation: {exc}"
        _record_run_failure(repository, run_id, detail)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=detail,
        ) from exc
    if existing_run.get("langsmith_thread_url") and not result.get("langsmith_thread_url"):
        result["langsmith_thread_url"] = existing_run["langsmith_thread_url"]
    elif existing_run.get("langsmith_thread_url"):
        result["langsmith_thread_url"] = existing_run["langsmith_thread_url"]
    if existing_run.get("langsmith_thread_id") and not result.get("langsmith_thread_id"):
        result["langsmith_thread_id"] = existing_run["langsmith_thread_id"]
    elif existing_run.get("langsmith_thread_id"):
        result["langsmith_thread_id"] = existing_run["langsmith_thread_id"]
    _persist_run_result(repository, run_id, result)
    repository.add_message(existing_run["session_id"], "assistant", result["reply"], run_id=run_id)
    session_memory.refresh_session_memory(
        existing_run["session_id"],
        run_id=run_id,
        intent=result["intent"],
        route=result["route"],
        status=result["status"],
        reply=result["reply"],
        pending_actions=result["pending_actions"],
    )
    return ChatResponse(**result)


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
        langsmith_thread_id=existing_run["langsmith_thread_id"],
        langsmith_thread_url=existing_run["langsmith_thread_url"],
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
    knowledge_index = request.app.state.knowledge_index
    try:
        result = knowledge_index.search(payload.model_dump())
    except RuntimeError as exc:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc)) from exc
    return KnowledgeSearchResponse(**result)


@router.get("/runs/{run_id}/events")
async def stream_run_events(
    run_id: str,
    request: Request,
    follow: bool = Query(default=True),
) -> StreamingResponse:
    repository = request.app.state.repository
    existing_run = repository.get_run(run_id)
    if existing_run is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Run not found.")

    async def event_stream():
        last_sequence = 0
        idle_loops = 0
        first_pass_completed = False

        while True:
            if await request.is_disconnected():
                break

            events = repository.get_run_events(run_id, after_sequence=last_sequence)
            for event in events:
                last_sequence = event["sequence"]
                payload = RunEventResponse(**event).model_dump_json()
                yield f"id: {event['sequence']}\nevent: run_event\ndata: {payload}\n\n"
                idle_loops = 0

            current_run = repository.get_run(run_id)
            if current_run is None:
                break
            if current_run["status"] in TERMINAL_STATUSES:
                break
            if not follow and first_pass_completed:
                break

            first_pass_completed = True
            idle_loops += 1
            if idle_loops >= 20:
                yield ": keep-alive\n\n"
                idle_loops = 0
            await asyncio.sleep(0.25)

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        },
    )
