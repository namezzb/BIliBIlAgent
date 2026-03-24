import asyncio
from uuid import uuid4

from fastapi import APIRouter, HTTPException, Query, Request, status
from fastapi.responses import StreamingResponse

from app.api.schemas import (
    ChatRequest,
    ChatResponse,
    RunEventResponse,
    RunConfirmationRequest,
    RunDetailResponse,
)


router = APIRouter(prefix="/api", tags=["agent"])
TERMINAL_STATUSES = {"completed", "cancelled", "failed"}


@router.post("/chat", response_model=ChatResponse)
def chat(request: Request, payload: ChatRequest) -> ChatResponse:
    repository = request.app.state.repository
    orchestrator = request.app.state.orchestrator

    session_id = payload.session_id or str(uuid4())
    if repository.session_exists(session_id):
        repository.touch_session(session_id)
    else:
        repository.create_session(session_id)

    run_id = str(uuid4())
    repository.create_run(run_id, session_id, status="running")
    repository.add_message(session_id, "user", payload.message, run_id=run_id)
    messages = [
        {"role": item["role"], "content": item["content"]}
        for item in repository.get_messages(session_id)
    ]

    result = orchestrator.invoke_chat(
        session_id=session_id,
        run_id=run_id,
        message=payload.message,
        messages=messages,
    )
    repository.update_run(
        run_id,
        intent=result["intent"],
        route=result["route"],
        status=result["status"],
        requires_confirmation=result["requires_confirmation"],
        approval_status=result["approval_status"],
        latest_reply=result["reply"],
        pending_actions=result["pending_actions"],
    )
    repository.add_message(session_id, "assistant", result["reply"], run_id=run_id)
    return ChatResponse(**result)


@router.post("/runs/{run_id}/confirm", response_model=ChatResponse)
def confirm_run(
    run_id: str,
    payload: RunConfirmationRequest,
    request: Request,
) -> ChatResponse:
    repository = request.app.state.repository
    orchestrator = request.app.state.orchestrator

    existing_run = repository.get_run(run_id)
    if existing_run is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Run not found.")
    if existing_run["status"] != "awaiting_confirmation":
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Run is not waiting for confirmation."
        )

    result = orchestrator.resume_run(run_id, payload.approved)
    repository.update_run(
        run_id,
        intent=result["intent"],
        route=result["route"],
        status=result["status"],
        requires_confirmation=result["requires_confirmation"],
        approval_status=result["approval_status"],
        latest_reply=result["reply"],
        pending_actions=result["pending_actions"],
    )
    repository.add_message(existing_run["session_id"], "assistant", result["reply"], run_id=run_id)
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
        status=existing_run["status"],
        reply=existing_run["latest_reply"] or "",
        requires_confirmation=existing_run["requires_confirmation"],
        approval_status=existing_run["approval_status"],
        pending_actions=existing_run["pending_actions"],
        created_at=existing_run["created_at"],
        updated_at=existing_run["updated_at"],
        event_count=repository.get_run_event_count(run_id),
        steps=repository.get_run_steps(run_id),
    )


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
