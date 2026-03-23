from uuid import uuid4

from fastapi import APIRouter, HTTPException, Request, status

from app.api.schemas import (
    ChatRequest,
    ChatResponse,
    RunConfirmationRequest,
    RunDetailResponse,
)


router = APIRouter(prefix="/api", tags=["agent"])


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
        status=existing_run["status"],
        reply=existing_run["latest_reply"] or "",
        requires_confirmation=existing_run["requires_confirmation"],
        approval_status=existing_run["approval_status"],
        pending_actions=existing_run["pending_actions"],
        created_at=existing_run["created_at"],
        updated_at=existing_run["updated_at"],
        steps=repository.get_run_steps(run_id),
    )
