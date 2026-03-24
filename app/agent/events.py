from typing import Any

from app.agent.types import RunEvent


TERMINAL_STATUSES = {"completed", "cancelled", "failed"}


def aggregate_chat_response(run_id: str, events: list[RunEvent]) -> dict[str, Any]:
    intent: str | None = None
    status = "running"
    reply = ""
    requires_confirmation = False
    approval_status: str | None = None
    pending_actions: list[dict[str, Any]] = []

    for event in events:
        payload = event["payload"]
        event_type = event["type"]

        if event_type == "intent_classified":
            intent = str(payload.get("intent")) if payload.get("intent") is not None else intent
        elif event_type == "response_prepared":
            reply = str(payload.get("reply", reply))
            pending_actions = list(payload.get("pending_actions", pending_actions))
        elif event_type == "confirmation_required":
            requires_confirmation = True
            status = "awaiting_confirmation"
            pending_actions = list(payload.get("pending_actions", pending_actions))
            if payload.get("reply"):
                reply = str(payload["reply"])
        elif event_type == "confirmation_resolved":
            approval_status = "approved" if payload.get("approved") else "rejected"
            requires_confirmation = False
            if not payload.get("approved"):
                status = "cancelled"
            if payload.get("reply"):
                reply = str(payload["reply"])
        elif event_type == "tool_execution_finished":
            if payload.get("reply"):
                reply = str(payload["reply"])
        elif event_type == "run_completed":
            status = str(payload.get("status", status))
            if payload.get("reply"):
                reply = str(payload["reply"])
        elif event_type == "run_failed":
            status = "failed"
            if payload.get("reply"):
                reply = str(payload["reply"])

    if status not in TERMINAL_STATUSES and requires_confirmation:
        status = "awaiting_confirmation"
    elif status == "running" and events:
        status = "completed"

    return {
        "run_id": run_id,
        "intent": intent,
        "status": status,
        "reply": reply,
        "requires_confirmation": requires_confirmation,
        "approval_status": approval_status,
        "pending_actions": pending_actions,
    }
