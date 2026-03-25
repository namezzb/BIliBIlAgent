from typing import Any

from app.agent.types import RunEvent


TERMINAL_STATUSES = {"completed", "cancelled", "failed"}


def aggregate_chat_response(run_id: str, events: list[RunEvent]) -> dict[str, Any]:
    intent: str | None = None
    route: str | None = None
    status = "running"
    reply = ""
    requires_confirmation = False
    approval_status: str | None = None
    pending_actions: list[dict[str, Any]] = []
    execution_plan: dict[str, Any] | None = None
    approval_requested_at: str | None = None
    approval_resolved_at: str | None = None

    for event in events:
        payload = event["payload"]
        event_type = event["type"]

        if event_type == "intent_classified":
            intent = str(payload.get("intent")) if payload.get("intent") is not None else intent
            route = str(payload.get("route")) if payload.get("route") is not None else route
        elif event_type == "response_prepared":
            reply = str(payload.get("reply", reply))
            route = str(payload.get("route")) if payload.get("route") is not None else route
            pending_actions = list(payload.get("pending_actions", pending_actions))
            execution_plan = payload.get("execution_plan") or execution_plan
        elif event_type == "confirmation_required":
            requires_confirmation = True
            status = "awaiting_confirmation"
            route = str(payload.get("route")) if payload.get("route") is not None else route
            pending_actions = list(payload.get("pending_actions", pending_actions))
            execution_plan = payload.get("execution_plan") or execution_plan
            approval_requested_at = event["timestamp"]
            if payload.get("reply"):
                reply = str(payload["reply"])
        elif event_type == "confirmation_resolved":
            approval_status = "approved" if payload.get("approved") else "rejected"
            route = str(payload.get("route")) if payload.get("route") is not None else route
            requires_confirmation = False
            pending_actions = list(payload.get("pending_actions", pending_actions))
            execution_plan = payload.get("execution_plan") or execution_plan
            approval_resolved_at = event["timestamp"]
            if not payload.get("approved"):
                status = "cancelled"
            if payload.get("reply"):
                reply = str(payload["reply"])
        elif event_type == "tool_execution_started":
            pending_actions = list(payload.get("pending_actions", pending_actions))
            execution_plan = payload.get("execution_plan") or execution_plan
        elif event_type == "tool_execution_finished":
            pending_actions = list(payload.get("pending_actions", pending_actions))
            execution_plan = payload.get("execution_plan") or execution_plan
            if payload.get("reply"):
                reply = str(payload["reply"])
        elif event_type == "run_completed":
            status = str(payload.get("status", status))
            pending_actions = list(payload.get("pending_actions", pending_actions))
            execution_plan = payload.get("execution_plan") or execution_plan
            if payload.get("reply"):
                reply = str(payload["reply"])
        elif event_type == "run_failed":
            status = "failed"
            pending_actions = list(payload.get("pending_actions", pending_actions))
            execution_plan = payload.get("execution_plan") or execution_plan
            if payload.get("reply"):
                reply = str(payload["reply"])

    if status not in TERMINAL_STATUSES and requires_confirmation:
        status = "awaiting_confirmation"
    elif status == "running" and events:
        status = "completed"

    return {
        "run_id": run_id,
        "intent": intent,
        "route": route,
        "status": status,
        "reply": reply,
        "requires_confirmation": requires_confirmation,
        "approval_status": approval_status,
        "execution_plan": execution_plan,
        "pending_actions": pending_actions,
        "approval_requested_at": approval_requested_at,
        "approval_resolved_at": approval_resolved_at,
    }
