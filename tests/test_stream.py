"""
Unit tests for the native graph.astream() streaming path.

Covers:
  1. AgentOrchestrator.astream_chat  — token / node / done chunks
  2. AgentOrchestrator.astream_chat  — interrupt chunk when approval required
  3. AgentOrchestrator.astream_resume — resume after interrupt
  4. AgentOrchestrator.invoke_chat   — sync wrapper builds response from final_state
  5. POST /api/chat/stream           — SSE endpoint emits correct event lines
  6. POST /api/runs/{run_id}/confirm/stream — SSE resume endpoint
"""
import asyncio
import json
from typing import Any, AsyncGenerator
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest
from httpx import AsyncClient
from fastapi import FastAPI

# ── helpers ──────────────────────────────────────────────────────────────────

def _make_app(orchestrator, repository, session_memory, user_memory):
    """Minimal FastAPI app wired with test doubles."""
    from app.main import create_app
    from langgraph.checkpoint.memory import MemorySaver

    app = create_app(checkpointer=MemorySaver())
    app.state.orchestrator = orchestrator
    app.state.repository = repository
    app.state.session_memory = session_memory
    app.state.user_memory = user_memory
    return app


def _make_repository(run_id: str, session_id: str):
    repo = MagicMock()
    repo.get_run.return_value = {
        "run_id": run_id,
        "session_id": session_id,
        "status": "awaiting_confirmation",
        "intent": "tool_request",
        "route": "import_request",
        "requires_confirmation": True,
        "approval_status": None,
        "latest_reply": "needs approval",
        "pending_actions": [],
        "execution_plan": None,
        "approval_requested_at": None,
        "approval_resolved_at": None,
        "created_at": "2026-01-01T00:00:00",
        "updated_at": "2026-01-01T00:00:00",
    }
    repo.get_session.return_value = {"session_id": session_id, "user_id": None, "summary_text": None, "recent_context": {}}
    repo.create_run.return_value = None
    repo.add_message.return_value = str(uuid4())
    repo.update_run.return_value = None
    repo.get_run_event_count.return_value = 0
    repo.get_run_steps.return_value = []
    repo.upsert_run_step.return_value = None
    repo.append_run_event.return_value = None
    return repo


def _make_session_memory():
    sm = MagicMock()
    sm.load_session_context.return_value = {
        "messages": [],
        "session_summary": None,
        "recent_context": {},
    }
    sm.refresh_session_memory.return_value = None
    return sm


# ── 1. astream_chat — normal flow ────────────────────────────────────────────

@pytest.mark.asyncio
async def test_astream_chat_yields_token_and_done():
    """
    When the graph emits a messages chunk then an updates chunk,
    astream_chat should yield one 'token' chunk followed by a 'done' chunk.
    """
    from app.agent.service import AgentOrchestrator

    run_id = str(uuid4())
    session_id = str(uuid4())

    # Build minimal LangGraph-style chunks
    from langchain_core.messages import AIMessageChunk

    token_msg = AIMessageChunk(content="Hello")
    updates_chunk = {"finalize_run": {"status": "completed", "response": "Hello", "route": "general_chat", "intent": "general_chat"}}

    async def fake_astream(input_, config, stream_mode):
        # Simulate ("messages", (msg, metadata)) then ("updates", data)
        yield ("messages", (token_msg, {}))
        yield ("updates", updates_chunk)

    orch = MagicMock(spec=AgentOrchestrator)
    orch.graph = MagicMock()
    orch.graph.astream = fake_astream
    orch.graph.aget_state = AsyncMock(return_value=MagicMock(values={"status": "completed", "response": "Hello", "route": "general_chat", "intent": "general_chat"}))
    # Call the real astream_chat method bound to our mock
    orch.astream_chat = AgentOrchestrator.astream_chat.__get__(orch)

    chunks = []
    async for chunk in orch.astream_chat(
        session_id=session_id,
        run_id=run_id,
        message="hi",
        messages=[],
    ):
        chunks.append(chunk)

    types = [c["type"] for c in chunks]
    assert "token" in types
    assert types[-1] == "done"
    token_chunks = [c for c in chunks if c["type"] == "token"]
    assert token_chunks[0]["content"] == "Hello"
    done = chunks[-1]
    assert done["run_id"] == run_id


# ── 2. astream_chat — interrupt ───────────────────────────────────────────────

@pytest.mark.asyncio
async def test_astream_chat_yields_interrupt_chunk():
    """
    When the graph emits __interrupt__ in an updates chunk,
    astream_chat should yield an 'interrupt' chunk.
    """
    from app.agent.service import AgentOrchestrator
    from langgraph.types import Interrupt

    run_id = str(uuid4())
    session_id = str(uuid4())

    interrupt_value = {"question": "Approve?", "execution_plan": {"goal": "import", "steps": [], "tool_calls": [], "summary": ""}, "pending_actions": []}

    async def fake_astream(input_, config, stream_mode):
        yield ("updates", {"__interrupt__": [Interrupt(value=interrupt_value)]})

    orch = MagicMock(spec=AgentOrchestrator)
    orch.graph = MagicMock()
    orch.graph.astream = fake_astream
    orch.astream_chat = AgentOrchestrator.astream_chat.__get__(orch)

    chunks = []
    async for chunk in orch.astream_chat(
        session_id=session_id,
        run_id=run_id,
        message="导入收藏夹",
        messages=[],
    ):
        chunks.append(chunk)

    interrupt_chunks = [c for c in chunks if c["type"] == "interrupt"]
    assert len(interrupt_chunks) == 1
    assert interrupt_chunks[0]["data"] == interrupt_value
    # done chunk should still be last
    assert chunks[-1]["type"] == "done"


# ── 3. astream_resume ─────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_astream_resume_yields_token_and_done():
    from app.agent.service import AgentOrchestrator
    from langchain_core.messages import AIMessageChunk

    run_id = str(uuid4())
    token_msg = AIMessageChunk(content="Done!")

    async def fake_astream(command, config, stream_mode):
        yield ("messages", (token_msg, {}))
        yield ("updates", {"finalize_run": {"status": "completed", "response": "Done!", "approval_status": "approved"}})

    orch = MagicMock(spec=AgentOrchestrator)
    orch.graph = MagicMock()
    orch.graph.astream = fake_astream
    orch.astream_resume = AgentOrchestrator.astream_resume.__get__(orch)

    chunks = []
    async for chunk in orch.astream_resume(run_id, approved=True):
        chunks.append(chunk)

    types = [c["type"] for c in chunks]
    assert "token" in types
    assert types[-1] == "done"
    done = chunks[-1]
    assert done["approval_status"] == "approved"
    assert done["requires_confirmation"] is False


# ── 4. invoke_chat — builds response from final_state, no events.py ──────────

@pytest.mark.asyncio
async def test_invoke_chat_builds_response_from_state():
    """
    invoke_chat must NOT call aggregate_chat_response or append_run_event.
    It should build the response from the graph final_state directly.
    """
    from app.agent.service import AgentOrchestrator

    run_id = str(uuid4())
    session_id = str(uuid4())

    final_state = {
        "status": "completed",
        "response": "42",
        "intent": "general_chat",
        "route": "general_chat",
        "requires_confirmation": False,
        "approval_status": None,
        "pending_actions": [],
        "execution_plan": None,
    }

    async def fake_astream(input_, config, stream_mode):
        yield {"finalize_run": final_state}

    repo = MagicMock()
    repo.get_run.return_value = None
    runtime_audit = MagicMock()
    runtime_audit.environment = "test"
    runtime_audit.app_name = "test"
    runtime_audit.trace_request = MagicMock()
    trace_ctx = MagicMock()
    trace_ctx.__enter__ = MagicMock(return_value=trace_ctx)
    trace_ctx.__exit__ = MagicMock(return_value=False)
    trace_ctx.add_metadata = MagicMock()
    trace_ctx.end = MagicMock()
    runtime_audit.trace_request.return_value = trace_ctx
    runtime_audit.sanitize_payload = lambda x: x
    runtime_audit.build_reference = MagicMock(return_value={})

    orch = MagicMock(spec=AgentOrchestrator)
    orch.repository = repo
    orch.runtime_audit = runtime_audit
    orch.graph = MagicMock()
    orch.graph.astream = fake_astream
    orch.graph.aget_state = AsyncMock(return_value=MagicMock(values=final_state))
    orch._state_to_response = AgentOrchestrator._state_to_response.__get__(orch)
    orch._execution_goal = AgentOrchestrator._execution_goal.__get__(orch)
    orch._planned_tool_names = AgentOrchestrator._planned_tool_names.__get__(orch)
    orch.invoke_chat = AgentOrchestrator.invoke_chat.__get__(orch)

    result = await orch.invoke_chat(
        session_id=session_id,
        run_id=run_id,
        message="what is 6x7?",
        messages=[],
    )

    assert result["status"] == "completed"
    assert result["reply"] == "42"
    assert result["route"] == "general_chat"
    # Must NOT have called append_run_event (event system is gone)
    repo.append_run_event.assert_not_called()


# ── 5. POST /api/chat/stream SSE endpoint ─────────────────────────────────────

@pytest.mark.asyncio
async def test_chat_stream_endpoint_emits_sse_lines():
    """
    POST /api/chat/stream should return text/event-stream.
    Each SSE line should be parseable JSON with correct chunk types.
    """
    from app.main import create_app
    from langgraph.checkpoint.memory import MemorySaver

    run_id = str(uuid4())
    session_id = str(uuid4())

    async def fake_astream_chat(**kwargs) -> AsyncGenerator[dict, None]:
        yield {"type": "token", "content": "Hi"}
        yield {"type": "node", "node": "finalize_run", "data": {}}
        yield {
            "type": "done",
            "run_id": run_id,
            "status": "completed",
            "reply": "Hi",
            "intent": "general_chat",
            "route": "general_chat",
            "requires_confirmation": False,
            "approval_status": None,
            "execution_plan": None,
            "pending_actions": [],
        }

    orchestrator = MagicMock()
    orchestrator.astream_chat = fake_astream_chat

    repo = _make_repository(run_id, session_id)
    repo.get_session.return_value = {"session_id": session_id, "user_id": None, "summary_text": None, "recent_context": {}}
    sm = _make_session_memory()
    um = MagicMock()
    um.build_context_message.return_value = None

    app = create_app(checkpointer=MemorySaver())
    app.state.orchestrator = orchestrator
    app.state.repository = repo
    app.state.session_memory = sm
    app.state.user_memory = um
    # Stub other required state
    app.state.bilibili_favorite_folder_service = MagicMock()
    app.state.bilibili_import_pipeline = MagicMock()
    app.state.bilibili_import_tool = MagicMock()
    app.state.knowledge_index = MagicMock()
    app.state.knowledge_retrieval = MagicMock()
    app.state.knowledge_retrieval_tool = MagicMock()
    app.state.knowledge_qa = MagicMock()
    app.state.runtime_audit = MagicMock()

    from httpx import ASGITransport
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.post(
            "/api/chat/stream",
            json={"session_id": session_id, "message": "hello"},
        )

    assert resp.status_code == 200
    assert "text/event-stream" in resp.headers["content-type"]

    lines = [l for l in resp.text.split("\n") if l.startswith("data: ")]
    assert len(lines) >= 2  # at least token + done + [DONE]

    parsed = []
    for line in lines:
        raw = line[6:].strip()
        if raw != "[DONE]":
            parsed.append(json.loads(raw))

    types = [c["type"] for c in parsed]
    assert "token" in types
    assert "done" in types


# ── 6. POST /api/runs/{run_id}/confirm/stream SSE endpoint ───────────────────

@pytest.mark.asyncio
async def test_confirm_stream_endpoint_emits_sse_lines():
    from app.main import create_app
    from langgraph.checkpoint.memory import MemorySaver

    run_id = str(uuid4())
    session_id = str(uuid4())

    async def fake_astream_resume(rid, approved) -> AsyncGenerator[dict, None]:
        yield {"type": "token", "content": "Executing..."}
        yield {
            "type": "done",
            "run_id": rid,
            "status": "completed",
            "reply": "Import done.",
            "intent": "tool_request",
            "route": "import_request",
            "requires_confirmation": False,
            "approval_status": "approved",
            "execution_plan": None,
            "pending_actions": [],
        }

    orchestrator = MagicMock()
    orchestrator.astream_resume = fake_astream_resume

    repo = _make_repository(run_id, session_id)
    sm = _make_session_memory()
    um = MagicMock()
    um.build_context_message.return_value = None

    app = create_app(checkpointer=MemorySaver())
    app.state.orchestrator = orchestrator
    app.state.repository = repo
    app.state.session_memory = sm
    app.state.user_memory = um
    app.state.bilibili_favorite_folder_service = MagicMock()
    app.state.bilibili_import_pipeline = MagicMock()
    app.state.bilibili_import_tool = MagicMock()
    app.state.knowledge_index = MagicMock()
    app.state.knowledge_retrieval = MagicMock()
    app.state.knowledge_retrieval_tool = MagicMock()
    app.state.knowledge_qa = MagicMock()
    app.state.runtime_audit = MagicMock()

    from httpx import ASGITransport
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.post(
            f"/api/runs/{run_id}/confirm/stream",
            json={"approved": True},
        )

    assert resp.status_code == 200
    assert "text/event-stream" in resp.headers["content-type"]

    lines = [l for l in resp.text.split("\n") if l.startswith("data: ")]
    parsed = []
    for line in lines:
        raw = line[6:].strip()
        if raw != "[DONE]":
            parsed.append(json.loads(raw))

    types = [c["type"] for c in parsed]
    assert "token" in types
    done_chunks = [c for c in parsed if c["type"] == "done"]
    assert done_chunks[0]["approval_status"] == "approved"
    assert done_chunks[0]["status"] == "completed"
