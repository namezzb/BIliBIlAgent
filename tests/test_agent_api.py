import json
from pathlib import Path

from fastapi.testclient import TestClient

from app.core.config import Settings
from app.main import create_app


def build_client(tmp_path: Path) -> TestClient:
    settings = Settings(
        app_db_path=tmp_path / "app.db",
        checkpoint_db_path=tmp_path / "checkpoints.db",
        data_dir=tmp_path,
    )
    app = create_app(settings)
    return TestClient(app)


def test_general_chat_returns_completed_when_llm_is_unconfigured(tmp_path: Path) -> None:
    with build_client(tmp_path) as client:
        response = client.post("/api/chat", json={"message": "hello"})

        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "completed"
        assert body["intent"] == "general_chat"
        assert body["route"] == "general_chat"
        assert body["requires_confirmation"] is False
        assert body["session_id"]
        assert body["run_id"]


def test_import_request_requires_confirmation_and_can_be_approved(tmp_path: Path) -> None:
    with build_client(tmp_path) as client:
        first_response = client.post("/api/chat", json={"message": "请帮我导入这个收藏夹"})
        assert first_response.status_code == 200
        first_body = first_response.json()
        assert first_body["status"] == "awaiting_confirmation"
        assert first_body["intent"] == "tool_request"
        assert first_body["route"] == "import_request"
        assert first_body["requires_confirmation"] is True
        assert first_body["pending_actions"]
        assert first_body["pending_actions"][0]["tool"] == "bilibili_import"

        confirm_response = client.post(
            f"/api/runs/{first_body['run_id']}/confirm",
            json={"approved": True},
        )
        assert confirm_response.status_code == 200
        confirm_body = confirm_response.json()
        assert confirm_body["status"] == "completed"
        assert confirm_body["approval_status"] == "approved"


def test_run_detail_returns_steps(tmp_path: Path) -> None:
    with build_client(tmp_path) as client:
        response = client.post("/api/chat", json={"message": "知识库里这个视频讲了什么"})
        run_id = response.json()["run_id"]

        detail_response = client.get(f"/api/runs/{run_id}")

        assert detail_response.status_code == 200
        body = detail_response.json()
        assert body["run_id"] == run_id
        assert body["route"] == "video_knowledge_query"
        assert body["event_count"] >= 4
        assert len(body["steps"]) >= 3


def test_completed_run_event_stream_replays_and_closes(tmp_path: Path) -> None:
    with build_client(tmp_path) as client:
        response = client.post("/api/chat", json={"message": "hello"})
        run_id = response.json()["run_id"]

        with client.stream("GET", f"/api/runs/{run_id}/events?follow=false") as stream_response:
            assert stream_response.status_code == 200
            events = []
            for line in stream_response.iter_lines():
                if line.startswith("data: "):
                    events.append(json.loads(line[6:]))

        event_types = [event["type"] for event in events]
        assert event_types[0] == "run_started"
        assert "intent_classified" in event_types
        assert event_types[-1] == "run_completed"
        assert events[2]["payload"]["route"] == "general_chat"


def test_tool_request_event_stream_contains_confirmation_event(tmp_path: Path) -> None:
    with build_client(tmp_path) as client:
        first_response = client.post("/api/chat", json={"message": "请帮我导入这个收藏夹"})
        run_id = first_response.json()["run_id"]

        with client.stream("GET", f"/api/runs/{run_id}/events?follow=false") as stream_response:
            assert stream_response.status_code == 200
            event_types = []
            for line in stream_response.iter_lines():
                if line.startswith("data: "):
                    payload = json.loads(line[6:])
                    event_types.append(payload["type"])
                    if payload["type"] == "confirmation_required":
                        assert payload["payload"]["route"] == "import_request"
                        break

        assert "confirmation_required" in event_types

        confirm_response = client.post(
            f"/api/runs/{run_id}/confirm",
            json={"approved": True},
        )
        assert confirm_response.status_code == 200

        with client.stream("GET", f"/api/runs/{run_id}/events?follow=false") as stream_response:
            replayed_event_types = []
            for line in stream_response.iter_lines():
                if line.startswith("data: "):
                    replayed_event_types.append(json.loads(line[6:])["type"])

        assert "tool_execution_started" in replayed_event_types
        assert "tool_execution_finished" in replayed_event_types
        assert replayed_event_types[-1] == "run_completed"


def test_retry_request_uses_retry_route_and_tool(tmp_path: Path) -> None:
    with build_client(tmp_path) as client:
        response = client.post("/api/chat", json={"message": "重试失败的视频导入"})

        assert response.status_code == 200
        body = response.json()
        assert body["intent"] == "tool_request"
        assert body["route"] == "retry_request"
        assert body["status"] == "awaiting_confirmation"
        assert body["pending_actions"][0]["tool"] == "bilibili_retry"


def test_favorite_knowledge_query_route_is_exposed(tmp_path: Path) -> None:
    with build_client(tmp_path) as client:
        response = client.post(
            "/api/chat",
            json={"message": "这个收藏夹里有哪些讲向量数据库的相关视频"},
        )

        assert response.status_code == 200
        body = response.json()
        assert body["intent"] == "knowledge_query"
        assert body["route"] == "favorite_knowledge_query"
        assert body["status"] == "completed"


def test_session_detail_returns_history_and_recent_context(tmp_path: Path) -> None:
    with build_client(tmp_path) as client:
        first = client.post("/api/chat", json={"message": "hello"})
        session_id = first.json()["session_id"]
        client.post("/api/chat", json={"session_id": session_id, "message": "我刚刚问了什么？"})

        detail = client.get(f"/api/sessions/{session_id}")

        assert detail.status_code == 200
        body = detail.json()
        assert body["session_id"] == session_id
        assert len(body["messages"]) == 4
        assert body["recent_context"]["last_run_id"]
        assert body["recent_context"]["last_user_message"] == "我刚刚问了什么？"


def test_session_follow_up_uses_previous_message_when_no_llm_is_configured(tmp_path: Path) -> None:
    with build_client(tmp_path) as client:
        first = client.post("/api/chat", json={"message": "我最喜欢的水果是什么？"})
        session_id = first.json()["session_id"]

        second = client.post(
            "/api/chat",
            json={"session_id": session_id, "message": "我刚刚问了什么？"},
        )

        assert second.status_code == 200
        body = second.json()
        assert "我最喜欢的水果是什么？" in body["reply"]


def test_long_session_generates_summary_and_sessions_are_isolated(tmp_path: Path) -> None:
    with build_client(tmp_path) as client:
        first = client.post("/api/chat", json={"message": "第一句"})
        session_a = first.json()["session_id"]
        session_b = client.post("/api/chat", json={"message": "另一条会话"}).json()["session_id"]

        for content in ["第二句", "第三句", "第四句", "第五句"]:
            client.post("/api/chat", json={"session_id": session_a, "message": content})

        detail_a = client.get(f"/api/sessions/{session_a}")
        detail_b = client.get(f"/api/sessions/{session_b}")

        assert detail_a.status_code == 200
        assert detail_b.status_code == 200
        body_a = detail_a.json()
        body_b = detail_b.json()
        assert body_a["summary_text"]
        assert len(body_a["messages"]) > len(body_b["messages"])
        assert all(message["content"] != "另一条会话" for message in body_a["messages"])
