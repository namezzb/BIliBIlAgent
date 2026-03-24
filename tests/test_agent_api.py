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
