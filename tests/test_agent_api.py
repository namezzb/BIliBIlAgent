import json
from pathlib import Path

from fastapi.testclient import TestClient

from app.core.config import Settings
from app.db.repository import SQLiteRepository
from app.main import create_app
from app.services.llm import OpenAICompatibleLLM
from app.services.session_memory import SessionMemoryManager


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
        assert body["user_id"] is None
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


class FakeSummaryLLM:
    def summarize_conversation(self, messages: list[dict[str, str]]) -> str | None:
        return "Compressed summary from model."


def test_summary_is_injected_as_assistant_context_message(tmp_path: Path) -> None:
    repository = SQLiteRepository(tmp_path / "memory.db")
    repository.initialize()
    session_id = "session-1"
    repository.create_session(session_id)

    for index in range(5):
        repository.add_message(session_id, "user", f"user-{index}")
        repository.add_message(session_id, "assistant", f"assistant-{index}")

    manager = SessionMemoryManager(repository, FakeSummaryLLM())
    manager.refresh_session_memory(
        session_id,
        run_id="run-1",
        intent="general_chat",
        route="general_chat",
        status="completed",
        reply="assistant-4",
        pending_actions=[],
    )

    session_context = manager.load_session_context(session_id)

    assert session_context["session_summary"] == "Compressed summary from model."
    assert session_context["messages"][0]["role"] == "assistant"
    assert "Compressed summary from model." in session_context["messages"][0]["content"]


def test_user_memory_command_requires_user_id(tmp_path: Path) -> None:
    with build_client(tmp_path) as client:
        response = client.post("/api/chat", json={"message": "记住偏好: reply_language=zh-CN"})

        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "completed"
        assert "user_id" in body["reply"]


def test_user_memory_chat_command_persists_and_get_endpoint_returns_profile(tmp_path: Path) -> None:
    with build_client(tmp_path) as client:
        response = client.post(
            "/api/chat",
            json={"user_id": "user-1", "message": "记住偏好: reply_language=zh-CN"},
        )

        assert response.status_code == 200
        assert "已保存长期记忆" in response.json()["reply"]

        detail = client.get("/api/users/user-1/memory")
        assert detail.status_code == 200
        body = detail.json()
        assert body["user_id"] == "user-1"
        assert body["preferences"]["reply_language"]["value"] == "zh-CN"
        assert body["preferences"]["reply_language"]["source_type"] == "chat_command"
        assert body["preferences"]["reply_language"]["source_text"] == "记住偏好: reply_language=zh-CN"
        assert body["preferences"]["reply_language"]["source_run_id"]
        assert body["preferences"]["reply_language"]["confirmed"] is True


def test_user_memory_rest_patch_is_partial_and_delete_removes_entry(tmp_path: Path) -> None:
    with build_client(tmp_path) as client:
        patch_one = client.patch(
            "/api/users/user-2/memory",
            json={"preferences": {"timezone": "Asia/Shanghai"}},
        )
        assert patch_one.status_code == 200
        assert patch_one.json()["preferences"]["timezone"]["source_type"] == "api"

        patch_two = client.patch(
            "/api/users/user-2/memory",
            json={"aliases": {"me": "张三"}},
        )
        assert patch_two.status_code == 200
        body = patch_two.json()
        assert body["preferences"]["timezone"]["value"] == "Asia/Shanghai"
        assert body["aliases"]["me"]["value"] == "张三"

        deleted = client.delete("/api/users/user-2/memory/aliases/me")
        assert deleted.status_code == 200
        deleted_body = deleted.json()
        assert "me" not in deleted_body["aliases"]
        assert deleted_body["preferences"]["timezone"]["value"] == "Asia/Shanghai"


def test_user_memory_is_loaded_across_sessions_for_same_user(tmp_path: Path) -> None:
    with build_client(tmp_path) as client:
        client.post(
            "/api/chat",
            json={"user_id": "user-3", "message": "记住偏好: reply_language=zh-CN"},
        )

        second_response = client.post(
            "/api/chat",
            json={"user_id": "user-3", "message": "hello again"},
        )
        run_id = second_response.json()["run_id"]

        with client.stream("GET", f"/api/runs/{run_id}/events?follow=false") as stream_response:
            assert stream_response.status_code == 200
            events = []
            for line in stream_response.iter_lines():
                if line.startswith("data: "):
                    events.append(json.loads(line[6:]))

        context_event = next(event for event in events if event["type"] == "context_loaded")
        assert context_event["payload"]["user_memory_present"] is True


def test_view_user_memory_command_returns_summary(tmp_path: Path) -> None:
    with build_client(tmp_path) as client:
        client.patch(
            "/api/users/user-4/memory",
            json={"default_scopes": {"favorite_folder": "ai-learning"}},
        )

        response = client.post(
            "/api/chat",
            json={"user_id": "user-4", "message": "查看长期记忆"},
        )

        assert response.status_code == 200
        body = response.json()
        assert "当前长期记忆" in body["reply"]
        assert "favorite_folder = ai-learning" in body["reply"]


class RecordingLLM(OpenAICompatibleLLM):
    def __init__(self) -> None:
        super().__init__(
            api_key="test-key",
            base_url="https://example.com/v1",
            model="test-model",
            summary_model="test-summary-model",
            embedding_model="test-embedding-model",
            system_prompt="base system prompt",
        )
        self.last_messages: list[dict[str, str]] = []

    def _create_chat_completion(
        self,
        *,
        messages: list[dict[str, str]],
        model: str,
        temperature: float,
    ) -> str | None:
        self.last_messages = messages
        return "ok"


def test_llm_chat_injects_user_memory_after_base_system_prompt() -> None:
    llm = RecordingLLM()

    result = llm.chat(
        [
            {"role": "assistant", "content": "[Conversation summary for context only]\nsummary"},
            {"role": "user", "content": "hello"},
        ],
        extra_system_messages=["long-term memory"],
    )

    assert result == "ok"
    assert llm.last_messages[0] == {"role": "system", "content": "base system prompt"}
    assert llm.last_messages[1] == {"role": "system", "content": "long-term memory"}
    assert llm.last_messages[2]["role"] == "assistant"
