import json
import sqlite3
from pathlib import Path

from fastapi.testclient import TestClient

from app.core.config import Settings
from app.db.repository import SQLiteRepository
from app.main import create_app
from app.services.knowledge_index import KnowledgeIndexService
from app.services.llm import OpenAICompatibleLLM
from app.services.session_memory import SessionMemoryManager


def build_client(tmp_path: Path) -> TestClient:
    settings = Settings(
        app_db_path=tmp_path / "app.db",
        checkpoint_db_path=tmp_path / "checkpoints.db",
        chroma_persist_dir=tmp_path / "chroma",
        data_dir=tmp_path,
    )
    app = create_app(settings)
    return TestClient(app)


class FakeVectorIndex:
    def __init__(self, *, fail_upsert: bool = False) -> None:
        self.fail_upsert = fail_upsert
        self.documents: dict[str, dict[str, object]] = {}

    def upsert(
        self,
        *,
        ids: list[str],
        embeddings: list[list[float]],
        documents: list[str],
        metadatas: list[dict[str, object]],
    ) -> None:
        if self.fail_upsert:
            raise RuntimeError("forced upsert failure")
        for vector_id, embedding, document, metadata in zip(
            ids,
            embeddings,
            documents,
            metadatas,
            strict=False,
        ):
            self.documents[vector_id] = {
                "embedding": embedding,
                "document": document,
                "metadata": metadata,
            }

    def delete(self, ids: list[str]) -> None:
        for vector_id in ids:
            self.documents.pop(vector_id, None)

    def query(self, *, query_embedding: list[float], n_results: int) -> dict[str, object]:
        selected = list(self.documents.items())[:n_results]
        return {
            "ids": [[item[0] for item in selected]],
            "distances": [[0.05 * index for index, _ in enumerate(selected)]],
        }

    def count(self) -> int:
        return len(self.documents)


def fake_embed_texts(texts: list[str]) -> list[list[float]]:
    return [[float(index)] for index, _ in enumerate(texts)]


def failing_embed_texts(texts: list[str]) -> list[list[float]]:
    raise RuntimeError("missing embedding credentials")


def install_fake_knowledge_index(
    client: TestClient,
    *,
    embedder=fake_embed_texts,
    vector_index: FakeVectorIndex | None = None,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
) -> FakeVectorIndex:
    active_vector_index = vector_index or FakeVectorIndex()
    client.app.state.knowledge_index = KnowledgeIndexService(
        repository=client.app.state.repository,
        vector_index=active_vector_index,
        embed_texts=embedder,
        embedding_model="test-embedding-model",
        embedding_version="test-v1",
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    return active_vector_index


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


def build_knowledge_payload() -> dict[str, object]:
    return {
        "favorite_folders": [
            {
                "favorite_folder_id": "fav-ai",
                "title": "AI Favorites",
                "intro": "AI topics",
            },
            {
                "favorite_folder_id": "fav-db",
                "title": "Database Favorites",
                "intro": "DB topics",
            },
        ],
        "videos": [
            {
                "video_id": "video-rag",
                "bvid": "BV1RAG",
                "title": "RAG Intro",
                "favorite_folder_ids": ["fav-ai", "fav-db"],
                "pages": [
                    {
                        "page_id": "page-rag-1",
                        "page_number": 1,
                        "title": "Overview",
                        "text_blocks": [
                            {
                                "text": "rag retrieval augmented generation basics",
                                "source_type": "subtitle",
                                "source_language": "zh-CN",
                                "start_ms": 0,
                                "end_ms": 2000,
                            },
                            {
                                "text": "vector database indexing practice",
                                "source_type": "subtitle",
                                "source_language": "zh-CN",
                                "start_ms": 2001,
                                "end_ms": 4000,
                            },
                        ],
                    }
                ],
            },
            {
                "video_id": "video-asr",
                "bvid": "BV1ASR",
                "title": "ASR Fallback",
                "favorite_folder_ids": ["fav-ai"],
                "pages": [
                    {
                        "page_id": "page-asr-1",
                        "page_number": 1,
                        "title": "ASR page",
                        "text_blocks": [
                            {
                                "text": "speech recognition fallback pipeline",
                                "source_type": "asr",
                                "source_language": "en",
                                "start_ms": 0,
                                "end_ms": 1800,
                            }
                        ],
                    }
                ],
            },
        ],
    }


def build_long_knowledge_payload() -> dict[str, object]:
    return {
        "favorite_folders": [
            {
                "favorite_folder_id": "fav-long",
                "title": "Long Favorites",
                "intro": "Long text",
            }
        ],
        "videos": [
            {
                "video_id": "video-long",
                "bvid": "BV1LONG",
                "title": "Long Video",
                "favorite_folder_ids": ["fav-long"],
                "pages": [
                    {
                        "page_id": "page-long-2",
                        "page_number": 2,
                        "title": "Part 2",
                        "text_blocks": [
                            {
                                "text": "B" * 1400,
                                "source_type": "subtitle",
                                "source_language": "zh-CN",
                                "start_ms": 0,
                                "end_ms": 1000,
                            }
                        ],
                    },
                    {
                        "page_id": "page-long-1",
                        "page_number": 1,
                        "title": "Part 1",
                        "text_blocks": [
                            {
                                "text": "A" * 1400,
                                "source_type": "subtitle",
                                "source_language": "zh-CN",
                                "start_ms": 0,
                                "end_ms": 1000,
                            }
                        ],
                    },
                ],
            }
        ],
    }


def test_debug_index_knowledge_persists_normalized_entities(tmp_path: Path) -> None:
    with build_client(tmp_path) as client:
        install_fake_knowledge_index(client)

        response = client.post("/api/knowledge/debug/index", json=build_knowledge_payload())

        assert response.status_code == 200
        body = response.json()
        assert body["favorite_folder_count"] == 2
        assert body["video_count"] == 2
        assert body["page_count"] == 2
        assert body["chunk_count"] == 2

        repository: SQLiteRepository = client.app.state.repository
        details = repository.get_knowledge_chunk_details(["video-rag:chunk:0", "video-asr:chunk:0"])
        detail_by_chunk = {detail["chunk_id"]: detail for detail in details}
        assert len(detail_by_chunk["video-rag:chunk:0"]["favorite_folders"]) == 2
        assert detail_by_chunk["video-rag:chunk:0"]["video"]["video_id"] == "video-rag"
        assert detail_by_chunk["video-rag:chunk:0"]["text"] == (
            "rag retrieval augmented generation basics\nvector database indexing practice"
        )
        assert detail_by_chunk["video-asr:chunk:0"]["source_type"] == "asr"


def test_knowledge_search_returns_filtered_hits_with_source_details(tmp_path: Path) -> None:
    with build_client(tmp_path) as client:
        install_fake_knowledge_index(client)
        client.post("/api/knowledge/debug/index", json=build_knowledge_payload())

        response = client.post(
            "/api/knowledge/search",
            json={
                "query": "rag",
                "top_k": 5,
                "favorite_folder_ids": ["fav-db"],
                "source_types": ["subtitle"],
            },
        )

        assert response.status_code == 200
        body = response.json()
        assert body["total_hits"] >= 1
        first_hit = body["hits"][0]
        assert first_hit["video"]["video_id"] == "video-rag"
        assert {folder["favorite_folder_id"] for folder in first_hit["favorite_folders"]} >= {
            "fav-db"
        }
        assert first_hit["source_type"] == "subtitle"
        assert "page" not in first_hit


def test_knowledge_search_returns_empty_hits_when_index_is_empty(tmp_path: Path) -> None:
    with build_client(tmp_path) as client:
        install_fake_knowledge_index(client)

        response = client.post("/api/knowledge/search", json={"query": "nothing"})

        assert response.status_code == 200
        assert response.json() == {"query": "nothing", "total_hits": 0, "hits": []}


def test_debug_index_fails_cleanly_when_embedding_generation_fails(tmp_path: Path) -> None:
    with build_client(tmp_path) as client:
        vector_index = install_fake_knowledge_index(client, embedder=failing_embed_texts)

        response = client.post("/api/knowledge/debug/index", json=build_knowledge_payload())

        assert response.status_code == 503
        assert "Embedding generation failed" in response.json()["detail"]
        assert vector_index.count() == 0
        repository: SQLiteRepository = client.app.state.repository
        assert repository.get_knowledge_chunk_details(["video-rag:chunk:0"]) == []


def test_debug_index_compensates_vector_writes_on_repository_failure(tmp_path: Path) -> None:
    with build_client(tmp_path) as client:
        vector_index = install_fake_knowledge_index(client)
        repository: SQLiteRepository = client.app.state.repository

        original_method = repository.upsert_knowledge_bundle

        def fail_upsert_knowledge_bundle(**kwargs):  # type: ignore[no-untyped-def]
            raise sqlite3.IntegrityError("forced sqlite failure")

        repository.upsert_knowledge_bundle = fail_upsert_knowledge_bundle  # type: ignore[assignment]
        try:
            response = client.post("/api/knowledge/debug/index", json=build_knowledge_payload())
        finally:
            repository.upsert_knowledge_bundle = original_method  # type: ignore[assignment]

        assert response.status_code == 500
        assert vector_index.count() == 0


def test_debug_index_rejects_duplicate_video_imports(tmp_path: Path) -> None:
    with build_client(tmp_path) as client:
        install_fake_knowledge_index(client)
        payload = build_knowledge_payload()

        first_response = client.post("/api/knowledge/debug/index", json=payload)
        second_response = client.post("/api/knowledge/debug/index", json=payload)

        assert first_response.status_code == 200
        assert second_response.status_code == 409
        assert "Videos already indexed" in second_response.json()["detail"]


def test_debug_index_chunks_long_video_text_by_video_with_overlap(tmp_path: Path) -> None:
    with build_client(tmp_path) as client:
        install_fake_knowledge_index(client, chunk_size=1000, chunk_overlap=200)

        response = client.post("/api/knowledge/debug/index", json=build_long_knowledge_payload())

        assert response.status_code == 200
        assert response.json()["chunk_count"] == 4

        repository: SQLiteRepository = client.app.state.repository
        details = repository.get_knowledge_chunk_details(
            [
                "video-long:chunk:0",
                "video-long:chunk:1",
                "video-long:chunk:2",
                "video-long:chunk:3",
            ]
        )
        chunk_texts = [detail["text"] for detail in details]

        assert chunk_texts[0].startswith("A" * 1000)
        assert chunk_texts[1].startswith("A" * 600)
        assert chunk_texts[0][-200:] == chunk_texts[1][:200]
        assert chunk_texts[2][-200:] == chunk_texts[3][:200]
        assert set(chunk_texts[0]) == {"A"}
        assert set(chunk_texts[1]) == {"A"}
        assert set(chunk_texts[2]) == {"B"}
        assert set(chunk_texts[3]) == {"B"}


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
