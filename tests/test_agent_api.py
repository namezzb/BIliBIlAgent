import json
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from uuid import uuid4

import httpx
import pytest
from fastapi.testclient import TestClient

from app.agent.tools import build_bilibili_import_tool, build_tool_registry
from app.core.config import Settings
from app.db.repository import SQLiteRepository
from app.main import create_app
from app.services.bilibili_favorites import (
    BilibiliFavoriteFolderAuthError,
    BilibiliFavoriteFolderResponseError,
    BilibiliFavoriteFolderService,
    BilibiliFavoriteFolderUpstreamError,
)
from app.services.bilibili_import import BilibiliImportPipeline
from app.services.knowledge_index import KnowledgeIndexService
from app.services.llm import OpenAICompatibleLLM
from app.services.runtime_audit import LangSmithRuntimeAudit
from app.services.session_memory import SessionMemoryManager


class FakeTraceRun:
    def __init__(self, name: str, *, metadata: dict[str, object] | None = None) -> None:
        self.id = str(uuid4())
        self.name = name
        self.metadata = metadata or {}
        self.outputs: dict[str, object] | None = None

    def add_metadata(self, metadata: dict[str, object]) -> None:
        self.metadata.update(metadata)

    def end(self, outputs: dict[str, object] | None = None) -> None:
        self.outputs = outputs

    def get_url(self) -> str:
        return f"https://smith.example.test/runs/{self.id}"


class FakeRuntimeAudit:
    def __init__(self) -> None:
        self.app_name = "BIliBIlAgent API"
        self.environment = "test"
        self.requests: list[dict[str, object]] = []
        self.spans: list[dict[str, object]] = []

    @contextmanager
    def trace_request(
        self,
        *,
        name: str,
        inputs: dict[str, object],
        metadata: dict[str, object],
        tags: list[str] | None = None,
    ):
        run = FakeTraceRun(name, metadata=metadata)
        self.requests.append(
            {"name": name, "inputs": inputs, "metadata": metadata, "tags": tags or [], "run": run}
        )
        yield run

    @contextmanager
    def trace_span(
        self,
        *,
        name: str,
        run_type: str,
        inputs: dict[str, object],
        metadata: dict[str, object] | None = None,
        tags: list[str] | None = None,
    ):
        run = FakeTraceRun(name, metadata=metadata)
        self.spans.append(
            {
                "name": name,
                "run_type": run_type,
                "inputs": inputs,
                "metadata": metadata or {},
                "tags": tags or [],
                "run": run,
            }
        )
        yield run

    def build_reference(
        self,
        *,
        run_id: str,
        trace_run: FakeTraceRun,
        existing_url: str | None = None,
    ) -> dict[str, str | None]:
        return {
            "langsmith_thread_id": run_id,
            "langsmith_thread_url": existing_url or trace_run.get_url(),
        }

    def sanitize_payload(self, value):
        return value

    def close(self) -> None:
        return None


def build_client(
    tmp_path: Path,
    *,
    runtime_audit: FakeRuntimeAudit | None = None,
    raise_server_exceptions: bool = True,
) -> TestClient:
    settings = Settings(
        app_db_path=tmp_path / "app.db",
        checkpoint_db_path=tmp_path / "checkpoints.db",
        chroma_persist_dir=tmp_path / "chroma",
        data_dir=tmp_path,
        langsmith_tracing=True,
        langsmith_api_key="test-langsmith-key",
        langsmith_project="bilibilagent-tests",
    )
    app = create_app(settings, runtime_audit=runtime_audit or FakeRuntimeAudit())
    return TestClient(app, raise_server_exceptions=raise_server_exceptions)


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


class FakeBilibiliFavoriteFolderService:
    def __init__(
        self
    ) -> None:
        self.start_result = {
            "qr_url": "https://passport.bilibili.com/h5-app/passport/login/scan",
            "qrcode_key": "qr-key-1",
            "expires_in_seconds": 180,
        }
        self.poll_result = {
            "status": "success",
            "message": "登录成功。",
            "cookie": "DedeUserID=1; SESSDATA=test",
            "refresh_token": "refresh-1",
            "account": {"mid": 1, "uname": "tester", "is_login": True},
        }
        self.folder_result = {
            "account": {"mid": 1, "uname": "tester", "is_login": True},
            "total": 1,
            "folders": [
                {
                    "favorite_folder_id": "101",
                    "title": "AI Favorites",
                    "intro": "AI folder",
                    "cover": "https://img.example.test/cover.jpg",
                    "media_count": 12,
                    "folder_attr": 0,
                    "owner_mid": 1,
                }
            ],
        }
        self.folder_items_result = {
            "account": {"mid": 1, "uname": "tester", "is_login": True},
            "folder": {
                "favorite_folder_id": "101",
                "title": "AI Favorites",
                "intro": "AI folder",
                "cover": "https://img.example.test/cover.jpg",
                "media_count": 3,
                "folder_attr": 0,
                "owner_mid": 1,
            },
            "page": 1,
            "page_size": 20,
            "total": 3,
            "total_pages": 1,
            "has_more": False,
            "items": [
                {
                    "item_id": "fav-video-1",
                    "favorite_folder_id": "101",
                    "item_type": 2,
                    "media_type": 2,
                    "selectable": True,
                    "unsupported_reason": None,
                    "video_id": "BV1SUB",
                    "aid": 1001,
                    "bvid": "BV1SUB",
                    "title": "Subtitle Video",
                    "cover": "https://img.example.test/sub.jpg",
                    "intro": "has subtitle",
                    "duration": 120,
                    "upper_mid": 1,
                    "upper_name": "tester",
                    "fav_time": 1710000000,
                    "pubtime": 1700000000,
                },
                {
                    "item_id": "fav-article-1",
                    "favorite_folder_id": "101",
                    "item_type": 12,
                    "media_type": 12,
                    "selectable": False,
                    "unsupported_reason": "Unsupported favorite item type: 12",
                    "video_id": "article-1",
                    "aid": None,
                    "bvid": None,
                    "title": "Unsupported Article",
                    "cover": None,
                    "intro": "unsupported",
                    "duration": 0,
                    "upper_mid": 1,
                    "upper_name": "tester",
                    "fav_time": 1710000010,
                    "pubtime": 1700000010,
                },
                {
                    "item_id": "fav-video-2",
                    "favorite_folder_id": "101",
                    "item_type": 2,
                    "media_type": 2,
                    "selectable": True,
                    "unsupported_reason": None,
                    "video_id": "BV1ASR",
                    "aid": 1002,
                    "bvid": "BV1ASR",
                    "title": "ASR Video",
                    "cover": "https://img.example.test/asr.jpg",
                    "intro": "no subtitle",
                    "duration": 180,
                    "upper_mid": 1,
                    "upper_name": "tester",
                    "fav_time": 1710000020,
                    "pubtime": 1700000020,
                },
            ],
        }
        self.video_views = {
            "BV1SUB": {
                "bvid": "BV1SUB",
                "title": "Subtitle Video",
                "subtitle": {"list": [{"lan": "zh-CN", "subtitle_url": "https://sub.example.test/1.json"}]},
                "pages": [{"cid": 201, "page": 1, "part": "P1", "duration": 120}],
            },
            "BV1ASR": {
                "bvid": "BV1ASR",
                "title": "ASR Video",
                "subtitle": {"list": []},
                "pages": [{"cid": 202, "page": 1, "part": "P1", "duration": 180}],
            },
        }
        self.subtitle_payload = {
            "body": [
                {"from": 0, "to": 1.2, "content": "rag basics"},
                {"from": 1.2, "to": 2.5, "content": "vector indexing"},
            ]
        }
        self.playurl_payloads = {
            202: {
                "dash": {
                    "audio": [
                        {
                            "baseUrl": "https://media.example.test/audio-202.m4a",
                        }
                    ]
                }
            }
        }
        self.start_error: Exception | None = None
        self.poll_error: Exception | None = None
        self.folder_error: Exception | None = None
        self.folder_items_error: Exception | None = None
        self.calls: list[tuple[str, object]] = []

    def start_qr_login(self) -> dict[str, object]:
        self.calls.append(("start_qr_login", None))
        if self.start_error:
            raise self.start_error
        return self.start_result

    def poll_qr_login(self, qrcode_key: str) -> dict[str, object]:
        self.calls.append(("poll_qr_login", qrcode_key))
        if self.poll_error:
            raise self.poll_error
        return self.poll_result

    def list_favorite_folders(self, cookie: str, *, folder_type: int = 2) -> dict[str, object]:
        self.calls.append(("list_favorite_folders", {"cookie": cookie, "folder_type": folder_type}))
        if self.folder_error:
            raise self.folder_error
        return self.folder_result

    def list_folder_items(
        self,
        cookie: str,
        favorite_folder_id: str,
        *,
        pn: int = 1,
        ps: int = 20,
        keyword: str = "",
        order: str = "mtime",
    ) -> dict[str, object]:
        self.calls.append(
            (
                "list_folder_items",
                {
                    "cookie": cookie,
                    "favorite_folder_id": favorite_folder_id,
                    "pn": pn,
                    "ps": ps,
                    "keyword": keyword,
                    "order": order,
                },
            )
        )
        if self.folder_items_error:
            raise self.folder_items_error
        return self.folder_items_result

    def list_all_folder_items(
        self,
        cookie: str,
        favorite_folder_id: str,
        *,
        keyword: str = "",
        order: str = "mtime",
    ) -> dict[str, object]:
        self.calls.append(
            (
                "list_all_folder_items",
                {
                    "cookie": cookie,
                    "favorite_folder_id": favorite_folder_id,
                    "keyword": keyword,
                    "order": order,
                },
            )
        )
        if self.folder_items_error:
            raise self.folder_items_error
        return self.folder_items_result

    def get_video_view(
        self,
        cookie: str,
        *,
        bvid: str | None = None,
        aid: int | None = None,
    ) -> dict[str, object]:
        self.calls.append(("get_video_view", {"cookie": cookie, "bvid": bvid, "aid": aid}))
        key = bvid or str(aid)
        return self.video_views[key]

    def get_playurl(
        self,
        cookie: str,
        *,
        bvid: str,
        cid: int,
    ) -> dict[str, object]:
        self.calls.append(("get_playurl", {"cookie": cookie, "bvid": bvid, "cid": cid}))
        return self.playurl_payloads[cid]

    def fetch_subtitle_body(self, subtitle_url: str, *, cookie: str | None = None) -> dict[str, object]:
        self.calls.append(("fetch_subtitle_body", {"subtitle_url": subtitle_url, "cookie": cookie}))
        return self.subtitle_payload

    def choose_subtitle_entry(self, subtitles: list[dict[str, object]]) -> dict[str, object] | None:
        self.calls.append(("choose_subtitle_entry", {"count": len(subtitles)}))
        return subtitles[0] if subtitles else None

    def subtitle_payload_to_blocks(self, payload: dict[str, object]) -> list[dict[str, object]]:
        self.calls.append(("subtitle_payload_to_blocks", None))
        return [
            {
                "text": str(item["content"]),
                "start_ms": int(float(item["from"]) * 1000),
                "end_ms": int(float(item["to"]) * 1000),
                "block_index": index,
            }
            for index, item in enumerate(payload["body"])
        ]


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
    client.app.state.bilibili_import_pipeline = BilibiliImportPipeline(
        repository=client.app.state.repository,
        favorites_service=client.app.state.bilibili_favorite_folder_service,
        knowledge_index=client.app.state.knowledge_index,
        runtime_audit=client.app.state.runtime_audit,
    )
    client.app.state.bilibili_import_tool = build_bilibili_import_tool(
        client.app.state.bilibili_import_pipeline
    )
    client.app.state.orchestrator.tools = build_tool_registry(client.app.state.bilibili_import_tool)
    return active_vector_index


def install_fake_bilibili_favorite_folder_service(
    client: TestClient,
    service: FakeBilibiliFavoriteFolderService | None = None,
) -> FakeBilibiliFavoriteFolderService:
    active_service = service or FakeBilibiliFavoriteFolderService()
    client.app.state.bilibili_favorite_folder_service = active_service
    client.app.state.bilibili_import_pipeline = BilibiliImportPipeline(
        repository=client.app.state.repository,
        favorites_service=client.app.state.bilibili_favorite_folder_service,
        knowledge_index=client.app.state.knowledge_index,
        runtime_audit=client.app.state.runtime_audit,
    )
    client.app.state.bilibili_import_tool = build_bilibili_import_tool(
        client.app.state.bilibili_import_pipeline
    )
    client.app.state.orchestrator.tools = build_tool_registry(client.app.state.bilibili_import_tool)
    return active_service


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
        assert body["langsmith_thread_id"] == body["run_id"]
        assert body["langsmith_thread_url"]


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
        assert first_body["execution_plan"]["goal"].startswith("Import the requested")
        assert first_body["execution_plan"]["steps"][0]["status"] == "pending"
        assert first_body["execution_plan"]["tool_calls"][0]["args"]["request_message"] == (
            "请帮我导入这个收藏夹"
        )
        assert first_body["langsmith_thread_id"] == first_body["run_id"]
        initial_thread_url = first_body["langsmith_thread_url"]
        assert initial_thread_url

        confirm_response = client.post(
            f"/api/runs/{first_body['run_id']}/confirm",
            json={"approved": True},
        )
        assert confirm_response.status_code == 200
        confirm_body = confirm_response.json()
        assert confirm_body["status"] == "completed"
        assert confirm_body["approval_status"] == "approved"
        assert confirm_body["execution_plan"]["steps"][0]["status"] == "completed"
        assert confirm_body["langsmith_thread_id"] == first_body["run_id"]
        assert confirm_body["langsmith_thread_url"] == initial_thread_url


def test_run_detail_returns_steps(tmp_path: Path) -> None:
    with build_client(tmp_path) as client:
        response = client.post("/api/chat", json={"message": "知识库里这个视频讲了什么"})
        run_id = response.json()["run_id"]

        detail_response = client.get(f"/api/runs/{run_id}")

        assert detail_response.status_code == 200
        body = detail_response.json()
        assert body["run_id"] == run_id
        assert body["route"] == "video_knowledge_query"
        assert body["langsmith_thread_id"] == run_id
        assert body["langsmith_thread_url"]
        assert body["execution_plan"] is None
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
                        assert payload["payload"]["execution_plan"]["tool_calls"][0]["tool"] == (
                            "bilibili_import"
                        )
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
        assert body["execution_plan"]["tool_calls"][0]["action"] == "prepare_retry_plan"


def test_tool_run_detail_returns_execution_plan_and_approval_timestamps(tmp_path: Path) -> None:
    with build_client(tmp_path) as client:
        first_response = client.post("/api/chat", json={"message": "请帮我导入这个收藏夹"})
        run_id = first_response.json()["run_id"]
        client.post(f"/api/runs/{run_id}/confirm", json={"approved": True})

        detail_response = client.get(f"/api/runs/{run_id}")

        assert detail_response.status_code == 200
        body = detail_response.json()
        assert body["execution_plan"]["tool_calls"][0]["tool"] == "bilibili_import"
        assert body["approval_requested_at"]
        assert body["approval_resolved_at"]
        assert body["pending_actions"][0]["tool"] == "bilibili_import"


def test_rejected_tool_request_is_cancelled_without_tool_execution(tmp_path: Path) -> None:
    with build_client(tmp_path) as client:
        first_response = client.post("/api/chat", json={"message": "请帮我导入这个收藏夹"})
        run_id = first_response.json()["run_id"]

        confirm_response = client.post(
            f"/api/runs/{run_id}/confirm",
            json={"approved": False},
        )

        assert confirm_response.status_code == 200
        body = confirm_response.json()
        assert body["status"] == "cancelled"
        assert body["approval_status"] == "rejected"
        assert body["execution_plan"]["steps"][0]["status"] == "cancelled"

        with client.stream("GET", f"/api/runs/{run_id}/events?follow=false") as stream_response:
            replayed_event_types = []
            for line in stream_response.iter_lines():
                if line.startswith("data: "):
                    replayed_event_types.append(json.loads(line[6:])["type"])

        assert "tool_execution_started" not in replayed_event_types
        assert replayed_event_types[-1] == "run_completed"


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


def test_runtime_audit_records_planned_tools_and_tool_target(tmp_path: Path) -> None:
    runtime_audit = FakeRuntimeAudit()
    with build_client(tmp_path, runtime_audit=runtime_audit) as client:
        first_response = client.post("/api/chat", json={"message": "请帮我导入这个收藏夹"})
        run_id = first_response.json()["run_id"]
        client.post(f"/api/runs/{run_id}/confirm", json={"approved": True})

    approval_span = next(span for span in runtime_audit.spans if span["name"] == "agent.approval_gate")
    tool_span = next(span for span in runtime_audit.spans if span["name"] == "tool.bilibili_import.execute_import")

    assert approval_span["run"].metadata["planned_tools"] == ["bilibili_import.execute_import"]
    assert tool_span["run"].metadata["tool_target"] == "favorite-folder-ingestion"


def test_app_startup_requires_langsmith_config(tmp_path: Path) -> None:
    settings = Settings(
        app_db_path=tmp_path / "app.db",
        checkpoint_db_path=tmp_path / "checkpoints.db",
        chroma_persist_dir=tmp_path / "chroma",
        data_dir=tmp_path,
    )
    app = create_app(settings)

    with pytest.raises(RuntimeError, match="LANGSMITH_TRACING"):
        with TestClient(app):
            pass


def test_run_failure_marks_local_run_as_failed(tmp_path: Path) -> None:
    with build_client(tmp_path, raise_server_exceptions=False) as client:
        def broken_apply_chat_command(*args, **kwargs):
            raise RuntimeError("forced chat command failure")

        client.app.state.user_memory.apply_chat_command = broken_apply_chat_command

        response = client.post(
            "/api/chat",
            json={"user_id": "user-fail", "message": "记住偏好: reply_language=zh-CN"},
        )

        assert response.status_code == 500
        body = response.json()
        assert "Agent run failed" in body["detail"]

        with sqlite3.connect(tmp_path / "app.db") as connection:
            connection.row_factory = sqlite3.Row
            row = connection.execute(
                """
                SELECT run_id, status, latest_reply
                FROM runs
                ORDER BY created_at DESC
                LIMIT 1
                """
            ).fetchone()

        assert row is not None
        assert row["status"] == "failed"
        assert "forced chat command failure" in row["latest_reply"]


def test_langsmith_runtime_audit_redacts_secret_fields() -> None:
    audit = LangSmithRuntimeAudit.__new__(LangSmithRuntimeAudit)

    payload = {
        "api_key": "secret-value",
        "nested": {"cookie": "session-token", "normal": "A" * 1205},
    }
    sanitized = audit.sanitize_payload(payload)

    assert sanitized["api_key"] == "<redacted>"
    assert sanitized["nested"]["cookie"] == "<redacted>"
    assert sanitized["nested"]["normal"].endswith("...<truncated>")


def test_bilibili_qr_start_endpoint_returns_qr_payload(tmp_path: Path) -> None:
    with build_client(tmp_path) as client:
        service = install_fake_bilibili_favorite_folder_service(client)

        response = client.post("/api/bilibili/auth/qr/start")

        assert response.status_code == 200
        body = response.json()
        assert body["qrcode_key"] == "qr-key-1"
        assert body["expires_in_seconds"] == 180
        assert service.calls == [("start_qr_login", None)]


def test_bilibili_qr_poll_endpoint_returns_cookie_and_account(tmp_path: Path) -> None:
    with build_client(tmp_path) as client:
        service = install_fake_bilibili_favorite_folder_service(client)

        response = client.get("/api/bilibili/auth/qr/poll", params={"qrcode_key": "qr-key-1"})

        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "success"
        assert body["cookie"] == "DedeUserID=1; SESSDATA=test"
        assert body["account"]["uname"] == "tester"
        assert service.calls == [("poll_qr_login", "qr-key-1")]


def test_bilibili_favorite_folder_endpoint_returns_normalized_folders(tmp_path: Path) -> None:
    with build_client(tmp_path) as client:
        service = install_fake_bilibili_favorite_folder_service(client)

        response = client.get(
            "/api/bilibili/favorite-folders",
            headers={"X-Bilibili-Cookie": "SESSDATA=test"},
        )

        assert response.status_code == 200
        body = response.json()
        assert body["account"]["uname"] == "tester"
        assert body["total"] == 1
        assert body["folders"][0]["favorite_folder_id"] == "101"
        assert body["folders"][0]["media_count"] == 12
        assert service.calls == [
            ("list_favorite_folders", {"cookie": "SESSDATA=test", "folder_type": 2})
        ]


def test_bilibili_favorite_folder_endpoint_requires_cookie_header(tmp_path: Path) -> None:
    with build_client(tmp_path) as client:
        response = client.get("/api/bilibili/favorite-folders")

        assert response.status_code == 401
        assert "X-Bilibili-Cookie" in response.json()["detail"]


def test_bilibili_favorite_folder_endpoint_returns_auth_error(tmp_path: Path) -> None:
    with build_client(tmp_path) as client:
        service = install_fake_bilibili_favorite_folder_service(client)
        service.folder_error = BilibiliFavoriteFolderAuthError(
            "Bilibili login state is invalid or expired. Please login again."
        )

        response = client.get(
            "/api/bilibili/favorite-folders",
            headers={"X-Bilibili-Cookie": "SESSDATA=expired"},
        )

        assert response.status_code == 401
        assert "重新扫码登录" in response.json()["detail"]


def test_bilibili_favorite_folder_videos_endpoint_returns_selectable_items(tmp_path: Path) -> None:
    with build_client(tmp_path) as client:
        service = install_fake_bilibili_favorite_folder_service(client)

        response = client.get(
            "/api/bilibili/favorite-folders/101/videos",
            headers={"X-Bilibili-Cookie": "SESSDATA=test"},
        )

        assert response.status_code == 200
        body = response.json()
        assert body["folder"]["favorite_folder_id"] == "101"
        assert body["total"] == 3
        assert body["items"][0]["video_id"] == "BV1SUB"
        assert body["items"][1]["selectable"] is False
        assert "Unsupported favorite item type" in body["items"][1]["unsupported_reason"]
        assert service.calls[0][0] == "list_folder_items"


def test_submit_bilibili_import_runs_pipeline_and_records_asr_fallback(tmp_path: Path) -> None:
    runtime_audit = FakeRuntimeAudit()
    with build_client(tmp_path, runtime_audit=runtime_audit) as client:
        install_fake_knowledge_index(client)
        install_fake_bilibili_favorite_folder_service(client)

        response = client.post(
            "/api/bilibili/imports",
            headers={"X-Bilibili-Cookie": "SESSDATA=test"},
            json={
                "favorite_folder_id": "101",
                "selected_video_ids": ["BV1SUB", "BV1ASR"],
            },
        )

        assert response.status_code == 202
        body = response.json()
        assert body["status"] == "running"
        assert body["route"] == "import_request"
        assert body["requires_confirmation"] is False
        assert body["execution_plan"]["tool_calls"][0]["action"] == "execute_import"

        run_detail = client.get(f"/api/runs/{body['run_id']}")
        assert run_detail.status_code == 200
        detail = run_detail.json()
        assert detail["status"] == "completed"
        assert "indexed=1" in detail["reply"]
        assert "needs_asr=1" in detail["reply"]
        assert detail["execution_plan"]["steps"][0]["status"] == "completed"

        items = client.app.state.repository.get_import_run_items(body["run_id"])
        assert [item["status"] for item in items] == ["needs_asr", "indexed"]
        assert items[0]["needs_asr"] is True
        assert items[0]["asr_job"]["pages"][0]["media_url"] == "https://media.example.test/audio-202.m4a"
        assert items[1]["manifest"]["source_type"] == "subtitle"

        event_types = [event["type"] for event in client.app.state.repository.get_run_events(body["run_id"])]
        assert "import_started" in event_types
        assert "import_index_completed" in event_types
        assert event_types[-1] == "run_completed"

        submit_trace = next(
            request for request in runtime_audit.requests if request["name"] == "agent.submit_bilibili_import"
        )
        import_trace = next(
            request for request in runtime_audit.requests if request["name"] == "agent.import_selected_videos"
        )
        assert submit_trace["metadata"]["run_id"] == body["run_id"]
        assert import_trace["metadata"]["run_id"] == body["run_id"]


def test_submit_bilibili_import_rejects_invalid_selection(tmp_path: Path) -> None:
    with build_client(tmp_path) as client:
        install_fake_knowledge_index(client)
        install_fake_bilibili_favorite_folder_service(client)

        response = client.post(
            "/api/bilibili/imports",
            headers={"X-Bilibili-Cookie": "SESSDATA=test"},
            json={
                "favorite_folder_id": "101",
                "selected_video_ids": ["article-1"],
            },
        )

        assert response.status_code == 400
        assert "unsupported favorites" in response.json()["detail"]


def test_submit_bilibili_import_skips_duplicate_knowledge_videos(tmp_path: Path) -> None:
    with build_client(tmp_path) as client:
        install_fake_knowledge_index(client)
        install_fake_bilibili_favorite_folder_service(client)

        first = client.post(
            "/api/bilibili/imports",
            headers={"X-Bilibili-Cookie": "SESSDATA=test"},
            json={
                "favorite_folder_id": "101",
                "selected_video_ids": ["BV1SUB"],
            },
        )
        assert first.status_code == 202

        second = client.post(
            "/api/bilibili/imports",
            headers={"X-Bilibili-Cookie": "SESSDATA=test"},
            json={
                "favorite_folder_id": "101",
                "selected_video_ids": ["BV1SUB"],
            },
        )

        assert second.status_code == 202
        detail = client.get(f"/api/runs/{second.json()['run_id']}")
        assert detail.status_code == 200
        assert "skipped_duplicate=1" in detail.json()["reply"]

        items = client.app.state.repository.get_import_run_items(second.json()["run_id"])
        assert items[0]["status"] == "skipped_duplicate"


def test_manual_and_agent_import_share_same_bilibili_import_tool(tmp_path: Path) -> None:
    with build_client(tmp_path) as client:
        install_fake_knowledge_index(client)
        install_fake_bilibili_favorite_folder_service(client)

        agent_response = client.post("/api/chat", json={"message": "请帮我导入这个收藏夹"})
        assert agent_response.status_code == 200
        assert agent_response.json()["execution_plan"]["tool_calls"][0]["action"] == "execute_import"

        shared_tool = client.app.state.bilibili_import_tool
        assert client.app.state.orchestrator.tools[("bilibili_import", "execute_import")] is shared_tool


def test_bilibili_qr_poll_service_parses_cookie_and_account() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/qrcode/poll"):
            headers = [
                ("set-cookie", "SESSDATA=token-1; Path=/; Domain=.bilibili.com; HttpOnly"),
                ("set-cookie", "DedeUserID=42; Path=/; Domain=.bilibili.com"),
            ]
            return httpx.Response(
                200,
                headers=headers,
                json={
                    "code": 0,
                    "data": {
                        "code": 0,
                        "message": "0",
                        "refresh_token": "refresh-1",
                    },
                },
            )
        if request.url.path.endswith("/x/web-interface/nav"):
            assert request.headers["cookie"] == "DedeUserID=42; SESSDATA=token-1"
            return httpx.Response(
                200,
                json={"code": 0, "data": {"isLogin": True, "mid": 42, "uname": "alice"}},
            )
        raise AssertionError(f"Unexpected path: {request.url.path}")

    service = BilibiliFavoriteFolderService(transport=httpx.MockTransport(handler))

    result = service.poll_qr_login("qr-key-1")

    assert result["status"] == "success"
    assert result["cookie"] == "DedeUserID=42; SESSDATA=token-1"
    assert result["account"]["uname"] == "alice"


def test_bilibili_qr_poll_service_maps_pending_statuses() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={"code": 0, "data": {"code": 86101, "message": "Not scanned"}},
        )

    service = BilibiliFavoriteFolderService(transport=httpx.MockTransport(handler))

    result = service.poll_qr_login("qr-key-1")

    assert result["status"] == "pending_scan"
    assert result["cookie"] is None


def test_bilibili_favorite_folder_service_fetches_normalized_folders() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/x/web-interface/nav"):
            return httpx.Response(
                200,
                json={"code": 0, "data": {"isLogin": True, "mid": 42, "uname": "alice"}},
            )
        if request.url.path.endswith("/x/v3/fav/folder/created/list-all"):
            assert request.url.params["up_mid"] == "42"
            assert request.url.params["type"] == "2"
            return httpx.Response(
                200,
                json={
                    "code": 0,
                    "data": {
                        "list": [
                            {"id": 2, "title": "Folder B", "media_count": 5, "mid": 42, "attr": 0},
                            {"id": 1, "title": "Folder A", "media_count": 8, "mid": 42, "attr": 0},
                        ]
                    },
                },
            )
        if request.url.path.endswith("/x/v3/fav/folder/info"):
            media_id = request.url.params["media_id"]
            if media_id == "1":
                return httpx.Response(
                    200,
                    json={
                        "code": 0,
                        "data": {
                            "id": 1,
                            "mid": 42,
                            "title": "Folder A",
                            "intro": "Alpha",
                            "cover": "https://img/a.jpg",
                            "attr": 0,
                            "media_count": 8,
                            "upper": {"mid": 42},
                        },
                    },
                )
            return httpx.Response(
                200,
                json={
                    "code": 0,
                    "data": {
                        "id": 2,
                        "mid": 42,
                        "title": "Folder B",
                        "intro": "Beta",
                        "cover": "https://img/b.jpg",
                        "attr": 1,
                        "media_count": 5,
                        "upper": {"mid": 42},
                    },
                },
            )
        raise AssertionError(f"Unexpected path: {request.url.path}")

    service = BilibiliFavoriteFolderService(transport=httpx.MockTransport(handler))

    result = service.list_favorite_folders("SESSDATA=test")

    assert result["account"]["mid"] == 42
    assert result["total"] == 2
    assert [folder["favorite_folder_id"] for folder in result["folders"]] == ["1", "2"]


def test_bilibili_favorite_folder_service_maps_auth_errors() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"code": -101, "message": "not login"})

    service = BilibiliFavoriteFolderService(transport=httpx.MockTransport(handler))

    with pytest.raises(BilibiliFavoriteFolderAuthError):
        service.list_favorite_folders("SESSDATA=test")


def test_bilibili_favorite_folder_service_raises_response_error_on_invalid_json() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text="not-json")

    service = BilibiliFavoriteFolderService(transport=httpx.MockTransport(handler))

    with pytest.raises(BilibiliFavoriteFolderResponseError):
        service.list_favorite_folders("SESSDATA=test")


def test_bilibili_favorite_folder_service_maps_timeout_to_upstream_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("timed out")

    service = BilibiliFavoriteFolderService(transport=httpx.MockTransport(handler))

    with pytest.raises(BilibiliFavoriteFolderUpstreamError):
        service.start_qr_login()


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
            runtime_audit=FakeRuntimeAudit(),
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
