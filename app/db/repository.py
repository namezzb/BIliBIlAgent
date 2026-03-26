import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from app.agent.types import RunEvent, RunEventType
from app.db.schema import SCHEMA_STATEMENTS


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class SQLiteRepository:
    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path

    def initialize(self) -> None:
        with self._connect() as connection:
            for statement in SCHEMA_STATEMENTS:
                connection.execute(statement)
            self._ensure_sessions_user_id_column(connection)
            self._ensure_sessions_memory_columns(connection)
            self._ensure_runs_route_column(connection)
            self._ensure_runs_langsmith_columns(connection)
            self._ensure_runs_execution_plan_columns(connection)
            self._ensure_knowledge_text_chunks_video_schema(connection)
            self._ensure_knowledge_chunk_pages_schema(connection)
            connection.commit()

    def _ensure_sessions_user_id_column(self, connection: sqlite3.Connection) -> None:
        rows = connection.execute("PRAGMA table_info(sessions)").fetchall()
        columns = {row["name"] for row in rows}
        if "user_id" not in columns:
            connection.execute("ALTER TABLE sessions ADD COLUMN user_id TEXT")

    def _ensure_sessions_memory_columns(self, connection: sqlite3.Connection) -> None:
        rows = connection.execute("PRAGMA table_info(sessions)").fetchall()
        columns = {row["name"] for row in rows}
        if "summary_text" not in columns:
            connection.execute("ALTER TABLE sessions ADD COLUMN summary_text TEXT")
        if "recent_context_json" not in columns:
            connection.execute("ALTER TABLE sessions ADD COLUMN recent_context_json TEXT")

    def _ensure_runs_route_column(self, connection: sqlite3.Connection) -> None:
        rows = connection.execute("PRAGMA table_info(runs)").fetchall()
        columns = {row["name"] for row in rows}
        if "route" not in columns:
            connection.execute("ALTER TABLE runs ADD COLUMN route TEXT")

    def _ensure_runs_langsmith_columns(self, connection: sqlite3.Connection) -> None:
        rows = connection.execute("PRAGMA table_info(runs)").fetchall()
        columns = {row["name"] for row in rows}
        if "langsmith_thread_id" not in columns:
            connection.execute("ALTER TABLE runs ADD COLUMN langsmith_thread_id TEXT")
        if "langsmith_thread_url" not in columns:
            connection.execute("ALTER TABLE runs ADD COLUMN langsmith_thread_url TEXT")

    def _ensure_runs_execution_plan_columns(self, connection: sqlite3.Connection) -> None:
        rows = connection.execute("PRAGMA table_info(runs)").fetchall()
        columns = {row["name"] for row in rows}
        if "execution_plan_json" not in columns:
            connection.execute("ALTER TABLE runs ADD COLUMN execution_plan_json TEXT")
        if "approval_requested_at" not in columns:
            connection.execute("ALTER TABLE runs ADD COLUMN approval_requested_at TEXT")
        if "approval_resolved_at" not in columns:
            connection.execute("ALTER TABLE runs ADD COLUMN approval_resolved_at TEXT")

    def _ensure_knowledge_text_chunks_video_schema(self, connection: sqlite3.Connection) -> None:
        rows = connection.execute("PRAGMA table_info(knowledge_text_chunks)").fetchall()
        if not rows:
            return
        columns = {row["name"] for row in rows}
        if "video_id" in columns:
            return

        connection.execute("DROP INDEX IF EXISTS idx_knowledge_text_chunks_page_id")
        connection.execute("ALTER TABLE knowledge_text_chunks RENAME TO knowledge_text_chunks_legacy")
        connection.execute(
            """
            CREATE TABLE knowledge_text_chunks (
                chunk_id TEXT PRIMARY KEY,
                video_id TEXT NOT NULL,
                source_type TEXT NOT NULL,
                source_language TEXT,
                block_index INTEGER NOT NULL,
                text TEXT NOT NULL,
                start_ms INTEGER,
                end_ms INTEGER,
                embedding_model TEXT NOT NULL,
                embedding_version TEXT NOT NULL,
                index_status TEXT NOT NULL,
                vector_document_id TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
            """
        )
        connection.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_knowledge_text_chunks_video_id
            ON knowledge_text_chunks (video_id)
            """
        )
        connection.execute("DROP TABLE knowledge_text_chunks_legacy")

    def _ensure_knowledge_chunk_pages_schema(self, connection: sqlite3.Connection) -> None:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS knowledge_chunk_pages (
                chunk_id TEXT NOT NULL,
                page_id TEXT NOT NULL,
                created_at TEXT NOT NULL,
                PRIMARY KEY (chunk_id, page_id)
            )
            """
        )
        connection.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_knowledge_chunk_pages_page_id
            ON knowledge_chunk_pages (page_id)
            """
        )

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.db_path)
        connection.row_factory = sqlite3.Row
        return connection

    def session_exists(self, session_id: str) -> bool:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT 1 FROM sessions WHERE session_id = ?",
                (session_id,),
            ).fetchone()
        return row is not None

    def create_session(self, session_id: str, user_id: str | None = None) -> None:
        timestamp = utc_now()
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO sessions (session_id, user_id, created_at, updated_at)
                VALUES (?, ?, ?, ?)
                """,
                (session_id, user_id, timestamp, timestamp),
            )
            connection.commit()

    def touch_session(self, session_id: str) -> None:
        with self._connect() as connection:
            connection.execute(
                "UPDATE sessions SET updated_at = ? WHERE session_id = ?",
                (utc_now(), session_id),
            )
            connection.commit()

    def set_session_user_id(self, session_id: str, user_id: str) -> None:
        with self._connect() as connection:
            connection.execute(
                "UPDATE sessions SET user_id = ?, updated_at = ? WHERE session_id = ?",
                (user_id, utc_now(), session_id),
            )
            connection.commit()

    def update_session_memory(
        self,
        session_id: str,
        *,
        summary_text: str | None,
        recent_context: dict[str, Any],
    ) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                UPDATE sessions
                SET summary_text = ?, recent_context_json = ?, updated_at = ?
                WHERE session_id = ?
                """,
                (
                    summary_text,
                    json.dumps(recent_context, ensure_ascii=False),
                    utc_now(),
                    session_id,
                ),
            )
            connection.commit()

    def get_session(self, session_id: str) -> dict[str, Any] | None:
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT session_id, user_id, summary_text, recent_context_json, created_at, updated_at
                FROM sessions
                WHERE session_id = ?
                """,
                (session_id,),
            ).fetchone()
        if row is None:
            return None
        result = dict(row)
        recent_context_json = result.pop("recent_context_json")
        result["recent_context"] = json.loads(recent_context_json) if recent_context_json else {}
        return result

    def add_message(
        self,
        session_id: str,
        role: str,
        content: str,
        run_id: str | None = None,
    ) -> str:
        message_id = str(uuid4())
        timestamp = utc_now()
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO messages (message_id, session_id, run_id, role, content, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (message_id, session_id, run_id, role, content, timestamp),
            )
            connection.execute(
                "UPDATE sessions SET updated_at = ? WHERE session_id = ?",
                (timestamp, session_id),
            )
            connection.commit()
        return message_id

    def get_messages(self, session_id: str) -> list[dict[str, Any]]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT message_id, session_id, run_id, role, content, created_at
                FROM messages
                WHERE session_id = ?
                ORDER BY created_at ASC
                """,
                (session_id,),
            ).fetchall()
        return [dict(row) for row in rows]

    def create_run(self, run_id: str, session_id: str, status: str) -> None:
        timestamp = utc_now()
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO runs (
                    run_id,
                    session_id,
                    status,
                    created_at,
                    updated_at
                ) VALUES (?, ?, ?, ?, ?)
                """,
                (run_id, session_id, status, timestamp, timestamp),
            )
            connection.commit()

    def update_run(
        self,
        run_id: str,
        *,
        intent: str | None = None,
        route: str | None = None,
        langsmith_thread_id: str | None = None,
        langsmith_thread_url: str | None = None,
        status: str | None = None,
        requires_confirmation: bool | None = None,
        approval_status: str | None = None,
        latest_reply: str | None = None,
        pending_actions: list[dict[str, Any]] | None = None,
        execution_plan: dict[str, Any] | None = None,
        approval_requested_at: str | None = None,
        approval_resolved_at: str | None = None,
    ) -> None:
        assignments: list[str] = ["updated_at = ?"]
        values: list[Any] = [utc_now()]

        if intent is not None:
            assignments.append("intent = ?")
            values.append(intent)
        if route is not None:
            assignments.append("route = ?")
            values.append(route)
        if langsmith_thread_id is not None:
            assignments.append("langsmith_thread_id = ?")
            values.append(langsmith_thread_id)
        if langsmith_thread_url is not None:
            assignments.append("langsmith_thread_url = ?")
            values.append(langsmith_thread_url)
        if status is not None:
            assignments.append("status = ?")
            values.append(status)
        if requires_confirmation is not None:
            assignments.append("requires_confirmation = ?")
            values.append(int(requires_confirmation))
        if approval_status is not None:
            assignments.append("approval_status = ?")
            values.append(approval_status)
        if latest_reply is not None:
            assignments.append("latest_reply = ?")
            values.append(latest_reply)
        if pending_actions is not None:
            assignments.append("pending_actions_json = ?")
            values.append(json.dumps(pending_actions, ensure_ascii=False))
        if execution_plan is not None:
            assignments.append("execution_plan_json = ?")
            values.append(json.dumps(execution_plan, ensure_ascii=False))
        if approval_requested_at is not None:
            assignments.append("approval_requested_at = ?")
            values.append(approval_requested_at)
        if approval_resolved_at is not None:
            assignments.append("approval_resolved_at = ?")
            values.append(approval_resolved_at)

        values.append(run_id)
        with self._connect() as connection:
            connection.execute(
                f"UPDATE runs SET {', '.join(assignments)} WHERE run_id = ?",
                values,
            )
            connection.commit()

    def get_run(self, run_id: str) -> dict[str, Any] | None:
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT run_id, session_id, intent, route, langsmith_thread_id,
                       langsmith_thread_url, status, requires_confirmation,
                       approval_status, latest_reply, pending_actions_json,
                       execution_plan_json, approval_requested_at, approval_resolved_at,
                       created_at, updated_at
                FROM runs
                WHERE run_id = ?
                """,
                (run_id,),
            ).fetchone()
        if row is None:
            return None
        result = dict(row)
        result["requires_confirmation"] = bool(result["requires_confirmation"])
        pending_actions_json = result.pop("pending_actions_json")
        result["pending_actions"] = json.loads(pending_actions_json) if pending_actions_json else []
        execution_plan_json = result.pop("execution_plan_json")
        result["execution_plan"] = json.loads(execution_plan_json) if execution_plan_json else None
        return result

    def get_user_memory_profile(self, user_id: str) -> dict[str, Any] | None:
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT user_id, profile_json, created_at, updated_at
                FROM user_memory_profiles
                WHERE user_id = ?
                """,
                (user_id,),
            ).fetchone()
        if row is None:
            return None
        return {
            "user_id": row["user_id"],
            "profile": json.loads(row["profile_json"]),
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
        }

    def upsert_user_memory_profile(self, user_id: str, profile: dict[str, Any]) -> dict[str, Any]:
        existing_profile = self.get_user_memory_profile(user_id)
        timestamp = utc_now()
        created_at = existing_profile["created_at"] if existing_profile is not None else timestamp
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO user_memory_profiles (user_id, profile_json, created_at, updated_at)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(user_id) DO UPDATE SET
                    profile_json = excluded.profile_json,
                    updated_at = excluded.updated_at
                """,
                (
                    user_id,
                    json.dumps(profile, ensure_ascii=False),
                    created_at,
                    timestamp,
                ),
            )
            connection.commit()
        return {
            "user_id": user_id,
            "profile": profile,
            "created_at": created_at,
            "updated_at": timestamp,
        }

    def upsert_knowledge_bundle(
        self,
        *,
        favorite_folders: list[dict[str, Any]],
        videos: list[dict[str, Any]],
        favorite_video_links: list[dict[str, str]],
        pages: list[dict[str, Any]],
        chunks: list[dict[str, Any]],
        chunk_page_links: list[dict[str, str]],
    ) -> None:
        timestamp = utc_now()

        with self._connect() as connection:
            for folder in favorite_folders:
                connection.execute(
                    """
                    INSERT INTO knowledge_favorite_folders (
                        favorite_folder_id,
                        title,
                        intro,
                        created_at,
                        updated_at
                    ) VALUES (?, ?, ?, ?, ?)
                    ON CONFLICT(favorite_folder_id) DO UPDATE SET
                        title = excluded.title,
                        intro = excluded.intro,
                        updated_at = excluded.updated_at
                    """,
                    (
                        folder["favorite_folder_id"],
                        folder["title"],
                        folder.get("intro"),
                        timestamp,
                        timestamp,
                    ),
                )

            for video in videos:
                connection.execute(
                    """
                    INSERT INTO knowledge_videos (
                        video_id,
                        bvid,
                        title,
                        created_at,
                        updated_at
                    ) VALUES (?, ?, ?, ?, ?)
                    ON CONFLICT(video_id) DO UPDATE SET
                        bvid = excluded.bvid,
                        title = excluded.title,
                        updated_at = excluded.updated_at
                    """,
                    (
                        video["video_id"],
                        video.get("bvid"),
                        video["title"],
                        timestamp,
                        timestamp,
                    ),
                )

            for link in favorite_video_links:
                connection.execute(
                    """
                    INSERT INTO knowledge_favorite_videos (
                        favorite_folder_id,
                        video_id,
                        created_at
                    ) VALUES (?, ?, ?)
                    ON CONFLICT(favorite_folder_id, video_id) DO NOTHING
                    """,
                    (
                        link["favorite_folder_id"],
                        link["video_id"],
                        timestamp,
                    ),
                )

            for page in pages:
                connection.execute(
                    """
                    INSERT INTO knowledge_video_pages (
                        page_id,
                        video_id,
                        page_number,
                        title,
                        created_at,
                        updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?)
                    ON CONFLICT(page_id) DO UPDATE SET
                        video_id = excluded.video_id,
                        page_number = excluded.page_number,
                        title = excluded.title,
                        updated_at = excluded.updated_at
                    """,
                    (
                        page["page_id"],
                        page["video_id"],
                        page["page_number"],
                        page["title"],
                        timestamp,
                        timestamp,
                    ),
                )

            for chunk in chunks:
                connection.execute(
                    """
                    INSERT INTO knowledge_text_chunks (
                        chunk_id,
                        video_id,
                        source_type,
                        source_language,
                        block_index,
                        text,
                        start_ms,
                        end_ms,
                        embedding_model,
                        embedding_version,
                        index_status,
                        vector_document_id,
                        created_at,
                        updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(chunk_id) DO UPDATE SET
                        video_id = excluded.video_id,
                        source_type = excluded.source_type,
                        source_language = excluded.source_language,
                        block_index = excluded.block_index,
                        text = excluded.text,
                        start_ms = excluded.start_ms,
                        end_ms = excluded.end_ms,
                        embedding_model = excluded.embedding_model,
                        embedding_version = excluded.embedding_version,
                        index_status = excluded.index_status,
                        vector_document_id = excluded.vector_document_id,
                        updated_at = excluded.updated_at
                    """,
                    (
                        chunk["chunk_id"],
                        chunk["video_id"],
                        chunk["source_type"],
                        chunk.get("source_language"),
                        chunk["block_index"],
                        chunk["text"],
                        chunk.get("start_ms"),
                        chunk.get("end_ms"),
                        chunk["embedding_model"],
                        chunk["embedding_version"],
                        chunk["index_status"],
                        chunk["vector_document_id"],
                        timestamp,
                        timestamp,
                    ),
                )

            if chunk_page_links:
                distinct_chunk_ids = sorted({str(link["chunk_id"]) for link in chunk_page_links})
                placeholders = ", ".join("?" for _ in distinct_chunk_ids)
                connection.execute(
                    f"DELETE FROM knowledge_chunk_pages WHERE chunk_id IN ({placeholders})",
                    distinct_chunk_ids,
                )
                for link in chunk_page_links:
                    connection.execute(
                        """
                        INSERT INTO knowledge_chunk_pages (
                            chunk_id,
                            page_id,
                            created_at
                        ) VALUES (?, ?, ?)
                        ON CONFLICT(chunk_id, page_id) DO NOTHING
                        """,
                        (
                            link["chunk_id"],
                            link["page_id"],
                            timestamp,
                        ),
                    )

            connection.commit()

    def get_existing_knowledge_video_ids(self, video_ids: list[str]) -> list[str]:
        if not video_ids:
            return []

        placeholders = ", ".join("?" for _ in video_ids)
        with self._connect() as connection:
            rows = connection.execute(
                f"""
                SELECT video_id
                FROM knowledge_videos
                WHERE video_id IN ({placeholders})
                ORDER BY video_id ASC
                """,
                video_ids,
            ).fetchall()
        return [str(row["video_id"]) for row in rows]

    def list_knowledge_favorite_folders(self) -> list[dict[str, Any]]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT favorite_folder_id, title, intro
                FROM knowledge_favorite_folders
                ORDER BY title COLLATE NOCASE ASC
                """
            ).fetchall()
        return [dict(row) for row in rows]

    def list_knowledge_videos(self) -> list[dict[str, Any]]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT video_id, bvid, title
                FROM knowledge_videos
                ORDER BY title COLLATE NOCASE ASC
                """
            ).fetchall()
        return [dict(row) for row in rows]

    def get_knowledge_chunk_details(self, chunk_ids: list[str]) -> list[dict[str, Any]]:
        if not chunk_ids:
            return []

        placeholders = ", ".join("?" for _ in chunk_ids)
        with self._connect() as connection:
            rows = connection.execute(
                f"""
                SELECT
                    c.chunk_id,
                    c.video_id,
                    c.source_type,
                    c.source_language,
                    c.block_index,
                    c.text,
                    c.start_ms,
                    c.end_ms,
                    c.embedding_model,
                    c.embedding_version,
                    c.index_status,
                    c.vector_document_id,
                    v.video_id,
                    v.bvid,
                    v.title AS video_title,
                    f.favorite_folder_id,
                    f.title AS favorite_folder_title,
                    f.intro AS favorite_folder_intro,
                    p.page_id,
                    p.page_number,
                    p.title AS page_title
                FROM knowledge_text_chunks c
                JOIN knowledge_videos v ON v.video_id = c.video_id
                LEFT JOIN knowledge_favorite_videos fv ON fv.video_id = v.video_id
                LEFT JOIN knowledge_favorite_folders f ON f.favorite_folder_id = fv.favorite_folder_id
                LEFT JOIN knowledge_chunk_pages cp ON cp.chunk_id = c.chunk_id
                LEFT JOIN knowledge_video_pages p ON p.page_id = cp.page_id
                WHERE c.chunk_id IN ({placeholders})
                ORDER BY c.chunk_id ASC
                """,
                chunk_ids,
            ).fetchall()

        grouped: dict[str, dict[str, Any]] = {}
        favorite_folder_ids_by_chunk: dict[str, set[str]] = {}
        page_ids_by_chunk: dict[str, set[str]] = {}
        for row in rows:
            chunk_id = str(row["chunk_id"])
            if chunk_id not in grouped:
                grouped[chunk_id] = {
                    "chunk_id": chunk_id,
                    "video_id": row["video_id"],
                    "source_type": row["source_type"],
                    "source_language": row["source_language"],
                    "block_index": row["block_index"],
                    "text": row["text"],
                    "start_ms": row["start_ms"],
                    "end_ms": row["end_ms"],
                    "embedding_model": row["embedding_model"],
                    "embedding_version": row["embedding_version"],
                    "index_status": row["index_status"],
                    "vector_document_id": row["vector_document_id"],
                    "video": {
                        "video_id": row["video_id"],
                        "bvid": row["bvid"],
                        "title": row["video_title"],
                    },
                    "favorite_folders": [],
                    "pages": [],
                }
                favorite_folder_ids_by_chunk[chunk_id] = set()
                page_ids_by_chunk[chunk_id] = set()
            if row["favorite_folder_id"] is not None:
                favorite_folder_id = str(row["favorite_folder_id"])
                if favorite_folder_id not in favorite_folder_ids_by_chunk[chunk_id]:
                    favorite_folder_ids_by_chunk[chunk_id].add(favorite_folder_id)
                    grouped[chunk_id]["favorite_folders"].append(
                        {
                            "favorite_folder_id": favorite_folder_id,
                            "title": row["favorite_folder_title"],
                            "intro": row["favorite_folder_intro"],
                        }
                    )
            if row["page_id"] is not None:
                page_id = str(row["page_id"])
                if page_id not in page_ids_by_chunk[chunk_id]:
                    page_ids_by_chunk[chunk_id].add(page_id)
                    grouped[chunk_id]["pages"].append(
                        {
                            "page_id": page_id,
                            "page_number": int(row["page_number"]),
                            "title": row["page_title"],
                        }
                    )

        for detail in grouped.values():
            detail["favorite_folders"].sort(key=lambda item: str(item["favorite_folder_id"]))
            detail["pages"].sort(key=lambda item: (int(item["page_number"]), str(item["page_id"])))

        return [grouped[chunk_id] for chunk_id in chunk_ids if chunk_id in grouped]

    def upsert_run_step(
        self,
        run_id: str,
        step_key: str,
        step_name: str,
        status: str,
        *,
        input_summary: str | None = None,
        output_summary: str | None = None,
    ) -> None:
        timestamp = utc_now()
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO run_steps (
                    run_id,
                    step_key,
                    step_name,
                    status,
                    input_summary,
                    output_summary,
                    created_at,
                    updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(run_id, step_key) DO UPDATE SET
                    step_name = excluded.step_name,
                    status = excluded.status,
                    input_summary = excluded.input_summary,
                    output_summary = excluded.output_summary,
                    updated_at = excluded.updated_at
                """,
                (
                    run_id,
                    step_key,
                    step_name,
                    status,
                    input_summary,
                    output_summary,
                    timestamp,
                    timestamp,
                ),
            )
            connection.commit()

    def get_run_steps(self, run_id: str) -> list[dict[str, Any]]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT step_key, step_name, status, input_summary, output_summary, updated_at
                FROM run_steps
                WHERE run_id = ?
                ORDER BY id ASC
                """,
                (run_id,),
            ).fetchall()
        return [dict(row) for row in rows]

    def append_run_event(
        self,
        run_id: str,
        event_type: RunEventType,
        payload: dict[str, Any],
    ) -> RunEvent:
        event_id = str(uuid4())
        timestamp = utc_now()
        payload_json = json.dumps(payload, ensure_ascii=False)
        with self._connect() as connection:
            row = connection.execute(
                "SELECT COALESCE(MAX(sequence), 0) + 1 AS next_sequence FROM run_events WHERE run_id = ?",
                (run_id,),
            ).fetchone()
            sequence = int(row["next_sequence"])
            connection.execute(
                """
                INSERT INTO run_events (
                    event_id,
                    run_id,
                    sequence,
                    event_type,
                    payload_json,
                    created_at
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (event_id, run_id, sequence, event_type, payload_json, timestamp),
            )
            connection.commit()
        return {
            "event_id": event_id,
            "run_id": run_id,
            "sequence": sequence,
            "type": event_type,
            "timestamp": timestamp,
            "payload": payload,
        }

    def get_run_events(
        self,
        run_id: str,
        *,
        after_sequence: int = 0,
    ) -> list[RunEvent]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT event_id, run_id, sequence, event_type, payload_json, created_at
                FROM run_events
                WHERE run_id = ? AND sequence > ?
                ORDER BY sequence ASC
                """,
                (run_id, after_sequence),
            ).fetchall()
        return [
            {
                "event_id": row["event_id"],
                "run_id": row["run_id"],
                "sequence": row["sequence"],
                "type": row["event_type"],
                "timestamp": row["created_at"],
                "payload": json.loads(row["payload_json"]),
            }
            for row in rows
        ]

    def get_run_event_count(self, run_id: str) -> int:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT COUNT(*) AS total FROM run_events WHERE run_id = ?",
                (run_id,),
            ).fetchone()
        return int(row["total"]) if row is not None else 0

    def upsert_import_run_item(
        self,
        run_id: str,
        *,
        favorite_folder_id: str,
        video_id: str,
        bvid: str | None,
        title: str,
        status: str,
        needs_asr: bool = False,
        failure_reason: str | None = None,
        retryable: bool = False,
        manifest: dict[str, Any] | None = None,
        asr_job: dict[str, Any] | None = None,
    ) -> None:
        timestamp = utc_now()
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO import_run_items (
                    run_id,
                    favorite_folder_id,
                    video_id,
                    bvid,
                    title,
                    status,
                    needs_asr,
                    failure_reason,
                    retryable,
                    manifest_json,
                    asr_job_json,
                    created_at,
                    updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(run_id, video_id) DO UPDATE SET
                    favorite_folder_id = excluded.favorite_folder_id,
                    bvid = excluded.bvid,
                    title = excluded.title,
                    status = excluded.status,
                    needs_asr = excluded.needs_asr,
                    failure_reason = excluded.failure_reason,
                    retryable = excluded.retryable,
                    manifest_json = excluded.manifest_json,
                    asr_job_json = excluded.asr_job_json,
                    updated_at = excluded.updated_at
                """,
                (
                    run_id,
                    favorite_folder_id,
                    video_id,
                    bvid,
                    title,
                    status,
                    int(needs_asr),
                    failure_reason,
                    int(retryable),
                    json.dumps(manifest, ensure_ascii=False) if manifest is not None else None,
                    json.dumps(asr_job, ensure_ascii=False) if asr_job is not None else None,
                    timestamp,
                    timestamp,
                ),
            )
            connection.commit()

    def get_import_run_items(self, run_id: str) -> list[dict[str, Any]]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT
                    run_id,
                    favorite_folder_id,
                    video_id,
                    bvid,
                    title,
                    status,
                    needs_asr,
                    failure_reason,
                    retryable,
                    manifest_json,
                    asr_job_json,
                    created_at,
                    updated_at
                FROM import_run_items
                WHERE run_id = ?
                ORDER BY id ASC
                """,
                (run_id,),
            ).fetchall()
        results: list[dict[str, Any]] = []
        for row in rows:
            item = dict(row)
            item["needs_asr"] = bool(item["needs_asr"])
            item["retryable"] = bool(item["retryable"])
            manifest_json = item.pop("manifest_json")
            asr_job_json = item.pop("asr_job_json")
            item["manifest"] = json.loads(manifest_json) if manifest_json else None
            item["asr_job"] = json.loads(asr_job_json) if asr_job_json else None
            results.append(item)
        return results
