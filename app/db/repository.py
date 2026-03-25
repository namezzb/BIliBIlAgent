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
        status: str | None = None,
        requires_confirmation: bool | None = None,
        approval_status: str | None = None,
        latest_reply: str | None = None,
        pending_actions: list[dict[str, Any]] | None = None,
    ) -> None:
        assignments: list[str] = ["updated_at = ?"]
        values: list[Any] = [utc_now()]

        if intent is not None:
            assignments.append("intent = ?")
            values.append(intent)
        if route is not None:
            assignments.append("route = ?")
            values.append(route)
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
                SELECT run_id, session_id, intent, route, status, requires_confirmation,
                       approval_status, latest_reply, pending_actions_json,
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
