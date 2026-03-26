from __future__ import annotations

from typing import Any

from app.db.repository import SQLiteRepository
from app.services.llm import OpenAICompatibleLLM


SUMMARY_TRIGGER_MESSAGES = 8
RECENT_MESSAGE_LIMIT = 6
SUMMARY_LINE_LIMIT = 6


class SessionMemoryManager:
    def __init__(
        self,
        repository: SQLiteRepository,
        llm: OpenAICompatibleLLM,
    ) -> None:
        self.repository = repository
        self.llm = llm

    def load_session_context(self, session_id: str) -> dict[str, Any]:
        session = self.repository.get_session(session_id) or {}
        messages = self.repository.get_messages(session_id)
        prompt_messages = self._build_prompt_messages(messages, session.get("summary_text"))
        return {
            "messages": prompt_messages,
            "session_summary": session.get("summary_text"),
            "recent_context": session.get("recent_context", {}),
            "message_count": len(messages),
        }

    def refresh_session_memory(
        self,
        session_id: str,
        *,
        run_id: str,
        intent: str | None,
        route: str | None,
        status: str,
        reply: str,
        pending_actions: list[dict[str, Any]],
        retrieval_result: dict[str, Any] | None = None,
    ) -> None:
        session = self.repository.get_session(session_id) or {}
        messages = self.repository.get_messages(session_id)
        summary_text = self._build_summary(messages)
        recent_context = {
            "last_run_id": run_id,
            "last_intent": intent,
            "last_route": route,
            "last_status": status,
            "last_user_message": self._last_message_content(messages, "user"),
            "last_assistant_reply": reply,
            "last_pending_actions": pending_actions,
            "message_count": len(messages),
        }
        if retrieval_result is not None:
            recent_context["last_retrieval"] = {
                "query": retrieval_result.get("query"),
                "route": retrieval_result.get("route"),
                "resolved_scope": retrieval_result.get("resolved_scope", {}),
                "total_hits": retrieval_result.get("total_hits", 0),
                "top_sources": retrieval_result.get("top_sources", []),
            }
        elif isinstance(session.get("recent_context"), dict) and session["recent_context"].get("last_retrieval"):
            recent_context["last_retrieval"] = session["recent_context"]["last_retrieval"]
        self.repository.update_session_memory(
            session_id,
            summary_text=summary_text,
            recent_context=recent_context,
        )

    def get_session_detail(self, session_id: str) -> dict[str, Any] | None:
        session = self.repository.get_session(session_id)
        if session is None:
            return None
        messages = self.repository.get_messages(session_id)
        return {
            **session,
            "messages": [
                {
                    "message_id": message["message_id"],
                    "run_id": message["run_id"],
                    "role": message["role"],
                    "content": message["content"],
                    "created_at": message["created_at"],
                }
                for message in messages
            ],
        }

    def _build_prompt_messages(
        self,
        messages: list[dict[str, Any]],
        summary_text: str | None,
    ) -> list[dict[str, str]]:
        recent_messages = messages[-RECENT_MESSAGE_LIMIT:]
        prompt_messages: list[dict[str, str]] = []
        if summary_text:
            prompt_messages.append(
                {
                    "role": "assistant",
                    "content": f"[Conversation summary for context only]\n{summary_text}",
                }
            )
        prompt_messages.extend(
            {"role": message["role"], "content": message["content"]}
            for message in recent_messages
        )
        return prompt_messages

    def _build_summary(self, messages: list[dict[str, Any]]) -> str | None:
        if len(messages) < SUMMARY_TRIGGER_MESSAGES:
            return None

        older_messages = messages[:-RECENT_MESSAGE_LIMIT]
        if not older_messages:
            return None

        model_summary = None
        try:
            model_summary = self.llm.summarize_conversation(
                [
                    {"role": message["role"], "content": message["content"]}
                    for message in older_messages
                ]
            )
        except Exception:
            pass
        if model_summary:
            return model_summary

        return self._build_fallback_summary(older_messages)

    def _build_fallback_summary(self, older_messages: list[dict[str, Any]]) -> str | None:
        if not older_messages:
            return None

        lines = []
        for message in older_messages[-SUMMARY_LINE_LIMIT:]:
            role = "User" if message["role"] == "user" else "Assistant"
            content = str(message["content"]).replace("\n", " ").strip()
            if len(content) > 120:
                content = f"{content[:117]}..."
            lines.append(f"{role}: {content}")

        if not lines:
            return None
        return "\n".join(lines)

    def _last_message_content(self, messages: list[dict[str, Any]], role: str) -> str | None:
        for message in reversed(messages):
            if message["role"] == role:
                return str(message["content"])
        return None
