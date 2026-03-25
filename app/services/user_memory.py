from __future__ import annotations

import re
from datetime import datetime, timezone
from typing import Any

from app.db.repository import SQLiteRepository


USER_MEMORY_GROUPS = ("preferences", "aliases", "default_scopes")
UPSERT_COMMAND_GROUPS = {
    "记住偏好": "preferences",
    "记住别名": "aliases",
    "记住默认范围": "default_scopes",
}
DELETE_COMMAND_GROUPS = {
    "删除偏好": "preferences",
    "删除别名": "aliases",
    "删除默认范围": "default_scopes",
}


class UserMemoryManager:
    def __init__(self, repository: SQLiteRepository) -> None:
        self.repository = repository

    def is_chat_command(self, message: str) -> bool:
        return self.parse_chat_command(message) is not None

    def parse_chat_command(self, message: str) -> dict[str, str] | None:
        content = message.strip()
        if content == "查看长期记忆":
            return {"operation": "list"}

        upsert_match = re.match(r"^(记住偏好|记住别名|记住默认范围)\s*[:：]\s*([^=]+?)\s*=\s*(.+)$", content)
        if upsert_match:
            command, key, value = upsert_match.groups()
            return {
                "operation": "upsert",
                "group": UPSERT_COMMAND_GROUPS[command],
                "key": key.strip(),
                "value": value.strip(),
            }

        delete_match = re.match(r"^(删除偏好|删除别名|删除默认范围)\s*[:：]\s*(.+)$", content)
        if delete_match:
            command, key = delete_match.groups()
            return {
                "operation": "delete",
                "group": DELETE_COMMAND_GROUPS[command],
                "key": key.strip(),
            }

        return None

    def get_profile_detail(self, user_id: str) -> dict[str, Any]:
        profile_record = self.repository.get_user_memory_profile(user_id)
        if profile_record is None:
            return {
                "user_id": user_id,
                "preferences": {},
                "aliases": {},
                "default_scopes": {},
                "created_at": None,
                "updated_at": None,
            }

        profile = self._normalize_profile(profile_record.get("profile"))
        return {
            "user_id": user_id,
            "preferences": profile["preferences"],
            "aliases": profile["aliases"],
            "default_scopes": profile["default_scopes"],
            "created_at": profile_record["created_at"],
            "updated_at": profile_record["updated_at"],
        }

    def build_context_message(self, user_id: str) -> str | None:
        detail = self.get_profile_detail(user_id)
        sections: list[str] = []
        labels = {
            "preferences": "Preferences",
            "aliases": "Aliases",
            "default_scopes": "Default scopes",
        }

        for group in USER_MEMORY_GROUPS:
            entries = detail[group]
            if not entries:
                continue
            lines = [f"{labels[group]}:"]
            for key, entry in sorted(entries.items()):
                lines.append(f"- {key}: {entry['value']}")
            sections.append("\n".join(lines))

        if not sections:
            return None

        return (
            "[User long-term memory. Apply it only when relevant and do not override the "
            "user's current request.]\n"
            + "\n\n".join(sections)
        )

    def apply_chat_command(self, user_id: str, message: str, source_run_id: str) -> str:
        command = self.parse_chat_command(message)
        if command is None:
            raise ValueError("Message is not a supported user-memory command.")

        operation = command["operation"]
        if operation == "list":
            return self._format_profile_summary(self.get_profile_detail(user_id))

        if operation == "delete":
            deleted = self.delete_entry(user_id, command["group"], command["key"])
            if deleted:
                return f"已删除长期记忆：{command['group']}.{command['key']}。"
            return f"没有找到长期记忆：{command['group']}.{command['key']}。"

        self.upsert_entries(
            user_id,
            {command["group"]: {command["key"]: command["value"]}},
            source_type="chat_command",
            source_run_id=source_run_id,
            source_text=message,
        )
        return f"已保存长期记忆：{command['group']}.{command['key']} = {command['value']}。"

    def upsert_entries(
        self,
        user_id: str,
        updates: dict[str, dict[str, str]],
        *,
        source_type: str,
        source_run_id: str | None,
        source_text: str | None,
    ) -> dict[str, Any]:
        profile_record = self.repository.get_user_memory_profile(user_id)
        profile = self._normalize_profile(
            profile_record["profile"] if profile_record is not None else None
        )
        timestamp = self._utc_now()

        for group, entries in updates.items():
            if group not in USER_MEMORY_GROUPS:
                raise ValueError(f"Unsupported user-memory group: {group}")
            for key, value in entries.items():
                existing_entry = profile[group].get(key, {})
                profile[group][key] = {
                    "value": value,
                    "source_type": source_type,
                    "source_run_id": source_run_id,
                    "source_text": source_text,
                    "confirmed": True,
                    "created_at": existing_entry.get("created_at", timestamp),
                    "updated_at": timestamp,
                }

        stored = self.repository.upsert_user_memory_profile(user_id, profile)
        return self.get_profile_detail(stored["user_id"])

    def delete_entry(self, user_id: str, group: str, key: str) -> bool:
        profile_record = self.repository.get_user_memory_profile(user_id)
        if profile_record is None:
            return False

        profile = self._normalize_profile(profile_record["profile"])
        group_entries = profile[group]
        if key not in group_entries:
            return False

        del group_entries[key]
        self.repository.upsert_user_memory_profile(user_id, profile)
        return True

    def _normalize_profile(self, profile: dict[str, Any] | None) -> dict[str, dict[str, Any]]:
        normalized = {group: {} for group in USER_MEMORY_GROUPS}
        if not profile:
            return normalized

        for group in USER_MEMORY_GROUPS:
            group_entries = profile.get(group, {})
            if isinstance(group_entries, dict):
                normalized[group] = dict(group_entries)
        return normalized

    def _format_profile_summary(self, detail: dict[str, Any]) -> str:
        lines = ["当前长期记忆："]
        labels = {
            "preferences": "偏好",
            "aliases": "别名",
            "default_scopes": "默认范围",
        }
        has_entries = False

        for group in USER_MEMORY_GROUPS:
            entries = detail[group]
            if not entries:
                continue
            has_entries = True
            lines.append(f"{labels[group]}:")
            for key, entry in sorted(entries.items()):
                lines.append(f"- {key} = {entry['value']}")

        if not has_entries:
            return "当前还没有保存任何长期记忆。"

        return "\n".join(lines)

    def _utc_now(self) -> str:
        return datetime.now(timezone.utc).isoformat()
