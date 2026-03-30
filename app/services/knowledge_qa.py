from __future__ import annotations

from typing import Any

from langchain_core.prompts import ChatPromptTemplate

from app.services.llm import OpenAICompatibleLLM


class KnowledgeGroundedQAService:
    def __init__(self, llm: OpenAICompatibleLLM) -> None:
        self.llm = llm
        self.prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    (
                        "You answer only from the provided Bilibili knowledge context. "
                        "Do not invent facts outside the retrieved evidence. "
                        "If the context is insufficient, say the knowledge base does not contain enough information. "
                        "Keep the answer concise, grounded, and end with a single '来源:' line."
                    ),
                ),
                (
                    "human",
                    (
                        "用户问题:\n{question}\n\n"
                        "检索上下文:\n{context}\n\n"
                        "请直接回答，并在最后追加一行来源，至少包含视频标题；"
                        "如果命中了具体分页，请写出 P{{page_number}}。"
                    ),
                ),
            ]
        )

    def answer(self, *, question: str, retrieval_result: dict[str, Any]) -> str:
        hits = list(retrieval_result.get("hits", []))
        if not hits:
            return "知识库暂无相关内容，暂时无法基于已导入的视频给出回答。"

        if not self.llm.api_key:
            return self._fallback_answer(question=question, hits=hits)

        prompt_messages = self.prompt.format_messages(
            question=question,
            context=str(retrieval_result.get("serialized_context", "")),
        )
        response = self.llm.chat(self._to_chat_messages(prompt_messages))
        return self._ensure_sources(response, hits)

    def _fallback_answer(self, *, question: str, hits: list[dict[str, Any]]) -> str:
        unique_videos: list[dict[str, Any]] = []
        seen_video_ids: set[str] = set()
        for hit in hits:
            video_id = str(hit["video"]["video_id"])
            if video_id in seen_video_ids:
                continue
            seen_video_ids.add(video_id)
            unique_videos.append(hit)
            if len(unique_videos) >= 3:
                break

        if any(token in question for token in ("哪些", "还有", "相关视频")):
            descriptions = []
            for hit in unique_videos:
                excerpt = str(hit["text"]).replace("\n", " ").strip()
                if len(excerpt) > 80:
                    excerpt = f"{excerpt[:77]}..."
                descriptions.append(f"{hit['video']['title']}：{excerpt}")
            reply = "根据知识库检索结果，相关内容包括：\n" + "\n".join(descriptions)
        else:
            primary = hits[0]
            excerpt = str(primary["text"]).replace("\n", " ").strip()
            if len(excerpt) > 160:
                excerpt = f"{excerpt[:157]}..."
            reply = f"根据知识库检索结果，{primary['video']['title']}主要提到：{excerpt}"

        return self._ensure_sources(reply, hits)

    def _ensure_sources(self, response: str, hits: list[dict[str, Any]]) -> str:
        sources = self._format_sources(hits)
        cleaned = response.strip()
        if "来源:" in cleaned:
            return cleaned
        return f"{cleaned}\n\n来源: {sources}"

    def _format_sources(self, hits: list[dict[str, Any]]) -> str:
        formatted: list[str] = []
        seen: set[str] = set()
        for hit in hits:
            pages = ", ".join(
                f"P{page['page_number']}" for page in hit.get("pages", []) if page.get("page_number") is not None
            )
            label = str(hit["video"]["title"])
            if pages:
                label = f"{label} ({pages})"
            if label in seen:
                continue
            seen.add(label)
            formatted.append(label)
            if len(formatted) >= 3:
                break
        return "；".join(formatted)

    def _to_chat_messages(self, prompt_messages: list[Any]) -> list[dict[str, str]]:
        converted: list[dict[str, str]] = []
        role_map = {"human": "user", "ai": "assistant"}
        for message in prompt_messages:
            role = role_map.get(getattr(message, "type", ""), getattr(message, "type", "user"))
            content = message.content if isinstance(message.content, str) else str(message.content)
            converted.append({"role": role, "content": content})
        return converted
