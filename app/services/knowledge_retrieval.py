from __future__ import annotations

import re
from typing import Any

from langchain_core.documents import Document

from app.db.repository import SQLiteRepository
from app.services.knowledge_index import KnowledgeIndexService


BV_PATTERN = re.compile(r"\bBV[0-9A-Za-z]+\b", re.IGNORECASE)
AV_PATTERN = re.compile(r"\bav\d+\b", re.IGNORECASE)
PAGE_PATTERN = re.compile(r"(?:第\s*(\d+)\s*[Pp集期]|(?:^|[^\w])([Pp])\s*(\d+)\b|\b(\d+)\s*[Pp]\b)")
SOURCE_TYPE_PATTERN = re.compile(r"\b(asr|subtitle|字幕)\b", re.IGNORECASE)
PREVIOUS_SCOPE_HINTS = (
    "这个视频",
    "这期视频",
    "该视频",
    "这个收藏夹",
    "该收藏夹",
    "上一条",
    "上一个",
    "还讲了什么",
    "还有哪些",
    "继续",
    "那这个",
)


class KnowledgeRetrievalService:
    def __init__(
        self,
        repository: SQLiteRepository,
        knowledge_index: KnowledgeIndexService,
    ) -> None:
        self.repository = repository
        self.knowledge_index = knowledge_index

    def search(self, payload: dict[str, Any]) -> dict[str, Any]:
        return self.knowledge_index.search(payload)

    def retrieve_for_question(
        self,
        *,
        message: str,
        route: str | None,
        recent_context: dict[str, object] | None = None,
        top_k: int = 5,
    ) -> dict[str, Any]:
        resolved_scope = self._resolve_scope(
            message=message,
            route=route,
            recent_context=recent_context or {},
        )
        search_payload = {
            "query": message,
            "top_k": top_k,
            "favorite_folder_ids": resolved_scope["favorite_folder_ids"],
            "video_ids": resolved_scope["video_ids"],
            "page_numbers": resolved_scope["page_numbers"],
            "source_types": resolved_scope["source_types"],
        }
        search_result = self.search(search_payload)
        hits = list(search_result.get("hits", []))
        documents = self._build_documents(hits)
        top_sources = self._top_sources(hits)
        return {
            "query": message,
            "route": route,
            "resolved_scope": resolved_scope,
            "total_hits": int(search_result.get("total_hits", 0)),
            "hits": hits,
            "documents": documents,
            "serialized_context": self._serialize_hits(hits),
            "top_sources": top_sources,
        }

    def _resolve_scope(
        self,
        *,
        message: str,
        route: str | None,
        recent_context: dict[str, object],
    ) -> dict[str, list[Any]]:
        lowered = message.lower()
        favorite_folder_ids: list[str] = []
        video_ids: list[str] = []
        page_numbers: list[int] = sorted(self._extract_page_numbers(message))
        source_types: list[str] = sorted(self._extract_source_types(lowered))

        videos = self.repository.list_knowledge_videos()
        folders = self.repository.list_knowledge_favorite_folders()

        explicit_video_ids = self._match_video_ids(lowered, videos)
        explicit_folder_ids = self._match_favorite_folder_ids(lowered, folders)

        if explicit_video_ids:
            video_ids = explicit_video_ids
        if explicit_folder_ids:
            favorite_folder_ids = explicit_folder_ids

        previous_scope = self._get_previous_scope(recent_context)
        if self._should_use_previous_scope(lowered) or (
            route == "video_knowledge_query" and not video_ids and previous_scope.get("video_ids")
        ):
            video_ids = video_ids or list(previous_scope.get("video_ids", []))
            if route == "favorite_knowledge_query":
                favorite_folder_ids = favorite_folder_ids or list(
                    previous_scope.get("favorite_folder_ids", [])
                )

        if route == "favorite_knowledge_query" and not favorite_folder_ids:
            favorite_folder_ids = list(previous_scope.get("favorite_folder_ids", []))
        if route == "video_knowledge_query" and not video_ids:
            video_ids = list(previous_scope.get("video_ids", []))

        if page_numbers and not video_ids and previous_scope.get("video_ids"):
            video_ids = list(previous_scope.get("video_ids", []))

        return {
            "favorite_folder_ids": favorite_folder_ids,
            "video_ids": video_ids,
            "page_numbers": page_numbers,
            "source_types": source_types,
        }

    def _get_previous_scope(self, recent_context: dict[str, object]) -> dict[str, list[Any]]:
        last_retrieval = recent_context.get("last_retrieval")
        if not isinstance(last_retrieval, dict):
            return {}
        scope = last_retrieval.get("resolved_scope")
        if not isinstance(scope, dict):
            return {}
        return {
            "favorite_folder_ids": list(scope.get("favorite_folder_ids", [])),
            "video_ids": list(scope.get("video_ids", [])),
            "page_numbers": list(scope.get("page_numbers", [])),
            "source_types": list(scope.get("source_types", [])),
        }

    def _match_video_ids(self, lowered: str, videos: list[dict[str, Any]]) -> list[str]:
        matches: list[str] = []
        bv_matches = {match.upper() for match in BV_PATTERN.findall(lowered)}
        av_matches = {match.lower() for match in AV_PATTERN.findall(lowered)}
        for video in videos:
            video_id = str(video["video_id"])
            bvid = str(video.get("bvid") or "").upper()
            title = str(video["title"]).lower()
            if video_id.lower() in lowered or (bvid and bvid in bv_matches) or title in lowered:
                matches.append(video_id)
                continue
            if video.get("video_id") and str(video["video_id"]).lower() in av_matches:
                matches.append(video_id)
        return sorted(set(matches))

    def _match_favorite_folder_ids(
        self,
        lowered: str,
        folders: list[dict[str, Any]],
    ) -> list[str]:
        matches: list[str] = []
        for folder in folders:
            favorite_folder_id = str(folder["favorite_folder_id"])
            title = str(folder["title"]).lower()
            if favorite_folder_id.lower() in lowered or (title and title in lowered):
                matches.append(favorite_folder_id)
        return sorted(set(matches))

    def _extract_page_numbers(self, message: str) -> set[int]:
        page_numbers: set[int] = set()
        for match in PAGE_PATTERN.findall(message):
            for candidate in match:
                if candidate and candidate.isdigit():
                    page_numbers.add(int(candidate))
        return page_numbers

    def _extract_source_types(self, lowered: str) -> set[str]:
        matches: set[str] = set()
        for match in SOURCE_TYPE_PATTERN.findall(lowered):
            token = match.lower()
            if token == "字幕":
                matches.add("subtitle")
            else:
                matches.add(token)
        return matches

    def _should_use_previous_scope(self, lowered: str) -> bool:
        return any(token in lowered for token in PREVIOUS_SCOPE_HINTS)

    def _build_documents(self, hits: list[dict[str, Any]]) -> list[Document]:
        documents: list[Document] = []
        for hit in hits:
            documents.append(
                Document(
                    page_content=str(hit["text"]),
                    metadata={
                        "chunk_id": hit["chunk_id"],
                        "score": hit["score"],
                        "source_type": hit["source_type"],
                        "source_language": hit.get("source_language"),
                        "favorite_folders": hit.get("favorite_folders", []),
                        "pages": hit.get("pages", []),
                        "video": hit["video"],
                    },
                )
            )
        return documents

    def _serialize_hits(self, hits: list[dict[str, Any]]) -> str:
        if not hits:
            return ""

        serialized_parts: list[str] = []
        for index, hit in enumerate(hits, start=1):
            folder_titles = ", ".join(
                str(folder["title"]) for folder in hit.get("favorite_folders", []) if folder.get("title")
            )
            page_labels = ", ".join(
                f"P{page['page_number']} {page['title']}"
                for page in hit.get("pages", [])
                if page.get("page_number") is not None
            )
            serialized_parts.append(
                "\n".join(
                    [
                        f"[Hit {index}] score={float(hit['score']):.3f}",
                        f"Video: {hit['video']['title']} ({hit['video'].get('bvid') or hit['video']['video_id']})",
                        f"Favorite folders: {folder_titles or 'N/A'}",
                        f"Pages: {page_labels or 'N/A'}",
                        f"Source type: {hit['source_type']}",
                        f"Content: {hit['text']}",
                    ]
                )
            )
        return "\n\n".join(serialized_parts)

    def _top_sources(self, hits: list[dict[str, Any]]) -> list[str]:
        sources: list[str] = []
        seen: set[str] = set()
        for hit in hits:
            source = self._format_source_label(hit)
            if source in seen:
                continue
            seen.add(source)
            sources.append(source)
            if len(sources) >= 3:
                break
        return sources

    def _format_source_label(self, hit: dict[str, Any]) -> str:
        folder_titles = [str(folder["title"]) for folder in hit.get("favorite_folders", []) if folder.get("title")]
        page_numbers = [f"P{page['page_number']}" for page in hit.get("pages", []) if page.get("page_number") is not None]
        parts = [str(hit["video"]["title"])]
        if folder_titles:
            parts.append(f"收藏夹: {' / '.join(folder_titles)}")
        if page_numbers:
            parts.append(f"分页: {', '.join(page_numbers)}")
        return " | ".join(parts)
