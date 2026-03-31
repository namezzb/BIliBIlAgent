from __future__ import annotations

from pathlib import Path
from typing import Any

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.db.repository import SQLiteRepository


class DuplicateKnowledgeVideoError(ValueError):
    def __init__(self, video_ids: list[str]) -> None:
        self.video_ids = sorted(video_ids)
        joined = ", ".join(self.video_ids)
        super().__init__(f"Videos already indexed and cannot be imported again: {joined}")


class KnowledgeIndexService:
    def __init__(
        self,
        repository: SQLiteRepository,
        *,
        lc_embeddings: Embeddings,
        persist_dir: Path,
        collection_name: str,
        embedding_model: str,
        embedding_version: str,
        chunk_size: int,
        chunk_overlap: int,
    ) -> None:
        self.repository = repository
        self.embedding_model = embedding_model
        self.embedding_version = embedding_version
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("knowledge_chunk_overlap must be smaller than knowledge_chunk_size.")

        # Use langchain-chroma with cosine space so score = cosine similarity in [0,1]
        self.vector_store = Chroma(
            collection_name=collection_name,
            embedding_function=lc_embeddings,
            persist_directory=str(persist_dir),
            collection_metadata={"hnsw:space": "cosine"},
        )

        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            add_start_index=True,
            is_separator_regex=False,
            separators=[
                "\n\n",
                "\n",
                " ",
                ".",
                ",",
                "\uff0c",
                "\u3001",
                "\uff0e",
                "\u3002",
                "",
            ],
        )

    def index_documents(self, payload: dict[str, Any]) -> dict[str, Any]:
        folders = [dict(item) for item in payload.get("favorite_folders", [])]
        videos = [dict(item) for item in payload.get("videos", [])]
        folder_ids = {folder["favorite_folder_id"] for folder in folders}

        if not folders:
            raise ValueError("At least one favorite folder is required.")
        if not videos:
            raise ValueError("At least one video is required.")

        duplicate_video_ids = self.repository.get_existing_knowledge_video_ids(
            [str(video["video_id"]) for video in videos]
        )
        if duplicate_video_ids:
            raise DuplicateKnowledgeVideoError(duplicate_video_ids)

        normalized_videos: list[dict[str, Any]] = []
        favorite_video_links: list[dict[str, str]] = []
        pages: list[dict[str, Any]] = []
        chunks: list[dict[str, Any]] = []
        chunk_page_links: list[dict[str, str]] = []
        lc_docs: list[Document] = []
        vector_ids: list[str] = []

        for video in videos:
            video_id = str(video["video_id"])
            favorite_folder_ids = [str(folder_id) for folder_id in video.get("favorite_folder_ids", [])]
            if not favorite_folder_ids:
                raise ValueError(f"Video {video_id} must reference at least one favorite folder.")
            missing_folders = [folder_id for folder_id in favorite_folder_ids if folder_id not in folder_ids]
            if missing_folders:
                raise ValueError(
                    f"Video {video_id} references unknown favorite folders: {', '.join(missing_folders)}"
                )

            normalized_videos.append(
                {
                    "video_id": video_id,
                    "bvid": video.get("bvid"),
                    "title": video["title"],
                }
            )
            favorite_video_links.extend(
                {
                    "favorite_folder_id": folder_id,
                    "video_id": video_id,
                }
                for folder_id in favorite_folder_ids
            )

            page_items = [dict(page) for page in video.get("pages", [])]
            if not page_items:
                raise ValueError(f"Video {video_id} must include at least one page.")

            pages.extend(
                {
                    "page_id": str(page["page_id"]),
                    "video_id": video_id,
                    "page_number": int(page["page_number"]),
                    "title": page["title"],
                }
                for page in page_items
            )

            page_items.sort(key=lambda item: int(item["page_number"]))
            full_text, page_spans, source_types, source_languages = self._build_video_text(page_items)
            if not full_text:
                raise ValueError(f"Video {video_id} has no usable text blocks for indexing.")

            source_type = self._aggregate_source_type(source_types)
            source_language = self._aggregate_source_language(source_languages)

            for chunk_index, document in enumerate(self._chunk_text(full_text)):
                chunk_text = document.page_content.strip()
                chunk_start = int(document.metadata.get("start_index", 0))
                chunk_end = chunk_start + len(chunk_text)
                chunk_id = f"{video_id}:chunk:{chunk_index}"
                vector_document_id = f"{self.embedding_version}:{chunk_id}"
                chunks.append(
                    {
                        "chunk_id": chunk_id,
                        "video_id": video_id,
                        "source_type": source_type,
                        "source_language": source_language,
                        "block_index": chunk_index,
                        "text": chunk_text,
                        "start_ms": None,
                        "end_ms": None,
                        "embedding_model": self.embedding_model,
                        "embedding_version": self.embedding_version,
                        "index_status": "indexed",
                        "vector_document_id": vector_document_id,
                    }
                )
                chunk_page_links.extend(
                    {
                        "chunk_id": chunk_id,
                        "page_id": str(page["page_id"]),
                    }
                    for page in page_spans
                    if chunk_start < int(page["end_index"]) and chunk_end > int(page["start_index"])
                )
                # Build LangChain Document for vector store (embedding done internally)
                lc_docs.append(
                    Document(
                        page_content=chunk_text,
                        metadata={
                            "chunk_id": chunk_id,
                            "video_id": video_id,
                            "source_type": source_type,
                        },
                    )
                )
                vector_ids.append(vector_document_id)

        # langchain-chroma handles embedding internally via lc_embeddings
        try:
            self.vector_store.add_documents(lc_docs, ids=vector_ids)
        except Exception as exc:
            self._safe_delete_vectors(vector_ids)
            raise RuntimeError(f"Vector index upsert failed: {exc}") from exc

        try:
            self.repository.upsert_knowledge_bundle(
                favorite_folders=folders,
                videos=normalized_videos,
                favorite_video_links=favorite_video_links,
                pages=pages,
                chunks=chunks,
                chunk_page_links=chunk_page_links,
            )
        except Exception:
            self._safe_delete_vectors(vector_ids)
            raise

        return {
            "favorite_folder_count": len(folders),
            "video_count": len(normalized_videos),
            "page_count": len(pages),
            "chunk_count": len(chunks),
            "embedding_model": self.embedding_model,
            "embedding_version": self.embedding_version,
        }

    def search(self, payload: dict[str, Any]) -> dict[str, Any]:
        query = str(payload["query"])
        top_k = int(payload.get("top_k", 5))

        # Check if collection is empty
        if self.vector_store._collection.count() == 0:
            return {"query": query, "total_hits": 0, "hits": []}

        candidate_count = max(top_k * 5, 20)

        # similarity_search_with_relevance_scores returns (Document, score)
        # score is cosine similarity in [0, 1] because hnsw:space=cosine
        try:
            results = self.vector_store.similarity_search_with_relevance_scores(
                query, k=candidate_count
            )
        except Exception as exc:
            raise RuntimeError(f"Vector search failed: {exc}") from exc

        if not results:
            return {"query": query, "total_hits": 0, "hits": []}

        candidate_chunk_ids: list[str] = []
        score_by_chunk_id: dict[str, float] = {}
        for doc, score in results:
            chunk_id = doc.metadata.get("chunk_id")
            if chunk_id:
                candidate_chunk_ids.append(str(chunk_id))
                score_by_chunk_id[str(chunk_id)] = float(score)

        if not candidate_chunk_ids:
            return {"query": query, "total_hits": 0, "hits": []}

        details = self.repository.get_knowledge_chunk_details(candidate_chunk_ids)
        filtered_hits: list[dict[str, Any]] = []
        for detail in details:
            if not self._matches_filters(detail, payload):
                continue
            filtered_hits.append(
                {
                    "score": score_by_chunk_id.get(detail["chunk_id"], 0.0),
                    "chunk_id": detail["chunk_id"],
                    "text": detail["text"],
                    "source_type": detail["source_type"],
                    "source_language": detail["source_language"],
                    "start_ms": detail["start_ms"],
                    "end_ms": detail["end_ms"],
                    "favorite_folders": detail["favorite_folders"],
                    "pages": detail.get("pages", []),
                    "video": detail["video"],
                }
            )
            if len(filtered_hits) >= top_k:
                break

        return {
            "query": query,
            "total_hits": len(filtered_hits),
            "hits": filtered_hits,
        }

    def _build_video_text(
        self,
        page_items: list[dict[str, Any]],
    ) -> tuple[str, list[dict[str, Any]], set[str], set[str]]:
        full_text_parts: list[str] = []
        page_spans: list[dict[str, Any]] = []
        source_types: set[str] = set()
        source_languages: set[str] = set()
        cursor = 0

        for page in page_items:
            page_text_parts: list[str] = []
            for block in page.get("text_blocks", []):
                block_text = str(block["text"]).strip()
                if not block_text:
                    continue
                page_text_parts.append(block_text)
                source_types.add(str(block["source_type"]))
                language = block.get("source_language")
                if language:
                    source_languages.add(str(language))

            page_text = "\n".join(page_text_parts).strip()
            if not page_text:
                continue

            if full_text_parts:
                full_text_parts.append("\n")
                cursor += 1

            start_index = cursor
            full_text_parts.append(page_text)
            cursor += len(page_text)
            page_spans.append(
                {
                    "page_id": str(page["page_id"]),
                    "page_number": int(page["page_number"]),
                    "title": page["title"],
                    "start_index": start_index,
                    "end_index": cursor,
                }
            )

        return "".join(full_text_parts), page_spans, source_types, source_languages

    def _chunk_text(self, text: str) -> list[Document]:
        content = text.strip()
        if not content:
            return []
        return [
            document
            for document in self.text_splitter.create_documents([content])
            if document.page_content.strip()
        ]

    def _matches_filters(self, detail: dict[str, Any], payload: dict[str, Any]) -> bool:
        favorite_folder_ids = {str(item) for item in payload.get("favorite_folder_ids", [])}
        video_ids = {str(item) for item in payload.get("video_ids", [])}
        source_types = {str(item) for item in payload.get("source_types", [])}
        page_numbers = {int(item) for item in payload.get("page_numbers", [])}

        if favorite_folder_ids:
            detail_folder_ids = {
                str(folder["favorite_folder_id"]) for folder in detail.get("favorite_folders", [])
            }
            if not detail_folder_ids.intersection(favorite_folder_ids):
                return False
        if video_ids and str(detail["video"]["video_id"]) not in video_ids:
            return False
        if source_types and str(detail["source_type"]) not in source_types:
            return False
        if page_numbers:
            detail_page_numbers = {
                int(page["page_number"]) for page in detail.get("pages", []) if page.get("page_number") is not None
            }
            if not detail_page_numbers.intersection(page_numbers):
                return False
        return True

    def _aggregate_source_type(self, source_types: set[str]) -> str:
        if not source_types:
            return "unknown"
        if len(source_types) == 1:
            return next(iter(source_types))
        return "mixed"

    def _aggregate_source_language(self, source_languages: set[str]) -> str | None:
        if not source_languages:
            return None
        if len(source_languages) == 1:
            return next(iter(source_languages))
        return "mixed"

    def _safe_delete_vectors(self, vector_ids: list[str]) -> None:
        try:
            self.vector_store.delete(ids=vector_ids)
        except Exception:
            pass
