from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Protocol

import chromadb
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.db.repository import SQLiteRepository


class DuplicateKnowledgeVideoError(ValueError):
    def __init__(self, video_ids: list[str]) -> None:
        self.video_ids = sorted(video_ids)
        joined = ", ".join(self.video_ids)
        super().__init__(f"Videos already indexed and cannot be imported again: {joined}")


class VectorIndex(Protocol):
    def upsert(
        self,
        *,
        ids: list[str],
        embeddings: list[list[float]],
        documents: list[str],
        metadatas: list[dict[str, Any]],
    ) -> None: ...

    def delete(self, ids: list[str]) -> None: ...

    def query(self, *, query_embedding: list[float], n_results: int) -> dict[str, Any]: ...

    def count(self) -> int: ...


class ChromaVectorIndex:
    def __init__(self, persist_dir: Path, collection_name: str) -> None:
        self.client = chromadb.PersistentClient(path=str(persist_dir))
        self.collection = self.client.get_or_create_collection(name=collection_name)

    def upsert(
        self,
        *,
        ids: list[str],
        embeddings: list[list[float]],
        documents: list[str],
        metadatas: list[dict[str, Any]],
    ) -> None:
        self.collection.upsert(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas,
        )

    def delete(self, ids: list[str]) -> None:
        if not ids:
            return
        self.collection.delete(ids=ids)

    def query(self, *, query_embedding: list[float], n_results: int) -> dict[str, Any]:
        return self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            include=["distances", "documents", "metadatas"],
        )

    def count(self) -> int:
        return int(self.collection.count())


class KnowledgeIndexService:
    def __init__(
        self,
        repository: SQLiteRepository,
        *,
        vector_index: VectorIndex,
        embed_texts: Callable[[list[str]], list[list[float]]],
        embedding_model: str,
        embedding_version: str,
        chunk_size: int,
        chunk_overlap: int,
    ) -> None:
        self.repository = repository
        self.vector_index = vector_index
        self.embed_texts = embed_texts
        self.embedding_model = embedding_model
        self.embedding_version = embedding_version
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("knowledge_chunk_overlap must be smaller than knowledge_chunk_size.")
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
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
        vector_ids: list[str] = []
        vector_texts: list[str] = []
        vector_metadatas: list[dict[str, Any]] = []

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
            aggregated_text_parts: list[str] = []
            source_types: set[str] = set()
            source_languages: set[str] = set()

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
                if page_text_parts:
                    aggregated_text_parts.append("\n".join(page_text_parts))

            full_text = "\n".join(aggregated_text_parts).strip()
            if not full_text:
                raise ValueError(f"Video {video_id} has no usable text blocks for indexing.")

            source_type = self._aggregate_source_type(source_types)
            source_language = self._aggregate_source_language(source_languages)

            for chunk_index, chunk_text in enumerate(self._chunk_text(full_text)):
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
                vector_ids.append(vector_document_id)
                vector_texts.append(chunk_text)
                vector_metadatas.append(
                    {
                        "chunk_id": chunk_id,
                        "video_id": video_id,
                        "source_type": source_type,
                    }
                )

        try:
            embeddings = self.embed_texts(vector_texts)
        except Exception as exc:
            raise RuntimeError(f"Embedding generation failed: {exc}") from exc

        try:
            self.vector_index.upsert(
                ids=vector_ids,
                embeddings=embeddings,
                documents=vector_texts,
                metadatas=vector_metadatas,
            )
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
        if self.vector_index.count() == 0:
            return {"query": query, "total_hits": 0, "hits": []}

        try:
            query_embedding = self.embed_texts([query])[0]
        except Exception as exc:
            raise RuntimeError(f"Embedding generation failed: {exc}") from exc

        candidate_count = max(top_k * 5, 20)
        raw = self.vector_index.query(query_embedding=query_embedding, n_results=candidate_count)
        ids = raw.get("ids", [[]])
        distances = raw.get("distances", [[]])
        if not ids or not ids[0]:
            return {"query": query, "total_hits": 0, "hits": []}

        candidate_ids = [
            str(vector_id).split(":", 1)[1]
            for vector_id in ids[0]
            if ":" in str(vector_id)
        ]
        distance_by_chunk_id = {
            str(vector_id).split(":", 1)[1]: float(distance)
            for vector_id, distance in zip(ids[0], distances[0], strict=False)
            if ":" in str(vector_id)
        }

        details = self.repository.get_knowledge_chunk_details(candidate_ids)
        filtered_hits: list[dict[str, Any]] = []
        for detail in details:
            if not self._matches_filters(detail, payload):
                continue
            filtered_hits.append(
                {
                    "score": max(0.0, 1.0 - distance_by_chunk_id.get(detail["chunk_id"], 1.0)),
                    "chunk_id": detail["chunk_id"],
                    "text": detail["text"],
                    "source_type": detail["source_type"],
                    "source_language": detail["source_language"],
                    "start_ms": detail["start_ms"],
                    "end_ms": detail["end_ms"],
                    "favorite_folders": detail["favorite_folders"],
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

    def _chunk_text(self, text: str) -> list[str]:
        content = text.strip()
        if not content:
            return []
        return [chunk.strip() for chunk in self.text_splitter.split_text(content) if chunk.strip()]

    def _matches_filters(self, detail: dict[str, Any], payload: dict[str, Any]) -> bool:
        favorite_folder_ids = {str(item) for item in payload.get("favorite_folder_ids", [])}
        video_ids = {str(item) for item in payload.get("video_ids", [])}
        source_types = {str(item) for item in payload.get("source_types", [])}

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
            self.vector_index.delete(vector_ids)
        except Exception:
            pass
