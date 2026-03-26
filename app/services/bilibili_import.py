from __future__ import annotations

from typing import Any

from app.db.repository import SQLiteRepository
from app.services.bilibili_favorites import BilibiliFavoriteFolderService
from app.services.knowledge_index import DuplicateKnowledgeVideoError, KnowledgeIndexService
from app.services.runtime_audit import LangSmithRuntimeAudit, NoOpRuntimeAudit


class BilibiliImportPipeline:
    def __init__(
        self,
        *,
        repository: SQLiteRepository,
        favorites_service: BilibiliFavoriteFolderService,
        knowledge_index: KnowledgeIndexService,
        runtime_audit: LangSmithRuntimeAudit | None = None,
    ) -> None:
        self.repository = repository
        self.favorites_service = favorites_service
        self.knowledge_index = knowledge_index
        self.runtime_audit = runtime_audit or NoOpRuntimeAudit()

    def build_execution_plan(
        self,
        *,
        favorite_folder_id: str,
        selected_video_ids: list[str],
    ) -> dict[str, Any]:
        step_status = "approved"
        return {
            "goal": "Import the selected Bilibili videos into the knowledge base.",
            "summary": (
                "The backend will fetch video metadata, download subtitles when available, "
                "prepare ASR fallback jobs when subtitles are missing, and index subtitle "
                "text into the knowledge base."
            ),
            "steps": [
                {
                    "id": "tool_call_1",
                    "title": "Run bilibili_import.execute_import",
                    "description": (
                        f"Import {len(selected_video_ids)} selected videos from favorite folder "
                        f"{favorite_folder_id}."
                    ),
                    "tool": "bilibili_import",
                    "action": "execute_import",
                    "status": step_status,
                }
            ],
            "tool_calls": [
                {
                    "tool": "bilibili_import",
                    "action": "execute_import",
                    "target": favorite_folder_id,
                    "description": (
                        f"Execute the import pipeline for {len(selected_video_ids)} selected videos."
                    ),
                    "args": {
                        "favorite_folder_id": favorite_folder_id,
                        "selected_video_ids": list(selected_video_ids),
                    },
                    "side_effect": True,
                }
            ],
        }

    def handle_agent_import_request(
        self,
        *,
        request_message: str | None,
        target: str | None,
    ) -> str:
        scoped_target = target or "favorite-folder-ingestion"
        return (
            "Execution was approved, but the real import pipeline requires explicit "
            "favorite_folder_id, selected_video_ids, and a current Bilibili Cookie. "
            f"Current agent request target={scoped_target}. Use the favorite-folder "
            "selection flow or submit `/api/bilibili/imports` after choosing videos."
        )

    def validate_selected_items(
        self,
        *,
        cookie: str,
        favorite_folder_id: str,
        selected_video_ids: list[str],
    ) -> dict[str, Any]:
        normalized_ids = [str(video_id).strip() for video_id in selected_video_ids if str(video_id).strip()]
        unique_ids = list(dict.fromkeys(normalized_ids))
        if not unique_ids:
            raise ValueError("At least one selected_video_id is required.")

        payload = self.favorites_service.list_all_folder_items(cookie, favorite_folder_id)
        item_by_video_id = {
            str(item["video_id"]): item
            for item in payload["items"]
            if item.get("video_id")
        }
        missing_ids = [video_id for video_id in unique_ids if video_id not in item_by_video_id]
        if missing_ids:
            raise ValueError(
                "Selected videos are not present in the favorite folder: "
                + ", ".join(sorted(missing_ids))
            )

        selected_items = [item_by_video_id[video_id] for video_id in unique_ids]
        unsupported = [item["video_id"] for item in selected_items if not item.get("selectable")]
        if unsupported:
            raise ValueError(
                "Selected items contain unsupported favorites: " + ", ".join(sorted(unsupported))
            )

        return {
            "account": payload["account"],
            "folder": payload["folder"],
            "selected_video_ids": unique_ids,
            "selected_items": selected_items,
        }

    def execute_selected_videos(
        self,
        *,
        run_id: str,
        session_id: str,
        user_id: str | None,
        cookie: str,
        favorite_folder_id: str,
        selected_video_ids: list[str],
    ) -> str:
        existing_run = self.repository.get_run(run_id)

        with self.runtime_audit.trace_request(
            name="agent.import_selected_videos",
            inputs={
                "run_id": run_id,
                "session_id": session_id,
                "favorite_folder_id": favorite_folder_id,
                "selected_video_ids": list(selected_video_ids),
            },
            metadata={
                "thread_id": run_id,
                "run_id": run_id,
                "session_id": session_id,
                "user_id": user_id,
                "environment": self.runtime_audit.environment,
                "app_name": self.runtime_audit.app_name,
                "operation": "bilibili_import",
            },
            tags=["agent", "import", self.runtime_audit.environment],
        ) as trace_run:
            try:
                self.repository.update_run(
                    run_id,
                    intent="tool_request",
                    route="import_request",
                    status="running",
                    requires_confirmation=False,
                    approval_status="approved",
                )

                self.repository.upsert_run_step(
                    run_id,
                    "execute_import",
                    "bilibili_import.execute_import",
                    "running",
                    input_summary=f"favorite_folder_id={favorite_folder_id}",
                    output_summary=f"queued {len(selected_video_ids)} selected video(s)",
                )
                self.repository.append_run_event(
                    run_id,
                    "import_started",
                    {
                        "route": "import_request",
                        "favorite_folder_id": favorite_folder_id,
                        "selected_video_count": len(selected_video_ids),
                    },
                )

                validated = self.validate_selected_items(
                    cookie=cookie,
                    favorite_folder_id=favorite_folder_id,
                    selected_video_ids=selected_video_ids,
                )
                folder = dict(validated["folder"])
                selected_items = [dict(item) for item in validated["selected_items"]]
                self.repository.upsert_run_step(
                    run_id,
                    "validate_selection",
                    "validate_selection",
                    "completed",
                    input_summary=f"{len(selected_video_ids)} selection(s)",
                    output_summary=f"validated {len(selected_items)} selected video(s)",
                )
                self.repository.append_run_event(
                    run_id,
                    "import_selection_validated",
                    {
                        "route": "import_request",
                        "favorite_folder_id": favorite_folder_id,
                        "selected_video_ids": [item["video_id"] for item in selected_items],
                    },
                )

                existing_video_ids = set(
                    self.repository.get_existing_knowledge_video_ids(
                        [str(item["video_id"]) for item in selected_items]
                    )
                )
                indexable_videos: list[dict[str, Any]] = []
                pending_index_items: list[dict[str, Any]] = []
                stats = {
                    "indexed": 0,
                    "needs_asr": 0,
                    "failed": 0,
                    "skipped_duplicate": 0,
                }

                for item in selected_items:
                    video_id = str(item["video_id"])
                    if video_id in existing_video_ids:
                        self.repository.upsert_import_run_item(
                            run_id,
                            favorite_folder_id=favorite_folder_id,
                            video_id=video_id,
                            bvid=item.get("bvid"),
                            title=str(item.get("title") or video_id),
                            status="skipped_duplicate",
                            manifest={
                                "video_id": video_id,
                                "favorite_folder_id": favorite_folder_id,
                            },
                        )
                        self.repository.upsert_run_step(
                            run_id,
                            f"import_item_{video_id}",
                            f"import_item:{video_id}",
                            "completed",
                            input_summary="duplicate check",
                            output_summary="skipped duplicate knowledge video",
                        )
                        self.repository.append_run_event(
                            run_id,
                            "import_item_processed",
                            {
                                "video_id": video_id,
                                "status": "skipped_duplicate",
                                "title": item.get("title"),
                            },
                        )
                        stats["skipped_duplicate"] += 1
                        continue

                    item_result = self._process_selected_item(
                        run_id=run_id,
                        favorite_folder=folder,
                        item=item,
                        cookie=cookie,
                    )
                    if item_result["status"] == "ready_for_index":
                        indexable_videos.append(item_result["knowledge_video"])
                        pending_index_items.append(item_result)
                        continue

                    stats[str(item_result["status"])] += 1

                if indexable_videos:
                    try:
                        index_result = self.knowledge_index.index_documents(
                            {
                                "favorite_folders": [folder],
                                "videos": indexable_videos,
                            }
                        )
                    except DuplicateKnowledgeVideoError as exc:
                        duplicate_ids = set(exc.video_ids)
                        for item_result in pending_index_items:
                            video_id = str(item_result["video_id"])
                            if video_id not in duplicate_ids:
                                continue
                            self.repository.upsert_import_run_item(
                                run_id,
                                favorite_folder_id=favorite_folder_id,
                                video_id=video_id,
                                bvid=item_result.get("bvid"),
                                title=str(item_result.get("title") or video_id),
                                status="skipped_duplicate",
                                manifest=item_result.get("manifest"),
                            )
                            self.repository.upsert_run_step(
                                run_id,
                                f"import_item_{video_id}",
                                f"import_item:{video_id}",
                                "completed",
                                input_summary="knowledge index duplicate check",
                                output_summary="skipped duplicate during indexing",
                            )
                            self.repository.append_run_event(
                                run_id,
                                "import_item_processed",
                                {
                                    "video_id": video_id,
                                    "status": "skipped_duplicate",
                                    "title": item_result.get("title"),
                                },
                            )
                            stats["skipped_duplicate"] += 1
                    else:
                        self.repository.upsert_run_step(
                            run_id,
                            "knowledge_index",
                            "knowledge_index",
                            "completed",
                            input_summary=f"{len(indexable_videos)} subtitle-backed video(s)",
                            output_summary=(
                                f"indexed {index_result['video_count']} video(s), "
                                f"{index_result['chunk_count']} chunk(s)"
                            ),
                        )
                        self.repository.append_run_event(
                            run_id,
                            "import_index_completed",
                            {
                                "video_count": index_result["video_count"],
                                "chunk_count": index_result["chunk_count"],
                                "page_count": index_result["page_count"],
                            },
                        )
                        for item_result in pending_index_items:
                            video_id = str(item_result["video_id"])
                            self.repository.upsert_import_run_item(
                                run_id,
                                favorite_folder_id=favorite_folder_id,
                                video_id=video_id,
                                bvid=item_result.get("bvid"),
                                title=str(item_result.get("title") or video_id),
                                status="indexed",
                                manifest=item_result.get("manifest"),
                            )
                            self.repository.upsert_run_step(
                                run_id,
                                f"import_item_{video_id}",
                                f"import_item:{video_id}",
                                "completed",
                                input_summary="subtitle pipeline",
                                output_summary="indexed into knowledge base",
                            )
                            self.repository.append_run_event(
                                run_id,
                                "import_item_processed",
                                {
                                    "video_id": video_id,
                                    "status": "indexed",
                                    "title": item_result.get("title"),
                                },
                            )
                            stats["indexed"] += 1

                final_reply = self._build_completion_reply(stats)
                execution_plan = self._mark_execution_plan_status(
                    self.build_execution_plan(
                        favorite_folder_id=favorite_folder_id,
                        selected_video_ids=selected_video_ids,
                    ),
                    "completed",
                )
                self.repository.update_run(
                    run_id,
                    status="completed",
                    latest_reply=final_reply,
                    execution_plan=execution_plan,
                    approval_status="approved",
                )
                self.repository.upsert_run_step(
                    run_id,
                    "execute_import",
                    "bilibili_import.execute_import",
                    "completed",
                    input_summary=f"favorite_folder_id={favorite_folder_id}",
                    output_summary=final_reply,
                )
                self.repository.append_run_event(
                    run_id,
                    "run_completed",
                    {
                        "status": "completed",
                        "route": "import_request",
                        "reply": final_reply,
                        "execution_plan": execution_plan,
                    },
                )
                trace_run.end(
                    outputs=self.runtime_audit.sanitize_payload(
                        {
                            "run_id": run_id,
                            "status": "completed",
                            "summary": stats,
                            "reply": final_reply,
                        }
                    )
                )
                return final_reply
            except Exception as exc:
                detail = f"Import run failed: {exc}"
                execution_plan = self._mark_execution_plan_status(
                    self.build_execution_plan(
                        favorite_folder_id=favorite_folder_id,
                        selected_video_ids=selected_video_ids,
                    ),
                    "failed",
                )
                self.repository.upsert_run_step(
                    run_id,
                    "execute_import",
                    "bilibili_import.execute_import",
                    "failed",
                    input_summary=f"favorite_folder_id={favorite_folder_id}",
                    output_summary=detail,
                )
                self.repository.append_run_event(
                    run_id,
                    "run_failed",
                    {
                        "status": "failed",
                        "route": "import_request",
                        "reply": detail,
                        "execution_plan": execution_plan,
                    },
                )
                self.repository.update_run(
                    run_id,
                    status="failed",
                    latest_reply=detail,
                    execution_plan=execution_plan,
                    approval_status="approved",
                )
                trace_run.end(
                    outputs=self.runtime_audit.sanitize_payload(
                        {"run_id": run_id, "status": "failed", "error": detail}
                    )
                )
                raise

    def _process_selected_item(
        self,
        *,
        run_id: str,
        favorite_folder: dict[str, Any],
        item: dict[str, Any],
        cookie: str,
    ) -> dict[str, Any]:
        video_id = str(item["video_id"])
        title = str(item.get("title") or video_id)
        bvid = item.get("bvid")
        aid = item.get("aid")

        try:
            view = self.favorites_service.get_video_view(cookie, bvid=bvid, aid=aid)
        except Exception as exc:
            self.repository.upsert_import_run_item(
                run_id,
                favorite_folder_id=str(favorite_folder["favorite_folder_id"]),
                video_id=video_id,
                bvid=bvid,
                title=title,
                status="failed",
                failure_reason=str(exc),
                retryable=True,
            )
            self.repository.upsert_run_step(
                run_id,
                f"import_item_{video_id}",
                f"import_item:{video_id}",
                "failed",
                input_summary="fetch video view",
                output_summary=str(exc),
            )
            self.repository.append_run_event(
                run_id,
                "import_item_processed",
                {
                    "video_id": video_id,
                    "status": "failed",
                    "title": title,
                    "failure_reason": str(exc),
                },
            )
            return {"status": "failed", "video_id": video_id, "title": title}

        pages = view.get("pages") or []
        if not pages:
            detail = "Bilibili view payload does not contain pages."
            self.repository.upsert_import_run_item(
                run_id,
                favorite_folder_id=str(favorite_folder["favorite_folder_id"]),
                video_id=video_id,
                bvid=bvid,
                title=title,
                status="failed",
                failure_reason=detail,
            )
            self.repository.upsert_run_step(
                run_id,
                f"import_item_{video_id}",
                f"import_item:{video_id}",
                "failed",
                input_summary="validate video pages",
                output_summary=detail,
            )
            self.repository.append_run_event(
                run_id,
                "import_item_processed",
                {
                    "video_id": video_id,
                    "status": "failed",
                    "title": title,
                    "failure_reason": detail,
                },
            )
            return {"status": "failed", "video_id": video_id, "title": title}

        subtitle_candidates = ((view.get("subtitle") or {}).get("list")) or []
        subtitle_entry = self.favorites_service.choose_subtitle_entry(subtitle_candidates)
        subtitle_url = subtitle_entry.get("subtitle_url") if subtitle_entry else None
        language = subtitle_entry.get("lan") if subtitle_entry else None
        page_records = [
            {
                "cid": page.get("cid"),
                "page_number": int(page.get("page") or 1),
                "title": str(page.get("part") or f"P{page.get('page') or 1}"),
                "duration": page.get("duration"),
            }
            for page in pages
        ]

        if subtitle_url:
            try:
                subtitle_payload = self.favorites_service.fetch_subtitle_body(
                    str(subtitle_url),
                    cookie=cookie,
                )
                subtitle_blocks = self.favorites_service.subtitle_payload_to_blocks(subtitle_payload)
            except Exception as exc:
                return self._build_asr_fallback_item(
                    run_id=run_id,
                    favorite_folder=favorite_folder,
                    item=item,
                    view=view,
                    pages=page_records,
                    cookie=cookie,
                    failure_reason=f"subtitle_fetch_failed: {exc}",
                    retryable=True,
                )
            if subtitle_blocks:
                knowledge_video = {
                    "video_id": video_id,
                    "bvid": str(view.get("bvid") or bvid or ""),
                    "title": str(view.get("title") or title),
                    "favorite_folder_ids": [str(favorite_folder["favorite_folder_id"])],
                    "pages": [
                        {
                            "page_id": f"{video_id}:p{page['page_number']}",
                            "page_number": page["page_number"],
                            "title": page["title"],
                            "text_blocks": (
                                [
                                    {
                                        "text": block["text"],
                                        "source_type": "subtitle",
                                        "source_language": language,
                                        "start_ms": block["start_ms"],
                                        "end_ms": block["end_ms"],
                                    }
                                    for block in subtitle_blocks
                                ]
                                if page["page_number"] == page_records[0]["page_number"]
                                else []
                            ),
                        }
                        for page in page_records
                    ],
                }
                manifest = {
                    "favorite_folder_id": str(favorite_folder["favorite_folder_id"]),
                    "video_id": video_id,
                    "bvid": str(view.get("bvid") or bvid or ""),
                    "title": str(view.get("title") or title),
                    "source_type": "subtitle",
                    "language": language,
                    "page_count": len(page_records),
                    "pages": page_records,
                }
                return {
                    "status": "ready_for_index",
                    "video_id": video_id,
                    "bvid": str(view.get("bvid") or bvid or ""),
                    "title": str(view.get("title") or title),
                    "knowledge_video": knowledge_video,
                    "manifest": manifest,
                }

        return self._build_asr_fallback_item(
            run_id=run_id,
            favorite_folder=favorite_folder,
            item=item,
            view=view,
            pages=page_records,
            cookie=cookie,
            failure_reason="subtitle_missing",
            retryable=False,
        )

    def _build_asr_fallback_item(
        self,
        *,
        run_id: str,
        favorite_folder: dict[str, Any],
        item: dict[str, Any],
        view: dict[str, Any],
        pages: list[dict[str, Any]],
        cookie: str,
        failure_reason: str,
        retryable: bool,
    ) -> dict[str, Any]:
        video_id = str(item["video_id"])
        bvid = str(view.get("bvid") or item.get("bvid") or "")
        title = str(view.get("title") or item.get("title") or video_id)
        asr_pages: list[dict[str, Any]] = []
        page_failures = 0
        for page in pages:
            media_url = None
            retryable_page = retryable
            page_failure_reason = failure_reason
            cid = page.get("cid")
            if bvid and cid:
                try:
                    playurl = self.favorites_service.get_playurl(cookie, bvid=bvid, cid=int(cid))
                except Exception as exc:
                    page_failure_reason = f"playurl_fetch_failed: {exc}"
                    retryable_page = True
                    page_failures += 1
                else:
                    media_url = self._select_media_url(playurl)
                    if media_url is None:
                        page_failure_reason = "playurl_fetch_failed: no playable media url"
                        retryable_page = True
                        page_failures += 1

            asr_pages.append(
                {
                    "page_id": f"{video_id}:p{page['page_number']}",
                    "cid": cid,
                    "page_number": page["page_number"],
                    "title": page["title"],
                    "duration": page.get("duration"),
                    "needs_asr": True,
                    "media_url": media_url,
                    "source_url": (
                        f"https://www.bilibili.com/video/{bvid}?p={page['page_number']}"
                        if bvid
                        else None
                    ),
                    "failure_reason": page_failure_reason,
                    "retryable": retryable_page,
                }
            )

        status = "needs_asr" if any(page["media_url"] for page in asr_pages) else "failed"
        manifest = {
            "favorite_folder_id": str(favorite_folder["favorite_folder_id"]),
            "video_id": video_id,
            "bvid": bvid or None,
            "title": title,
            "source_type": "none",
            "page_count": len(pages),
            "pages": pages,
            "failure_reason": failure_reason,
        }
        asr_job = {
            "favorite_folder_id": str(favorite_folder["favorite_folder_id"]),
            "video_id": video_id,
            "bvid": bvid or None,
            "title": title,
            "needs_asr": status == "needs_asr",
            "pages": asr_pages,
        }
        self.repository.upsert_import_run_item(
            run_id,
            favorite_folder_id=str(favorite_folder["favorite_folder_id"]),
            video_id=video_id,
            bvid=bvid or None,
            title=title,
            status=status,
            needs_asr=status == "needs_asr",
            failure_reason=None if status == "needs_asr" else failure_reason,
            retryable=retryable or page_failures > 0,
            manifest=manifest,
            asr_job=asr_job,
        )
        self.repository.upsert_run_step(
            run_id,
            f"import_item_{video_id}",
            f"import_item:{video_id}",
            "completed" if status == "needs_asr" else "failed",
            input_summary="subtitle/asr fallback pipeline",
            output_summary="prepared ASR fallback job" if status == "needs_asr" else failure_reason,
        )
        self.repository.append_run_event(
            run_id,
            "import_item_processed",
            {
                "video_id": video_id,
                "status": status,
                "title": title,
                "needs_asr": status == "needs_asr",
            },
        )
        return {"status": status, "video_id": video_id, "title": title}

    def _select_media_url(self, payload: dict[str, Any]) -> str | None:
        dash = payload.get("dash") or {}
        audio = dash.get("audio") or []
        if audio and isinstance(audio[0], dict):
            return audio[0].get("baseUrl") or audio[0].get("base_url")
        durl = payload.get("durl") or []
        if durl and isinstance(durl[0], dict):
            return durl[0].get("url")
        return None

    def _build_completion_reply(self, stats: dict[str, int]) -> str:
        return (
            "Import finished. "
            f"indexed={stats['indexed']}, "
            f"needs_asr={stats['needs_asr']}, "
            f"skipped_duplicate={stats['skipped_duplicate']}, "
            f"failed={stats['failed']}."
        )

    def _mark_execution_plan_status(
        self,
        execution_plan: dict[str, Any],
        status: str,
    ) -> dict[str, Any]:
        plan = dict(execution_plan)
        steps = []
        for step in plan.get("steps", []):
            updated = dict(step)
            updated["status"] = status
            steps.append(updated)
        plan["steps"] = steps
        return plan
