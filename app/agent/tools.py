from typing import Any

from langchain_core.tools import BaseTool, tool
from pydantic import BaseModel, Field


class BilibiliImportToolInput(BaseModel):
    request_message: str | None = None
    target: str | None = None
    run_id: str | None = None
    session_id: str | None = None
    user_id: str | None = None
    cookie: str | None = Field(default=None, repr=False)
    favorite_folder_id: str | None = None
    selected_video_ids: list[str] = Field(default_factory=list)


def build_bilibili_import_tool(import_pipeline) -> BaseTool:
    @tool("bilibili_import", args_schema=BilibiliImportToolInput)
    def bilibili_import(
        request_message: str | None = None,
        target: str | None = None,
        run_id: str | None = None,
        session_id: str | None = None,
        user_id: str | None = None,
        cookie: str | None = None,
        favorite_folder_id: str | None = None,
        selected_video_ids: list[str] | None = None,
    ) -> str:
        """Execute Bilibili import requests from either the manual UI flow or the agent flow."""
        if (
            run_id
            and session_id
            and cookie
            and favorite_folder_id
            and selected_video_ids
        ):
            return import_pipeline.execute_selected_videos(
                run_id=run_id,
                session_id=session_id,
                user_id=user_id,
                cookie=cookie,
                favorite_folder_id=favorite_folder_id,
                selected_video_ids=list(selected_video_ids),
            )
        return import_pipeline.handle_agent_import_request(
            request_message=request_message,
            target=target,
        )

    return bilibili_import


class BilibiliRetryToolInput(BaseModel):
    request_message: str = Field(min_length=1)
    target: str = Field(min_length=1)


class KnowledgeRetrievalToolInput(BaseModel):
    message: str = Field(min_length=1)
    route: str | None = None
    recent_context: dict[str, Any] = Field(default_factory=dict)
    top_k: int = Field(default=5, ge=1, le=10)


@tool(args_schema=BilibiliRetryToolInput)
def bilibili_retry(request_message: str, target: str) -> str:
    """Run the approved Bilibili retry placeholder tool for failed ingestion items."""
    return (
        "Execution was approved. The placeholder bilibili retry tool ran for "
        f"{target} based on request: {request_message}. Real retry tools are not "
        "wired in yet."
    )


def build_knowledge_retrieval_tool(retrieval_service) -> BaseTool:
    @tool("knowledge_retrieval", args_schema=KnowledgeRetrievalToolInput, response_format="content_and_artifact")
    def knowledge_retrieval(
        message: str,
        route: str | None = None,
        recent_context: dict[str, Any] | None = None,
        top_k: int = 5,
    ) -> tuple[str, dict[str, Any]]:
        """Retrieve grounded Bilibili knowledge for a chat question."""
        result = retrieval_service.retrieve_for_question(
            message=message,
            route=route,
            recent_context=recent_context or {},
            top_k=top_k,
        )
        artifact = {
            "query": result["query"],
            "route": result["route"],
            "resolved_scope": result["resolved_scope"],
            "total_hits": result["total_hits"],
            "hits": result["hits"],
            "top_sources": result["top_sources"],
        }
        return str(result["serialized_context"]), artifact

    return knowledge_retrieval


def build_tool_registry(import_tool: BaseTool) -> dict[tuple[str, str], BaseTool]:
    return {
        ("bilibili_import", "execute_import"): import_tool,
        ("bilibili_retry", "prepare_retry_plan"): bilibili_retry,
    }
