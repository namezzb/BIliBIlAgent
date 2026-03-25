from langchain_core.tools import BaseTool, tool
from pydantic import BaseModel, Field


class BilibiliImportToolInput(BaseModel):
    request_message: str = Field(min_length=1)
    target: str = Field(min_length=1)


@tool(args_schema=BilibiliImportToolInput)
def bilibili_import(request_message: str, target: str) -> str:
    """Run the approved Bilibili import placeholder tool for the requested scope."""
    return (
        "Execution was approved. The placeholder bilibili import tool ran for "
        f"{target} based on request: {request_message}. Real Bilibili import "
        "tools are not wired in yet."
    )


class BilibiliRetryToolInput(BaseModel):
    request_message: str = Field(min_length=1)
    target: str = Field(min_length=1)


@tool(args_schema=BilibiliRetryToolInput)
def bilibili_retry(request_message: str, target: str) -> str:
    """Run the approved Bilibili retry placeholder tool for failed ingestion items."""
    return (
        "Execution was approved. The placeholder bilibili retry tool ran for "
        f"{target} based on request: {request_message}. Real retry tools are not "
        "wired in yet."
    )


TOOL_REGISTRY: dict[tuple[str, str], BaseTool] = {
    ("bilibili_import", "prepare_import_plan"): bilibili_import,
    ("bilibili_retry", "prepare_retry_plan"): bilibili_retry,
}
