from contextlib import asynccontextmanager

from fastapi import FastAPI
from langsmith.middleware import TracingMiddleware

from app.agent.tools import (
    build_bilibili_import_tool,
    build_knowledge_retrieval_tool,
    build_tool_registry,
)
from app.agent.service import AgentOrchestrator
from app.api.routes.chat import router as chat_router
from app.core.config import Settings, get_settings
from app.db.repository import SQLiteRepository
from app.services.bilibili_favorites import BilibiliFavoriteFolderService
from app.services.bilibili_import import BilibiliImportPipeline
from app.services.knowledge_index import ChromaVectorIndex, KnowledgeIndexService
from app.services.knowledge_qa import KnowledgeGroundedQAService
from app.services.knowledge_retrieval import KnowledgeRetrievalService
from app.services.llm import OpenAICompatibleLLM
from app.services.runtime_audit import LangSmithRuntimeAudit
from app.services.session_memory import SessionMemoryManager
from app.services.user_memory import UserMemoryManager


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings: Settings = app.state.settings
    repository = SQLiteRepository(settings.app_db_path)
    repository.initialize()
    runtime_audit = app.state.runtime_audit or LangSmithRuntimeAudit(
        enabled=settings.langsmith_tracing,
        api_key=settings.langsmith_api_key,
        project_name=settings.langsmith_project,
        endpoint=settings.langsmith_endpoint,
        workspace_id=settings.langsmith_workspace_id,
        web_url=settings.langsmith_web_url,
        app_name=settings.app_name,
        environment=settings.environment,
    )
    llm = OpenAICompatibleLLM(
        api_key=settings.llm_api_key,
        base_url=settings.llm_base_url,
        model=settings.llm_model,
        summary_model=settings.summary_model,
        embedding_model=settings.embedding_model,
        system_prompt=settings.llm_system_prompt,
        runtime_audit=runtime_audit,
    )
    user_memory = UserMemoryManager(repository)
    bilibili_favorites = BilibiliFavoriteFolderService()
    knowledge_index = KnowledgeIndexService(
        repository=repository,
        vector_index=ChromaVectorIndex(
            persist_dir=settings.chroma_persist_dir,
            collection_name=settings.chroma_collection_name,
        ),
        embed_texts=llm.embed_texts,
        embedding_model=settings.embedding_model,
        embedding_version=settings.knowledge_embedding_version,
        chunk_size=settings.knowledge_chunk_size,
        chunk_overlap=settings.knowledge_chunk_overlap,
    )
    bilibili_import_pipeline = BilibiliImportPipeline(
        repository=repository,
        favorites_service=bilibili_favorites,
        knowledge_index=knowledge_index,
        runtime_audit=runtime_audit,
    )
    knowledge_retrieval_service = KnowledgeRetrievalService(repository, knowledge_index)
    knowledge_retrieval_tool = build_knowledge_retrieval_tool(knowledge_retrieval_service)
    knowledge_qa = KnowledgeGroundedQAService(llm)
    bilibili_import_tool = build_bilibili_import_tool(bilibili_import_pipeline)
    orchestrator = AgentOrchestrator(
        repository=repository,
        llm=llm,
        checkpoint_db_path=settings.checkpoint_db_path,
        user_memory=user_memory,
        runtime_audit=runtime_audit,
        tool_registry=build_tool_registry(bilibili_import_tool),
        knowledge_retrieval_service=knowledge_retrieval_service,
        knowledge_retrieval_tool=knowledge_retrieval_tool,
        knowledge_qa=knowledge_qa,
    )
    session_memory = SessionMemoryManager(repository, llm)

    app.state.repository = repository
    app.state.orchestrator = orchestrator
    app.state.session_memory = session_memory
    app.state.user_memory = user_memory
    app.state.bilibili_favorite_folder_service = bilibili_favorites
    app.state.bilibili_import_pipeline = bilibili_import_pipeline
    app.state.bilibili_import_tool = bilibili_import_tool
    app.state.knowledge_index = knowledge_index
    app.state.knowledge_retrieval = knowledge_retrieval_service
    app.state.knowledge_retrieval_tool = knowledge_retrieval_tool
    app.state.knowledge_qa = knowledge_qa
    app.state.runtime_audit = runtime_audit

    try:
        yield
    finally:
        orchestrator.close()
        runtime_audit.close()


def create_app(
    settings: Settings | None = None,
    *,
    runtime_audit: LangSmithRuntimeAudit | None = None,
) -> FastAPI:
    app_settings = settings or get_settings()
    app_settings.ensure_directories()

    app = FastAPI(
        title=app_settings.app_name,
        version=app_settings.app_version,
        description="Backend service for Bilibili favorite-folder ingestion and QA.",
        lifespan=lifespan,
    )
    app.state.settings = app_settings
    app.state.runtime_audit = runtime_audit
    app.add_middleware(TracingMiddleware)
    app.include_router(chat_router)

    @app.get("/")
    def read_root() -> dict[str, str]:
        return {
            "name": "BIliBIlAgent",
            "status": "ready",
            "message": "Use /health to verify service status.",
        }

    @app.get("/health")
    def health_check() -> dict[str, str]:
        return {
            "status": "ok",
            "service": "bilibilagent-backend",
        }

    return app


app = create_app()
