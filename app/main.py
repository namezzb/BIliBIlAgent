from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.agent.service import AgentOrchestrator
from app.api.routes.chat import router as chat_router
from app.core.config import Settings, get_settings
from app.db.repository import SQLiteRepository
from app.services.llm import OpenAICompatibleLLM
from app.services.session_memory import SessionMemoryManager


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings: Settings = app.state.settings
    repository = SQLiteRepository(settings.app_db_path)
    repository.initialize()
    llm = OpenAICompatibleLLM(
        api_key=settings.llm_api_key,
        base_url=settings.llm_base_url,
        model=settings.llm_model,
        summary_model=settings.summary_model,
        embedding_model=settings.embedding_model,
        system_prompt=settings.llm_system_prompt,
    )
    orchestrator = AgentOrchestrator(
        repository=repository,
        llm=llm,
        checkpoint_db_path=settings.checkpoint_db_path,
    )
    session_memory = SessionMemoryManager(repository, llm)

    app.state.repository = repository
    app.state.orchestrator = orchestrator
    app.state.session_memory = session_memory

    try:
        yield
    finally:
        orchestrator.close()


def create_app(settings: Settings | None = None) -> FastAPI:
    app_settings = settings or get_settings()
    app_settings.ensure_directories()

    app = FastAPI(
        title=app_settings.app_name,
        version=app_settings.app_version,
        description="Backend service for Bilibili favorite-folder ingestion and QA.",
        lifespan=lifespan,
    )
    app.state.settings = app_settings
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
