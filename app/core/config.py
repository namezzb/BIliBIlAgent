from functools import lru_cache
from pathlib import Path

from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "BIliBIlAgent API"
    app_version: str = "0.1.0"
    environment: str = "development"
    data_dir: Path = Path("data")
    app_db_path: Path = Path("data/app.db")
    checkpoint_db_path: Path = Path("data/checkpoints.db")
    chroma_persist_dir: Path = Path("data/chroma")
    chroma_collection_name: str = "knowledge_chunks"
    knowledge_embedding_version: str = "v1"
    knowledge_chunk_size: int = 1000
    knowledge_chunk_overlap: int = 200
    llm_api_key: str | None = Field(
        default=None,
        validation_alias=AliasChoices("LLM_API_KEY", "OPENROUTER_API_KEY"),
    )
    llm_base_url: str = Field(
        default="https://openrouter.ai/api/v1",
        validation_alias=AliasChoices("LLM_BASE_URL", "OPENROUTER_BASE_URL"),
    )
    llm_model: str = Field(
        default="qwen/qwen3-235b-a22b-2507",
        validation_alias=AliasChoices("LLM_MODEL", "OPENROUTER_MODEL"),
    )
    summary_model: str | None = Field(
        default=None,
        validation_alias=AliasChoices("SUMMARY_MODEL", "OPENROUTER_SUMMARY_MODEL"),
    )
    embedding_model: str = Field(
        default="openai/text-embedding-3-small",
        validation_alias=AliasChoices("EMBEDDING_MODEL", "OPENROUTER_EMBEDDING_MODEL"),
    )
    embedding_api_key: str | None = Field(
        default=None,
        validation_alias=AliasChoices("EMBEDDING_API_KEY"),
    )
    embedding_base_url: str | None = Field(
        default=None,
        validation_alias=AliasChoices("EMBEDDING_BASE_URL"),
    )
    llm_system_prompt: str = (
        "You are BIliBIlAgent. Answer directly when the user asks for general chat. "
        "Keep responses concise and helpful."
    )

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    def ensure_directories(self) -> None:
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.app_db_path.parent.mkdir(parents=True, exist_ok=True)
        self.checkpoint_db_path.parent.mkdir(parents=True, exist_ok=True)
        self.chroma_persist_dir.mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    settings = Settings()
    settings.ensure_directories()
    return settings
