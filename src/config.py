from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Centralized runtime configuration for the Agentic RAG system."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    project_root: Path = Field(
        default_factory=lambda: Path(__file__).resolve().parents[1]
    )
    data_dir: Path = Field(
        default_factory=lambda: Path(__file__).resolve().parents[1] / "data"
    )
    chroma_dir: Path = Field(
        default_factory=lambda: Path(__file__).resolve().parents[1] / "chroma_db"
    )
    evaluation_dir: Path = Field(
        default_factory=lambda: Path(__file__).resolve().parents[1] / "evaluation_logs"
    )
    upload_dir: Path = Field(
        default_factory=lambda: Path(__file__).resolve().parents[1] / "data" / "uploads"
    )

    collection_name: str = "agentic_rag_knowledge_base"
    chunk_size: int = 500
    chunk_overlap: int = 50
    top_k: int = 1
    retrieval_score_threshold: float = 1.3
    search_loop_limit: int = 3

    embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"

    openai_api_key: SecretStr | None = Field(
        default=None, validation_alias="OPENAI_API_KEY"
    )
    openai_model_name: str = "gpt-4o-mini"
    openai_temperature: float = 0.0

    ollama_base_url: str = "http://localhost:11434"
    ollama_model_name: str = "llama3.1"

    streamlit_page_title: str = "Agentic RAG Knowledge System"
    streamlit_page_icon: str = "🧠"

    def ensure_directories(self) -> None:
        """Create all persistence directories before runtime begins."""

        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.chroma_dir.mkdir(parents=True, exist_ok=True)
        self.evaluation_dir.mkdir(parents=True, exist_ok=True)
        self.upload_dir.mkdir(parents=True, exist_ok=True)

    @property
    def has_openai_key(self) -> bool:
        return self.openai_api_key is not None and bool(
            self.openai_api_key.get_secret_value().strip()
        )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    settings = Settings()
    settings.ensure_directories()
    return settings


settings = get_settings()
