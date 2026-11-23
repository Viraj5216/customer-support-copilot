from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


REPO_ROOT = Path(__file__).resolve().parents[1]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(REPO_ROOT / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    gemini_api_key: SecretStr | None = None
    gemini_model: str = "gemini-3.5-flash-lite"
    gemini_fallback_model: str | None = None

    llm_primary_timeout_seconds: float = Field(default=10, gt=0)
    llm_timeout_seconds: float = Field(default=30, gt=0)
    llm_stream_timeout_seconds: float = Field(default=45, gt=0)
    llm_fallback_timeout_seconds: float = Field(default=30, gt=0)
    llm_fallback_stream_timeout_seconds: float = Field(default=45, gt=0)
    llm_max_attempts_per_model: int = Field(default=2, ge=1)
    llm_total_deadline_seconds: float = Field(default=75, gt=0)
    llm_model_failure_threshold: int = Field(default=2, ge=1)
    llm_model_cooldown_seconds: float = Field(default=60, gt=0)
    llm_retry_initial_backoff_seconds: float = Field(default=0.25, ge=0)
    llm_retry_max_backoff_seconds: float = Field(default=2, ge=0)

    retrieval_max_distance: float | None = Field(default=1.35, gt=0)
    retrieval_high_confidence_distance: float = Field(default=0.934, gt=0)

    rag_section_aware_embeddings: bool = False
    rag_hybrid_enabled: bool = False
    rag_reranker_enabled: bool = False
    rag_reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    rag_reranker_min_score: float = -10.73

    langfuse_public_key: SecretStr | None = None
    langfuse_secret_key: SecretStr | None = None
    langfuse_user_id_salt: SecretStr | None = None
    langfuse_base_url: str = "https://us.cloud.langfuse.com"

    auth_session_idle_timeout_seconds: int = Field(default=1800, ge=60)

    order_date_order: Literal["DMY", "MDY"] = "DMY"
    app_timezone: str = "Asia/Kolkata"

    orders_db_rel_path: Path = Path("data/orders.db")
    chroma_db_rel_path: Path = Path("chroma")
    policies_file_rel_path: Path = Path("data/company_knowledge_base.txt")

    chroma_collection_name: str = "company_policies"
    embedding_model_name: str = "all-MiniLM-L6-v2"

    @property
    def orders_db_path(self) -> Path:
        return (REPO_ROOT / self.orders_db_rel_path).resolve()

    @property
    def chroma_db_path(self) -> Path:
        return (REPO_ROOT / self.chroma_db_rel_path).resolve()

    @property
    def policies_file_path(self) -> Path:
        return (REPO_ROOT / self.policies_file_rel_path).resolve()


@lru_cache
def get_settings() -> Settings:
    settings = Settings()

    import logging

    legacy_paths = (
        ("ORDERS_DB_REL_PATH", settings.orders_db_path, (Path("data/orders.db"),)),
        ("CHROMA_DB_REL_PATH", settings.chroma_db_path, (Path("chroma"),)),
        (
            "POLICIES_FILE_REL_PATH",
            settings.policies_file_path,
            (Path("data/company_knowledge_base.txt"),),
        ),
    )

    for setting_name, configured_path, fallback_rel_paths in legacy_paths:
        if configured_path.exists():
            continue

        fallback = next(
            (
                REPO_ROOT / rel_path
                for rel_path in fallback_rel_paths
                if (REPO_ROOT / rel_path).exists()
            ),
            None,
        )

        if fallback is not None and fallback != configured_path:
            logging.getLogger(__name__).warning(
                "Configured %s path does not exist; default path '%s' exists.",
                setting_name,
                fallback.relative_to(REPO_ROOT),
            )

    return settings
