"""Settings from environment variables (prefix `RAG_`). Defaults run fully offline."""

from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="RAG_", env_file=".env", extra="ignore")

    # Storage
    store: Literal["memory", "postgres"] = "memory"
    database_url: str = "postgresql+asyncpg://rag:rag@localhost:55433/rag"
    redis_url: str | None = None  # None → in-process cache and inline ingestion
    upload_dir: str = "data/uploads"

    # Providers ("fake" = deterministic offline implementations)
    embedding_provider: Literal["hashing", "openai", "gemini", "local"] = "hashing"
    embedding_model: str = "hashing-v1"
    embedding_dim: int = 384
    llm_provider: Literal["fake", "openai", "gemini"] = "fake"
    llm_model: str = "fake-extractive-v1"
    reranker: Literal["none", "lexical", "cross-encoder"] = "lexical"
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    openai_api_key: SecretStr | None = None
    gemini_api_key: SecretStr | None = None

    # Retrieval
    chunk_max_tokens: int = 256
    chunk_overlap_tokens: int = 32
    top_k: int = 5
    candidate_k: int = 20
    rrf_k: int = 60
    refusal_threshold: float = Field(0.2, ge=0.0, le=1.0)
    context_token_budget: int = 1500

    # Auth
    jwt_secret: SecretStr = SecretStr("dev-only-change-me-dev-only-change-me")
    jwt_access_ttl_minutes: int = 30
    jwt_refresh_ttl_days: int = 7

    # Cost table file
    models_config: str = "config/models.yaml"


@lru_cache
def get_settings() -> Settings:
    return Settings()
