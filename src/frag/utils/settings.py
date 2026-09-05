"""Typed configuration, loaded once from the environment (and .env).

Every tunable the graph reads lives here, so behaviour is inspectable in one place.
Import `settings` and read fields; tests override with Settings(...) explicitly.
"""

from __future__ import annotations

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # LLM (OpenRouter, OpenAI-compatible)
    openrouter_api_key: str | None = Field(default=None, alias="OPENROUTER_API_KEY")
    openrouter_base_url: str = Field(
        default="https://openrouter.ai/api/v1", alias="OPENROUTER_BASE_URL"
    )
    actor_model: str = Field(default="meta-llama/llama-3.3-70b-instruct", alias="ACTOR_MODEL")
    critic_model: str = Field(default="meta-llama/llama-3.3-70b-instruct", alias="CRITIC_MODEL")
    grader_model: str = Field(default="meta-llama/llama-3.3-70b-instruct", alias="GRADER_MODEL")
    agent_model: str = Field(default="meta-llama/llama-3.3-70b-instruct", alias="AGENT_MODEL")
    llm_temperature: float = Field(default=0.0, alias="LLM_TEMPERATURE")
    llm_timeout_s: int = Field(default=120, alias="LLM_TIMEOUT")
    llm_max_retries: int = Field(default=4, alias="LLM_MAX_RETRIES")

    # retrieval
    embedding_model: str = Field(default="BAAI/bge-base-en-v1.5", alias="EMBEDDING_MODEL")
    qdrant_url: str = Field(default="http://localhost:6333", alias="QDRANT_URL")
    qdrant_collection: str = Field(default="frag", alias="QDRANT_COLLECTION")
    top_k: int = Field(default=8, alias="TOP_K")
    rerank: bool = Field(default=False, alias="RERANK")
    reranker_model: str = Field(
        default="cross-encoder/ms-marco-MiniLM-L-6-v2", alias="RERANKER_MODEL"
    )

    # corrective retrieval (CRAG)
    corrective: bool = Field(default=False, alias="CORRECTIVE")
    corrective_max_rewrites: int = Field(default=2, alias="CORRECTIVE_MAX_REWRITES")
    corrective_min_relevant: int = Field(default=1, alias="CORRECTIVE_MIN_RELEVANT")

    # gates
    critic: bool = Field(default=False, alias="CRITIC")
    critic_min_score: float = Field(default=0.8, alias="CRITIC_MIN_SCORE")
    router_llm: bool = Field(default=False, alias="ROUTER_LLM")
    agent_max_turns: int = Field(default=6, alias="AGENT_MAX_TURNS")
    agent_max_tool_calls: int = Field(default=8, alias="AGENT_MAX_TOOL_CALLS")

    # observability (self-hosted Langfuse)
    langfuse_public_key: str | None = Field(default=None, alias="LANGFUSE_PUBLIC_KEY")
    langfuse_secret_key: str | None = Field(default=None, alias="LANGFUSE_SECRET_KEY")
    langfuse_host: str = Field(default="http://localhost:3000", alias="LANGFUSE_HOST")

    log_level: str = Field(default="INFO", alias="LOG_LEVEL")
    log_json: bool = Field(default=True, alias="LOG_JSON")

    @property
    def tracing_enabled(self) -> bool:
        return bool(self.langfuse_public_key and self.langfuse_secret_key)


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
