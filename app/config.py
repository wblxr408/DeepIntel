"""
Configuration management for Agentic Deep Research System.

All configuration is loaded from environment variables with type validation.
No hardcoded values. Supports multi-LLM provider fallback.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

ProviderName = Literal["qwen", "deepseek", "openai", "ollama", "anthropic"]


class LLMSettings(BaseSettings):
    """LLM provider configuration with fallback strategy."""

    provider: ProviderName = Field(
        default="qwen",
        description="Primary LLM provider"
    )
    model: str = Field(
        default="qwen-plus",
        description="Model name for primary provider"
    )
    api_key: str = Field(
        default="",
        description="API key for primary provider"
    )
    api_base: str | None = Field(
        default=None,
        description="Custom API base URL (for proxies/ollama)"
    )
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    max_tokens: int = Field(default=8192, ge=256)

    # Fallback settings
    fallback_provider: ProviderName | None = Field(default=None)
    fallback_model: str | None = Field(default=None)
    fallback_api_key: str | None = Field(default=None)

    model_config = SettingsConfigDict(env_prefix="LLM_", env_file=".env", env_file_encoding="utf-8", extra="ignore")


class DatabaseSettings(BaseSettings):
    """PostgreSQL + pgvector configuration."""

    url: str = Field(
        default="postgresql://postgres:postgres@localhost:5432/deepintel",
        description="PostgreSQL connection URL"
    )
    pool_size: int = Field(default=10, ge=1, le=100)
    max_overflow: int = Field(default=20, ge=0)

    model_config = SettingsConfigDict(env_prefix="DATABASE_", env_file=".env", env_file_encoding="utf-8", extra="ignore")


class RedisSettings(BaseSettings):
    """Redis cache configuration."""

    url: str = Field(default="redis://localhost:6379/0")
    session_ttl: int = Field(default=604800, description="7 days in seconds")  # 7 days
    cache_ttl: int = Field(default=259200, description="3 days in seconds")     # 3 days

    model_config = SettingsConfigDict(env_prefix="REDIS_", env_file=".env", env_file_encoding="utf-8", extra="ignore")


class RAGSettings(BaseSettings):
    """RAG system configuration."""

    embed_model: str = Field(default="BAAI/bge-zh-qwen2-int8")
    embed_dimension: int = Field(default=1024)
    rerank_model: str = Field(default="BAAI/bge-reranker-v2-m3")
    rerank_device: Literal["cuda", "cpu"] = Field(default="cuda")
    retrieval_top_k: int = Field(default=20, description="Initial retrieval count")
    rerank_top_n: int = Field(default=10, description="Final count after reranking")
    rrf_k: int = Field(default=60, description="RRF fusion parameter")

    model_config = SettingsConfigDict(env_prefix="RAG_", env_file=".env", env_file_encoding="utf-8", extra="ignore")


class BrowserSettings(BaseSettings):
    """Playwright browser pool configuration."""

    headless: bool = Field(default=True)
    pool_size: int = Field(default=3, ge=1, le=10)
    navigation_timeout: int = Field(default=30000, description="ms")
    user_agent: str = Field(
        default="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                "(KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36"
    )
    accept_lang: str = Field(default="zh-CN,zh;q=0.9,en;q=0.8")

    # Extraction levels
    snippet_max_chars: int = Field(default=500)
    skim_max_chars: int = Field(default=2000)
    deep_max_chars: int = Field(default=5000)

    model_config = SettingsConfigDict(env_prefix="PLAYWRIGHT_", env_file=".env", env_file_encoding="utf-8", extra="ignore")


class APISettings(BaseSettings):
    """FastAPI server configuration."""

    host: str = Field(default="0.0.0.0")
    port: int = Field(default=8000, ge=1, le=65535)
    reload: bool = Field(default=False)
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = Field(default="INFO")
    cors_origins: list[str] = Field(default=["http://localhost:5173", "http://localhost:3000"])
    trusted_hosts: list[str] = Field(default=["localhost", "127.0.0.1"])

    model_config = SettingsConfigDict(env_prefix="API_", env_file=".env", env_file_encoding="utf-8", extra="ignore")


class SecuritySettings(BaseSettings):
    """Authentication, encryption and request-boundary configuration."""

    auth_enabled: bool = Field(default=True)
    environment: Literal["development", "production"] = Field(default="development")
    encryption_key: str = Field(default="")
    session_ttl_seconds: int = Field(default=604800, ge=300, le=2592000)
    secure_cookies: bool = Field(default=True)
    cookie_name: str = Field(default="deepintel_session")
    login_rate_limit: int = Field(default=10, ge=1, le=100)
    upload_rate_limit: int = Field(default=30, ge=1, le=500)
    upload_max_bytes: int = Field(default=25 * 1024 * 1024, ge=1024)
    upload_max_pages: int = Field(default=500, ge=1, le=10000)

    model_config = SettingsConfigDict(env_prefix="SECURITY_", env_file=".env", env_file_encoding="utf-8", extra="ignore")

    def validate_production_requirements(self, cors_origins: list[str]) -> None:
        """Fail closed for deployment settings that protect persistent secrets."""
        if self.environment != "production":
            return
        if self.auth_enabled and (not self.encryption_key or self.encryption_key.startswith("replace-with-")):
            raise ValueError("SECURITY_ENCRYPTION_KEY must be set to a non-example value in production")
        if self.auth_enabled and not self.secure_cookies:
            raise ValueError("SECURITY_SECURE_COOKIES must be true in production")
        if "*" in cors_origins:
            raise ValueError("API_CORS_ORIGINS cannot include '*' in production")


class ResilienceSettings(BaseSettings):
    """Network and optional-capability failure handling."""

    capability_retry_seconds: int = Field(default=30, ge=1, le=3600)
    llm_connect_timeout_seconds: float = Field(default=10.0, gt=0)
    llm_read_timeout_seconds: float = Field(default=90.0, gt=0)
    llm_write_timeout_seconds: float = Field(default=30.0, gt=0)
    llm_pool_timeout_seconds: float = Field(default=10.0, gt=0)
    llm_max_retries: int = Field(default=1, ge=0, le=5)

    model_config = SettingsConfigDict(env_prefix="RESILIENCE_", env_file=".env", env_file_encoding="utf-8", extra="ignore")


class HarnessSettings(BaseSettings):
    """Harness supervisor state configuration."""

    state_root: str = Field(default="./data/harness")
    worker_id: str = Field(default="worker-1")

    model_config = SettingsConfigDict(env_prefix="HARNESS_", env_file=".env", env_file_encoding="utf-8", extra="ignore")


class SkillSettings(BaseSettings):
    """Skill matching/runtime settings."""

    match_top_k: int = Field(default=5, ge=1, le=20)
    coarse_candidate_floor: int = Field(default=12, ge=1, le=200)
    broad_fallback_limit: int = Field(default=32, ge=1, le=500)
    content_l1_cache_size: int = Field(default=128, ge=8, le=2048)
    match_cache_ttl: int = Field(default=900, ge=30, le=86400)
    content_cache_ttl: int = Field(default=1800, ge=30, le=86400)

    model_config = SettingsConfigDict(env_prefix="SKILLS_", env_file=".env", env_file_encoding="utf-8", extra="ignore")


class Settings(BaseSettings):
    """Root configuration aggregating all sub-settings."""

    # Sub-configurations
    llm: LLMSettings = Field(default_factory=LLMSettings)
    database: DatabaseSettings = Field(default_factory=DatabaseSettings)
    redis: RedisSettings = Field(default_factory=RedisSettings)
    rag: RAGSettings = Field(default_factory=RAGSettings)
    browser: BrowserSettings = Field(default_factory=BrowserSettings)
    api: APISettings = Field(default_factory=APISettings)
    security: SecuritySettings = Field(default_factory=SecuritySettings)
    resilience: ResilienceSettings = Field(default_factory=ResilienceSettings)
    harness: HarnessSettings = Field(default_factory=HarnessSettings)
    skills: SkillSettings = Field(default_factory=SkillSettings)

    # Global flags
    sse_enabled: bool = Field(default=True)
    observability_enabled: bool = Field(default=True)

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    def validate_startup_security(self) -> None:
        self.security.validate_production_requirements(self.api.cors_origins)


@lru_cache
def get_settings() -> Settings:
    """Singleton settings instance. Cached for performance."""
    return Settings()
