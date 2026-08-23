"""
Database connection management and models.

PostgreSQL + pgvector for document storage and hybrid retrieval.
Redis for session cache and LLM response cache.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import asyncpg

from app.config import get_settings
from app.db.text_search import regconfig_sql_literal, resolve_text_search_config

if TYPE_CHECKING:
    import redis.asyncio as redis

logger = logging.getLogger(__name__)

# Global pool instances
_db_pool: Any | None = None
_redis_client: Any | None = None
_fts_config: str | None = None


async def init_db() -> asyncpg.Pool:
    """
    Initialize PostgreSQL connection pool and create tables if needed.
    """
    global _db_pool, _fts_config
    if _db_pool is not None:
        return _db_pool

    settings = get_settings()
    pool = await asyncpg.create_pool(
        settings.database.url,
        min_size=2,
        max_size=20,
    )

    # Create tables
    async with pool.acquire() as conn:
        _fts_config = await resolve_text_search_config(conn, log=logger)
        fts_config_sql = regconfig_sql_literal(_fts_config)
        await conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS documents (
                id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                source_id UUID,
                source_name TEXT,
                source_type VARCHAR(20) DEFAULT 'manual',
                chunk_index INTEGER DEFAULT 0,
                chunk_count INTEGER DEFAULT 1,
                content TEXT NOT NULL,
                metadata JSONB DEFAULT '{}',
                embedding VECTOR(1024),
                created_at TIMESTAMP DEFAULT NOW(),
                updated_at TIMESTAMP DEFAULT NOW()
            )
        """)
        await conn.execute("ALTER TABLE documents ADD COLUMN IF NOT EXISTS source_id UUID")
        await conn.execute("ALTER TABLE documents ADD COLUMN IF NOT EXISTS source_name TEXT")
        await conn.execute("ALTER TABLE documents ADD COLUMN IF NOT EXISTS source_type VARCHAR(20) DEFAULT 'manual'")
        await conn.execute("ALTER TABLE documents ADD COLUMN IF NOT EXISTS chunk_index INTEGER DEFAULT 0")
        await conn.execute("ALTER TABLE documents ADD COLUMN IF NOT EXISTS chunk_count INTEGER DEFAULT 1")
        await conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_documents_embedding
            ON documents USING ivfflat (embedding vector_cosine_ops)
        """)
        try:
            await conn.execute(f"""
                CREATE INDEX IF NOT EXISTS idx_documents_fts
                ON documents USING gin (to_tsvector({fts_config_sql}, content))
            """)
        except (OSError, RuntimeError, asyncpg.PostgresError) as exc:
            logger.warning(
                "Failed to create FTS index with config '%s', falling back to 'simple': %s",
                _fts_config,
                exc,
            )
            _fts_config = "simple"
            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_documents_fts
                ON documents USING gin (to_tsvector('simple', content))
            """)

        await conn.execute("""
            CREATE TABLE IF NOT EXISTS document_sources (
                id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                name TEXT NOT NULL,
                group_name TEXT NOT NULL,
                source_type VARCHAR(20) DEFAULT 'manual',
                file_name TEXT,
                file_ext VARCHAR(20),
                status VARCHAR(20) DEFAULT 'active',
                original_text TEXT,
                chunk_size INTEGER DEFAULT 400,
                chunk_overlap INTEGER DEFAULT 80,
                metadata JSONB DEFAULT '{}',
                created_at TIMESTAMP DEFAULT NOW(),
                updated_at TIMESTAMP DEFAULT NOW()
            )
        """)
        await conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_document_sources_group
            ON document_sources (group_name)
        """)

        await conn.execute("""
            CREATE TABLE IF NOT EXISTS research_sessions (
                id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                user_query TEXT NOT NULL,
                status VARCHAR(20) DEFAULT 'pending',
                research_plan JSONB DEFAULT '[]',
                skill_context JSONB DEFAULT '{}',
                guardrail_decision JSONB DEFAULT NULL,
                guardrail_trace JSONB DEFAULT '[]',
                evidence_status JSONB DEFAULT NULL,
                review_status JSONB DEFAULT NULL,
                prompt_profile VARCHAR(50),
                prompt_template TEXT,
                enabled_tools JSONB DEFAULT '[]',
                final_report TEXT,
                citations JSONB DEFAULT '[]',
                agent_trace JSONB DEFAULT '[]',
                created_at TIMESTAMP DEFAULT NOW(),
                updated_at TIMESTAMP DEFAULT NOW(),
                completed_at TIMESTAMP
            )
        """)
        await conn.execute("""
            ALTER TABLE research_sessions
            ADD COLUMN IF NOT EXISTS skill_context JSONB DEFAULT '{}'
        """)

        await conn.execute("""
            CREATE TABLE IF NOT EXISTS citations (
                id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                session_id UUID REFERENCES research_sessions(id) ON DELETE CASCADE,
                citation_id VARCHAR(20),
                source_url TEXT,
                source_title TEXT,
                source_type VARCHAR(20),
                extracted_evidence TEXT,
                relevance_score FLOAT,
                access_timestamp TIMESTAMP,
                created_at TIMESTAMP DEFAULT NOW()
            )
        """)
        await conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_citations_session
            ON citations (session_id)
        """)
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS tool_call_audit (
                call_id VARCHAR(64) PRIMARY KEY,
                session_id UUID REFERENCES research_sessions(id) ON DELETE CASCADE,
                node_id VARCHAR(64),
                agent_type VARCHAR(50) NOT NULL,
                tool_name VARCHAR(100) NOT NULL,
                args_json JSONB DEFAULT '{}',
                args_hash VARCHAR(64),
                status VARCHAR(32) NOT NULL,
                error_category VARCHAR(64),
                error_message TEXT,
                retry_count INTEGER DEFAULT 0,
                result_summary TEXT,
                result_hash VARCHAR(64),
                tokens_used INTEGER DEFAULT 0,
                cost_usd NUMERIC(10, 6) DEFAULT 0,
                decision_id VARCHAR(100),
                approved_by VARCHAR(100),
                server_fingerprint VARCHAR(255),
                safety_json JSONB DEFAULT '{}',
                usage_source VARCHAR(32) DEFAULT 'provider',
                estimated BOOLEAN DEFAULT FALSE,
                started_at TIMESTAMP,
                completed_at TIMESTAMP,
                created_at TIMESTAMP DEFAULT NOW()
            )
        """)
        await conn.execute("""
            ALTER TABLE tool_call_audit
            ADD COLUMN IF NOT EXISTS safety_json JSONB DEFAULT '{}'
        """)
        await conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_tool_call_audit_session
            ON tool_call_audit (session_id)
        """)
        await conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_tool_call_audit_tool
            ON tool_call_audit (tool_name)
        """)
        await conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_tool_call_audit_status
            ON tool_call_audit (status)
        """)
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS session_budget_state (
                session_id UUID PRIMARY KEY REFERENCES research_sessions(id) ON DELETE CASCADE,
                max_total_tokens INTEGER DEFAULT 0,
                max_cost_usd NUMERIC(10, 6) DEFAULT 0,
                max_tool_calls INTEGER DEFAULT 0,
                max_wall_clock_seconds INTEGER DEFAULT 0,
                max_retries_per_tool INTEGER DEFAULT 0,
                used_total_tokens INTEGER DEFAULT 0,
                used_cost_usd NUMERIC(10, 6) DEFAULT 0,
                used_tool_calls INTEGER DEFAULT 0,
                elapsed_wall_clock_seconds INTEGER DEFAULT 0,
                hard_stop_reason VARCHAR(64),
                estimated_usage_count INTEGER DEFAULT 0,
                updated_at TIMESTAMP DEFAULT NOW()
            )
        """)
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS approval_requests (
                approval_id VARCHAR(64) PRIMARY KEY,
                session_id UUID REFERENCES research_sessions(id) ON DELETE CASCADE,
                node_id VARCHAR(64),
                tool_name VARCHAR(100) NOT NULL,
                risk_level VARCHAR(20) DEFAULT 'high',
                reason TEXT,
                request_payload_json JSONB DEFAULT '{}',
                status VARCHAR(20) DEFAULT 'pending',
                requested_at TIMESTAMP DEFAULT NOW(),
                resolved_at TIMESTAMP,
                resolved_by VARCHAR(100),
                comment TEXT
            )
        """)
        await conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_approval_requests_session
            ON approval_requests (session_id)
        """)
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS session_runtime_state (
                session_id UUID PRIMARY KEY REFERENCES research_sessions(id) ON DELETE CASCADE,
                runtime_status VARCHAR(50) NOT NULL,
                public_status VARCHAR(20) NOT NULL,
                current_batch_json JSONB DEFAULT '[]',
                retryable_failure_count INTEGER DEFAULT 0,
                terminal_failure_reason TEXT,
                pending_approval_count INTEGER DEFAULT 0,
                checkpoint_seq INTEGER DEFAULT 0,
                last_checkpoint_ref TEXT,
                last_heartbeat_at TIMESTAMP DEFAULT NOW(),
                harness_state_version INTEGER DEFAULT 1
            )
        """)
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS mcp_server_registry (
                server_id VARCHAR(100) PRIMARY KEY,
                transport VARCHAR(50) NOT NULL,
                endpoint TEXT NOT NULL,
                trust_status VARCHAR(20) DEFAULT 'untrusted',
                allowed_tools_json JSONB DEFAULT '[]',
                read_only_default BOOLEAN DEFAULT TRUE,
                secret_policy_json JSONB DEFAULT '{}',
                fingerprint VARCHAR(255) NOT NULL,
                created_at TIMESTAMP DEFAULT NOW(),
                updated_at TIMESTAMP DEFAULT NOW()
            )
        """)
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS system_config (
                config_key VARCHAR(50) PRIMARY KEY,
                config_data JSONB NOT NULL DEFAULT '{}',
                description TEXT,
                created_at TIMESTAMP DEFAULT NOW(),
                updated_at TIMESTAMP DEFAULT NOW()
            )
        """)
        await conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_system_config_key
            ON system_config (config_key)
        """)
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS skills (
                id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                markdown_content TEXT NOT NULL,
                metadata_json JSONB NOT NULL DEFAULT '{}',
                is_deleted BOOLEAN DEFAULT FALSE,
                created_at TIMESTAMP DEFAULT NOW(),
                updated_at TIMESTAMP DEFAULT NOW()
            )
        """)
        await conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_skills_updated_at
            ON skills (updated_at DESC)
        """)
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS skill_meta (
                id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                slug VARCHAR(120) NOT NULL UNIQUE,
                name VARCHAR(120) NOT NULL,
                description TEXT NOT NULL,
                current_version INTEGER NOT NULL DEFAULT 1,
                enabled BOOLEAN DEFAULT TRUE,
                priority INTEGER DEFAULT 100,
                tags_json JSONB DEFAULT '[]',
                keywords_json JSONB DEFAULT '[]',
                trigger_patterns_json JSONB DEFAULT '[]',
                allowed_tools_json JSONB DEFAULT '[]',
                agent_hints_json JSONB DEFAULT '{}',
                domain VARCHAR(100),
                tenant_id VARCHAR(100),
                project_id VARCHAR(100),
                scope VARCHAR(20) DEFAULT 'global',
                coarse_terms_json JSONB DEFAULT '[]',
                metadata_json JSONB DEFAULT '{}',
                is_deleted BOOLEAN DEFAULT FALSE,
                created_at TIMESTAMP DEFAULT NOW(),
                updated_at TIMESTAMP DEFAULT NOW()
            )
        """)
        await conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_skill_meta_updated_at
            ON skill_meta (updated_at DESC)
        """)
        await conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_skill_meta_enabled
            ON skill_meta (enabled, is_deleted, priority DESC)
        """)
        await conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_skill_meta_scope
            ON skill_meta (scope, tenant_id, project_id)
        """)
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS skill_content (
                id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                skill_id UUID REFERENCES skill_meta(id) ON DELETE CASCADE,
                version INTEGER NOT NULL,
                markdown_content TEXT NOT NULL,
                body TEXT NOT NULL,
                overview TEXT DEFAULT '',
                prompt TEXT DEFAULT '',
                constraints TEXT DEFAULT '',
                content_hash VARCHAR(64),
                created_at TIMESTAMP DEFAULT NOW(),
                UNIQUE(skill_id, version)
            )
        """)
        await conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_skill_content_skill_version
            ON skill_content (skill_id, version DESC)
        """)

        # Security state is authoritative in PostgreSQL. Redis remains a cache and
        # must never decide whether a credential is valid.
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS auth_principals (
                id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                username VARCHAR(120) NOT NULL UNIQUE,
                password_hash TEXT NOT NULL,
                session_version INTEGER NOT NULL DEFAULT 1,
                is_active BOOLEAN NOT NULL DEFAULT TRUE,
                created_at TIMESTAMP NOT NULL DEFAULT NOW(),
                updated_at TIMESTAMP NOT NULL DEFAULT NOW()
            )
        """)
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS auth_sessions (
                id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                principal_id UUID NOT NULL REFERENCES auth_principals(id) ON DELETE CASCADE,
                token_hash CHAR(64) NOT NULL UNIQUE,
                session_version INTEGER NOT NULL,
                expires_at TIMESTAMP NOT NULL,
                revoked_at TIMESTAMP,
                created_at TIMESTAMP NOT NULL DEFAULT NOW(),
                last_used_at TIMESTAMP NOT NULL DEFAULT NOW()
            )
        """)
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS auth_api_tokens (
                id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                principal_id UUID NOT NULL REFERENCES auth_principals(id) ON DELETE CASCADE,
                token_hash CHAR(64) NOT NULL UNIQUE,
                token_prefix VARCHAR(16) NOT NULL,
                label VARCHAR(120) NOT NULL,
                session_version INTEGER NOT NULL,
                expires_at TIMESTAMP,
                revoked_at TIMESTAMP,
                created_at TIMESTAMP NOT NULL DEFAULT NOW(),
                last_used_at TIMESTAMP
            )
        """)
        await conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_auth_sessions_valid
            ON auth_sessions (token_hash, expires_at) WHERE revoked_at IS NULL
        """)
        await conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_auth_api_tokens_valid
            ON auth_api_tokens (token_hash) WHERE revoked_at IS NULL
        """)
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS capability_health_audit (
                id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                capability VARCHAR(80) NOT NULL,
                state VARCHAR(20) NOT NULL,
                reason_code VARCHAR(120),
                created_at TIMESTAMP NOT NULL DEFAULT NOW()
            )
        """)

        # Keep a single explicit owner even in self-hosted mode. New code must
        # always scope data by it, which makes later multi-user migration safe.
        for table in ("documents", "document_sources", "research_sessions", "tool_call_audit", "approval_requests", "system_config", "skill_meta"):
            await conn.execute(f"ALTER TABLE {table} ADD COLUMN IF NOT EXISTS owner_id UUID")
        await conn.execute("CREATE INDEX IF NOT EXISTS idx_documents_owner ON documents (owner_id)")
        await conn.execute("CREATE INDEX IF NOT EXISTS idx_document_sources_owner ON document_sources (owner_id)")
        await conn.execute("CREATE INDEX IF NOT EXISTS idx_research_sessions_owner ON research_sessions (owner_id)")
        await conn.execute("CREATE INDEX IF NOT EXISTS idx_skill_meta_owner ON skill_meta (owner_id)")
        await conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_skill_meta_owner_active ON skill_meta (owner_id, is_deleted, enabled)"
        )

    _db_pool = pool
    logger.info("Database initialized successfully")
    return pool


async def get_db_pool() -> asyncpg.Pool:
    """Get the database connection pool."""
    if _db_pool is None:
        return await init_db()
    return _db_pool


async def get_text_search_config() -> str:
    """Return the resolved PostgreSQL text search configuration."""
    global _fts_config
    if _fts_config is not None:
        return _fts_config

    pool = await get_db_pool()
    async with pool.acquire() as conn:
        _fts_config = await resolve_text_search_config(conn, log=logger)
    return _fts_config


async def get_redis() -> redis.Redis:
    """Get the Redis client."""
    global _redis_client
    if _redis_client is None:
        import redis.asyncio as redis
        settings = get_settings()
        _redis_client = redis.from_url(
            settings.redis.url,
            encoding="utf-8",
            decode_responses=True,
        )
    return _redis_client


async def close_db():
    """Close all database connections."""
    global _db_pool, _redis_client, _fts_config
    if _db_pool:
        await _db_pool.close()
        _db_pool = None
    _fts_config = None
    if _redis_client:
        await _redis_client.close()
        _redis_client = None


# ==============================================================
# SQLAlchemy-style models (using raw asyncpg for simplicity)
# ==============================================================

class Document:
    """Document model for the knowledge base."""

    @staticmethod
    async def create(
        content: str,
        metadata: dict[str, Any],
        embedding: list[float] | None = None,
        source_id: str | None = None,
        source_name: str | None = None,
        source_type: str = "manual",
        chunk_index: int = 0,
        chunk_count: int = 1,
        owner_id: str | None = None,
    ) -> str:
        """Create a new document and return its ID."""
        pool = await get_db_pool()
        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                INSERT INTO documents (
                    source_id, source_name, source_type,
                    chunk_index, chunk_count,
                    content, metadata, embedding, owner_id
                )
                VALUES ($1::uuid, $2, $3, $4, $5, $6, $7, $8::vector, $9::uuid)
                RETURNING id
                """,
                source_id,
                source_name,
                source_type,
                chunk_index,
                chunk_count,
                content,
                metadata,
                embedding,
                owner_id,
            )
            return str(row["id"])

    @staticmethod
    async def search_by_vector(
        embedding: list[float],
        owner_id: str,
        top_k: int = 10,
    ) -> list[dict[str, Any]]:
        """Search documents by vector similarity."""
        pool = await get_db_pool()
        async with pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT id, content, metadata,
                       1 - (embedding <=> $1::vector) AS similarity
                FROM documents
                WHERE owner_id = $2::uuid
                ORDER BY embedding <=> $1::vector
                LIMIT $3
                """,
                embedding,
                owner_id,
                top_k,
            )
            return [dict(r) for r in rows]

    @staticmethod
    async def count(owner_id: str) -> int:
        """Count total documents."""
        pool = await get_db_pool()
        async with pool.acquire() as conn:
            row = await conn.fetchrow("SELECT COUNT(*) FROM documents WHERE owner_id = $1::uuid", owner_id)
            return int(row["count"])


class DocumentSource:
    """Document source model."""

    @staticmethod
    async def create(
        *,
        name: str,
        group_name: str,
        source_type: str = "manual",
        file_name: str | None = None,
        file_ext: str | None = None,
        original_text: str | None = None,
        chunk_size: int = 400,
        chunk_overlap: int = 80,
        metadata: dict[str, Any] | None = None,
        owner_id: str | None = None,
    ) -> str:
        pool = await get_db_pool()
        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                INSERT INTO document_sources (
                    name, group_name, source_type, file_name, file_ext,
                    original_text, chunk_size, chunk_overlap, metadata, owner_id
                )
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10::uuid)
                RETURNING id
                """,
                name,
                group_name,
                source_type,
                file_name,
                file_ext,
                original_text,
                chunk_size,
                chunk_overlap,
                metadata or {},
                owner_id,
            )
            return str(row["id"])
