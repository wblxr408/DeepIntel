"""
Database migration script.

Initializes the PostgreSQL schema with pgvector extension,
creates all tables, indexes, and applies initial seed data.
"""

from __future__ import annotations

import asyncio
import logging

import asyncpg

from app.db.text_search import regconfig_sql_literal, resolve_text_search_config

logger = logging.getLogger(__name__)

# SQL for schema creation
SCHEMA_SQL = """
-- Enable pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- ==============================================================
-- Documents table (knowledge base with vector embeddings)
-- ==============================================================
CREATE TABLE IF NOT EXISTS documents (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    source_id UUID,
    source_name TEXT,
    source_type VARCHAR(20) DEFAULT 'manual',
    chunk_index INTEGER DEFAULT 0,
    chunk_count INTEGER DEFAULT 1,
    content TEXT NOT NULL,
    metadata JSONB DEFAULT '{}',
    embedding VECTOR({dimension}),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

ALTER TABLE documents ADD COLUMN IF NOT EXISTS source_id UUID;
ALTER TABLE documents ADD COLUMN IF NOT EXISTS source_name TEXT;
ALTER TABLE documents ADD COLUMN IF NOT EXISTS source_type VARCHAR(20) DEFAULT 'manual';
ALTER TABLE documents ADD COLUMN IF NOT EXISTS chunk_index INTEGER DEFAULT 0;
ALTER TABLE documents ADD COLUMN IF NOT EXISTS chunk_count INTEGER DEFAULT 1;

-- Vector similarity index (IVF for approximate nearest neighbor)
CREATE INDEX IF NOT EXISTS idx_documents_embedding
    ON documents USING ivfflat (embedding vector_cosine_ops)
    WITH (lists = 100);

-- Full-text search index
CREATE INDEX IF NOT EXISTS idx_documents_fts
    ON documents USING gin (to_tsvector({fts_config}, content));

-- Metadata filtering index
CREATE INDEX IF NOT EXISTS idx_documents_metadata
    ON documents USING gin (metadata);

-- Knowledge source table
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
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_document_sources_group
    ON document_sources (group_name);

-- ==============================================================
-- Research sessions table
-- ==============================================================
CREATE TABLE IF NOT EXISTS research_sessions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_query TEXT NOT NULL,
    status VARCHAR(20) DEFAULT 'pending'
        CHECK (status IN ('pending', 'running', 'completed', 'failed')),
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
    error_message TEXT,
    revision_count INTEGER DEFAULT 0,
    total_tokens INTEGER DEFAULT 0,
    total_cost_usd NUMERIC(10, 6) DEFAULT 0,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    completed_at TIMESTAMP WITH TIME ZONE
);

-- Status query index
CREATE INDEX IF NOT EXISTS idx_sessions_status ON research_sessions (status);
CREATE INDEX IF NOT EXISTS idx_sessions_created ON research_sessions (created_at DESC);

-- ==============================================================
-- Citations table
-- ==============================================================
CREATE TABLE IF NOT EXISTS citations (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    session_id UUID REFERENCES research_sessions(id) ON DELETE CASCADE,
    citation_id VARCHAR(20) NOT NULL,
    source_url TEXT,
    source_title TEXT,
    source_type VARCHAR(20) DEFAULT 'web'
        CHECK (source_type IN ('web', 'document', 'knowledge_base')),
    extracted_evidence TEXT,
    relevance_score FLOAT DEFAULT 0,
    access_timestamp TIMESTAMP WITH TIME ZONE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    UNIQUE(session_id, citation_id)
);

CREATE INDEX IF NOT EXISTS idx_citations_session ON citations (session_id);

-- ==============================================================
-- Agent trace events table (for detailed analytics)
-- ==============================================================
CREATE TABLE IF NOT EXISTS agent_events (
    id BIGSERIAL PRIMARY KEY,
    session_id UUID REFERENCES research_sessions(id) ON DELETE CASCADE,
    agent_name VARCHAR(50) NOT NULL,
    event_type VARCHAR(50) NOT NULL,
    content TEXT,
    duration_ms INTEGER,
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_events_session ON agent_events (session_id);
CREATE INDEX IF NOT EXISTS idx_events_agent ON agent_events (agent_name);
CREATE INDEX IF NOT EXISTS idx_events_type ON agent_events (event_type);

-- ==============================================================
-- Tool call metrics table
-- ==============================================================
CREATE TABLE IF NOT EXISTS tool_metrics (
    id BIGSERIAL PRIMARY KEY,
    session_id UUID REFERENCES research_sessions(id) ON DELETE CASCADE,
    tool_name VARCHAR(100) NOT NULL,
    call_count INTEGER DEFAULT 0,
    success_count INTEGER DEFAULT 0,
    error_count INTEGER DEFAULT 0,
    total_duration_ms BIGINT DEFAULT 0,
    avg_duration_ms FLOAT DEFAULT 0,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_tool_metrics_session ON tool_metrics (session_id);

-- ==============================================================
-- Tool call audit table (for forensic-grade per-call tracing)
-- ==============================================================
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
    started_at TIMESTAMP WITH TIME ZONE,
    completed_at TIMESTAMP WITH TIME ZONE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_tool_call_audit_session ON tool_call_audit (session_id);
CREATE INDEX IF NOT EXISTS idx_tool_call_audit_tool ON tool_call_audit (tool_name);
CREATE INDEX IF NOT EXISTS idx_tool_call_audit_status ON tool_call_audit (status);
ALTER TABLE tool_call_audit ADD COLUMN IF NOT EXISTS safety_json JSONB DEFAULT '{}';

-- ==============================================================
-- Session budget state table (session-level budget source of truth)
-- ==============================================================
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
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- ==============================================================
-- Approval requests table
-- ==============================================================
CREATE TABLE IF NOT EXISTS approval_requests (
    approval_id VARCHAR(64) PRIMARY KEY,
    session_id UUID REFERENCES research_sessions(id) ON DELETE CASCADE,
    node_id VARCHAR(64),
    tool_name VARCHAR(100) NOT NULL,
    risk_level VARCHAR(20) DEFAULT 'high',
    reason TEXT,
    request_payload_json JSONB DEFAULT '{}',
    status VARCHAR(20) DEFAULT 'pending',
    requested_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    resolved_at TIMESTAMP WITH TIME ZONE,
    resolved_by VARCHAR(100),
    comment TEXT
);

CREATE INDEX IF NOT EXISTS idx_approval_requests_session ON approval_requests (session_id);

-- ==============================================================
-- Session runtime state table
-- ==============================================================
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
    last_heartbeat_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    harness_state_version INTEGER DEFAULT 1
);

-- ==============================================================
-- MCP server registry table
-- ==============================================================
CREATE TABLE IF NOT EXISTS mcp_server_registry (
    server_id VARCHAR(100) PRIMARY KEY,
    transport VARCHAR(50) NOT NULL,
    endpoint TEXT NOT NULL,
    trust_status VARCHAR(20) DEFAULT 'untrusted',
    allowed_tools_json JSONB DEFAULT '[]',
    read_only_default BOOLEAN DEFAULT TRUE,
    secret_policy_json JSONB DEFAULT '{}',
    fingerprint VARCHAR(255) NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- ==============================================================
-- System configuration table (for frontend-configurable settings)
-- ==============================================================
CREATE TABLE IF NOT EXISTS system_config (
    config_key VARCHAR(50) PRIMARY KEY,
    config_data JSONB NOT NULL DEFAULT '{}',
    description TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_system_config_key ON system_config (config_key);

-- ==============================================================
-- Skills table (legacy raw markdown store)
-- ==============================================================
CREATE TABLE IF NOT EXISTS skills (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    markdown_content TEXT NOT NULL,
    metadata_json JSONB NOT NULL DEFAULT '{}',
    is_deleted BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_skills_updated_at ON skills (updated_at DESC);
CREATE INDEX IF NOT EXISTS idx_skills_metadata_json ON skills USING gin (metadata_json);

-- ==============================================================
-- Skill meta/content tables
-- ==============================================================
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
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_skill_meta_updated_at ON skill_meta (updated_at DESC);
CREATE INDEX IF NOT EXISTS idx_skill_meta_enabled ON skill_meta (enabled, is_deleted, priority DESC);
CREATE INDEX IF NOT EXISTS idx_skill_meta_scope ON skill_meta (scope, tenant_id, project_id);
CREATE INDEX IF NOT EXISTS idx_skill_meta_metadata_json ON skill_meta USING gin (metadata_json);

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
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    UNIQUE(skill_id, version)
);

CREATE INDEX IF NOT EXISTS idx_skill_content_skill_version ON skill_content (skill_id, version DESC);

-- ==============================================================
-- Trigger: auto-update updated_at
-- ==============================================================
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ language 'plpgsql';

CREATE OR REPLACE TRIGGER update_documents_updated_at
    BEFORE UPDATE ON documents
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE OR REPLACE TRIGGER update_sessions_updated_at
    BEFORE UPDATE ON research_sessions
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
"""


async def run_migration(
    database_url: str,
    embed_dimension: int = 1024,
    drop_existing: bool = False,
) -> None:
    """
    Run database migrations.

    Args:
        database_url: PostgreSQL connection URL
        embed_dimension: Embedding vector dimension (for pgvector column)
        drop_existing: If True, drop existing tables before creating
    """
    logger.info("Starting database migration...")

    pool = await asyncpg.create_pool(
        database_url,
        min_size=1,
        max_size=5,
    )

    try:
        async with pool.acquire() as conn:
            # Enable pgvector extension
            await conn.execute("CREATE EXTENSION IF NOT EXISTS vector")

            if drop_existing:
                logger.warning("Dropping existing tables (drop_existing=True)")
                await conn.execute("""
                    DROP TABLE IF EXISTS skill_content CASCADE;
                    DROP TABLE IF EXISTS skill_meta CASCADE;
                    DROP TABLE IF EXISTS skills CASCADE;
                    DROP TABLE IF EXISTS tool_metrics CASCADE;
                    DROP TABLE IF EXISTS tool_call_audit CASCADE;
                    DROP TABLE IF EXISTS session_budget_state CASCADE;
                    DROP TABLE IF EXISTS agent_events CASCADE;
                    DROP TABLE IF EXISTS citations CASCADE;
                    DROP TABLE IF EXISTS research_sessions CASCADE;
                    DROP TABLE IF EXISTS documents CASCADE;
                """)

            # Create schema
            fts_config = await resolve_text_search_config(conn, log=logger)
            try:
                schema = SCHEMA_SQL.format(
                    dimension=embed_dimension,
                    fts_config=regconfig_sql_literal(fts_config),
                )
                await conn.execute(schema)
            except (OSError, RuntimeError, asyncpg.PostgresError) as exc:
                logger.warning(
                    "Migration failed with text search config '%s'; retrying with 'simple': %s",
                    fts_config,
                    exc,
                )
                schema = SCHEMA_SQL.format(
                    dimension=embed_dimension,
                    fts_config="'simple'",
                )
                await conn.execute(schema)

            logger.info("Database migration completed successfully")

            # Verify tables
            tables = await conn.fetch("""
                SELECT table_name
                FROM information_schema.tables
                WHERE table_schema = 'public'
                ORDER BY table_name
            """)
            logger.info(f"Created tables: {[t['table_name'] for t in tables]}")

    finally:
        await pool.close()


def main() -> None:
    """CLI entry point for migration."""
    import argparse

    from app.config import get_settings

    parser = argparse.ArgumentParser(description="Run database migrations")
    parser.add_argument(
        "--drop",
        action="store_true",
        help="Drop existing tables before creating (destructive!)",
    )
    parser.add_argument(
        "--dimension",
        type=int,
        default=1024,
        help="Embedding dimension (default: 1024)",
    )
    args = parser.parse_args()

    settings = get_settings()
    asyncio.run(
        run_migration(
            database_url=settings.database.url,
            embed_dimension=args.dimension,
            drop_existing=args.drop,
        )
    )


if __name__ == "__main__":
    main()
