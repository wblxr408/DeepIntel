"""
Pytest configuration and shared fixtures for the DeepIntel test suite.

This file provides:
- Database fixtures (in-memory or mocked)
- Mock LLM clients
- Test client fixtures
- Agent fixtures
- Temp directory fixtures
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# Add app to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


# ==============================================================
# Async Event Loop Fixture
# ==============================================================

@pytest.fixture(scope="session")
def event_loop_policy():
    """Use the default event loop policy."""
    return asyncio.DefaultEventLoopPolicy()


@pytest.fixture
def event_loop():
    """Create a new event loop for each test."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


# ==============================================================
# Mock LLM Client Fixture
# ==============================================================

@pytest.fixture
def mock_llm_response() -> dict[str, Any]:
    """Default mock LLM response for planner."""
    return {
        "choices": [
            {
                "message": {
                    "content": json.dumps({
                        "steps": [
                            {
                                "step_id": "step-1",
                                "description": "Search market data",
                                "assigned_agent": "search",
                                "target_query": "China EV market statistics 2025",
                                "status": "pending",
                            },
                            {
                                "step_id": "step-2",
                                "description": "Deep read news",
                                "assigned_agent": "browser",
                                "target_query": "Latest EV policy news",
                                "status": "pending",
                            },
                            {
                                "step_id": "step-3",
                                "description": "Query knowledge base",
                                "assigned_agent": "rag",
                                "target_query": "EV market background",
                                "status": "pending",
                            },
                        ]
                    })
                }
            }
        ]
    }


@pytest.fixture
def mock_llm_client(mock_llm_response: dict[str, Any]) -> MagicMock:
    """Mock OpenAI-compatible LLM client."""
    client = MagicMock()
    client.api_key = "test-key"
    client.base_url = None
    client.chat.completions.create = MagicMock(return_value=MagicMock(**mock_llm_response))
    return client


@pytest.fixture
def mock_llm_fallback_response() -> dict[str, Any]:
    """Mock LLM response for analyst."""
    return {
        "choices": [
            {
                "message": {
                    "content": "# Research Analysis\n\nBased on the evidence collected, here are the key findings."
                }
            }
        ]
    }


# ==============================================================
# Mock Browser Fixture
# ==============================================================

@pytest.fixture
def mock_browser_page() -> MagicMock:
    """Mock Playwright page."""
    page = MagicMock()
    page.title = AsyncMock(return_value="Test Page Title")
    page.query_selector = AsyncMock(return_value=None)
    page.query_selector_all = AsyncMock(return_value=[])
    page.goto = AsyncMock(return_value=MagicMock(status=200))
    page.set_extra_http_headers = AsyncMock()
    return page


@pytest.fixture
def mock_browser() -> MagicMock:
    """Mock Playwright browser."""
    browser = MagicMock()
    browser.new_page = AsyncMock(return_value=mock_browser_page())
    browser.close = AsyncMock()
    return browser


# ==============================================================
# Mock Database Fixtures
# ==============================================================

@pytest.fixture
def mock_db_pool() -> MagicMock:
    """Mock asyncpg connection pool."""
    pool = MagicMock()
    conn = MagicMock()
    conn.fetch = AsyncMock(return_value=[])
    conn.fetchrow = AsyncMock(return_value=None)
    conn.execute = AsyncMock(return_value="OK")

    pool.acquire = MagicMock(return_value=MagicMock(
        __aenter__=AsyncMock(return_value=conn),
        __aexit__=AsyncMock(return_value=None),
    ))
    return pool


# ==============================================================
# Mock Redis Fixture
# ==============================================================

@pytest.fixture
def mock_redis() -> MagicMock:
    """Mock Redis client."""
    redis = MagicMock()
    redis.ping = AsyncMock(return_value=True)
    redis.set = AsyncMock(return_value=True)
    redis.get = AsyncMock(return_value=None)
    redis.delete = AsyncMock(return_value=1)
    return redis


# ==============================================================
# State Fixtures
# ==============================================================

@pytest.fixture
def sample_state() -> dict[str, Any]:
    """Sample initial research state."""
    from app.graph.state import create_initial_state
    return create_initial_state("Test research query", "test-session-001")


@pytest.fixture
def sample_plan_steps() -> list[dict[str, Any]]:
    """Sample plan steps."""
    return [
        {
            "step_id": "step-1",
            "description": "Search for market data",
            "assigned_agent": "search",
            "target_query": "China EV market 2025",
            "status": "pending",
            "evidence_ids": [],
            "retry_count": 0,
        },
        {
            "step_id": "step-2",
            "description": "Deep read news",
            "assigned_agent": "browser",
            "target_query": "EV policy news",
            "status": "pending",
            "evidence_ids": [],
            "retry_count": 0,
        },
    ]


# ==============================================================
# Agent Fixtures
# ==============================================================

@pytest.fixture
def planner_agent(mock_llm_client: MagicMock) -> MagicMock:
    """Planner agent with mocked LLM."""
    from app.agents.planner import PlannerAgent
    with patch.object(PlannerAgent, "__init__", lambda self: None):
        agent = PlannerAgent()
        agent._client = mock_llm_client
        agent.model = "qwen-plus"
        agent.provider = "qwen"
    return agent


@pytest.fixture
def search_agent() -> MagicMock:
    """Search agent fixture."""
    from app.agents.search import SearchAgent
    with patch.object(SearchAgent, "__init__", lambda self: None):
        agent = SearchAgent()
        agent._client = None
        agent.model = "qwen-plus"
    return agent


@pytest.fixture
def analyst_agent(mock_llm_client: MagicMock) -> MagicMock:
    """Analyst agent with mocked LLM."""
    from app.agents.analyst import AnalystAgent
    with patch.object(AnalystAgent, "__init__", lambda self: None):
        agent = AnalystAgent()
        agent._client = mock_llm_client
        agent.model = "qwen-plus"
    return agent


@pytest.fixture
def reflection_agent(mock_llm_client: MagicMock) -> MagicMock:
    """Reflection agent with mocked LLM."""
    from app.agents.reflection import ReflectionAgent
    with patch.object(ReflectionAgent, "__init__", lambda self: None):
        agent = ReflectionAgent()
        agent._client = mock_llm_client
        agent.model = "qwen-plus"
    return agent


# ==============================================================
# SSE Fixtures
# ==============================================================

@pytest.fixture
def sse_manager():
    """SSE manager fixture."""
    from app.observability.sse_manager import SSEManager
    return SSEManager()


# ==============================================================
# Graph Fixtures
# ==============================================================

@pytest.fixture
def research_graph():
    """Compiled research graph."""
    from app.graph.compiler import compile_research_graph
    return compile_research_graph()


# ==============================================================
# Temp Directory Fixtures
# ==============================================================

@pytest.fixture
def temp_metrics_dir(tmp_path) -> Any:
    """Temp directory for metrics data."""
    metrics_dir = tmp_path / "metrics"
    metrics_dir.mkdir()
    return metrics_dir


# ==============================================================
# Config Override Fixtures
# ==============================================================

@pytest.fixture(autouse=True)
def mock_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    """Mock settings for all tests."""
    import app.config as config_module

    class MockDatabaseSettings:
        url = "postgresql://test:test@localhost:5432/test"
        pool_size = 5
        max_overflow = 10

    class MockRedisSettings:
        url = "redis://localhost:6379/0"
        session_ttl = 3600
        cache_ttl = 1800

    class MockLLMSettings:
        provider = "openai"
        model = "gpt-4o-mini"
        api_key = "test-key"
        api_base = None
        temperature = 0.7
        max_tokens = 2048
        fallback_provider = None
        fallback_model = None
        fallback_api_key = None

    class MockRAGSettings:
        embed_model = "BAAI/bge-zh-qwen2-int8"
        embed_dimension = 1024
        rerank_model = "BAAI/bge-reranker-v2-m3"
        rerank_device = "cpu"
        retrieval_top_k = 20
        rerank_top_n = 10
        rrf_k = 60

    class MockBrowserSettings:
        headless = True
        pool_size = 2
        navigation_timeout = 15000
        user_agent = "TestBot/1.0"
        accept_lang = "en-US,en;q=0.9"
        snippet_max_chars = 200
        skim_max_chars = 1000
        deep_max_chars = 3000

    class MockAPISettings:
        host = "0.0.0.0"
        port = 8000
        reload = False
        log_level = "INFO"
        cors_origins = ("http://localhost:5173",)

    class MockSettings:
        llm = MockLLMSettings()
        database = MockDatabaseSettings()
        redis = MockRedisSettings()
        rag = MockRAGSettings()
        browser = MockBrowserSettings()
        api = MockAPISettings()
        skills = type("MockSkillSettings", (), {
            "match_top_k": 5,
            "coarse_candidate_floor": 12,
            "broad_fallback_limit": 32,
            "content_l1_cache_size": 64,
            "match_cache_ttl": 900,
            "content_cache_ttl": 1800,
        })()
        sse_enabled = True
        observability_enabled = True

    monkeypatch.setattr(config_module, "get_settings", lambda: MockSettings())
