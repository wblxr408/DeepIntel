"""
Search tools for LangChain/LangGraph integration.

Provides DuckDuckGo search as a LangChain-compatible tool.
"""

from __future__ import annotations

import logging

from langchain_core.tools import tool

logger = logging.getLogger(__name__)


@tool
def duckduckgo_search(query: str) -> str:
    """
    Search the web using DuckDuckGo. Use this for quick factual lookups,
    statistics, market data, and news.

    Args:
        query: The search query (be specific and use Chinese for Chinese sources)

    Returns:
        JSON string of search results with title, URL, and snippet
    """
    import json
    try:
        from app.agents.search import SearchAgent

        results = [
            result.model_dump()
            for result in SearchAgent().execute_search(query)
        ]

        return json.dumps(results, ensure_ascii=False, indent=2)

    except (OSError, RuntimeError, TypeError, ValueError, KeyError) as e:
        logger.error(f"DuckDuckGo search error: {e}")
        return json.dumps({"error": str(e)})


def get_search_tools():
    """Return all search tools for LangGraph."""
    return [duckduckgo_search]
