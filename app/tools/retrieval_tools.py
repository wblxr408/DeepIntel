"""
Retrieval tools for LangChain/LangGraph integration.

Provides RAG retrieval as LangChain-compatible tools.
"""

from __future__ import annotations

import json
import logging
from contextvars import ContextVar

from langchain_core.tools import tool

logger = logging.getLogger(__name__)
_retrieval_owner_id: ContextVar[str | None] = ContextVar("retrieval_owner_id", default=None)


@tool
def knowledge_base_search(
    query: str,
    top_k: int = 5,
    group: str | None = None,
) -> str:
    """
    Search the internal knowledge base for relevant documents and context.

    Use this for:
    - Domain background knowledge
    - Previous research reports
    - Technical documentation
    - Historical data and trends

    Args:
        query: The search query
        top_k: Number of top results to return (default 5)
        group: Optional knowledge source group. Matches metadata.group,
            metadata.source_group, or metadata.knowledge_group.

    Returns:
        JSON string of retrieved document chunks with metadata
    """
    owner_id = _retrieval_owner_id.get()
    if not owner_id:
        return json.dumps({"error": "owner_scope_required"})

    try:
        from app.agents.rag import RAGAgent
        from app.graph.state import PlanStep

        # Quick single-query RAG retrieval
        step = PlanStep(
            description=f"Knowledge base search: {query}",
            assigned_agent="rag",
            target_query=query,
        )
        agent = RAGAgent()
        results = agent.execute([step], query, group=group, owner_id=owner_id)

        output = []
        for r in results[:top_k]:
            output.append({
                "chunk_id": r.chunk_id,
                "content": r.content[:500],
                "metadata": r.metadata,
                "rerank_score": r.rerank_score,
                "source": r.metadata.get("title", "Unknown") if r.metadata else "Unknown",
            })

        return json.dumps(output, ensure_ascii=False, indent=2)

    except (OSError, RuntimeError, TypeError, ValueError, KeyError) as e:
        logger.error(f"Knowledge base search error: {e}")
        return json.dumps({"error": str(e)})


def get_retrieval_tools(owner_id: str):
    """Return tools bound to a server-derived owner, never an LLM argument."""
    @tool
    def scoped_knowledge_base_search(query: str, top_k: int = 5, group: str | None = None) -> str:
        token = _retrieval_owner_id.set(owner_id)
        try:
            return knowledge_base_search.invoke({"query": query, "top_k": top_k, "group": group})
        finally:
            _retrieval_owner_id.reset(token)

    return [scoped_knowledge_base_search]
