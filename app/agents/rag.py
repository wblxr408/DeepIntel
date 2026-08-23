"""
RAG Agent: hybrid retrieval from the knowledge base.

Implements vector search + BM25 + RRF fusion + BGE Reranker.
"""

from __future__ import annotations

import logging

from typing_extensions import Self

from app.config import get_settings
from app.db.connection import get_text_search_config
from app.db.text_search import regconfig_sql_literal
from app.graph.state import Citation, PlanStep, RAGResult
from app.security.capabilities import CapabilityUnavailable, capability_registry

logger = logging.getLogger(__name__)


class RAGAgent:
    """
    Performs hybrid retrieval combining vector similarity and BM25 keyword matching.

    Pipeline:
    1. Embed query with BGE
    2. ANN search via pgvector
    3. BM25 search via PostgreSQL full-text
    4. RRF fusion
    5. BGE Reranker for final ranking

    Lifecycle:
    - Use close() or async context manager to properly release resources
    - The agent lazily initializes the DB pool and models on first use
    """

    def __init__(self):
        settings = get_settings()
        self.rag_cfg = settings.rag
        self._db_pool = None
        self._embedder = None
        self._reranker = None

    async def _get_db_pool(self):
        """Get or create the database connection pool."""
        if self._db_pool is None:
            import asyncpg
            db_settings = get_settings().database
            self._db_pool = await asyncpg.create_pool(
                db_settings.url,
                min_size=2,
                max_size=10,
            )
        return self._db_pool

    async def _get_embedder(self):
        """Lazy-load the embedding model."""
        if self._embedder is None:
            from app.rag.embedder import Embedder
            self._embedder = Embedder()
        return self._embedder

    async def _get_reranker(self):
        """Lazy-load the reranker model."""
        if self._reranker is None:
            from app.rag.reranker import Reranker
            self._reranker = Reranker()
        return self._reranker

    async def close(self) -> None:
        """
        Close all resources held by the agent.

        Call this when done using the agent, or use the async context manager:
            async with RAGAgent() as agent:
                results = await agent.execute_async(steps, query)
        """
        if self._db_pool is not None:
            await self._db_pool.close()
            self._db_pool = None
            logger.info("RAG Agent: database pool closed")

    async def __aenter__(self) -> Self:
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit - ensures cleanup."""
        await self.close()

    async def execute_async(
        self,
        plan_steps: list[PlanStep],
        user_query: str,
        group: str | None = None,
        owner_id: str | None = None,
    ) -> list[RAGResult]:
        """
        Execute RAG retrieval for the given plan steps.
        """
        if not owner_id:
            logger.warning("RAG retrieval rejected because owner scope is missing")
            return []

        logger.info(f"RAG Agent: processing {len(plan_steps)} steps")

        try:
            pool = await capability_registry.run("rag_database", self._get_db_pool)
            embedder = await capability_registry.run("embedding", self._get_embedder)
        except CapabilityUnavailable as exc:
            logger.warning("RAG unavailable: %s", exc)
            return []
        reranker = None
        try:
            reranker = await capability_registry.run("reranker", self._get_reranker)
        except CapabilityUnavailable as exc:
            logger.warning("Reranker unavailable; using hybrid ranking: %s", exc)

        results: list[RAGResult] = []

        for step in plan_steps:
            try:
                # Embed the query
                query_embedding = await embedder.embed(step.target_query)

                # Fetch from DB
                async with pool.acquire() as conn:
                    fts_config = await get_text_search_config()
                    fts_config_sql = regconfig_sql_literal(fts_config)

                    # pgvector ANN search
                    vector_rows = await conn.fetch("""
                        SELECT id, content, metadata,
                               1 - (embedding <=> $1::vector) AS similarity
                        FROM documents
                        WHERE (
                            $2::text IS NULL
                            OR metadata->>'group' = $2::text
                            OR metadata->>'source_group' = $2::text
                            OR metadata->>'knowledge_group' = $2::text
                        )
                        AND ($3::uuid IS NULL OR owner_id = $3::uuid)
                        ORDER BY embedding <=> $1::vector
                        LIMIT 30
                    """, query_embedding, group, owner_id)

                    # BM25 search
                    bm25_rows = await conn.fetch(
                        f"""
                        SELECT id, content, metadata,
                               ts_rank(
                                   to_tsvector({fts_config_sql}, content),
                                   plainto_tsquery({fts_config_sql}, $1)
                               ) AS bm25_score
                        FROM documents
                        WHERE to_tsvector({fts_config_sql}, content) @@ plainto_tsquery({fts_config_sql}, $1)
                        AND (
                            $2::text IS NULL
                            OR metadata->>'group' = $2::text
                            OR metadata->>'source_group' = $2::text
                            OR metadata->>'knowledge_group' = $2::text
                        )
                        AND ($3::uuid IS NULL OR owner_id = $3::uuid)
                        ORDER BY bm25_score DESC
                        LIMIT 30
                        """,
                        step.target_query,
                        group,
                        owner_id,
                    )

                # Build result lists
                vector_results = {
                    str(r["id"]): {"row": r, "score": float(r["similarity"])}
                    for r in vector_rows
                }
                bm25_results = {
                    str(r["id"]): {"row": r, "score": float(r["bm25_score"]) + 0.001}
                    for r in bm25_rows
                }

                # RRF Fusion
                all_ids = set(vector_results) | set(bm25_results)
                rrf_scores: dict[str, float] = {}

                for doc_id in all_ids:
                    vector_results.get(doc_id, {})
                    bm25_results.get(doc_id, {})
                    v_rank = list(vector_results.keys()).index(doc_id) + 1 if doc_id in vector_results else len(vector_results) + 1
                    b_rank = list(bm25_results.keys()).index(doc_id) + 1 if doc_id in bm25_results else len(bm25_results) + 1

                    k = self.rag_cfg.rrf_k
                    rrf_scores[doc_id] = (1 / (k + v_rank)) + (1 / (k + b_rank))

                # Get top candidates for reranking
                top_ids = sorted(rrf_scores, key=rrf_scores.get, reverse=True)[:15]

                # Prepare for reranker
                all_rows = {str(r["id"]): r for r in vector_rows + bm25_rows}
                rerank_docs = [
                    all_rows[did]["content"]
                    for did in top_ids
                    if did in all_rows
                ]

                if not rerank_docs:
                    continue

                # Rerank
                if reranker is not None:
                    try:
                        reranked = await capability_registry.run(
                            "reranker",
                        lambda query=step.target_query, documents=rerank_docs: reranker.rerank(
                            query=query,
                            documents=documents,
                                top_n=self.rag_cfg.rerank_top_n,
                            ),
                        )
                    except CapabilityUnavailable:
                        reranked = [rrf_scores[doc_id] for doc_id in top_ids]
                else:
                    reranked = [rrf_scores[doc_id] for doc_id in top_ids]

                # Build final results
                for i, (rerank_score, doc_id) in enumerate(zip(reranked, top_ids)):
                    row = all_rows[doc_id]
                    meta = row["metadata"] or {}

                    citation = Citation(
                        source_url=meta.get("url", ""),
                        source_title=meta.get("title", "Document"),
                        source_type="knowledge_base",
                        extracted_evidence=row["content"][:300],
                        relevance_score=rerank_score,
                        chunk_id=str(row["id"]),
                    )

                    results.append(RAGResult(
                        chunk_id=str(row["id"]),
                        content=row["content"],
                        metadata=meta,
                        vector_score=vector_results.get(doc_id, {}).get("score", 0.0),
                        bm25_score=bm25_results.get(doc_id, {}).get("score", 0.0),
                        rrf_score=rrf_scores.get(doc_id, 0.0),
                        rerank_score=rerank_score,
                        citation=citation,
                    ))

            except (OSError, RuntimeError, TypeError, ValueError, KeyError) as e:
                logger.error(f"RAG step error for '{step.target_query}': {e}")
                continue

        logger.info(f"RAG Agent: {len(results)} results retrieved")
        return results

    def execute(
        self,
        plan_steps: list[PlanStep],
        user_query: str,
        group: str | None = None,
        owner_id: str | None = None,
    ) -> list[RAGResult]:
        """Synchronous wrapper."""
        try:
            import asyncio
            return asyncio.run(self.execute_async(plan_steps, user_query, group=group, owner_id=owner_id))
        except RuntimeError:
            import nest_asyncio
            nest_asyncio.apply()
            import asyncio
            return asyncio.run(self.execute_async(plan_steps, user_query, group=group, owner_id=owner_id))

    def execute_retrieval(
        self,
        query: str,
        context: str = "",
        group: str | None = None,
        owner_id: str | None = None,
    ) -> list[RAGResult]:
        """
        Execute RAG retrieval for a single query (主题 3 工具接口).

        直接执行单个检索查询，供 DAG executor 调用。
        """
        from app.graph.state import PlanStep
        step = PlanStep(target_query=query, node_type="rag")
        return self.execute([step], context or query, group=group, owner_id=owner_id)

    def execute_for_node(
        self,
        node_query: str,
        context: str = "",
        group: str | None = None,
        owner_id: str | None = None,
    ) -> list[RAGResult]:
        """兼容接口：根据节点 query 执行检索。"""
        return self.execute_retrieval(node_query, context, group=group, owner_id=owner_id)
