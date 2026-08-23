"""
Tests for the RAG module (embedding, retrieval, reranking).
"""

from unittest.mock import patch

import pytest

from app.rag.retriever import HybridRetriever


class TestHybridRetriever:
    def test_rrf_fusion_empty(self):
        retriever = HybridRetriever()
        result = retriever.reciprocal_rank_fusion([])
        assert result == []

    def test_rrf_fusion_single_list(self):
        retriever = HybridRetriever()
        result = retriever.reciprocal_rank_fusion([[
            ("doc1", 0.9),
            ("doc2", 0.7),
            ("doc3", 0.5),
        ]])
        assert len(result) == 3
        # Should be sorted by RRF score (which equals rank in single list)
        assert result[0][0] == "doc1"
        assert result[1][0] == "doc2"
        assert result[2][0] == "doc3"

    def test_rrf_fusion_multiple_lists(self):
        retriever = HybridRetriever()
        list1 = [("doc1", 0.9), ("doc2", 0.7), ("doc3", 0.5)]
        list2 = [("doc3", 0.9), ("doc1", 0.6), ("doc4", 0.8)]

        result = retriever.reciprocal_rank_fusion([list1, list2], k=60)
        result_dict = {doc_id: score for doc_id, score in result}

        # doc1 appears first in list1 and second in list2
        # doc3 appears third in list1 and first in list2
        # Both should have high scores
        assert "doc1" in result_dict
        assert "doc3" in result_dict

        # doc2 only in list1
        assert "doc2" in result_dict

        # doc4 only in list2
        assert "doc4" in result_dict

    def test_rrf_fusion_preserves_all_docs(self):
        retriever = HybridRetriever()
        list1 = [("a", 1.0), ("b", 1.0)]
        list2 = [("c", 1.0), ("d", 1.0)]

        result = retriever.reciprocal_rank_fusion([list1, list2], k=60)
        result_ids = {doc_id for doc_id, _ in result}

        assert result_ids == {"a", "b", "c", "d"}

    def test_rrf_k_parameter_effect(self):
        retriever = HybridRetriever()
        list1 = [("doc1", 1.0), ("doc2", 1.0)]

        # With higher k, lower-ranked docs get relatively more weight
        result_low_k = retriever.reciprocal_rank_fusion([list1], k=1)
        result_high_k = retriever.reciprocal_rank_fusion([list1], k=100)

        # Ratio of scores should differ
        score_low_k = result_low_k[0][1] / result_low_k[1][1]
        score_high_k = result_high_k[0][1] / result_high_k[1][1]

        # Higher k makes the ratio closer to 1 (less difference between ranks)
        assert score_high_k < score_low_k


class TestEmbedder:
    def test_embedder_import(self):
        from app.rag.embedder import Embedder
        # Don't actually load the model in test
        with patch.object(Embedder, '__init__', lambda self, **kw: None):
            embedder = Embedder()
            embedder.model_name = "test-model"
            embedder.device = "cpu"
            embedder._model = None
            assert embedder.model_name == "test-model"


class TestReranker:
    def test_reranker_import(self):
        from app.rag.reranker import Reranker
        with patch.object(Reranker, '__init__', lambda self, **kw: None):
            reranker = Reranker()
            reranker.model_name = "test-reranker"
            reranker.device = "cpu"
            reranker._model = None
            reranker._tokenizer = None
            assert reranker.model_name == "test-reranker"


@pytest.mark.asyncio
async def test_rag_agent_scopes_vector_and_bm25_queries_to_owner(monkeypatch):
    from app.agents.rag import RAGAgent
    from app.graph.state import PlanStep

    owner_id = "00000000-0000-0000-0000-000000000001"
    calls: list[tuple[str, tuple[object, ...]]] = []

    class Conn:
        async def fetch(self, query, *args):
            calls.append((query, args))
            return []

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return False

    class Pool:
        def acquire(self):
            return Conn()

    class Embedder:
        async def embed(self, query):
            return [0.1, 0.2]

    agent = RAGAgent()
    agent._db_pool = Pool()
    agent._embedder = Embedder()

    async def capability_run(name, initializer):
        if name == "reranker":
            from app.security.capabilities import (
                CapabilityHealth,
                CapabilityStatus,
                CapabilityUnavailable,
            )
            raise CapabilityUnavailable(CapabilityStatus(name, CapabilityHealth.DEGRADED, "offline"))
        return await initializer()

    monkeypatch.setattr("app.agents.rag.capability_registry.run", capability_run)
    monkeypatch.setattr("app.agents.rag.get_text_search_config", _fts_config)

    result = await agent.execute_async([PlanStep(node_type="rag", query="private data")], "private data", owner_id=owner_id)

    assert result == []
    assert len(calls) == 2
    assert all(args[-1] == owner_id for _, args in calls)
    assert all("owner_id = $3::uuid" in query for query, _ in calls)


async def _fts_config():
    return "simple"


def test_knowledge_base_tool_refuses_unscoped_retrieval():
    from app.tools.retrieval_tools import knowledge_base_search

    response = knowledge_base_search.invoke({"query": "private data"})

    assert '"owner_scope_required"' in response


@pytest.mark.asyncio
async def test_rag_agent_refuses_unscoped_retrieval_without_initializing_dependencies(monkeypatch):
    from app.agents.rag import RAGAgent
    from app.graph.state import PlanStep

    agent = RAGAgent()

    async def fail_if_called(*args, **kwargs):
        raise AssertionError("unscoped retrieval must not initialize capabilities")

    monkeypatch.setattr("app.agents.rag.capability_registry.run", fail_if_called)

    assert await agent.execute_async([PlanStep(node_type="rag", query="private data")], "private data") == []


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
