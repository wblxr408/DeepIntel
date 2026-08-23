"""
Tests for individual agent implementations.
"""

import asyncio
from unittest.mock import MagicMock, patch

import pytest

from app.graph.state import AgentType, Evidence, StepStatus


class TestPlannerAgent:
    def test_planner_import(self):
        from app.agents.planner import PlannerAgent
        agent = PlannerAgent()
        assert agent is not None

    def test_fallback_plan_generation(self):
        from app.agents.planner import PlannerAgent

        agent = PlannerAgent()
        # Mock the client to fail
        agent._client = MagicMock()
        agent._client.chat.completions.create.side_effect = RuntimeError("API Error")

        with patch("app.agents.planner.create_fallback_llm_client", return_value=None):
            steps = agent.create_plan("中国新能源车市场分析")
        assert len(steps) == 4
        assert all(s["status"] == StepStatus.PENDING for s in steps)
        # Should have one of each agent type
        agent_types = {s.get("assigned_agent") or s.get("node_type") for s in steps}
        assert "search" in agent_types
        assert "browser" in agent_types
        assert "rag" in agent_types
        assert "analyst" in agent_types

    def test_fact_lookup_uses_minimal_dag(self):
        from app.agents.planner import PlannerAgent

        dag = PlannerAgent().create_dag("量子计算是什么")

        assert len(dag.nodes) == 3
        assert [node.node_type for node in dag.nodes] == ["rag", "search", "analyst"]


class TestSearchAgent:
    def test_search_agent_import(self):
        from app.agents.search import SearchAgent
        agent = SearchAgent()
        assert agent is not None

    def test_execute_search_uses_ten_second_timeout(self, monkeypatch):
        from app.agents.search import SearchAgent

        captured = {"timeouts": []}

        def fake_get(url, params=None, timeout=None, headers=None):
            captured["timeouts"].append(timeout)

            class _Response:
                status_code = 200
                text = ""

                def json(self):
                    return {}

            return _Response()

        monkeypatch.setattr("requests.get", fake_get)

        agent = SearchAgent()
        result = agent._execute_search("test query")

        assert captured["timeouts"] == [10, 10]
        assert result == []

    def test_parse_duckduckgo_html_results(self):
        from app.agents.search import SearchAgent

        html = '''
        <div class="result">
          <div class="result__body">
            <a class="result__a" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fexample.com%2Fdoc">Example &amp; Title</a>
            <a class="result__snippet">Useful snippet about the topic.</a>
          </div>
        </div>
        '''

        results = SearchAgent()._parse_duckduckgo_html(html, "topic")

        assert len(results) == 1
        assert results[0].url == "https://example.com/doc"
        assert results[0].title == "Example & Title"

    def test_deduplication(self):
        from app.agents.search import SearchAgent, SearchResult

        agent = SearchAgent()
        results = [
            SearchResult(url="https://example.com/1", title="A", snippet="x", relevance_score=0.5),
            SearchResult(url="https://example.com/1", title="A", snippet="x", relevance_score=0.8),
            SearchResult(url="https://example.com/2", title="B", snippet="y", relevance_score=0.6),
        ]
        deduped = agent._deduplicate(results)
        assert len(deduped) == 2
        # Should keep the one with higher score
        assert next(r for r in deduped if r.url == "https://example.com/1").relevance_score == 0.8


class TestBrowserAgent:
    def test_browser_agent_import(self):
        from app.agents.browser import BrowserAgent
        agent = BrowserAgent()
        assert agent is not None

    def test_page_classification(self):
        from unittest.mock import MagicMock

        from app.agents.browser import BrowserAgent, PageType

        agent = BrowserAgent()

        # News article
        result = asyncio.run(agent._classify_page(MagicMock(), "https://news.example.com/article/123"))
        assert result == PageType.NEWS_ARTICLE

        # GitHub
        result = asyncio.run(agent._classify_page(MagicMock(), "https://github.com/user/repo"))
        assert result == PageType.TECHNICAL

        # Search result
        result = asyncio.run(agent._classify_page(MagicMock(), "https://www.google.com/search?q=AI"))
        assert result == PageType.SEARCH_RESULT

        # Social
        result = asyncio.run(agent._classify_page(MagicMock(), "https://zhihu.com/question/123"))
        assert result == PageType.SOCIAL

        # General
        result = asyncio.run(agent._classify_page(MagicMock(), "https://example.com/page"))
        assert result == PageType.GENERAL

    def test_fallback_result(self):
        from app.agents.browser import BrowserAgent

        agent = BrowserAgent()
        result = agent._fallback_result("https://failed.com")
        assert result.url == "https://failed.com"
        assert result.extracted_content == ""
        assert len(result.citations) == 0

    def test_fallback_result_preserves_error_message(self):
        from app.agents.browser import BrowserAgent

        agent = BrowserAgent()
        result = agent._fallback_result("https://failed.com", error_message="navigation timeout")
        assert result.error_message == "navigation timeout"


class TestRAGAgent:
    def test_rag_agent_import(self):
        from app.agents.rag import RAGAgent
        agent = RAGAgent()
        assert agent is not None


class TestAnalystAgent:
    def test_analyst_import(self):
        from app.agents.analyst import AnalystAgent
        agent = AnalystAgent()
        assert agent is not None

    def test_evidence_formatting(self):
        from app.agents.analyst import AnalystAgent, Evidence

        agent = AnalystAgent()
        evidence = [
            Evidence(
                content="这是测试内容的一部分",
                source_title="测试来源",
                source_url="https://test.com",
                source_type="web",
                collected_by=AgentType.SEARCH,
            )
        ]
        formatted = agent._format_evidence(evidence)
        assert "测试来源" in formatted
        assert "这是测试内容的一部分" in formatted


class TestReflectionAgent:
    def test_reflection_import(self):
        from app.agents.reflection import ReflectionAgent
        agent = ReflectionAgent()
        assert agent is not None

    def test_default_result(self):
        from app.agents.reflection import ReflectionAgent

        agent = ReflectionAgent()
        result = agent._default_result()
        assert result.overall_confidence == 0.0
        assert result.needs_revision is True
        assert "Reflection validation failed" in (result.revision_focus or "")
        assert len(result.hallucinated_claims) == 0


class TestReportAgent:
    def test_report_import(self):
        from app.agents.report import ReportAgent
        agent = ReportAgent()
        assert agent is not None

    def test_generate_stream_emits_chunks_and_citations(self):
        from app.agents.report import ReportAgent

        agent = ReportAgent()
        agent._client = MagicMock()
        agent._client.chat.completions.create.return_value = [
            MagicMock(choices=[MagicMock(delta=MagicMock(content="# Title\n"))]),
            MagicMock(choices=[MagicMock(delta=MagicMock(content="Body [citation:1]"))]),
        ]

        chunks = []
        citations = []
        report, report_citations = agent.generate_stream(
            user_query="Test topic",
            analysis="Test analysis",
            evidence_list=[
                Evidence(
                    content="Evidence body",
                    source_title="Source 1",
                    source_url="https://example.com",
                    source_type="web",
                    collected_by=AgentType.SEARCH,
                )
            ],
            reflection={"overall_confidence": 0.9},
            on_chunk=chunks.append,
            on_citation=citations.append,
        )

        assert report.startswith("# Title")
        assert len(chunks) >= 2
        assert len(citations) == 1
        assert len(report_citations) == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
