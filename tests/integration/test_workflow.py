"""
Integration tests for the complete research workflow.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.graph.state import StepStatus, create_initial_state


class TestResearchWorkflowIntegration:
    """Integration tests for the full research workflow."""

    @pytest.mark.asyncio
    async def test_workflow_initial_state(self):
        """Test that initial state is created correctly."""
        state = create_initial_state("Test query", "test-session")
        assert state["task_id"] == "test-session"
        assert state["user_query"] == "Test query"
        assert state["status"] == "pending"
        assert state["revision_count"] == 0
        assert state["tool_histories"] == []
        assert state["collected_evidence"] == []
        assert state["current_executing_nodes"] == []
        assert state["node_outcomes"] == []
        assert state["runtime_status"] == "pending"

    def test_collect_usage_metrics_reads_provider_usage(self):
        from app.llm_client import collect_usage_metrics

        response = SimpleNamespace(
            usage=SimpleNamespace(prompt_tokens=120, completion_tokens=30, total_tokens=150),
        )
        usage = collect_usage_metrics(
            response=response,
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "hello"}],
            completion_text="world",
        )

        assert usage["prompt_tokens"] == 120
        assert usage["completion_tokens"] == 30
        assert usage["total_tokens"] == 150
        assert usage["estimated"] is False

    def test_collect_usage_metrics_falls_back_to_estimate(self):
        from app.llm_client import collect_usage_metrics

        usage = collect_usage_metrics(
            response=None,
            model="unknown-model",
            messages=[{"role": "user", "content": "hello world"}],
            completion_text="response text",
        )

        assert usage["total_tokens"] > 0
        assert usage["estimated"] is True

    @pytest.mark.asyncio
    async def test_workflow_with_mocked_llm(
        self,
        sample_state: dict,
        mock_llm_client: MagicMock,
    ):
        """Test workflow execution with mocked LLM."""
        from app.graph.compiler import compile_research_graph

        # This test verifies the graph compiles and can be invoked
        graph = compile_research_graph()
        assert graph is not None
        assert hasattr(graph, "builder")

    def test_workflow_state_serialization(self, sample_plan_steps: list):
        """Test that plan steps can be serialized and deserialized."""
        from app.graph.state import PlanStep, deserialize_steps

        # Serialize
        steps = [PlanStep.model_validate(s) for s in sample_plan_steps]
        serialized = [s.model_dump() for s in steps]

        # Deserialize
        deserialized = deserialize_steps(serialized)
        assert len(deserialized) == 2
        assert deserialized[0].assigned_agent == "search"
        assert deserialized[1].assigned_agent == "browser"

    def test_plan_step_status_transitions(self):
        """Test that plan step status transitions work correctly."""
        from app.graph.state import PlanStep

        step = PlanStep(
            description="Test step",
            assigned_agent="search",
            target_query="test query",
        )
        assert step.status == StepStatus.PENDING

        step.status = StepStatus.RUNNING
        assert step.status == StepStatus.RUNNING

        step.status = StepStatus.DONE
        assert step.status == StepStatus.DONE

    def test_fact_lookup_does_not_force_no_evidence_reject(self):
        from app.guardrails import build_guardrail_decision

        decision = build_guardrail_decision("量子计算是什么")

        assert decision.intent == "fact_lookup"
        assert decision.reject_if_no_evidence is False


class TestAgentCollaboration:
    """Tests for agent-to-agent collaboration."""

    def test_planner_generates_plan(self, planner_agent: MagicMock):
        """Test that planner generates a valid plan."""
        # The planner should return PlanStep objects
        plan = planner_agent.create_plan("Test query")

        # Should have fallback plan (3 steps) since LLM is mocked
        assert len(plan) == 4
        agent_types = {step.get("assigned_agent") or step.get("node_type") for step in plan}
        assert "search" in agent_types
        assert "browser" in agent_types
        assert "rag" in agent_types
        assert "analyst" in agent_types

    def test_analyst_validates_evidence(self, analyst_agent: MagicMock):
        """Test that analyst can process evidence."""
        from app.graph.state import AgentType, Evidence

        evidence = [
            Evidence(
                content="Test evidence content",
                source_title="Test Source",
                source_url="https://test.com",
                source_type="web",
                collected_by=AgentType.SEARCH,
            )
        ]

        # Analyst should return analysis text
        analysis = analyst_agent.analyze("Test query", evidence)
        assert isinstance(analysis, str)
        assert len(analysis) > 0

    def test_reflection_detects_hallucinations(self, reflection_agent: MagicMock):
        """Test that reflection agent can detect hallucinations."""
        from app.graph.state import AgentType, Evidence

        evidence = [
            Evidence(
                content="Some evidence",
                source_title="Source",
                source_type="web",
                collected_by=AgentType.ANALYST,
            )
        ]

        # Should return a ReflectionResult
        result = reflection_agent.reflect("Test query", "Analysis text", evidence)
        assert result.overall_confidence == 0.0
        assert result.needs_revision is True


class TestSSEIntegration:
    """Tests for SSE streaming functionality."""

    @pytest.mark.asyncio
    async def test_sse_publish_and_stream(self, sse_manager):
        """Test SSE publish and stream."""
        session_id = "test-session-001"

        # Publish an event
        await sse_manager.publish(session_id, "agent_start", {"agent": "planner"})
        await sse_manager.publish(session_id, "done", {"status": "completed"})

        # Stream events
        events = []
        async for event in sse_manager.stream(session_id):
            events.append(event)
            if event["event"] == "done":
                break

        assert len(events) >= 1
        assert events[0]["event"] == "connected"
        assert any(event["event"] == "agent_start" for event in events)
        assert any(event["event"] == "done" for event in events)

    @pytest.mark.asyncio
    async def test_sse_wait_for_completion_accepts_workflow_error(self, sse_manager):
        """SSE completion waiter should stop on workflow_error."""
        session_id = "test-workflow-error-session"

        await sse_manager.publish(session_id, "agent_start", {"agent": "planner"})
        await sse_manager.publish(session_id, "workflow_error", {"error": "boom"})

        events = await sse_manager.wait_for_completion(session_id, timeout=1.0)

        assert any(event["event"] == "workflow_error" for event in events)

    @pytest.mark.asyncio
    async def test_sse_multiple_sessions(self, sse_manager):
        """Test SSE with multiple concurrent sessions."""
        sessions = ["session-1", "session-2", "session-3"]

        # Publish to each session
        for sid in sessions:
            await sse_manager.publish(sid, "agent_start", {"agent": f"agent-{sid}"})

        # Each should have independent queue
        for sid in sessions:
            queue = await sse_manager._get_queue(sid)
            assert queue is not None

    @pytest.mark.asyncio
    async def test_sse_history(self, sse_manager):
        """Test SSE event history."""
        session_id = "test-history-session"

        # Publish multiple events
        for i in range(5):
            await sse_manager.publish(session_id, f"event_{i}", {"index": i})

        # Get history
        history = sse_manager.get_history(session_id)
        assert len(history) == 5

    @pytest.mark.asyncio
    async def test_sse_replays_history_for_late_subscriber(self, sse_manager):
        """Late SSE subscribers should receive buffered history."""
        session_id = "test-late-session"

        await sse_manager.publish(session_id, "agent_start", {"agent": "planner"})
        await sse_manager.publish(session_id, "report_chunk", {"chunk": "hello"})

        events = []
        async for event in sse_manager.stream(session_id):
            events.append(event)
            if len(events) >= 3:
                break

        assert [event["event"] for event in events[:3]] == ["connected", "agent_start", "report_chunk"]

    @pytest.mark.asyncio
    async def test_sse_session_close(self, sse_manager):
        """Test SSE session cleanup."""
        session_id = "test-close-session"

        # Create session and publish
        await sse_manager.publish(session_id, "test", {})
        assert session_id in sse_manager._sessions

        # Close session
        sse_manager.close_session(session_id)
        assert session_id not in sse_manager._sessions


class TestResearchResultNormalization:
    """Tests for research result response normalization."""

    def test_normalize_tool_audit_rows_joins_outcomes(self):
        from app.api.research import _normalize_tool_audit_rows

        rows = _normalize_tool_audit_rows(
            session_id="test-session",
            tool_histories=[
                {
                    "agent_type": "search",
                    "tool_calls": [
                        {
                            "call_id": "call-1",
                            "tool_name": "duckduckgo_search",
                            "args": {"query": "openai"},
                            "status": "success",
                            "result_summary": "1 result",
                            "tokens_used": 12,
                            "cost_usd": 0.01,
                            "started_at": "2026-05-27T00:00:00",
                            "completed_at": "2026-05-27T00:00:01",
                        }
                    ],
                }
            ],
            node_outcomes=[
                {
                    "node_id": "n1",
                    "tool_call_id": "call-1",
                    "tool_name": "duckduckgo_search",
                    "status": "success",
                    "retry_count": 1,
                    "error_category": None,
                    "error_message": None,
                    "tokens_used": 12,
                    "cost_usd": 0.01,
                }
            ],
        )

        assert len(rows) == 1
        row = rows[0]
        assert row[0] == "call-1"
        assert row[1] == "test-session"
        assert row[2] == "n1"
        assert row[3] == "search"
        assert row[4] == "duckduckgo_search"
        assert row[8] is None
        assert row[10] == 1
        assert row[13] == 12
        assert row[14] == 0.01

    def test_iter_state_updates_unwraps_langgraph_node_chunks(self):
        from app.api.research import _iter_state_updates

        chunk = {
            "report": {
                "final_report": "report text",
                "status": "completed",
            }
        }

        assert list(_iter_state_updates(chunk)) == [
            {
                "final_report": "report text",
                "status": "completed",
            }
        ]

    def test_iter_state_updates_keeps_plain_state_chunks(self):
        from app.api.research import _iter_state_updates

        chunk = {
            "final_report": "report text",
            "status": "completed",
        }

        assert list(_iter_state_updates(chunk)) == [chunk]

    @pytest.mark.asyncio
    async def test_research_result_normalizes_citations_shape(self):
        from app.api.research import get_research_result

        row = {
            "id": "test-session",
            "user_query": "Test query",
            "status": "completed",
            "final_report": "report",
            "citations": {"citation_id": "citation:1", "source_url": "https://example.com"},
            "agent_trace": [],
            "review_status": {
                "runtime_status": "completed",
                "tool_audit_summary": {"total_calls": 3, "error_calls": 1},
            },
            "created_at": SimpleNamespace(isoformat=lambda: "2026-05-23T00:00:00"),
            "completed_at": None,
        }

        class FakeConn:
            async def fetchrow(self, *args, **kwargs):
                return row

            async def fetch(self, *args, **kwargs):
                return []

            def acquire(self):
                return self

            async def __aenter__(self):
                return self

            async def __aexit__(self, exc_type, exc, tb):
                return None

        class FakePool:
            def acquire(self):
                return FakeConn()

        with patch("app.api.research.get_db_pool", AsyncMock(return_value=FakePool())):
            response = await get_research_result("test-session")

        assert response["citations"] == [{"citation_id": "citation:1", "source_url": "https://example.com"}]
        assert response["runtime_status"] == "completed"
        assert response["tool_audit_summary"] == {"total_calls": 3, "error_calls": 1}

    @pytest.mark.asyncio
    async def test_get_research_tool_calls_returns_persisted_audit_rows(self):
        from app.api.research import get_research_tool_calls

        session_row = {"id": "test-session"}
        tool_rows = [
            {
                "call_id": "call-1",
                "session_id": "test-session",
                "node_id": "n1",
                "agent_type": "search",
                "tool_name": "duckduckgo_search",
                "args_json": {"query": "openai"},
                "args_hash": "abc",
                "status": "success",
                "error_category": None,
                "error_message": None,
                "retry_count": 1,
                "result_summary": "1 result",
                "result_hash": "def",
                "tokens_used": 20,
                "cost_usd": 0.02,
                "decision_id": None,
                "approved_by": None,
                "server_fingerprint": None,
                "safety_json": {"source": "harness_tool_layer", "audited": True},
                "usage_source": "provider",
                "estimated": False,
                "started_at": SimpleNamespace(isoformat=lambda: "2026-05-27T00:00:00"),
                "completed_at": SimpleNamespace(isoformat=lambda: "2026-05-27T00:00:01"),
                "created_at": SimpleNamespace(isoformat=lambda: "2026-05-27T00:00:02"),
            }
        ]

        class FakeConn:
            async def fetchrow(self, query, *args, **kwargs):
                if "FROM research_sessions" in query:
                    return session_row
                return None

            async def fetch(self, *args, **kwargs):
                return tool_rows

            def acquire(self):
                return self

            async def __aenter__(self):
                return self

            async def __aexit__(self, exc_type, exc, tb):
                return None

        class FakePool:
            def acquire(self):
                return FakeConn()

        with patch("app.api.research.get_db_pool", AsyncMock(return_value=FakePool())):
            response = await get_research_tool_calls("test-session")

        assert len(response) == 1
        assert response[0].call_id == "call-1"
        assert response[0].node_id == "n1"
        assert response[0].tool_name == "duckduckgo_search"
        assert response[0].args_json == {"query": "openai"}
        assert response[0].retry_count == 1
        assert response[0].safety_json == {"source": "harness_tool_layer", "audited": True}
        assert response[0].usage_source == "provider"
        assert response[0].estimated is False


class TestGraphCompilation:
    """Tests for graph compilation and structure."""

    def test_graph_has_all_nodes(self, research_graph):
        """Test that compiled graph has all required nodes."""
        nodes = set(research_graph.builder.nodes)
        required_nodes = {
            "planner", "dag_executor", "dag_aggregator",
            "search", "browser", "rag",
            "analyst", "reflection", "replan", "report",
        }
        assert required_nodes.issubset(nodes)

    def test_graph_edges_exist(self, research_graph):
        """Test that expected edges exist in the graph."""
        edges = []
        for start, end in research_graph.builder.edges:
            edges.append((start, end))

        # Should have edges from START to planner
        assert any(start == "__start__" and end == "planner" for start, end in edges)

        # Should have planner to dag executor
        assert any(start == "planner" and end == "dag_executor" for start, end in edges)

        # Should have edges from sub-agents to dag aggregator
        for agent in ["search", "browser", "rag"]:
            assert any(start == agent and end == "dag_aggregator" for start, end in edges)

        # Should have edge from analyst to reflection
        assert any(start == "analyst" and end == "reflection" for start, end in edges)

        # Should have replan loop back into dag executor
        assert any(start == "replan" and end == "dag_executor" for start, end in edges)

        # Should have edge from report to END
        assert any(start == "report" and end == "__end__" for start, end in edges)

    def test_graph_compiles_without_error(self):
        """Test that graph compiles successfully."""
        from app.graph.compiler import compile_research_graph
        # Should not raise
        graph = compile_research_graph()
        assert graph is not None


class TestMetricsCollectors:
    """Tests for metrics collectors."""

    def test_langgraph_metrics_collector(self):
        """Test LangGraph metrics collector."""
        from metrics.langgraph_workflow.collector import LangGraphMetricsCollector

        collector = LangGraphMetricsCollector(storage_path="metrics/test_langgraph/data")
        collector.record_workflow_start("session-1", "Test query")
        collector.record_node_start("session-1", "planner")
        collector.record_node_end("session-1", "planner", status="completed")
        collector.record_workflow_end("session-1", status="completed")

        metrics = collector.get_metrics()
        assert metrics["summary"]["total_workflows"] == 1
        assert metrics["summary"]["completed"] == 1

    def test_sse_metrics_collector(self):
        """Test SSE metrics collector."""
        from metrics.observability.collector import ObservabilityMetricsCollector

        collector = ObservabilityMetricsCollector(storage_path="metrics/test_sse/data")
        collector.record_connection_start("session-1")
        collector.record_event_publish("session-1", "thought", 100)
        collector.record_event_publish("session-1", "tool_call", 50)
        collector.record_connection_end("session-1", events_sent=2, bytes_sent=150)

        metrics = collector.get_metrics()
        assert metrics["summary"]["total_connections"] == 1
        assert metrics["summary"]["total_events_published"] == 2

    def test_research_quality_metrics_collector(self):
        """Test research quality metrics collector."""
        from metrics.research_quality.collector import ResearchQualityCollector

        collector = ResearchQualityCollector(storage_path="metrics/test_quality/data")
        collector.record_report_quality(
            session_id="session-1",
            query="Test query",
            total_claims=10,
            verified_claims=9,
            hallucinated_claims=1,
            citation_count=8,
            plan_coverage=0.9,
            avg_confidence=0.85,
        )

        metrics = collector.get_metrics()
        assert metrics["summary"]["total_reports"] == 1

    def test_rag_evaluation_metrics_collector(self):
        """Test layered RAG evaluation metrics collector."""
        from metrics.rag_evaluation.collector import RAGEvaluationCollector

        collector = RAGEvaluationCollector(storage_path="metrics/test_rag_eval/data")
        collector.record_retrieval_eval(
            question_id="q1",
            query="What is RAG evaluation?",
            expected_chunk_ids=["chunk-2"],
            retrieved_chunk_ids=["chunk-9", "chunk-2", "chunk-4"],
            top_k=3,
        )
        collector.record_generation_eval(
            question_id="q1",
            query="What is RAG evaluation?",
            answer="RAG evaluation should separate retrieval from generation.",
            faithfulness=0.92,
            answer_relevancy=0.89,
            context_recall=0.81,
            context_precision=0.78,
            judge_model="gpt-4o-mini",
        )

        metrics = collector.get_metrics()
        assert metrics["retrieval"]["summary"]["total_queries"] == 1
        assert metrics["retrieval"]["summary"]["hit_rate"] == 1.0
        assert metrics["retrieval"]["summary"]["mrr"] == 0.5
        assert metrics["generation"]["summary"]["total_answers"] == 1
        assert metrics["generation"]["summary"]["faithfulness_avg"] == 0.92
        assert metrics["generation"]["summary"]["answer_relevancy_avg"] == 0.89

    def test_rag_offline_evaluator_uses_real_retrieval_interface(self):
        """Test offline evaluator against a retrieval runner interface."""
        from types import SimpleNamespace

        from app.rag.evaluation import RAGEvalExample, RAGOfflineEvaluator
        from metrics.rag_evaluation.collector import RAGEvaluationCollector

        class FakeRetrievalRunner:
            def execute_retrieval(self, query: str, context: str = "", group: str | None = None):
                return [
                    SimpleNamespace(chunk_id="chunk-9", content="irrelevant"),
                    SimpleNamespace(chunk_id="chunk-2", content="relevant"),
                ]

        collector = RAGEvaluationCollector(storage_path="metrics/test_rag_eval_runner/data")
        evaluator = RAGOfflineEvaluator(
            retrieval_runner=FakeRetrievalRunner(),
            collector=collector,
        )
        metrics = evaluator.evaluate_dataset(
            [
                RAGEvalExample(
                    question_id="q1",
                    query="How do we evaluate RAG?",
                    expected_chunk_ids=["chunk-2"],
                    top_k=5,
                )
            ],
            run_generation_eval=False,
        )

        assert metrics["retrieval"]["summary"]["total_queries"] == 1
        assert metrics["retrieval"]["summary"]["hit_rate"] == 1.0
        assert metrics["retrieval"]["summary"]["mrr"] == 0.5

    def test_rag_offline_evaluator_generation_uses_generated_answer_and_context_metrics(self):
        """Test generation evaluation uses the system-generated answer and computes context metrics."""
        from types import SimpleNamespace

        from app.rag.evaluation import RAGEvalExample, RAGOfflineEvaluator
        from metrics.rag_evaluation.collector import RAGEvaluationCollector

        class FakeRetrievalRunner:
            def execute_retrieval(self, query: str, context: str = "", group: str | None = None):
                return [
                    SimpleNamespace(chunk_id="chunk-1", content="ctx1", metadata={"title": "Doc1"}),
                    SimpleNamespace(chunk_id="chunk-3", content="ctx3", metadata={"title": "Doc3"}),
                    SimpleNamespace(chunk_id="chunk-2", content="ctx2", metadata={"title": "Doc2"}),
                ]

        class FakeAnswerGenerator:
            def generate_answer(self, query: str, retrieved_results: list[SimpleNamespace]) -> str:
                return f"generated answer for {query} with {len(retrieved_results)} contexts"

        class FakeEvaluator(RAGOfflineEvaluator):
            def _judge_generation(self, *, query: str, retrieved_contexts: list[str], answer: str):
                assert answer.startswith("generated answer")
                return {
                    "faithfulness": 0.91,
                    "answer_relevancy": 0.88,
                    "context_recall": None,
                    "context_precision": None,
                }

        collector = RAGEvaluationCollector(storage_path="metrics/test_rag_eval_generation/data")
        evaluator = FakeEvaluator(
            retrieval_runner=FakeRetrievalRunner(),
            answer_generator=FakeAnswerGenerator(),
            collector=collector,
        )
        metrics = evaluator.evaluate_dataset(
            [
                RAGEvalExample(
                    question_id="q2",
                    query="How is RAG graded?",
                    expected_chunk_ids=["chunk-1", "chunk-2"],
                    ground_truth_answer="gold answer",
                    top_k=3,
                )
            ],
            run_generation_eval=True,
        )

        summary = metrics["generation"]["summary"]
        assert summary["total_answers"] == 1
        assert summary["faithfulness_avg"] == 0.91
        assert summary["answer_relevancy_avg"] == 0.88
        assert summary["context_recall_avg"] == 1.0
        assert round(summary["context_precision_avg"], 4) == round((1.0 + (2 / 3)) / 2, 4)

    def test_rag_retrieval_summary_reports_recall_at_k(self):
        """Retrieval summary exposes recall@k alongside hit_rate / mrr."""
        from metrics.rag_evaluation.collector import RAGEvaluationCollector

        collector = RAGEvaluationCollector(storage_path="metrics/test_rag_eval/data")
        # 1 of 2 gold chunks retrieved within top_k -> recall 0.5, hit 1.0.
        collector.record_retrieval_eval(
            question_id="q1",
            query="multi gold query",
            expected_chunk_ids=["chunk-1", "chunk-2"],
            retrieved_chunk_ids=["chunk-9", "chunk-2", "chunk-4"],
            top_k=3,
        )
        summary = collector.get_metrics()["retrieval"]["summary"]
        assert summary["hit_rate"] == 1.0
        assert summary["recall_at_k"] == 0.5

    def test_rag_retrieval_groups_by_query_type_and_traces_failures(self):
        """Per-query-type breakdown isolates weak categories; failures are traced."""
        from metrics.rag_evaluation.collector import RAGEvaluationCollector

        collector = RAGEvaluationCollector(storage_path="metrics/test_rag_eval/data")
        # synonym query hits at rank 1
        collector.record_retrieval_eval(
            question_id="q1",
            query="差旅报销限额是多少",
            expected_chunk_ids=["EXP-03"],
            retrieved_chunk_ids=["EXP-03", "EXP-09"],
            top_k=10,
            query_type="synonym",
        )
        # numeric_code query misses entirely -> failure
        collector.record_retrieval_eval(
            question_id="q2",
            query="ORD20260418 为什么没发货",
            expected_chunk_ids=["ORD-22"],
            retrieved_chunk_ids=["ORD-01", "ORD-07"],
            top_k=10,
            query_type="numeric_code",
        )

        retrieval = collector.get_metrics()["retrieval"]
        by_type = retrieval["by_query_type"]

        assert by_type["synonym"]["hit_rate"] == 1.0
        assert by_type["synonym"]["mrr"] == 1.0
        assert by_type["numeric_code"]["hit_rate"] == 0.0
        assert by_type["numeric_code"]["recall_at_k"] == 0.0

        failures = retrieval["failures"]
        assert len(failures) == 1
        assert failures[0]["question_id"] == "q2"
        assert failures[0]["query_type"] == "numeric_code"

    def test_rag_offline_evaluator_threads_query_type(self):
        """The evaluator carries each example's query_type into per-type metrics."""
        from types import SimpleNamespace

        from app.rag.evaluation import RAGEvalExample, RAGOfflineEvaluator
        from metrics.rag_evaluation.collector import RAGEvaluationCollector

        class FakeRetrievalRunner:
            def execute_retrieval(self, query: str, context: str = "", group: str | None = None):
                return [SimpleNamespace(chunk_id="chunk-1", content="ctx")]

        collector = RAGEvaluationCollector(storage_path="metrics/test_rag_eval_runner/data")
        evaluator = RAGOfflineEvaluator(
            retrieval_runner=FakeRetrievalRunner(),
            collector=collector,
        )
        metrics = evaluator.evaluate_dataset(
            [
                RAGEvalExample(
                    question_id="q1",
                    query="缩写术语 query",
                    expected_chunk_ids=["chunk-1"],
                    top_k=10,
                    query_type="abbreviation",
                )
            ],
            run_generation_eval=False,
        )
        assert "abbreviation" in metrics["retrieval"]["by_query_type"]
        assert metrics["retrieval"]["by_query_type"]["abbreviation"]["hit_rate"] == 1.0


class TestEdgeRouting:
    """Tests for edge routing logic."""

    def test_should_revise_under_limit(self):
        """Test revision routing when under limit."""
        from app.graph.compiler import should_revise

        state = {
            "revision_needed": True,
            "revision_count": 2,
        }
        assert should_revise(state) == "replan"

    def test_should_revise_at_limit(self):
        """Test revision routing when at limit."""
        from app.graph.compiler import should_revise

        state = {
            "revision_needed": True,
            "revision_count": 3,
        }
        assert should_revise(state) == "generate_report"

    def test_should_not_revise(self):
        """Test no revision when not needed."""
        from app.graph.compiler import should_revise

        state = {
            "revision_needed": False,
            "revision_count": 0,
        }
        assert should_revise(state) == "generate_report"


class TestRRFMatrices:
    """Tests for RRF fusion matrix calculations."""

    def test_rrf_single_list(self):
        """Test RRF with single result list."""
        from app.rag.retriever import HybridRetriever

        retriever = HybridRetriever()
        result = retriever.reciprocal_rank_fusion([
            [("doc1", 1.0), ("doc2", 0.8), ("doc3", 0.6)],
        ])

        assert len(result) == 3
        assert result[0][0] == "doc1"
        assert result[1][0] == "doc2"
        assert result[2][0] == "doc3"

    def test_rrf_multiple_lists(self):
        """Test RRF with multiple result lists."""
        from app.rag.retriever import HybridRetriever

        retriever = HybridRetriever()
        list1 = [("doc1", 1.0), ("doc2", 0.8)]
        list2 = [("doc2", 1.0), ("doc1", 0.7)]

        result = retriever.reciprocal_rank_fusion([list1, list2], k=60)
        result_dict = {doc_id: score for doc_id, score in result}

        assert "doc1" in result_dict
        assert "doc2" in result_dict
        # Both docs should have higher scores due to appearing in both lists
        assert result_dict["doc1"] > 0
        assert result_dict["doc2"] > 0

    def test_rrf_empty_input(self):
        """Test RRF with empty input."""
        from app.rag.retriever import HybridRetriever

        retriever = HybridRetriever()
        result = retriever.reciprocal_rank_fusion([])
        assert result == []

    def test_rrf_different_k_values(self):
        """Test that different k values affect scores."""
        from app.rag.retriever import HybridRetriever

        retriever = HybridRetriever()
        list1 = [("doc1", 1.0), ("doc2", 0.8)]

        result_low_k = retriever.reciprocal_rank_fusion([list1], k=1)
        result_high_k = retriever.reciprocal_rank_fusion([list1], k=100)

        # With higher k, the relative difference between ranks is smaller
        ratio_low = result_low_k[0][1] / result_low_k[1][1]
        ratio_high = result_high_k[0][1] / result_high_k[1][1]

        assert ratio_low > ratio_high


class TestAPIEndpoints:
    """Tests for FastAPI endpoints."""

    @pytest.mark.asyncio
    async def test_health_endpoint(self):
        """Test health check endpoint."""
        from app.main import app

        # This would need the app to be properly configured
        # For now, just verify the app exists
        assert app is not None
        assert app.title == "Agentic Deep Research System"

    def test_research_request_validation(self):
        """Test ResearchRequest model validation."""
        from app.api.research import ResearchRequest

        # Valid request
        req = ResearchRequest(query="Test query")
        assert req.query == "Test query"
        assert req.session_id is None
        assert req.max_revision == 3

        # Query too short
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            ResearchRequest(query="Hi")  # min_length=5

    def test_research_status_model(self):
        """Test ResearchStatus model."""
        from app.api.research import ResearchStatus

        status = ResearchStatus(
            session_id="test-123",
            status="running",
            created_at="2026-01-01T00:00:00",
            runtime_status="running",
        )
        assert status.session_id == "test-123"
        assert status.status == "running"
        assert status.runtime_status == "running"

    @pytest.mark.asyncio
    async def test_research_status_includes_runtime_fields(self):
        from app.api.research import get_research_status

        row = {
            "id": "test-session",
            "user_query": "Test query",
            "status": "running",
            "review_status": {
                "runtime_status": "retryable_failed",
                "requires_confirmation": False,
                "pending_approval_count": 1,
                "budget_state": {"budget_profile": "medium"},
                "last_error_category": "timeout",
            },
            "max_total_tokens": 60000,
            "max_cost_usd": 0.5,
            "max_tool_calls": 12,
            "max_wall_clock_seconds": 420,
            "used_total_tokens": 140,
            "used_cost_usd": 0.0012,
            "used_tool_calls": 2,
            "elapsed_wall_clock_seconds": 8,
            "hard_stop_reason": None,
            "created_at": SimpleNamespace(isoformat=lambda: "2026-05-27T00:00:00"),
            "updated_at": SimpleNamespace(isoformat=lambda: "2026-05-27T00:01:00"),
            "completed_at": None,
        }

        class FakeConn:
            async def fetchrow(self, *args, **kwargs):
                return row

            def acquire(self):
                return self

            async def __aenter__(self):
                return self

            async def __aexit__(self, exc_type, exc, tb):
                return None

        class FakePool:
            def acquire(self):
                return FakeConn()

        with patch("app.api.research.get_db_pool", AsyncMock(return_value=FakePool())):
            response = await get_research_status("test-session")

        assert response.runtime_status == "retryable_failed"
        assert response.pending_approval_count == 1
        assert response.last_error_category == "timeout"
        assert response.budget_state["max_tool_calls"] == 12
        assert response.budget_state["used_total_tokens"] == 140

    @pytest.mark.asyncio
    async def test_create_research_returns_session_budget_thresholds(self):
        from app.api.research import ResearchRequest, create_research

        class FakeConn:
            async def execute(self, *args, **kwargs):
                return "OK"

            def acquire(self):
                return self

            async def __aenter__(self):
                return self

            async def __aexit__(self, exc_type, exc, tb):
                return None

        class FakePool:
            def acquire(self):
                return FakeConn()

        background_tasks = SimpleNamespace(tasks=[], add_task=lambda *args, **kwargs: None)
        fake_registry = SimpleNamespace(
            resolve_for_session=AsyncMock(return_value=SimpleNamespace(as_dict=lambda: {
                "effective_tool_allowlist": ["search", "rag"],
                "effective_skill_ids": [],
                "resolved_skills": [],
                "effective_prompt_sections": {},
                "effective_agent_hints": {},
            }))
        )

        with patch("app.api.research.get_db_pool", AsyncMock(return_value=FakePool())), \
                patch("app.api.research.get_skill_registry", return_value=fake_registry):
            response = await create_research(
                ResearchRequest(query="请研究中国 AI Agent 市场格局"),
                background_tasks,
            )

        assert response.status == "running"
        assert response.budget["max_tool_calls"] == 12
        assert response.budget["max_wall_clock_seconds"] == 420
        assert response.skill_context["effective_tool_allowlist"] == ["search", "rag"]

    @pytest.mark.asyncio
    async def test_create_research_passes_skill_scope_and_overrides_to_registry(self):
        from app.api.research import ResearchRequest, create_research

        class FakeConn:
            async def execute(self, *args, **kwargs):
                return "OK"

            def acquire(self):
                return self

            async def __aenter__(self):
                return self

            async def __aexit__(self, exc_type, exc, tb):
                return None

        class FakePool:
            def acquire(self):
                return FakeConn()

        background_tasks = SimpleNamespace(tasks=[], add_task=lambda *args, **kwargs: None)
        fake_registry = SimpleNamespace(
            resolve_for_session=AsyncMock(return_value=SimpleNamespace(as_dict=lambda: {
                "effective_tool_allowlist": ["rag"],
                "effective_skill_ids": ["skill-1"],
                "resolved_skills": [],
                "effective_prompt_sections": {"prompt": "Use internal finance heuristics."},
                "effective_agent_hints": {"planner": ["Prefer earnings materials."]},
            }))
        )

        request = ResearchRequest(
            query="请研究中国 AI Agent 产业链中的财报与融资情况",
            enabled_skill_ids=["skill-1"],
            disabled_skill_ids=["skill-2"],
            skill_tenant_id="tenant-a",
            skill_project_id="project-a",
        )

        with patch("app.api.research.get_db_pool", AsyncMock(return_value=FakePool())), \
                patch("app.api.research.get_skill_registry", return_value=fake_registry):
            response = await create_research(request, background_tasks)

        fake_registry.resolve_for_session.assert_awaited_once_with(
            query=request.query,
            manually_enabled_skill_ids=["skill-1"],
            manually_disabled_skill_ids=["skill-2"],
            tenant_id="tenant-a",
            project_id="project-a",
        )
        assert response.status == "running"
        assert response.skill_context["effective_skill_ids"] == ["skill-1"]
        assert response.skill_context["effective_tool_allowlist"] == ["rag"]
