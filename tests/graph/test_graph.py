"""
Tests for the research graph workflow.
"""

from datetime import timedelta

import pytest

from app.core.time import utc_now_naive
from app.graph.compiler import compile_research_graph, should_revise
from app.graph.state import PlanStep, StepStatus, create_initial_state


class TestResearchState:
    def test_create_initial_state(self):
        state = create_initial_state("测试研究查询", "test-123")
        assert state["task_id"] == "test-123"
        assert state["user_query"] == "测试研究查询"
        assert state["status"] == "pending"
        assert state["revision_count"] == 0
        assert state["search_results"] == []

    def test_research_state_has_required_keys(self):
        state = create_initial_state("query")
        required_keys = {
            "task_id", "user_query", "created_at", "status", "session",
            "dag", "current_executing_nodes", "completed_nodes",
            "node_outcomes",
            "tool_histories", "collected_evidence", "verification",
            "search_results", "browser_results", "rag_results", "aggregated_evidence",
            "revision_needed", "revision_count", "analysis",
            "final_report", "citations", "guardrail_decision", "evidence_status",
            "review_status", "failure_memory", "user_confirmed", "allow_web_after_rag_hit", "rag_group",
            "retrieval_policy", "runtime_status", "budget_state", "pending_approvals", "output_length",
            "skill_context",
            "agent_trace", "guardrail_trace", "errors"
        }
        assert set(state.keys()) == required_keys


class TestPlanStep:
    def test_plan_step_defaults(self):
        step = PlanStep(
            description="搜索市场数据",
            assigned_agent="search",
            target_query="2025年中国AI市场"
        )
        assert step.status == StepStatus.PENDING
        assert step.retry_count == 0
        assert step.evidence_ids == []
        assert len(step.step_id) == 7

    def test_plan_step_serialization(self):
        step = PlanStep(
            description="深度阅读文章",
            assigned_agent="browser",
            target_query="AI发展趋势"
        )
        data = step.model_dump()
        assert isinstance(data, dict)
        assert data["description"] == "深度阅读文章"
        assert data["assigned_agent"] == "browser"


class TestConditionalRouting:
    def test_should_revise_under_limit(self):
        state = {
            "revision_needed": True,
            "revision_count": 2,
        }
        assert should_revise(state) == "replan"

    def test_should_revise_at_limit(self):
        state = {
            "revision_needed": True,
            "revision_count": 3,
        }
        assert should_revise(state) == "generate_report"

    def test_should_not_revise(self):
        state = {
            "revision_needed": False,
            "revision_count": 0,
        }
        assert should_revise(state) == "generate_report"


class TestGraphCompilation:
    def test_compile_research_graph(self):
        graph = compile_research_graph()
        assert graph is not None
        # Verify it's a compiled graph
        assert hasattr(graph, "builder")
        assert hasattr(graph, "config")

    def test_graph_has_required_nodes(self):
        graph = compile_research_graph()
        nodes = set(graph.builder.nodes)
        required_nodes = {
            "planner", "search", "browser", "rag", "mcp", "approval_reviewer",
            "analyst", "reflection", "report"
        }
        assert required_nodes.issubset(nodes)

    def test_search_node_returns_tool_history(self):
        from unittest.mock import patch

        from app.graph.compiler import search_node
        from app.graph.state import (
            AgentType,
            DAGDefinition,
            PlanNode,
            SearchResult,
            serialize_dag,
        )

        node = PlanNode(node_type="search", query="test query")
        state = create_initial_state("test query", "session-1")
        state["dag"] = serialize_dag(DAGDefinition(dag_name="test", nodes=[node], edges=[]))
        state["executing_nodes"] = [node.node_id]

        with patch("app.agents.search.SearchAgent.execute_search", return_value=[
            SearchResult(
                url="https://example.com",
                title="Example",
                snippet="Example snippet",
                relevance_score=0.9,
            )
        ]):
            result = search_node(state)

        assert len(result["tool_histories"]) == 1
        history = result["tool_histories"][0]
        assert history["agent_type"] == AgentType.SEARCH.value
        assert history["tool_calls"][0]["status"] == "success"
        assert "dag" not in result

    def test_search_node_handles_tool_error(self):
        from unittest.mock import patch

        from app.graph.compiler import search_node
        from app.graph.state import (
            DAGDefinition,
            PlanNode,
            serialize_dag,
        )

        node = PlanNode(node_type="search", query="test query")
        state = create_initial_state("test query", "session-1")
        state["dag"] = serialize_dag(DAGDefinition(dag_name="test", nodes=[node], edges=[]))
        state["executing_nodes"] = [node.node_id]

        with patch("app.agents.search.SearchAgent.execute_search", side_effect=RuntimeError("boom")):
            result = search_node(state)

        assert len(result["tool_histories"]) == 1
        assert result["tool_histories"][0]["tool_calls"][0]["status"] == "error"
        assert result["node_outcomes"][0]["status"] == "retryable_error"
        assert result["node_outcomes"][0]["error_category"] == "temporary_db"

    def test_search_node_returns_tool_trace_events(self):
        from unittest.mock import patch

        from app.graph.compiler import search_node
        from app.graph.state import (
            DAGDefinition,
            PlanNode,
            serialize_dag,
        )

        node = PlanNode(node_type="search", query="test query")
        state = create_initial_state("test query", "session-1")
        state["dag"] = serialize_dag(DAGDefinition(dag_name="test", nodes=[node], edges=[]))
        state["executing_nodes"] = [node.node_id]

        with patch("app.agents.search.SearchAgent.execute_search", return_value=[]):
            result = search_node(state)

        event_types = [event["event_type"] for event in result["agent_trace"]]
        assert "tool_start" in event_types
        assert "tool_error" in event_types

    def test_browser_node_treats_fallback_error_as_retryable_failure(self):
        from unittest.mock import patch

        from app.graph.compiler import browser_node
        from app.graph.state import (
            BrowserResult,
            DAGDefinition,
            PlanNode,
            serialize_dag,
        )

        node = PlanNode(node_type="browser", query="https://example.com")
        state = create_initial_state("test query", "session-1")
        state["dag"] = serialize_dag(DAGDefinition(dag_name="test", nodes=[node], edges=[]))
        state["executing_nodes"] = [node.node_id]

        with patch("app.agents.browser.BrowserAgent.execute_browse", return_value=[
            BrowserResult(
                url="https://example.com",
                title="https://example.com",
                extracted_content="",
                citations=[],
                extraction_level="snippet",
                citation="https://example.com",
                error_message="navigation timeout",
            )
        ]):
            result = browser_node(state)

        assert result["tool_histories"][0]["tool_calls"][0]["status"] == "error"
        assert result["node_outcomes"][0]["status"] == "retryable_error"
        assert result["node_outcomes"][0]["error_category"] == "timeout"

    def test_dag_aggregator_keeps_failed_node_failed(self):
        from app.graph.compiler import dag_results_aggregator
        from app.graph.state import (
            DAGDefinition,
            PlanNode,
            RuntimeStatus,
            deserialize_dag,
            serialize_dag,
        )

        node = PlanNode(node_id="s1", node_type="search", query="test query")
        state = create_initial_state("test query", "session-1")
        state["dag"] = serialize_dag(DAGDefinition(dag_name="test", nodes=[node], edges=[]))
        state["current_executing_nodes"] = ["s1"]
        state["node_outcomes"] = [{
            "node_id": "s1",
            "tool_call_id": "call-1",
            "tool_name": "duckduckgo_search",
            "status": "retryable_error",
            "error_category": "timeout",
            "error_message": "request timeout",
            "retry_count": 1,
            "tokens_used": 0,
            "cost_usd": 0.0,
            "result_count": 0,
            "approval_request_id": None,
        }]

        result = dag_results_aggregator(state)
        planned = deserialize_dag(result["dag"])
        failed = next(node for node in planned.nodes if node.node_id == "s1")

        assert failed.status == StepStatus.FAILED
        assert "s1" not in result["completed_nodes"]
        assert result["runtime_status"] == RuntimeStatus.RETRYABLE_FAILED.value

    def test_dag_executor_requeues_retryable_failed_node_under_limit(self):
        from app.graph.compiler import dag_executor_node
        from app.graph.state import (
            DAGDefinition,
            PlanNode,
            deserialize_dag,
            serialize_dag,
        )

        node = PlanNode(node_id="s1", node_type="search", query="retry query")
        node.status = StepStatus.FAILED
        node.retry_count = 1
        node.last_error_category = "timeout"
        state = create_initial_state("test query", "session-1")
        state["dag"] = serialize_dag(DAGDefinition(dag_name="test", nodes=[node], edges=[]))
        state["budget_state"] = {"max_retries_per_tool": 2}

        result = dag_executor_node(state)
        planned = deserialize_dag(result["dag"])
        retried = next(item for item in planned.nodes if item.node_id == "s1")

        assert retried.status == StepStatus.PENDING
        assert result["current_executing_nodes"] == ["s1"]

    def test_dag_executor_defers_retry_until_backoff_expires(self):
        from app.graph.compiler import dag_executor_node
        from app.graph.state import DAGDefinition, PlanNode, serialize_dag

        node = PlanNode(node_id="s1", node_type="search", query="retry query")
        node.status = StepStatus.FAILED
        node.retry_count = 1
        node.last_error_category = "timeout"
        state = create_initial_state("test query", "session-1")
        state["dag"] = serialize_dag(DAGDefinition(dag_name="test", nodes=[node], edges=[]))
        state["budget_state"] = {"max_retries_per_tool": 2}
        state["failure_memory"] = {
            "records": [{
                "node_type": "search",
                "query": "retry query",
                "last_error_category": "timeout",
                "retry_count": 1,
                "repeat_blocked": False,
                "next_retry_at": (utc_now_naive() + timedelta(seconds=30)).isoformat(),
                "backoff_seconds": 30,
            }],
            "max_retries_per_tool": 2,
            "repeat_blocked_nodes": [],
        }

        result = dag_executor_node(state)

        assert result["current_executing_nodes"] == []
        assert result["agent_trace"][0]["agent"] == "retry_scheduler"

    def test_dag_executor_does_not_retry_terminal_or_exhausted_node(self):
        from app.graph.compiler import dag_executor_node
        from app.graph.state import (
            DAGDefinition,
            PlanNode,
            serialize_dag,
        )

        node = PlanNode(node_id="s1", node_type="search", query="bad query")
        node.status = StepStatus.FAILED
        node.retry_count = 2
        node.last_error_category = "schema_validation"
        node.terminal_failure = True
        state = create_initial_state("test query", "session-1")
        state["dag"] = serialize_dag(DAGDefinition(dag_name="test", nodes=[node], edges=[]))
        state["budget_state"] = {"max_retries_per_tool": 2}

        result = dag_executor_node(state)
        assert result["current_executing_nodes"] == []

    def test_dag_aggregator_builds_failure_memory(self):
        from app.graph.compiler import dag_results_aggregator
        from app.graph.state import DAGDefinition, PlanNode, serialize_dag

        node = PlanNode(node_id="s1", node_type="search", query="retry query")
        state = create_initial_state("test query", "session-1")
        state["dag"] = serialize_dag(DAGDefinition(dag_name="test", nodes=[node], edges=[]))
        state["current_executing_nodes"] = ["s1"]
        state["node_outcomes"] = [{
            "node_id": "s1",
            "tool_call_id": "call-1",
            "tool_name": "duckduckgo_search",
            "status": "retryable_error",
            "error_category": "timeout",
            "error_message": "request timeout",
            "retry_count": 1,
            "tokens_used": 0,
            "cost_usd": 0.0,
            "result_count": 0,
            "approval_request_id": None,
        }]
        state["budget_state"] = {"max_retries_per_tool": 2}

        result = dag_results_aggregator(state)
        memory = result["failure_memory"]

        assert memory["records"][0]["node_type"] == "search"
        assert memory["records"][0]["retry_count"] == 1
        assert memory["records"][0]["repeat_blocked"] is False
        assert memory["records"][0]["backoff_seconds"] == 2
        assert memory["records"][0]["next_retry_at"] is not None

    def test_should_continue_dag_blocks_on_pending_approval(self):
        from app.graph.compiler import should_continue_dag
        from app.graph.state import DAGDefinition, PlanNode, serialize_dag

        node = PlanNode(node_id="s1", node_type="search", query="test query")
        state = create_initial_state("test query", "session-1")
        state["dag"] = serialize_dag(DAGDefinition(dag_name="test", nodes=[node], edges=[]))
        state["pending_approvals"] = [{"approval_id": "ap-1"}]

        assert should_continue_dag(state) == "approval_reviewer"

    def test_should_continue_dag_keeps_retryable_backoff_node_in_loop(self):
        from app.graph.compiler import should_continue_dag
        from app.graph.state import DAGDefinition, PlanNode, serialize_dag

        node = PlanNode(node_id="s1", node_type="search", query="retry query")
        node.status = StepStatus.FAILED
        node.retry_count = 1
        node.last_error_category = "timeout"
        state = create_initial_state("test query", "session-1")
        state["dag"] = serialize_dag(DAGDefinition(dag_name="test", nodes=[node], edges=[]))
        state["failure_memory"] = {
            "records": [{
                "node_type": "search",
                "query": "retry query",
                "last_error_category": "timeout",
                "retry_count": 1,
                "repeat_blocked": False,
                "next_retry_at": (utc_now_naive() + timedelta(seconds=30)).isoformat(),
                "backoff_seconds": 30,
            }],
            "max_retries_per_tool": 2,
            "repeat_blocked_nodes": [],
        }

        assert should_continue_dag(state) == "continue"

    def test_approval_reviewer_sets_paused_runtime(self):
        from app.graph.compiler import approval_reviewer_node

        state = create_initial_state("test query", "session-1")
        state["pending_approvals"] = [{"approval_id": "ap-1", "status": "pending"}]

        result = approval_reviewer_node(state)
        assert result["runtime_status"] == "awaiting_approval"
        assert result["status"] == "paused"

    def test_replan_node_regenerates_dag_with_failure_hints(self):
        from unittest.mock import patch

        from app.graph.compiler import replan_node
        from app.graph.state import (
            DAGDefinition,
            PlanNode,
            deserialize_dag,
            serialize_dag,
        )

        old_node = PlanNode(node_id="s1", node_type="search", query="old query")
        new_node = PlanNode(node_id="s2", node_type="rag", query="new query")
        state = create_initial_state("test query", "session-1")
        state["dag"] = serialize_dag(DAGDefinition(dag_name="old", nodes=[old_node], edges=[]))
        state["failure_memory"] = {
            "records": [{
                "node_type": "search",
                "query": "old query",
                "last_error_category": "timeout",
                "retry_count": 2,
                "repeat_blocked": True,
            }],
            "max_retries_per_tool": 2,
            "repeat_blocked_nodes": ["s1"],
        }

        with patch("app.agents.planner.PlannerAgent.create_dag", return_value=DAGDefinition(dag_name="new", nodes=[new_node], edges=[])) as mock_create:
            result = replan_node(state)

        planned = deserialize_dag(result["dag"])
        assert planned.dag_name == "new"
        assert any(node.node_type == "rag" for node in planned.nodes)
        assert result["completed_nodes"] == []
        assert "Known failures:" in mock_create.call_args.kwargs["planning_hints"]

    @pytest.mark.asyncio
    async def test_mcp_node_blocks_untrusted_server(self):
        from unittest.mock import AsyncMock, patch

        from app.graph.compiler import mcp_node
        from app.graph.state import DAGDefinition, PlanNode, serialize_dag

        node = PlanNode(node_id="m1", node_type="mcp", query="unknown:write_tool")
        state = create_initial_state("test query", "session-1")
        state["dag"] = serialize_dag(DAGDefinition(dag_name="test", nodes=[node], edges=[]))
        state["executing_nodes"] = [node.node_id]

        class FakeConn:
            async def fetchrow(self, *args, **kwargs):
                return None

            def acquire(self):
                return self

            async def __aenter__(self):
                return self

            async def __aexit__(self, exc_type, exc, tb):
                return None

        class FakePool:
            def acquire(self):
                return FakeConn()

        with patch("app.governance.mcp.get_db_pool", AsyncMock(return_value=FakePool())):
            result = await mcp_node(state)

        assert result["node_outcomes"][0]["status"] == "terminal_error"

    def test_execute_tool_batch_includes_dag_payload(self):
        from app.graph.compiler import execute_tool_batch
        from app.graph.state import (
            DAGDefinition,
            PlanNode,
            serialize_dag,
        )

        node = PlanNode(node_type="search", query="test query")
        state = create_initial_state("test query", "session-1")
        state["dag"] = serialize_dag(DAGDefinition(dag_name="test", nodes=[node], edges=[]))
        state["current_executing_nodes"] = [node.node_id]
        state["guardrail_decision"] = {"enabled_tools": ["search"]}

        sends = execute_tool_batch(state)

        assert len(sends) == 1
        payload = sends[0].arg
        assert "dag" in payload
        assert payload["executing_nodes"] == [node.node_id]

    def test_execute_tool_batch_respects_skill_tool_allowlist(self):
        from app.graph.compiler import execute_tool_batch
        from app.graph.state import DAGDefinition, PlanNode, serialize_dag

        search = PlanNode(node_id="s1", node_type="search", query="search query")
        browser = PlanNode(node_id="b1", node_type="browser", query="browse query")
        rag = PlanNode(node_id="r1", node_type="rag", query="rag query")
        state = create_initial_state("test query", "session-1")
        state["dag"] = serialize_dag(DAGDefinition(dag_name="test", nodes=[search, browser, rag], edges=[]))
        state["current_executing_nodes"] = [search.node_id, browser.node_id, rag.node_id]
        state["guardrail_decision"] = {"enabled_tools": ["search", "browser", "rag"]}
        state["skill_context"] = {"effective_tool_allowlist": ["rag"]}

        sends = execute_tool_batch(state)

        assert len(sends) == 1
        assert sends[0].node == "rag"
        assert sends[0].arg["current_executing_nodes"] == ["r1"]

    def test_planner_node_passes_skill_prompt_and_hints(self):
        from unittest.mock import patch

        from app.graph.compiler import planner_node
        from app.graph.state import DAGDefinition, PlanNode

        search_node_only = PlanNode(node_id="s1", node_type="search", query="test query")
        dag = DAGDefinition(dag_name="test", nodes=[search_node_only], edges=[])
        state = create_initial_state("test query", "session-1")
        state["skill_context"] = {
            "effective_prompt_sections": {
                "overview": "Finance domain context.",
                "prompt": "Prioritize earnings materials.",
                "constraints": "Avoid browser unless filings are required.",
            },
            "effective_agent_hints": {
                "planner": ["Prefer earnings, guidance, and filings."],
            },
        }

        with patch("app.agents.planner.PlannerAgent.create_dag", return_value=dag) as mock_create:
            planner_node(state)

        assert mock_create.call_args.kwargs["skill_prompt"] == (
            "Overview:\nFinance domain context.\n\n"
            "Prompt:\nPrioritize earnings materials.\n\n"
            "Constraints:\nAvoid browser unless filings are required."
        )
        assert mock_create.call_args.kwargs["planner_hints"] == ["Prefer earnings, guidance, and filings."]

    def test_planner_enforces_internal_rag_before_search(self):
        from unittest.mock import patch

        from app.graph.compiler import planner_node
        from app.graph.state import (
            DAGDefinition,
            PlanNode,
            deserialize_dag,
        )

        search_node_only = PlanNode(node_id="s1", node_type="search", query="test query")
        dag = DAGDefinition(dag_name="test", nodes=[search_node_only], edges=[])
        state = create_initial_state("test query", "session-1")

        with patch("app.agents.planner.PlannerAgent.create_dag", return_value=dag):
            result = planner_node(state)

        planned = deserialize_dag(result["dag"])
        rag_nodes = [node for node in planned.nodes if node.node_type == "rag"]
        search_nodes = [node for node in planned.nodes if node.node_type == "search"]

        assert rag_nodes
        assert search_nodes[0].depends_on == [rag_nodes[0].node_id]

    def test_dag_aggregator_skips_web_after_rag_hit_by_default(self):
        from app.graph.compiler import dag_results_aggregator
        from app.graph.state import (
            DAGDefinition,
            PlanNode,
            deserialize_dag,
            serialize_dag,
        )

        rag = PlanNode(node_id="r1", node_type="rag", query="internal")
        search = PlanNode(node_id="s1", node_type="search", query="web", depends_on=["r1"])
        rag.result = {"results": [{"content": "internal evidence"}]}
        state = create_initial_state("test query", "session-1")
        state["dag"] = serialize_dag(DAGDefinition(dag_name="test", nodes=[rag, search], edges=[]))
        state["current_executing_nodes"] = ["r1"]
        state["retrieval_policy"] = {"allow_web_after_rag_hit": False}

        result = dag_results_aggregator(state)
        planned = deserialize_dag(result["dag"])
        skipped = next(node for node in planned.nodes if node.node_id == "s1")

        assert skipped.status == StepStatus.SKIPPED
        assert "s1" in result["completed_nodes"]
        assert result["retrieval_policy"]["web_search_required"] is False

    def test_dag_aggregator_allows_web_when_rag_empty(self):
        from app.graph.compiler import dag_results_aggregator
        from app.graph.state import (
            DAGDefinition,
            PlanNode,
            deserialize_dag,
            serialize_dag,
        )

        rag = PlanNode(node_id="r1", node_type="rag", query="internal")
        search = PlanNode(node_id="s1", node_type="search", query="web", depends_on=["r1"])
        rag.result = {"results": []}
        state = create_initial_state("test query", "session-1")
        state["dag"] = serialize_dag(DAGDefinition(dag_name="test", nodes=[rag, search], edges=[]))
        state["current_executing_nodes"] = ["r1"]
        state["retrieval_policy"] = {"allow_web_after_rag_hit": False}

        result = dag_results_aggregator(state)
        planned = deserialize_dag(result["dag"])
        web = next(node for node in planned.nodes if node.node_id == "s1")

        assert web.status == StepStatus.PENDING
        assert result["retrieval_policy"]["web_search_required"] is True

    def test_dag_aggregator_triggers_budget_breaker_on_tool_call_limit(self):
        from app.graph.compiler import dag_results_aggregator
        from app.graph.state import (
            DAGDefinition,
            PlanNode,
            RuntimeStatus,
            deserialize_dag,
            serialize_dag,
        )

        current = PlanNode(node_id="r1", node_type="rag", query="internal")
        current.result = {"results": [{"content": "hit"}]}
        pending = PlanNode(node_id="s1", node_type="search", query="web", depends_on=["r1"])
        state = create_initial_state("test query", "session-1")
        state["dag"] = serialize_dag(DAGDefinition(dag_name="test", nodes=[current, pending], edges=[]))
        state["current_executing_nodes"] = ["r1"]
        state["tool_histories"] = [
            {
                "agent_type": "search",
                "tool_calls": [
                    {
                        "call_id": "call-1",
                        "tool_name": "duckduckgo_search",
                        "status": "success",
                        "tokens_used": 0,
                        "cost_usd": 0.0,
                    }
                ],
            },
            {
                "agent_type": "rag",
                "tool_calls": [
                    {
                        "call_id": "call-2",
                        "tool_name": "knowledge_base_search",
                        "status": "success",
                        "tokens_used": 0,
                        "cost_usd": 0.0,
                    }
                ],
            },
        ]
        state["budget_state"] = {"max_tool_calls": 2, "max_wall_clock_seconds": 9999}

        result = dag_results_aggregator(state)
        planned = deserialize_dag(result["dag"])
        skipped = next(node for node in planned.nodes if node.node_id == "s1")

        assert result["budget_state"]["hard_stop_reason"] == "tool_call_limit"
        assert result["runtime_status"] == RuntimeStatus.TERMINAL_FAILED.value
        assert skipped.status == StepStatus.SKIPPED
        assert "s1" in result["completed_nodes"]

    def test_should_continue_dag_exits_on_budget_hard_stop(self):
        from app.graph.compiler import should_continue_dag
        from app.graph.state import DAGDefinition, PlanNode, serialize_dag

        node = PlanNode(node_id="s1", node_type="search", query="test query")
        state = create_initial_state("test query", "session-1")
        state["dag"] = serialize_dag(DAGDefinition(dag_name="test", nodes=[node], edges=[]))
        state["budget_state"] = {"hard_stop_reason": "tool_call_limit"}

        assert should_continue_dag(state) == "analyst"

    def test_should_revise_blocked_by_budget_hard_stop(self):
        from app.graph.compiler import should_revise

        state = {
            "revision_needed": True,
            "revision_count": 0,
            "budget_state": {"hard_stop_reason": "tool_call_limit"},
            "session": {"max_revisions": 3},
        }

        assert should_revise(state) == "generate_report"

    def test_apply_llm_usage_to_state_updates_session_and_budget(self):
        from app.graph.compiler import _apply_llm_usage_to_state

        state = create_initial_state("test query", "session-1")
        session, budget_state = _apply_llm_usage_to_state(
            state,
            "planner",
            {
                "prompt_tokens": 100,
                "completion_tokens": 40,
                "total_tokens": 140,
                "cost_usd": 0.0012,
                "estimated": False,
                "model": "gpt-4o-mini",
            },
        )

        assert session["total_tokens"] == 140
        assert session["total_cost_usd"] == 0.0012
        assert budget_state["used_total_tokens"] == 140
        assert budget_state["used_cost_usd"] == 0.0012
        assert budget_state["llm_usage"][0]["agent"] == "planner"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
