"""
LangGraph StateGraph compiler for the research workflow.

===============================================================
五大核心主题在编译器中的映射
===============================================================

主题 1 - Autonomous Research Workflow:
    - compile_research_graph() 构建完整工作流
    - 从 START → Planner → DAG执行 → Analyst → Reflection → Report → END
    - 整个工作流由 LangGraph 编排，无需人工干预

主题 2 - Research DAG Generation:
    - planner_node 调用 PlannerAgent 生成 DAG
    - dag_node 执行 DAG（按拓扑序执行可并行节点）
    - after_planner 分发 DAG 节点到对应的工具 Agent
    - DAG 执行顺序由 get_executable_order() 计算

主题 3 - Tool-driven Multi-Agent Collaboration:
    - 每个节点执行前/后记录 ToolCallRecord
    - tool_histories 存储每个 Agent 的工具调用历史
    - send_node() 工具调用封装器

主题 4 - Long-running Stateful Agent:
    - PostgresSaver checkpointer 保存状态快照
    - session 字段存储会话元数据
    - revision_count 追踪重规划次数

主题 5 - Self-Reflection & Verification:
    - reflection_node 调用 ReflectionAgent
    - verification_result 决定是否需要重规划
    - 重规划循环最多 3 次

===============================================================
"""

from __future__ import annotations

import asyncio
import logging
import time
from datetime import datetime, timedelta
from typing import Any

from langgraph.constants import END, START, Send
from langgraph.graph import StateGraph

from app.config import get_settings
from app.core.time import utc_now_naive
from app.governance import McpPolicyProxy, McpToolRequest
from app.graph.state import (
    AgentType,
    DAGDefinition,
    NodeOutcome,
    PlanEdge,
    PlanNode,
    ResearchState,
    RuntimeStatus,
    StepStatus,
    TaskStatus,
    ToolCallRecord,
    ToolInvocationHistory,
    VerificationResult,
    deserialize_dag,
    serialize_dag,
)
from app.guardrails import (
    build_answer_gate_message,
    build_evidence_gate,
    build_guardrail_decision,
    get_research_budget,
    record_guardrail_event,
    should_require_action_approval,
)
from app.observability.trace import EventType

logger = logging.getLogger(__name__)

TOOL_NODE_TYPES = {"search", "browser", "rag", "mcp"}
WEB_NODE_TYPES = {"search", "browser"}


def _skill_context(state: ResearchState) -> dict[str, Any]:
    return dict(state.get("skill_context") or state.get("session", {}).get("skill_context") or {})


def _skill_prompt(state: ResearchState) -> str:
    context = _skill_context(state)
    parts: list[str] = []
    prompt_sections = context.get("effective_prompt_sections") or {}
    for key in ("overview", "prompt", "constraints"):
        value = prompt_sections.get(key)
        if isinstance(value, str) and value.strip():
            parts.append(f"{key.capitalize()}:\n{value.strip()}")
    return "\n\n".join(parts).strip()


def _agent_skill_hints(state: ResearchState, agent_name: str) -> list[str]:
    context = _skill_context(state)
    hints = (context.get("effective_agent_hints") or {}).get(agent_name) or []
    return [item for item in hints if isinstance(item, str) and item.strip()]


def _effective_tool_allowlist(state: ResearchState, decision: dict[str, Any]) -> set[str]:
    skill_tools = set(_skill_context(state).get("effective_tool_allowlist") or [])
    decision_tools = set(decision.get("enabled_tools", ["search", "browser", "rag"]))
    if skill_tools:
        return decision_tools.intersection(skill_tools)
    return decision_tools


def _research_budget(state: ResearchState) -> dict[str, int | float]:
    return get_research_budget(state.get("output_length") or state.get("session", {}).get("output_length"))


def _build_budget_state(state: ResearchState) -> dict[str, Any]:
    """Compute session-level budget usage and breaker status from runtime state."""
    configured = dict(_research_budget(state))
    existing_budget = state.get("budget_state") or {}
    for key in (
        "max_total_tokens",
        "max_cost_usd",
        "max_tool_calls",
        "max_wall_clock_seconds",
        "budget_profile",
        "estimated",
    ):
        if key in existing_budget and existing_budget[key] is not None:
            configured[key] = existing_budget[key]
    tool_histories = [
        history for history in state.get("tool_histories", [])
        if isinstance(history, dict)
    ]
    tool_calls = [
        call
        for history in tool_histories
        for call in history.get("tool_calls", [])
        if isinstance(call, dict)
    ]
    used_total_tokens = sum(int(call.get("tokens_used") or 0) for call in tool_calls)
    used_cost_usd = round(sum(float(call.get("cost_usd") or 0.0) for call in tool_calls), 6)
    used_tool_calls = len(tool_calls)
    if existing_budget.get("used_total_tokens"):
        used_total_tokens = max(used_total_tokens, int(existing_budget.get("used_total_tokens") or 0))
    if existing_budget.get("used_cost_usd"):
        used_cost_usd = max(used_cost_usd, float(existing_budget.get("used_cost_usd") or 0.0))
    started_at = state.get("created_at") or state.get("session", {}).get("created_at")
    elapsed_wall_clock_seconds = 0
    if started_at:
        try:
            elapsed_wall_clock_seconds = max(
                0,
                int((utc_now_naive() - datetime.fromisoformat(started_at)).total_seconds()),
            )
        except ValueError:
            elapsed_wall_clock_seconds = 0

    hard_stop_reason = None
    for key, reason in (
        ("max_tool_calls", "tool_call_limit"),
        ("max_total_tokens", "token_limit"),
        ("max_cost_usd", "cost_limit"),
        ("max_wall_clock_seconds", "wall_clock_limit"),
    ):
        limit = configured.get(key)
        if limit is None:
            continue
        current = {
            "max_tool_calls": used_tool_calls,
            "max_total_tokens": used_total_tokens,
            "max_cost_usd": used_cost_usd,
            "max_wall_clock_seconds": elapsed_wall_clock_seconds,
        }[key]
        if current >= limit:
            hard_stop_reason = reason
            break

    warning = False
    if not hard_stop_reason:
        warning_checks = (
            configured.get("max_tool_calls") and used_tool_calls >= int(configured["max_tool_calls"] * 0.8),
            configured.get("max_total_tokens") and used_total_tokens >= int(configured["max_total_tokens"] * 0.8),
            configured.get("max_cost_usd") and used_cost_usd >= float(configured["max_cost_usd"]) * 0.8,
            configured.get("max_wall_clock_seconds") and elapsed_wall_clock_seconds >= int(configured["max_wall_clock_seconds"] * 0.8),
        )
        warning = any(bool(check) for check in warning_checks)

    return {
        **configured,
        "used_total_tokens": used_total_tokens,
        "used_cost_usd": used_cost_usd,
        "used_tool_calls": used_tool_calls,
        "elapsed_wall_clock_seconds": elapsed_wall_clock_seconds,
        "hard_stop_reason": hard_stop_reason,
        "warning": warning,
    }


def _apply_budget_breaker(
    dag: DAGDefinition,
    budget_state: dict[str, Any],
) -> tuple[DAGDefinition, list[str]]:
    """Skip remaining retrievable nodes once hard budget limits are reached."""
    if not budget_state.get("hard_stop_reason"):
        return dag, []
    skipped_nodes: list[str] = []
    for node in dag.nodes:
        if node.node_type not in TOOL_NODE_TYPES:
            continue
        if node.status == StepStatus.PENDING:
            node.status = StepStatus.SKIPPED
            node.last_error = budget_state["hard_stop_reason"]
            node.last_error_category = "budget_exceeded"
            skipped_nodes.append(node.node_id)
    return dag, skipped_nodes


def _apply_llm_usage_to_state(
    state: ResearchState,
    agent_name: str,
    usage: dict[str, Any] | None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Merge LLM usage into session + budget state and emit a traceable summary."""
    session = dict(state.get("session", {}))
    budget_state = dict(state.get("budget_state") or {})
    if not usage:
        return session, budget_state

    session["total_tokens"] = int(session.get("total_tokens", 0) or 0) + int(usage.get("total_tokens", 0) or 0)
    session["total_cost_usd"] = round(
        float(session.get("total_cost_usd", 0.0) or 0.0) + float(usage.get("cost_usd", 0.0) or 0.0),
        6,
    )
    budget_state["used_total_tokens"] = int(budget_state.get("used_total_tokens", 0) or 0) + int(usage.get("total_tokens", 0) or 0)
    budget_state["used_cost_usd"] = round(
        float(budget_state.get("used_cost_usd", 0.0) or 0.0) + float(usage.get("cost_usd", 0.0) or 0.0),
        6,
    )
    budget_state.setdefault("llm_usage", [])
    budget_state["llm_usage"].append({
        "agent": agent_name,
        **usage,
    })
    return session, budget_state


def _node_identifier(node: PlanNode) -> str:
    return f"{node.node_type}:{node.query.strip().lower()}"


def _build_failure_memory(state: ResearchState, dag: DAGDefinition) -> dict[str, Any]:
    """Summarize failures so retries and replanning can avoid repeating mistakes."""
    budget = dict(_research_budget(state))
    max_retries = int(
        (state.get("budget_state") or {}).get("max_retries_per_tool")
        or budget.get("max_retries_per_tool")
        or 2
    )
    records: dict[str, dict[str, Any]] = {}
    repeated_terminal_nodes: list[str] = []

    for node in dag.nodes:
        if node.node_type not in TOOL_NODE_TYPES:
            continue
        key = _node_identifier(node)
        record = records.setdefault(key, {
            "node_type": node.node_type,
            "query": node.query,
            "last_error_category": None,
            "last_error": None,
            "retry_count": 0,
            "terminal_failure": False,
            "repeat_blocked": False,
            "next_retry_at": None,
            "backoff_seconds": 0,
        })
        previous_retry_count = int(record.get("retry_count") or 0)
        record["last_error_category"] = node.last_error_category or record["last_error_category"]
        record["last_error"] = node.last_error or record["last_error"]
        record["retry_count"] = max(previous_retry_count, int(node.retry_count or 0))
        record["terminal_failure"] = bool(record["terminal_failure"] or node.terminal_failure)
        if node.status == StepStatus.FAILED and node.last_error_category and not node.terminal_failure:
            existing_next_retry_at = record.get("next_retry_at")
            if not existing_next_retry_at or previous_retry_count != int(node.retry_count or 0):
                next_retry_at, backoff_seconds = _compute_next_retry_at(
                    node.last_error_category,
                    int(node.retry_count or 0),
                )
                record["next_retry_at"] = next_retry_at
                record["backoff_seconds"] = backoff_seconds
        if record["terminal_failure"] or record["retry_count"] >= max_retries:
            record["repeat_blocked"] = True
            repeated_terminal_nodes.append(node.node_id)

    return {
        "records": list(records.values()),
        "max_retries_per_tool": max_retries,
        "repeat_blocked_nodes": sorted(set(repeated_terminal_nodes)),
    }


def _format_failure_memory_notes(failure_memory: dict[str, Any] | None) -> str:
    """Convert failure memory into concise planner hints."""
    if not failure_memory:
        return ""
    records = failure_memory.get("records") or []
    notable = [
        record for record in records
        if record.get("repeat_blocked") or record.get("retry_count", 0) > 0
    ]
    if not notable:
        return ""
    lines = [
        "Avoid repeating known failed tool patterns from this session.",
        "Known failures:",
    ]
    for record in notable[:8]:
        lines.append(
            f"- {record.get('node_type')} | {record.get('query')} | "
            f"error={record.get('last_error_category') or 'unknown'} | "
            f"retries={record.get('retry_count', 0)} | "
            f"repeat_blocked={bool(record.get('repeat_blocked'))} | "
            f"next_retry_at={record.get('next_retry_at')}"
        )
    lines.append("Prefer alternative tools or narrower queries when a pattern is repeat_blocked.")
    return "\n".join(lines)


def _should_retry_node(
    node: PlanNode,
    budget_state: dict[str, Any],
    failure_memory: dict[str, Any] | None,
) -> tuple[bool, str | None]:
    """Decide whether a failed node should be retried automatically."""
    if node.terminal_failure:
        return False, "terminal_failure"
    if node.status != StepStatus.FAILED:
        return False, "not_failed"
    if node.last_error_category is None:
        return False, "missing_error_category"

    max_retries = int(
        budget_state.get("max_retries_per_tool")
        or _research_budget({"output_length": None}).get("max_retries_per_tool")
        or 2
    )
    if int(node.retry_count or 0) >= max_retries:
        return False, "max_retries_reached"

    record = None
    for item in (failure_memory or {}).get("records", []):
        if item.get("node_type") == node.node_type and item.get("query") == node.query:
            record = item
            break
    if record and record.get("repeat_blocked"):
        return False, "repeat_blocked"

    retryable_categories = {"timeout", "upstream_429", "upstream_5xx", "network", "temporary_db"}
    if node.last_error_category not in retryable_categories:
        return False, "non_retryable_category"
    if record and record.get("next_retry_at"):
        try:
            if utc_now_naive() < datetime.fromisoformat(str(record["next_retry_at"])):
                return False, "backoff_active"
        except ValueError:
            pass
    return True, None


def _base_backoff_seconds(error_category: str | None) -> int:
    mapping = {
        "timeout": 2,
        "network": 2,
        "temporary_db": 2,
        "upstream_5xx": 3,
        "upstream_429": 5,
    }
    return mapping.get(error_category or "", 2)


def _compute_next_retry_at(error_category: str | None, retry_count: int) -> tuple[str, int]:
    """Return ISO timestamp + seconds for exponential backoff."""
    base = _base_backoff_seconds(error_category)
    exponent = max(0, retry_count - 1)
    backoff_seconds = min(base * (2 ** exponent), 60)
    next_retry_at = utc_now_naive() + timedelta(seconds=backoff_seconds)
    return next_retry_at.isoformat(), backoff_seconds


def _enforce_internal_first_dag(dag: DAGDefinition, user_query: str) -> DAGDefinition:
    """Ensure every web retrieval node runs only after internal RAG has been checked."""
    rag_nodes = [node for node in dag.nodes if node.node_type == "rag"]
    if not rag_nodes:
        rag_node = PlanNode(
            node_id="internal_rag_first",
            node_type="rag",
            query=f"内部知识库检索：{user_query}",
            depends_on=[],
            parallel=True,
        )
        dag.nodes.insert(0, rag_node)
        rag_nodes = [rag_node]

    rag_ids = [node.node_id for node in rag_nodes]
    existing_edges = {(edge.from_node, edge.to_node) for edge in dag.edges}
    for node in dag.nodes:
        if node.node_type not in WEB_NODE_TYPES:
            continue
        for rag_id in rag_ids:
            if rag_id == node.node_id or rag_id in node.depends_on:
                continue
            node.depends_on.append(rag_id)
            if (rag_id, node.node_id) not in existing_edges:
                dag.edges.append(PlanEdge(from_node=rag_id, to_node=node.node_id, edge_type="sequential"))
                existing_edges.add((rag_id, node.node_id))

    return dag


def _result_count(node: PlanNode) -> int:
    if not node.result:
        return 0
    results = node.result.get("results")
    return len(results) if isinstance(results, list) else 0


def _classify_error_category(error: str | None) -> str:
    """Map runtime failures into retryable/terminal governance buckets."""
    message = (error or "").lower()
    if any(token in message for token in ("timeout", "timed out")):
        return "timeout"
    if any(token in message for token in ("429", "rate limit")):
        return "upstream_429"
    if any(token in message for token in ("503", "502", "500", "service unavailable", "bad gateway")):
        return "upstream_5xx"
    if any(token in message for token in ("connection", "network", "dns")):
        return "network"
    if any(token in message for token in ("database", "db")):
        return "temporary_db"
    return "temporary_db"


def _is_terminal_error_category(category: str | None) -> bool:
    return category in {
        "schema_validation",
        "permission_denied",
        "approval_denied",
        "untrusted_mcp_server",
        "tool_not_allowed",
        "domain_blocked",
        "unsupported_input",
    }


def _make_node_outcome(
    *,
    node_id: str,
    tool_history: dict[str, Any],
    tool_name: str,
    status: str,
    retry_count: int = 0,
    error_category: str | None = None,
    error_message: str | None = None,
    result_count: int = 0,
    approval_request_id: str | None = None,
) -> dict[str, Any]:
    tool_call = (tool_history.get("tool_calls") or [{}])[-1]
    outcome = NodeOutcome(
        node_id=node_id,
        tool_call_id=tool_call.get("call_id"),
        tool_name=tool_name,
        status=status,
        error_category=error_category,
        error_message=error_message,
        retry_count=retry_count,
        tokens_used=int(tool_call.get("tokens_used") or 0),
        cost_usd=float(tool_call.get("cost_usd") or 0.0),
        result_count=result_count,
        approval_request_id=approval_request_id,
    )
    return outcome.model_dump()


def _make_approval_request(
    *,
    node_id: str,
    tool_name: str,
    reason: str,
    request_payload: dict[str, Any],
    risk_level: str = "high",
) -> dict[str, Any]:
    return {
        "approval_id": f"ap-{node_id}-{tool_name}-{int(time.time() * 1000)}",
        "node_id": node_id,
        "tool_name": tool_name,
        "risk_level": risk_level,
        "reason": reason,
        "request_payload": request_payload,
        "status": "pending",
        "requested_at": utc_now_naive().isoformat(),
    }


def _skip_pending_web_nodes(dag: DAGDefinition) -> list[str]:
    skipped: list[str] = []
    for node in dag.nodes:
        if node.node_type in WEB_NODE_TYPES and node.status == StepStatus.PENDING:
            node.status = StepStatus.SKIPPED
            skipped.append(node.node_id)
    return skipped


# ==============================================================
# 主题 2 & 3: 工具调用追踪装饰器
# ==============================================================

def track_tool_call(agent_type: AgentType, tool_name: str):
    """
    装饰器：追踪工具调用（主题 3）。

    在节点函数执行时记录 ToolCallRecord，
    方便后续分析和优化工具使用效率。
    """
    def decorator(func):
        async def async_wrapper(state: ResearchState, *args, **kwargs) -> dict:
            agent_str = agent_type.value
            call_record = ToolCallRecord(
                agent_type=agent_type,
                tool_name=tool_name,
                args={"task_id": state.get("task_id"), "query": state.get("user_query")},
                status="running",
            )

            # 记录开始
            state.setdefault("tool_histories", [])
            history = ToolInvocationHistory(
                agent_type=agent_type,
                tool_calls=[call_record],
            )
            state["tool_histories"].append(history.model_dump())

            try:
                start_time = time.perf_counter()
                result = await func(state, *args, **kwargs)
                if (state["tool_histories"]
                        and state["tool_histories"][-1].get("agent_type") == agent_str):
                    state["tool_histories"][-1]["tool_calls"][-1]["status"] = "success"
                    state["tool_histories"][-1]["tool_calls"][-1]["completed_at"] = utc_now_naive().isoformat()
                    state["tool_histories"][-1]["tool_calls"][-1]["duration_ms"] = int((time.perf_counter() - start_time) * 1000)
                    state["tool_histories"][-1]["tool_calls"][-1]["result_summary"] = str(result)[:200]
                return result
            except (OSError, RuntimeError, TypeError, ValueError, KeyError) as e:
                if (state["tool_histories"]
                        and state["tool_histories"][-1].get("agent_type") == agent_str):
                    state["tool_histories"][-1]["tool_calls"][-1]["status"] = "error"
                    state["tool_histories"][-1]["tool_calls"][-1]["completed_at"] = utc_now_naive().isoformat()
                    state["tool_histories"][-1]["tool_calls"][-1]["error"] = str(e)
                raise

        def sync_wrapper(state: ResearchState, *args, **kwargs) -> dict:
            agent_str = agent_type.value
            call_record = ToolCallRecord(
                agent_type=agent_type,
                tool_name=tool_name,
                args={"task_id": state.get("task_id"), "query": state.get("user_query")},
                status="running",
            )

            state.setdefault("tool_histories", [])
            history = ToolInvocationHistory(
                agent_type=agent_type,
                tool_calls=[call_record],
            )
            state["tool_histories"].append(history.model_dump())
            try:
                start_time = time.perf_counter()
                result = func(state, *args, **kwargs)
                if (state["tool_histories"]
                        and state["tool_histories"][-1].get("agent_type") == agent_str):
                    state["tool_histories"][-1]["tool_calls"][-1]["status"] = "success"
                    state["tool_histories"][-1]["tool_calls"][-1]["completed_at"] = utc_now_naive().isoformat()
                    state["tool_histories"][-1]["tool_calls"][-1]["duration_ms"] = int((time.perf_counter() - start_time) * 1000)
                    state["tool_histories"][-1]["tool_calls"][-1]["result_summary"] = str(result)[:200]
                return result
            except (OSError, RuntimeError, TypeError, ValueError, KeyError) as e:
                if (state["tool_histories"]
                        and state["tool_histories"][-1].get("agent_type") == agent_str):
                    state["tool_histories"][-1]["tool_calls"][-1]["status"] = "error"
                    state["tool_histories"][-1]["tool_calls"][-1]["completed_at"] = utc_now_naive().isoformat()
                    state["tool_histories"][-1]["tool_calls"][-1]["error"] = str(e)
                raise

        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper

    return decorator


def _make_tool_history(
    *,
    agent_type: AgentType,
    tool_name: str,
    task_id: str | None,
    query: str,
) -> dict[str, Any]:
    """Create a structured tool history entry for a single tool invocation."""
    history = ToolInvocationHistory(
        agent_type=agent_type,
        tool_calls=[ToolCallRecord(
            agent_type=agent_type,
            tool_name=tool_name,
            args={
                "task_id": task_id,
                "query": query,
            },
            safety_json={
                "source": "harness_tool_layer",
                "audited": True,
            },
            status="running",
        )],
    )
    return history.model_dump()


def _append_trace_event(
    state: ResearchState,
    event_type: EventType,
    agent: str,
    content: str,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Emit both agent trace and SSE-friendly state updates for the UI."""
    from app.graph.state import AgentEvent

    event = AgentEvent(agent=agent, event_type=event_type.value, content=content).model_dump()
    if metadata:
        event.update(metadata)
    state.setdefault("agent_trace", []).append(event)
    return event


def _finish_tool_history(
    history: dict[str, Any],
    *,
    status: str,
    result_summary: str | None = None,
    error: str | None = None,
) -> None:
    """Finalize a structured tool history entry."""
    tool_call = history["tool_calls"][-1]
    tool_call["status"] = status
    tool_call["completed_at"] = utc_now_naive().isoformat()
    started_at = datetime.fromisoformat(tool_call["started_at"])
    tool_call["duration_ms"] = int((utc_now_naive() - started_at).total_seconds() * 1000)
    if result_summary is not None:
        tool_call["result_summary"] = result_summary[:200]
    if error is not None:
        tool_call["error"] = error


def _update_tool_safety(history: dict[str, Any], **metadata: Any) -> None:
    """Merge structured safety metadata into the current tool call audit record."""
    tool_call = history["tool_calls"][-1]
    safety = dict(tool_call.get("safety_json") or {})
    for key, value in metadata.items():
        if value is not None:
            safety[key] = value
    tool_call["safety_json"] = safety


# ==============================================================
# 主题 1 & 2: Planner Node - DAG 生成
# ==============================================================

def planner_node(state: ResearchState) -> dict:
    """
    Planner Agent: 将用户查询转换为可执行的 DAG（主题 1 & 2）。

    主题 1: 这是自主研究工作流的起点
    主题 2: Planner 生成研究计划的 DAG 结构

    输入: user_query
    输出: dag（序列化的 DAGDefinition）, agent_trace
    """
    from app.agents.planner import PlannerAgent
    from app.observability.trace import EventType, emit_event

    # 追踪事件
    emit_event(state, EventType.AGENT_START, "planner", f"Planning research for: {state['user_query']}")

    # 调用 Planner Agent 生成 DAG
    agent = PlannerAgent()
    planning_hints = _format_failure_memory_notes(state.get("failure_memory"))
    dag: DAGDefinition = agent.create_dag(
        state["user_query"],
        planning_hints=planning_hints or None,
        skill_prompt=_skill_prompt(state) or None,
        planner_hints=_agent_skill_hints(state, "planner"),
    )
    dag = _enforce_internal_first_dag(dag, state["user_query"])
    decision = build_guardrail_decision(state["user_query"], user_confirmed=state.get("user_confirmed", False))
    record_guardrail_event(
        state,
        agent="planner",
        event_type="guardrail_decision",
        content="planner routed",
        metadata=decision.model_dump(),
    )

    # 追踪 DAG 生成
    emit_event(
        state, EventType.AGENT_COMPLETE, "planner",
        f"Generated DAG '{dag.dag_name}' with {len(dag.nodes)} nodes, "
        f"execution order: {dag.get_executable_order()}"
    )

    # 更新会话元数据
    session = state.get("session", {})
    session["updated_at"] = utc_now_naive().isoformat()
    session, budget_state = _apply_llm_usage_to_state(state, "planner", agent.last_usage)

    return {
        "dag": serialize_dag(dag),
        "status": TaskStatus.RUNNING.value,
        "session": session,
        "budget_state": budget_state,
        "guardrail_decision": decision.model_dump(),
        "failure_memory": state.get("failure_memory"),
        "skill_context": _skill_context(state),
    }


# ==============================================================
# 主题 1 & 2: DAG 执行 Node - 按拓扑序执行
# ==============================================================

def dag_executor_node(state: ResearchState) -> dict:
    """
    DAG 执行器：按拓扑序分发 DAG 节点（主题 1 & 2）。

    主题 2: 根据 DAG 的节点依赖关系，分发到对应的工具 Agent
    主题 3: 使用 Send API 实现并行执行

    输入: dag（DAGDefinition）
    输出: current_executing_nodes, agent_trace
    """
    from app.observability.trace import EventType, emit_event

    dag = deserialize_dag(state["dag"])
    execution_order = dag.get_executable_order()
    budget_state = _build_budget_state(state)
    failure_memory = state.get("failure_memory")

    # 当前批次节点
    all_completed = set(state.get("completed_nodes", []))
    current_batch = []
    deferred_retries: list[dict[str, Any]] = []

    for node in dag.nodes:
        if node.node_type not in TOOL_NODE_TYPES:
            continue
        should_retry, reason = _should_retry_node(node, budget_state, failure_memory)
        if should_retry:
            node.status = StepStatus.PENDING
        elif reason == "backoff_active":
            for item in (failure_memory or {}).get("records", []):
                if item.get("node_type") == node.node_type and item.get("query") == node.query:
                    deferred_retries.append({
                        "node_id": node.node_id,
                        "next_retry_at": item.get("next_retry_at"),
                        "backoff_seconds": item.get("backoff_seconds", 0),
                    })
                    break

    for batch in execution_order:
        # 找出当前批次中未完成的节点
        for node_id in batch:
            if node_id not in all_completed:
                node = next((n for n in dag.nodes if n.node_id == node_id), None)
                if node and node.node_type in TOOL_NODE_TYPES and node.status in (StepStatus.PENDING, StepStatus.RUNNING):
                    current_batch.append(node_id)

        if current_batch:
            break

    if not current_batch:
        # 所有节点都已执行完成
        if deferred_retries:
            next_retry_at = min(
                (
                    datetime.fromisoformat(item["next_retry_at"])
                    for item in deferred_retries
                    if item.get("next_retry_at")
                ),
                default=None,
            )
            if next_retry_at is not None:
                delay_seconds = max(0.0, min((next_retry_at - utc_now_naive()).total_seconds(), 5.0))
                if delay_seconds > 0:
                    time.sleep(delay_seconds)
        emit_event(state, EventType.AGENT_COMPLETE, "dag_executor", "All DAG nodes completed")
        trace = []
        if deferred_retries:
            trace.append(_append_trace_event(
                state,
                EventType.AGENT_COMPLETE,
                "retry_scheduler",
                f"Deferred {len(deferred_retries)} retryable node(s) by exponential backoff.",
                {"deferred_retries": deferred_retries},
            ))
        return {"current_executing_nodes": [], "agent_trace": trace}

    emit_event(
        state, EventType.AGENT_START, "dag_executor",
        f"Executing batch: {current_batch} (parallel)"
    )

    return {
        "dag": serialize_dag(dag),
        "current_executing_nodes": current_batch,
        "agent_trace": [
            _append_trace_event(
                state,
                EventType.AGENT_COMPLETE,
                "retry_scheduler",
                f"Deferred {len(deferred_retries)} retryable node(s) by exponential backoff.",
                {"deferred_retries": deferred_retries},
            ),
        ] if deferred_retries else [],
    }


def dag_results_aggregator(state: ResearchState) -> dict:
    """
    DAG 结果聚合器：收集工具 Agent 的结果，更新 DAG 状态（主题 2）。

    当一批并行节点执行完成后，调用此函数更新 DAG 状态。
    """
    from app.observability.trace import EventType

    dag = deserialize_dag(state["dag"])
    current_nodes = set(state.get("current_executing_nodes", []))
    latest_outcomes = {
        outcome["node_id"]: outcome
        for outcome in state.get("node_outcomes", [])
        if isinstance(outcome, dict) and outcome.get("node_id") in current_nodes
    }

    completed = set(state.get("completed_nodes", []))
    runtime_status = state.get("runtime_status", RuntimeStatus.RUNNING.value)

    for node in dag.nodes:
        if node.node_id not in current_nodes:
            continue
        outcome = latest_outcomes.get(node.node_id)
        if not outcome and node.status == StepStatus.DONE:
            completed.add(node.node_id)
            continue
        if not outcome and node.result is not None and node.status != StepStatus.FAILED and not node.waiting_approval:
            node.status = StepStatus.DONE
            completed.add(node.node_id)
            continue
        if not outcome:
            continue
        outcome_status = outcome.get("status")
        node.last_error = outcome.get("error_message")
        node.last_error_category = outcome.get("error_category")
        node.waiting_approval = outcome_status == "awaiting_approval"
        node.terminal_failure = outcome_status == "terminal_error"
        node.retry_count = max(node.retry_count, int(outcome.get("retry_count") or 0))

        if outcome_status == "success":
            node.status = StepStatus.DONE
            completed.add(node.node_id)
        elif outcome_status == "skipped":
            node.status = StepStatus.SKIPPED
            completed.add(node.node_id)
        elif outcome_status == "awaiting_approval":
            node.status = StepStatus.PENDING
            runtime_status = RuntimeStatus.AWAITING_APPROVAL.value
        elif outcome_status == "terminal_error":
            node.status = StepStatus.FAILED
            runtime_status = RuntimeStatus.TERMINAL_FAILED.value
        elif outcome_status == "retryable_error":
            node.status = StepStatus.FAILED
            if runtime_status not in (RuntimeStatus.AWAITING_APPROVAL.value, RuntimeStatus.TERMINAL_FAILED.value):
                runtime_status = RuntimeStatus.RETRYABLE_FAILED.value

    completed_list = list(completed)
    retrieval_policy = dict(state.get("retrieval_policy") or {})
    retrieval_policy.setdefault("mode", "internal_first")
    retrieval_policy.setdefault("allow_web_after_rag_hit", bool(state.get("allow_web_after_rag_hit", False)))
    retrieval_policy.setdefault("rag_group", state.get("rag_group"))
    budget_state = _build_budget_state(state)

    rag_hit_count = sum(
        _result_count(node)
        for node in dag.nodes
        if node.node_type == "rag" and node.status == StepStatus.DONE
    )
    retrieval_policy["rag_hit_count"] = rag_hit_count

    skipped_web_nodes: list[str] = []
    if current_nodes and all(
        (node.node_type == "rag")
        for node in dag.nodes
        if node.node_id in current_nodes
    ):
        if rag_hit_count > 0:
            if retrieval_policy["allow_web_after_rag_hit"]:
                retrieval_policy["web_search_required"] = True
                retrieval_policy["web_search_reason"] = "rag_hit_user_allowed_web"
            else:
                retrieval_policy["web_search_required"] = False
                retrieval_policy["web_search_reason"] = "rag_hit_user_skipped_web"
                skipped_web_nodes = _skip_pending_web_nodes(dag)
                completed_list = list(set(completed_list) | set(skipped_web_nodes))
        else:
            retrieval_policy["web_search_required"] = True
            retrieval_policy["web_search_reason"] = "rag_empty_auto_web"

    trace = []
    if skipped_web_nodes:
        trace.append(_append_trace_event(
            state,
            EventType.AGENT_COMPLETE,
            "retrieval_policy",
            f"Internal RAG found {rag_hit_count} chunks; skipped {len(skipped_web_nodes)} web nodes by user policy.",
            {
                "retrieval_policy": retrieval_policy,
                "skipped_web_nodes": skipped_web_nodes,
            },
        ))
    elif current_nodes and any(node.node_type == "rag" for node in dag.nodes if node.node_id in current_nodes):
        trace.append(_append_trace_event(
            state,
            EventType.AGENT_COMPLETE,
            "retrieval_policy",
            f"Internal RAG found {rag_hit_count} chunks; web search policy: {retrieval_policy.get('web_search_reason')}.",
            {"retrieval_policy": retrieval_policy},
        ))

    breaker_skipped_nodes: list[str] = []
    if budget_state.get("warning"):
        trace.append(_append_trace_event(
            state,
            EventType.AGENT_COMPLETE,
            "budget_guardrail",
            f"Budget warning: tool_calls={budget_state['used_tool_calls']}, "
            f"tokens={budget_state['used_total_tokens']}, cost={budget_state['used_cost_usd']}, "
            f"wall_clock={budget_state['elapsed_wall_clock_seconds']}s",
            {"budget_state": budget_state},
        ))
    if budget_state.get("hard_stop_reason"):
        dag, breaker_skipped_nodes = _apply_budget_breaker(dag, budget_state)
        if breaker_skipped_nodes:
            completed_list = list(set(completed_list) | set(breaker_skipped_nodes))
        runtime_status = RuntimeStatus.TERMINAL_FAILED.value
        trace.append(_append_trace_event(
            state,
            EventType.AGENT_COMPLETE,
            "budget_guardrail",
            f"Budget breaker triggered: {budget_state['hard_stop_reason']}. "
            f"Skipped {len(breaker_skipped_nodes)} pending tool nodes.",
            {
                "budget_state": budget_state,
                "skipped_nodes": breaker_skipped_nodes,
            },
        ))

    failure_memory = _build_failure_memory(state, dag)
    if failure_memory.get("records"):
        trace.append(_append_trace_event(
            state,
            EventType.AGENT_COMPLETE,
            "failure_memory",
            f"Captured {len(failure_memory['records'])} failure memory records",
            {"failure_memory": failure_memory},
        ))

    return {
        "dag": serialize_dag(dag),
        "completed_nodes": completed_list,
        "retrieval_policy": retrieval_policy,
        "runtime_status": runtime_status,
        "budget_state": budget_state,
        "failure_memory": failure_memory,
        "agent_trace": trace,
    }


def should_continue_dag(state: ResearchState) -> str:
    """
    判断 DAG 是否还有未执行的节点（主题 1 & 2）。

    返回:
    - "continue": 还有节点需要执行
    - "analyst": 所有节点都已执行，进入分析阶段
    """
    dag = deserialize_dag(state["dag"])
    completed = set(state.get("completed_nodes", []))
    if state.get("pending_approvals"):
        return "approval_reviewer"
    budget_state = state.get("budget_state") or {}
    if budget_state.get("hard_stop_reason"):
        return "analyst"

    pending = [
        n for n in dag.nodes
        if n.node_type in TOOL_NODE_TYPES
        and n.node_id not in completed
        and n.status != StepStatus.SKIPPED
        and not getattr(n, "waiting_approval", False)
    ]

    deferred_retry_exists = False
    failure_memory = state.get("failure_memory") or {}
    records = failure_memory.get("records") or []
    if pending:
        for node in pending:
            if node.status != StepStatus.FAILED:
                deferred_retry_exists = True
                break
            for record in records:
                if (
                    record.get("node_type") == node.node_type
                    and record.get("query") == node.query
                    and record.get("next_retry_at")
                    and not record.get("repeat_blocked")
                ):
                    deferred_retry_exists = True
                    break
            if deferred_retry_exists:
                break

    if deferred_retry_exists:
        return "continue"
    return "analyst"


def approval_reviewer_node(state: ResearchState) -> dict:
    """Independent reviewer branch for action-level approvals."""
    from app.graph.state import AgentEvent

    pending = [
        approval for approval in state.get("pending_approvals", [])
        if isinstance(approval, dict) and approval.get("status", "pending") == "pending"
    ]
    trace = [AgentEvent(
        agent="approval_reviewer",
        event_type="agent_complete",
        content=f"Paused for {len(pending)} approval request(s)",
    ).model_dump()]
    return {
        "runtime_status": RuntimeStatus.AWAITING_APPROVAL.value,
        "status": TaskStatus.PAUSED.value,
        "agent_trace": trace,
    }


# ==============================================================
# 主题 3: 工具 Agent Nodes - Search / Browser / RAG
# ==============================================================

def search_node(state: ResearchState) -> dict:
    """
    Search Agent: 执行搜索工具（主题 3）。

    主题 3: 工具驱动多 Agent
    - 从 DAG 获取当前批次中类型为 search 的节点
    - 执行搜索工具
    - 记录工具调用历史

    输入: dag, current_executing_nodes
    输出: collected_evidence, tool_histories, agent_trace
    """
    from app.agents.search import SearchAgent
    from app.graph.state import AgentEvent, Evidence
    from app.guardrails import record_guardrail_event, validate_tool_invocation
    from app.observability.trace import EventType

    dag = deserialize_dag(state["dag"])
    executing_nodes = state.get("executing_nodes", state.get("current_executing_nodes", []))

    # 找出当前批次中分配给 search 的节点
    search_nodes = [
        n for n in dag.nodes
        if n.node_id in executing_nodes and n.node_type == "search"
    ]

    if not search_nodes:
        return {"collected_evidence": [], "tool_histories": [], "agent_trace": [], "node_outcomes": []}

    trace = [
        _append_trace_event(state, EventType.AGENT_START, "search", f"Executing {len(search_nodes)} search nodes")
    ]

    agent = SearchAgent()
    all_evidence = []
    tool_histories = []
    node_outcomes = []
    budget = _research_budget(state)
    max_queries = budget.get("search_max_queries", 4)
    max_results = budget.get("search_max_results", 10)

    for node in search_nodes[:max_queries]:
        trace.append(_append_trace_event(
            state,
            EventType.TOOL_START,
            "search",
            f"Searching: {node.query}",
            {"tool_name": "duckduckgo_search", "args": {"query": node.query}},
        ))
        tool_history = _make_tool_history(
            agent_type=AgentType.SEARCH,
            tool_name="execute_search",
            task_id=state.get("task_id"),
            query=node.query,
        )
        valid, reason = validate_tool_invocation("duckduckgo_search", {"query": node.query})
        if not valid:
            node.status = StepStatus.FAILED
            node.terminal_failure = True
            node.last_error = reason or "invalid_args"
            node.last_error_category = "schema_validation"
            _update_tool_safety(tool_history, schema_validated=False, sanitized_error=True)
            _finish_tool_history(tool_history, status="error", error=reason or "invalid_args")
            record_guardrail_event(
                state,
                agent="search",
                event_type="tool_blocked",
                content=reason or "invalid_args",
                metadata={"tool": "duckduckgo_search", "query": node.query},
            )
            tool_histories.append(tool_history)
            node_outcomes.append(_make_node_outcome(
                node_id=node.node_id,
                tool_history=tool_history,
                tool_name="duckduckgo_search",
                status="terminal_error",
                retry_count=node.retry_count,
                error_category="schema_validation",
                error_message=reason or "invalid_args",
            ))
            continue
        require_approval, approval_reason = should_require_action_approval(
            tool_name="duckduckgo_search",
            decision=state.get("guardrail_decision") and build_guardrail_decision(
                state["user_query"],
                user_confirmed=state.get("user_confirmed", False),
            ),
            readonly=True,
        )
        if require_approval:
            approval = _make_approval_request(
                node_id=node.node_id,
                tool_name="duckduckgo_search",
                reason=approval_reason or "action_requires_approval",
                request_payload={"query": node.query},
            )
            node.waiting_approval = True
            _update_tool_safety(
                tool_history,
                schema_validated=True,
                approval_required=True,
                sanitized_error=True,
            )
            _finish_tool_history(tool_history, status="error", error=approval["reason"])
            tool_histories.append(tool_history)
            node_outcomes.append(_make_node_outcome(
                node_id=node.node_id,
                tool_history=tool_history,
                tool_name="duckduckgo_search",
                status="awaiting_approval",
                retry_count=node.retry_count,
                error_category="approval_required",
                error_message=approval["reason"],
                approval_request_id=approval["approval_id"],
            ))
            return {
                "collected_evidence": [e.model_dump() for e in all_evidence],
                "tool_histories": tool_histories,
                "node_outcomes": node_outcomes,
                "pending_approvals": [approval],
                "agent_trace": trace,
            }
        # 更新节点状态为 RUNNING
        node.status = StepStatus.RUNNING

        try:
            results = agent.execute_search(node.query)[:max_results]
            node.result = {"results": [r.model_dump() for r in results]}
            node.confidence = 0.9 if results else 0.0
            _finish_tool_history(
                tool_history,
                status="success",
                result_summary=f"{len(results)} search results",
            )
            _update_tool_safety(tool_history, schema_validated=True)
            node.status = StepStatus.DONE
            node.last_error = None
            node.last_error_category = None
            node.terminal_failure = False

            # 转换为 Evidence
            for r in results:
                evidence = Evidence(
                    content=f"{r.title}: {r.snippet}",
                    source_url=r.url,
                    source_title=r.title,
                    source_type="web",
                    collected_by=AgentType.SEARCH,
                )
                all_evidence.append(evidence)

            trace.append(_append_trace_event(
                state,
                EventType.TOOL_COMPLETE if results else EventType.TOOL_ERROR,
                "search",
                f"Found {len(results)} results for: {node.query}" if results else f"No web search results for: {node.query}",
                {
                    "tool_name": "duckduckgo_search",
                    "status": "success" if results else "error",
                    "result_summary": f"{len(results)} search results",
                    "error": None if results else "no_search_results",
                },
            ))
            node_outcomes.append(_make_node_outcome(
                node_id=node.node_id,
                tool_history=tool_history,
                tool_name="duckduckgo_search",
                status="success",
                retry_count=node.retry_count,
                result_count=len(results),
            ))
        except (OSError, RuntimeError, TypeError, ValueError, KeyError) as e:
            node.status = StepStatus.FAILED
            node.retry_count += 1
            category = _classify_error_category(str(e))
            node.last_error = str(e)
            node.last_error_category = category
            node.terminal_failure = _is_terminal_error_category(category)
            trace.append(_append_trace_event(
                state,
                EventType.TOOL_ERROR,
                "search",
                f"Search failed: {e!s}",
                {"tool_name": "duckduckgo_search", "status": "error", "error": str(e)},
            ))
            _finish_tool_history(tool_history, status="error", error=str(e))
            node_outcomes.append(_make_node_outcome(
                node_id=node.node_id,
                tool_history=tool_history,
                tool_name="duckduckgo_search",
                status="terminal_error" if node.terminal_failure else "retryable_error",
                retry_count=node.retry_count,
                error_category=category,
                error_message=str(e),
            ))
        tool_histories.append(tool_history)

    trace.append(AgentEvent(
        agent="search",
        event_type="agent_complete",
        content=f"Collected {len(all_evidence)} evidence from search"
    ).model_dump())

    return {
        "collected_evidence": [e.model_dump() for e in all_evidence],
        "tool_histories": tool_histories,
        "node_outcomes": node_outcomes,
        "agent_trace": trace,
    }


def browser_node(state: ResearchState) -> dict:
    """
    Browser Agent: 执行浏览器工具（主题 3）。

    主题 3: 工具驱动多 Agent
    - 从 DAG 获取当前批次中类型为 browser 的节点
    - 执行 Playwright 浏览器工具
    - 支持深度页面提取

    输入: dag, current_executing_nodes
    输出: collected_evidence, tool_histories, agent_trace
    """
    from app.agents.browser import BrowserAgent
    from app.graph.state import AgentEvent, Evidence
    from app.guardrails import record_guardrail_event, validate_tool_invocation
    from app.observability.trace import EventType

    dag = deserialize_dag(state["dag"])
    executing_nodes = state.get("executing_nodes", state.get("current_executing_nodes", []))

    browser_nodes = [
        n for n in dag.nodes
        if n.node_id in executing_nodes and n.node_type == "browser"
    ]

    if not browser_nodes:
        return {"collected_evidence": [], "tool_histories": [], "agent_trace": [], "node_outcomes": []}

    trace = [
        _append_trace_event(state, EventType.AGENT_START, "browser", f"Executing {len(browser_nodes)} browser nodes")
    ]

    agent = BrowserAgent()
    all_evidence = []
    tool_histories = []
    node_outcomes = []
    budget = _research_budget(state)
    max_results = budget.get("browser_max_results", 2)

    for node in browser_nodes[:max_results]:
        trace.append(_append_trace_event(
            state,
            EventType.TOOL_START,
            "browser",
            f"Browsing: {node.query}",
            {"tool_name": "browse_webpage", "args": {"query": node.query, "url": node.query}},
        ))
        tool_history = _make_tool_history(
            agent_type=AgentType.BROWSER,
            tool_name="execute_browse",
            task_id=state.get("task_id"),
            query=node.query,
        )
        valid, reason = validate_tool_invocation("browse_webpage", {"url": node.query, "max_chars": 2000})
        if not valid:
            node.status = StepStatus.FAILED
            node.terminal_failure = True
            node.last_error = reason or "invalid_args"
            node.last_error_category = "schema_validation"
            _update_tool_safety(tool_history, schema_validated=False, sanitized_error=True)
            _finish_tool_history(tool_history, status="error", error=reason or "invalid_args")
            record_guardrail_event(
                state,
                agent="browser",
                event_type="tool_blocked",
                content=reason or "invalid_args",
                metadata={"tool": "browse_webpage", "url": node.query},
            )
            tool_histories.append(tool_history)
            node_outcomes.append(_make_node_outcome(
                node_id=node.node_id,
                tool_history=tool_history,
                tool_name="browse_webpage",
                status="terminal_error",
                retry_count=node.retry_count,
                error_category="schema_validation",
                error_message=reason or "invalid_args",
            ))
            continue
        require_approval, approval_reason = should_require_action_approval(
            tool_name="browse_webpage",
            decision=state.get("guardrail_decision") and build_guardrail_decision(
                state["user_query"],
                user_confirmed=state.get("user_confirmed", False),
            ),
            readonly=True,
        )
        if require_approval:
            approval = _make_approval_request(
                node_id=node.node_id,
                tool_name="browse_webpage",
                reason=approval_reason or "action_requires_approval",
                request_payload={"url": node.query, "max_chars": 2000},
            )
            node.waiting_approval = True
            _update_tool_safety(
                tool_history,
                schema_validated=True,
                approval_required=True,
                sanitized_error=True,
            )
            _finish_tool_history(tool_history, status="error", error=approval["reason"])
            tool_histories.append(tool_history)
            node_outcomes.append(_make_node_outcome(
                node_id=node.node_id,
                tool_history=tool_history,
                tool_name="browse_webpage",
                status="awaiting_approval",
                retry_count=node.retry_count,
                error_category="approval_required",
                error_message=approval["reason"],
                approval_request_id=approval["approval_id"],
            ))
            return {
                "collected_evidence": [e.model_dump() for e in all_evidence],
                "tool_histories": tool_histories,
                "node_outcomes": node_outcomes,
                "pending_approvals": [approval],
                "agent_trace": trace,
            }
        node.status = StepStatus.RUNNING

        try:
            results = agent.execute_browse(node.query)
            browser_error = next(
                (result.error_message for result in results if getattr(result, "error_message", None)),
                None,
            )
            has_content = any((result.extracted_content or "").strip() for result in results)
            if browser_error and not has_content:
                raise RuntimeError(browser_error)
            node.result = {"results": [r.model_dump() for r in results]}
            node.confidence = 0.85 if results else 0.0
            _finish_tool_history(
                tool_history,
                status="success",
                result_summary=f"{len(results)} browser pages",
            )
            _update_tool_safety(tool_history, schema_validated=True)
            node.status = StepStatus.DONE
            node.last_error = None
            node.last_error_category = None
            node.terminal_failure = False

            for r in results:
                evidence = Evidence(
                    content=r.extracted_content,
                    source_url=r.url,
                    source_title=r.title,
                    source_type="web",
                    collected_by=AgentType.BROWSER,
                )
                all_evidence.append(evidence)

            trace.append(_append_trace_event(
                state, EventType.TOOL_COMPLETE, "browser",
                f"Extracted {len(results)} pages for: {node.query}",
                {"tool_name": "browse_webpage", "status": "success", "result_summary": f"{len(results)} browser pages"},
            ))
            node_outcomes.append(_make_node_outcome(
                node_id=node.node_id,
                tool_history=tool_history,
                tool_name="browse_webpage",
                status="success",
                retry_count=node.retry_count,
                result_count=len(results),
            ))
        except (OSError, RuntimeError, TypeError, ValueError, KeyError) as e:
            node.status = StepStatus.FAILED
            node.retry_count += 1
            category = _classify_error_category(str(e))
            node.last_error = str(e)
            node.last_error_category = category
            node.terminal_failure = _is_terminal_error_category(category)
            trace.append(_append_trace_event(
                state,
                EventType.TOOL_ERROR,
                "browser",
                f"Browse failed: {e!s}",
                {"tool_name": "browse_webpage", "status": "error", "error": str(e)},
            ))
            _finish_tool_history(tool_history, status="error", error=str(e))
            node_outcomes.append(_make_node_outcome(
                node_id=node.node_id,
                tool_history=tool_history,
                tool_name="browse_webpage",
                status="terminal_error" if node.terminal_failure else "retryable_error",
                retry_count=node.retry_count,
                error_category=category,
                error_message=str(e),
            ))
        tool_histories.append(tool_history)

    trace.append(AgentEvent(
        agent="browser",
        event_type="agent_complete",
        content=f"Collected {len(all_evidence)} evidence from browser"
    ).model_dump())

    return {
        "collected_evidence": [e.model_dump() for e in all_evidence],
        "tool_histories": tool_histories,
        "node_outcomes": node_outcomes,
        "agent_trace": trace,
    }


def rag_node(state: ResearchState) -> dict:
    """
    RAG Agent: 执行混合检索工具（主题 3）。

    主题 3: 工具驱动多 Agent
    - 从 DAG 获取当前批次中类型为 rag 的节点
    - 执行混合检索（Rerank + Citation）
    - 使用 pgvector 向量检索

    输入: dag, current_executing_nodes
    输出: collected_evidence, tool_histories, agent_trace
    """
    from app.agents.rag import RAGAgent
    from app.graph.state import AgentEvent, Evidence
    from app.guardrails import record_guardrail_event, validate_tool_invocation
    from app.observability.trace import EventType

    dag = deserialize_dag(state["dag"])
    executing_nodes = state.get("executing_nodes", state.get("current_executing_nodes", []))

    rag_nodes = [
        n for n in dag.nodes
        if n.node_id in executing_nodes and n.node_type == "rag"
    ]

    if not rag_nodes:
        return {"collected_evidence": [], "tool_histories": [], "agent_trace": [], "node_outcomes": []}

    trace = [
        _append_trace_event(state, EventType.AGENT_START, "rag", f"Executing {len(rag_nodes)} RAG nodes")
    ]

    agent = RAGAgent()
    all_evidence = []
    tool_histories = []
    node_outcomes = []
    budget = _research_budget(state)
    max_results = budget.get("rag_max_results", 8)

    for node in rag_nodes[:1]:
        trace.append(_append_trace_event(
            state,
            EventType.TOOL_START,
            "rag",
            f"Retrieving: {node.query}",
            {"tool_name": "knowledge_base_search", "args": {"query": node.query}},
        ))
        tool_history = _make_tool_history(
            agent_type=AgentType.RAG,
            tool_name="execute_retrieval",
            task_id=state.get("task_id"),
            query=node.query,
        )
        valid, reason = validate_tool_invocation("knowledge_base_search", {"query": node.query, "top_k": 10})
        if not valid:
            node.status = StepStatus.FAILED
            node.terminal_failure = True
            node.last_error = reason or "invalid_args"
            node.last_error_category = "schema_validation"
            _update_tool_safety(tool_history, schema_validated=False, sanitized_error=True)
            _finish_tool_history(tool_history, status="error", error=reason or "invalid_args")
            record_guardrail_event(
                state,
                agent="rag",
                event_type="tool_blocked",
                content=reason or "invalid_args",
                metadata={"tool": "knowledge_base_search", "query": node.query},
            )
            tool_histories.append(tool_history)
            node_outcomes.append(_make_node_outcome(
                node_id=node.node_id,
                tool_history=tool_history,
                tool_name="knowledge_base_search",
                status="terminal_error",
                retry_count=node.retry_count,
                error_category="schema_validation",
                error_message=reason or "invalid_args",
            ))
            continue
        require_approval, approval_reason = should_require_action_approval(
            tool_name="knowledge_base_search",
            decision=state.get("guardrail_decision") and build_guardrail_decision(
                state["user_query"],
                user_confirmed=state.get("user_confirmed", False),
            ),
            readonly=True,
        )
        if require_approval:
            approval = _make_approval_request(
                node_id=node.node_id,
                tool_name="knowledge_base_search",
                reason=approval_reason or "action_requires_approval",
                request_payload={"query": node.query, "top_k": 10},
            )
            node.waiting_approval = True
            _update_tool_safety(
                tool_history,
                schema_validated=True,
                approval_required=True,
                sanitized_error=True,
            )
            _finish_tool_history(tool_history, status="error", error=approval["reason"])
            tool_histories.append(tool_history)
            node_outcomes.append(_make_node_outcome(
                node_id=node.node_id,
                tool_history=tool_history,
                tool_name="knowledge_base_search",
                status="awaiting_approval",
                retry_count=node.retry_count,
                error_category="approval_required",
                error_message=approval["reason"],
                approval_request_id=approval["approval_id"],
            ))
            return {
                "collected_evidence": [e.model_dump() for e in all_evidence],
                "tool_histories": tool_histories,
                "node_outcomes": node_outcomes,
                "pending_approvals": [approval],
                "agent_trace": trace,
            }
        node.status = StepStatus.RUNNING

        try:
            results = agent.execute_retrieval(
                node.query,
                state["user_query"],
                group=state.get("rag_group"),
                owner_id=state.get("session", {}).get("owner_id"),
            )[:max_results]
            node.result = {"results": [r.model_dump() for r in results]}
            node.confidence = 0.88 if results else 0.0
            _finish_tool_history(
                tool_history,
                status="success",
                result_summary=f"{len(results)} rag chunks",
            )
            _update_tool_safety(tool_history, schema_validated=True)
            node.status = StepStatus.DONE
            node.last_error = None
            node.last_error_category = None
            node.terminal_failure = False

            for r in results:
                evidence = Evidence(
                    content=r.content,
                    source_url=r.metadata.get("url") if r.metadata else None,
                    source_title=r.metadata.get("title") if r.metadata else None,
                    source_type="knowledge_base",
                    collected_by=AgentType.RAG,
                )
                all_evidence.append(evidence)

            trace.append(_append_trace_event(
                state, EventType.TOOL_COMPLETE, "rag",
                f"Retrieved {len(results)} chunks for: {node.query}",
                {"tool_name": "knowledge_base_search", "status": "success", "result_summary": f"{len(results)} rag chunks"},
            ))
            node_outcomes.append(_make_node_outcome(
                node_id=node.node_id,
                tool_history=tool_history,
                tool_name="knowledge_base_search",
                status="success",
                retry_count=node.retry_count,
                result_count=len(results),
            ))
        except (OSError, RuntimeError, TypeError, ValueError, KeyError) as e:
            node.status = StepStatus.FAILED
            node.retry_count += 1
            category = _classify_error_category(str(e))
            node.last_error = str(e)
            node.last_error_category = category
            node.terminal_failure = _is_terminal_error_category(category)
            trace.append(_append_trace_event(
                state,
                EventType.TOOL_ERROR,
                "rag",
                f"RAG retrieval failed: {e!s}",
                {"tool_name": "knowledge_base_search", "status": "error", "error": str(e)},
            ))
            _finish_tool_history(tool_history, status="error", error=str(e))
            node_outcomes.append(_make_node_outcome(
                node_id=node.node_id,
                tool_history=tool_history,
                tool_name="knowledge_base_search",
                status="terminal_error" if node.terminal_failure else "retryable_error",
                retry_count=node.retry_count,
                error_category=category,
                error_message=str(e),
            ))
        tool_histories.append(tool_history)

    trace.append(AgentEvent(
        agent="rag",
        event_type="agent_complete",
        content=f"Collected {len(all_evidence)} evidence from RAG"
    ).model_dump())

    return {
        "collected_evidence": [e.model_dump() for e in all_evidence],
        "tool_histories": tool_histories,
        "node_outcomes": node_outcomes,
        "agent_trace": trace,
    }


# ==============================================================
# 主题 1 & 3: 批量执行 - Fan-out / Fan-in
# ==============================================================

def execute_tool_batch(state: ResearchState) -> list[Send]:
    """
    批量执行工具（主题 1 & 3）。

    主题 1: 自主工作流的并行执行
    主题 3: 工具驱动的多 Agent 协作

    从当前批次节点中，根据节点类型分发到对应的工具 Agent。
    返回 Send 列表，实现并行执行。
    """
    executing_nodes = state.get("current_executing_nodes", [])

    if not executing_nodes:
        return []

    dag = deserialize_dag(state["dag"])
    decision = state.get("guardrail_decision") or {}
    enabled_tools = _effective_tool_allowlist(state, decision)
    sends = []

    # 子节点在 LangGraph 中接收到的是 Send payload，不一定包含完整 state。
    # 显式带上后续节点需要的上下文，避免子节点丢失 dag / query / session。
    node_payload = {
        "dag": state.get("dag"),
        "task_id": state.get("task_id"),
        "user_query": state.get("user_query"),
        "session": state.get("session", {}),
        "guardrail_decision": state.get("guardrail_decision"),
        "allow_web_after_rag_hit": state.get("allow_web_after_rag_hit", False),
        "rag_group": state.get("rag_group"),
        "retrieval_policy": state.get("retrieval_policy"),
    }

    for node_id in executing_nodes:
        node = next((n for n in dag.nodes if n.node_id == node_id), None)
        if not node:
            continue

        # 根据节点类型分发
        if node.node_type in ("search", "browser", "rag", "mcp") and node.node_type in enabled_tools:
            sends.append(Send(node.node_type, {
                **node_payload,
                "executing_nodes": [node_id],
                "current_executing_nodes": [node_id],
            }))

    return sends


# ==============================================================
# 主题 1 & 5: Analyst / Reflection / Report Nodes
# ==============================================================

def analyst_node(state: ResearchState) -> dict:
    """
    Analyst Agent: 综合证据生成分析（主题 1）。

    主题 1: 自主工作流中的分析阶段
    输入所有收集到的证据，生成结构化分析

    输入: collected_evidence
    输出: analysis, agent_trace
    """
    from app.agents.analyst import AnalystAgent
    from app.graph.state import AgentEvent, deserialize_evidence
    from app.observability.trace import EventType

    evidence_list = [deserialize_evidence(e) for e in state.get("collected_evidence", [])]
    evidence_gate = build_evidence_gate(evidence_list)
    decision = state.get("guardrail_decision") or {}
    reject_if_no_evidence = bool(decision.get("reject_if_no_evidence", True))

    start_event = _append_trace_event(
        state, EventType.AGENT_START, "analyst",
        f"Analyzing {len(evidence_list)} evidence items"
    )

    if not evidence_gate.allowed and reject_if_no_evidence:
        message = build_answer_gate_message(evidence_gate)
        record_guardrail_event(
            state,
            agent="analyst",
            event_type="answer_gate",
            content=message,
            metadata=evidence_gate.model_dump(),
        )
        trace = [start_event, AgentEvent(
            agent="analyst",
            event_type="agent_complete",
            content=message,
        ).model_dump()]
        return {
            "analysis": message,
            "evidence_status": evidence_gate.model_dump(),
            "revision_needed": False,
            "agent_trace": trace,
        }

    agent = AnalystAgent()
    analysis_text = agent.analyze(
        state["user_query"],
        evidence_list,
        skill_prompt=_skill_prompt(state) or None,
        analyst_hints=_agent_skill_hints(state, "analyst"),
    )
    session, budget_state = _apply_llm_usage_to_state(state, "analyst", agent.last_usage)

    complete_content = (
        "Analysis complete without external evidence"
        if not evidence_gate.allowed
        else "Analysis complete"
    )
    complete_event = _append_trace_event(state, EventType.AGENT_COMPLETE, "analyst", complete_content)

    trace = [start_event, complete_event, AgentEvent(
        agent="analyst",
        event_type="agent_complete",
        content=f"Generated {len(analysis_text)} chars of analysis from {len(evidence_list)} evidence items"
    ).model_dump()]

    return {
        "analysis": analysis_text,
        "evidence_status": evidence_gate.model_dump(),
        "session": session,
        "budget_state": budget_state,
        "agent_trace": trace,
    }


def reflection_node(state: ResearchState) -> dict:
    """
    Reflection Agent: 自校验与验证（主题 1 & 5）。

    主题 5: Self-Reflection & Verification
    - 校验分析的事实性、一致性、完整性
    - 检测幻觉声明
    - 决定是否需要重规划

    输入: analysis, collected_evidence
    输出: verification, revision_needed, agent_trace
    """
    from app.agents.reflection import ReflectionAgent
    from app.graph.state import AgentEvent, deserialize_evidence
    from app.observability.trace import EventType, emit_event

    evidence_list = [deserialize_evidence(e) for e in state.get("collected_evidence", [])]

    emit_event(
        state, EventType.AGENT_START, "reflection",
        f"Validating analysis with {len(evidence_list)} evidence items"
    )

    agent = ReflectionAgent()
    verification: VerificationResult = agent.reflect(
        state["user_query"],
        state["analysis"],
        evidence_list,
        skill_prompt=_skill_prompt(state) or None,
        reflection_hints=_agent_skill_hints(state, "reflection"),
    )
    session, budget_state = _apply_llm_usage_to_state(state, "reflection", agent.last_usage)

    # 追踪校验结果
    if verification.needs_revision:
        emit_event(
            state, EventType.AGENT_COMPLETE, "reflection",
            f"Verification failed: confidence={verification.overall_confidence:.2f}, "
            f"hallucination_rate={verification.hallucination_rate:.2%}, "
            f"needs_revision=True"
        )
    else:
        emit_event(
            state, EventType.AGENT_COMPLETE, "reflection",
            f"Verification passed: confidence={verification.overall_confidence:.2f}, "
            f"citation_coverage={verification.citation_coverage:.2%}"
        )

    trace = [AgentEvent(
        agent="reflection",
        event_type="agent_complete",
        content=f"Verification: confidence={verification.overall_confidence:.2f}, "
                f"needs_revision={verification.needs_revision}"
    ).model_dump()]

    return {
        "verification": verification.model_dump(),
        "revision_needed": verification.needs_revision,
        "session": session,
        "budget_state": budget_state,
        "agent_trace": trace,
    }


def should_revise(state: ResearchState) -> str:
    """
    条件路由：判断是否需要重规划（主题 5）。

    主题 5: 自校验失败 → 重规划循环
    - 检查 verification.needs_revision
    - 检查 revision_count < max_revisions
    """
    revision_count = state.get("revision_count", 0)
    max_revisions = state.get("session", {}).get("max_revisions", 3)
    budget_state = state.get("budget_state") or {}

    if budget_state.get("hard_stop_reason"):
        return "generate_report"
    if state.get("revision_needed") and revision_count < max_revisions:
        return "replan"
    return "generate_report"


def replan_node(state: ResearchState) -> dict:
    """
    重规划节点：增加重规划计数，重置 DAG（主题 1 & 5）。

    主题 5: 重新调用 Planner Agent 生成新的 DAG
    主题 1: 自主工作流的迭代循环
    """
    revision_count = state.get("revision_count", 0) + 1

    session = state.get("session", {})
    session["revision_count"] = revision_count
    session["updated_at"] = utc_now_naive().isoformat()

    from app.agents.planner import PlannerAgent

    failure_memory = state.get("failure_memory")
    planning_hints = _format_failure_memory_notes(failure_memory)
    agent = PlannerAgent()
    dag = agent.create_dag(
        state["user_query"],
        planning_hints=planning_hints or None,
        skill_prompt=_skill_prompt(state) or None,
        planner_hints=_agent_skill_hints(state, "planner"),
    )
    dag = _enforce_internal_first_dag(dag, state["user_query"])
    session, budget_state = _apply_llm_usage_to_state(state, "replan", agent.last_usage)

    from app.observability.trace import EventType, emit_event
    emit_event(
        state, EventType.AGENT_START, "replan",
        f"Replanning (attempt {revision_count})"
    )

    return {
        "revision_count": revision_count,
        "session": session,
        "dag": serialize_dag(dag),
        "completed_nodes": [],
        "current_executing_nodes": [],
        "collected_evidence": [],
        "analysis": "",
        "node_outcomes": [],
        "failure_memory": failure_memory,
        "budget_state": budget_state,
    }


async def report_node(state: ResearchState) -> dict:
    """
    Report Generator: 生成最终报告（主题 1）。

    主题 1: 自主工作流的输出阶段
    - 基于分析、证据、引用生成 Markdown 报告
    - 添加置信度和来源信息

    输入: analysis, collected_evidence, verification
    输出: final_report, status
    """
    from app.agents.report import ReportAgent
    from app.graph.state import AgentEvent, deserialize_evidence
    from app.llm_client import StreamInterruptedError
    from app.observability.sse_manager import get_sse_manager
    from app.observability.trace import EventType

    evidence_list = [deserialize_evidence(e) for e in state.get("collected_evidence", [])]
    evidence_gate = build_evidence_gate(evidence_list)
    decision = state.get("guardrail_decision") or {}
    reject_if_no_evidence = bool(decision.get("reject_if_no_evidence", True))
    verification = state.get("verification")
    session_id = state.get("task_id")
    sse = get_sse_manager()
    pending_tasks = []

    start_event = _append_trace_event(state, EventType.AGENT_START, "report", "Generating final report")

    if not evidence_gate.allowed and reject_if_no_evidence:
        message = build_answer_gate_message(evidence_gate)
        record_guardrail_event(
            state,
            agent="report",
            event_type="answer_gate",
            content=message,
            metadata=evidence_gate.model_dump(),
        )
        trace = [start_event, AgentEvent(
            agent="report",
            event_type="agent_complete",
            content=message,
        ).model_dump()]
        session = state.get("session", {})
        session["status"] = TaskStatus.COMPLETED.value
        session["completed_at"] = utc_now_naive().isoformat()
        return {
            "final_report": f"# Research Report\n\n{message}\n",
            "citations": [],
            "status": TaskStatus.COMPLETED.value,
            "session": session,
            "review_status": {
                "blocked": True,
                **evidence_gate.model_dump(),
            },
            "agent_trace": trace,
        }

    agent = ReportAgent()

    def on_chunk(chunk: str) -> None:
        if session_id:
            pending_tasks.append(asyncio.create_task(
                sse.publish(session_id, EventType.REPORT_CHUNK.value, {
                    "agent": "report",
                    "chunk": chunk,
                    "is_partial": True,
                })
            ))

    def on_citation(citation) -> None:
        if session_id:
            pending_tasks.append(asyncio.create_task(
                sse.publish(session_id, EventType.REPORT_CITATION.value, {
                    "agent": "report",
                    "citation_id": citation.citation_id,
                    "source": citation.source_title or citation.source_url or "",
                    "source_url": citation.source_url,
                    "source_title": citation.source_title,
                })
            ))

    try:
        report, citations = agent.generate_stream(
            user_query=state["user_query"],
            analysis=state["analysis"],
            evidence_list=evidence_list,
            reflection=verification,
            output_length=state.get("output_length") or state.get("session", {}).get("output_length"),
            skill_prompt=_skill_prompt(state) or None,
            report_hints=_agent_skill_hints(state, "report"),
            on_chunk=on_chunk,
            on_citation=on_citation,
        )
    except StreamInterruptedError:
        if pending_tasks:
            await asyncio.gather(*pending_tasks)
        session = state.get("session", {})
        session["status"] = TaskStatus.FAILED.value
        return {
            "status": TaskStatus.FAILED.value,
            "session": session,
            "review_status": {
                "blocked": False,
                "last_error_category": "llm_stream_interrupted",
                "retryable": True,
            },
            "errors": ["llm_stream_interrupted_after_output"],
            "agent_trace": [start_event, AgentEvent(
                agent="report",
                event_type="agent_error",
                content="LLM stream interrupted after output; retry requires a new report request.",
            ).model_dump()],
        }
    session, budget_state = _apply_llm_usage_to_state(state, "report", agent.last_usage)

    if pending_tasks:
        await asyncio.gather(*pending_tasks)

    # 更新会话状态
    session["status"] = TaskStatus.COMPLETED.value
    session["completed_at"] = utc_now_naive().isoformat()

    complete_event = _append_trace_event(state, EventType.AGENT_COMPLETE, "report", f"Report generated: {len(report)} chars")

    trace = [start_event, complete_event, AgentEvent(
        agent="report",
        event_type="agent_complete",
        content=f"Report: {len(report)} chars, {len(citations)} citations"
    ).model_dump()]

    return {
        "final_report": report,
        "citations": [c.model_dump() for c in citations],
        "status": TaskStatus.COMPLETED.value,
        "session": session,
        "budget_state": budget_state,
        "review_status": {
            "blocked": False,
            **evidence_gate.model_dump(),
            "verification": verification,
        },
        "agent_trace": trace,
    }


# ==============================================================
# 主题 1: Graph 编译
# ==============================================================

def compile_research_graph() -> StateGraph:
    """
    编译并返回完整的研究 StateGraph（主题 1）。

    架构（5 大主题的整合）：

    START → planner (生成 DAG)
           ↓ (主题 2: DAG 生成)
      dag_executor (分发节点批次)
           ↓ (Send API: 主题 1 & 3 并行执行)
      search / browser / rag (工具调用)
           ↓ (主题 2: 结果聚合)
      should_continue_dag (条件路由)
           ↓ continue / analyst
      analyst (综合分析)
           ↓ (主题 5: 自校验)
      reflection (验证)
           ↓ (主题 5: 条件分支)
      replan ← (重规划循环，最多 3 次)
           ↓ generate_report
      report (报告生成)
           ↓
          END
    """
    settings = get_settings()

    # 构建图
    builder = StateGraph(ResearchState)

    # === 节点定义 ===
    builder.add_node("planner", planner_node)
    builder.add_node("dag_executor", dag_executor_node)
    builder.add_node("dag_aggregator", dag_results_aggregator)
    builder.add_node("approval_reviewer", approval_reviewer_node)
    builder.add_node("analyst", analyst_node)
    builder.add_node("reflection", reflection_node)
    builder.add_node("replan", replan_node)
    builder.add_node("report", report_node)

    # 工具节点（通过 Send API 调用）
    builder.add_node("search", search_node)
    builder.add_node("browser", browser_node)
    builder.add_node("rag", rag_node)
    builder.add_node("mcp", mcp_node)

    # === 边定义 ===
    # 入口
    builder.add_edge(START, "planner")

    # Planner → DAG 执行器
    builder.add_edge("planner", "dag_executor")

    # DAG 执行器 → 工具节点（并行）
    builder.add_conditional_edges(
        "dag_executor",
        execute_tool_batch,
        ["search", "browser", "rag", "mcp"],
    )

    # 工具节点 → 结果聚合
    builder.add_edge("search", "dag_aggregator")
    builder.add_edge("browser", "dag_aggregator")
    builder.add_edge("rag", "dag_aggregator")
    builder.add_edge("mcp", "dag_aggregator")

    # 结果聚合 → 判断是否继续 DAG 或进入分析
    builder.add_conditional_edges(
        "dag_aggregator",
        should_continue_dag,
        {
            "continue": "dag_executor",  # 继续执行下一批节点
            "approval_reviewer": "approval_reviewer",
            "analyst": "analyst",         # DAG 全部完成，进入分析
        },
    )

    builder.add_edge("approval_reviewer", END)

    # 分析 → 反思
    builder.add_edge("analyst", "reflection")

    # 反思 → 重规划或报告（条件分支）
    builder.add_conditional_edges(
        "reflection",
        should_revise,
        {
            "replan": "replan",
            "generate_report": "report",
        },
    )

    # 重规划 → DAG 执行器（重新开始）
    builder.add_edge("replan", "dag_executor")

    # 报告 → 结束
    builder.add_edge("report", END)

    # === 检查点配置（主题 4）===
    checkpointer = None

    if settings.redis.url:
        try:
            import redis as sync_redis
            from langgraph.checkpoint.redis import RedisSaver
            r = sync_redis.from_url(settings.redis.url)
            checkpointer = RedisSaver(r)
        except (ImportError, OSError, RuntimeError, TypeError, ValueError, KeyError):
            logger.warning("Redis checkpointer is unavailable; using in-memory checkpoints")

    if not checkpointer:
        from langgraph.checkpoint.memory import MemorySaver
        checkpointer = MemorySaver()

    return builder.compile(checkpointer=checkpointer)
async def mcp_node(state: ResearchState) -> dict:
    """MCP tool node guarded by McpPolicyProxy."""
    from app.graph.state import AgentEvent
    from app.observability.trace import EventType

    dag = deserialize_dag(state["dag"])
    executing_nodes = state.get("executing_nodes", state.get("current_executing_nodes", []))
    mcp_nodes = [n for n in dag.nodes if n.node_id in executing_nodes and n.node_type == "mcp"]
    if not mcp_nodes:
        return {"collected_evidence": [], "tool_histories": [], "agent_trace": [], "node_outcomes": []}

    proxy = McpPolicyProxy()
    trace = [
        _append_trace_event(state, EventType.AGENT_START, "mcp", f"Executing {len(mcp_nodes)} MCP nodes")
    ]
    tool_histories = []
    node_outcomes = []
    for node in mcp_nodes:
        tool_history = _make_tool_history(
            agent_type=AgentType.BROWSER,
            tool_name="mcp_proxy",
            task_id=state.get("task_id"),
            query=node.query,
        )
        request = McpToolRequest(
            session_id=state.get("task_id"),
            decision_id=None,
            tool_name=node.query.split(":", 1)[-1] if ":" in node.query else node.query,
            requested_args={"query": node.query},
            server_id=node.query.split(":", 1)[0] if ":" in node.query else "default",
        )
        result = await proxy.invoke(request)
        if result.status == "awaiting_approval":
            approval = result.approval_request or _make_approval_request(
                node_id=node.node_id,
                tool_name=request.tool_name,
                reason="mcp_write_requires_approval",
                request_payload={"server_id": request.server_id, "args": request.requested_args},
            )
            _update_tool_safety(
                tool_history,
                approval_required=True,
                sanitized_error=True,
                read_only_default=False,
            )
            _finish_tool_history(tool_history, status="error", error=approval["reason"])
            tool_histories.append(tool_history)
            node_outcomes.append(_make_node_outcome(
                node_id=node.node_id,
                tool_history=tool_history,
                tool_name=request.tool_name,
                status="awaiting_approval",
                retry_count=node.retry_count,
                error_category="approval_required",
                error_message=approval["reason"],
                approval_request_id=approval["approval_id"],
            ))
            return {
                "tool_histories": tool_histories,
                "node_outcomes": node_outcomes,
                "pending_approvals": [approval],
                "agent_trace": trace,
            }
        if result.status == "terminal_error":
            _update_tool_safety(
                tool_history,
                sanitized_error=True,
                redaction_applied=result.redaction_applied,
            )
            _finish_tool_history(tool_history, status="error", error=result.error_message or result.error_category or "mcp_error")
            tool_histories.append(tool_history)
            node_outcomes.append(_make_node_outcome(
                node_id=node.node_id,
                tool_history=tool_history,
                tool_name=request.tool_name,
                status="terminal_error",
                retry_count=node.retry_count,
                error_category=result.error_category,
                error_message=result.error_message,
            ))
            continue
        _finish_tool_history(tool_history, status="success", result_summary=result.result_summary or "mcp success")
        _update_tool_safety(
            tool_history,
            schema_validated=True,
            redaction_applied=result.redaction_applied,
            server_trusted=True,
        )
        tool_history["tool_calls"][-1]["server_fingerprint"] = result.server_fingerprint
        tool_histories.append(tool_history)
        node_outcomes.append(_make_node_outcome(
            node_id=node.node_id,
            tool_history=tool_history,
            tool_name=request.tool_name,
            status="success",
            retry_count=node.retry_count,
            result_count=1 if result.payload is not None else 0,
        ))
        trace.append(AgentEvent(
            agent="mcp",
            event_type="tool_complete",
            content=f"MCP tool {request.tool_name} succeeded",
        ).model_dump())
    return {
        "collected_evidence": [],
        "tool_histories": tool_histories,
        "node_outcomes": node_outcomes,
        "agent_trace": trace,
    }
