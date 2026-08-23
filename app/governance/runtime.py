from __future__ import annotations

import hashlib
from typing import Any

from app.core.time import utc_now_naive
from app.db.connection import get_db_pool
from app.db.json import dumps_json
from app.graph.state import RuntimeStatus


def _stable_json_hash(value: Any) -> str | None:
    if value is None:
        return None
    payload = dumps_json(value)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def public_status_from_runtime(runtime_status: str | None) -> str:
    if runtime_status in (
        RuntimeStatus.PENDING.value,
        RuntimeStatus.APPROVED.value,
        RuntimeStatus.AWAITING_APPROVAL.value,
    ):
        return "pending"
    if runtime_status in (
        RuntimeStatus.RUNNING.value,
        RuntimeStatus.RETRYABLE_FAILED.value,
    ):
        return "running"
    if runtime_status in (
        RuntimeStatus.TERMINAL_FAILED.value,
        RuntimeStatus.CANCELED.value,
    ):
        return "failed"
    if runtime_status == RuntimeStatus.COMPLETED.value:
        return "completed"
    return "running"


def build_runtime_review_status(
    *,
    review_status: dict[str, Any] | None,
    runtime_status: str,
    pending_approval_count: int,
    budget_state: dict[str, Any] | None,
    last_error_category: str | None = None,
    tool_audit_summary: dict[str, Any] | None = None,
) -> dict[str, Any]:
    merged = dict(review_status or {})
    merged["runtime_status"] = runtime_status
    merged["pending_approval_count"] = pending_approval_count
    if budget_state is not None:
        merged["budget_state"] = budget_state
    if last_error_category:
        merged["last_error_category"] = last_error_category
    if tool_audit_summary is not None:
        merged["tool_audit_summary"] = tool_audit_summary
    return merged


def normalize_tool_audit_rows(
    *,
    session_id: str,
    tool_histories: list[dict[str, Any]],
    node_outcomes: list[dict[str, Any]],
) -> list[tuple[Any, ...]]:
    outcomes_by_call_id: dict[str, dict[str, Any]] = {}
    for outcome in node_outcomes:
        if not isinstance(outcome, dict):
            continue
        call_id = outcome.get("tool_call_id")
        if call_id:
            outcomes_by_call_id[str(call_id)] = outcome

    rows: list[tuple[Any, ...]] = []
    for history in tool_histories:
        if not isinstance(history, dict):
            continue
        agent_type = history.get("agent_type")
        for call in history.get("tool_calls", []):
            if not isinstance(call, dict):
                continue
            call_id = str(call.get("call_id"))
            if not call_id:
                continue
            outcome = outcomes_by_call_id.get(call_id, {})
            args_json = call.get("args") or {}
            result_summary = call.get("result_summary")
            safety_json = call.get("safety_json") or {}
            usage_source = "provider"
            estimated = False
            if call.get("usage_estimated") or outcome.get("usage_estimated"):
                usage_source = "estimated"
                estimated = True
            rows.append((
                call_id,
                session_id,
                outcome.get("node_id"),
                agent_type,
                call.get("tool_name"),
                dumps_json(args_json),
                _stable_json_hash(args_json),
                call.get("status") or "pending",
                outcome.get("error_category"),
                outcome.get("error_message") or call.get("error"),
                int(outcome.get("retry_count") or 0),
                result_summary,
                _stable_json_hash(result_summary),
                int(call.get("tokens_used") or outcome.get("tokens_used") or 0),
                float(call.get("cost_usd") or outcome.get("cost_usd") or 0.0),
                call.get("decision_id"),
                call.get("approved_by"),
                call.get("server_fingerprint"),
                dumps_json(safety_json),
                usage_source,
                estimated,
                call.get("started_at"),
                call.get("completed_at"),
            ))
    return rows


class RuntimePersistence:
    """Crash-safe persistence for runtime state, audits, approvals, and budgets."""

    async def persist_runtime_snapshot(
        self,
        *,
        session_id: str,
        state: dict[str, Any],
        current_batch: list[str] | None = None,
        checkpoint_ref: str | None = None,
    ) -> None:
        pool = await get_db_pool()
        runtime_status = state.get("runtime_status", RuntimeStatus.RUNNING.value)
        public_status = public_status_from_runtime(runtime_status)
        budget_state = dict(state.get("budget_state") or {})
        pending_approvals = [
            approval for approval in state.get("pending_approvals", [])
            if isinstance(approval, dict)
        ]
        tool_histories = [
            history for history in state.get("tool_histories", [])
            if isinstance(history, dict)
        ]
        node_outcomes = [
            outcome for outcome in state.get("node_outcomes", [])
            if isinstance(outcome, dict)
        ]

        tool_audit_rows = normalize_tool_audit_rows(
            session_id=session_id,
            tool_histories=tool_histories,
            node_outcomes=node_outcomes,
        )
        tool_call_total = sum(
            len(history.get("tool_calls", []))
            for history in tool_histories
        )
        tool_call_errors = sum(
            sum(1 for call in history.get("tool_calls", []) if call.get("status") == "error")
            for history in tool_histories
        )
        last_error_category = None
        for outcome in reversed(node_outcomes):
            if outcome.get("error_category"):
                last_error_category = outcome.get("error_category")
                break

        review_status = build_runtime_review_status(
            review_status=state.get("review_status"),
            runtime_status=runtime_status,
            pending_approval_count=len(pending_approvals),
            budget_state=budget_state,
            last_error_category=last_error_category,
            tool_audit_summary={
                "total_calls": tool_call_total,
                "error_calls": tool_call_errors,
                "persisted_calls": len(tool_audit_rows),
            },
        )

        now = utc_now_naive()
        async with pool.acquire() as conn, conn.transaction():
                await conn.execute(
                    """
                    INSERT INTO session_runtime_state (
                        session_id,
                        runtime_status,
                        public_status,
                        current_batch_json,
                        retryable_failure_count,
                        terminal_failure_reason,
                        pending_approval_count,
                        checkpoint_seq,
                        last_checkpoint_ref,
                        last_heartbeat_at,
                        harness_state_version
                    )
                    VALUES (
                        $1::uuid, $2, $3, $4::jsonb, $5, $6, $7, $8, $9, $10, $11
                    )
                    ON CONFLICT (session_id) DO UPDATE SET
                        runtime_status = EXCLUDED.runtime_status,
                        public_status = EXCLUDED.public_status,
                        current_batch_json = EXCLUDED.current_batch_json,
                        retryable_failure_count = EXCLUDED.retryable_failure_count,
                        terminal_failure_reason = EXCLUDED.terminal_failure_reason,
                        pending_approval_count = EXCLUDED.pending_approval_count,
                        checkpoint_seq = EXCLUDED.checkpoint_seq,
                        last_checkpoint_ref = EXCLUDED.last_checkpoint_ref,
                        last_heartbeat_at = EXCLUDED.last_heartbeat_at,
                        harness_state_version = EXCLUDED.harness_state_version
                    """,
                    session_id,
                    runtime_status,
                    public_status,
                    dumps_json(current_batch or state.get("current_executing_nodes", [])),
                    sum(1 for outcome in node_outcomes if outcome.get("status") == "retryable_error"),
                    budget_state.get("hard_stop_reason") or last_error_category,
                    len(pending_approvals),
                    int(state.get("session", {}).get("checkpoint_seq", 0) or 0),
                    checkpoint_ref,
                    now,
                    int(state.get("session", {}).get("harness_state_version", 1) or 1),
                )
                await conn.execute(
                    """
                    INSERT INTO session_budget_state (
                        session_id,
                        max_total_tokens,
                        max_cost_usd,
                        max_tool_calls,
                        max_wall_clock_seconds,
                        max_retries_per_tool,
                        used_total_tokens,
                        used_cost_usd,
                        used_tool_calls,
                        elapsed_wall_clock_seconds,
                        hard_stop_reason,
                        estimated_usage_count,
                        updated_at
                    )
                    VALUES ($1::uuid, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13)
                    ON CONFLICT (session_id) DO UPDATE SET
                        max_total_tokens = EXCLUDED.max_total_tokens,
                        max_cost_usd = EXCLUDED.max_cost_usd,
                        max_tool_calls = EXCLUDED.max_tool_calls,
                        max_wall_clock_seconds = EXCLUDED.max_wall_clock_seconds,
                        max_retries_per_tool = EXCLUDED.max_retries_per_tool,
                        used_total_tokens = EXCLUDED.used_total_tokens,
                        used_cost_usd = EXCLUDED.used_cost_usd,
                        used_tool_calls = EXCLUDED.used_tool_calls,
                        elapsed_wall_clock_seconds = EXCLUDED.elapsed_wall_clock_seconds,
                        hard_stop_reason = EXCLUDED.hard_stop_reason,
                        estimated_usage_count = EXCLUDED.estimated_usage_count,
                        updated_at = EXCLUDED.updated_at
                    """,
                    session_id,
                    int(budget_state.get("max_total_tokens", 0) or 0),
                    float(budget_state.get("max_cost_usd", 0.0) or 0.0),
                    int(budget_state.get("max_tool_calls", 0) or 0),
                    int(budget_state.get("max_wall_clock_seconds", 0) or 0),
                    int(budget_state.get("max_retries_per_tool", 0) or 0),
                    int(budget_state.get("used_total_tokens", 0) or 0),
                    float(budget_state.get("used_cost_usd", 0.0) or 0.0),
                    int(budget_state.get("used_tool_calls", 0) or 0),
                    int(budget_state.get("elapsed_wall_clock_seconds", 0) or 0),
                    budget_state.get("hard_stop_reason"),
                    int(sum(1 for usage in budget_state.get("llm_usage", []) if usage.get("estimated"))),
                    now,
                )
                await conn.execute(
                    """
                    UPDATE research_sessions
                    SET status = $1,
                        review_status = $2::jsonb,
                        total_tokens = $3,
                        total_cost_usd = $4,
                        updated_at = $5
                    WHERE id = $6::uuid
                    """,
                    public_status,
                    dumps_json(review_status),
                    int(state.get("session", {}).get("total_tokens", budget_state.get("used_total_tokens", 0)) or 0),
                    float(state.get("session", {}).get("total_cost_usd", budget_state.get("used_cost_usd", 0.0)) or 0.0),
                    now,
                    session_id,
                )
                if tool_audit_rows:
                    await conn.executemany(
                        """
                        INSERT INTO tool_call_audit (
                            call_id,
                            session_id,
                            node_id,
                            agent_type,
                            tool_name,
                            args_json,
                            args_hash,
                            status,
                            error_category,
                            error_message,
                            retry_count,
                            result_summary,
                            result_hash,
                            tokens_used,
                            cost_usd,
                            decision_id,
                            approved_by,
                            server_fingerprint,
                            safety_json,
                            usage_source,
                            estimated,
                            started_at,
                            completed_at
                        )
                        VALUES (
                            $1, $2::uuid, $3, $4, $5, $6::jsonb, $7, $8, $9, $10,
                            $11, $12, $13, $14, $15, $16, $17, $18, $19::jsonb, $20, $21, $22, $23
                        )
                        ON CONFLICT (call_id) DO UPDATE SET
                            node_id = EXCLUDED.node_id,
                            agent_type = EXCLUDED.agent_type,
                            tool_name = EXCLUDED.tool_name,
                            args_json = EXCLUDED.args_json,
                            args_hash = EXCLUDED.args_hash,
                            status = EXCLUDED.status,
                            error_category = EXCLUDED.error_category,
                            error_message = EXCLUDED.error_message,
                            retry_count = EXCLUDED.retry_count,
                            result_summary = EXCLUDED.result_summary,
                            result_hash = EXCLUDED.result_hash,
                            tokens_used = EXCLUDED.tokens_used,
                            cost_usd = EXCLUDED.cost_usd,
                            decision_id = EXCLUDED.decision_id,
                            approved_by = EXCLUDED.approved_by,
                            server_fingerprint = EXCLUDED.server_fingerprint,
                            safety_json = EXCLUDED.safety_json,
                            usage_source = EXCLUDED.usage_source,
                            estimated = EXCLUDED.estimated,
                            started_at = EXCLUDED.started_at,
                            completed_at = EXCLUDED.completed_at
                        """,
                        tool_audit_rows,
                    )
                if pending_approvals:
                    await conn.executemany(
                        """
                        INSERT INTO approval_requests (
                            approval_id,
                            session_id,
                            node_id,
                            tool_name,
                            risk_level,
                            reason,
                            request_payload_json,
                            status,
                            requested_at,
                            resolved_at,
                            resolved_by,
                            comment
                        )
                        VALUES ($1, $2::uuid, $3, $4, $5, $6, $7::jsonb, $8, $9, $10, $11, $12)
                        ON CONFLICT (approval_id) DO UPDATE SET
                            node_id = EXCLUDED.node_id,
                            tool_name = EXCLUDED.tool_name,
                            risk_level = EXCLUDED.risk_level,
                            reason = EXCLUDED.reason,
                            request_payload_json = EXCLUDED.request_payload_json,
                            status = EXCLUDED.status,
                            resolved_at = EXCLUDED.resolved_at,
                            resolved_by = EXCLUDED.resolved_by,
                            comment = EXCLUDED.comment
                        """,
                        [
                            (
                                approval["approval_id"],
                                session_id,
                                approval.get("node_id"),
                                approval.get("tool_name"),
                                approval.get("risk_level", "high"),
                                approval.get("reason"),
                                dumps_json(approval.get("request_payload") or {}),
                                approval.get("status", "pending"),
                                approval.get("requested_at") or now,
                                approval.get("resolved_at"),
                                approval.get("resolved_by"),
                                approval.get("comment"),
                            )
                            for approval in pending_approvals
                            if approval.get("approval_id")
                        ],
                    )
