"""
Research API endpoints: SSE streaming and research session management.
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from collections.abc import AsyncGenerator
from typing import Any

import asyncpg
from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException
from pydantic import BaseModel, Field
from sse_starlette.sse import EventSourceResponse

from app.config import get_settings
from app.core.time import utc_now_naive
from app.db.connection import get_db_pool
from app.db.json import dumps_json
from app.governance import HarnessSupervisor, RuntimePersistence
from app.governance.runtime import normalize_tool_audit_rows, public_status_from_runtime
from app.graph.compiler import compile_research_graph
from app.graph.state import (
    RuntimeStatus,
    TaskStatus,
    create_initial_state,
)
from app.guardrails import (
    build_guardrail_decision,
    build_review_status,
    compose_guardrail_prompt,
    get_research_budget,
    normalize_research_length,
)
from app.observability.sse_manager import get_sse_manager
from app.security.auth import Principal, require_principal
from app.skills import get_skill_registry

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1/research", tags=["research"], dependencies=[Depends(require_principal)])
principal_dependency = Depends(require_principal)
ACCUMULATING_STATE_KEYS = {
    "agent_trace",
    "tool_histories",
    "collected_evidence",
    "search_results",
    "browser_results",
    "rag_results",
    "aggregated_evidence",
    "node_outcomes",
    "pending_approvals",
}

KNOWN_STATE_KEYS = {
    "task_id",
    "user_query",
    "created_at",
    "status",
    "session",
    "dag",
    "current_executing_nodes",
    "completed_nodes",
    "node_outcomes",
    "tool_histories",
    "collected_evidence",
    "search_results",
    "browser_results",
    "rag_results",
    "aggregated_evidence",
    "verification",
    "revision_needed",
    "revision_count",
    "analysis",
    "final_report",
    "citations",
    "guardrail_decision",
    "evidence_status",
    "review_status",
    "user_confirmed",
    "allow_web_after_rag_hit",
    "rag_group",
    "retrieval_policy",
    "runtime_status",
    "budget_state",
    "pending_approvals",
    "agent_trace",
    "guardrail_trace",
    "errors",
}


def _principal_id(principal: Principal | Any) -> str | None:
    """Keep direct unit calls compatible; HTTP requests always receive a Principal."""
    return principal.id if isinstance(principal, Principal) else None


def _iter_state_updates(chunk: dict[str, Any]):
    """Yield state update mappings from raw LangGraph stream chunks."""
    if any(key in KNOWN_STATE_KEYS for key in chunk):
        yield chunk
        return

    for value in chunk.values():
        if isinstance(value, dict):
            yield value


def _normalize_citations(value: Any) -> list[dict[str, Any]]:
    """Return citations as a list of dicts regardless of stored shape."""
    if isinstance(value, list):
        normalized: list[dict[str, Any]] = []
        for item in value:
            if isinstance(item, dict):
                normalized.append(item)
            else:
                normalized.append({"value": item})
        return normalized
    if isinstance(value, dict):
        return [value]
    return []


def _normalize_tool_audit_rows(
    session_id: str,
    tool_histories: list[dict[str, Any]],
    node_outcomes: list[dict[str, Any]],
) -> list[tuple[Any, ...]]:
    """Backward-compatible wrapper around governance runtime normalization."""
    return normalize_tool_audit_rows(
        session_id=session_id,
        tool_histories=tool_histories,
        node_outcomes=node_outcomes,
    )


# ==============================================================
# Request/Response Models
# ==============================================================

class ResearchRequest(BaseModel):
    """Request body for starting a research task."""
    query: str = Field(..., min_length=5, max_length=2000, description="Research query")
    session_id: str | None = Field(default=None, description="Optional session ID for continuation")
    max_revision: int = Field(default=3, ge=1, le=5, description="Max revision loops")
    user_confirmed: bool = Field(default=False, description="User confirmed high-risk task")
    allow_web_after_rag_hit: bool = Field(
        default=False,
        description="If internal RAG finds evidence, also run internet search.",
    )
    rag_group: str | None = Field(
        default=None,
        max_length=100,
        description="Optional internal RAG source group filter.",
    )
    output_length: str = Field(
        default="medium",
        description="Output length: short, medium, long.",
    )
    enabled_skill_ids: list[str] = Field(
        default_factory=list,
        description="Explicitly enable selected skill IDs for this session.",
    )
    disabled_skill_ids: list[str] = Field(
        default_factory=list,
        description="Explicitly disable selected skill IDs for this session.",
    )
    skill_tenant_id: str | None = Field(
        default=None,
        max_length=100,
        description="Optional tenant scope for skill resolution.",
    )
    skill_project_id: str | None = Field(
        default=None,
        max_length=100,
        description="Optional project scope for skill resolution.",
    )


class ResearchStatus(BaseModel):
    """Response for research status query."""
    session_id: str
    status: str
    created_at: str
    updated_at: str | None = None
    runtime_status: str | None = None
    requires_confirmation: bool = False
    pending_approval_count: int = 0
    budget_state: dict[str, Any] | None = None
    last_error_category: str | None = None


class ResearchResponse(BaseModel):
    """Response for research creation."""
    session_id: str
    status: str
    message: str
    requires_confirmation: bool = False
    output_length: str = "medium"
    budget: dict[str, int | float] = Field(default_factory=dict)
    skill_context: dict[str, Any] = Field(default_factory=dict)


class ToolCallAuditRecord(BaseModel):
    """Single persisted tool call audit row."""
    call_id: str
    session_id: str
    node_id: str | None = None
    agent_type: str
    tool_name: str
    args_json: dict[str, Any] = Field(default_factory=dict)
    args_hash: str | None = None
    status: str
    error_category: str | None = None
    error_message: str | None = None
    retry_count: int = 0
    result_summary: str | None = None
    result_hash: str | None = None
    tokens_used: int = 0
    cost_usd: float = 0.0
    decision_id: str | None = None
    approved_by: str | None = None
    server_fingerprint: str | None = None
    safety_json: dict[str, Any] = Field(default_factory=dict)
    usage_source: str | None = None
    estimated: bool = False
    started_at: str | None = None
    completed_at: str | None = None
    created_at: str | None = None


class ApprovalRequestRecord(BaseModel):
    approval_id: str
    session_id: str
    node_id: str | None = None
    tool_name: str
    risk_level: str
    reason: str | None = None
    request_payload_json: dict[str, Any] = Field(default_factory=dict)
    status: str
    requested_at: str | None = None
    resolved_at: str | None = None
    resolved_by: str | None = None
    comment: str | None = None


class ApprovalActionRequest(BaseModel):
    action: str = Field(..., pattern="^(approve|reject)$")
    approved_by: str = Field(..., min_length=1, max_length=100)
    comment: str | None = Field(default=None, max_length=1000)


# ==============================================================
# SSE Event Streaming
# ==============================================================

async def research_event_generator(
    session_id: str,
) -> AsyncGenerator[dict[str, Any], None]:
    """
    SSE event generator for research progress.
    Yields events as SSE-compatible dictionaries.
    """
    sse = get_sse_manager()

    # Send initial connection event
    yield {
        "event": "connected",
        "data": {"session_id": session_id, "timestamp": utc_now_naive().isoformat()},
    }

    try:
        # Stream events from the queue
        async for event in sse.stream(session_id):
            yield event
    except asyncio.CancelledError:
        logger.info(f"SSE stream cancelled for session {session_id}")
        yield {
            "event": "disconnected",
            "data": {"session_id": session_id},
        }


async def _owned_session_exists(session_id: str, owner_id: str) -> bool:
    pool = await get_db_pool()
    async with pool.acquire() as conn:
        return bool(await conn.fetchval(
            "SELECT EXISTS(SELECT 1 FROM research_sessions WHERE id = $1::uuid AND owner_id = $2::uuid)",
            session_id,
            owner_id,
        ))


@router.get("/stream/{session_id}")
async def stream_research(session_id: str, principal: Principal = principal_dependency):
    """
    SSE endpoint for real-time research progress streaming.

    Clients should connect with:
        const eventSource = new EventSource(`/api/v1/research/stream/${sessionId}`);

    Event types:
    - connected: Initial connection confirmation
    - agent_start/end: Agent execution lifecycle
    - thought: LLM reasoning process
    - tool_call/result/error: Tool invocations
    - state_update: Workflow state changes
    - reflection: Reflection result
    - report_chunk: Report content chunks
    - done: Task completion
    - workflow_error: Workflow/business error occurred
    """
    if not await _owned_session_exists(session_id, principal.id):
        raise HTTPException(status_code=404, detail="Session not found")
    return EventSourceResponse(
        research_event_generator(session_id),
        media_type="text/event-stream",
    )


# ==============================================================
# Research Task Management
# ==============================================================

@router.post("", response_model=ResearchResponse)
async def create_research(
    request: ResearchRequest,
    background_tasks: BackgroundTasks,
    principal: Principal = principal_dependency,
):
    """
    Start a new research task.

    Creates a research session, saves it to the database,
    and dispatches the LangGraph workflow to run in background.
    """
    # Generate or use provided session ID
    session_id = request.session_id or str(uuid.uuid4())
    decision = build_guardrail_decision(request.query, user_confirmed=request.user_confirmed)
    output_length = normalize_research_length(request.output_length)
    budget = get_research_budget(output_length)
    skill_context = (await get_skill_registry(_principal_id(principal)).resolve_for_session(
        query=request.query,
        manually_enabled_skill_ids=request.enabled_skill_ids,
        manually_disabled_skill_ids=request.disabled_skill_ids,
        tenant_id=request.skill_tenant_id,
        project_id=request.skill_project_id,
    )).as_dict()

    logger.info(f"Creating research session: {session_id}, query: {request.query[:50]}")

    # Save to database
    pool = await get_db_pool()
    async with pool.acquire() as conn:
        review_status = build_review_status(
            blocked=decision.must_confirm and not request.user_confirmed,
            requires_confirmation=decision.must_confirm and not request.user_confirmed,
            approved=request.user_confirmed or not decision.must_confirm,
            reason="pending_confirmation" if decision.must_confirm and not request.user_confirmed else None,
            risk_level=decision.risk_level,
            intent=decision.intent,
            prompt_profile=decision.prompt_profile,
        )
        await conn.execute(
            """
            INSERT INTO research_sessions (id, user_query, status, guardrail_decision, guardrail_trace,
                                           evidence_status, review_status, prompt_profile, prompt_template,
                                            enabled_tools, skill_context, owner_id, created_at, updated_at)
            VALUES ($1::uuid, $2, $3, $4::jsonb, $5::jsonb, $6::jsonb, $7::jsonb, $8, $9, $10::jsonb, $11::jsonb, $12::uuid, $13, $13)
            ON CONFLICT (id) DO UPDATE SET
                user_query = $2,
                status = $3,
                guardrail_decision = $4::jsonb,
                guardrail_trace = $5::jsonb,
                evidence_status = $6::jsonb,
                review_status = $7::jsonb,
                prompt_profile = $8,
                prompt_template = $9,
                enabled_tools = $10::jsonb,
                skill_context = $11::jsonb,
                updated_at = $13
            """,
            session_id,
            request.query,
            "pending" if decision.must_confirm and not request.user_confirmed else "running",
            dumps_json(decision.model_dump()),
            dumps_json([]),
            dumps_json(None),
            dumps_json(review_status),
            decision.prompt_profile.value,
            compose_guardrail_prompt(request.query, decision),
            dumps_json(skill_context.get("effective_tool_allowlist") or decision.enabled_tools),
            dumps_json(skill_context),
            _principal_id(principal),
            utc_now_naive(),
        )

    if decision.must_confirm and not request.user_confirmed:
        return ResearchResponse(
            session_id=session_id,
            status="pending_confirmation",
            message="High-risk request requires user confirmation before execution.",
            requires_confirmation=True,
            output_length=output_length.value,
            budget=budget,
            skill_context=skill_context,
        )

    # Start background execution
    background_tasks.add_task(
        run_research_workflow,
        session_id=session_id,
        query=request.query,
        max_revision=request.max_revision,
        user_confirmed=request.user_confirmed,
        allow_web_after_rag_hit=request.allow_web_after_rag_hit,
        rag_group=request.rag_group,
        output_length=output_length.value,
        skill_context=skill_context,
        owner_id=_principal_id(principal),
    )

    return ResearchResponse(
        session_id=session_id,
        status="running",
        message=f"Research task started. Connect to /api/v1/research/stream/{session_id} for updates.",
        requires_confirmation=False,
        output_length=output_length.value,
        budget=budget,
        skill_context=skill_context,
    )


@router.get("/status/{session_id}", response_model=ResearchStatus)
async def get_research_status(session_id: str, principal: Principal = principal_dependency):
    """Get the current status of a research session."""
    pool = await get_db_pool()
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            """
            SELECT rs.id, rs.user_query, rs.status, rs.review_status, rs.created_at, rs.updated_at, rs.completed_at,
                   sbs.max_total_tokens, sbs.max_cost_usd, sbs.max_tool_calls, sbs.max_wall_clock_seconds,
                   sbs.used_total_tokens, sbs.used_cost_usd, sbs.used_tool_calls,
                   sbs.elapsed_wall_clock_seconds, sbs.hard_stop_reason
            FROM research_sessions
            rs
            LEFT JOIN session_budget_state sbs ON sbs.session_id = rs.id
            WHERE rs.id = $1::uuid AND rs.owner_id = $2::uuid
            """,
            session_id, _principal_id(principal),
        )

    if not row:
        raise HTTPException(status_code=404, detail="Session not found")

    review_status = row.get("review_status") or {}
    budget_state = review_status.get("budget_state")
    if row.get("max_tool_calls") is not None:
        budget_state = {
            "max_total_tokens": int(row.get("max_total_tokens") or 0),
            "max_cost_usd": float(row.get("max_cost_usd") or 0.0),
            "max_tool_calls": int(row.get("max_tool_calls") or 0),
            "max_wall_clock_seconds": int(row.get("max_wall_clock_seconds") or 0),
            "used_total_tokens": int(row.get("used_total_tokens") or 0),
            "used_cost_usd": float(row.get("used_cost_usd") or 0.0),
            "used_tool_calls": int(row.get("used_tool_calls") or 0),
            "elapsed_wall_clock_seconds": int(row.get("elapsed_wall_clock_seconds") or 0),
            "hard_stop_reason": row.get("hard_stop_reason"),
        }
    pending_approval_count = int(review_status.get("pending_approval_count", 0) or 0)
    return ResearchStatus(
        session_id=str(row["id"]),
        status=row["status"],
        created_at=row["created_at"].isoformat(),
        updated_at=row["updated_at"].isoformat() if row["updated_at"] else None,
        runtime_status=review_status.get("runtime_status"),
        requires_confirmation=bool(review_status.get("requires_confirmation", False)),
        pending_approval_count=pending_approval_count,
        budget_state=budget_state,
        last_error_category=review_status.get("last_error_category"),
    )


@router.get("/{session_id}")
async def get_research_result(session_id: str, principal: Principal = principal_dependency):
    """Get the final research result."""
    pool = await get_db_pool()
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            """
            SELECT id, user_query, status, final_report, citations, agent_trace, review_status, created_at, completed_at
                 , skill_context
            FROM research_sessions
            WHERE id = $1::uuid AND owner_id = $2::uuid
            """,
            session_id, _principal_id(principal),
        )
        citation_rows = await conn.fetch(
            """
            SELECT citation_id, source_url, source_title, source_type,
                   extracted_evidence, relevance_score, access_timestamp
            FROM citations
            WHERE session_id = $1::uuid
              AND EXISTS (SELECT 1 FROM research_sessions WHERE id = $1::uuid AND owner_id = $2::uuid)
            ORDER BY CAST(SPLIT_PART(citation_id, ':', 2) AS INTEGER)
            """,
            session_id, _principal_id(principal),
        )

    if not row:
        raise HTTPException(status_code=404, detail="Session not found")

    citations = [dict(citation) for citation in citation_rows] if citation_rows else _normalize_citations(row["citations"])

    review_status = row.get("review_status") or {}
    return {
        "session_id": str(row["id"]),
        "query": row["user_query"],
        "status": row["status"],
        "runtime_status": review_status.get("runtime_status"),
        "report": row["final_report"],
        "citations": citations,
        "agent_trace": row["agent_trace"],
        "tool_audit_summary": review_status.get("tool_audit_summary", {}),
        "skill_context": row.get("skill_context") or {},
        "created_at": row["created_at"].isoformat(),
        "completed_at": row["completed_at"].isoformat() if row["completed_at"] else None,
    }


@router.get("/{session_id}/tool-calls", response_model=list[ToolCallAuditRecord])
async def get_research_tool_calls(session_id: str, principal: Principal = principal_dependency):
    """Get persisted per-call tool audit rows for a research session."""
    pool = await get_db_pool()
    async with pool.acquire() as conn:
        session_exists = await conn.fetchrow(
            "SELECT id FROM research_sessions WHERE id = $1::uuid AND owner_id = $2::uuid",
            session_id, _principal_id(principal),
        )
        if not session_exists:
            raise HTTPException(status_code=404, detail="Session not found")

        rows = await conn.fetch(
            """
            SELECT call_id, session_id, node_id, agent_type, tool_name,
                   args_json, args_hash, status, error_category, error_message,
                   retry_count, result_summary, result_hash, tokens_used, cost_usd,
                   decision_id, approved_by, server_fingerprint, safety_json, usage_source, estimated,
                   started_at, completed_at, created_at
            FROM tool_call_audit
            WHERE session_id = $1::uuid
              AND EXISTS (
                  SELECT 1 FROM research_sessions
                  WHERE id = $1::uuid AND owner_id = $2::uuid
              )
            ORDER BY created_at ASC, call_id ASC
            """,
            session_id,
            _principal_id(principal),
        )

    result: list[ToolCallAuditRecord] = []
    for row in rows:
        result.append(ToolCallAuditRecord(
            call_id=row["call_id"],
            session_id=str(row["session_id"]),
            node_id=row["node_id"],
            agent_type=row["agent_type"],
            tool_name=row["tool_name"],
            args_json=row["args_json"] or {},
            args_hash=row["args_hash"],
            status=row["status"],
            error_category=row["error_category"],
            error_message=row["error_message"],
            retry_count=int(row["retry_count"] or 0),
            result_summary=row["result_summary"],
            result_hash=row["result_hash"],
            tokens_used=int(row["tokens_used"] or 0),
            cost_usd=float(row["cost_usd"] or 0.0),
            decision_id=row["decision_id"],
            approved_by=row["approved_by"],
            server_fingerprint=row["server_fingerprint"],
            safety_json=row["safety_json"] or {},
            usage_source=row["usage_source"],
            estimated=bool(row["estimated"]),
            started_at=row["started_at"].isoformat() if row["started_at"] else None,
            completed_at=row["completed_at"].isoformat() if row["completed_at"] else None,
            created_at=row["created_at"].isoformat() if row["created_at"] else None,
        ))
    return result


@router.get("/{session_id}/approvals", response_model=list[ApprovalRequestRecord])
async def get_research_approvals(session_id: str, principal: Principal = principal_dependency):
    """Get approval requests for a research session."""
    pool = await get_db_pool()
    async with pool.acquire() as conn:
        session_exists = await conn.fetchrow(
            "SELECT id FROM research_sessions WHERE id = $1::uuid AND owner_id = $2::uuid",
            session_id, _principal_id(principal),
        )
        if not session_exists:
            raise HTTPException(status_code=404, detail="Session not found")
        rows = await conn.fetch(
            """
            SELECT approval_id, session_id, node_id, tool_name, risk_level, reason,
                   request_payload_json, status, requested_at, resolved_at, resolved_by, comment
            FROM approval_requests
            WHERE session_id = $1::uuid
              AND EXISTS (
                  SELECT 1 FROM research_sessions
                  WHERE id = $1::uuid AND owner_id = $2::uuid
              )
            ORDER BY requested_at ASC, approval_id ASC
            """,
            session_id,
            _principal_id(principal),
        )
    return [
        ApprovalRequestRecord(
            approval_id=row["approval_id"],
            session_id=str(row["session_id"]),
            node_id=row["node_id"],
            tool_name=row["tool_name"],
            risk_level=row["risk_level"],
            reason=row["reason"],
            request_payload_json=row["request_payload_json"] or {},
            status=row["status"],
            requested_at=row["requested_at"].isoformat() if row["requested_at"] else None,
            resolved_at=row["resolved_at"].isoformat() if row["resolved_at"] else None,
            resolved_by=row["resolved_by"],
            comment=row["comment"],
        )
        for row in rows
    ]


@router.post("/{session_id}/approvals/{approval_id}")
async def resolve_research_approval(
    session_id: str,
    approval_id: str,
    request: ApprovalActionRequest,
    principal: Principal = principal_dependency,
):
    """Resolve an approval request and persist decision for harness-driven resume."""
    pool = await get_db_pool()
    approval_payload: dict[str, Any] | None = None
    async with pool.acquire() as conn:
        approval = await conn.fetchrow(
            """
            SELECT approval_id, session_id, status, request_payload_json
            FROM approval_requests
            WHERE approval_id = $1 AND session_id = $2::uuid
              AND EXISTS (SELECT 1 FROM research_sessions WHERE id = $2::uuid AND owner_id = $3::uuid)
            """,
            approval_id,
            session_id, _principal_id(principal),
        )
        if not approval:
            raise HTTPException(status_code=404, detail="Approval request not found")
        if approval["status"] not in {"pending", "awaiting_approval"}:
            raise HTTPException(status_code=409, detail="Approval request already resolved")
        resolved_status = "approved" if request.action == "approve" else "rejected"
        await conn.execute(
            """
            UPDATE approval_requests
            SET status = $1,
                resolved_at = $2,
                resolved_by = $3,
                comment = $4
            WHERE approval_id = $5 AND session_id = $6::uuid
            """,
            resolved_status,
            utc_now_naive(),
            request.approved_by,
            request.comment,
            approval_id,
            session_id,
        )
        approval_payload = approval["request_payload_json"] or {}
    if resolved_status == "approved":
        harness = HarnessSupervisor(get_settings().harness.state_root)
        task = harness.get_task(session_id)
        checkpoint = ((task or {}).get("checkpoint") or {})
        state_snapshot = checkpoint.get("state_snapshot")
        if isinstance(state_snapshot, dict):
            pending = []
            for item in state_snapshot.get("pending_approvals", []):
                if not isinstance(item, dict):
                    continue
                if item.get("approval_id") == approval_id:
                    item = {
                        **item,
                        "status": "approved",
                        "resolved_at": utc_now_naive().isoformat(),
                        "resolved_by": request.approved_by,
                        "comment": request.comment,
                    }
                pending.append(item)
            state_snapshot["pending_approvals"] = [
                item for item in pending
                if item.get("status") not in {"approved", "rejected"}
            ]
            state_snapshot["runtime_status"] = RuntimeStatus.RUNNING.value
            graph = compile_research_graph()
            config = {"configurable": {"thread_id": session_id}}
            graph.update_state(config, state_snapshot, as_node="approval_resume")
            asyncio.create_task(
                run_research_workflow(
                    session_id=session_id,
                    query=state_snapshot.get("user_query", ""),
                    max_revision=int(state_snapshot.get("session", {}).get("max_revisions", 3) or 3),
                    user_confirmed=bool(state_snapshot.get("user_confirmed", False)),
                    allow_web_after_rag_hit=bool(state_snapshot.get("allow_web_after_rag_hit", False)),
                    rag_group=state_snapshot.get("rag_group"),
                    output_length=state_snapshot.get("output_length", "medium"),
                    owner_id=state_snapshot.get("session", {}).get("owner_id"),
                )
            )
    return {
        "approval_id": approval_id,
        "session_id": session_id,
        "status": resolved_status,
        "approved_by": request.approved_by,
        "request_payload": approval_payload,
    }


# ==============================================================
# Background Workflow Execution
# ==============================================================

async def run_research_workflow(
    session_id: str,
    query: str,
    max_revision: int = 3,
    user_confirmed: bool = False,
    allow_web_after_rag_hit: bool = False,
    rag_group: str | None = None,
    output_length: str = "medium",
    skill_context: dict[str, Any] | None = None,
    owner_id: str | None = None,
):
    """
    Execute the LangGraph research workflow in background.

    This function:
    1. Compiles the StateGraph
    2. Creates the initial state
    3. Streams events via SSE
    4. Saves final results to the database
    """
    sse = get_sse_manager()
    runtime_persistence = RuntimePersistence()
    harness = HarnessSupervisor(get_settings().harness.state_root)
    settings = get_settings()
    budget = get_research_budget(output_length)

    try:
        # Emit start event
        await sse.publish(session_id, "workflow_start", {
            "session_id": session_id,
            "query": query,
            "timestamp": utc_now_naive().isoformat(),
        })

        # Create initial state
        state = create_initial_state(query, session_id)
        state.setdefault("session", {})["owner_id"] = owner_id
        state["session"]["max_revisions"] = max_revision
        state["session"]["output_length"] = output_length
        decision = build_guardrail_decision(query)
        state["guardrail_decision"] = decision.model_dump()
        state["user_confirmed"] = user_confirmed
        state["allow_web_after_rag_hit"] = allow_web_after_rag_hit
        state["rag_group"] = rag_group
        state["output_length"] = output_length
        state["retrieval_policy"] = {
            "mode": "internal_first",
            "allow_web_after_rag_hit": allow_web_after_rag_hit,
            "rag_group": rag_group,
            "rag_hit_count": 0,
            "web_search_required": None,
            "web_search_reason": None,
        }
        state["session"]["prompt_profile"] = decision.prompt_profile.value
        state["session"]["enabled_tools"] = (skill_context or {}).get("effective_tool_allowlist") or decision.enabled_tools
        state["session"]["prompt_template"] = compose_guardrail_prompt(query, decision)
        state["runtime_status"] = RuntimeStatus.RUNNING.value
        state["skill_context"] = skill_context or {}
        state["budget_state"] = {
            "budget_profile": output_length,
            "estimated": True,
            **budget,
            "used_total_tokens": 0,
            "used_cost_usd": 0.0,
            "used_tool_calls": 0,
            "elapsed_wall_clock_seconds": 0,
            "hard_stop_reason": None,
            "warning": False,
        }
        state["session"]["harness_state_version"] = 1
        state["session"]["checkpoint_seq"] = 0
        state["session"]["skill_context"] = skill_context or {}
        await runtime_persistence.persist_runtime_snapshot(
            session_id=session_id,
            state=state,
            current_batch=[],
            checkpoint_ref=None,
        )
        harness.upsert_task(
            session_id=session_id,
            public_status="running",
            runtime_status=RuntimeStatus.RUNNING.value,
            budget=state["budget_state"],
            current_batch=[],
            checkpoint_seq=0,
            pending_approval_id=None,
            used_total_tokens=0,
            used_cost_usd=0.0,
            worker_id=settings.harness.worker_id,
        )

        # Compile graph
        graph = compile_research_graph()

        # Run with streaming
        config = {
            "configurable": {
                "thread_id": session_id,
            }
        }

        # Run the graph and accumulate state from all chunks
        # FIX: Use a dict to accumulate state, not just last chunk
        accumulated_state: dict[str, Any] = {}
        last_chunk: dict[str, Any] = {}
        async for chunk in graph.astream(state, config):
            # Emit state updates
            if isinstance(chunk, dict):
                last_chunk = chunk
                # Merge chunk into accumulated state
                for update in _iter_state_updates(chunk):
                    for key, value in update.items():
                        if key in ACCUMULATING_STATE_KEYS and isinstance(value, list):
                            accumulated_state.setdefault(key, [])
                            accumulated_state[key].extend(value)
                        else:
                            accumulated_state[key] = value

                        if key == "agent_trace":
                            # Forward agent trace events to SSE
                            for event in value:
                                event_type = event.get("event_type", "trace") if isinstance(event, dict) else "trace"
                                await sse.publish(session_id, event_type, event)
                        else:
                            # Forward other state updates
                            await sse.publish(session_id, "state_update", {
                                "key": key,
                                "value": str(value)[:500] if value else "",
                            })
                checkpoint_state = accumulated_state if accumulated_state else last_chunk
                checkpoint_runtime = checkpoint_state.get("runtime_status", RuntimeStatus.RUNNING.value)
                checkpoint_batch = checkpoint_state.get("current_executing_nodes", [])
                checkpoint_ref = None
                try:
                    snapshot = graph.get_state(config)
                    checkpoint_ref = getattr(snapshot, "config", None)
                except (OSError, RuntimeError, TypeError, ValueError):
                    checkpoint_ref = None
                checkpoint_seq = int(checkpoint_state.get("session", {}).get("checkpoint_seq", 0) or 0) + 1
                checkpoint_state.setdefault("session", {})
                checkpoint_state["session"]["checkpoint_seq"] = checkpoint_seq
                await runtime_persistence.persist_runtime_snapshot(
                    session_id=session_id,
                    state=checkpoint_state,
                    current_batch=checkpoint_batch,
                    checkpoint_ref=str(checkpoint_ref) if checkpoint_ref is not None else None,
                )
                pending_approvals = checkpoint_state.get("pending_approvals", [])
                pending_approval_id = None
                if pending_approvals:
                    latest = pending_approvals[-1]
                    if isinstance(latest, dict):
                        pending_approval_id = latest.get("approval_id")
                harness.upsert_task(
                    session_id=session_id,
                    public_status=public_status_from_runtime(checkpoint_runtime),
                    runtime_status=checkpoint_runtime,
                    budget=dict(checkpoint_state.get("budget_state") or {}),
                    current_batch=checkpoint_batch,
                    checkpoint_seq=checkpoint_seq,
                    pending_approval_id=pending_approval_id,
                    used_total_tokens=int((checkpoint_state.get("budget_state") or {}).get("used_total_tokens", 0) or 0),
                    used_cost_usd=float((checkpoint_state.get("budget_state") or {}).get("used_cost_usd", 0.0) or 0.0),
                    last_error={"category": checkpoint_state.get("review_status", {}).get("last_error_category")},
                    worker_id=settings.harness.worker_id,
                    state_snapshot=checkpoint_state,
                )

        # Determine final state
        # Use accumulated state, with fallback to last chunk
        final_state: dict[str, Any] = accumulated_state if accumulated_state else last_chunk

        # Save results to database
        if final_state:
            pool = await get_db_pool()
            async with pool.acquire() as conn:
                completed_at = utc_now_naive()
                citations = final_state.get("citations", [])
                tool_histories = final_state.get("tool_histories", [])
                tool_call_total = sum(
                    len(history.get("tool_calls", []))
                    for history in tool_histories
                    if isinstance(history, dict)
                )
                tool_call_errors = sum(
                    sum(1 for call in history.get("tool_calls", []) if call.get("status") == "error")
                    for history in tool_histories
                    if isinstance(history, dict)
                )
                pending_approval_count = len(final_state.get("pending_approvals", []))
                runtime_status = final_state.get("runtime_status", final_state.get("status", TaskStatus.COMPLETED.value))
                review_status = dict(final_state.get("review_status") or {})
                tool_audit_rows = normalize_tool_audit_rows(
                    session_id=session_id,
                    tool_histories=tool_histories,
                    node_outcomes=final_state.get("node_outcomes", []),
                )
                session_budget_state = dict(final_state.get("budget_state") or {})
                review_status.update({
                    "runtime_status": runtime_status,
                    "pending_approval_count": pending_approval_count,
                    "budget_state": session_budget_state,
                    "tool_audit_summary": {
                        "total_calls": tool_call_total,
                        "error_calls": tool_call_errors,
                        "persisted_calls": len(tool_audit_rows),
                    },
                })
                last_error_category = None
                for outcome in reversed(final_state.get("node_outcomes", [])):
                    if isinstance(outcome, dict) and outcome.get("error_category"):
                        last_error_category = outcome.get("error_category")
                        break
                if last_error_category:
                    review_status["last_error_category"] = last_error_category
                async with conn.transaction():
                    await conn.execute(
                        """
                        UPDATE research_sessions
                        SET status = $1,
                            guardrail_decision = $2::jsonb,
                            guardrail_trace = $3::jsonb,
                            evidence_status = $4::jsonb,
                            review_status = $5::jsonb,
                            prompt_profile = $6,
                            prompt_template = $7,
                            enabled_tools = $8::jsonb,
                            final_report = $9,
                            citations = $10::jsonb,
                            agent_trace = $11::jsonb,
                            total_tokens = $12,
                            total_cost_usd = $13,
                            updated_at = $14,
                            completed_at = $14
                        WHERE id = $15::uuid
                        """,
                        final_state.get("status", TaskStatus.COMPLETED.value),
                        dumps_json(final_state.get("guardrail_decision")),
                        dumps_json(final_state.get("guardrail_trace", [])),
                        dumps_json(final_state.get("evidence_status")),
                        dumps_json(review_status),
                        final_state.get("session", {}).get("prompt_profile"),
                        final_state.get("session", {}).get("prompt_template"),
                        dumps_json(final_state.get("session", {}).get("enabled_tools", [])),
                        final_state.get("final_report", ""),
                        dumps_json(citations),
                        dumps_json(final_state.get("agent_trace", [])),
                        int(final_state.get("session", {}).get("total_tokens", session_budget_state.get("used_total_tokens", 0)) or 0),
                        float(final_state.get("session", {}).get("total_cost_usd", session_budget_state.get("used_cost_usd", 0.0)) or 0.0),
                        completed_at,
                        session_id,
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
                            updated_at
                        )
                        VALUES ($1::uuid, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12)
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
                            updated_at = EXCLUDED.updated_at
                        """,
                        session_id,
                        int(session_budget_state.get("max_total_tokens", 0) or 0),
                        float(session_budget_state.get("max_cost_usd", 0.0) or 0.0),
                        int(session_budget_state.get("max_tool_calls", 0) or 0),
                        int(session_budget_state.get("max_wall_clock_seconds", 0) or 0),
                        int(session_budget_state.get("max_retries_per_tool", 0) or 0),
                        int(session_budget_state.get("used_total_tokens", 0) or 0),
                        float(session_budget_state.get("used_cost_usd", 0.0) or 0.0),
                        int(session_budget_state.get("used_tool_calls", 0) or 0),
                        int(session_budget_state.get("elapsed_wall_clock_seconds", 0) or 0),
                        session_budget_state.get("hard_stop_reason"),
                        completed_at,
                    )
                    await conn.execute(
                        "DELETE FROM citations WHERE session_id = $1::uuid",
                        session_id,
                    )
                    if citations:
                        await conn.executemany(
                            """
                            INSERT INTO citations (
                                session_id,
                                citation_id,
                                source_url,
                                source_title,
                                source_type,
                                extracted_evidence,
                                relevance_score,
                                access_timestamp
                            )
                            VALUES ($1::uuid, $2, $3, $4, $5, $6, $7, $8)
                            """,
                            [
                                (
                                    session_id,
                                    citation.get("citation_id"),
                                    citation.get("source_url"),
                                    citation.get("source_title"),
                                    citation.get("source_type", "web"),
                                    citation.get("extracted_evidence"),
                                    citation.get("relevance_score", 0.0),
                                    completed_at,
                                )
                                for citation in citations
                            ],
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
                await runtime_persistence.persist_runtime_snapshot(
                    session_id=session_id,
                    state=final_state,
                    current_batch=final_state.get("current_executing_nodes", []),
                    checkpoint_ref="final",
                )
                harness.upsert_task(
                    session_id=session_id,
                    public_status=final_state.get("status", public_status_from_runtime(runtime_status)),
                    runtime_status=runtime_status,
                    budget=session_budget_state,
                    current_batch=final_state.get("current_executing_nodes", []),
                    checkpoint_seq=int(final_state.get("session", {}).get("checkpoint_seq", 0) or 0),
                    pending_approval_id=None,
                    used_total_tokens=int(session_budget_state.get("used_total_tokens", 0) or 0),
                    used_cost_usd=float(session_budget_state.get("used_cost_usd", 0.0) or 0.0),
                    worker_id=settings.harness.worker_id,
                    state_snapshot=final_state,
                )
                harness.clear_active_if_idle()

        # Emit completion
        await sse.publish(session_id, "done", {
            "session_id": session_id,
            "status": final_state.get("status", "completed") if final_state else "completed",
            "timestamp": utc_now_naive().isoformat(),
        })

        logger.info(f"Research workflow completed: {session_id}")

    except Exception as exc:
        logger.exception("Research workflow error for %s", session_id)

        # Emit error
        await sse.publish(session_id, "workflow_error", {
            "session_id": session_id,
            "error": str(exc),
            "timestamp": utc_now_naive().isoformat(),
        })

        # Update database status - FIX: properly log errors instead of bare except:pass
        try:
            pool = await get_db_pool()
            async with pool.acquire() as conn:
                await conn.execute(
                    """
                    UPDATE research_sessions
                    SET status = 'failed', updated_at = $1
                    WHERE id = $2::uuid
                    """,
                    utc_now_naive(),
                    session_id,
                )
            harness.upsert_task(
                session_id=session_id,
                public_status="failed",
                runtime_status=RuntimeStatus.TERMINAL_FAILED.value,
                budget={},
                current_batch=[],
                checkpoint_seq=0,
                pending_approval_id=None,
                last_error={"category": "workflow_error", "message": str(exc)},
                worker_id=settings.harness.worker_id,
                state_snapshot=None,
            )
        except (OSError, RuntimeError, asyncpg.PostgresError) as db_error:
            # FIX: Log the error instead of silently swallowing
            logger.error(
                f"Failed to update session {session_id} status to 'failed': {db_error}"
            )
