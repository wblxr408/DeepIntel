"""
Agent Tracer: instruments the research workflow with observability events.

Provides structured logging and SSE event emission for agent traces.
"""

from __future__ import annotations

import logging
import time
from enum import Enum
from typing import Any
from uuid import uuid4

from app.core.time import utc_now_naive
from app.observability.sse_manager import get_sse_manager

logger = logging.getLogger(__name__)


class EventType(str, Enum):
    """Standard event types for agent tracing."""
    AGENT_START = "agent_start"
    AGENT_COMPLETE = "agent_complete"
    AGENT_END = "agent_end"  # backward compat
    THOUGHT = "thought"
    TOOL_CALL = "tool_call"
    TOOL_START = "tool_start"
    TOOL_COMPLETE = "tool_complete"
    TOOL_ERROR = "tool_error"
    TOOL_RESULT = "tool_result"
    STATE_UPDATE = "state_update"
    REFLECTION = "reflection"
    REPORT_CHUNK = "report_chunk"
    REPORT_CITATION = "report_citation"
    WORKFLOW_ERROR = "workflow_error"
    DONE = "done"


def emit_event(state: dict, event_type: EventType, agent: str, content: str) -> None:
    """
    Emit an event to the agent_trace in state.

    This is a convenience function that creates an AgentEvent and appends it
    to state["agent_trace"]. Used by compiler.py nodes.
    """
    from app.graph.state import AgentEvent

    event = AgentEvent(
        agent=agent,
        event_type=event_type.value,
        content=content,
    )
    state.setdefault("agent_trace", []).append(event.model_dump())


class AgentTracer:
    """
    Traces agent execution and emits events to SSE and structured logs.

    Usage:
        tracer = AgentTracer(session_id="abc123", agent="planner")
        tracer.start()
        tracer.thought("Planning research steps...")
        tracer.tool_call("duckduckgo_search", {"query": "AI market 2025"})
        tracer.tool_result("success", {"results": 15})
        tracer.end()

    Each event is:
    - Logged with structlog
    - Published to SSE (if enabled)
    - Stored in the agent_trace list in ResearchState
    """

    def __init__(
        self,
        session_id: str,
        agent: str,
        enabled: bool = True,
    ):
        self.session_id = session_id
        self.agent = agent
        self.enabled = enabled
        self.start_time: float | None = None
        self._sse = get_sse_manager() if enabled else None

    def start(self) -> None:
        """Mark the start of an agent's execution."""
        self.start_time = time.time()
        self._emit(EventType.AGENT_START, {"agent": self.agent})

    def thought(self, content: str) -> None:
        """Record an LLM thought/reasoning step."""
        self._emit(EventType.THOUGHT, {
            "agent": self.agent,
            "content": content,
            "timestamp": utc_now_naive().isoformat(),
        })

    def tool_call(self, tool_name: str, args: dict[str, Any]) -> str:
        """Record a tool call and return a call ID."""
        call_id = str(uuid4())[:12]
        self._emit(EventType.TOOL_CALL, {
            "call_id": call_id,
            "tool": tool_name,
            "args": args,
            "agent": self.agent,
        })
        return call_id

    def tool_result(
        self,
        call_id: str,
        status: str,
        result: Any,
        duration_ms: int | None = None,
    ) -> None:
        """Record the result of a tool call."""
        elapsed = int((time.time() - self.start_time) * 1000) if self.start_time else 0
        self._emit(EventType.TOOL_RESULT, {
            "call_id": call_id,
            "status": status,
            "agent": self.agent,
            "duration_ms": duration_ms or elapsed,
            "result_summary": str(result)[:200] if result else "",
        })

    def tool_error(
        self,
        call_id: str,
        error: str,
        retry_count: int = 0,
    ) -> None:
        """Record a tool call error."""
        self._emit(EventType.TOOL_ERROR, {
            "call_id": call_id,
            "error": error,
            "retry_count": retry_count,
            "agent": self.agent,
        })

    def state_update(self, step_index: int, status: str, details: str = "") -> None:
        """Record a workflow state update."""
        self._emit(EventType.STATE_UPDATE, {
            "step_index": step_index,
            "status": status,
            "details": details,
        })

    def reflection_result(
        self,
        confidence: float,
        hallucination_count: int,
        needs_revision: bool,
    ) -> None:
        """Record a reflection result."""
        self._emit(EventType.REFLECTION, {
            "confidence": confidence,
            "hallucination_count": hallucination_count,
            "needs_revision": needs_revision,
        })

    def report_chunk(self, chunk: str, is_partial: bool = True) -> None:
        """Record a report content chunk."""
        self._emit(EventType.REPORT_CHUNK, {
            "chunk": chunk,
            "is_partial": is_partial,
        })

    def report_citation(self, citation_id: str, source: str) -> None:
        """Record a citation added to the report."""
        self._emit(EventType.REPORT_CITATION, {
            "citation_id": citation_id,
            "source": source,
        })

    def error(self, error_type: str, message: str, recoverable: bool = True) -> None:
        """Record an error."""
        self._emit(EventType.WORKFLOW_ERROR, {
            "error_type": error_type,
            "message": message,
            "recoverable": recoverable,
            "agent": self.agent,
        })

    def end(self, summary: str = "") -> None:
        """Mark the end of an agent's execution."""
        duration_ms = int((time.time() - self.start_time) * 1000) if self.start_time else 0
        self._emit(EventType.AGENT_END, {
            "agent": self.agent,
            "duration_ms": duration_ms,
            "summary": summary,
        })

    def _emit(self, event_type: EventType, data: dict[str, Any]) -> None:
        """Emit an event to logs and SSE."""
        if not self.enabled:
            return

        # Structured log
        logger.info(
            f"[{self.agent}] {event_type.value}",
            extra={
                "session_id": self.session_id,
                "agent": self.agent,
                "event_type": event_type.value,
                **data,
            },
        )

        # SSE publish - use running loop to avoid deprecation warnings
        if self._sse:
            try:
                import asyncio
                loop = asyncio.get_running_loop()
                loop.create_task(
                    self._sse.publish(self.session_id, event_type.value, data)
                )
            except RuntimeError:
                # No event loop running - skip SSE for this event
                pass
