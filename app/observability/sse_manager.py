"""
SSE Manager: manages Server-Sent Events for real-time streaming.

Handles multiple concurrent SSE connections with per-session event routing.
Thread-safe implementation with proper lock management.
"""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import AsyncGenerator
from typing import Any

logger = logging.getLogger(__name__)


class SSEManager:
    """
    Manages SSE connections and event streaming to clients.

    Architecture:
    - Each research session has its own event queue
    - Events are pushed as JSON-formatted SSE messages
    - Clients connect via GET /api/v1/research/stream/{session_id}
    - Events are sent via publish_event(session_id, event_type, data)

    Event types:
    - agent_start: An agent node begins execution
    - thought: LLM thinking process (streamed)
    - tool_call: Tool invocation request
    - tool_result: Tool result returned
    - tool_error: Tool call failed
    - state_update: Workflow state changed
    - reflection: Reflection result available
    - report_chunk: Report content chunk (Markdown)
    - report_citation: Citation added to report
    - workflow_error: Workflow/business error occurred
    - done: Task completed

    Thread Safety:
    - _get_queue is protected by a dedicated lock dictionary
    - Uses setdefault pattern to atomically create locks
    - History updates use the session-specific lock
    """

    def __init__(self):
        self._sessions: dict[str, asyncio.Queue] = {}
        self._locks: dict[str, asyncio.Lock] = {}
        self._events_per_session: dict[str, list[dict]] = {}
        self._max_history = 500
        # Lock to protect session creation (avoid race condition)
        self._session_creation_lock = asyncio.Lock()

    async def _get_queue(self, session_id: str) -> asyncio.Queue:
        """
        Get or create an event queue for a session.

        Thread-safe: uses dedicated lock to prevent race conditions
        during concurrent session creation.
        """
        # Fast path: check without lock (avoids lock contention)
        if session_id in self._sessions:
            return self._sessions[session_id]

        # Slow path: create session with lock
        async with self._session_creation_lock:
            # Double-check after acquiring lock
            if session_id not in self._sessions:
                self._sessions[session_id] = asyncio.Queue(maxsize=100)
                # Use setdefault to atomically get/create the lock
                # This fixes the bug where get() with default didn't store the lock
                self._locks[session_id] = asyncio.Lock()
                self._events_per_session[session_id] = []
            return self._sessions[session_id]

    async def publish(
        self,
        session_id: str,
        event_type: str,
        data: dict[str, Any],
    ) -> None:
        """
        Publish an event to a session's event queue.

        Args:
            session_id: The research session ID
            event_type: Type of event (e.g., "thought", "tool_call")
            data: Event payload
        """
        queue = await self._get_queue(session_id)

        event = {
            "type": event_type,
            "data": data,
        }

        # Store in history - use setdefault to get or create lock atomically
        # This fixes the bug where locks.get() created throwaway locks
        async with self._locks.setdefault(session_id, asyncio.Lock()):
            history = self._events_per_session.get(session_id, [])
            history.append(event)
            if len(history) > self._max_history:
                history[:] = history[-self._max_history:]
            self._events_per_session[session_id] = history

        # Push to queue (non-blocking)
        try:
            queue.put_nowait(event)
        except asyncio.QueueFull:
            logger.warning(f"SSE queue full for session {session_id}, dropping event")

    async def stream(self, session_id: str) -> AsyncGenerator[dict[str, Any], None]:
        """
        Stream events to a client via SSE.

        Yields event dictionaries suitable for EventSourceResponse.
        """
        queue = await self._get_queue(session_id)

        # Send initial connection event
        yield {
            "event": "connected",
            "data": json.dumps({"session_id": session_id}),
        }

        try:
            # Replay buffered history so late subscribers do not miss the
            # startup burst before the EventSource handshake completes.
            async with self._locks.setdefault(session_id, asyncio.Lock()):
                history = list(self._events_per_session.get(session_id, []))

            for event in history:
                yield {
                    "event": event["type"],
                    "data": json.dumps(event["data"], ensure_ascii=False),
                }

            while True:
                event = await queue.get()
                yield {
                    "event": event["type"],
                    "data": json.dumps(event["data"], ensure_ascii=False),
                }
        except asyncio.CancelledError:
            logger.info(f"SSE stream cancelled for session {session_id}")
            raise

    def get_history(self, session_id: str) -> list[dict]:
        """Get the event history for a session."""
        return self._events_per_session.get(session_id, [])

    def close_session(self, session_id: str) -> None:
        """
        Clean up a session's resources.

        Note: This should be called when a session is permanently closed.
        For SSE disconnection, just close the connection - the session
        will remain for potential reconnection.
        """
        if session_id in self._sessions:
            del self._sessions[session_id]
        if session_id in self._locks:
            del self._locks[session_id]
        if session_id in self._events_per_session:
            del self._events_per_session[session_id]
        logger.info(f"Closed SSE session {session_id}")

    async def event_generator(
        self, session_id: str
    ) -> AsyncGenerator[dict[str, Any], None]:
        """Alias for stream(), compatible with EventSourceResponse."""
        async for event in self.stream(session_id):
            yield event

    async def wait_for_completion(
        self,
        session_id: str,
        timeout: float = 300.0,
    ) -> list[dict]:
        """
        Wait for a session to complete (receives 'done' or 'workflow_error' event).

        Useful for testing and synchronous use cases.

        Args:
            session_id: The session to wait for
            timeout: Maximum seconds to wait

        Returns:
            List of all events received
        """
        events = []
        loop = asyncio.get_running_loop()
        deadline = loop.time() + timeout

        async for event in self.stream(session_id):
            events.append(event)
            if event.get("event") in ("done", "workflow_error"):
                return events

            if loop.time() > deadline:
                logger.warning(f"Timeout waiting for session {session_id}")
                return events

        return events


# Global SSE manager instance
_sse_manager: SSEManager | None = None


def get_sse_manager() -> SSEManager:
    """Get the global SSE manager instance."""
    global _sse_manager
    if _sse_manager is None:
        _sse_manager = SSEManager()
    return _sse_manager
