"""Single-flight, throttled health tracking for optional external capabilities."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import asdict, dataclass
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from typing import Any

from app.config import get_settings


class CapabilityHealth(StrEnum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    RECOVERING = "recovering"


@dataclass
class CapabilityStatus:
    capability: str
    state: CapabilityHealth = CapabilityHealth.HEALTHY
    reason_code: str | None = None
    last_failure_at: datetime | None = None
    retry_after_seconds: int | None = None

    def as_event(self, *, fallback_used: bool = False) -> dict[str, Any]:
        event = asdict(self)
        event["state"] = self.state.value
        event["last_failure_at"] = self.last_failure_at.isoformat() if self.last_failure_at else None
        event["fallback_used"] = fallback_used
        return event


class CapabilityRegistry:
    def __init__(self) -> None:
        self._statuses: dict[str, CapabilityStatus] = {}
        self._locks: dict[str, asyncio.Lock] = {}

    def status(self, capability: str) -> CapabilityStatus:
        return self._statuses.setdefault(capability, CapabilityStatus(capability=capability))

    def mark_failure(self, capability: str, exc: BaseException | str) -> CapabilityStatus:
        status = self.status(capability)
        status.state = CapabilityHealth.DEGRADED
        status.reason_code = type(exc).__name__ if isinstance(exc, BaseException) else str(exc)
        status.last_failure_at = datetime.now(UTC)
        resilience = getattr(get_settings(), "resilience", None)
        status.retry_after_seconds = getattr(resilience, "capability_retry_seconds", 30)
        return status

    def mark_healthy(self, capability: str) -> CapabilityStatus:
        status = self.status(capability)
        status.state = CapabilityHealth.HEALTHY
        status.reason_code = None
        status.last_failure_at = None
        status.retry_after_seconds = None
        return status

    def can_attempt(self, capability: str) -> bool:
        status = self.status(capability)
        if status.state == CapabilityHealth.HEALTHY or not status.last_failure_at:
            return True
        elapsed = datetime.now(UTC) - status.last_failure_at
        resilience = getattr(get_settings(), "resilience", None)
        return elapsed >= timedelta(seconds=getattr(resilience, "capability_retry_seconds", 30))

    async def run(self, capability: str, operation: Callable[[], Awaitable[Any]]) -> Any:
        """Run an optional operation once; suppress retry storms during cooldown."""
        if not self.can_attempt(capability):
            raise CapabilityUnavailable(self.status(capability))
        lock = self._locks.setdefault(capability, asyncio.Lock())
        async with lock:
            if not self.can_attempt(capability):
                raise CapabilityUnavailable(self.status(capability))
            status = self.status(capability)
            status.state = CapabilityHealth.RECOVERING
            try:
                result = await operation()
            except Exception as exc:
                raise CapabilityUnavailable(self.mark_failure(capability, exc)) from exc
            self.mark_healthy(capability)
            return result


class CapabilityUnavailable(RuntimeError):
    def __init__(self, status: CapabilityStatus):
        self.status = status
        super().__init__(f"{status.capability} is {status.state.value}: {status.reason_code}")


capability_registry = CapabilityRegistry()
