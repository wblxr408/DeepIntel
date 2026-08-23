"""Time helpers with explicit UTC semantics for legacy naive database fields."""

from __future__ import annotations

from datetime import UTC, datetime


def utc_now_naive() -> datetime:
    """Return the current UTC instant for TIMESTAMP WITHOUT TIME ZONE storage."""
    return datetime.now(UTC).replace(tzinfo=None)
