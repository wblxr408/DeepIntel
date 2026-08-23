"""
Helpers for PostgreSQL full-text search configuration selection.
"""

from __future__ import annotations

import logging
from typing import Any

import asyncpg

logger = logging.getLogger(__name__)

DEFAULT_FALLBACK_CONFIGS = ("simple", "english")


def build_text_search_candidates(preferred: str) -> list[str]:
    """Build an ordered list of text search configurations to try."""
    candidates: list[str] = []
    for candidate in (preferred, *DEFAULT_FALLBACK_CONFIGS):
        normalized = candidate.strip()
        if normalized and normalized not in candidates:
            candidates.append(normalized)
    return candidates


async def resolve_text_search_config(
    conn: Any,
    preferred: str = "chinese",
    *,
    log: logging.Logger | None = None,
) -> str:
    """
    Resolve the first available PostgreSQL text search configuration.

    Falls back to built-in configurations when the preferred one is unavailable.
    """
    active_logger = log or logger

    for candidate in build_text_search_candidates(preferred):
        try:
            exists = await conn.fetchval(
                "SELECT EXISTS (SELECT 1 FROM pg_ts_config WHERE cfgname = $1::text)",
                candidate,
            )
        except (OSError, RuntimeError, asyncpg.PostgresError) as exc:
            active_logger.debug(
                "Text search config candidate '%s' could not be resolved: %s",
                candidate,
                exc,
            )
            continue

        if exists:
            if candidate != preferred:
                active_logger.warning(
                    "Preferred text search configuration '%s' is unavailable; using '%s' instead",
                    preferred,
                    candidate,
                )
            return candidate

    raise RuntimeError(
        "No PostgreSQL text search configuration is available from candidates: "
        + ", ".join(build_text_search_candidates(preferred))
    )


def regconfig_sql_literal(config_name: str) -> str:
    """Return a safely quoted SQL string literal for a regconfig name."""
    return "'" + config_name.replace("'", "''") + "'"
