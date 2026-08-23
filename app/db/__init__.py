"""Database layer with lazy exports."""

from __future__ import annotations

from typing import Any

__all__ = [
    "CitationRecord",
    "Document",
    "DocumentModel",
    "ResearchSession",
    "close_db",
    "get_db_pool",
    "get_redis",
    "get_text_search_config",
    "init_db",
]


def __getattr__(name: str) -> Any:
    if name in {"init_db", "get_db_pool", "get_text_search_config", "get_redis", "close_db", "Document"}:
        from app.db.connection import (
            Document,
            close_db,
            get_db_pool,
            get_redis,
            get_text_search_config,
            init_db,
        )
        return {
            "init_db": init_db,
            "get_db_pool": get_db_pool,
            "get_text_search_config": get_text_search_config,
            "get_redis": get_redis,
            "close_db": close_db,
            "Document": Document,
        }[name]
    if name in {"DocumentModel", "ResearchSession", "CitationRecord"}:
        from app.db.models import CitationRecord, ResearchSession
        from app.db.models import Document as DocumentModel
        return {
            "DocumentModel": DocumentModel,
            "ResearchSession": ResearchSession,
            "CitationRecord": CitationRecord,
        }[name]
    raise AttributeError(name)
