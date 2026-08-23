from __future__ import annotations

import pytest
from fastapi import HTTPException

from app.api import backup
from app.security.auth import Principal

OWNER = Principal("00000000-0000-0000-0000-000000000001", "owner", "test")


class _Transaction:
    def __init__(self, conn) -> None:
        self.conn = conn

    async def __aenter__(self):
        self.conn.transaction_entered += 1
        return self

    async def __aexit__(self, exc_type, exc, tb):
        self.conn.transaction_error = exc_type is not None
        return False


class _Conn:
    def __init__(self) -> None:
        self.transaction_entered = 0
        self.transaction_error = False

    async def fetchval(self, *args):
        return False

    def transaction(self):
        return _Transaction(self)

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        return False


class _Pool:
    def __init__(self, conn) -> None:
        self.conn = conn

    def acquire(self):
        return self.conn


async def _pool(pool):
    return pool


def _source(name: str) -> dict[str, object]:
    return {"name": name, "group_name": "group", "original_text": f"content-{name}"}


@pytest.mark.asyncio
async def test_backup_import_uses_one_transaction_and_rolls_back_batch(monkeypatch):
    conn = _Conn()
    calls = 0

    async def ingest(**kwargs):
        nonlocal calls
        calls += 1
        assert kwargs["connection"] is conn
        assert kwargs["owner_id"] == OWNER.id
        if calls == 2:
            raise RuntimeError("second source fails")
        return {"source_id": "first"}

    monkeypatch.setattr(backup, "get_db_pool", lambda: _pool(_Pool(conn)))
    monkeypatch.setattr(backup, "_ingest_source", ingest)

    with pytest.raises(RuntimeError, match="second source fails"):
        await backup.import_backup(backup.BackupImportRequest(version=1, sources=[_source("one"), _source("two")]), OWNER)

    assert conn.transaction_entered == 1
    assert conn.transaction_error is True


@pytest.mark.asyncio
async def test_backup_import_rejects_unrecognized_data_before_ingestion(monkeypatch):
    conn = _Conn()
    monkeypatch.setattr(backup, "get_db_pool", lambda: _pool(_Pool(conn)))

    with pytest.raises(HTTPException, match="invalid_backup_source"):
        await backup.import_backup(backup.BackupImportRequest(version=1, sources=[{"name": "x", "unknown": True}]), OWNER)

    assert conn.transaction_entered == 1
    assert conn.transaction_error is True
