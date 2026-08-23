from __future__ import annotations

import pytest

from app.api import config


class _Connection:
    def __init__(self) -> None:
        self.fetchrow_calls: list[tuple[str, tuple[object, ...]]] = []
        self.execute_calls: list[tuple[str, tuple[object, ...]]] = []

    async def fetchrow(self, query: str, *args: object):
        self.fetchrow_calls.append((query, args))

    async def execute(self, query: str, *args: object):
        self.execute_calls.append((query, args))

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args: object):
        return False


class _Pool:
    def __init__(self, connection: _Connection) -> None:
        self.connection = connection

    def acquire(self) -> _Connection:
        return self.connection


async def _pool(pool: _Pool) -> _Pool:
    return pool


@pytest.mark.asyncio
async def test_llm_config_lookup_is_scoped_to_the_requesting_owner(monkeypatch):
    connection = _Connection()
    owner_id = "00000000-0000-0000-0000-000000000001"
    monkeypatch.setattr(config, "get_db_pool", lambda: _pool(_Pool(connection)))

    assert await config.get_llm_config_from_db(owner_id) is None

    query, args = connection.fetchrow_calls[0]
    assert "owner_id = $1::uuid" in query
    assert args == (owner_id, f"llm:{owner_id}")


@pytest.mark.asyncio
async def test_llm_config_save_records_owner_scope(monkeypatch):
    connection = _Connection()
    owner_id = "00000000-0000-0000-0000-000000000001"
    request = config.LLMConfigRequest(provider="openai", model="gpt-4o-mini", api_key="secret")
    monkeypatch.setattr(config, "get_db_pool", lambda: _pool(_Pool(connection)))
    monkeypatch.setattr(config, "encrypt_secret", lambda value: "enc:ciphertext")

    await config.save_llm_config_to_db(request, owner_id)

    query, args = connection.execute_calls[0]
    assert "owner_id" in query
    assert args[0] == f"llm:{owner_id}"
    assert args[2] == owner_id
    assert "secret" not in args[1]
