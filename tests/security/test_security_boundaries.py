from __future__ import annotations

from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.security import HTTPAuthorizationCredentials
from fastapi.testclient import TestClient

from app.llm_adapters import AnthropicCompatibleClient
from app.security import auth
from app.security.middleware import InMemoryRateLimiter, SecurityHeadersMiddleware


class _Transaction:
    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        return False


class _Connection:
    def __init__(self) -> None:
        self.fetchval_result = None
        self.fetchrow_result = None
        self.executed: list[tuple[str, tuple[object, ...]]] = []

    async def fetchval(self, query: str, *args):
        return self.fetchval_result

    async def fetchrow(self, query: str, *args):
        return self.fetchrow_result

    async def execute(self, query: str, *args):
        self.executed.append((query, args))
        return "UPDATE 1"

    def transaction(self):
        return _Transaction()

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        return False


class _Pool:
    def __init__(self, conn: _Connection) -> None:
        self.conn = conn

    def acquire(self):
        return self.conn


def _settings(*, auth_enabled: bool = True):
    return SimpleNamespace(
        security=SimpleNamespace(
            auth_enabled=auth_enabled,
            cookie_name="session",
            session_ttl_seconds=3600,
            login_rate_limit=1,
            upload_rate_limit=1,
        )
    )


@pytest.mark.asyncio
async def test_auth_database_paths_and_revocation(monkeypatch):
    conn = _Connection()
    monkeypatch.setattr(auth, "get_db_pool", lambda: _pool_async(_Pool(conn)))
    monkeypatch.setattr(auth, "get_settings", _settings)

    conn.fetchval_result = True
    assert await auth.is_initialized() is True

    with pytest.raises(ValueError):
        await auth.bootstrap_admin("admin", "strong-password-123")

    conn.fetchval_result = False
    conn.fetchrow_result = {"id": "00000000-0000-0000-0000-000000000001", "username": "owner"}
    principal = await auth.bootstrap_admin("Owner", "strong-password-123")
    assert principal.username == "owner"
    assert any("pg_advisory_xact_lock" in query for query, _ in conn.executed)

    conn.fetchval_result = 3
    token, expires_at = await auth.create_session(principal)
    assert token.startswith("ds_") and expires_at.year >= 2026

    encoded = auth.hash_password("strong-password-123")
    conn.fetchrow_result = {"id": principal.id, "username": "owner", "password_hash": encoded}
    assert await auth.authenticate_password("OWNER", "strong-password-123") == auth.Principal(principal.id, "owner", "session")
    assert await auth.authenticate_password("owner", "wrong-password") is None

    conn.fetchrow_result = {"id": principal.id, "username": "owner"}
    assert await auth._principal_from_token("token", "api_token") == auth.Principal(principal.id, "owner", "api_token")
    await auth.revoke_all_sessions(principal.id, new_password="another-strong-password")
    assert any("auth_api_tokens" in query for query, _ in conn.executed)


@pytest.mark.asyncio
async def test_auth_principal_rejects_missing_and_accepts_bearer(monkeypatch):
    from starlette.requests import Request

    monkeypatch.setattr(auth, "get_settings", lambda: _settings(auth_enabled=False))
    request = Request({"type": "http", "headers": [], "path": "/api/v1/test"})
    assert (await auth.get_principal(request, None)).auth_type == "disabled"

    monkeypatch.setattr(auth, "get_settings", _settings)
    with pytest.raises(Exception) as rejected:
        await auth.get_principal(request, None)
    assert rejected.value.status_code == 401

    expected = auth.Principal("00000000-0000-0000-0000-000000000001", "owner", "api_token")
    monkeypatch.setattr(auth, "_principal_from_token", _return_async(expected))
    credentials = HTTPAuthorizationCredentials(scheme="Bearer", credentials="raw")
    assert await auth.get_principal(request, credentials) == expected


def test_security_middleware_blocks_and_sets_headers(monkeypatch):
    from app.security import middleware

    middleware.rate_limiter = InMemoryRateLimiter()
    monkeypatch.setattr(middleware, "get_settings", _settings)
    app = FastAPI()
    app.add_middleware(SecurityHeadersMiddleware)

    @app.get("/api/v1/private")
    async def private():
        return {"ok": True}

    @app.post("/api/v1/auth/login")
    async def login():
        return {"ok": True}

    with TestClient(app) as client:
        denied = client.get("/api/v1/private")
        allowed = client.get("/api/v1/private", headers={"Authorization": "Bearer token"})
        first_login = client.post("/api/v1/auth/login")
        second_login = client.post("/api/v1/auth/login")

    assert denied.status_code == 401
    assert allowed.headers["X-Frame-Options"] == "DENY"
    assert allowed.headers["Content-Security-Policy"].startswith("default-src")
    assert first_login.status_code == 200 and second_login.status_code == 429


class _Response:
    def __init__(self, payload: dict, lines: list[str] | None = None) -> None:
        self.payload = payload
        self.lines = lines or []

    def raise_for_status(self):
        return None

    def json(self):
        return self.payload

    def iter_lines(self):
        return iter(self.lines)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


class _HttpClient:
    def __init__(self) -> None:
        self.post_body = None

    def post(self, url, *, headers, json):
        self.post_body = json
        return _Response({"content": [{"type": "text", "text": "answer"}], "usage": {"input_tokens": 2, "output_tokens": 3}})

    def stream(self, *args, **kwargs):
        return _Response({}, ['event: ping', 'data: {"type":"content_block_delta","delta":{"text":"stream"}}'])


def test_anthropic_adapter_normalizes_regular_and_streaming_responses():
    http = _HttpClient()
    client = AnthropicCompatibleClient(http, "https://anthropic.example", {"x-api-key": "secret"})
    messages = [{"role": "system", "content": "system"}, {"role": "user", "content": "question"}]

    response = client.chat.completions.create(model="claude", messages=messages, temperature=0.2)
    streamed = list(client.chat.completions.create(model="claude", messages=messages, stream=True))

    assert response.choices[0].message.content == "answer"
    assert response.usage.prompt_tokens == 2
    assert http.post_body["system"] == "system"
    assert streamed[0].choices[0].delta.content == "stream"


async def _pool_async(pool):
    return pool


def _return_async(value):
    async def _inner(*args, **kwargs):
        return value

    return _inner
