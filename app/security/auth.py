"""Single-administrator authentication backed by PostgreSQL, not Redis."""

from __future__ import annotations

import hashlib
import secrets
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

import bcrypt
from fastapi import Depends, HTTPException, Request, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from app.config import get_settings
from app.db.connection import get_db_pool

bearer_scheme = HTTPBearer(auto_error=False)
credentials_dependency = Depends(bearer_scheme)
RESERVED_USERNAMES = {"admin", "root", "system", "support", "api", "anonymous"}


@dataclass(frozen=True)
class Principal:
    id: str
    username: str
    auth_type: str


def hash_password(password: str) -> str:
    if len(password) < 12:
        raise ValueError("password must be at least 12 characters")
    return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")


def verify_password(password: str, encoded: str) -> bool:
    return bcrypt.checkpw(password.encode("utf-8"), encoded.encode("utf-8"))


def token_hash(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def new_secret(prefix: str) -> str:
    return f"{prefix}_{secrets.token_urlsafe(32)}"


async def is_initialized() -> bool:
    pool = await get_db_pool()
    async with pool.acquire() as conn:
        return bool(await conn.fetchval("SELECT EXISTS(SELECT 1 FROM auth_principals)"))


async def bootstrap_admin(username: str, password: str) -> Principal:
    normalized = username.strip().lower()
    if not normalized or len(normalized) > 120 or normalized in RESERVED_USERNAMES:
        raise ValueError("username is invalid or reserved")
    encoded = hash_password(password)
    pool = await get_db_pool()
    async with pool.acquire() as conn, conn.transaction():
            # Serializes first initialization across processes.
            await conn.execute("SELECT pg_advisory_xact_lock(73120401)")
            if await conn.fetchval("SELECT EXISTS(SELECT 1 FROM auth_principals)"):
                raise RuntimeError("administrator already initialized")
            row = await conn.fetchrow(
                """INSERT INTO auth_principals (username, password_hash)
                   VALUES ($1, $2) RETURNING id, username""",
                normalized,
                encoded,
            )
            owner_id = row["id"]
            for table in ("documents", "document_sources", "research_sessions", "tool_call_audit", "approval_requests", "system_config", "skill_meta"):
                await conn.execute(f"UPDATE {table} SET owner_id = $1 WHERE owner_id IS NULL", owner_id)
    return Principal(id=str(row["id"]), username=row["username"], auth_type="bootstrap")


async def create_session(principal: Principal) -> tuple[str, datetime]:
    token = new_secret("ds")
    expires_at = (datetime.now(UTC) + timedelta(seconds=get_settings().security.session_ttl_seconds)).replace(tzinfo=None)
    pool = await get_db_pool()
    async with pool.acquire() as conn:
        version = await conn.fetchval("SELECT session_version FROM auth_principals WHERE id = $1::uuid", principal.id)
        await conn.execute(
            """INSERT INTO auth_sessions (principal_id, token_hash, session_version, expires_at)
               VALUES ($1::uuid, $2, $3, $4)""",
            principal.id, token_hash(token), version, expires_at,
        )
    return token, expires_at


async def authenticate_password(username: str, password: str) -> Principal | None:
    pool = await get_db_pool()
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            "SELECT id, username, password_hash FROM auth_principals WHERE username = $1 AND is_active = TRUE",
            username.strip().lower(),
        )
    if not row or not verify_password(password, row["password_hash"]):
        return None
    return Principal(id=str(row["id"]), username=row["username"], auth_type="session")


async def _principal_from_token(raw_token: str, kind: str) -> Principal | None:
    table = "auth_api_tokens" if kind == "api_token" else "auth_sessions"
    pool = await get_db_pool()
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            f"""SELECT p.id, p.username
                 FROM {table} t JOIN auth_principals p ON p.id = t.principal_id
                 WHERE t.token_hash = $1 AND t.revoked_at IS NULL AND p.is_active = TRUE
                   AND t.session_version = p.session_version
                   AND (t.expires_at IS NULL OR t.expires_at > NOW())""",
            token_hash(raw_token),
        )
        if row:
            await conn.execute(f"UPDATE {table} SET last_used_at = NOW() WHERE token_hash = $1", token_hash(raw_token))
    return Principal(id=str(row["id"]), username=row["username"], auth_type=kind) if row else None


async def get_principal(request: Request, credentials: HTTPAuthorizationCredentials | None = credentials_dependency) -> Principal:
    if not get_settings().security.auth_enabled:
        return Principal(id="00000000-0000-0000-0000-000000000000", username="local", auth_type="disabled")
    raw = credentials.credentials if credentials else request.cookies.get(get_settings().security.cookie_name)
    principal = await _principal_from_token(raw, "api_token" if credentials else "session") if raw else None
    if principal is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="authentication_required", headers={"WWW-Authenticate": "Bearer"})
    request.state.principal = principal
    return principal


principal_dependency = Depends(get_principal)


async def require_principal(principal: Principal = principal_dependency) -> Principal:
    return principal


async def revoke_all_sessions(principal_id: str, *, new_password: str | None = None) -> None:
    pool = await get_db_pool()
    async with pool.acquire() as conn, conn.transaction():
            if new_password:
                await conn.execute(
                    "UPDATE auth_principals SET password_hash = $1, session_version = session_version + 1, updated_at = NOW() WHERE id = $2::uuid",
                    hash_password(new_password), principal_id,
                )
            else:
                await conn.execute("UPDATE auth_principals SET session_version = session_version + 1, updated_at = NOW() WHERE id = $1::uuid", principal_id)
            await conn.execute("UPDATE auth_sessions SET revoked_at = NOW() WHERE principal_id = $1::uuid AND revoked_at IS NULL", principal_id)
            await conn.execute("UPDATE auth_api_tokens SET revoked_at = NOW() WHERE principal_id = $1::uuid AND revoked_at IS NULL", principal_id)
