"""Administrator bootstrap, session and API-token management endpoints."""

from __future__ import annotations

from datetime import UTC, datetime

from fastapi import APIRouter, Depends, HTTPException, Request, Response, status
from pydantic import BaseModel, Field

from app.config import get_settings
from app.db.connection import get_db_pool
from app.security.auth import (
    Principal,
    authenticate_password,
    bootstrap_admin,
    create_session,
    get_principal,
    is_initialized,
    new_secret,
    revoke_all_sessions,
    token_hash,
)

router = APIRouter(prefix="/api/v1/auth", tags=["auth"])
principal_dependency = Depends(get_principal)


class BootstrapRequest(BaseModel):
    username: str = Field(min_length=3, max_length=120, pattern=r"^[A-Za-z0-9_.-]+$")
    password: str = Field(min_length=12, max_length=256)


class LoginRequest(BaseModel):
    username: str = Field(min_length=1, max_length=120)
    password: str = Field(min_length=1, max_length=256)


class PasswordChangeRequest(BaseModel):
    current_password: str = Field(min_length=1, max_length=256)
    new_password: str = Field(min_length=12, max_length=256)


class TokenCreateRequest(BaseModel):
    label: str = Field(min_length=1, max_length=120)
    expires_in_days: int | None = Field(default=None, ge=1, le=365)


@router.get("/status")
async def auth_status():
    return {"initialized": await is_initialized(), "auth_enabled": get_settings().security.auth_enabled}


def _set_session_cookie(response: Response, token: str, expires_at: datetime) -> None:
    settings = get_settings().security
    response.set_cookie(
        settings.cookie_name,
        token,
        httponly=True,
        secure=settings.secure_cookies,
        samesite="strict",
        max_age=settings.session_ttl_seconds,
        path="/",
    )


@router.post("/initialize", status_code=status.HTTP_201_CREATED)
async def initialize(payload: BootstrapRequest, response: Response):
    if await is_initialized():
        raise HTTPException(status_code=409, detail="administrator_already_initialized")
    try:
        principal = await bootstrap_admin(payload.username, payload.password)
    except (ValueError, RuntimeError) as exc:
        raise HTTPException(status_code=400 if isinstance(exc, ValueError) else 409, detail=str(exc)) from exc
    token, expires_at = await create_session(principal)
    _set_session_cookie(response, token, expires_at)
    return {"principal": {"id": principal.id, "username": principal.username}, "expires_at": expires_at.isoformat()}


@router.post("/login")
async def login(payload: LoginRequest, response: Response):
    principal = await authenticate_password(payload.username, payload.password)
    if principal is None:
        raise HTTPException(status_code=401, detail="invalid_credentials")
    token, expires_at = await create_session(principal)
    _set_session_cookie(response, token, expires_at)
    return {"principal": {"id": principal.id, "username": principal.username}, "expires_at": expires_at.isoformat()}


@router.post("/logout", status_code=status.HTTP_204_NO_CONTENT)
async def logout(request: Request, response: Response, principal: Principal = principal_dependency):
    raw = request.cookies.get(get_settings().security.cookie_name)
    if raw:
        pool = await get_db_pool()
        async with pool.acquire() as conn:
            await conn.execute("UPDATE auth_sessions SET revoked_at = NOW() WHERE token_hash = $1", token_hash(raw))
    response.delete_cookie(get_settings().security.cookie_name, path="/")


@router.get("/me")
async def me(principal: Principal = principal_dependency):
    return {"id": principal.id, "username": principal.username, "auth_type": principal.auth_type}


@router.post("/password")
async def change_password(payload: PasswordChangeRequest, principal: Principal = principal_dependency):
    current = await authenticate_password(principal.username, payload.current_password)
    if current is None:
        raise HTTPException(status_code=401, detail="invalid_credentials")
    await revoke_all_sessions(principal.id, new_password=payload.new_password)
    return {"status": "password_changed_all_credentials_revoked"}


@router.post("/tokens", status_code=status.HTTP_201_CREATED)
async def create_api_token(payload: TokenCreateRequest, principal: Principal = principal_dependency):
    from datetime import timedelta
    token = new_secret("di")
    expires_at = (datetime.now(UTC) + timedelta(days=payload.expires_in_days)).replace(tzinfo=None) if payload.expires_in_days else None
    pool = await get_db_pool()
    async with pool.acquire() as conn:
        version = await conn.fetchval("SELECT session_version FROM auth_principals WHERE id = $1::uuid", principal.id)
        row = await conn.fetchrow(
            """INSERT INTO auth_api_tokens (principal_id, token_hash, token_prefix, label, session_version, expires_at)
               VALUES ($1::uuid, $2, $3, $4, $5, $6) RETURNING id""",
            principal.id, token_hash(token), token[:12], payload.label, version, expires_at,
        )
    return {"id": str(row["id"]), "label": payload.label, "token": token, "expires_at": expires_at.isoformat() if expires_at else None}


@router.get("/tokens")
async def list_api_tokens(principal: Principal = principal_dependency):
    pool = await get_db_pool()
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """SELECT id, token_prefix, label, expires_at, revoked_at, created_at, last_used_at
               FROM auth_api_tokens WHERE principal_id = $1::uuid ORDER BY created_at DESC""", principal.id,
        )
    return {"items": [{**dict(row), "id": str(row["id"])} for row in rows]}


@router.delete("/tokens/{token_id}", status_code=status.HTTP_204_NO_CONTENT)
async def revoke_api_token(token_id: str, principal: Principal = principal_dependency):
    pool = await get_db_pool()
    async with pool.acquire() as conn:
        result = await conn.execute(
            "UPDATE auth_api_tokens SET revoked_at = NOW() WHERE id = $1::uuid AND principal_id = $2::uuid AND revoked_at IS NULL",
            token_id, principal.id,
        )
    if result.endswith("0"):
        raise HTTPException(status_code=404, detail="token_not_found")
