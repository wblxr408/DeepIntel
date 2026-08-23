"""
Health check API endpoints.
"""

from __future__ import annotations

import logging
import sys

import asyncpg
from fastapi import APIRouter, Response, status
from pydantic import BaseModel
from redis.exceptions import RedisError

from app.core.time import utc_now_naive

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1", tags=["health"])


class HealthResponse(BaseModel):
    status: str
    timestamp: str
    version: str
    python_version: str


class ReadinessResponse(BaseModel):
    status: str
    database: str
    redis: str
    auth: str
    capabilities: dict[str, str]
    timestamp: str


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """Basic health check endpoint."""
    return HealthResponse(
        status="healthy",
        timestamp=utc_now_naive().isoformat(),
        version="1.0.0",
        python_version=sys.version,
    )


@router.get("/ready", response_model=ReadinessResponse)
async def readiness_check(response: Response):
    """
    Readiness check: verify all dependencies are available.

    Checks:
    - PostgreSQL connection
    - Redis connection
    """
    db_status = "unavailable"
    redis_status = "unavailable"
    auth_status = "unavailable"

    # Check database
    try:
        from app.db.connection import get_db_pool
        pool = await get_db_pool()
        async with pool.acquire() as conn:
            await conn.fetchval("SELECT 1")
        db_status = "connected"
    except (OSError, RuntimeError, asyncpg.PostgresError):
        logger.warning("Database readiness check failed")

    # Check Redis
    try:
        from app.db.connection import get_redis
        redis = await get_redis()
        await redis.ping()
        redis_status = "connected"
    except (OSError, RuntimeError, RedisError):
        logger.warning("Redis readiness check failed")

    try:
        from app.config import get_settings
        settings = get_settings()
        if not settings.security.auth_enabled or settings.security.encryption_key:
            auth_status = "configured"
    except (RuntimeError, ValueError):
        logger.warning("Security readiness check failed")

    # Redis is a cache/optional SSE optimization. Database and security material
    # are the authority required to accept authenticated requests.
    overall_status = "ready" if (db_status == "connected" and auth_status == "configured") else "not_ready"
    if overall_status != "ready":
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    from app.security.capabilities import capability_registry
    capabilities = {name: item.state.value for name, item in capability_registry._statuses.items()}

    return ReadinessResponse(
        status=overall_status,
        database=db_status,
        redis=redis_status,
        auth=auth_status,
        capabilities=capabilities,
        timestamp=utc_now_naive().isoformat(),
    )
