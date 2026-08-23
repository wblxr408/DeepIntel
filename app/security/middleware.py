"""HTTP boundary protections applied once for every request."""

from __future__ import annotations

import time
from collections import defaultdict, deque

from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse

from app.config import get_settings


class InMemoryRateLimiter:
    """Conservative local limiter; failed shared caches never grant access."""
    def __init__(self) -> None:
        self._hits: dict[str, deque[float]] = defaultdict(deque)

    def allowed(self, key: str, limit: int, period: int = 60) -> bool:
        now = time.monotonic()
        bucket = self._hits[key]
        while bucket and now - bucket[0] >= period:
            bucket.popleft()
        if len(bucket) >= limit:
            return False
        bucket.append(now)
        return True


rate_limiter = InMemoryRateLimiter()
PUBLIC_PATHS = {"/", "/api/v1/health", "/api/v1/ready", "/api/v1/auth/status", "/api/v1/auth/initialize", "/api/v1/auth/login"}


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):  # type: ignore[no-untyped-def]
        settings = get_settings()
        path = request.url.path
        client = request.client.host if request.client else "unknown"
        if path == "/api/v1/auth/login" and not rate_limiter.allowed(f"login:{client}", settings.security.login_rate_limit):
            return JSONResponse({"detail": "rate_limited"}, status_code=429)
        if path == "/api/v1/documents/upload" and not rate_limiter.allowed(f"upload:{client}", settings.security.upload_rate_limit):
            return JSONResponse({"detail": "rate_limited"}, status_code=429)
        if settings.security.auth_enabled and path.startswith("/api/") and path not in PUBLIC_PATHS:
            authorization = request.headers.get("authorization")
            if not authorization and not request.cookies.get(settings.security.cookie_name):
                return JSONResponse({"detail": "authentication_required"}, status_code=401)
        response = await call_next(request)
        response.headers["Content-Security-Policy"] = "default-src 'self'; base-uri 'self'; frame-ancestors 'none'; object-src 'none'"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        response.headers["Permissions-Policy"] = "geolocation=(), microphone=(), camera=()"
        return response
