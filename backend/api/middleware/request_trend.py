"""请求趋势采集中间件 — 统计 API 调用量供 Dashboard 使用"""

from __future__ import annotations

import time

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request

from infrastructure.audit.trends import record_request


class RequestTrendMiddleware(BaseHTTPMiddleware):
    """Record authenticated organization API requests into UTC minute buckets."""

    SKIP_EXACT = {
        "/metrics",
        "/health",
        "/api/health",
        "/api/health/live",
        "/api/health/ready",
        "/api/v1/health",
        "/openapi.json",
        "/docs",
        "/redoc",
        "/favicon.ico",
        "/api/v1/admin/request-trends",
        "/api/admin/request-trends",
    }

    async def dispatch(self, request: Request, call_next):
        """Dispatch the request trend middleware."""
        started = time.perf_counter()
        raw_path = request.url.path
        should_record = raw_path.startswith("/api") and raw_path not in self.SKIP_EXACT
        status_code = 500
        try:
            response = await call_next(request)
            status_code = response.status_code
            return response
        finally:
            user = getattr(request.state, "user", None)
            org_id = str(getattr(user, "org_id", "") or "")
            if should_record and org_id:
                route = request.scope.get("route")
                path = str(getattr(route, "path", raw_path))
                try:
                    record_request(
                        path,
                        org_id=org_id,
                        is_qa="/qa/ask" in path,
                        method=request.method,
                        status_code=status_code,
                        duration_ms=(time.perf_counter() - started) * 1000,
                    )
                except Exception:
                    # Metrics must never alter the business response.
                    pass
