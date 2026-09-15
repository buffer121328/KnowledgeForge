"""JWT/API Key authentication middleware using authoritative auth services."""

from __future__ import annotations

import logging
from typing import Any
from uuid import uuid4

from fastapi import Request
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

from auth.apikey_service import APIKeyService, get_default_service
from auth.config import AuthSettings
from auth.jwt_service import AuthService
from auth.token_blacklist import TokenBlacklist
from auth.user_service import UserService
from domain.identity import Permission, UserContext, UserRole
from infrastructure.security.security_state import SecurityStateUnavailableError

logger = logging.getLogger(__name__)

PUBLIC_PATHS: set[str] = {
    "/api/v1/auth/login", "/api/v1/auth/register", "/api/v1/auth/refresh",
    "/api/health", "/api/health/live", "/api/health/ready",
    "/api/v1/health", "/api/v1/health/live", "/api/v1/health/ready",
    "/docs", "/openapi.json", "/redoc", "/metrics",
}
PUBLIC_RESOURCE_PATHS: set[str] = {
    "/auth/login", "/auth/register", "/auth/refresh",
    "/health", "/health/live", "/health/ready",
    "/docs", "/openapi.json", "/redoc", "/metrics",
}


def _is_public_path(path: str) -> bool:
    """Report whether the public path."""
    if path in PUBLIC_PATHS:
        return True
    prefix = "/api/v1"
    if path.startswith(prefix) and path[len(prefix):] in PUBLIC_RESOURCE_PATHS:
        return True
    return False


class JWTAuthMiddleware(BaseHTTPMiddleware):
    """Authenticate Bearer JWTs and API Keys before permission middleware."""

    def __init__(
        self,
        app: Any,
        settings: AuthSettings | None = None,
        redis_client: Any = None,
        api_key_service: APIKeyService | None = None,
        user_service: UserService | None = None,
    ) -> None:
        """Initialize the JWT auth middleware."""
        super().__init__(app)
        self.settings = settings or AuthSettings()
        self.auth_service = AuthService(self.settings)
        self.token_blacklist = TokenBlacklist(self.settings, redis_client)
        self.api_key_service = api_key_service or get_default_service()
        self.user_service = user_service or UserService()

    async def dispatch(self, request: Request, call_next):
        """Dispatch the jwt auth middleware."""
        path = request.url.path
        if (
            path.startswith("/api/")
            and not path.startswith("/api/v1/")
            and path not in PUBLIC_PATHS
        ):
            return await call_next(request)
        if _is_public_path(path):
            return await call_next(request)
        auth_header = request.headers.get("Authorization", "")
        raw_key = request.headers.get("X-API-Key", "")
        try:
            if auth_header.startswith("Bearer "):
                return await self._handle_bearer_token(
                    request,
                    call_next,
                    auth_header[7:],
                )
            if raw_key:
                return await self._handle_api_key(request, call_next, raw_key)
        except SecurityStateUnavailableError:
            request_id = uuid4().hex
            return JSONResponse(
                status_code=503,
                content={
                    "detail": {
                        "code": "security_state_unavailable",
                        "message": "安全状态服务暂不可用，请稍后重试。",
                        "request_id": request_id,
                    }
                },
                headers={"X-Request-ID": request_id},
            )
        return JSONResponse(status_code=401, content={"detail": "缺少认证信息，请提供 Authorization 或 X-API-Key 头"})

    async def _handle_bearer_token(self, request: Request, call_next, token: str):
        """Handle the bearer token."""
        if await self.token_blacklist.contains(token):
            return JSONResponse(status_code=401, content={"detail": "Token 已注销"})
        payload = self.auth_service.decode_token(token)
        if not payload:
            return JSONResponse(status_code=401, content={"detail": "Token 无效或已过期"})
        if payload.type != "access":
            return JSONResponse(status_code=401, content={"detail": "请使用 Access Token"})
        if not self.auth_service.is_authorization_current(payload):
            return JSONResponse(status_code=401, content={"detail": "Token 授权已更新，请重新登录"})
        user_record = self.user_service.find_by_id(payload.sub, raise_on_missing=False)
        if not user_record:
            return JSONResponse(status_code=401, content={"detail": "Token 无效或已过期"})
        if not user_record.get("is_active", True):
            return JSONResponse(status_code=401, content={"detail": "账号已被禁用"})
        if payload.token_version != user_record.get("token_version", 0):
            return JSONResponse(status_code=401, content={"detail": "Token 已失效，请重新登录"})
        current_org = str(user_record.get("org_id") or "")
        if not current_org or payload.org_id != current_org:
            return JSONResponse(status_code=401, content={"detail": "Token 无效或已过期"})
        request.state.user = self.auth_service.token_to_context(payload)
        return await call_next(request)

    async def _handle_api_key(self, request: Request, call_next, raw_key: str):
        """Resolve a raw Key through the same service used by API Key routes."""
        api_key = self.api_key_service.verify(raw_key, record_use=False)
        if not api_key:
            return JSONResponse(status_code=401, content={"detail": "API Key 无效"})
        owner = self.user_service.find_by_id(
            api_key.user_id,
            raise_on_missing=False,
        )
        if (
            not owner
            or not owner.get("is_active", True)
            or not api_key.org_id
            or owner.get("org_id") != api_key.org_id
        ):
            return JSONResponse(status_code=401, content={"detail": "API Key 无效"})
        permissions = [
            scope if isinstance(scope, Permission) else Permission(scope)
            for scope in api_key.scopes
        ]
        if not self.api_key_service.mark_used(api_key.id, raw_key=raw_key):
            return JSONResponse(status_code=401, content={"detail": "API Key 无效"})
        request.state.user = UserContext(
            user_id=api_key.user_id,
            username=f"apikey:{api_key.name}",
            role=UserRole.API_USER,
            org_id=api_key.org_id,
            permissions=permissions,
            api_key_id=api_key.id,
        )
        return await call_next(request)
