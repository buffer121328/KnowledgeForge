"""Permission enforcement middleware and permission checks."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable
from uuid import uuid4

from fastapi import Request
from fastapi.responses import JSONResponse
from starlette.types import ASGIApp, Receive, Scope, Send

from api.middleware.auth import PUBLIC_PATHS, _is_public_path
from api.middleware.permission_audit import _record_audit
from api.middleware.permission_registry import _match_route
from api.middleware.permission_resolver import permission_resolver
from auth.config import AuthSettings
from domain.identity import Permission
from infrastructure.audit.log import AuditService, AuditWriteError, get_audit_service

if TYPE_CHECKING:
    from domain.identity import UserContext


logger = logging.getLogger(__name__)


# ── ASGI 权限校验中间件 ───────────────────────────────────────

class PermissionMiddleware:
    """
    ASGI 权限校验中间件。

    工作流程:
      1. 公开路径直接放行
      2. 从 request.state.user 获取已认证的用户上下文
         （由 JWTAuthMiddleware 注入）
      3. 根据 HTTP method + path 匹配路由权限注册表
      4. 如果匹配到权限要求，检查用户是否拥有
      5. 通过 → 放行；拒绝 → 返回 403

    注意: 此中间件必须在 JWTAuthMiddleware 之后注册，
    因为它依赖 request.state.user 已被注入。
    """

    def __init__(
        self,
        app: ASGIApp,
        settings: AuthSettings | None = None,
        strict: bool = True,
        audit_service: AuditService | None = None,
    ):
        """Initialize the permission middleware."""
        self.app = app
        self.settings = settings or AuthSettings()
        self.resolver = permission_resolver
        self.strict = strict  # True=严格模式(未注册路由也检查)，False=宽松模式(未注册放行)
        self.audit_service = audit_service or get_audit_service()

    async def __call__(self, scope: Scope, receive: Receive, send: Send):
        """Execute the permission middleware as a callable."""
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        request = Request(scope, receive)

        # 1. 公开路径放行
        if _is_public_path(request.url.path):
            await self.app(scope, receive, send)
            return

        # 2. 获取用户上下文
        user = getattr(request.state, "user", None)
        if not user:
            # 用户未认证，交给 JWTAuthMiddleware 处理
            await self.app(scope, receive, send)
            return

        # 3. 匹配路由权限
        method = request.method
        path = request.url.path
        required_perms = _match_route(method, path)

        # 4. 无权限要求 → 放行
        if required_perms is None:
            if self.strict:
                # 严格模式: 未注册路由默认需要 admin:user 权限
                required_perms = [Permission.ADMIN_USER]
            else:
                await self.app(scope, receive, send)
                return

        # 5. 空权限列表 → 公开路由
        if not required_perms:
            await self.app(scope, receive, send)
            return

        # 6. 检查权限
        granted = self.resolver.has_permission(user, required_perms)

        # 7. 审计日志
        user_perms = [p.value for p in self.resolver.resolve_user_permissions(user)]
        try:
            _record_audit(
                user_id=user.user_id,
                username=user.username,
                method=method,
                path=path,
                required=[p.value for p in required_perms],
                user_perms=user_perms,
                granted=granted,
                reason="" if granted else "权限不足",
                org_id=user.org_id,
                ip=request.client.host if request.client else "",
                user_agent=request.headers.get("user-agent", ""),
                audit_service=self.audit_service,
            )
        except AuditWriteError:
            logger.error(
                "required_permission_audit_failed user=%s org=%s method=%s path=%s",
                user.user_id,
                user.org_id,
                method,
                path,
            )
            if granted:
                request_id = uuid4().hex
                response = JSONResponse(
                    status_code=503,
                    content={
                        "detail": {
                            "code": "audit_persistence_unavailable",
                            "message": "安全审计暂时不可用",
                            "request_id": request_id,
                        }
                    },
                    headers={"X-Request-ID": request_id},
                )
                await response(scope, receive, send)
                return

        # 8. 放行 or 拒绝
        if granted:
            await self.app(scope, receive, send)
        else:
            response = JSONResponse(
                status_code=403,
                content={
                    "detail": "权限不足",
                    "required": [p.value for p in required_perms],
                    "your_permissions": user_perms[:10],
                },
            )
            await response(scope, receive, send)


# ── 装饰器风格权限声明 ────────────────────────────────────────

def require_permissions(
    *permissions: Permission,
    require_all: bool = False,
):
    """
    装饰器风格的权限声明 — 用于在路由函数上声明权限要求。

    用法:
        @router.get("/api/data")
        @require_permissions(Permission.DOC_READ)
        async def get_data():
            ...

    这会同时:
      1. 注册到 ROUTE_PERMISSIONS（供中间件使用）
      2. 添加路由的 openapi_extra 元数据（供文档显示）
    """
    def decorator(func: Callable) -> Callable:
        # 从路由函数提取 method 和 path（如果可用）
        """Handle decorator for the module."""
        if hasattr(func, "__wrapped__"):
            func = func.__wrapped__

        # 存储权限元数据
        if not hasattr(func, "_required_permissions"):
            func._required_permissions = []
        func._required_permissions.extend(permissions)
        func._require_all = require_all

        return func

    return decorator


# ── 权限校验结果模型 ──────────────────────────────────────────

@dataclass
class PermissionCheckResult:
    """权限校验结果"""
    granted: bool
    user_id: str
    username: str
    method: str
    path: str
    required: list[str]
    user_permissions: list[str]
    message: str = ""


def check_permission(
    user: UserContext,
    method: str,
    path: str,
) -> PermissionCheckResult:
    """
    编程式权限校验 — 用于在业务代码中手动检查权限。

    Returns:
        PermissionCheckResult 包含校验结果和详细信息
    """
    required = _match_route(method, path)
    if required is None:
        return PermissionCheckResult(
            granted=True,
            user_id=user.user_id,
            username=user.username,
            method=method,
            path=path,
            required=[],
            user_permissions=[p.value for p in user.permissions],
            message="路由未注册权限要求",
        )

    user_perms = permission_resolver.resolve_user_permissions(user)
    granted = permission_resolver.has_permission(user, required)

    return PermissionCheckResult(
        granted=granted,
        user_id=user.user_id,
        username=user.username,
        method=method,
        path=path,
        required=[p.value for p in required],
        user_permissions=[p.value for p in user_perms],
        message="" if granted else "权限不足",
    )
