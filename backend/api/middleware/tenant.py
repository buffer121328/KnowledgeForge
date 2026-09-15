"""多租户上下文中间件

策略:
  - 行级隔离（tenant_id 字段）+ 应用层过滤
  - tenant_id 只从已认证 UserContext 提取
  - 注入 request.state.tenant_id
  - 默认使用 UserContext.org_id 作为 tenant_id

使用方式（在 main.py 中）:
  app.add_middleware(TenantMiddleware)

查询时:
  results = await vector_store.search(query, tenant_id=request.state.tenant_id)
"""

from __future__ import annotations

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request

from shared.utils.logging import bind_context, get_logger

logger = get_logger(__name__)


class TenantMiddleware(BaseHTTPMiddleware):
    """租户上下文中间件

    注入 request.state.tenant_id，供下游服务使用。
    """

    async def dispatch(self, request: Request, call_next):
        # Tenant authority comes exclusively from the authenticated context.
        """Dispatch the tenant middleware."""
        user = getattr(request.state, "user", None)
        tenant_id = ""
        if user and getattr(user, "org_id", None):
            tenant_id = user.org_id

        request.state.tenant_id = tenant_id

        # Bind only authenticated tenant data to logging context.
        if tenant_id:
            bind_context(tenant_id=tenant_id)

        return await call_next(request)
