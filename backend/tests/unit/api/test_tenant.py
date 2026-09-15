"""租户中间件的单元测试"""
from __future__ import annotations

from unittest.mock import MagicMock

from starlette.applications import Starlette
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Route
from starlette.testclient import TestClient

from api.middleware.tenant import TenantMiddleware


async def _index(request: Request):
    """回显 tenant_id"""
    tenant_id = getattr(request.state, "tenant_id", "")
    return JSONResponse({"tenant_id": tenant_id})


class _UserInjectorMiddleware(BaseHTTPMiddleware):
    """测试用 - 在 TenantMiddleware 之前注入模拟的 user"""

    def __init__(self, app, org_id: str = "org_from_jwt"):
        super().__init__(app)
        self.org_id = org_id

    async def dispatch(self, request, call_next):
        user = MagicMock()
        user.org_id = self.org_id
        request.state.user = user
        return await call_next(request)


class TestTenantMiddleware:
    def test_tenant_header_is_not_an_authority(self):
        """未经认证的 X-Tenant-Id 不得建立租户上下文"""
        app = Starlette(routes=[Route("/", _index)])
        app.add_middleware(TenantMiddleware)
        client = TestClient(app)
        response = client.get("/", headers={"X-Tenant-Id": "org_from_header"})
        assert response.status_code == 200
        assert response.json()["tenant_id"] == ""

    def test_tenant_empty_when_no_context(self):
        """测试无上下文时 tenant_id 为空字符串"""
        app = Starlette(routes=[Route("/", _index)])
        app.add_middleware(TenantMiddleware)
        client = TestClient(app)
        response = client.get("/")
        assert response.status_code == 200
        assert response.json()["tenant_id"] == ""

    def test_tenant_from_user_context(self):
        """测试从 UserContext.org_id 提取 tenant_id

        中间件执行顺序（Starlette 逆序执行）:
          先注册 TenantMiddleware -> 后注册 UserInjector
          执行: UserInjector 先（注入 user）-> TenantMiddleware -> handler
        """
        app = Starlette(routes=[Route("/", _index)])
        app.add_middleware(TenantMiddleware)
        app.add_middleware(_UserInjectorMiddleware, org_id="org_from_jwt")
        client = TestClient(app)
        response = client.get("/")
        assert response.status_code == 200
        assert response.json()["tenant_id"] == "org_from_jwt"
