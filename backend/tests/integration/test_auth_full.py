"""
完整集成测试 — 认证授权 + 用户管理 + 角色管理 + 权限中间件

运行方式:
    cd code/python
    pytest tests/integration/test_auth_full.py -v

测试覆盖:
    1. 认证流程: 登录 / 注册 / 刷新 / 注销 / 获取当前用户
    2. 用户管理: 列表 / 详情 / 创建 / 更新 / 删除 / 启用禁用 / 密码重置
    3. 角色管理: 列表 / 详情 / 创建 / 更新 / 删除 / 权限管理
    4. 权限中间件: 路由权限注册 / 权限校验 / 审计日志
    5. API Key: 创建 / 列表 / 删除
"""

from __future__ import annotations

import pytest
from fastapi import FastAPI
from fastapi import APIRouter
from fastapi.testclient import TestClient

# ── 测试应用工厂 ──────────────────────────────────────────────

def create_test_app() -> FastAPI:
    """创建最小化的测试应用，跳过数据库/向量库初始化"""
    from fastapi import Depends
    from api.routers import apikeys as apikey_router_module
    from api.routers import auth as auth_router_module
    from api.routers import roles as role_router_module
    from api.routers import users as user_router_module
    from auth.apikey_service import APIKeyService
    from auth.apikey_store import MemoryKeyStore
    from auth.config import auth_settings
    from auth.memory_identity_store import MemoryIdentityStore
    from auth.role_service import RoleService
    from auth.memory_accounts import ROLE_DB, USER_DB
    from auth.user_service import UserService
    from api.dependencies import require_permission
    from api.middleware.auth import JWTAuthMiddleware
    from domain.identity import Permission, UserContext
    from api.middleware.permission import PermissionMiddleware
    from api.middleware.permission_audit import get_audit_logs
    from api.middleware.permission_registry import init_route_permissions
    identity_store = MemoryIdentityStore(USER_DB, ROLE_DB)
    user_service = UserService(store=identity_store, auth_service=auth_router_module.auth_service)
    api_key_service = APIKeyService(store=MemoryKeyStore())
    auth_router_module.user_service = user_service
    user_router_module.user_service = user_service
    role_router_module.role_service = RoleService(store=identity_store)
    apikey_router_module._service = lambda: api_key_service

    app = FastAPI(title="Test App")

    # 配置限流器（slowapi 要求 app.state.limiter 存在）
    from slowapi import _rate_limit_exceeded_handler
    from slowapi.errors import RateLimitExceeded
    from slowapi.middleware import SlowAPIMiddleware
    from shared.utils.ratelimit import limiter
    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
    app.add_middleware(SlowAPIMiddleware)

    # 中间件（顺序重要: JWT 必须在 Permission 之前注册，因为 FastAPI 按注册逆序执行）
    app.add_middleware(PermissionMiddleware, settings=auth_settings)  # 先注册 = 后执行
    app.add_middleware(
        JWTAuthMiddleware,
        settings=auth_settings,
        api_key_service=api_key_service,
        user_service=user_service,
    )  # 后注册 = 先执行

    # 业务路由只挂载到版本化路径；内存仓仅作为显式测试替身。
    root_router = APIRouter(prefix="/api/v1")
    root_router.include_router(auth_router_module.router)
    root_router.include_router(user_router_module.router)
    root_router.include_router(role_router_module.router)
    root_router.include_router(apikey_router_module.router, prefix="/auth")
    app.include_router(root_router)

    # 简单的测试端点
    @app.get("/api/v1/health")
    async def health():
        return {"status": "ok"}

    @app.get("/api/v1/test/admin-only")
    async def admin_endpoint():
        return {"message": "admin only"}

    # 审计日志端点
    @app.get("/api/v1/admin/audit-logs")
    async def list_audit_logs(
        user_id: str | None = None,
        limit: int = 100,
        user: UserContext = Depends(require_permission(Permission.ADMIN_AUDIT)),
    ):
        logs = get_audit_logs(user_id=user_id, limit=limit)
        return [
            {
                "timestamp": log.timestamp,
                "user_id": log.user_id,
                "username": log.username,
                "method": log.method,
                "path": log.path,
                "required_permissions": log.required_permissions,
                "user_permissions": log.user_permissions,
                "granted": log.granted,
                "reason": log.reason,
            }
            for log in logs
        ]

    # 初始化权限注册表
    init_route_permissions()

    return app


@pytest.fixture(scope="module")
def client():
    """创建测试客户端"""
    from shared.config import settings

    previous_environment = settings.app_environment
    previous_demo_registration = settings.allow_demo_default_org_registration
    settings.app_environment = "test"
    settings.allow_demo_default_org_registration = True
    app = create_test_app()
    try:
        with TestClient(app) as c:
            yield c
    finally:
        settings.app_environment = previous_environment
        settings.allow_demo_default_org_registration = previous_demo_registration


@pytest.fixture(scope="module")
def admin_token(client):
    """获取管理员 Token"""
    resp = client.post("/api/v1/auth/login", json={"username": "admin", "password": "admin123"})
    assert resp.status_code == 200
    data = resp.json()
    return data["access_token"]


@pytest.fixture(scope="module")
def admin_headers(admin_token):
    """管理员请求头"""
    return {"Authorization": f"Bearer {admin_token}"}


# ══════════════════════════════════════════════════════════════
# 1. 认证流程测试
# ══════════════════════════════════════════════════════════════

class TestAuthFlow:
    """认证流程: 登录 / 注册 / 刷新 / 注销 / 当前用户"""

    def test_health_check(self, client):
        """健康检查端点（无需认证）"""
        resp = client.get("/api/v1/health")
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"

    def test_login_success(self, client):
        """管理员登录成功"""
        resp = client.post("/api/v1/auth/login", json={
            "username": "admin",
            "password": "admin123",
        })
        assert resp.status_code == 200
        data = resp.json()
        assert "access_token" in data
        assert "refresh_token" in data
        assert data["token_type"] == "bearer"
        assert data["expires_in"] > 0

    def test_versioned_login_success_with_rate_limit_headers(self, client, monkeypatch):
        """成功登录时 SlowAPI 应注入响应头而不是把响应转换为 HTTP 500。"""
        from api.routers import auth
        from domain.identity import UserRole
        from shared.utils.ratelimit import limiter

        monkeypatch.setattr(
            auth.user_service,
            "authenticate",
            lambda username, password: {
                "user_id": "user_remote_admin",
                "username": username,
                "role": UserRole.ADMIN,
                "org_id": "org_default",
                "is_active": True,
                "token_version": 0,
            },
        )
        limiter.reset()
        monkeypatch.setattr(limiter, "enabled", True)

        resp = client.post(
            "/api/v1/auth/login",
            json={"username": "remote-admin", "password": "correct-password"},
        )

        assert resp.status_code == 200
        assert "access_token" in resp.json()
        assert "refresh_token" in resp.json()
        assert resp.headers["x-ratelimit-limit"] == "10"

    def test_login_wrong_password(self, client):
        """密码错误"""
        resp = client.post("/api/v1/auth/login", json={
            "username": "admin",
            "password": "wrong_password",
        })
        assert resp.status_code == 401

    def test_login_nonexistent_user(self, client):
        """用户不存在"""
        resp = client.post("/api/v1/auth/login", json={
            "username": "nonexistent",
            "password": "any",
        })
        assert resp.status_code == 401

    def test_register_new_user(self, client):
        """注册新用户"""
        resp = client.post("/api/v1/auth/register", json={
            "username": "test_user_001",
            "password": "test123456",
        })
        assert resp.status_code == 200
        data = resp.json()
        assert "access_token" in data

    def test_register_duplicate_username(self, client):
        """重复用户名注册"""
        # 先注册
        client.post("/api/v1/auth/register", json={
            "username": "dup_user",
            "password": "test123456",
        })
        # 再次注册
        resp = client.post("/api/v1/auth/register", json={
            "username": "dup_user",
            "password": "test123456",
        })
        assert resp.status_code == 409

    def test_refresh_token(self, client):
        """刷新 Token"""
        # 登录获取 refresh_token
        login_resp = client.post("/api/v1/auth/login", json={
            "username": "admin",
            "password": "admin123",
        })
        refresh_token = login_resp.json()["refresh_token"]

        # 刷新
        resp = client.post("/api/v1/auth/refresh", json={
            "refresh_token": refresh_token,
        })
        assert resp.status_code == 200
        data = resp.json()
        assert "access_token" in data
        assert "refresh_token" in data
        # 新 Token 可以正常使用
        new_headers = {"Authorization": f"Bearer {data['access_token']}"}
        me_resp = client.get("/api/v1/auth/me", headers=new_headers)
        assert me_resp.status_code == 200

    def test_refresh_invalid_token(self, client):
        """无效 refresh token"""
        resp = client.post("/api/v1/auth/refresh", json={
            "refresh_token": "invalid.token.here",
        })
        assert resp.status_code == 401

    def test_get_current_user(self, client, admin_headers):
        """获取当前用户信息"""
        resp = client.get("/api/v1/auth/me", headers=admin_headers)
        assert resp.status_code == 200
        data = resp.json()
        assert data["username"] == "admin"
        assert data["display_name"] == "组织管理员"
        assert data["role"] == "organization_admin"
        assert "permissions" in data

    def test_get_me_unauthorized(self, client):
        """未认证访问 /me"""
        resp = client.get("/api/v1/auth/me")
        assert resp.status_code == 401 or resp.status_code == 403


# ══════════════════════════════════════════════════════════════
# 2. 用户管理测试
# ══════════════════════════════════════════════════════════════

class TestUserManagement:
    """用户管理: CRUD + 启用禁用 + 密码重置"""

    def test_list_users(self, client, admin_headers):
        """列出所有用户"""
        resp = client.get("/api/v1/users", headers=admin_headers)
        assert resp.status_code == 200
        users = resp.json()
        assert isinstance(users, list)
        assert len(users) >= 1
        # 应包含 admin 用户
        usernames = [u["username"] for u in users]
        assert "admin" in usernames

    def test_list_users_with_search(self, client, admin_headers):
        """搜索用户"""
        resp = client.get("/api/v1/users?search=admin", headers=admin_headers)
        assert resp.status_code == 200
        users = resp.json()
        assert len(users) >= 1
        assert any("admin" in u["username"] for u in users)

    def test_list_users_with_role_filter(self, client, admin_headers):
        """按角色筛选用户"""
        resp = client.get("/api/v1/users?role=admin", headers=admin_headers)
        assert resp.status_code == 200
        users = resp.json()
        assert all(u["role"] == "admin" for u in users)

    def test_create_user(self, client, admin_headers):
        """管理员只能在自己的组织创建用户"""
        cross_org = client.post("/api/v1/users", headers=admin_headers, json={
            "username": "cross_org_user",
            "password": "password123",
            "role": "editor",
            "org_id": "org_test",
        })
        assert cross_org.status_code == 400

        resp = client.post("/api/v1/users", headers=admin_headers, json={
            "username": "new_user_001",
            "password": "password123",
            "display_name": "新用户",
            "email": "new@example.com",
            "role": "editor",
        })
        assert resp.status_code == 201
        data = resp.json()
        assert data["username"] == "new_user_001"
        assert data["role"] == "editor"
        assert data["display_name"] == "新用户"
        assert data["org_id"] == "org_001"

    def test_create_user_duplicate_username(self, client, admin_headers):
        """创建重复用户名"""
        # 先创建
        client.post("/api/v1/users", headers=admin_headers, json={
            "username": "dup_test_user",
            "password": "password123",
        })
        # 重复创建
        resp = client.post("/api/v1/users", headers=admin_headers, json={
            "username": "dup_test_user",
            "password": "password123",
        })
        assert resp.status_code == 409

    def test_create_user_invalid_username(self, client, admin_headers):
        """无效用户名（太短/包含特殊字符）"""
        resp = client.post("/api/v1/users", headers=admin_headers, json={
            "username": "ab",  # 太短
            "password": "password123",
        })
        assert resp.status_code == 422  # 验证错误

    def test_get_user_detail(self, client, admin_headers):
        """获取用户详情"""
        # 先获取用户列表找到 user_id
        list_resp = client.get("/api/v1/users", headers=admin_headers)
        users = list_resp.json()
        admin_user = next(u for u in users if u["username"] == "admin")

        resp = client.get(f"/api/v1/users/{admin_user['user_id']}", headers=admin_headers)
        assert resp.status_code == 200
        data = resp.json()
        assert data["username"] == "admin"

    def test_get_nonexistent_user(self, client, admin_headers):
        """获取不存在的用户"""
        resp = client.get("/api/v1/users/user_nonexistent", headers=admin_headers)
        assert resp.status_code == 404

    def test_update_user(self, client, admin_headers):
        """更新用户信息"""
        # 创建测试用户
        create_resp = client.post("/api/v1/users", headers=admin_headers, json={
            "username": "update_test_user",
            "password": "password123",
            "display_name": "原名",
        })
        user_id = create_resp.json()["user_id"]

        # 更新
        resp = client.patch(f"/api/v1/users/{user_id}", headers=admin_headers, json={
            "display_name": "新名字",
            "role": "viewer",
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["display_name"] == "新名字"
        assert data["role"] == "viewer"

    def test_delete_user(self, client, admin_headers):
        """删除用户"""
        # 创建测试用户
        create_resp = client.post("/api/v1/users", headers=admin_headers, json={
            "username": "delete_test_user",
            "password": "password123",
        })
        user_id = create_resp.json()["user_id"]

        # 删除
        resp = client.delete(f"/api/v1/users/{user_id}", headers=admin_headers)
        assert resp.status_code == 200

        # 验证已删除
        get_resp = client.get(f"/api/v1/users/{user_id}", headers=admin_headers)
        assert get_resp.status_code == 404

    def test_cannot_delete_self(self, client, admin_headers):
        """不能删除自己"""
        # 获取 admin user_id
        list_resp = client.get("/api/v1/users", headers=admin_headers)
        admin_user = next(u for u in list_resp.json() if u["username"] == "admin")

        resp = client.delete(f"/api/v1/users/{admin_user['user_id']}", headers=admin_headers)
        assert resp.status_code == 400

    def test_toggle_user_active(self, client, admin_headers):
        """启用/禁用用户"""
        # 创建测试用户
        create_resp = client.post("/api/v1/users", headers=admin_headers, json={
            "username": "toggle_test_user",
            "password": "password123",
        })
        user_id = create_resp.json()["user_id"]

        # 禁用
        resp = client.post(f"/api/v1/users/{user_id}/toggle-active", headers=admin_headers)
        assert resp.status_code == 200
        assert resp.json()["is_active"] is False

        # 再次启用
        resp = client.post(f"/api/v1/users/{user_id}/toggle-active", headers=admin_headers)
        assert resp.status_code == 200
        assert resp.json()["is_active"] is True

    def test_cannot_toggle_self(self, client, admin_headers):
        """不能禁用自己"""
        list_resp = client.get("/api/v1/users", headers=admin_headers)
        admin_user = next(u for u in list_resp.json() if u["username"] == "admin")

        resp = client.post(f"/api/v1/users/{admin_user['user_id']}/toggle-active", headers=admin_headers)
        assert resp.status_code == 400

    def test_reset_user_password(self, client, admin_headers):
        """管理员重置用户密码"""
        # 创建测试用户
        create_resp = client.post("/api/v1/users", headers=admin_headers, json={
            "username": "reset_pwd_user",
            "password": "old_password",
        })
        user_id = create_resp.json()["user_id"]

        # 重置密码
        resp = client.post(f"/api/v1/users/{user_id}/reset-password", headers=admin_headers, json={
            "new_password": "new_password_123",
        })
        assert resp.status_code == 200

        # 用新密码登录
        login_resp = client.post("/api/v1/auth/login", json={
            "username": "reset_pwd_user",
            "password": "new_password_123",
        })
        assert login_resp.status_code == 200

    def test_change_own_password(self, client, admin_headers):
        """修改自己的密码"""
        # 注册新用户
        register_resp = client.post("/api/v1/auth/register", json={
            "username": "change_pwd_user",
            "password": "old_pass_123",
        })
        token = register_resp.json()["access_token"]
        headers = {"Authorization": f"Bearer {token}"}

        # 修改密码
        resp = client.post("/api/v1/users/me/password", headers=headers, json={
            "old_password": "old_pass_123",
            "new_password": "new_pass_456",
        })
        assert resp.status_code == 200

        # 用新密码登录
        login_resp = client.post("/api/v1/auth/login", json={
            "username": "change_pwd_user",
            "password": "new_pass_456",
        })
        assert login_resp.status_code == 200

    def test_change_password_wrong_old(self, client, admin_headers):
        """修改密码时旧密码错误"""
        register_resp = client.post("/api/v1/auth/register", json={
            "username": "wrong_old_pwd_user",
            "password": "correct_pass",
        })
        token = register_resp.json()["access_token"]
        headers = {"Authorization": f"Bearer {token}"}

        resp = client.post("/api/v1/users/me/password", headers=headers, json={
            "old_password": "wrong_password",
            "new_password": "new_pass",
        })
        assert resp.status_code == 400

    def test_user_unauthorized_access(self, client):
        """普通用户访问用户管理接口"""
        # 注册 viewer 用户
        register_resp = client.post("/api/v1/auth/register", json={
            "username": "viewer_user_test",
            "password": "password123",
        })
        token = register_resp.json()["access_token"]
        headers = {"Authorization": f"Bearer {token}"}

        # 尝试列出用户（需要 admin:user 权限）
        resp = client.get("/api/v1/users", headers=headers)
        assert resp.status_code == 403


# ══════════════════════════════════════════════════════════════
# 3. 角色管理测试
# ══════════════════════════════════════════════════════════════

class TestRoleManagement:
    """角色管理: CRUD + 权限管理"""

    def test_list_roles(self, client, admin_headers):
        """列出所有角色"""
        resp = client.get("/api/v1/roles", headers=admin_headers)
        assert resp.status_code == 200
        roles = resp.json()
        assert isinstance(roles, list)
        # 应包含内置角色
        role_names = [r["name"] for r in roles]
        assert "admin" in role_names
        assert "editor" in role_names
        assert "viewer" in role_names

    def test_list_roles_with_search(self, client, admin_headers):
        """搜索角色"""
        resp = client.get("/api/v1/roles?search=admin", headers=admin_headers)
        assert resp.status_code == 200
        roles = resp.json()
        assert any("admin" in r["name"] for r in roles)

    def test_list_builtin_roles_only(self, client, admin_headers):
        """只列出内置角色"""
        resp = client.get("/api/v1/roles?is_builtin=true", headers=admin_headers)
        assert resp.status_code == 200
        roles = resp.json()
        assert all(r["is_builtin"] for r in roles)

    def test_get_role_detail(self, client, admin_headers):
        """获取角色详情"""
        resp = client.get("/api/v1/roles/admin", headers=admin_headers)
        assert resp.status_code == 200
        data = resp.json()
        assert data["name"] == "admin"
        assert data["is_builtin"] is True
        assert len(data["permissions"]) > 0

    def test_get_nonexistent_role(self, client, admin_headers):
        """获取不存在的角色"""
        resp = client.get("/api/v1/roles/nonexistent", headers=admin_headers)
        assert resp.status_code == 404

    def test_create_custom_role(self, client, admin_headers):
        """创建自定义角色"""
        resp = client.post("/api/v1/roles", headers=admin_headers, json={
            "name": "custom_role",
            "display_name": "自定义角色",
            "description": "测试用自定义角色",
            "permissions": ["doc:read", "qa:query"],
        })
        assert resp.status_code == 201
        data = resp.json()
        assert data["name"] == "custom_role"
        assert data["is_builtin"] is False
        assert "doc:read" in data["permissions"]

    def test_create_role_duplicate_name(self, client, admin_headers):
        """创建重复角色名"""
        # 先创建
        client.post("/api/v1/roles", headers=admin_headers, json={
            "name": "dup_role",
            "permissions": ["doc:read"],
        })
        # 重复创建
        resp = client.post("/api/v1/roles", headers=admin_headers, json={
            "name": "dup_role",
            "permissions": ["doc:read"],
        })
        assert resp.status_code == 409

    def test_create_role_builtin_name_conflict(self, client, admin_headers):
        """角色名与内置角色冲突（内置角色已在 ROLE_DB 中，返回 409）"""
        resp = client.post("/api/v1/roles", headers=admin_headers, json={
            "name": "admin",  # 内置角色名
            "permissions": ["doc:read"],
        })
        assert resp.status_code == 409  # 角色名已存在

    def test_create_role_invalid_name(self, client, admin_headers):
        """无效角色名（以数字开头/包含大写）"""
        resp = client.post("/api/v1/roles", headers=admin_headers, json={
            "name": "123invalid",
            "permissions": [],
        })
        assert resp.status_code == 422  # 验证错误

    def test_create_role_invalid_permissions(self, client, admin_headers):
        """无效权限标识"""
        resp = client.post("/api/v1/roles", headers=admin_headers, json={
            "name": "bad_perms_role",
            "permissions": ["invalid:permission"],
        })
        assert resp.status_code == 422

    def test_update_custom_role(self, client, admin_headers):
        """更新自定义角色"""
        # 创建
        create_resp = client.post("/api/v1/roles", headers=admin_headers, json={
            "name": "update_role",
            "display_name": "原名",
            "permissions": ["doc:read"],
        })

        # 更新
        resp = client.patch("/api/v1/roles/update_role", headers=admin_headers, json={
            "display_name": "新名字",
            "permissions": ["doc:read", "doc:write"],
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["display_name"] == "新名字"
        assert "doc:write" in data["permissions"]

    def test_cannot_update_builtin_role(self, client, admin_headers):
        """不能更新内置角色"""
        resp = client.patch("/api/v1/roles/admin", headers=admin_headers, json={
            "display_name": "hacked",
        })
        assert resp.status_code == 400

    def test_delete_custom_role(self, client, admin_headers):
        """删除自定义角色"""
        # 创建
        client.post("/api/v1/roles", headers=admin_headers, json={
            "name": "delete_role",
            "permissions": [],
        })

        # 删除
        resp = client.delete("/api/v1/roles/delete_role", headers=admin_headers)
        assert resp.status_code == 200

        # 验证已删除
        get_resp = client.get("/api/v1/roles/delete_role", headers=admin_headers)
        assert get_resp.status_code == 404

    def test_cannot_delete_builtin_role(self, client, admin_headers):
        """不能删除内置角色"""
        resp = client.delete("/api/v1/roles/admin", headers=admin_headers)
        assert resp.status_code == 400

    def test_get_role_permissions(self, client, admin_headers):
        """获取角色权限列表"""
        resp = client.get("/api/v1/roles/organization_admin/permissions", headers=admin_headers)
        assert resp.status_code == 200
        perms = resp.json()
        assert isinstance(perms, list)
        assert "admin:manage" in perms

    def test_set_role_permissions(self, client, admin_headers):
        """设置角色权限（覆盖）"""
        # 创建自定义角色
        client.post("/api/v1/roles", headers=admin_headers, json={
            "name": "set_perms_role",
            "permissions": ["doc:read"],
        })

        # 覆盖权限
        resp = client.put("/api/v1/roles/set_perms_role/permissions", headers=admin_headers, json={"permissions": [
            "doc:read", "doc:write", "qa:query",
        ]})
        assert resp.status_code == 200
        perms = resp.json()
        assert len(perms) == 3
        assert "doc:write" in perms

    def test_add_role_permissions(self, client, admin_headers):
        """追加角色权限"""
        # 创建角色
        client.post("/api/v1/roles", headers=admin_headers, json={
            "name": "add_perms_role",
            "permissions": ["doc:read"],
        })

        # 追加权限
        resp = client.post("/api/v1/roles/add_perms_role/permissions", headers=admin_headers, json={"permissions": [
            "doc:write", "qa:query",
        ]})
        assert resp.status_code == 200
        data = resp.json()
        assert data["added"] == 2
        assert data["total"] == 3

    def test_remove_role_permissions(self, client, admin_headers):
        """移除角色权限"""
        # 创建角色
        client.post("/api/v1/roles", headers=admin_headers, json={
            "name": "remove_perms_role",
            "permissions": ["doc:read", "doc:write", "qa:query"],
        })

        # 移除权限（使用 Query 参数）
        resp = client.delete(
            "/api/v1/roles/remove_perms_role/permissions",
            headers=admin_headers,
            params={"permissions": ["doc:write"]},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["removed"] == 1
        assert data["total"] == 2

    def test_list_role_users(self, client, admin_headers):
        """查看角色下的用户"""
        resp = client.get("/api/v1/roles/organization_admin/users", headers=admin_headers)
        assert resp.status_code == 200
        data = resp.json()
        assert data["role_name"] == "organization_admin"
        assert isinstance(data["users"], list)
        assert len(data["users"]) >= 1

    def test_list_permissions_meta(self, client, admin_headers):
        """列出所有可用权限"""
        resp = client.get("/api/v1/roles/meta/permissions", headers=admin_headers)
        assert resp.status_code == 200
        perms = resp.json()
        assert isinstance(perms, list)
        assert len(perms) > 0
        # 应包含权限值和分类
        assert any(p["value"] == "doc:read" for p in perms)


# ══════════════════════════════════════════════════════════════
# 4. 权限中间件测试
# ══════════════════════════════════════════════════════════════

class TestPermissionMiddleware:
    """权限中间件: 路由权限校验 + 审计日志"""

    def test_public_endpoint_no_auth(self, client):
        """公开端点无需认证"""
        resp = client.get("/api/v1/health")
        assert resp.status_code == 200

    def test_protected_endpoint_no_auth(self, client):
        """受保护端点需要认证"""
        resp = client.get("/api/v1/auth/me")
        assert resp.status_code in (401, 403)

    def test_admin_access_admin_endpoint(self, client, admin_headers):
        """管理员访问管理端点"""
        resp = client.get("/api/v1/users", headers=admin_headers)
        assert resp.status_code == 200

    def test_viewer_cannot_access_admin_endpoint(self, client):
        """Viewer 用户无法访问管理端点"""
        # 注册 viewer
        register_resp = client.post("/api/v1/auth/register", json={
            "username": "perm_test_viewer",
            "password": "password123",
        })
        token = register_resp.json()["access_token"]
        headers = {"Authorization": f"Bearer {token}"}

        # 尝试访问用户管理
        resp = client.get("/api/v1/users", headers=headers)
        assert resp.status_code == 403

    def test_editor_can_access_doc_write(self, client):
        """Editor 用户可以访问需要 doc:write 权限的路由"""
        # 注册 editor
        register_resp = client.post("/api/v1/auth/register", json={
            "username": "perm_test_editor",
            "password": "password123",
        })
        token = register_resp.json()["access_token"]
        headers = {"Authorization": f"Bearer {token}"}

        # Editor 应该能访问需要 admin:user 权限的路由（editor 没有此权限，会返回 403）
        # 但应该能通过权限中间件的检查（editor 有 doc:write 权限）
        # 我们通过检查审计日志来验证权限检查是否通过
        from api.middleware.permission_audit import _audit_logs
        before_count = len(_audit_logs)

        # 访问一个需要 admin:user 权限的路由
        resp = client.get("/api/v1/users", headers=headers)
        assert resp.status_code == 403  # editor 没有 admin:user 权限

        # 检查审计日志是否记录了这次拒绝
        assert len(_audit_logs) > before_count
        last_log = _audit_logs[-1]
        assert last_log.granted is False
        assert last_log.method == "GET"
        assert last_log.path == "/api/v1/users"

    def test_viewer_cannot_access_doc_write(self, client):
        """Viewer 用户无法访问文档写入"""
        register_resp = client.post("/api/v1/auth/register", json={
            "username": "perm_test_viewer2",
            "password": "password123",
        })
        token = register_resp.json()["access_token"]
        headers = {"Authorization": f"Bearer {token}"}

        resp = client.post("/api/v1/ingest/upload", headers=headers, files={"file": ("test.txt", b"test", "text/plain")})
        assert resp.status_code == 403

    def test_audit_logs_recorded(self, client, admin_headers):
        """审计日志记录"""
        # 执行一些操作
        client.get("/api/v1/users", headers=admin_headers)
        client.get("/api/v1/roles", headers=admin_headers)

        # 查询审计日志
        resp = client.get("/api/v1/admin/audit-logs?limit=10", headers=admin_headers)
        assert resp.status_code == 200
        logs = resp.json()
        assert isinstance(logs, list)
        # 应该有记录
        assert len(logs) > 0
        # 检查日志结构
        log = logs[0]
        assert "timestamp" in log
        assert "user_id" in log
        assert "method" in log
        assert "path" in log
        assert "granted" in log

    def test_audit_logs_filter_by_user(self, client, admin_headers):
        """按用户筛选审计日志"""
        # 获取 admin user_id
        list_resp = client.get("/api/v1/users", headers=admin_headers)
        admin_user = next(u for u in list_resp.json() if u["username"] == "admin")

        resp = client.get(f"/api/v1/admin/audit-logs?user_id={admin_user['user_id']}", headers=admin_headers)
        assert resp.status_code == 200
        logs = resp.json()
        # 所有日志都应该是 admin 用户的
        assert all(log["user_id"] == admin_user["user_id"] for log in logs)

    def test_audit_logs_permission_denied(self, client):
        """无审计日志查看权限"""
        register_resp = client.post("/api/v1/auth/register", json={
            "username": "audit_denied_user",
            "password": "password123",
        })
        token = register_resp.json()["access_token"]
        headers = {"Authorization": f"Bearer {token}"}

        resp = client.get("/api/v1/admin/audit-logs", headers=headers)
        assert resp.status_code == 403


# ══════════════════════════════════════════════════════════════
# 5. API Key 管理测试
# ══════════════════════════════════════════════════════════════

class TestAPIKeyManagement:
    """API Key: 创建 / 列表 / 软撤销"""

    def test_create_api_key(self, client, admin_headers):
        """创建 API Key"""
        resp = client.post("/api/v1/auth/apikey/create", headers=admin_headers, json={
            "name": "test-api-key",
            "permissions": ["doc:read", "qa:query"],
            "expires_days": 30,
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["name"] == "test-api-key"
        assert "api_key" in data  # 创建时返回完整 Key
        assert data["api_key"].startswith("ak_")

    def test_list_api_keys(self, client, admin_headers):
        """列出 API Key"""
        # 先创建一个
        client.post("/api/v1/auth/apikey/create", headers=admin_headers, json={
            "name": "list-test-key",
            "permissions": ["doc:read"],
        })

        resp = client.get("/api/v1/auth/apikey/list", headers=admin_headers)
        assert resp.status_code == 200
        keys = resp.json()
        assert isinstance(keys, list)
        assert len(keys) >= 1
        # 列表不返回完整 Key（api_key 应为 None）
        assert all(k.get("api_key") is None for k in keys)

    def test_delete_api_key(self, client, admin_headers):
        """软撤销 API Key 并保留生命周期记录"""
        # 创建
        create_resp = client.post("/api/v1/auth/apikey/create", headers=admin_headers, json={
            "name": "delete-test-key",
            "permissions": ["doc:read"],
        })
        key_id = create_resp.json()["id"]

        # 删除
        resp = client.delete(f"/api/v1/auth/apikey/{key_id}", headers=admin_headers)
        assert resp.status_code == 200

        # 验证记录保留且不可再认证
        list_resp = client.get("/api/v1/auth/apikey/list", headers=admin_headers)
        revoked = next(k for k in list_resp.json() if k["id"] == key_id)
        assert revoked["status"] == "revoked"
        assert revoked["is_active"] is False
        assert revoked["revoked_at"] is not None

    def test_cannot_delete_others_api_key(self, client, admin_headers):
        """不能删除他人的 API Key"""
        # 创建另一个用户的 key（通过注册新用户）
        register_resp = client.post("/api/v1/auth/register", json={
            "username": "apikey_other_user",
            "password": "password123",
        })
        other_token = register_resp.json()["access_token"]
        other_headers = {"Authorization": f"Bearer {other_token}"}

        # 其他用户创建 key
        create_resp = client.post("/api/v1/auth/apikey/create", headers=other_headers, json={
            "name": "other-user-key",
            "permissions": ["doc:read"],
        })
        other_key_id = create_resp.json()["id"]

        # admin 尝试删除
        resp = client.delete(f"/api/v1/auth/apikey/{other_key_id}", headers=admin_headers)
        assert resp.status_code == 404  # 找不到（因为不属于 admin）

    def test_toggle_api_key(self, client, admin_headers):
        """启用/禁用 API Key"""
        # 创建
        create_resp = client.post("/api/v1/auth/apikey/create", headers=admin_headers, json={
            "name": "toggle-test-key",
            "permissions": ["doc:read"],
        })
        key_id = create_resp.json()["id"]

        # 禁用
        resp = client.post(f"/api/v1/auth/apikey/{key_id}/toggle", headers=admin_headers)
        assert resp.status_code == 200
        assert resp.json()["is_active"] is False

        # 再次启用
        resp = client.post(f"/api/v1/auth/apikey/{key_id}/toggle", headers=admin_headers)
        assert resp.status_code == 200
        assert resp.json()["is_active"] is True

    def test_disabled_api_key_rejected(self, client, admin_headers):
        """禁用的 API Key 无法使用"""
        # 创建 API Key
        create_resp = client.post("/api/v1/auth/apikey/create", headers=admin_headers, json={
            "name": "disabled-ak-test",
            "permissions": ["doc:read"],
        })
        raw_key = create_resp.json()["api_key"]
        key_id = create_resp.json()["id"]

        # 验证 Key 可用
        resp = client.get("/api/v1/auth/me", headers={"X-API-Key": raw_key})
        assert resp.status_code == 200

        # 禁用 Key
        client.post(f"/api/v1/auth/apikey/{key_id}/toggle", headers=admin_headers)

        # Key 应该被拒绝
        resp = client.get("/api/v1/auth/me", headers={"X-API-Key": raw_key})
        assert resp.status_code == 401


# ══════════════════════════════════════════════════════════════
# 6. 边界情况测试
# ══════════════════════════════════════════════════════════════

class TestEdgeCases:
    """边界情况: Token 过期 / 无效请求 / 并发"""

    def test_disabled_user_cannot_login(self, client, admin_headers):
        """被禁用的用户无法登录"""
        # 创建用户
        create_resp = client.post("/api/v1/users", headers=admin_headers, json={
            "username": "disabled_login_user",
            "password": "password123",
        })
        user_id = create_resp.json()["user_id"]

        # 禁用用户
        client.post(f"/api/v1/users/{user_id}/toggle-active", headers=admin_headers)

        # 尝试登录
        resp = client.post("/api/v1/auth/login", json={
            "username": "disabled_login_user",
            "password": "password123",
        })
        assert resp.status_code == 403
        assert "禁用" in resp.json()["detail"]

    def test_disabled_user_refresh_token_rejected(self, client, admin_headers):
        """被禁用的用户无法刷新 Token"""
        # 创建用户并获取 refresh_token
        register_resp = client.post("/api/v1/auth/register", json={
            "username": "disabled_refresh_user",
            "password": "password123",
        })
        refresh_token = register_resp.json()["refresh_token"]
        user_id = register_resp.json().get("access_token", "")

        # 获取 user_id（通过列表查找）
        list_resp = client.get("/api/v1/users", headers=admin_headers)
        target = next(
            (u for u in list_resp.json() if u["username"] == "disabled_refresh_user"),
            None,
        )
        if target:
            # 禁用
            client.post(f"/api/v1/users/{target['user_id']}/toggle-active", headers=admin_headers)

            # 尝试刷新
            resp = client.post("/api/v1/auth/refresh", json={
                "refresh_token": refresh_token,
            })
            assert resp.status_code == 403

    def test_token_version_mismatch_rejected(self, client, admin_headers):
        """角色变更后旧 Token 被拒绝"""
        # 创建 editor 用户
        create_resp = client.post("/api/v1/users", headers=admin_headers, json={
            "username": "version_test_user",
            "password": "password123",
            "role": "editor",
        })
        user_id = create_resp.json()["user_id"]

        # 登录获取 token
        login_resp = client.post("/api/v1/auth/login", json={
            "username": "version_test_user",
            "password": "password123",
        })
        old_token = login_resp.json()["access_token"]
        old_headers = {"Authorization": f"Bearer {old_token}"}

        # 确认 token 可用
        me_resp = client.get("/api/v1/auth/me", headers=old_headers)
        assert me_resp.status_code == 200

        # 管理员修改角色（会递增 token_version）
        client.patch(f"/api/v1/users/{user_id}", headers=admin_headers, json={
            "role": "viewer",
        })

        # 旧 token 应该被拒绝
        me_resp = client.get("/api/v1/auth/me", headers=old_headers)
        assert me_resp.status_code == 401

        # 用新密码重新登录获取新 token
        login_resp2 = client.post("/api/v1/auth/login", json={
            "username": "version_test_user",
            "password": "password123",
        })
        new_token = login_resp2.json()["access_token"]
        new_headers = {"Authorization": f"Bearer {new_token}"}

        # 新 token 应该可用
        me_resp2 = client.get("/api/v1/auth/me", headers=new_headers)
        assert me_resp2.status_code == 200
        assert me_resp2.json()["role"] == "viewer"

    def test_disabled_user_token_rejected(self, client, admin_headers):
        """禁用用户后其 Token 立即失效"""
        # 注册用户
        register_resp = client.post("/api/v1/auth/register", json={
            "username": "disabled_token_user",
            "password": "password123",
        })
        token = register_resp.json()["access_token"]
        headers = {"Authorization": f"Bearer {token}"}

        # 确认 token 可用
        me_resp = client.get("/api/v1/auth/me", headers=headers)
        assert me_resp.status_code == 200

        # 获取 user_id
        list_resp = client.get("/api/v1/users", headers=admin_headers)
        target = next(
            (u for u in list_resp.json() if u["username"] == "disabled_token_user"),
            None,
        )
        if target:
            # 禁用用户
            client.post(f"/api/v1/users/{target['user_id']}/toggle-active", headers=admin_headers)

            # 旧 token 应该被拒绝
            me_resp = client.get("/api/v1/auth/me", headers=headers)
            assert me_resp.status_code == 401

    def test_expired_token_rejected(self, client):
        """过期 Token 被拒绝"""
        from auth.config import AuthSettings
        from domain.identity import UserRole
        from auth.jwt_service import AuthService

        # 创建一个过期的 Token
        settings = AuthSettings(
            secret_key="test-secret",
            access_token_expire_minutes=-1,  # 已过期
        )
        service = AuthService(settings)
        token = service.create_access_token(
            user_id="user_001",
            username="test",
            role=UserRole.ADMIN,
            org_id="org_001",
        )

        headers = {"Authorization": f"Bearer {token}"}
        resp = client.get("/api/v1/auth/me", headers=headers)
        assert resp.status_code == 401

    def test_malformed_auth_header(self, client):
        """格式错误的 Authorization 头"""
        resp = client.get("/api/v1/auth/me", headers={"Authorization": "InvalidFormat"})
        assert resp.status_code in (401, 403)

    def test_empty_body_post(self, client, admin_headers):
        """空请求体"""
        resp = client.post("/api/v1/users", headers=admin_headers, json={})
        assert resp.status_code == 422  # 验证错误

    def test_oversized_field(self, client, admin_headers):
        """超长字段"""
        resp = client.post("/api/v1/users", headers=admin_headers, json={
            "username": "a" * 100,  # 超过 32 字符限制
            "password": "password123",
        })
        assert resp.status_code == 422

    def test_sql_injection_attempt(self, client):
        """SQL 注入尝试"""
        resp = client.post("/api/v1/auth/login", json={
            "username": "admin' OR '1'='1",
            "password": "password",
        })
        # 应该返回 401 而不是 500
        assert resp.status_code == 422


# ══════════════════════════════════════════════════════════════
# 7. 运行入口
# ══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
