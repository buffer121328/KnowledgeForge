"""ATDD coverage for the identity/API Key responsibility boundaries."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from starlette.requests import Request
from starlette.responses import Response

from api.middleware.auth import JWTAuthMiddleware
from api.routers.roles import router as roles_router
from api.dependencies import get_current_user
from api.routers import users as users_module
from api.routers.users import router as users_router
from auth import apikey_service as apikey_module
from auth.apikey_service import APIKeyService, get_default_service
from auth.apikey_store import MemoryKeyStore
from auth.role_service import IdentityServiceError, RoleService
from auth.memory_accounts import ROLE_DB, USER_DB
from auth.user_service import UserService
from domain.identity import Permission, UserContext, UserRole


def _clear_default_api_keys() -> None:
    service = get_default_service()
    assert isinstance(service.store, MemoryKeyStore)
    service.store.clear()


@pytest.fixture(autouse=True)
def reset_authoritative_api_key_store(monkeypatch):
    monkeypatch.setattr(
        apikey_module,
        "_default_service",
        APIKeyService(store=MemoryKeyStore()),
    )
    _clear_default_api_keys()
    yield
    _clear_default_api_keys()


@pytest.mark.asyncio
async def test_middleware_verifies_keys_created_by_authoritative_service():
    service = get_default_service()
    api_key, raw_key = service.create_key(
        name="boundary",
        user_id="user_001",
        org_id="org_001",
        role=UserRole.ADMIN,
        scopes=[Permission.DOC_READ],
        actor_permissions=list(Permission),
        actor_username="admin",
    )

    app = FastAPI()
    middleware = JWTAuthMiddleware(app, user_service=UserService(users=USER_DB))
    scope = {
        "type": "http",
        "method": "GET",
        "path": "/api/docs",
        "headers": [],
        "app": app,
        "scheme": "http",
        "server": ("testserver", 80),
    }
    request = Request(scope)
    seen = {}

    async def call_next(current_request: Request) -> Response:
        seen["user"] = current_request.state.user
        return Response(status_code=204)

    response = await middleware._handle_api_key(request, call_next, raw_key)

    assert middleware.api_key_service is service
    assert response.status_code == 204
    assert seen["user"].api_key_id == api_key.id
    assert seen["user"].user_id == "user_001"
    assert api_key.last_used_at is not None


@pytest.mark.asyncio
@pytest.mark.parametrize("mutator", ["disable", "expire"])
async def test_middleware_rejects_disabled_or_expired_authoritative_keys(mutator: str):
    service = get_default_service()
    api_key, raw_key = service.create_key(
        name="boundary",
        user_id="user_001",
        org_id="org_001",
        role=UserRole.ADMIN,
        actor_permissions=list(Permission),
        actor_username="admin",
    )
    if mutator == "disable":
        api_key.is_active = False
    else:
        api_key.expires_at = (datetime.now(timezone.utc) - timedelta(days=1)).isoformat()
    service.store.save(api_key)

    app = FastAPI()
    middleware = JWTAuthMiddleware(app)
    request = Request(
        {
            "type": "http",
            "method": "GET",
            "path": "/api/docs",
            "headers": [],
            "app": app,
            "scheme": "http",
            "server": ("testserver", 80),
        }
    )

    response = await middleware._handle_api_key(request, lambda _: Response(), raw_key)

    assert response.status_code == 401
    assert not hasattr(request.state, "user")


def test_static_identity_routes_precede_dynamic_identifier_routes():
    user_paths = [route.path for route in users_router.routes]
    role_paths = [route.path for route in roles_router.routes]

    assert user_paths.index("/users/batch/role") < user_paths.index("/users/{user_id}")
    assert user_paths.index("/users/me/password") < user_paths.index("/users/{user_id}")
    assert role_paths.index("/roles/meta/permissions") < role_paths.index("/roles/{role_name}")


def test_identity_services_preserve_self_and_builtin_role_protections():
    user_service = UserService(USER_DB)
    role_service = RoleService(ROLE_DB, USER_DB)
    admin = USER_DB["admin"]

    with pytest.raises(IdentityServiceError) as delete_error:
        user_service.delete_user(admin["user_id"], actor_user_id=admin["user_id"])
    assert delete_error.value.status_code == 400
    assert delete_error.value.detail == "不能删除自己的账号"

    with pytest.raises(IdentityServiceError) as role_error:
        role_service.delete_role("admin")
    assert role_error.value.status_code == 400
    assert role_error.value.detail == "内置角色不可删除"


def test_user_management_api_derives_department_responsibility_from_role(monkeypatch):
    """User management derives manager responsibility from the selected role."""
    service = UserService(users={})
    monkeypatch.setattr(users_module, "user_service", service)
    app = FastAPI()
    app.include_router(users_router)
    app.dependency_overrides[get_current_user] = lambda: UserContext(
        user_id="admin-a",
        username="admin-a",
        role=UserRole.ORGANIZATION_ADMIN,
        org_id="org-a",
        permissions=list(Permission),
    )
    client = TestClient(app)

    created = client.post(
        "/users",
        json={
            "username": "finance_manager",
            "password": "password-123",
            "display_name": "财务负责人",
            "email": "finance-manager@example.com",
            "role": "admin",
            "department_id": "finance",
            "is_department_manager": False,
        },
    )

    assert created.status_code == 201
    assert created.json()["department_id"] == "finance"
    assert created.json()["is_department_manager"] is True

    promoted = client.patch(
        f"/users/{created.json()['user_id']}",
        json={
            "role": "organization_admin",
            "department_id": None,
            "is_department_manager": False,
        },
    )

    assert promoted.status_code == 200
    assert promoted.json()["role"] == "organization_admin"
    assert promoted.json()["department_id"] is None
    assert promoted.json()["is_department_manager"] is False
