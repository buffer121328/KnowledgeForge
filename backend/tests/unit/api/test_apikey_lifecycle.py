"""ATDD coverage for bounded, least-privilege API Key lifecycle behavior."""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from starlette.requests import Request
from starlette.responses import Response

from api.dependencies.auth import get_current_user
from api.middleware.auth import JWTAuthMiddleware
from api.middleware.permission_resolver import PermissionResolver
from api.routers import apikeys as apikey_router
from auth.apikey_service import (
    APIKeyLifecycleError,
    APIKeyLifecycleEvent,
    APIKeyPolicy,
    APIKeyService,
)
from auth.apikey_store import MemoryKeyStore
from domain.identity import Permission, UserContext, UserRole


NOW = datetime(2026, 7, 26, 12, 0, tzinfo=timezone.utc)


@pytest.fixture
def events() -> list[APIKeyLifecycleEvent]:
    return []


@pytest.fixture
def policy() -> APIKeyPolicy:
    return APIKeyPolicy(
        min_expiry_days=2,
        default_expiry_days=30,
        max_expiry_days=90,
        stale_warning_days=14,
        expiry_warning_days=7,
    )


@pytest.fixture
def service(
    policy: APIKeyPolicy,
    events: list[APIKeyLifecycleEvent],
) -> APIKeyService:
    return APIKeyService(
        store=MemoryKeyStore(),
        policy=policy,
        event_sink=events.append,
        clock=lambda: NOW,
    )


def _create(
    service: APIKeyService,
    *,
    scopes: list[Permission] | list[str] | None = None,
    actor_permissions: list[Permission] | None = None,
    expires_days: int | None = None,
    user_id: str = "user-a",
    org_id: str = "org-a",
):
    return service.create_key(
        name="automation",
        user_id=user_id,
        org_id=org_id,
        role=UserRole.ADMIN,
        scopes=scopes or [],
        actor_permissions=actor_permissions
        if actor_permissions is not None
        else list(Permission),
        actor_username="admin",
        expires_days=expires_days,
    )


def test_default_expiry_is_mandatory_and_explicit_expiry_is_bounded(
    service: APIKeyService,
) -> None:
    api_key, _ = _create(service)
    assert api_key.expires_at == (NOW + timedelta(days=30)).isoformat()

    for invalid_days in (0, -1, 1, 91):
        with pytest.raises(APIKeyLifecycleError) as error:
            _create(service, expires_days=invalid_days)
        assert error.value.code == "invalid_api_key_expiry"

    assert len(service.list_keys("user-a", "org-a")) == 1


@pytest.mark.parametrize("expires_at", [None, "not-an-iso-time"])
def test_missing_or_malformed_legacy_expiry_fails_closed_without_last_use(
    service: APIKeyService,
    expires_at: str | None,
) -> None:
    api_key, raw_key = _create(service)
    api_key.expires_at = expires_at
    service.store.save(api_key)

    assert service.verify(raw_key) is None
    assert api_key.last_used_at is None
    assert service.status(api_key) == "invalid_expiry"


def test_unknown_or_excessive_scope_rejects_entire_creation(
    service: APIKeyService,
) -> None:
    with pytest.raises(APIKeyLifecycleError) as unknown:
        _create(
            service,
            scopes=["doc:read", "unknown:scope"],
            actor_permissions=[Permission.DOC_READ],
        )
    assert unknown.value.code == "invalid_api_key_scope"

    with pytest.raises(APIKeyLifecycleError) as excessive:
        _create(
            service,
            scopes=[Permission.DOC_WRITE],
            actor_permissions=[Permission.DOC_READ],
        )
    assert excessive.value.code == "api_key_scope_not_allowed"
    assert service.list_keys("user-a", "org-a") == []


def test_scope_update_only_reduces_and_preserves_state_on_expansion(
    service: APIKeyService,
    events: list[APIKeyLifecycleEvent],
) -> None:
    api_key, _ = _create(
        service,
        scopes=[Permission.DOC_READ, Permission.QA_QUERY],
    )

    reduced = service.update_scopes(
        api_key.id,
        "user-a",
        "org-a",
        [Permission.DOC_READ],
        actor_permissions=list(Permission),
        actor_username="admin",
    )
    assert reduced is not None
    assert reduced.scopes == [Permission.DOC_READ]
    assert events[-1].action == "auth.apikey_scope_update"

    with pytest.raises(APIKeyLifecycleError) as expansion:
        service.update_scopes(
            api_key.id,
            "user-a",
            "org-a",
            [Permission.DOC_READ, Permission.QA_QUERY],
            actor_permissions=list(Permission),
        )
    assert expansion.value.code == "api_key_scope_expansion_forbidden"
    assert service.get_owned_key(api_key.id, "user-a", "org-a").scopes == [
        Permission.DOC_READ
    ]


def test_revoke_is_soft_irreversible_and_owner_org_bound(
    service: APIKeyService,
) -> None:
    api_key, raw_key = _create(service, scopes=[Permission.DOC_READ])

    revoked = service.revoke(api_key.id, "user-a", "org-a", actor_username="admin")

    assert revoked is not None
    assert revoked.revoked_at == NOW.isoformat()
    assert revoked.revoked_by == "user-a"
    assert service.verify(raw_key) is None
    assert service.list_keys("user-a", "org-a")[0].id == api_key.id
    assert service.status(revoked) == "revoked"
    assert service.get_owned_key(api_key.id, "user-a", "org-b") is None

    with pytest.raises(APIKeyLifecycleError) as toggle_error:
        service.toggle(api_key.id, "user-a", "org-a")
    assert toggle_error.value.code == "api_key_revoked"

    with pytest.raises(APIKeyLifecycleError) as scope_error:
        service.update_scopes(
            api_key.id,
            "user-a",
            "org-a",
            [],
            actor_permissions=list(Permission),
        )
    assert scope_error.value.code == "api_key_revoked"


def test_rotation_rolls_back_when_required_audit_fails(
    policy: APIKeyPolicy,
) -> None:
    def sink(event: APIKeyLifecycleEvent) -> None:
        if event.action == "auth.apikey_rotate":
            raise RuntimeError("audit backend path must not leak")

    service = APIKeyService(
        store=MemoryKeyStore(),
        policy=policy,
        event_sink=sink,
        clock=lambda: NOW,
    )
    api_key, old_raw = _create(service, scopes=[Permission.DOC_READ])
    before_hash = api_key.key_hash

    with pytest.raises(APIKeyLifecycleError) as error:
        service.rotate(
            api_key.id,
            "user-a",
            "org-a",
            expires_days=30,
            actor_username="admin",
        )

    assert error.value.code == "api_key_audit_unavailable"
    restored = service.get_owned_key(api_key.id, "user-a", "org-a")
    assert restored.key_hash == before_hash
    assert restored.rotation_count == 0
    assert service.verify(old_raw) is not None
    assert "audit backend" not in str(error.value)


def test_lifecycle_warnings_are_computed_without_secret_material(
    service: APIKeyService,
) -> None:
    api_key, _ = _create(service, expires_days=30)
    api_key.created_at = (NOW - timedelta(days=20)).isoformat()
    assert service.warnings(api_key) == ["never_used"]

    api_key.last_used_at = (NOW - timedelta(days=20)).isoformat()
    api_key.expires_at = (NOW + timedelta(days=5)).isoformat()
    assert service.warnings(api_key) == ["stale", "expires_soon"]


def _admin(org_id: str = "org-a") -> UserContext:
    return UserContext(
        user_id="user-a",
        username="admin",
        role=UserRole.ADMIN,
        org_id=org_id,
        permissions=list(Permission),
    )


def _client(
    monkeypatch: pytest.MonkeyPatch,
    service: APIKeyService,
    *,
    user: UserContext | None = None,
) -> TestClient:
    app = FastAPI()
    app.include_router(apikey_router.router, prefix="/api/auth")
    app.dependency_overrides[get_current_user] = lambda: user or _admin()
    monkeypatch.setattr(apikey_router, "_service", lambda: service)
    return TestClient(app)


def test_http_create_list_rotate_and_safe_lifecycle_audit(
    monkeypatch: pytest.MonkeyPatch,
    service: APIKeyService,
    events: list[APIKeyLifecycleEvent],
) -> None:
    client = _client(monkeypatch, service)

    created_response = client.post(
        "/api/auth/apikey/create",
        json={"name": "ci", "permissions": ["doc:read"]},
    )
    assert created_response.status_code == 200
    created = created_response.json()
    assert created["api_key"].startswith("ak_")
    assert created["status"] == "active"
    assert created["last_used_at"] is None
    assert "key_hash" not in created

    listed = client.get("/api/auth/apikey/list").json()
    assert listed[0]["api_key"] is None
    assert listed[0]["rotation_count"] == 0
    assert "key_hash" not in listed[0]

    rotated_response = client.post(
        f"/api/auth/apikey/{created['id']}/rotate",
        json={"expires_days": 45},
    )
    assert rotated_response.status_code == 200
    rotated = rotated_response.json()
    assert rotated["api_key"] != created["api_key"]
    assert rotated["rotation_count"] == 1
    assert service.verify(created["api_key"]) is None
    assert service.verify(rotated["api_key"]) is not None

    serialized_events = json.dumps(
        [event.__dict__ for event in events],
        default=str,
    ).lower()
    assert "auth.apikey_create" in serialized_events
    assert "auth.apikey_rotate" in serialized_events
    for forbidden in ("key_hash", "key_prefix", created["api_key"].lower()):
        assert forbidden not in serialized_events


@pytest.mark.parametrize(
    "payload",
    [
        {"name": "bad", "permissions": [], "expires_days": 0},
        {"name": "bad", "permissions": ["invalid:scope"]},
    ],
)
def test_http_creation_returns_stable_policy_errors(
    monkeypatch: pytest.MonkeyPatch,
    service: APIKeyService,
    payload: dict,
) -> None:
    response = _client(monkeypatch, service).post(
        "/api/auth/apikey/create",
        json=payload,
    )

    assert response.status_code == 422
    assert response.json()["detail"]


def test_http_owner_and_org_mismatch_is_concealed_and_scope_expansion_conflicts(
    monkeypatch: pytest.MonkeyPatch,
    service: APIKeyService,
) -> None:
    api_key, _ = _create(service, scopes=[Permission.DOC_READ])
    other_org = _admin("org-b")
    concealed = _client(monkeypatch, service, user=other_org).post(
        f"/api/auth/apikey/{api_key.id}/rotate",
        json={},
    )
    assert concealed.status_code == 404

    client = _client(monkeypatch, service)
    expansion = client.patch(
        f"/api/auth/apikey/{api_key.id}/scopes",
        json={"permissions": ["doc:read", "qa:query"]},
    )
    assert expansion.status_code == 409
    assert expansion.json()["detail"]["code"] == "api_key_scope_expansion_forbidden"


def test_required_create_audit_failure_removes_staged_key(
    monkeypatch: pytest.MonkeyPatch,
    policy: APIKeyPolicy,
) -> None:
    def failing_sink(event: APIKeyLifecycleEvent) -> None:
        raise RuntimeError("raw audit failure")

    service = APIKeyService(
        store=MemoryKeyStore(),
        policy=policy,
        event_sink=failing_sink,
        clock=lambda: NOW,
    )
    response = _client(monkeypatch, service).post(
        "/api/auth/apikey/create",
        json={"name": "ci", "permissions": ["doc:read"]},
    )

    assert response.status_code == 503
    assert response.json()["detail"]["code"] == "api_key_audit_unavailable"
    assert "raw audit" not in response.text
    assert service.list_keys("user-a", "org-a") == []


class _UserService:
    def __init__(self, record: dict | None):
        self.record = record

    def find_by_id(self, user_id: str, *, raise_on_missing: bool = True):
        return self.record


def _request() -> Request:
    return Request(
        {
            "type": "http",
            "method": "GET",
            "path": "/api/users",
            "headers": [],
            "app": FastAPI(),
            "scheme": "http",
            "server": ("testserver", 80),
        }
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "owner",
    [
        None,
        {"user_id": "user-a", "org_id": "org-a", "is_active": False},
        {"user_id": "user-a", "org_id": "org-b", "is_active": True},
    ],
)
async def test_api_key_auth_rejects_invalid_owner_before_last_use(
    service: APIKeyService,
    owner: dict | None,
) -> None:
    api_key, raw_key = _create(service, scopes=[Permission.DOC_READ])
    middleware = JWTAuthMiddleware(
        FastAPI(),
        api_key_service=service,
        user_service=_UserService(owner),
    )

    response = await middleware._handle_api_key(
        _request(),
        lambda _: Response(status_code=204),
        raw_key,
    )

    assert response.status_code == 401
    assert api_key.last_used_at is None


@pytest.mark.asyncio
async def test_empty_scope_api_key_never_inherits_creator_role(
    service: APIKeyService,
) -> None:
    api_key, raw_key = _create(service, scopes=[])
    owner = {"user_id": "user-a", "org_id": "org-a", "is_active": True}
    middleware = JWTAuthMiddleware(
        FastAPI(),
        api_key_service=service,
        user_service=_UserService(owner),
    )
    seen: dict = {}

    async def call_next(request: Request) -> Response:
        seen["user"] = request.state.user
        return Response(status_code=204)

    response = await middleware._handle_api_key(_request(), call_next, raw_key)

    assert response.status_code == 204
    context = seen["user"]
    assert context.api_key_id == api_key.id
    assert context.role == UserRole.API_USER
    assert PermissionResolver().resolve_user_permissions(context) == []
    assert api_key.last_used_at is not None
