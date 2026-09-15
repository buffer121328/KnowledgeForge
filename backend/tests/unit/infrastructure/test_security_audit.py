"""ATDD coverage for persistent, tenant-scoped security auditing."""

from __future__ import annotations

import csv
import io
import json
import stat
from dataclasses import asdict
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from api.dependencies.auth import get_current_user
from api.middleware.permission import PermissionMiddleware
from api.middleware.permission_registry import init_route_permissions
from api.routers import audit as audit_router
from domain.identity import Permission, UserContext, UserRole
from infrastructure.audit.log import (
    AuditAction,
    AuditIntegrityError,
    AuditResult,
    AuditService,
    AuditWriteError,
)
from infrastructure.audit.stores import FileAuditStore, MemoryAuditStore


def _admin(org_id: str = "org-a") -> UserContext:
    return UserContext(
        user_id=f"admin-{org_id}",
        username=f"admin-{org_id}",
        role=UserRole.ADMIN,
        org_id=org_id,
        permissions=[Permission.ADMIN_AUDIT],
    )


def _log(
    service: AuditService,
    *,
    org_id: str,
    user_id: str,
    action: AuditAction = AuditAction.LOGIN,
):
    return service.log(
        user_id=user_id,
        username=user_id,
        org_id=org_id,
        action=action,
    )


def test_file_store_persists_tenant_filtered_offset_queries_across_recreation(
    tmp_path: Path,
) -> None:
    path = tmp_path / "audit.jsonl"
    service = AuditService(FileAuditStore(str(path), max_bytes=1_000_000, backup_count=2))
    _log(service, org_id="org-a", user_id="a-1")
    _log(service, org_id="org-b", user_id="b-1")
    _log(service, org_id="org-a", user_id="a-2")

    recreated = AuditService(
        FileAuditStore(str(path), max_bytes=1_000_000, backup_count=2)
    )
    result = recreated.query(org_id="org-a", limit=1, offset=1)

    assert [entry.user_id for entry in result] == ["a-1"]
    assert stat.S_IMODE(path.stat().st_mode) == 0o600


def test_file_store_rejects_tampered_records(tmp_path: Path) -> None:
    path = tmp_path / "audit.jsonl"
    service = AuditService(FileAuditStore(str(path)))
    _log(service, org_id="org-a", user_id="a-1")

    record = json.loads(path.read_text(encoding="utf-8"))
    record["user_id"] = "tampered"
    path.write_text(json.dumps(record) + "\n", encoding="utf-8")

    with pytest.raises(AuditIntegrityError):
        service.query(org_id="org-a")


def test_file_store_rotates_and_verifies_retained_hash_chain(tmp_path: Path) -> None:
    path = tmp_path / "audit.jsonl"
    service = AuditService(FileAuditStore(str(path), max_bytes=700, backup_count=3))

    for index in range(8):
        service.log(
            user_id=f"user-{index}",
            username=f"user-{index}",
            org_id="org-a",
            action=AuditAction.LOGIN,
            metadata={"padding": "x" * 80},
            required=True,
        )

    assert (tmp_path / "audit.jsonl.1").exists()
    result = service.query(org_id="org-a", limit=100)
    assert result[0].user_id == "user-7"
    assert len(result) >= 2


class _FailingStore:
    def append(self, log) -> None:
        raise AuditWriteError("disk path and secret must not leak")

    def query(self, **kwargs):
        return []


async def _run_permission_middleware(
    middleware: PermissionMiddleware,
    user: UserContext,
) -> tuple[bool, list[dict]]:
    called = False
    sent: list[dict] = []

    async def receive():
        return {"type": "http.request", "body": b"", "more_body": False}

    async def send(message):
        sent.append(message)

    async def downstream(scope, receive, send):
        nonlocal called
        called = True
        await send({"type": "http.response.start", "status": 204, "headers": []})
        await send({"type": "http.response.body", "body": b""})

    middleware.app = downstream
    scope = {
        "type": "http",
        "asgi": {"version": "3.0"},
        "http_version": "1.1",
        "method": "GET",
        "scheme": "http",
        "path": "/api/users",
        "raw_path": b"/api/users",
        "query_string": b"",
        "headers": [],
        "client": ("127.0.0.1", 12345),
        "server": ("testserver", 80),
        "state": {"user": user},
    }
    await middleware(scope, receive, send)
    return called, sent


@pytest.mark.asyncio
async def test_permission_decisions_are_persistent_tenant_bound_and_minimal(
    tmp_path: Path,
) -> None:
    init_route_permissions()
    path = tmp_path / "permission-audit.jsonl"
    service = AuditService(FileAuditStore(str(path)))
    middleware = PermissionMiddleware(lambda *_: None, audit_service=service)
    user = UserContext(
        user_id="reader-a",
        username="reader",
        role=UserRole.API_USER,
        org_id="org-a",
        permissions=[Permission.ADMIN_USER],
    )

    called, _ = await _run_permission_middleware(middleware, user)

    assert called is True
    recreated = AuditService(FileAuditStore(str(path)))
    records = recreated.query(
        org_id="org-a", action=AuditAction.PERMISSION_CHECK.value
    )
    assert len(records) == 1
    record = records[0]
    assert record.result == AuditResult.SUCCESS.value
    assert record.resource == "GET /api/users"
    assert record.metadata == {
        "required_permissions": ["admin:user"],
        "user_permissions": ["admin:user"],
        "reason_code": "permission_granted",
    }
    assert "authorization" not in json.dumps(asdict(record)).lower()


@pytest.mark.asyncio
async def test_allowed_request_fails_closed_when_permission_audit_cannot_persist() -> None:
    init_route_permissions()
    middleware = PermissionMiddleware(
        lambda *_: None,
        audit_service=AuditService(_FailingStore()),
    )
    user = UserContext(
        user_id="reader-a",
        username="reader",
        role=UserRole.API_USER,
        org_id="org-a",
        permissions=[Permission.ADMIN_USER],
    )

    called, sent = await _run_permission_middleware(middleware, user)

    assert called is False
    start = next(message for message in sent if message["type"] == "http.response.start")
    body = next(message for message in sent if message["type"] == "http.response.body")
    assert start["status"] == 503
    assert b"audit_persistence_unavailable" in body["body"]
    assert b"disk path" not in body["body"]


@pytest.mark.asyncio
async def test_denied_permission_decision_is_persisted_without_invoking_endpoint(
    tmp_path: Path,
) -> None:
    init_route_permissions()
    service = AuditService(FileAuditStore(str(tmp_path / "denied.jsonl")))
    middleware = PermissionMiddleware(lambda *_: None, audit_service=service)
    user = UserContext(
        user_id="viewer-a",
        username="viewer",
        role=UserRole.VIEWER,
        org_id="org-a",
        permissions=[Permission.DOC_READ],
    )

    called, sent = await _run_permission_middleware(middleware, user)

    assert called is False
    start = next(message for message in sent if message["type"] == "http.response.start")
    assert start["status"] == 403
    records = service.query(
        org_id="org-a", action=AuditAction.PERMISSION_CHECK.value
    )
    assert len(records) == 1
    assert records[0].result == AuditResult.DENIED.value
    assert records[0].metadata["reason_code"] == "permission_denied"


def _audit_client(
    monkeypatch: pytest.MonkeyPatch,
    service: AuditService,
    *,
    org_id: str = "org-a",
) -> TestClient:
    app = FastAPI()
    app.include_router(audit_router.router, prefix="/api")
    app.dependency_overrides[get_current_user] = lambda: _admin(org_id)
    monkeypatch.setattr(audit_router, "get_audit_service", lambda: service)
    return TestClient(app)


def test_audit_query_is_tenant_scoped_and_offset_paginated(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = AuditService(MemoryAuditStore())
    _log(service, org_id="org-a", user_id="a-1")
    _log(service, org_id="org-b", user_id="b-1")
    _log(service, org_id="org-a", user_id="a-2")
    client = _audit_client(monkeypatch, service)

    response = client.get("/api/audit/logs?org_id=org-b&limit=1&offset=1")

    assert response.status_code == 200
    assert [entry["user_id"] for entry in response.json()] == ["a-1"]


@pytest.mark.parametrize(
    "query",
    [
        "start=not-a-time",
        "start=2026-07-27T00:00:00Z&end=2026-07-26T00:00:00Z",
    ],
)
def test_audit_query_rejects_invalid_time_ranges_with_stable_error(
    monkeypatch: pytest.MonkeyPatch,
    query: str,
) -> None:
    client = _audit_client(monkeypatch, AuditService(MemoryAuditStore()))

    response = client.get(f"/api/audit/logs?{query}")

    assert response.status_code == 400
    detail = response.json()["detail"]
    assert detail["code"] == "invalid_audit_time_range"
    assert detail["request_id"]
    assert response.headers["X-Request-ID"] == detail["request_id"]


class _IntegrityFailureStore(MemoryAuditStore):
    def query(self, **kwargs):
        raise AuditIntegrityError("tampered record detail")


def test_audit_query_returns_stable_integrity_error_without_partial_records(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = _audit_client(
        monkeypatch, AuditService(_IntegrityFailureStore())
    )

    response = client.get("/api/audit/logs")

    assert response.status_code == 503
    detail = response.json()["detail"]
    assert detail["code"] == "audit_integrity_error"
    assert "tampered" not in response.text
    assert response.headers["X-Request-ID"] == detail["request_id"]


def test_audit_export_is_tenant_scoped_minimal_and_audited(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = AuditService(MemoryAuditStore())
    exported = _log(service, org_id="org-a", user_id="a-1")
    exported.username = "=spreadsheet_formula()"
    service.log(
        user_id="b-1",
        username="b-1",
        org_id="org-b",
        action=AuditAction.LOGIN,
        metadata={"secret": "must-not-export"},
    )
    client = _audit_client(monkeypatch, service)

    response = client.get("/api/audit/logs/export?limit=20&offset=0")

    assert response.status_code == 200
    assert response.headers["X-Request-ID"]
    rows = list(csv.DictReader(io.StringIO(response.text)))
    assert [row["user_id"] for row in rows] == ["a-1"]
    assert rows[0]["username"] == "'=spreadsheet_formula()"
    assert "metadata" not in rows[0]
    assert "org-b" not in response.text
    export_events = service.query(
        org_id="org-a", action=AuditAction.AUDIT_EXPORT.value
    )
    assert len(export_events) == 1
    assert export_events[0].metadata == {
        "request_id": response.headers["X-Request-ID"],
        "filters": {
            "user_id": None,
            "action": None,
            "start": None,
            "end": None,
            "limit": 20,
            "offset": 0,
        },
        "record_count": 1,
    }
