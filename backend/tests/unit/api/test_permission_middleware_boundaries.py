"""Acceptance coverage for permission middleware responsibility boundaries."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from domain.identity import Permission, UserContext, UserRole


BACKEND_ROOT = Path(__file__).resolve().parents[3]


@pytest.fixture
def permission_state():
    from api.middleware import permission_audit, permission_registry

    original_rules = dict(permission_registry.ROUTE_PERMISSIONS)
    original_logs = list(permission_audit._audit_logs)
    original_capacity = permission_audit._MAX_AUDIT_LOGS
    try:
        yield
    finally:
        permission_registry.ROUTE_PERMISSIONS.clear()
        permission_registry.ROUTE_PERMISSIONS.update(original_rules)
        permission_audit._audit_logs[:] = original_logs
        permission_audit._MAX_AUDIT_LOGS = original_capacity


def _user(role: UserRole, permissions: list[Permission] | None = None) -> UserContext:
    return UserContext(
        user_id=f"user-{role.value}",
        username=role.value,
        role=role,
        org_id="org-test",
        permissions=permissions or [],
    )


def test_permission_middleware_does_not_reexport_focused_module_state():
    from api.middleware import permission

    for obsolete_export in (
        "ROUTE_PERMISSIONS",
        "PermissionAuditLog",
        "_audit_logs",
        "get_audit_logs",
        "init_route_permissions",
        "register_permission",
    ):
        assert not hasattr(permission, obsolete_export)


def test_focused_modules_do_not_import_permission_facade():
    for module_name in ("permission_registry", "permission_resolver", "permission_audit"):
        module_path = BACKEND_ROOT / "api" / "middleware" / f"{module_name}.py"
        tree = ast.parse(module_path.read_text(encoding="utf-8"))
        imported_modules = {
            node.module
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom) and node.module
        }
        imported_modules.update(
            alias.name
            for node in ast.walk(tree)
            if isinstance(node, ast.Import)
            for alias in node.names
        )
        assert "api.middleware.permission" not in imported_modules


def test_registry_registers_only_v1_parameterized_rules(permission_state):
    from api.middleware import permission_registry

    permission_registry.ROUTE_PERMISSIONS.clear()
    permission_registry.init_route_permissions()

    assert permission_registry._match_route("GET", "/api/docs") is None
    assert permission_registry._match_route("GET", "/api/v1/docs") == [Permission.DOC_READ]
    assert permission_registry._match_route("GET", "/api/v1/docs/departments") == [Permission.DOC_READ]
    assert permission_registry._match_route("GET", "/api/v1/docs/doc-123/file") == [Permission.DOC_READ]
    assert permission_registry._match_route("GET", "/api/v1/users/user-123") == [
        Permission.ADMIN_USER
    ]


def test_resolver_preserves_role_and_user_permission_behavior(permission_state):
    from api.middleware import permission, permission_registry, permission_resolver

    permission_registry.ROUTE_PERMISSIONS.clear()
    permission_registry.init_route_permissions()

    editor = _user(UserRole.EDITOR)
    viewer = _user(UserRole.VIEWER)
    explicit_admin = _user(UserRole.VIEWER, [Permission.ADMIN_USER])

    assert not permission_resolver.permission_resolver.has_permission(
        editor, [Permission.DOC_WRITE]
    )
    assert permission_resolver.permission_resolver.has_permission(editor, [Permission.DOC_READ])
    assert not permission_resolver.permission_resolver.has_permission(
        viewer, [Permission.ADMIN_USER]
    )
    assert permission_resolver.permission_resolver.resolve_user_permissions(
        explicit_admin
    ) is explicit_admin.permissions
    assert not permission.check_permission(editor, "POST", "/api/v1/ingest/upload").granted


def test_audit_retention_and_filtering_are_preserved(monkeypatch, permission_state):
    from api.middleware import permission_audit

    permission_audit._audit_logs.clear()
    monkeypatch.setattr(permission_audit, "_MAX_AUDIT_LOGS", 2)

    permission_audit._record_audit(
        "user-1", "one", "GET", "/api/v1/docs", ["doc:read"], ["doc:read"], True
    )
    permission_audit._record_audit(
        "user-2", "two", "GET", "/api/v1/users", ["admin:user"], [], False, "权限不足"
    )
    permission_audit._record_audit(
        "user-1", "one", "POST", "/api/v1/qa/ask", ["qa:query"], ["qa:query"], True
    )

    logs = permission_audit.get_audit_logs(limit=10)
    assert len(logs) == 2
    assert [log.user_id for log in logs] == ["user-2", "user-1"]
    assert logs[0].granted is False
    assert logs[0].reason == "权限不足"
    assert permission_audit.get_audit_logs(user_id="user-1") == [logs[-1]]
