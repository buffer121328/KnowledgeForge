"""Acceptance coverage for focused FastAPI application assembly."""

from __future__ import annotations

import ast
from pathlib import Path

from fastapi.testclient import TestClient


BACKEND_ROOT = Path(__file__).resolve().parents[3]


def _business_routes(app, prefix: str) -> dict[str, frozenset[str]]:
    paths = app.openapi()["paths"]
    return {
        path.removeprefix(prefix): frozenset(operations)
        for path, operations in paths.items()
        if path.startswith(f"{prefix}/")
        and (prefix != "/api" or not path.startswith("/api/v1/"))
    }


def _assembled_app(monkeypatch):
    from shared.config import settings

    monkeypatch.setattr(settings, "deepseek_api_key", "test-key")
    monkeypatch.setattr(settings, "dashscope_api_key", "test-key")
    from api.app import app

    return app


def test_main_reexports_the_assembled_application(monkeypatch):
    assembled_app = _assembled_app(monkeypatch)
    from api.main import app

    assert app is assembled_app


def test_business_routes_are_available_only_under_versioned_prefix(monkeypatch):
    app = _assembled_app(monkeypatch)

    versioned = _business_routes(app, "/api/v1")
    legacy = _business_routes(app, "/api")

    assert {"/docs", "/qa/ask"} <= set(versioned)
    assert "/docs" not in legacy
    assert "/qa/ask" not in legacy
    assert versioned["/docs"] == frozenset({"get"})
    assert versioned["/qa/ask"] == frozenset({"post"})


def test_protected_extracted_route_does_not_bypass_authentication(monkeypatch):
    app = _assembled_app(monkeypatch)

    client = TestClient(app)
    assert client.get("/api/docs").status_code == 404
    assert client.get("/api/v1/docs").status_code in {401, 403}


def test_application_assembly_has_no_legacy_fact_runtime_imports():
    source = (BACKEND_ROOT / "api" / "app.py").read_text(encoding="utf-8")

    for obsolete in (
        "SQLiteDocumentCatalogRepository",
        "get_security_state_backend",
        "security_state_backend",
        "document_catalog_backend",
        "qa_history_backend",
        "webhook_fact_backend",
        "task_history_backend",
        "Legacy API",
    ):
        assert obsolete not in source


def test_extracted_routers_do_not_import_application_assembly():
    for module_name in ("documents_ingest", "documents_read", "qa", "knowledge_graph", "admin"):
        module_path = BACKEND_ROOT / "api" / "routers" / f"{module_name}.py"
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

        assert "api.app" not in imported_modules
        assert "api.main" not in imported_modules


def test_effective_middleware_order_is_preserved(monkeypatch):
    app = _assembled_app(monkeypatch)

    middleware_names = [middleware.cls.__name__ for middleware in app.user_middleware]

    assert middleware_names.index("JWTAuthMiddleware") < middleware_names.index(
        "PermissionMiddleware"
    )
    assert middleware_names.index("JWTAuthMiddleware") < middleware_names.index(
        "RequestTrendMiddleware"
    )
    assert middleware_names.index("RequestTrendMiddleware") < middleware_names.index(
        "PermissionMiddleware"
    )
    assert middleware_names.index("PermissionMiddleware") < middleware_names.index("TenantMiddleware")
