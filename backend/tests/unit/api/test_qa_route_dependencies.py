"""Mounted-route coverage for explicit QA HTTP dependencies."""

from __future__ import annotations

from types import SimpleNamespace

from api.dependencies import get_current_user
from api.dependencies.qa import QARouteDependencies, get_qa_route_dependencies
from api.routers.qa import qa_router
from domain.identity import Permission, UserContext, UserRole
from fastapi import FastAPI
from fastapi.testclient import TestClient


def test_conversation_route_uses_overridden_tenant_scoped_history() -> None:
    """Mounted requests use the overrideable bundle and authenticated scope."""

    calls: list[dict] = []

    class History:
        def list_conversations(self, **kwargs):
            calls.append(kwargs)
            return SimpleNamespace(items=[], next_cursor=None)

    dependencies = QARouteDependencies(history=History())
    app = FastAPI()
    app.include_router(qa_router)
    app.dependency_overrides[get_current_user] = lambda: UserContext(
        user_id="user-a",
        username="alice",
        role=UserRole.VIEWER,
        org_id="tenant-a",
        permissions=[Permission.QA_HISTORY],
    )
    app.dependency_overrides[get_qa_route_dependencies] = lambda: dependencies

    response = TestClient(app).get("/qa/conversations?limit=10")

    assert response.status_code == 200
    assert response.json() == {"items": [], "next_cursor": None}
    assert calls == [
        {
            "tenant_id": "tenant-a",
            "user_id": "user-a",
            "cursor": None,
            "limit": 10,
        }
    ]
