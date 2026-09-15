"""Acceptance tests for knowledge graph subgraph route defaults."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from starlette.requests import Request

from api.routers.knowledge_graph import get_graph_subgraph
from domain.identity import Permission, UserContext, UserRole


def _user() -> UserContext:
    """Return an authenticated graph reader in a stable test tenant."""
    return UserContext(
        user_id="user-a",
        username="alice",
        role=UserRole.VIEWER,
        org_id="org-a",
        permissions=[Permission.GRAPH_READ],
    )


def _request(graph) -> Request:
    """Build a request whose tenant middleware context is intentionally empty."""
    request = Request(
        {
            "type": "http",
            "method": "GET",
            "path": "/api/v1/graph/subgraph",
            "headers": [],
            "client": ("203.0.113.10", 12345),
            "app": SimpleNamespace(state=SimpleNamespace(knowledge_graph=graph)),
        }
    )
    request.state.tenant_id = ""
    return request


@pytest.mark.asyncio
async def test_subgraph_route_uses_backend_default_limit_and_user_tenant() -> None:
    """A request without a browser limit uses the backend cap and authenticated tenant."""
    graph = SimpleNamespace(
        get_subgraph=AsyncMock(return_value={"nodes": [], "edges": [], "status": "ok"})
    )

    await get_graph_subgraph(_request(graph), user=_user())

    graph.get_subgraph.assert_awaited_once_with(
        keyword="",
        entity_type="",
        limit=200,
        tenant_id="org-a",
    )
