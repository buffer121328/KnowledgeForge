"""Regression coverage for catalog-backed dashboard graph statistics."""

from __future__ import annotations

from types import SimpleNamespace

from starlette.responses import Response
from unittest.mock import AsyncMock, Mock

import pytest

from api.routers.admin import get_stats
from domain.identity import Permission, UserContext, UserRole
from infrastructure.graph.neo4j_graph import KnowledgeGraphService


def _user() -> UserContext:
    """Return an administrator in one tenant."""

    return UserContext(
        user_id="admin-1",
        username="admin",
        role=UserRole.ADMIN,
        org_id="tenant-a",
        permissions=[Permission.ADMIN_MANAGE],
    )


def _request(*, catalog, graph) -> SimpleNamespace:
    """Build the router's minimal app-state dependency container."""

    return SimpleNamespace(
        app=SimpleNamespace(
            state=SimpleNamespace(
                document_catalog=catalog,
                knowledge_graph=graph,
                vector_store=SimpleNamespace(
                    get_stats=AsyncMock(return_value={"total_vectors": 0})
                ),
            )
        )
    )


@pytest.mark.asyncio
async def test_admin_stats_hide_residual_graph_counts_when_catalog_is_empty() -> None:
    """An empty authoritative catalog must override stale Neo4j counters."""

    catalog = SimpleNamespace(list_documents=Mock(return_value=[]))
    graph = SimpleNamespace(
        get_stats=AsyncMock(
            return_value={
                "total_entities": 198,
                "total_relations": 3,
                "nodes": 198,
                "edges": 3,
                "status": "ok",
            }
        )
    )

    transport_response = Response()
    response = await get_stats(
        _request(catalog=catalog, graph=graph),
        response=transport_response,
        user=_user(),
    )

    assert transport_response.headers["Cache-Control"] == "no-store"
    assert response.knowledge_graph == {
        "total_entities": 0,
        "total_relations": 0,
        "nodes": 0,
        "edges": 0,
        "status": "empty",
    }
    graph.get_stats.assert_not_awaited()


@pytest.mark.asyncio
async def test_admin_stats_scope_graph_counts_to_current_tenant_documents() -> None:
    """Current catalog records allow only same-tenant graph statistics."""

    catalog = SimpleNamespace(
        list_documents=Mock(
            return_value=[
                SimpleNamespace(department_id="finance", ingest_status="ingested"),
            ]
        )
    )
    graph = SimpleNamespace(
        get_stats=AsyncMock(
            return_value={
                "total_entities": 2,
                "total_relations": 1,
                "nodes": 2,
                "edges": 1,
                "status": "ok",
            }
        )
    )

    response = await get_stats(
        _request(catalog=catalog, graph=graph),
        response=Response(),
        user=_user(),
    )

    graph.get_stats.assert_awaited_once_with(tenant_id="tenant-a")
    assert response.knowledge_graph["nodes"] == 2
    assert response.knowledge_graph["edges"] == 1


@pytest.mark.asyncio
async def test_graph_service_builds_tenant_scoped_count_queries() -> None:
    """Neo4j aggregate queries must not mix statistics between tenants."""

    graph = KnowledgeGraphService()
    graph._driver = object()
    graph.execute_cypher = AsyncMock(side_effect=[[{"cnt": 8}], [{"cnt": 11}]])

    stats = await graph.get_stats(tenant_id="tenant-a")

    assert stats == {
        "total_entities": 8,
        "total_relations": 11,
        "nodes": 8,
        "edges": 11,
        "status": "ok",
    }
    first_query, first_params = graph.execute_cypher.await_args_list[0].args
    second_query, second_params = graph.execute_cypher.await_args_list[1].args
    assert "entity.tenant_id = $tenant_id" in first_query
    assert "source.tenant_id = $tenant_id" in second_query
    assert "target.tenant_id = $tenant_id" in second_query
    assert first_params == {"tenant_id": "tenant-a"}
    assert second_params == {"tenant_id": "tenant-a"}
