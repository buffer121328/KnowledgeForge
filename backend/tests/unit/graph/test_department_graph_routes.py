"""Acceptance tests for tenant-scoped company and department graph routes."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
from fastapi import HTTPException
from starlette.requests import Request

from api.routers.knowledge_graph import (
    get_claim_evidence,
    get_company_graph_overview,
    get_department_graph,
    get_graph_paths,
)
from domain.documents import DepartmentRecord, DocumentIngestStatus
from domain.identity import Permission, UserContext, UserRole


def _user(org_id: str = "tenant-a") -> UserContext:
    """Return a graph reader scoped to one organization."""

    return UserContext(
        user_id="user-a",
        username="alice",
        role=UserRole.VIEWER,
        org_id=org_id,
        permissions=[Permission.GRAPH_READ],
    )


def _request(graph, catalog) -> Request:
    """Build a request with graph and catalog services attached."""

    request = Request(
        {
            "type": "http",
            "method": "GET",
            "path": "/api/v1/graph/company-overview",
            "headers": [],
            "client": ("203.0.113.10", 12345),
            "app": SimpleNamespace(
                state=SimpleNamespace(knowledge_graph=graph, document_catalog=catalog)
            ),
        }
    )
    request.state.tenant_id = "tenant-a"
    return request


_DEFAULT_DOCUMENTS = object()


def _catalog(*, visible: bool = True, documents=_DEFAULT_DOCUMENTS):
    """Return a tenant catalog mock with configurable graph-eligible documents."""

    department = DepartmentRecord(
        department_id="finance",
        tenant_id="tenant-a",
        company_id="tenant-a",
        name="财务部",
        normalized_key="finance",
    )
    if documents is _DEFAULT_DOCUMENTS:
        documents = [
            SimpleNamespace(
                doc_id="doc-finance",
                department_id="finance",
                ingest_status=DocumentIngestStatus.INGESTED.value,
            )
        ]
    return SimpleNamespace(
        get_department=Mock(return_value=department if visible else None),
        list_documents=Mock(return_value=documents),
    )


def _stale_company_overview() -> dict:
    """Return the exact stale one-company/three-department projection seen by users."""

    departments = [
        {"department_id": "finance", "name": "财务部", "document_count": 5},
        {"department_id": "hr", "name": "人力资源部", "document_count": 21},
        {"department_id": "procurement", "name": "采购部", "document_count": 5},
    ]
    return {
        "company_id": "tenant-a",
        "company_name": "示例公司",
        "departments": departments,
        "nodes": [
            {"id": "company:tenant-a", "label": "示例公司", "type": "Company"},
            *[
                {
                    "id": f"department:{item['department_id']}",
                    "label": item["name"],
                    "type": "Department",
                    "department_id": item["department_id"],
                    "document_count": item["document_count"],
                }
                for item in departments
            ],
        ],
        "edges": [
            {
                "source": "company:tenant-a",
                "target": f"department:{item['department_id']}",
                "label": "HAS_DEPARTMENT",
                "path_count": item["document_count"],
            }
            for item in departments
        ],
        "status": "ok",
    }


@pytest.mark.asyncio
async def test_company_overview_is_tenant_scoped_and_typed() -> None:
    """Overview calls the graph with only the authenticated tenant."""

    graph = SimpleNamespace(
        get_company_overview=AsyncMock(return_value=_stale_company_overview())
    )

    response = await get_company_graph_overview(_request(graph, _catalog()), user=_user())

    graph.get_company_overview.assert_awaited_once_with("tenant-a")
    assert response.company_id == "tenant-a"
    assert [item.department_id for item in response.departments] == ["finance"]
    assert response.departments[0].document_count == 1
    assert [node.id for node in response.nodes] == [
        "company:tenant-a",
        "department:finance",
    ]
    assert [(edge.source, edge.target) for edge in response.edges] == [
        ("company:tenant-a", "department:finance")
    ]
    assert response.edges[0].path_count == 1


@pytest.mark.asyncio
async def test_company_overview_is_empty_when_catalog_has_no_graph_documents() -> None:
    """Deleted catalog state overrides stale one-company/three-department Neo4j data."""

    graph = SimpleNamespace(
        get_company_overview=AsyncMock(return_value=_stale_company_overview())
    )

    response = await get_company_graph_overview(
        _request(graph, _catalog(documents=[])),
        user=_user(),
    )

    assert response.company_id == "tenant-a"
    assert response.departments == []
    assert response.nodes == []
    assert response.edges == []
    assert response.status == "empty"
    graph.get_company_overview.assert_not_awaited()


@pytest.mark.asyncio
async def test_department_route_conceals_foreign_or_missing_department() -> None:
    """A department outside the authenticated catalog scope is returned as 404."""

    graph = SimpleNamespace(get_department_graph=AsyncMock())

    with pytest.raises(HTTPException) as exc_info:
        await get_department_graph(
            _request(graph, _catalog(visible=False)),
            "finance",
            user=_user(),
        )

    assert exc_info.value.status_code == 404
    graph.get_department_graph.assert_not_awaited()


@pytest.mark.asyncio
async def test_department_route_passes_cross_department_flag_and_limit() -> None:
    """Authorized drill-down preserves scope controls while bounding the result size."""

    graph = SimpleNamespace(
        get_department_graph=AsyncMock(
            return_value={"nodes": [], "edges": [], "status": "ok"}
        )
    )

    response = await get_department_graph(
        _request(graph, _catalog()),
        "finance",
        include_cross_department=True,
        limit=9999,
        user=_user(),
    )

    graph.get_department_graph.assert_awaited_once_with(
        "finance",
        tenant_id="tenant-a",
        include_cross_department=True,
        limit=500,
    )
    assert response.department_id == "finance"


@pytest.mark.asyncio
async def test_paths_are_bounded_and_department_authorized() -> None:
    """Path requests clamp hops/results and verify an optional department scope."""

    graph = SimpleNamespace(
        get_paths=AsyncMock(return_value={"paths": [], "path_count": 0, "status": "ok"})
    )

    response = await get_graph_paths(
        _request(graph, _catalog()),
        from_id="entity:a",
        to_id="entity:b",
        department_id="finance",
        include_cross_department=False,
        max_hops=99,
        limit=999,
        user=_user(),
    )

    graph.get_paths.assert_awaited_once_with(
        from_id="entity:a",
        to_id="entity:b",
        tenant_id="tenant-a",
        department_id="finance",
        include_cross_department=False,
        max_hops=6,
        limit=50,
    )
    assert response.path_count == 0


@pytest.mark.asyncio
async def test_claim_without_tenant_visible_evidence_is_concealed() -> None:
    """Missing or foreign claims are indistinguishable through the tenant-scoped API."""

    graph = SimpleNamespace(
        get_claim_evidence=AsyncMock(
            return_value={"claim_id": "claim-x", "evidence": [], "status": "ok"}
        )
    )

    with pytest.raises(HTTPException) as exc_info:
        await get_claim_evidence(
            _request(graph, _catalog()),
            "claim-x",
            user=_user(),
        )

    assert exc_info.value.status_code == 404


@pytest.mark.asyncio
async def test_graph_timeout_returns_degraded_timeout_state(monkeypatch) -> None:
    """Slow graph dependencies end with a finite timeout response, not an infinite request."""

    async def _slow_overview(_tenant_id: str):
        await asyncio.sleep(0.05)
        return {}

    graph = SimpleNamespace(get_company_overview=_slow_overview)

    monkeypatch.setattr("api.routers.knowledge_graph.GRAPH_QUERY_TIMEOUT_SECONDS", 0.001)
    response = await get_company_graph_overview(
        _request(graph, _catalog()),
        user=_user(),
    )

    assert response.status == "timeout"
    assert response.nodes == []

@pytest.mark.asyncio
async def test_graph_entity_update_uses_edit_permission_and_tenant() -> None:
    """Manual entity corrections stay tenant-scoped before returning the updated node."""

    graph = SimpleNamespace(
        update_entity=AsyncMock(
            return_value={
                "id": "entity:预算制度",
                "label": "预算制度",
                "type": "Policy",
                "description": "已校正",
            }
        )
    )
    from api.routers.knowledge_graph import update_graph_entity
    from api.schemas import GraphEntityUpdateRequest

    response = await update_graph_entity(
        _request(graph, _catalog()),
        GraphEntityUpdateRequest(node_id="entity:预算", label="预算制度", type="Policy", description="已校正"),
        user=UserContext(
            user_id="admin-a",
            username="admin",
            role=UserRole.ADMIN,
            org_id="tenant-a",
            permissions=[Permission.GRAPH_EDIT],
        ),
    )

    graph.update_entity.assert_awaited_once_with(
        node_id="entity:预算",
        tenant_id="tenant-a",
        label="预算制度",
        entity_type="Policy",
        description="已校正",
    )
    assert response.node.label == "预算制度"


@pytest.mark.asyncio
async def test_graph_relation_update_normalizes_through_graph_service() -> None:
    """Manual relation corrections are addressed by relation claim id."""

    graph = SimpleNamespace(
        update_relation_claim=AsyncMock(
            return_value={"claim_id": "claim-1", "relation_type": "DEPENDS_ON"}
        )
    )
    from api.routers.knowledge_graph import update_graph_relation
    from api.schemas import GraphRelationUpdateRequest

    response = await update_graph_relation(
        _request(graph, _catalog()),
        GraphRelationUpdateRequest(claim_id="claim-1", relation_type="depends on"),
        user=UserContext(
            user_id="admin-a",
            username="admin",
            role=UserRole.ADMIN,
            org_id="tenant-a",
            permissions=[Permission.GRAPH_EDIT],
        ),
    )

    graph.update_relation_claim.assert_awaited_once_with(
        claim_id="claim-1",
        tenant_id="tenant-a",
        relation_type="depends on",
    )
    assert response.relation_type == "DEPENDS_ON"
