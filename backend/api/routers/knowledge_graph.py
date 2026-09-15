"""Knowledge graph visualization HTTP routes."""

from __future__ import annotations

import asyncio
from collections import Counter
from collections.abc import Awaitable
from typing import Annotated, TypeVar

from fastapi import APIRouter, Depends, HTTPException, Query, Request, status
from pydantic import RootModel

from api.contracts import ResourceId
from api.dependencies import require_permission
from api.schemas import (
    CompanyGraphOverviewResponse,
    GraphEdgeItem,
    GraphEntityUpdateRequest,
    GraphEntityUpdateResponse,
    GraphEvidenceResponse,
    GraphNodeItem,
    GraphPathResponse,
    GraphRelationUpdateRequest,
    GraphRelationUpdateResponse,
    GraphSubgraphResponse,
)
from domain.documents import DocumentIngestStatus
from domain.identity import Permission, UserContext


class GraphEntityTypeList(RootModel[list[str]]):
    """Preserve the entity-type array response with a reusable schema."""


graph_router = APIRouter(prefix="/graph", tags=["知识图谱"])
GRAPH_QUERY_TIMEOUT_SECONDS = 5.0
T = TypeVar("T")


def _tenant_from_request(request: Request, user: UserContext | None = None) -> str | None:
    """Return the authenticated tenant, falling back to the user context."""

    tenant_id = getattr(request.state, "tenant_id", None) or None
    if tenant_id == "":
        tenant_id = None
    if tenant_id is None and user is not None:
        tenant_id = user.org_id or None
    return tenant_id


def _require_tenant(request: Request, user: UserContext) -> str:
    """Return the authenticated tenant or reject an incomplete identity context."""

    tenant_id = _tenant_from_request(request, user)
    if not tenant_id:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="租户上下文不可用")
    return tenant_id


def _require_department(request: Request, department_id: str, tenant_id: str) -> None:
    """Conceal departments that do not belong to the authenticated tenant/company."""

    catalog = getattr(request.app.state, "document_catalog", None)
    department = (
        catalog.get_department(
            department_id,
            tenant_id=tenant_id,
            company_id=tenant_id,
        )
        if catalog is not None
        else None
    )
    if department is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="部门不存在")


_GRAPH_VISIBLE_DOCUMENT_STATUSES = {
    DocumentIngestStatus.INGESTED.value,
    DocumentIngestStatus.LEGACY.value,
}


def _catalog_department_counts(request: Request, tenant_id: str) -> Counter[str]:
    """Return graph-eligible document counts from the lifecycle source of truth."""

    catalog = getattr(request.app.state, "document_catalog", None)
    if catalog is None:
        return Counter()
    documents = catalog.list_documents(tenant_id=tenant_id, company_id=tenant_id)
    return Counter(
        str(document.department_id)
        for document in documents
        if document.department_id
        and document.ingest_status in _GRAPH_VISIBLE_DOCUMENT_STATUSES
    )


def _catalog_backed_company_overview(
    data: dict,
    *,
    tenant_id: str,
    department_counts: Counter[str],
) -> dict:
    """Intersect Neo4j organization output with catalog-backed department facts."""

    status_value = str(data.get("status") or "ok")
    company_id = str(data.get("company_id") or tenant_id)
    company_name = str(data.get("company_name") or company_id)
    if not department_counts:
        return {
            "company_id": tenant_id,
            "company_name": "",
            "departments": [],
            "nodes": [],
            "edges": [],
            "status": status_value,
        }

    graph_departments = {
        str(item.get("department_id") or ""): item
        for item in data.get("departments", [])
        if item and str(item.get("department_id") or "") in department_counts
    }
    department_nodes = {
        str(node.get("department_id") or node.get("id", "").removeprefix("department:"))
        for node in data.get("nodes", [])
        if node.get("type") == "Department"
    }
    visible_department_ids = set(graph_departments) & department_nodes
    if not visible_department_ids:
        return {
            "company_id": company_id,
            "company_name": company_name,
            "departments": [],
            "nodes": [],
            "edges": [],
            "status": status_value,
        }

    departments = [
        {
            **graph_departments[department_id],
            "document_count": department_counts[department_id],
        }
        for department_id in sorted(
            visible_department_ids,
            key=lambda item: str(graph_departments[item].get("name") or item),
        )
    ]
    nodes: list[dict] = []
    for node in data.get("nodes", []):
        node_type = str(node.get("type") or "")
        if node_type == "Company":
            nodes.append(dict(node))
        elif node_type == "Department":
            department_id = str(
                node.get("department_id")
                or str(node.get("id") or "").removeprefix("department:")
            )
            if department_id in visible_department_ids:
                nodes.append(
                    {
                        **node,
                        "department_id": department_id,
                        "document_count": department_counts[department_id],
                    }
                )

    visible_node_ids = {str(node.get("id") or "") for node in nodes}
    edges: list[dict] = []
    for edge in data.get("edges", []):
        source_id = str(edge.get("source") or "")
        target_id = str(edge.get("target") or "")
        if source_id not in visible_node_ids or target_id not in visible_node_ids:
            continue
        projected_edge = dict(edge)
        if str(edge.get("label") or "") == "HAS_DEPARTMENT":
            department_id = target_id.removeprefix("department:")
            projected_edge["path_count"] = department_counts[department_id]
        edges.append(projected_edge)
    return {
        "company_id": company_id,
        "company_name": company_name,
        "departments": departments,
        "nodes": nodes,
        "edges": edges,
        "status": status_value,
    }


async def _await_graph_query(awaitable: Awaitable[T], fallback: T) -> T:
    """Bound graph dependency latency and return a typed degraded response on failure."""

    try:
        return await asyncio.wait_for(awaitable, timeout=GRAPH_QUERY_TIMEOUT_SECONDS)
    except TimeoutError:
        if isinstance(fallback, dict):
            fallback["status"] = "timeout"
        return fallback
    except Exception:
        if isinstance(fallback, dict):
            fallback["status"] = "disconnected"
        return fallback


@graph_router.get("/subgraph", response_model=GraphSubgraphResponse)
async def get_graph_subgraph(
    request: Request,
    q: Annotated[str, Query(max_length=255)] = "",
    entity_type: Annotated[str, Query(max_length=64)] = "",
    limit: Annotated[int, Query(ge=1, le=500)] = 200,
    user: UserContext = Depends(require_permission(Permission.GRAPH_READ)),
):
    """查询知识图谱子图（可视化）"""

    data = await request.app.state.knowledge_graph.get_subgraph(
        keyword=q,
        entity_type=entity_type if entity_type not in ("", "all") else "",
        limit=limit,
        tenant_id=_tenant_from_request(request, user),
    )
    return GraphSubgraphResponse(
        nodes=[GraphNodeItem(**node) for node in data.get("nodes", [])],
        edges=[GraphEdgeItem(**edge) for edge in data.get("edges", [])],
        status=data.get("status", "ok"),
    )


@graph_router.get("/company-overview", response_model=CompanyGraphOverviewResponse)
async def get_company_graph_overview(
    request: Request,
    user: UserContext = Depends(require_permission(Permission.GRAPH_READ)),
) -> CompanyGraphOverviewResponse:
    """Return the authenticated tenant's company and department overview."""

    tenant_id = _require_tenant(request, user)
    department_counts = _catalog_department_counts(request, tenant_id)
    if not department_counts:
        return CompanyGraphOverviewResponse(company_id=tenant_id, status="empty")

    data = await _await_graph_query(
        request.app.state.knowledge_graph.get_company_overview(tenant_id),
        {"company_id": tenant_id, "nodes": [], "edges": [], "departments": [], "status": "disconnected"},
    )
    return CompanyGraphOverviewResponse(
        **_catalog_backed_company_overview(
            data,
            tenant_id=tenant_id,
            department_counts=department_counts,
        )
    )


@graph_router.get("/departments/{department_id}", response_model=GraphSubgraphResponse)
async def get_department_graph(
    request: Request,
    department_id: ResourceId,
    include_cross_department: bool = False,
    limit: int = Query(default=200, ge=1, le=500),
    user: UserContext = Depends(require_permission(Permission.GRAPH_READ)),
) -> GraphSubgraphResponse:
    """Return one authorized department graph with optional cross-department evidence."""

    tenant_id = _require_tenant(request, user)
    _require_department(request, department_id, tenant_id)
    data = await _await_graph_query(
        request.app.state.knowledge_graph.get_department_graph(
            department_id,
            tenant_id=tenant_id,
            include_cross_department=include_cross_department,
            limit=max(1, min(int(limit), 500)),
        ),
        {"nodes": [], "edges": [], "status": "disconnected"},
    )
    return GraphSubgraphResponse(
        nodes=[GraphNodeItem(**node) for node in data.get("nodes", [])],
        edges=[GraphEdgeItem(**edge) for edge in data.get("edges", [])],
        status=data.get("status", "ok"),
        scope="department",
        department_id=department_id,
    )


@graph_router.get("/paths", response_model=GraphPathResponse)
async def get_graph_paths(
    request: Request,
    from_id: ResourceId,
    to_id: ResourceId,
    department_id: ResourceId | None = None,
    include_cross_department: bool = False,
    max_hops: int = Query(default=4, ge=1, le=6),
    limit: int = Query(default=20, ge=1, le=50),
    user: UserContext = Depends(require_permission(Permission.GRAPH_READ)),
) -> GraphPathResponse:
    """Return finite tenant-scoped paths between two selected graph nodes."""

    tenant_id = _require_tenant(request, user)
    scoped_department = department_id or None
    if scoped_department:
        _require_department(request, scoped_department, tenant_id)
    data = await _await_graph_query(
        request.app.state.knowledge_graph.get_paths(
            from_id=from_id,
            to_id=to_id,
            tenant_id=tenant_id,
            department_id=scoped_department,
            include_cross_department=include_cross_department,
            max_hops=max(1, min(int(max_hops), 6)),
            limit=max(1, min(int(limit), 50)),
        ),
        {"paths": [], "path_count": 0, "status": "disconnected"},
    )
    return GraphPathResponse(**data)


@graph_router.get("/claims/{claim_id}/evidence", response_model=GraphEvidenceResponse)
async def get_claim_evidence(
    request: Request,
    claim_id: ResourceId,
    user: UserContext = Depends(require_permission(Permission.GRAPH_READ)),
) -> GraphEvidenceResponse:
    """Return document/chunk evidence for one tenant-visible semantic claim."""

    tenant_id = _require_tenant(request, user)
    data = await _await_graph_query(
        request.app.state.knowledge_graph.get_claim_evidence(claim_id, tenant_id),
        {"claim_id": claim_id, "evidence": [], "status": "disconnected"},
    )
    if data.get("status") == "ok" and not data.get("evidence"):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="关系证据不存在")
    return GraphEvidenceResponse(**data)


@graph_router.patch("/entities", response_model=GraphEntityUpdateResponse)
async def update_graph_entity(
    request: Request,
    payload: GraphEntityUpdateRequest,
    user: UserContext = Depends(require_permission(Permission.GRAPH_EDIT)),
) -> GraphEntityUpdateResponse:
    """Apply an authorized user correction to one extracted entity."""

    tenant_id = _require_tenant(request, user)
    node = await request.app.state.knowledge_graph.update_entity(
        node_id=payload.node_id,
        tenant_id=tenant_id,
        label=payload.label,
        entity_type=payload.type,
        description=payload.description,
    )
    if not node:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="实体不存在")
    return GraphEntityUpdateResponse(node=GraphNodeItem(**node))


@graph_router.patch("/relations", response_model=GraphRelationUpdateResponse)
async def update_graph_relation(
    request: Request,
    payload: GraphRelationUpdateRequest,
    user: UserContext = Depends(require_permission(Permission.GRAPH_EDIT)),
) -> GraphRelationUpdateResponse:
    """Apply an authorized user correction to one extracted relation claim."""

    tenant_id = _require_tenant(request, user)
    relation = await request.app.state.knowledge_graph.update_relation_claim(
        claim_id=payload.claim_id,
        tenant_id=tenant_id,
        relation_type=payload.relation_type,
    )
    if not relation:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="关系不存在或类型非法")
    return GraphRelationUpdateResponse(**relation)


@graph_router.get("/types", response_model=GraphEntityTypeList)
async def list_graph_entity_types(
    request: Request,
    user: UserContext = Depends(require_permission(Permission.GRAPH_READ)),
):
    """列出图谱中已有实体类型"""

    return await request.app.state.knowledge_graph.list_entity_types(
        tenant_id=_tenant_from_request(request, user)
    )
