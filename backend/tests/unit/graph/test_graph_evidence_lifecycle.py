"""Service-level acceptance tests for idempotent relation evidence lifecycle."""

from __future__ import annotations

from unittest.mock import AsyncMock, Mock

import pytest

from domain.knowledge import Relation
from infrastructure.graph.neo4j_graph import KnowledgeGraphService
from infrastructure.graph.neo4j_queries import build_document_evidence_delete_queries


def _context(doc_id: str, department_id: str) -> dict[str, str]:
    """Build a stable catalog-backed graph evidence context."""

    return {
        "tenant_id": "tenant-a",
        "company_id": "tenant-a",
        "department_id": department_id,
        "doc_id": doc_id,
        "display_name": doc_id,
    }


@pytest.mark.asyncio
async def test_relation_evidence_retry_reuses_claim_and_evidence_ids() -> None:
    """Retrying the same document/chunk semantic triple is idempotent."""

    service = KnowledgeGraphService()
    service._driver = object()
    service.execute_cypher = AsyncMock(return_value=[])  # type: ignore[method-assign]
    relation = Relation(head="采购", relation="requires", tail="付款", confidence=0.9)

    first = await service.add_relation_evidence(
        relation,
        context=_context("doc-a", "finance"),
        chunk_id="doc-a#chunk-0",
    )
    second = await service.add_relation_evidence(
        relation,
        context=_context("doc-a", "finance"),
        chunk_id="doc-a#chunk-0",
    )

    assert first == second
    assert service.execute_cypher.await_count == 2
    first_params = service.execute_cypher.await_args_list[0].args[1]
    second_params = service.execute_cypher.await_args_list[1].args[1]
    assert first_params["claim_id"] == second_params["claim_id"]
    assert first_params["evidence_id"] == second_params["evidence_id"]


@pytest.mark.asyncio
async def test_two_documents_share_claim_but_keep_distinct_evidence_ids() -> None:
    """The semantic claim deduplicates while each supporting document remains traceable."""

    service = KnowledgeGraphService()
    service._driver = object()
    service.execute_cypher = AsyncMock(return_value=[])  # type: ignore[method-assign]
    relation = Relation(head="采购", relation="requires", tail="付款", confidence=0.9)

    first = await service.add_relation_evidence(
        relation,
        context=_context("doc-a", "finance"),
        chunk_id="doc-a#chunk-0",
    )
    second = await service.add_relation_evidence(
        relation,
        context=_context("doc-b", "procurement_warehouse"),
        chunk_id="doc-b#chunk-0",
    )

    assert first is not None and second is not None
    assert first[0] == second[0]
    assert first[1] != second[1]


def test_delete_query_removes_evidence_before_only_orphaned_claims() -> None:
    """Deletion checks remaining evidence after deleting the selected document evidence."""

    steps = build_document_evidence_delete_queries("doc-a", "tenant-a")

    assert len(steps) == 7
    assert all(step[1]["tenant_id"] == "tenant-a" for step in steps)
    assert all(step[1]["doc_id"] == "doc-a" for step in steps)
    assert "DETACH DELETE evidence" in steps[1][0]
    assert "DETACH DELETE document" in steps[2][0]
    assert "NOT EXISTS" in steps[3][0]
    assert "DETACH DELETE department" in steps[4][0]
    assert "DETACH DELETE company" in steps[5][0]
    assert "orphan_departments" in steps[6][0]
    assert "orphan_companies" in steps[6][0]

@pytest.mark.asyncio
async def test_delete_document_evidence_runs_ordered_neo4j5_compatible_queries() -> None:
    """Deletion avoids the compound CALL-after-delete syntax rejected in production."""

    service = KnowledgeGraphService()
    service._driver = object()
    service.execute_cypher = AsyncMock(  # type: ignore[method-assign]
        side_effect=[
            [{"claim_ids": ["claim-a", "claim-b"]}],
            [{"deleted_evidence": 2}],
            [{"deleted_documents": 1}],
            [{"deleted_claims": 1}],
            [{"deleted_departments": 1}],
            [{"deleted_companies": 1}],
            [{
                "documents": 0,
                "evidences": 0,
                "orphan_departments": 0,
                "orphan_companies": 0,
            }],
        ]
    )

    deleted = await service.delete_document_evidence("doc-a", "tenant-a")

    assert deleted == 2
    assert service.execute_cypher.await_count == 7
    queries = [call.args[0] for call in service.execute_cypher.await_args_list]
    assert all("DETACH DELETE document\n        CALL" not in query for query in queries)
    assert "claim_ids" in service.execute_cypher.await_args_list[3].args[1]


@pytest.mark.asyncio
async def test_document_delete_surfaces_graph_cleanup_failure_before_vector_deletion() -> None:
    """A Neo4j failure cannot be hidden behind an HTTP 200 complete-success response."""

    from types import SimpleNamespace

    from fastapi import HTTPException

    from api.routers.documents_read import delete_document
    from domain.identity import Permission, UserContext, UserRole
    from services.documents.lifecycle import DocumentLifecycleCoordinator, DocumentLifecycleDependencies

    vector_store = SimpleNamespace(
        get_chunks_by_doc_id=AsyncMock(return_value=[{"source": "/uploads/doc-a.txt"}]),
        delete_by_doc_id=AsyncMock(return_value=1),
    )
    knowledge_graph = SimpleNamespace(
        delete_document_evidence=AsyncMock(side_effect=RuntimeError("cypher failed")),
    )
    catalog = SimpleNamespace(
        get_document=lambda *_args, **_kwargs: SimpleNamespace(storage_reference="/uploads/doc-a.txt"),
    )
    request = SimpleNamespace(
        state=SimpleNamespace(tenant_id="tenant-a"),
        app=SimpleNamespace(state=SimpleNamespace()),
    )
    lifecycle = DocumentLifecycleCoordinator(
        DocumentLifecycleDependencies(
            catalog=catalog,
            vector_store=vector_store,
            knowledge_graph=knowledge_graph,
            storage_factory=lambda: SimpleNamespace(delete=Mock()),
        )
    )
    user = UserContext(
        user_id="user-a",
        username="alice",
        role=UserRole.ORGANIZATION_ADMIN,
        org_id="tenant-a",
        permissions=[Permission.DOC_DELETE],
    )

    with pytest.raises(HTTPException) as captured:
        await delete_document("doc-a", request, user, lifecycle)

    assert captured.value.status_code == 503
    assert captured.value.detail["code"] == "document_graph_cleanup_failed"
    vector_store.delete_by_doc_id.assert_not_awaited()
