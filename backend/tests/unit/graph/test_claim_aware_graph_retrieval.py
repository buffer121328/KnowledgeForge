"""Acceptance tests for tenant-safe claim/evidence graph retrieval."""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from services.qa.retrieval import graph_retrieve
from infrastructure.graph.neo4j_graph import KnowledgeGraphService
from infrastructure.graph.neo4j_queries import build_claim_context_search_query


def test_claim_context_query_is_tenant_department_and_result_bounded() -> None:
    """Seed, claim, evidence, document, and returned evidence stay authorized and bounded."""

    cypher, params = build_claim_context_search_query(
        keyword="采购",
        tenant_id="tenant-a",
        visible_department_ids=("finance", "procurement"),
        limit=200,
        evidence_limit=100,
    )

    assert params == {
        "keyword": "采购",
        "tenant_id": "tenant-a",
        "visible_department_ids": ["finance", "procurement"],
        "limit": 50,
        "evidence_limit": 20,
    }
    assert "seed.tenant_id = $tenant_id" in cypher
    assert "claim:RelationClaim {tenant_id: $tenant_id}" in cypher
    assert "evidence:RelationEvidence {tenant_id: $tenant_id}" in cypher
    assert "document:Document {tenant_id: $tenant_id}" in cypher
    assert "evidence.department_id IN $visible_department_ids" in cypher
    assert "collect(DISTINCT" in cypher
    assert "[..$evidence_limit]" in cypher
    assert "LIMIT $limit" in cypher


def test_claim_context_query_allows_tenant_wide_scope_without_request_filter() -> None:
    """Current tenant-wide roles omit the department predicate rather than trusting request data."""

    cypher, params = build_claim_context_search_query(
        keyword="采购",
        tenant_id="tenant-a",
        visible_department_ids=None,
        limit=8,
        evidence_limit=5,
    )

    assert "evidence.department_id IN $visible_department_ids" not in cypher
    assert params["visible_department_ids"] == []


@pytest.mark.asyncio
async def test_graph_service_executes_claim_query_with_server_scope() -> None:
    """The facade forwards only explicit server-derived tenant/department scope."""

    service = KnowledgeGraphService()
    service._driver = object()
    service.execute_cypher = AsyncMock(return_value=[{"claim_id": "claim-1"}])  # type: ignore[method-assign]

    rows = await service.search_claim_contexts(
        "采购",
        tenant_id="tenant-a",
        visible_department_ids=("finance",),
        limit=7,
        evidence_limit=3,
    )

    assert rows == [{"claim_id": "claim-1"}]
    params = service.execute_cypher.await_args.args[1]
    assert params["tenant_id"] == "tenant-a"
    assert params["visible_department_ids"] == ["finance"]
    assert params["limit"] == 7
    assert params["evidence_limit"] == 3


@pytest.mark.asyncio
async def test_graph_retrieve_emits_one_canonical_context_per_evidence_chunk() -> None:
    """One claim preserves claim lineage while exposing canonical chunk identities."""

    graph = AsyncMock()
    graph.search_claim_contexts.return_value = [
        {
            "claim_id": "claim-1",
            "relation_type": "DEPENDS_ON",
            "head_name": "采购部",
            "head_type": "Department",
            "tail_name": "财务审批",
            "tail_type": "Process",
            "claim_confidence": 0.91,
            "evidence_count": 2,
            "evidences": [
                {
                    "evidence_id": "evidence-1",
                    "doc_id": "doc-a",
                    "chunk_id": "doc-a#chunk-4",
                    "department_id": "procurement",
                    "source": "采购申请流程.txt",
                },
                {
                    "evidence_id": "evidence-2",
                    "doc_id": "doc-b",
                    "chunk_id": "doc-b#chunk-7",
                    "department_id": "finance",
                    "source": "财务审批制度.txt",
                },
            ],
        }
    ]

    outcome = await graph_retrieve(
        graph,
        "采购依赖什么审批？",
        {"entities": ["采购部"]},
        tenant_id="tenant-a",
        visible_department_ids=("procurement", "finance"),
    )

    assert outcome.unavailable is False
    assert len(outcome.contexts) == 2
    context = outcome.contexts[0]
    assert "采购部 -[DEPENDS_ON]-> 财务审批" in context.content
    assert "证据: 2 条" in context.content
    assert context.source == "采购申请流程.txt"
    assert context.metadata["claim_id"] == "claim-1"
    assert context.metadata["chunk_id"] == "doc-a#chunk-4"
    assert context.metadata["source_document_id"] == "doc-a"
    assert context.metadata["department_id"] == "procurement"
    assert context.metadata["doc_ids"] == ["doc-a", "doc-b"]
    assert context.metadata["chunk_ids"] == ["doc-a#chunk-4", "doc-b#chunk-7"]
    assert context.metadata["evidence_count"] == 2
    assert context.metadata["source"] == "采购申请流程.txt"
    assert 0 < context.score <= 1
    assert outcome.contexts[1].metadata["chunk_id"] == "doc-b#chunk-7"
    assert outcome.contexts[1].metadata["source_document_id"] == "doc-b"
    graph.search_claim_contexts.assert_awaited_once_with(
        "采购部",
        tenant_id="tenant-a",
        visible_department_ids=("procurement", "finance"),
        limit=8,
        evidence_limit=5,
    )


@pytest.mark.asyncio
async def test_graph_retrieve_deduplicates_same_claim_returned_for_multiple_keywords() -> None:
    """Multiple seed matches cannot duplicate one claim in final QA contexts."""

    row = {
        "claim_id": "claim-1",
        "relation_type": "DEPENDS_ON",
        "head_name": "采购部",
        "head_type": "Department",
        "tail_name": "财务审批",
        "tail_type": "Process",
        "claim_confidence": 0.8,
        "evidence_count": 1,
        "evidences": [
            {
                "evidence_id": "evidence-1",
                "doc_id": "doc-a",
                "chunk_id": "doc-a#chunk-1",
                "department_id": "finance",
                "source": "制度.txt",
            }
        ],
    }
    graph = AsyncMock()
    graph.search_claim_contexts.return_value = [row]

    outcome = await graph_retrieve(
        graph,
        "问题",
        {"entities": ["采购部", "财务审批"]},
        tenant_id="tenant-a",
    )

    assert len(outcome.contexts) == 1


@pytest.mark.asyncio
async def test_graph_retrieve_excludes_claim_without_visible_evidence() -> None:
    """Evidence-free legacy/claim hints do not enter deterministic answer contexts."""

    graph = AsyncMock()
    graph.search_claim_contexts.return_value = [
        {
            "claim_id": "claim-legacy",
            "relation_type": "DEPENDS_ON",
            "head_name": "采购部",
            "tail_name": "财务审批",
            "claim_confidence": 0.9,
            "evidence_count": 0,
            "evidences": [],
        }
    ]

    outcome = await graph_retrieve(
        graph,
        "问题",
        {"entities": ["采购部"]},
        tenant_id="tenant-a",
    )

    assert outcome.contexts == []
    assert outcome.unavailable is False


@pytest.mark.asyncio
async def test_graph_retrieve_excludes_evidence_without_canonical_chunk_id() -> None:
    graph = AsyncMock()
    graph.search_claim_contexts.return_value = [{
        "claim_id": "claim-1",
        "relation_type": "DEPENDS_ON",
        "head_name": "采购部",
        "tail_name": "财务审批",
        "claim_confidence": 0.9,
        "evidence_count": 1,
        "evidences": [{
            "evidence_id": "evidence-1",
            "doc_id": "doc-a",
            "chunk_id": "",
            "department_id": "finance",
            "source": "制度.txt",
        }],
    }]

    outcome = await graph_retrieve(
        graph, "问题", {"entities": ["采购部"]}, tenant_id="tenant-a"
    )

    assert outcome.contexts == []
