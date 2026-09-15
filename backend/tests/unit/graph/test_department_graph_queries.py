"""Deterministic query tests for organization hierarchy and relation evidence."""

from __future__ import annotations

from infrastructure.graph.neo4j_queries import (
    build_claim_evidence_query,
    build_company_overview_query,
    build_department_graph_query,
    build_department_metadata_sync_query,
    build_document_context_query,
    build_document_evidence_delete_queries,
    build_entity_update_query,
    build_paths_query,
    build_relation_claim_query,
    build_relation_claim_update_query,
)


def _context() -> dict:
    """Return one complete document context for graph query fixtures."""

    return {
        "tenant_id": "tenant-a",
        "company_id": "tenant-a",
        "department_id": "finance",
        "department_name": "财务部",
        "doc_id": "doc-a",
        "display_name": "资金制度",
        "uploaded_filename": "runtime.txt",
        "provenance_source_filename": "资金制度.docx",
        "relative_path": "finance/runtime.txt",
        "folder_path": "finance",
        "version": 1,
    }


def test_document_context_query_uses_tenant_scoped_merge_keys() -> None:
    """Company, department, and document ownership is deterministic and idempotent."""

    cypher, params = build_document_context_query(_context())

    assert "MERGE (company:Company {tenant_id: $tenant_id, company_id: $company_id})" in cypher
    assert "department_id: $department_id" in cypher
    assert "MERGE (document:Document {tenant_id: $tenant_id, doc_id: $doc_id})" in cypher
    assert "MERGE (department)-[:OWNS_DOCUMENT]->(document)" in cypher
    assert params["tenant_id"] == "tenant-a"
    assert params["provenance_source_filename"] == "资金制度.docx"


def test_document_context_and_sync_queries_use_canonical_department_names() -> None:
    context = {**_context(), "department_id": "procurement_warehouse", "department_name": ""}

    _cypher, params = build_document_context_query(context)
    sync_cypher, sync_params = build_department_metadata_sync_query(
        "tenant-a",
        {"administration": "行政管理部", "finance": "财务部"},
    )

    assert params["department_name"] == "采购/仓储部"
    assert "MATCH (department:Department" in sync_cypher
    assert sync_params == {
        "tenant_id": "tenant-a",
        "departments": [
            {"department_id": "administration", "name": "行政管理部"},
            {"department_id": "finance", "name": "财务部"},
        ],
    }


def test_relation_claim_query_deduplicates_claim_and_evidence_independently() -> None:
    """One claim can retain multiple deterministic document/chunk evidence records."""

    cypher, params = build_relation_claim_query(
        claim_id="claim-1",
        evidence_id="evidence-1",
        head="采购部",
        relation_type="APPROVES",
        tail="付款申请",
        confidence=0.9,
        context=_context(),
        chunk_id="doc-a#chunk-0",
        extractor="knowledge_extractor",
        model="fixture",
        version="1",
    )

    assert "MERGE (claim:RelationClaim {tenant_id: $tenant_id, claim_id: $claim_id})" in cypher
    assert "SET claim.confidence = CASE" in cypher
    assert "MERGE (evidence:RelationEvidence" in cypher
    assert "MERGE (document)-[:PROVIDES_EVIDENCE]->(evidence)" in cypher
    assert "MERGE (evidence)-[:SUPPORTS]->(claim)" in cypher
    assert params["claim_id"] == "claim-1"
    assert params["evidence_id"] == "evidence-1"


def test_document_deletion_removes_evidence_before_orphan_claims() -> None:
    """Deletion preserves claims until no tenant evidence remains."""

    steps = build_document_evidence_delete_queries("doc-a", "tenant-a")

    assert len(steps) == 7
    assert "PROVIDES_EVIDENCE" in steps[0][0]
    assert "DETACH DELETE evidence" in steps[1][0]
    assert "NOT EXISTS" in steps[3][0]
    assert "MATCH (department:Department {tenant_id: $tenant_id})" in steps[4][0]
    assert "DETACH DELETE department" in steps[4][0]
    assert "MATCH (company:Company {tenant_id: $tenant_id})" in steps[5][0]
    assert "DETACH DELETE company" in steps[5][0]
    assert steps[0][1] == {"tenant_id": "tenant-a", "doc_id": "doc-a"}


def test_overview_department_paths_and_evidence_are_tenant_bounded() -> None:
    """Every graph read query carries tenant and result bounds."""

    overview, overview_params = build_company_overview_query("tenant-a")
    department, department_params = build_department_graph_query(
        "tenant-a",
        "finance",
        include_cross_department=False,
        limit=10_000,
    )
    paths, path_params = build_paths_query(
        tenant_id="tenant-a",
        from_id="finance",
        to_id="procurement_warehouse",
        department_id="finance",
        include_cross_department=False,
        max_hops=999,
        limit=999,
    )
    evidence, evidence_params = build_claim_evidence_query("claim-1", "tenant-a")

    assert "tenant_id: $tenant_id" in overview
    assert "MATCH (company)-[:HAS_DEPARTMENT]->(department:Department)" in overview
    assert "MATCH (department)-[:OWNS_DOCUMENT]->(document:Document)" in overview
    assert "OPTIONAL MATCH (company)-[:HAS_DEPARTMENT]" not in overview
    assert overview_params == {"tenant_id": "tenant-a"}
    assert "evidence.tenant_id = $tenant_id" in department
    assert department_params["limit"] == 500
    assert "[*1..6]" in paths
    assert "all(node IN nodes(path) WHERE node.tenant_id = $tenant_id)" in paths
    assert path_params["limit"] == 50
    assert "RelationEvidence {tenant_id: $tenant_id}" in evidence
    assert evidence_params == {"tenant_id": "tenant-a", "claim_id": "claim-1"}


def test_manual_graph_correction_queries_are_tenant_bounded() -> None:
    """Entity and relation correction writes cannot cross tenant boundaries."""

    entity, entity_params = build_entity_update_query(
        tenant_id="tenant-a",
        node_id="entity:预算",
        label="预算制度",
        entity_type="Policy",
        description="已校正",
    )
    relation, relation_params = build_relation_claim_update_query(
        tenant_id="tenant-a",
        claim_id="claim-1",
        relation_type="DEPENDS_ON",
    )

    assert "MATCH (entity:Entity {tenant_id: $tenant_id, name: $entity_name})" in entity
    assert "entity.name" in entity
    assert entity_params["entity_name"] == "预算"
    assert "MATCH (claim:RelationClaim {tenant_id: $tenant_id, claim_id: $claim_id})" in relation
    assert "RelationEvidence {tenant_id: $tenant_id}" in relation
    assert relation_params == {"tenant_id": "tenant-a", "claim_id": "claim-1", "relation_type": "DEPENDS_ON"}
