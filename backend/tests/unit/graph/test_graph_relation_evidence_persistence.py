"""Deterministic persistence tests for claims with multiple source evidence records."""

from __future__ import annotations

import pytest

from domain.documents import DocType, DocumentChunk
from domain.knowledge import Entity, ExtractionResult, Relation
from workflows.ingestion_persistence import persist_extractions_to_graph


class RecordingGraph:
    """Record organization/entity/claim writes without a live Neo4j dependency."""

    def __init__(self) -> None:
        self.documents: list[dict] = []
        self.entities: list[tuple[str, str]] = []
        self.claims: list[tuple[str, str, str]] = []
        self.legacy_relations: list[str] = []

    async def upsert_document_context(self, context: dict) -> None:
        self.documents.append(context)

    async def upsert_entity(self, entity, *, source: str, tenant_id: str) -> None:
        self.entities.append((entity.name, tenant_id))

    async def add_relation_evidence(self, relation, *, context: dict, chunk_id: str) -> None:
        self.claims.append((relation.relation, str(context["doc_id"]), chunk_id))

    async def add_relation(self, relation, *, source: str, tenant_id: str) -> None:
        self.legacy_relations.append(relation.relation)


def _chunk(doc_id: str, index: int) -> DocumentChunk:
    """Build one catalog-backed chunk for a deterministic evidence fixture."""

    return DocumentChunk(
        content="evidence",
        doc_id=doc_id,
        chunk_index=index,
        doc_type=DocType.TEXT,
        tenant_id="tenant-a",
        metadata={
            "source": f"/{doc_id}.txt",
            "doc_id": doc_id,
            "company_id": "tenant-a",
            "department_id": "finance" if doc_id == "doc-a" else "procurement_warehouse",
            "display_name": doc_id,
        },
    )


@pytest.mark.asyncio
async def test_same_claim_from_two_documents_keeps_two_evidence_writes() -> None:
    """Claim deduplication is delegated to Neo4j while both source evidences survive."""

    chunks = [_chunk("doc-a", 0), _chunk("doc-b", 0)]
    relation = Relation(head="采购", relation="requires", tail="付款", confidence=0.9)
    extractions = [
        ExtractionResult(
            entities=[Entity(name="采购", type="Process"), Entity(name="付款", type="Process")],
            relations=[relation],
            events=[],
            source_chunk_id=chunks[0].chunk_id,
        ),
        ExtractionResult(
            entities=[],
            relations=[relation],
            events=[],
            source_chunk_id=chunks[1].chunk_id,
        ),
    ]
    graph = RecordingGraph()

    entities, relations = await persist_extractions_to_graph(
        chunks=chunks,
        extractions=extractions,
        knowledge_graph=graph,  # type: ignore[arg-type]
        tenant_id="tenant-a",
    )

    assert entities == 2
    assert relations == 2
    assert {context["doc_id"] for context in graph.documents} == {"doc-a", "doc-b"}
    assert graph.claims == [
        ("requires", "doc-a", "doc-a#chunk-0"),
        ("requires", "doc-b", "doc-b#chunk-0"),
    ]
    assert graph.legacy_relations == []
