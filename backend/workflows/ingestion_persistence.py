"""Shared vector/graph persistence helpers for ingestion workflows and workers."""

from __future__ import annotations

from domain.documents import DocumentChunk
from domain.knowledge import ExtractionResult
from infrastructure.graph.neo4j_graph import KnowledgeGraphService


async def persist_extractions_to_graph(
    *,
    chunks: list[DocumentChunk],
    extractions: list[ExtractionResult],
    knowledge_graph: KnowledgeGraphService,
    tenant_id: str,
) -> tuple[int, int]:
    """Persist organization nodes, extracted entities, claims, and per-chunk evidence."""

    chunk_by_id = {chunk.chunk_id: chunk for chunk in chunks}
    default_chunk = chunks[0] if chunks else None
    seen_documents: set[str] = set()
    for chunk in chunks:
        context = {**chunk.metadata, "tenant_id": chunk.tenant_id or tenant_id}
        doc_id = str(context.get("doc_id") or chunk.doc_id)
        context["doc_id"] = doc_id
        if doc_id in seen_documents or not context.get("company_id") or not context.get("department_id"):
            continue
        await knowledge_graph.upsert_document_context(context)
        seen_documents.add(doc_id)

    entities_stored = 0
    relations_stored = 0
    for extraction in extractions:
        chunk = chunk_by_id.get(extraction.source_chunk_id, default_chunk)
        source = str(chunk.metadata.get("source") or "") if chunk is not None else ""
        chunk_tenant = chunk.tenant_id if chunk is not None else tenant_id
        context = (
            {**chunk.metadata, "tenant_id": chunk_tenant, "doc_id": chunk.doc_id}
            if chunk is not None
            else {"tenant_id": tenant_id}
        )
        for entity in extraction.entities:
            await knowledge_graph.upsert_entity(
                entity,
                source=source,
                tenant_id=chunk_tenant,
            )
            entities_stored += 1
        for relation in extraction.relations:
            if context.get("company_id") and context.get("department_id") and context.get("doc_id"):
                await knowledge_graph.add_relation_evidence(
                    relation,
                    context=context,
                    chunk_id=extraction.source_chunk_id,
                )
            else:
                await knowledge_graph.add_relation(
                    relation,
                    source=source,
                    tenant_id=chunk_tenant,
                )
            relations_stored += 1
    return entities_stored, relations_stored
