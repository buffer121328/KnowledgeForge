"""ATDD for vector and graph persistence in asynchronous ingestion."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from domain.documents import DocType, DocumentChunk
from domain.knowledge import Entity, ExtractionResult, Relation
from infrastructure.tasks.celery_ingest_tasks import _run_ingest


def _fixture():
    chunk = DocumentChunk(
        content="采购审批制度",
        doc_id="doc-1",
        chunk_index=0,
        doc_type=DocType.WORD,
        metadata={"source": "/accepted/document.docx"},
        tenant_id="org-a",
    )
    extraction = ExtractionResult(
        entities=[Entity(name="采购部", type="Department")],
        relations=[Relation(head="采购部", relation="APPROVES", tail="申请")],
        events=[],
        source_chunk_id=chunk.chunk_id,
    )
    return chunk, extraction


@pytest.mark.asyncio
async def test_async_ingestion_persists_vectors_entities_and_relations(monkeypatch) -> None:
    chunk, extraction = _fixture()
    parser = MagicMock(parse=AsyncMock(return_value=[chunk]))
    extractor = MagicMock(extract=AsyncMock(return_value=[extraction]))
    vector = MagicMock(init=AsyncMock(), add_chunks=AsyncMock(return_value=1))
    sparse = MagicMock(add_chunks=AsyncMock(return_value=1))
    graph = MagicMock(
        init=AsyncMock(),
        upsert_entity=AsyncMock(),
        add_relation=AsyncMock(),
        close=AsyncMock(),
    )
    revision = MagicMock(advance=AsyncMock())
    monkeypatch.setattr("agents.document_parser.DocParserAgent", lambda: parser)
    monkeypatch.setattr("agents.knowledge_extractor.KnowledgeExtractAgent", lambda: extractor)
    monkeypatch.setattr("infrastructure.retrieval.vector_store.VectorStoreService", lambda: vector)
    monkeypatch.setattr("infrastructure.retrieval.bm25_index.NativeBM25Index", lambda *args, **kwargs: sparse)
    monkeypatch.setattr("infrastructure.graph.neo4j_graph.KnowledgeGraphService", lambda: graph)
    monkeypatch.setattr(
        "infrastructure.tasks.celery_ingest_tasks.KnowledgeRevisionStore",
        MagicMock(return_value=revision),
    )

    result = await _run_ingest("/accepted/document.docx", tenant_id="org-a")

    assert result["vectors_stored"] == 1
    assert result["sparse_stored"] == 1
    sparse.add_chunks.assert_awaited_once_with([chunk])
    assert result["entities_count"] == 1
    assert result["relations_count"] == 1
    graph.upsert_entity.assert_awaited_once_with(
        extraction.entities[0], source="/accepted/document.docx", tenant_id="org-a"
    )
    graph.add_relation.assert_awaited_once_with(
        extraction.relations[0], source="/accepted/document.docx", tenant_id="org-a"
    )
    graph.close.assert_awaited_once()
    revision.advance.assert_not_awaited()


@pytest.mark.asyncio
async def test_graph_persistence_failure_fails_ingestion_and_closes_driver(monkeypatch) -> None:
    chunk, extraction = _fixture()
    parser = MagicMock(parse=AsyncMock(return_value=[chunk]))
    extractor = MagicMock(extract=AsyncMock(return_value=[extraction]))
    vector = MagicMock(init=AsyncMock(), add_chunks=AsyncMock(return_value=1))
    graph = MagicMock(
        init=AsyncMock(),
        upsert_entity=AsyncMock(),
        add_relation=AsyncMock(side_effect=RuntimeError("graph unavailable")),
        close=AsyncMock(),
    )
    monkeypatch.setattr("agents.document_parser.DocParserAgent", lambda: parser)
    monkeypatch.setattr("agents.knowledge_extractor.KnowledgeExtractAgent", lambda: extractor)
    monkeypatch.setattr("infrastructure.retrieval.vector_store.VectorStoreService", lambda: vector)
    monkeypatch.setattr("infrastructure.graph.neo4j_graph.KnowledgeGraphService", lambda: graph)

    with pytest.raises(RuntimeError, match="graph unavailable"):
        await _run_ingest("/accepted/document.docx", tenant_id="org-a")

    graph.close.assert_awaited_once()


@pytest.mark.asyncio
async def test_sparse_persistence_failure_blocks_revision_advance(monkeypatch) -> None:
    chunk, extraction = _fixture()
    parser = MagicMock(parse=AsyncMock(return_value=[chunk]))
    extractor = MagicMock(extract=AsyncMock(return_value=[extraction]))
    vector = MagicMock(init=AsyncMock(), add_chunks=AsyncMock(return_value=1))
    sparse = MagicMock(add_chunks=AsyncMock(side_effect=OSError("sparse disk full")))
    graph = MagicMock(
        init=AsyncMock(),
        upsert_entity=AsyncMock(),
        add_relation=AsyncMock(),
        close=AsyncMock(),
    )
    revision = MagicMock(advance=AsyncMock())
    monkeypatch.setattr("agents.document_parser.DocParserAgent", lambda: parser)
    monkeypatch.setattr("agents.knowledge_extractor.KnowledgeExtractAgent", lambda: extractor)
    monkeypatch.setattr("infrastructure.retrieval.vector_store.VectorStoreService", lambda: vector)
    monkeypatch.setattr("infrastructure.retrieval.bm25_index.NativeBM25Index", lambda *args, **kwargs: sparse)
    monkeypatch.setattr("infrastructure.graph.neo4j_graph.KnowledgeGraphService", lambda: graph)
    monkeypatch.setattr(
        "infrastructure.tasks.celery_ingest_tasks.KnowledgeRevisionStore",
        MagicMock(return_value=revision),
    )

    with pytest.raises(OSError, match="sparse disk full"):
        await _run_ingest("/accepted/document.docx", tenant_id="org-a")

    revision.advance.assert_not_awaited()
    graph.close.assert_awaited_once()
