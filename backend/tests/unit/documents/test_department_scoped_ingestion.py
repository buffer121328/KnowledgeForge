"""Acceptance tests for stable per-document ingestion context propagation."""

from __future__ import annotations

from pathlib import Path
from types import MethodType

import pytest

from agents.document_parser import DocParserAgent
from domain.documents import DocType, DocumentChunk, IngestDocumentInput


@pytest.mark.asyncio
async def test_parser_preserves_catalog_context_on_every_chunk(tmp_path: Path) -> None:
    """Stable identity and provenance survive parsing without path-derived doc IDs."""

    source = tmp_path / "runtime.txt"
    source.write_text("first\nsecond", encoding="utf-8")
    parser = object.__new__(DocParserAgent)
    parser._classify = MethodType(lambda _self, _path: DocType.TEXT, parser)
    parser._parse_text = MethodType(lambda _self, _path: ["first\nsecond"], parser)

    async def build_chunks(
        _self,
        _texts,
        doc_id,
        doc_type,
        file_path,
        tenant_id,
    ):
        return [
            DocumentChunk(
                content="first",
                doc_id=doc_id,
                chunk_index=0,
                doc_type=doc_type,
                metadata={"source": file_path},
                tenant_id=tenant_id,
            ),
            DocumentChunk(
                content="second",
                doc_id=doc_id,
                chunk_index=1,
                doc_type=doc_type,
                metadata={"source": file_path},
                tenant_id=tenant_id,
            ),
        ]

    parser._chunk_texts_for_strategy = MethodType(build_chunks, parser)
    document_input = IngestDocumentInput(
        doc_id="stable-doc-id",
        tenant_id="tenant-a",
        company_id="tenant-a",
        department_id="finance",
        file_path=str(source),
        uploaded_filename="runtime.txt",
        display_name="资金制度",
        provenance_source_filename="资金制度.docx",
        relative_path="finance/runtime.txt",
        folder_path="finance",
        content_sha256="a" * 64,
        version=2,
        authority="formal_candidate",
        review_status="needs_review",
        sensitivity="internal_demo",
        external_source_id="source-1",
    )

    chunks = await parser.parse(
        str(source),
        tenant_id="ignored-tenant",
        document_input=document_input,
    )

    assert {chunk.doc_id for chunk in chunks} == {"stable-doc-id"}
    assert {chunk.tenant_id for chunk in chunks} == {"tenant-a"}
    for chunk in chunks:
        assert chunk.metadata["company_id"] == "tenant-a"
        assert chunk.metadata["department_id"] == "finance"
        assert chunk.metadata["relative_path"] == "finance/runtime.txt"
        assert chunk.metadata["display_name"] == "资金制度"
        assert chunk.metadata["provenance_source_filename"] == "资金制度.docx"
        assert chunk.metadata["content_sha256"] == "a" * 64
        assert chunk.metadata["version"] == 2
        assert chunk.metadata["authority"] == "formal_candidate"
        assert chunk.metadata["review_status"] == "needs_review"
        assert chunk.metadata["sensitivity"] == "internal_demo"


@pytest.mark.asyncio
async def test_parse_batch_rejects_misaligned_document_inputs() -> None:
    """Parallel path/input arrays cannot silently associate the wrong department."""

    parser = object.__new__(DocParserAgent)
    with pytest.raises(ValueError, match="document input count"):
        await parser.parse_batch(["a.txt"], document_inputs=[])
