"""Acceptance tests for the repository-owned BM25 index and retriever."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from services.qa.retrievers import NativeBM25Retriever
from domain.documents import DocType, DocumentChunk
from domain.retrieval import RetrievalRequest, RetrievalScope, RetrievalStatus
from infrastructure.retrieval.bm25_index import (
    BM25IndexContractError,
    NativeBM25Index,
    tokenize_bm25,
)


def _chunk(
    content: str,
    *,
    tenant_id: str = "tenant-a",
    department_id: str = "finance",
    doc_id: str = "doc-1",
    chunk_index: int = 0,
) -> DocumentChunk:
    return DocumentChunk(
        content=content,
        doc_id=doc_id,
        chunk_index=chunk_index,
        doc_type=DocType.TEXT,
        tenant_id=tenant_id,
        metadata={
            "source": f"/uploads/{doc_id}.txt",
            "file_name": f"{doc_id}.txt",
            "company_id": "company-a",
            "department_id": department_id,
            "review_status": "approved",
            "sensitivity": "internal",
            "version": 3,
        },
    )


def test_tokenizer_is_deterministic_for_mixed_chinese_and_english() -> None:
    first = tokenize_bm25("Knowledge Graph 知识图谱 V2.0")
    second = tokenize_bm25("ＫＮＯＷＬＥＤＧＥ graph 知识图谱 v2.0")

    assert first == second
    assert "knowledge" in first
    assert "graph" in first
    assert "知识" in first
    assert "图谱" in first
    assert "v2.0" in first


@pytest.mark.asyncio
async def test_search_is_deterministic_and_preserves_provenance(tmp_path: Path) -> None:
    index = NativeBM25Index(tmp_path)
    await index.add_chunks(
        [
            _chunk("财务预算审批流程", chunk_index=0),
            _chunk("研发发布流程", department_id="engineering", doc_id="doc-2"),
        ]
    )

    first = await index.search(
        "预算审批",
        tenant_id="tenant-a",
        visible_department_ids=("finance",),
        top_k=5,
    )
    second = await index.search(
        "预算审批",
        tenant_id="tenant-a",
        visible_department_ids=("finance",),
        top_k=5,
    )

    assert first == second
    assert len(first) == 1
    document, score = first[0]
    assert score > 0
    assert document["content"] == "财务预算审批流程"
    assert document["metadata"]["tenant_id"] == "tenant-a"
    assert document["metadata"]["department_id"] == "finance"
    assert document["metadata"]["doc_id"] == "doc-1"
    assert document["metadata"]["chunk_id"] == "doc-1#chunk-0"
    assert document["metadata"]["chunk_index"] == 0
    assert document["metadata"]["version"] == 3


@pytest.mark.asyncio
async def test_tenant_and_department_filters_run_before_scoring(tmp_path: Path) -> None:
    index = NativeBM25Index(tmp_path)
    await index.add_chunks(
        [
            _chunk("共享机密预算", tenant_id="tenant-a", department_id="finance"),
            _chunk(
                "共享机密预算",
                tenant_id="tenant-a",
                department_id="engineering",
                doc_id="doc-2",
            ),
            _chunk(
                "共享机密预算",
                tenant_id="tenant-b",
                department_id="finance",
                doc_id="doc-3",
            ),
        ]
    )

    results = await index.search(
        "机密预算",
        tenant_id="tenant-a",
        visible_department_ids=("finance",),
        top_k=10,
    )

    assert [item[0]["metadata"]["doc_id"] for item in results] == ["doc-1"]


@pytest.mark.asyncio
async def test_restricted_department_scope_rejects_legacy_missing_metadata(tmp_path: Path) -> None:
    index = NativeBM25Index(tmp_path)
    chunk = _chunk("预算", department_id="")
    await index.add_chunks([chunk])

    results = await index.search(
        "预算",
        tenant_id="tenant-a",
        visible_department_ids=("finance",),
    )

    assert results == []


@pytest.mark.asyncio
async def test_add_is_idempotent_update_replaces_and_delete_is_targeted(tmp_path: Path) -> None:
    index = NativeBM25Index(tmp_path)
    await index.add_chunks([_chunk("旧预算"), _chunk("保留内容", doc_id="doc-2")])
    await index.add_chunks([_chunk("新预算")])

    assert [item[0]["content"] for item in await index.search("预算", tenant_id="tenant-a")] == [
        "新预算"
    ]
    assert await index.delete_by_doc_id("doc-1", tenant_id="tenant-a") == 1
    assert await index.search("预算", tenant_id="tenant-a") == []
    assert len(await index.search("保留", tenant_id="tenant-a")) == 1


@pytest.mark.asyncio
async def test_rebuild_and_status_report_independent_generation(tmp_path: Path) -> None:
    index = NativeBM25Index(tmp_path)
    await index.add_chunks([_chunk("旧内容")])
    first_status = await index.get_status("tenant-a")

    rebuilt = await index.rebuild("tenant-a", [_chunk("新内容", doc_id="doc-9")])
    second_status = await index.get_status("tenant-a")

    assert rebuilt == 1
    assert first_status["available"] is True
    assert second_status["available"] is True
    assert second_status["record_count"] == 1
    assert first_status["generation"] != second_status["generation"]
    assert await index.search("旧", tenant_id="tenant-a") == []
    assert len(await index.search("新", tenant_id="tenant-a")) == 1


@pytest.mark.asyncio
async def test_incompatible_snapshot_fails_closed(tmp_path: Path) -> None:
    index = NativeBM25Index(tmp_path)
    await index.add_chunks([_chunk("预算")])
    current = Path((await index.get_status("tenant-a"))["path"])
    payload = json.loads(current.read_text(encoding="utf-8"))
    payload["manifest"]["tokenizer_version"] = "legacy-v0"
    current.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

    with pytest.raises(BM25IndexContractError):
        await index.search("预算", tenant_id="tenant-a")


@pytest.mark.asyncio
async def test_failed_generation_keeps_previous_snapshot(tmp_path: Path, monkeypatch) -> None:
    index = NativeBM25Index(tmp_path)
    await index.add_chunks([_chunk("稳定内容")])

    def fail_write(*args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(index, "_write_snapshot_file", fail_write)
    with pytest.raises(OSError, match="disk full"):
        await index.add_chunks([_chunk("不应生效")])

    assert len(await index.search("稳定", tenant_id="tenant-a")) == 1
    assert await index.search("不应", tenant_id="tenant-a") == []


@pytest.mark.asyncio
async def test_native_bm25_adapter_returns_repository_contexts(tmp_path: Path) -> None:
    index = NativeBM25Index(tmp_path)
    await index.add_chunks([_chunk("财务预算审批")])
    request = RetrievalRequest(
        question="预算",
        rewritten={"queries": ["预算"]},
        scope=RetrievalScope(
            tenant_id="tenant-a",
            visible_department_ids=("finance",),
        ),
    )

    outcome = await NativeBM25Retriever(index).retrieve(request)

    assert outcome.status is RetrievalStatus.SUCCESS
    assert outcome.contexts[0].retrieval_type == "bm25"
    assert outcome.contexts[0].metadata["raw_score"] > 0
    assert outcome.contexts[0].metadata["source_rank"] == 1


@pytest.mark.asyncio
async def test_native_bm25_adapter_reports_missing_index_and_invalid_scope(tmp_path: Path) -> None:
    index = NativeBM25Index(tmp_path)
    missing = await NativeBM25Retriever(index).retrieve(
        RetrievalRequest(
            question="预算",
            rewritten={"queries": ["预算"]},
            scope=RetrievalScope(tenant_id="tenant-a"),
        )
    )
    invalid_scope = await NativeBM25Retriever(index).retrieve(
        RetrievalRequest(
            question="预算",
            rewritten={"queries": ["预算"]},
            scope=RetrievalScope(tenant_id=""),
        )
    )

    assert missing.status is RetrievalStatus.UNAVAILABLE
    assert invalid_scope.status is RetrievalStatus.UNAUTHORIZED
