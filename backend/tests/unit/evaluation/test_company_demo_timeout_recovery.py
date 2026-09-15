"""Unit coverage for the constrained historic parser-timeout recovery command."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from types import SimpleNamespace

import pytest

from domain.documents import DocumentIngestStatus, DocumentRecord


@dataclass
class _Catalog:
    records: list[DocumentRecord]

    def list_documents(self, **_kwargs):
        return list(self.records)


@dataclass
class _Registry:
    updates: list[tuple[str, str]] = field(default_factory=list)

    def get_owned(self, task_id: str, tenant_id: str):
        assert task_id == "tsk-timeout"
        assert tenant_id == "org-1"
        return SimpleNamespace(actor_id="user-1", kind="folder_ingest")

    def update_state(self, task_id: str, state: str):
        self.updates.append((task_id, state))


def _record() -> DocumentRecord:
    return DocumentRecord(
        doc_id="doc-00",
        tenant_id="org-1",
        company_id="org-1",
        department_id="finance",
        folder_path="finance",
        relative_path="finance/a.docx",
        normalized_relative_path="finance/a.docx",
        uploaded_filename="a.docx",
        display_name="A",
        provenance_source_filename="a.docx",
        storage_reference="/accepted/a.docx",
        content_sha256="abc",
        size_bytes=1,
        mime_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        document_type="policy",
        source_format="docx",
        ingest_status=DocumentIngestStatus.PROCESSING.value,
        version=1,
        external_source_id="doc-00",
        metadata={"_upload_batch_id": "batch-1"},
    )


def _manifest(root) -> None:
    (root / "corpus_manifest.json").write_text(
        json.dumps({"documents": [{"source_document_id": f"doc-{index:02d}"} for index in range(41)]}),
        encoding="utf-8",
    )


@pytest.mark.asyncio
async def test_timeout_recovery_dry_run_requires_verified_batch_and_does_not_mutate(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    from evaluation.scripts import recover_company_demo_timeout as recovery

    _manifest(tmp_path)
    catalog = _Catalog([_record()])
    registry = _Registry()
    monkeypatch.setattr(recovery.celery_app, "AsyncResult", lambda _task_id: SimpleNamespace(state="FAILURE"))
    monkeypatch.setattr(recovery, "_parser_is_idle", lambda _task_id: True)
    monkeypatch.setattr(recovery, "get_task_registry", lambda: registry)
    monkeypatch.setattr(recovery, "PostgreSQLDocumentCatalogRepository", lambda _db: catalog)

    report = await recovery._recover(
        root=tmp_path,
        tenant_id="org-1",
        actor_id="user-1",
        task_id="tsk-timeout",
        upload_id="batch-1",
        apply=False,
    )

    assert report == {
        "task_id": "tsk-timeout",
        "task_state": "FAILURE",
        "selected_count": 1,
        "status": "validated",
    }
    assert registry.updates == []


@pytest.mark.asyncio
async def test_timeout_recovery_invokes_lifecycle_then_terminalizes_task(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    from evaluation.scripts import recover_company_demo_timeout as recovery

    _manifest(tmp_path)
    catalog = _Catalog([_record()])
    registry = _Registry()
    calls: list[tuple[str, bool]] = []

    class _Lifecycle:
        async def recover_interrupted_document(self, document_input, *, cleanup_partial_artifacts: bool):
            calls.append((document_input.doc_id, cleanup_partial_artifacts))
            return True

    monkeypatch.setattr(recovery.celery_app, "AsyncResult", lambda _task_id: SimpleNamespace(state="FAILURE"))
    monkeypatch.setattr(recovery, "_parser_is_idle", lambda _task_id: True)
    monkeypatch.setattr(recovery, "get_task_registry", lambda: registry)
    monkeypatch.setattr(recovery, "PostgreSQLDocumentCatalogRepository", lambda _db: catalog)
    monkeypatch.setattr(recovery, "_worker_lifecycle", lambda **_kwargs: _Lifecycle())

    report = await recovery._recover(
        root=tmp_path,
        tenant_id="org-1",
        actor_id="user-1",
        task_id="tsk-timeout",
        upload_id="batch-1",
        apply=True,
    )

    assert report["status"] == "recovered"
    assert report["recovered_count"] == 1
    assert calls == [("doc-00", True)]
    assert registry.updates == [("tsk-timeout", "failed")]
