from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace

import pytest

from domain.documents import DocumentIngestStatus, DocumentRecord
from infrastructure.documents.local_uploads import StagedUpload, UploadInspection, UploadPolicyError
from services.documents.submission import (
    DocumentSubmissionCoordinator,
    DocumentSubmissionDependencies,
    DocumentSubmissionError,
    DocumentSubmissionRequest,
    DocumentSubmissionSettings,
)


@dataclass
class FakeStorage:
    staged: StagedUpload
    accepted_path: str
    discarded: list[StagedUpload]
    deleted: list[tuple[str, str | None]]
    quarantined: list[StagedUpload]

    def stage(self, **_: object) -> StagedUpload:
        return self.staged

    def promote(self, *_: object, **__: object) -> str:
        return self.accepted_path

    def discard(self, staged_or_path: StagedUpload | str) -> bool:
        assert isinstance(staged_or_path, StagedUpload)
        self.discarded.append(staged_or_path)
        return True

    def delete(self, path: str, tenant_id: str | None = None) -> bool:
        self.deleted.append((path, tenant_id))
        return True

    def quarantine(self, staged: StagedUpload) -> str:
        self.quarantined.append(staged)
        return staged.path


class FakeCatalog:
    def __init__(self) -> None:
        self.records: dict[str, DocumentRecord] = {}
        self.departments: list[object] = []
        self.removed: list[tuple[str, str]] = []

    def upsert_department(self, department: object) -> None:
        self.departments.append(department)

    def create_document(self, record: DocumentRecord) -> None:
        self.records[record.doc_id] = record

    def update_document(self, record: DocumentRecord) -> None:
        self.records[record.doc_id] = record

    def get_document(self, doc_id: str, *, tenant_id: str) -> DocumentRecord | None:
        record = self.records.get(doc_id)
        return record if record is not None and record.tenant_id == tenant_id else None

    def remove_document(self, doc_id: str, *, tenant_id: str) -> None:
        self.records.pop(doc_id, None)
        self.removed.append((doc_id, tenant_id))


class FakePolicy:
    def __init__(self, inspection: UploadInspection | Exception) -> None:
        self.inspection = inspection

    def validate(self, *_: object) -> UploadInspection:
        if isinstance(self.inspection, Exception):
            raise self.inspection
        return self.inspection


class SuccessfulWorkflow:
    async def ainvoke(self, payload: dict) -> dict:
        await payload["progress_callback"]("extract", 2, 6, "知识抽取")
        return {
            "chunks": [SimpleNamespace(doc_type=SimpleNamespace(value="pdf"), doc_id="doc-1")],
            "entities_stored": 2,
            "relations_stored": 3,
        }


class FailingWorkflow:
    async def ainvoke(self, _: dict) -> dict:
        raise RuntimeError("downstream unavailable")


def _storage(tmp_path: Path) -> FakeStorage:
    staged_path = tmp_path / "staged.pdf"
    staged_path.write_bytes(b"hello")
    return FakeStorage(
        staged=StagedUpload(
            path=str(staged_path),
            tenant_id="tenant-a",
            tenant_key="tenant-key",
            extension=".pdf",
            size_bytes=5,
        ),
        accepted_path=str(tmp_path / "accepted.pdf"),
        discarded=[],
        deleted=[],
        quarantined=[],
    )


def _request() -> DocumentSubmissionRequest:
    return DocumentSubmissionRequest(
        tenant_id="tenant-a",
        user_id="user-a",
        username="alice",
        org_id="tenant-a",
        source=BytesIO(b"hello"),
        original_filename="report.pdf",
        content_type="application/pdf",
        upload_id="batch-1",
        client_file_id="file-1",
    )


def _coordinator(
    storage: FakeStorage,
    policy: FakePolicy,
    workflow: object | None,
    catalog: FakeCatalog | None,
    audit_events: list[tuple],
    webhook_events: list[tuple],
    metric_events: list[tuple],
) -> DocumentSubmissionCoordinator:
    def audit_denial(request: DocumentSubmissionRequest, code: str, size_bytes: int) -> None:
        audit_events.append(("denied", request.user_id, code, size_bytes))

    def audit_failure(request: DocumentSubmissionRequest, code: str) -> None:
        audit_events.append(("failure", request.user_id, code))

    def audit_success(request: DocumentSubmissionRequest, result: object, inspection: UploadInspection) -> None:
        audit_events.append(("success", request.user_id, result.doc_id, inspection.parser_type))

    async def trigger_webhook(request: DocumentSubmissionRequest, result: object) -> None:
        webhook_events.append((request.org_id, result.doc_id, result.chunks_count))

    return DocumentSubmissionCoordinator(
        DocumentSubmissionDependencies(
            storage_factory=lambda: storage,
            upload_policy=policy,
            catalog=catalog,
            ingest_workflow=workflow,
            settings=DocumentSubmissionSettings(
                max_file_bytes=100,
                tenant_quota_bytes=1_000,
                stream_chunk_bytes=16,
                quarantine_retention_hours=24,
            ),
            audit_denial=audit_denial,
            audit_failure=audit_failure,
            audit_success=audit_success,
            trigger_webhook=trigger_webhook,
            record_metrics=lambda doc_type, duration: metric_events.append((doc_type, duration)),
            allocate_document_id=lambda: "doc-1",
            clock=iter([100.0, 101.5]).__next__,
        )
    )


@pytest.mark.asyncio
async def test_submission_coordinates_catalog_progress_audit_webhook_and_metrics(tmp_path: Path) -> None:
    storage = _storage(tmp_path)
    catalog = FakeCatalog()
    audit_events: list[tuple] = []
    webhook_events: list[tuple] = []
    metric_events: list[tuple] = []

    result = await _coordinator(
        storage,
        FakePolicy(UploadInspection(parser_type="pdf", extension=".pdf", size_bytes=5)),
        SuccessfulWorkflow(),
        catalog,
        audit_events,
        webhook_events,
        metric_events,
    ).submit(_request())

    assert result.doc_id == "doc-1"
    assert result.chunks_count == 1
    assert result.entities_count == 2
    assert result.relations_count == 3
    record = catalog.records["doc-1"]
    assert record.ingest_status == DocumentIngestStatus.INGESTED.value
    assert record.metadata["_upload_batch_id"] == "batch-1"
    assert record.metadata["_upload_client_file_id"] == "file-1"
    assert record.metadata["_processing_step"] == "extract"
    assert audit_events == [("success", "user-a", "doc-1", "pdf")]
    assert webhook_events == [("tenant-a", "doc-1", 1)]
    assert metric_events == [("pdf", 1.5)]
    assert storage.discarded == []
    assert storage.deleted == []


@pytest.mark.asyncio
async def test_submission_rejection_discards_staging_and_records_denial(tmp_path: Path) -> None:
    storage = _storage(tmp_path)
    audit_events: list[tuple] = []

    with pytest.raises(DocumentSubmissionError) as captured:
        await _coordinator(
            storage,
            FakePolicy(UploadPolicyError("upload_unsupported_type", "文件类型不支持")),
            SuccessfulWorkflow(),
            FakeCatalog(),
            audit_events,
            [],
            [],
        ).submit(_request())

    assert captured.value.code == "upload_unsupported_type"
    assert storage.discarded == [storage.staged]
    assert audit_events == [("denied", "user-a", "upload_unsupported_type", 5)]


@pytest.mark.asyncio
async def test_submission_execution_failure_marks_catalog_failed_and_records_audit(tmp_path: Path) -> None:
    storage = _storage(tmp_path)
    catalog = FakeCatalog()
    audit_events: list[tuple] = []

    with pytest.raises(DocumentSubmissionError) as captured:
        await _coordinator(
            storage,
            FakePolicy(UploadInspection(parser_type="pdf", extension=".pdf", size_bytes=5)),
            FailingWorkflow(),
            catalog,
            audit_events,
            [],
            [],
        ).submit(_request())

    assert captured.value.code == "document_ingest_failed"
    record = catalog.records["doc-1"]
    assert record.ingest_status == DocumentIngestStatus.FAILED.value
    assert record.error_code == "document_ingest_failed"
    assert record.metadata["_ingest_stage_index"] == 3
    assert audit_events == [("failure", "user-a", "document_ingest_failed")]
    assert storage.deleted == []
