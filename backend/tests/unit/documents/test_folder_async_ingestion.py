"""ATDD coverage for bounded asynchronous logical-folder ingestion."""

from __future__ import annotations

from io import BytesIO
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import HTTPException, Response
from starlette.datastructures import Headers, UploadFile

from api.schemas import FolderManifestFileRequest, FolderUploadManifestRequest, IngestResponse
from domain.documents import DepartmentRecord, DocumentIngestStatus, DocumentRecord, IngestDocumentInput
from domain.identity import Permission, UserContext, UserRole


def _record(status: str = DocumentIngestStatus.ACCEPTED.value) -> DocumentRecord:
    return DocumentRecord(
        doc_id="doc-folder",
        tenant_id="tenant-a",
        company_id="tenant-a",
        department_id="finance",
        relative_path="finance/a.txt",
        normalized_relative_path="finance/a.txt",
        folder_path="finance",
        uploaded_filename="a.txt",
        provenance_source_filename="a.txt",
        version=1,
        display_name="A",
        storage_reference="/uploads/a.txt",
        content_sha256="abc",
        size_bytes=1,
        mime_type="text/plain",
        document_type="text",
        source_format="txt",
        ingest_status=status,
        created_by="user-a",
    )


def _prepared():
    from api.routers.documents_ingest import PreparedFolderMember

    record = _record()
    return PreparedFolderMember(
        record=record,
        document_input=IngestDocumentInput(
            doc_id=record.doc_id,
            tenant_id=record.tenant_id,
            company_id=record.company_id,
            department_id=record.department_id,
            file_path=record.storage_reference,
            uploaded_filename=record.uploaded_filename,
            display_name=record.display_name,
            provenance_source_filename=record.provenance_source_filename,
            relative_path=record.relative_path,
            folder_path=record.folder_path,
            content_sha256=record.content_sha256,
        ),
        response=IngestResponse(
            file_name="a.txt",
            chunks_count=0,
            entities_count=0,
            relations_count=0,
            status="accepted",
            doc_id=record.doc_id,
            client_file_id="client-a",
            relative_path=record.relative_path,
            department_id=record.department_id,
        ),
    )


def _manifest() -> FolderUploadManifestRequest:
    return FolderUploadManifestRequest(
        root_folder_name="root",
        company_id="tenant-a",
        department_mappings={"finance": "finance"},
        files=[
            FolderManifestFileRequest(
                client_file_id="client-a",
                relative_path="finance/a.txt",
                department_id="finance",
            )
        ],
    )


def _user() -> UserContext:
    return UserContext(
        user_id="user-a",
        username="alice",
        role=UserRole.EDITOR,
        org_id="tenant-a",
        permissions=[Permission.DOC_WRITE],
    )


@pytest.mark.asyncio
async def test_folder_upload_returns_202_processing_and_publishes_durable_task(monkeypatch) -> None:
    from api.routers import documents_ingest as router

    upload = UploadFile(file=BytesIO(b"a"), filename="a.txt", headers=Headers({"content-type": "text/plain"}))
    catalog = MagicMock()
    catalog.update_document_if_status.return_value = True
    registry = MagicMock()
    prepared = _prepared()
    monkeypatch.setattr(router, "_validate_folder_manifest", lambda *_args, **_kwargs: [(upload, _manifest().files[0], "finance/a.txt")])
    monkeypatch.setattr(router, "_ensure_manifest_departments", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(router, "_prepare_folder_member", AsyncMock(return_value=prepared))
    from services.documents.lifecycle import DocumentLifecycleCoordinator, DocumentLifecycleDependencies

    publish = MagicMock()
    lifecycle = DocumentLifecycleCoordinator(
        DocumentLifecycleDependencies(
            catalog=catalog,
            task_registry_factory=lambda: registry,
            new_task_id=lambda _org: "tsk-1",
            publish_folder_task=lambda **kwargs: publish(**kwargs),
        )
    )
    request = SimpleNamespace(state=SimpleNamespace(tenant_id="tenant-a"), app=SimpleNamespace(state=SimpleNamespace()))
    response = Response()

    results = await router.upload_folder(
        request,
        response,
        files=[upload],
        manifest=_manifest().model_dump_json(),
        user=_user(),
        lifecycle=lifecycle,
    )

    assert response.status_code == 202
    assert results[0].status == "processing"
    assert results[0].task_id == "tsk-1"
    registry.reserve.assert_called_once()
    publish.assert_called_once()
    assert publish.call_args.kwargs["task_id"] == "tsk-1"


@pytest.mark.asyncio
async def test_folder_publication_failure_marks_prepared_record_failed(monkeypatch) -> None:
    from api.routers import documents_ingest as router

    upload = UploadFile(file=BytesIO(b"a"), filename="a.txt", headers=Headers({"content-type": "text/plain"}))
    catalog = MagicMock()
    catalog.update_document_if_status.return_value = True
    registry = MagicMock()
    prepared = _prepared()
    monkeypatch.setattr(router, "_validate_folder_manifest", lambda *_args, **_kwargs: [(upload, _manifest().files[0], "finance/a.txt")])
    monkeypatch.setattr(router, "_ensure_manifest_departments", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(router, "_prepare_folder_member", AsyncMock(return_value=prepared))
    from services.documents.lifecycle import DocumentLifecycleCoordinator, DocumentLifecycleDependencies

    publish = MagicMock(side_effect=ConnectionError("broker"))
    lifecycle = DocumentLifecycleCoordinator(
        DocumentLifecycleDependencies(
            catalog=catalog,
            task_registry_factory=lambda: registry,
            new_task_id=lambda _org: "tsk-1",
            publish_folder_task=lambda **kwargs: publish(**kwargs),
        )
    )
    request = SimpleNamespace(state=SimpleNamespace(tenant_id="tenant-a"), app=SimpleNamespace(state=SimpleNamespace()))

    with pytest.raises(HTTPException) as captured:
        await router.upload_folder(
            request,
            Response(),
            files=[upload],
            manifest=_manifest().model_dump_json(),
            user=_user(),
            lifecycle=lifecycle,
        )

    assert captured.value.status_code == 503
    assert captured.value.detail["code"] == "folder_ingest_publish_failed"
    failed_record = catalog.update_document_if_status.call_args_list[-1].args[0]
    assert failed_record.ingest_status == DocumentIngestStatus.FAILED.value
    registry.update_state.assert_called_with("tsk-1", "publish_failed")


def test_catalog_guard_refuses_late_completion_after_delete(tmp_path) -> None:
    from infrastructure.documents.catalog import SQLiteDocumentCatalogRepository

    catalog = SQLiteDocumentCatalogRepository(tmp_path / "catalog.sqlite3")
    catalog.upsert_department(
        DepartmentRecord(
            department_id="finance",
            tenant_id="tenant-a",
            company_id="tenant-a",
            name="Finance",
            normalized_key="finance",
        )
    )
    processing = _record(DocumentIngestStatus.PROCESSING.value)
    catalog.create_document(processing)
    deleted = processing.with_updates(ingest_status=DocumentIngestStatus.DELETED.value)
    catalog.update_document(deleted)

    accepted = catalog.update_document_if_status(
        processing.with_updates(ingest_status=DocumentIngestStatus.INGESTED.value, chunks_count=3),
        allowed_statuses={DocumentIngestStatus.ACCEPTED.value, DocumentIngestStatus.PROCESSING.value},
    )

    assert accepted is False
    current = catalog.get_document(processing.doc_id, tenant_id=processing.tenant_id)
    assert current is not None
    assert current.ingest_status == DocumentIngestStatus.DELETED.value
    assert current.chunks_count == 0

@pytest.mark.asyncio
async def test_worker_late_success_after_delete_keeps_deleted_and_cleans_artifacts() -> None:
    from dataclasses import asdict

    from infrastructure.tasks.celery_ingest_tasks import _finalize_document_success

    deleted = _record(DocumentIngestStatus.DELETED.value)
    catalog = MagicMock()
    catalog.get_document.return_value = deleted
    catalog.update_document_if_status.return_value = False
    cleanup = AsyncMock()

    accepted = await _finalize_document_success(
        catalog,
        asdict(_prepared().document_input),
        {"chunks_count": 3, "entities_count": 2, "relations_count": 1},
        cleanup=cleanup,
    )

    assert accepted is False
    cleanup.assert_awaited_once_with("doc-folder", "tenant-a")
    target = catalog.update_document_if_status.call_args.args[0]
    assert target.ingest_status == DocumentIngestStatus.INGESTED.value


@pytest.mark.asyncio
async def test_duplicate_worker_success_does_not_delete_committed_artifacts() -> None:
    from dataclasses import asdict

    from infrastructure.tasks.celery_ingest_tasks import _finalize_document_success

    ingested = _record(DocumentIngestStatus.INGESTED.value).with_updates(chunks_count=3)
    catalog = MagicMock()
    catalog.get_document.return_value = ingested
    catalog.update_document_if_status.return_value = False
    cleanup = AsyncMock()

    accepted = await _finalize_document_success(
        catalog,
        asdict(_prepared().document_input),
        {"chunks_count": 3, "entities_count": 2, "relations_count": 1},
        cleanup=cleanup,
    )

    assert accepted is False
    cleanup.assert_not_awaited()


@pytest.mark.asyncio
async def test_worker_failure_marks_only_active_generation_failed_and_cleans_partial_artifacts() -> None:
    from dataclasses import asdict

    from infrastructure.tasks.celery_ingest_tasks import _finalize_document_failure

    processing = _record(DocumentIngestStatus.PROCESSING.value)
    catalog = MagicMock()
    catalog.get_document.return_value = processing
    catalog.update_document_if_status.return_value = True
    cleanup = AsyncMock()

    accepted = await _finalize_document_failure(
        catalog,
        asdict(_prepared().document_input),
        cleanup=cleanup,
    )

    assert accepted is True
    cleanup.assert_awaited_once_with("doc-folder", "tenant-a")
    target = catalog.update_document_if_status.call_args.args[0]
    assert target.ingest_status == DocumentIngestStatus.FAILED.value
    assert target.error_code == "task_execution_failed"


@pytest.mark.asyncio
async def test_interrupted_member_recovery_cleans_only_the_active_generation() -> None:
    from services.documents.lifecycle import DocumentLifecycleCoordinator, DocumentLifecycleDependencies

    processing = _record(DocumentIngestStatus.PROCESSING.value)
    catalog = MagicMock()
    catalog.get_document.return_value = processing
    catalog.update_document_if_status.return_value = True
    cleanup = AsyncMock()
    lifecycle = DocumentLifecycleCoordinator(
        DocumentLifecycleDependencies(catalog=catalog, cleanup_artifacts=cleanup)
    )

    recovered = await lifecycle.recover_interrupted_document(
        _prepared().document_input,
        cleanup_partial_artifacts=True,
    )

    assert recovered is True
    cleanup.assert_awaited_once_with("doc-folder", "tenant-a")
    target = catalog.update_document_if_status.call_args.args[0]
    assert target.ingest_status == DocumentIngestStatus.FAILED.value


@pytest.mark.asyncio
async def test_unstarted_member_recovery_never_cleans_artifacts() -> None:
    from services.documents.lifecycle import DocumentLifecycleCoordinator, DocumentLifecycleDependencies

    processing = _record(DocumentIngestStatus.PROCESSING.value)
    catalog = MagicMock()
    catalog.get_document.return_value = processing
    catalog.update_document_if_status.return_value = True
    cleanup = AsyncMock()
    lifecycle = DocumentLifecycleCoordinator(
        DocumentLifecycleDependencies(catalog=catalog, cleanup_artifacts=cleanup)
    )

    recovered = await lifecycle.recover_interrupted_document(
        _prepared().document_input,
        cleanup_partial_artifacts=False,
    )

    assert recovered is True
    cleanup.assert_not_awaited()
    target = catalog.update_document_if_status.call_args.args[0]
    assert target.ingest_status == DocumentIngestStatus.FAILED.value
