"""ATDD coverage for synchronized upload records and failed-document retry."""

from __future__ import annotations

from io import BytesIO
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient
from starlette.requests import Request
from starlette.responses import Response

from api.dependencies import get_current_user
from domain.documents import DepartmentRecord, DocumentIngestStatus, DocumentRecord
from domain.identity import Permission, UserContext, UserRole
from infrastructure.documents.catalog import SQLiteDocumentCatalogRepository
from infrastructure.documents.local_uploads import LocalUploadStorage


def _user(
    *,
    org_id: str = "tenant-a",
    permissions: list[Permission] | None = None,
) -> UserContext:
    """Build one authenticated tenant user for a retry acceptance scenario."""

    return UserContext(
        user_id="user-a",
        username="alice",
        role=UserRole.EDITOR,
        org_id=org_id,
        permissions=[Permission.DOC_READ, Permission.DOC_WRITE]
        if permissions is None
        else permissions,
    )


def _catalog(tmp_path: Path) -> SQLiteDocumentCatalogRepository:
    """Create an isolated document catalog with one tenant department."""

    catalog = SQLiteDocumentCatalogRepository(tmp_path / "catalog.sqlite3")
    catalog.upsert_department(
        DepartmentRecord(
            department_id="finance",
            tenant_id="tenant-a",
            company_id="tenant-a",
            name="财务部",
            normalized_key="finance",
        )
    )
    return catalog


def _accepted_record(
    tmp_path: Path,
    catalog: SQLiteDocumentCatalogRepository,
    *,
    doc_id: str = "doc-failed",
    status: str = DocumentIngestStatus.FAILED.value,
    legacy: bool = False,
) -> tuple[DocumentRecord, LocalUploadStorage]:
    """Persist one catalog record whose source is retained in accepted storage."""

    storage = LocalUploadStorage(str(tmp_path / "uploads"))
    staged = storage.stage(
        tenant_id="tenant-a",
        original_filename="预算制度.txt",
        source=BytesIO(b"safe"),
        max_file_bytes=1024,
        tenant_quota_bytes=4096,
    )
    accepted = storage.promote(
        staged,
        department_id="finance",
        doc_id=doc_id,
        original_filename="预算制度.txt",
    )
    record = DocumentRecord(
        doc_id=doc_id,
        tenant_id="tenant-a",
        company_id="tenant-a",
        department_id="finance",
        folder_path="finance",
        relative_path="finance/预算制度.txt",
        normalized_relative_path="finance/预算制度.txt",
        uploaded_filename="预算制度.txt",
        display_name="预算制度",
        provenance_source_filename="预算制度.txt",
        storage_reference=accepted,
        content_sha256="a" * 64,
        size_bytes=4,
        mime_type="text/plain",
        document_type="text",
        source_format="txt",
        ingest_status=status,
        version=1,
        created_by="user-a",
        created_at="2026-08-01T01:02:03+00:00",
        updated_at="2026-08-01T01:03:04+00:00",
        error_code="document_ingest_failed" if status == "failed" else "",
        legacy=legacy,
    )
    return catalog.create_document(record), storage


def _request(catalog, workflow, vector_items: list[dict] | None = None) -> Request:
    """Build a request carrying the document runtime dependencies."""

    app = FastAPI()
    app.state.document_catalog = catalog
    app.state.workflows = {"ingest": workflow}
    app.state.vector_store = SimpleNamespace(
        list_documents=AsyncMock(return_value=vector_items or [])
    )
    request = Request(
        {
            "type": "http",
            "method": "POST",
            "path": "/docs/doc-failed/retry",
            "headers": [],
            "app": app,
            "scheme": "http",
            "server": ("testserver", 80),
        }
    )
    request.state.tenant_id = "tenant-a"
    return request


@pytest.mark.asyncio
async def test_catalog_list_exposes_sync_fields_and_successful_retry_updates_same_record(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A durable failed row is retryable and keeps its identity on success."""

    from api.dependencies import documents as document_dependencies
    from api.routers import documents_read as documents

    catalog = _catalog(tmp_path)
    record, _storage = _accepted_record(tmp_path, catalog)
    workflow = SimpleNamespace(
        ainvoke=AsyncMock(
            return_value={
                "chunks": [SimpleNamespace(doc_type=SimpleNamespace(value="text"))],
                "entities_stored": 2,
                "relations_stored": 3,
            }
        )
    )
    audit = SimpleNamespace(log=Mock())
    webhook = SimpleNamespace(trigger=AsyncMock())
    monkeypatch.setattr(documents.settings, "upload_dir", str(tmp_path / "uploads"))
    monkeypatch.setattr(document_dependencies, "get_audit_service", lambda: audit)
    monkeypatch.setattr(document_dependencies, "get_webhook_service", lambda: webhook)
    request = _request(catalog, workflow)
    lifecycle = document_dependencies.get_document_lifecycle(request)

    listed = await documents.list_documents(request, _user())

    assert listed[0].created_at == "2026-08-01T01:02:03+00:00"
    assert listed[0].updated_at == "2026-08-01T01:03:04+00:00"
    assert listed[0].error_code == "document_ingest_failed"
    assert listed[0].retryable is True

    response = await documents.retry_document.__wrapped__(
        record.doc_id,
        request,
        Response(),
        _user(),
        lifecycle,
    )

    assert response.doc_id == record.doc_id
    assert response.status == "success"
    refreshed = catalog.get_document(record.doc_id, tenant_id="tenant-a")
    assert refreshed is not None
    assert refreshed.ingest_status == DocumentIngestStatus.INGESTED.value
    assert (refreshed.chunks_count, refreshed.entities_count, refreshed.relations_count) == (
        1,
        2,
        3,
    )
    payload = workflow.ainvoke.await_args.args[0]
    assert payload["document_inputs"][0].doc_id == record.doc_id
    assert payload["tenant_id"] == "tenant-a"
    audit.log.assert_called_once()
    webhook.trigger.assert_awaited_once()


@pytest.mark.asyncio
async def test_retry_workflow_failure_preserves_failed_state_and_safe_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A failed retry remains visible and never leaks the workflow exception."""

    from api.dependencies import documents as document_dependencies
    from api.routers import documents_read as documents

    catalog = _catalog(tmp_path)
    record, _storage = _accepted_record(tmp_path, catalog)
    workflow = SimpleNamespace(ainvoke=AsyncMock(side_effect=RuntimeError("secret detail")))
    audit = SimpleNamespace(log=Mock())
    monkeypatch.setattr(documents.settings, "upload_dir", str(tmp_path / "uploads"))
    monkeypatch.setattr(document_dependencies, "get_audit_service", lambda: audit)
    monkeypatch.setattr(document_dependencies, "get_webhook_service", lambda: SimpleNamespace(trigger=AsyncMock()))
    request = _request(catalog, workflow)
    lifecycle = document_dependencies.get_document_lifecycle(request)

    with pytest.raises(HTTPException) as exc_info:
        await documents.retry_document.__wrapped__(
            record.doc_id,
            request,
            Response(),
            _user(),
            lifecycle,
        )

    assert exc_info.value.status_code == 500
    assert exc_info.value.detail["code"] == "document_retry_failed"
    assert "secret detail" not in str(exc_info.value.detail)
    refreshed = catalog.get_document(record.doc_id, tenant_id="tenant-a")
    assert refreshed is not None
    assert refreshed.ingest_status == DocumentIngestStatus.FAILED.value
    assert refreshed.error_code == "document_retry_failed"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("status", "legacy"),
    [
        (DocumentIngestStatus.INGESTED.value, False),
        (DocumentIngestStatus.FAILED.value, True),
    ],
)
async def test_retry_rejects_owned_ineligible_records_without_ingestion(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    status: str,
    legacy: bool,
) -> None:
    """Successful and legacy catalog records cannot enter failed retry."""

    from api.dependencies import documents as document_dependencies
    from api.routers import documents_read as documents

    catalog = _catalog(tmp_path)
    record, _storage = _accepted_record(
        tmp_path,
        catalog,
        status=status,
        legacy=legacy,
    )
    workflow = SimpleNamespace(ainvoke=AsyncMock())
    monkeypatch.setattr(documents.settings, "upload_dir", str(tmp_path / "uploads"))
    request = _request(catalog, workflow)
    lifecycle = document_dependencies.get_document_lifecycle(request)

    with pytest.raises(HTTPException) as exc_info:
        await documents.retry_document.__wrapped__(
            record.doc_id,
            request,
            Response(),
            _user(),
            lifecycle,
        )

    assert exc_info.value.status_code == 409
    workflow.ainvoke.assert_not_awaited()


@pytest.mark.asyncio
async def test_retry_rejects_missing_source_and_conceals_cross_tenant_record(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Retry eligibility fails closed for missing files and other tenants."""

    from api.dependencies import documents as document_dependencies
    from api.routers import documents_read as documents

    catalog = _catalog(tmp_path)
    record, storage = _accepted_record(tmp_path, catalog)
    assert storage.delete(record.storage_reference, tenant_id="tenant-a") is True
    workflow = SimpleNamespace(ainvoke=AsyncMock())
    monkeypatch.setattr(documents.settings, "upload_dir", str(tmp_path / "uploads"))
    request = _request(catalog, workflow)
    lifecycle = document_dependencies.get_document_lifecycle(request)

    listed = await documents.list_documents(request, _user())
    assert listed[0].retryable is False

    with pytest.raises(HTTPException) as missing_source:
        await documents.retry_document.__wrapped__(
            record.doc_id,
            request,
            Response(),
            _user(),
            lifecycle,
        )
    assert missing_source.value.status_code == 409

    other_tenant_request = _request(catalog, workflow)
    other_tenant_request.state.tenant_id = "tenant-b"
    with pytest.raises(HTTPException) as hidden:
        await documents.retry_document.__wrapped__(
            record.doc_id,
            other_tenant_request,
            Response(),
            _user(org_id="tenant-b"),
            lifecycle,
        )
    assert hidden.value.status_code == 404
    workflow.ainvoke.assert_not_awaited()


def test_retry_route_requires_doc_write_before_handler_execution() -> None:
    """FastAPI rejects a read-only user at the existing RBAC boundary."""

    from api.routers import documents_read as documents

    app = FastAPI()
    app.include_router(documents.docs_router)
    app.dependency_overrides[get_current_user] = lambda: _user(
        permissions=[Permission.DOC_READ]
    )

    response = TestClient(app).post("/docs/doc-failed/retry")

    assert response.status_code == 403
