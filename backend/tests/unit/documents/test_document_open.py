"""ATDD coverage for tenant-scoped original document opening."""

from __future__ import annotations

from io import BytesIO
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient
from starlette.requests import Request

from domain.documents import DocumentRecord
from domain.identity import Permission, UserContext, UserRole
from infrastructure.documents.local_uploads import LocalUploadStorage


def _user(org_id: str = "tenant-a") -> UserContext:
    """Build an authenticated document reader."""

    return UserContext(
        user_id="user-a",
        username="alice",
        role=UserRole.VIEWER,
        org_id=org_id,
        permissions=[Permission.DOC_READ],
    )


def _request(catalog, tenant_id: str = "tenant-a") -> Request:
    """Build one document route request with an isolated runtime catalog."""

    app = FastAPI()
    app.state.document_catalog = catalog
    request = Request(
        {
            "type": "http",
            "method": "GET",
            "path": "/docs/doc-a/file",
            "headers": [],
            "app": app,
            "scheme": "http",
            "server": ("testserver", 80),
        }
    )
    request.state.tenant_id = tenant_id
    return request


def _record(storage_reference: str) -> DocumentRecord:
    """Build one catalog record for an accepted text document."""

    return DocumentRecord(
        doc_id="doc-a",
        tenant_id="tenant-a",
        company_id="tenant-a",
        department_id="finance",
        folder_path="finance",
        relative_path="finance/预算制度.txt",
        normalized_relative_path="finance/预算制度.txt",
        uploaded_filename="预算制度.txt",
        display_name="预算制度",
        provenance_source_filename="预算制度.txt",
        storage_reference=storage_reference,
        content_sha256="a" * 64,
        size_bytes=4,
        mime_type="text/plain",
        document_type="text",
        source_format="txt",
        ingest_status="ingested",
        version=1,
    )


@pytest.mark.asyncio
async def test_open_document_returns_inline_tenant_owned_source(tmp_path: Path, monkeypatch) -> None:
    """A tenant reader receives only the catalog-owned accepted source file."""

    from api.routers import documents_read as documents

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
        doc_id="doc-a",
        original_filename="预算制度.txt",
    )
    catalog = SimpleNamespace(get_document=Mock(return_value=_record(accepted)))
    monkeypatch.setattr(documents.settings, "upload_dir", str(tmp_path / "uploads"))

    response = await documents.open_document_file("doc-a", _request(catalog), _user())

    assert Path(response.path) == Path(accepted)
    assert response.media_type == "text/plain"
    assert response.headers["content-disposition"].startswith("inline;")
    catalog.get_document.assert_called_once_with("doc-a", tenant_id="tenant-a")


def test_document_file_route_injects_the_explicit_read_provider(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """The mounted route resolves source files through its dependency provider."""

    from api.dependencies import get_current_user
    from api.routers import documents_read as documents

    storage = LocalUploadStorage(str(tmp_path / "uploads"))
    staged = storage.stage(
        tenant_id="tenant-a",
        original_filename="policy.txt",
        source=BytesIO(b"safe"),
        max_file_bytes=1024,
        tenant_quota_bytes=4096,
    )
    accepted = storage.promote(
        staged,
        department_id="finance",
        doc_id="doc-a",
        original_filename="policy.txt",
    )
    catalog = SimpleNamespace(get_document=Mock(return_value=_record(accepted)))
    app = FastAPI()
    app.state.document_catalog = catalog
    app.state.vector_store = SimpleNamespace()
    app.include_router(documents.docs_router)
    app.dependency_overrides[get_current_user] = _user
    monkeypatch.setattr(documents.settings, "upload_dir", str(tmp_path / "uploads"))

    response = TestClient(app).get("/docs/doc-a/file")

    assert response.status_code == 200
    assert response.content == b"safe"
    assert response.headers["content-type"].startswith("text/plain")
    catalog.get_document.assert_called_once_with("doc-a", tenant_id="tenant-a")


@pytest.mark.asyncio
async def test_open_document_does_not_reveal_cross_tenant_record(tmp_path: Path, monkeypatch) -> None:
    """A document ID from another tenant is indistinguishable from a missing document."""

    from api.routers import documents_read as documents

    catalog = SimpleNamespace(get_document=Mock(return_value=None))
    monkeypatch.setattr(documents.settings, "upload_dir", str(tmp_path / "uploads"))

    with pytest.raises(HTTPException) as captured:
        await documents.open_document_file(
            "doc-a",
            _request(catalog, tenant_id="tenant-b"),
            _user("tenant-b"),
        )

    assert captured.value.status_code == 404
    assert captured.value.detail["code"] == "document_not_found"
    catalog.get_document.assert_called_once_with("doc-a", tenant_id="tenant-b")


@pytest.mark.asyncio
async def test_open_document_reports_missing_source_without_server_path(tmp_path: Path, monkeypatch) -> None:
    """A stale catalog source returns a bounded error without leaking storage paths."""

    from api.routers import documents_read as documents

    missing = tmp_path / "uploads" / "missing-secret-path.txt"
    catalog = SimpleNamespace(get_document=Mock(return_value=_record(str(missing))))
    monkeypatch.setattr(documents.settings, "upload_dir", str(tmp_path / "uploads"))

    with pytest.raises(HTTPException) as captured:
        await documents.open_document_file("doc-a", _request(catalog), _user())

    assert captured.value.status_code == 409
    assert captured.value.detail["code"] == "document_source_unavailable"
    assert str(missing) not in captured.value.detail["message"]
