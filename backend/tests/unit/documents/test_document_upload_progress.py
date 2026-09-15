"""Acceptance coverage for tenant-scoped document upload stage progress."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock
from uuid import uuid4

import pytest

from api.routers.documents_ingest import upload_progress
from domain.documents import (
    DocumentIngestStatus,
    DocumentRecord,
    document_ingest_processing_step,
    document_ingest_stage,
)
from domain.identity import Permission, UserContext, UserRole


def _record(status: str, *, tenant_id: str = "tenant-a", metadata: dict | None = None) -> DocumentRecord:
    return DocumentRecord(
        doc_id=str(uuid4()),
        tenant_id=tenant_id,
        company_id=tenant_id,
        department_id="finance",
        folder_path="finance",
        relative_path="finance/report.txt",
        normalized_relative_path="finance/report.txt",
        uploaded_filename="report.txt",
        display_name="report",
        provenance_source_filename="report.txt",
        storage_reference="tenant-safe-reference",
        content_sha256="a" * 64,
        size_bytes=1,
        mime_type="text/plain",
        document_type="text",
        source_format="txt",
        ingest_status=status,
        version=1,
        metadata=metadata or {},
    )


def _user(permission: Permission = Permission.DOC_READ) -> UserContext:
    return UserContext(
        user_id="user-a",
        username="alice",
        role=UserRole.EDITOR,
        org_id="tenant-a",
        permissions=[permission],
    )


def test_document_ingest_stage_is_normalized_and_failure_keeps_last_stage() -> None:
    assert document_ingest_stage(DocumentIngestStatus.PROCESSING.value) == (
        "processing", 3, 4, "解析与知识抽取"
    )
    assert document_ingest_stage(
        DocumentIngestStatus.FAILED.value,
        {"_ingest_stage_index": 2},
    ) == ("failed", 2, 4, "处理失败")
    assert document_ingest_stage(DocumentIngestStatus.LEGACY.value) == (
        "ingested", 4, 4, "入库完成"
    )


def test_document_ingest_processing_step_is_normalized() -> None:
    assert document_ingest_processing_step({"_processing_step": "store_graph"}) == (
        "store_graph", 4, 6, "写知识图谱"
    )
    assert document_ingest_processing_step({"_processing_step": "unknown"}) == (
        "parse", 1, 6, "解析文件"
    )


@pytest.mark.asyncio
async def test_upload_progress_is_tenant_scoped_and_aggregates_conservatively() -> None:
    upload_id = str(uuid4())
    catalog = MagicMock()
    catalog.list_documents.return_value = [
        _record(
            DocumentIngestStatus.PROCESSING.value,
            metadata={"_upload_batch_id": upload_id, "_upload_client_file_id": "client-1"},
        ),
        _record(
            DocumentIngestStatus.INGESTED.value,
            metadata={"_upload_batch_id": upload_id, "_upload_client_file_id": "client-2"},
        ),
        _record(
            DocumentIngestStatus.INGESTED.value,
            tenant_id="tenant-b",
            metadata={"_upload_batch_id": upload_id, "_upload_client_file_id": "other"},
        ),
    ]
    request = SimpleNamespace(state=SimpleNamespace(tenant_id="tenant-a"), app=SimpleNamespace(state=SimpleNamespace(document_catalog=catalog)))

    response = await upload_progress(upload_id, request, total_count=2, user=_user())

    assert response.total_count == 2
    assert response.completed_count == 1
    assert response.failed_count == 0
    assert response.stage_index == 3
    assert response.status == "processing"
    assert response.terminal is False
    assert len(response.items) == 2
    assert {item.client_file_id for item in response.items} == {"client-1", "client-2"}


@pytest.mark.asyncio
async def test_upload_progress_reports_terminal_failure_without_storage_details() -> None:
    upload_id = str(uuid4())
    catalog = MagicMock()
    catalog.list_documents.return_value = [
        _record(
            DocumentIngestStatus.FAILED.value,
            metadata={"_upload_batch_id": upload_id, "_ingest_stage_index": 2},
        ),
    ]
    request = SimpleNamespace(state=SimpleNamespace(tenant_id="tenant-a"), app=SimpleNamespace(state=SimpleNamespace(document_catalog=catalog)))

    response = await upload_progress(upload_id, request, total_count=1, user=_user())

    assert response.terminal is True
    assert response.status == "failed"
    assert response.stage_index == 2
    assert response.items[0].error_code == ""
    assert response.items[0].message == ""
    assert "tenant-safe-reference" not in response.model_dump_json()


@pytest.mark.asyncio
async def test_upload_progress_reports_slowest_active_processing_substage() -> None:
    upload_id = str(uuid4())
    catalog = MagicMock()
    catalog.list_documents.return_value = [
        _record(
            DocumentIngestStatus.PROCESSING.value,
            metadata={
                "_upload_batch_id": upload_id,
                "_upload_client_file_id": "client-1",
                "_processing_step": "store_sparse",
            },
        ),
        _record(
            DocumentIngestStatus.PROCESSING.value,
            metadata={
                "_upload_batch_id": upload_id,
                "_upload_client_file_id": "client-2",
                "_processing_step": "store_vectors",
            },
        ),
    ]
    request = SimpleNamespace(state=SimpleNamespace(tenant_id="tenant-a"), app=SimpleNamespace(state=SimpleNamespace(document_catalog=catalog)))

    response = await upload_progress(upload_id, request, total_count=2, user=_user())

    assert response.processing_step == "store_vectors"
    assert response.processing_step_index == 3
    assert response.processing_step_label == "写向量库"
