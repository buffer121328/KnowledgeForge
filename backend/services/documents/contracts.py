"""Explicit contracts for synchronous and folder document submission workflows."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Awaitable, BinaryIO, Callable, Protocol

from domain.documents import DepartmentRecord, DocumentRecord, IngestDocumentInput


class UploadStorage(Protocol):
    """Tenant-safe storage operations required for one document submission."""

    def stage(
        self,
        *,
        tenant_id: str,
        original_filename: str,
        source: BinaryIO,
        max_file_bytes: int,
        tenant_quota_bytes: int,
        chunk_bytes: int,
    ) -> object: ...

    def promote(
        self,
        staged: object,
        *,
        department_id: str | None = None,
        doc_id: str | None = None,
        original_filename: str | None = None,
    ) -> str: ...

    def quarantine(self, staged: object) -> str: ...

    def discard(self, staged_or_path: object | str) -> bool: ...

    def delete(self, path: str, tenant_id: str | None = None) -> bool: ...


@dataclass(frozen=True)
class UploadInspection:
    """Validated file metadata kept independent from its infrastructure adapter."""

    parser_type: str
    extension: str
    size_bytes: int


class UploadPolicy(Protocol):
    """Content validation performed after tenant-safe staging."""

    def validate(self, file_path: str, filename: str, content_type: str | None) -> UploadInspection: ...


class DocumentCatalog(Protocol):
    """Catalog lifecycle operations used by submission workflows."""

    def upsert_department(self, department: DepartmentRecord) -> object: ...

    def create_document(self, record: DocumentRecord) -> object: ...

    def update_document(self, record: DocumentRecord) -> object: ...

    def get_document(self, doc_id: str, *, tenant_id: str) -> DocumentRecord | None: ...

    def remove_document(self, doc_id: str, *, tenant_id: str) -> object: ...


class IngestWorkflow(Protocol):
    """Existing async workflow facade used by synchronous HTTP ingestion."""

    async def ainvoke(self, payload: dict[str, Any]) -> dict[str, Any]: ...


@dataclass(frozen=True)
class DocumentSubmissionSettings:
    """Settings-derived submission limits kept explicit for tests and providers."""

    max_file_bytes: int
    tenant_quota_bytes: int
    stream_chunk_bytes: int
    quarantine_retention_hours: int


@dataclass(frozen=True)
class DocumentSubmissionRequest:
    """Trusted actor context plus one already-extracted multipart file stream."""

    tenant_id: str
    user_id: str
    username: str
    org_id: str
    source: BinaryIO
    original_filename: str
    content_type: str | None
    upload_id: str = ""
    client_file_id: str = ""


@dataclass(frozen=True)
class DocumentSubmissionDenialRequest:
    """Trusted actor context for an upload denial raised at the HTTP boundary."""

    tenant_id: str
    user_id: str
    username: str
    org_id: str


@dataclass(frozen=True)
class DocumentSubmissionResult:
    """HTTP-neutral result that the Router serializes using its existing schema."""

    file_name: str
    chunks_count: int
    entities_count: int
    relations_count: int
    doc_id: str
    client_file_id: str = ""


@dataclass(frozen=True)
class FolderSubmissionRequest:
    """One prevalidated folder member prepared for asynchronous ingestion."""

    tenant_id: str
    user_id: str
    username: str
    org_id: str
    source: BinaryIO
    original_filename: str
    content_type: str | None
    company_id: str
    department_id: str
    relative_path: str
    normalized_relative_path: str
    metadata: dict[str, Any]
    display_name: str = ""
    provenance_source_filename: str = ""
    external_source_id: str = ""
    upload_id: str = ""
    client_file_id: str = ""


@dataclass(frozen=True)
class FolderSubmissionResult:
    """A prepared catalog record or one safe per-member rejection result."""

    record: DocumentRecord | None
    document_input: IngestDocumentInput | None
    file_name: str
    client_file_id: str
    relative_path: str
    department_id: str
    display_name: str
    provenance_source_filename: str
    error_code: str = ""
    message: str = ""

    @property
    def accepted(self) -> bool:
        """Return whether staging and catalog promotion completed."""

        return self.record is not None and self.document_input is not None


class DocumentSubmissionError(RuntimeError):
    """Safe coordination failure mapped to the existing HTTP error envelope."""

    def __init__(self, status_code: int, code: str, message: str) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.code = code
        self.message = message


AuditDenial = Callable[[DocumentSubmissionRequest | FolderSubmissionRequest, str, int], None]
UploadDenialAudit = Callable[[DocumentSubmissionDenialRequest, str, int, dict[str, Any]], None]
AuditFailure = Callable[[DocumentSubmissionRequest, str], None]
AuditSuccess = Callable[[DocumentSubmissionRequest, DocumentSubmissionResult, UploadInspection], None]
WebhookTrigger = Callable[[DocumentSubmissionRequest, DocumentSubmissionResult], Awaitable[None]]
MetricsRecorder = Callable[[str, float], None]


@dataclass(frozen=True)
class DocumentSubmissionDependencies:
    """Explicit collaborators for staging, ingestion, and observable side effects."""

    storage_factory: Callable[[], UploadStorage]
    upload_policy: UploadPolicy
    catalog: DocumentCatalog | None
    ingest_workflow: IngestWorkflow | None
    settings: DocumentSubmissionSettings
    audit_denial: AuditDenial
    audit_failure: AuditFailure
    audit_success: AuditSuccess
    trigger_webhook: WebhookTrigger
    record_metrics: MetricsRecorder
    allocate_document_id: Callable[[], str]
    audit_upload_denial: UploadDenialAudit | None = None
    clock: Callable[[], float] = time.time


__all__ = [
    "AuditDenial",
    "AuditFailure",
    "AuditSuccess",
    "DocumentCatalog",
    "DocumentSubmissionDenialRequest",
    "DocumentSubmissionDependencies",
    "DocumentSubmissionError",
    "DocumentSubmissionRequest",
    "DocumentSubmissionResult",
    "DocumentSubmissionSettings",
    "FolderSubmissionRequest",
    "FolderSubmissionResult",
    "IngestWorkflow",
    "MetricsRecorder",
    "UploadDenialAudit",
    "UploadInspection",
    "UploadPolicy",
    "UploadStorage",
    "WebhookTrigger",
]
