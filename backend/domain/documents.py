from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from enum import Enum
from pathlib import PurePosixPath, PureWindowsPath
from typing import Any, Protocol, runtime_checkable

__all__ = [
    "CatalogConflictError",
    "CatalogNotFoundError",
    "DepartmentRecord",
    "DocType",
    "DocumentCatalogRepository",
    "DocumentChunk",
    "DocumentIngestStatus",
    "DocumentIngestProcessingStep",
    "DocumentIngestStage",
    "document_ingest_processing_step",
    "document_ingest_stage",
    "DocumentRecord",
    "FolderManifestEntry",
    "FolderRecord",
    "FolderUploadManifest",
    "IngestDocumentInput",
    "normalize_relative_path",
    "normalize_uploaded_filename",
    "utc_now_iso",
]


def utc_now_iso() -> str:
    """Return a stable UTC timestamp for persisted catalog metadata."""

    return datetime.now(UTC).isoformat()


class CatalogConflictError(ValueError):
    """Raised when a catalog uniqueness rule rejects a write."""


class CatalogNotFoundError(LookupError):
    """Raised when a tenant-owned catalog record does not exist."""


class DocType(str, Enum):
    """Represent a parsed document type."""

    PDF = "pdf"
    WORD = "word"
    IMAGE = "image"
    TABLE = "table"
    TEXT = "text"
    MARKDOWN = "markdown"
    UNKNOWN = "unknown"


class DocumentIngestStatus(str, Enum):
    """Represent the durable catalog lifecycle for one document."""

    STAGING = "staging"
    ACCEPTED = "accepted"
    PROCESSING = "processing"
    INGESTED = "ingested"
    FAILED = "failed"
    DELETED = "deleted"
    LEGACY = "legacy"


class DocumentIngestStage(str, Enum):
    """Represent the four user-visible document-ingestion milestones."""

    STAGING = "staging"
    ACCEPTED = "accepted"
    PROCESSING = "processing"
    INGESTED = "ingested"
    FAILED = "failed"


class DocumentIngestProcessingStep(str, Enum):
    """Represent internal progress substages for long-running ingestion work."""

    PARSE = "parse"
    EXTRACT = "extract"
    STORE_VECTORS = "store_vectors"
    STORE_GRAPH = "store_graph"
    STORE_SPARSE = "store_sparse"
    ADVANCE_REVISION = "advance_revision"
    FINALIZE = "finalize"


_INGEST_STAGE_INDEX = {
    DocumentIngestStage.STAGING.value: 1,
    DocumentIngestStage.ACCEPTED.value: 2,
    DocumentIngestStage.PROCESSING.value: 3,
    DocumentIngestStage.INGESTED.value: 4,
    DocumentIngestStage.FAILED.value: 4,
}
_INGEST_STAGE_LABELS = {
    DocumentIngestStage.STAGING.value: "上传校验与落盘",
    DocumentIngestStage.ACCEPTED.value: "文件已接收",
    DocumentIngestStage.PROCESSING.value: "解析与知识抽取",
    DocumentIngestStage.INGESTED.value: "入库完成",
    DocumentIngestStage.FAILED.value: "处理失败",
}


def document_ingest_stage(status: str, metadata: dict[str, Any] | None = None) -> tuple[str, int, int, str]:
    """Normalize a durable ingest status into the public four-stage contract."""

    if status == DocumentIngestStatus.LEGACY.value:
        normalized = DocumentIngestStage.INGESTED.value
    else:
        normalized = status if status in _INGEST_STAGE_INDEX else DocumentIngestStage.STAGING.value
    stage_index = _INGEST_STAGE_INDEX.get(normalized, 1)
    if normalized == DocumentIngestStage.FAILED.value:
        candidate = (metadata or {}).get("_ingest_stage_index")
        if isinstance(candidate, int) and 1 <= candidate <= 4:
            stage_index = candidate
    return normalized, stage_index, 4, _INGEST_STAGE_LABELS[normalized]



_PROCESSING_STEP_TOTAL = 6
_PROCESSING_STEP_LABELS = {
    "parse": "解析文件",
    "extract": "知识抽取",
    "store_vectors": "写向量库",
    "store_graph": "写知识图谱",
    "store_sparse": "写稀疏索引",
    "advance_revision": "收尾提交",
    "finalize": "收尾提交",
}
_PROCESSING_STEP_INDEX = {
    "parse": 1,
    "extract": 2,
    "store_vectors": 3,
    "store_graph": 4,
    "store_sparse": 5,
    "advance_revision": 6,
    "finalize": 6,
}


def document_ingest_processing_step(metadata: dict[str, Any] | None = None) -> tuple[str, int, int, str]:
    """Normalize the current internal processing substage for long-running ingest work."""

    candidate = str((metadata or {}).get("_processing_step", "")).strip() or "parse"
    normalized = candidate if candidate in _PROCESSING_STEP_INDEX else "parse"
    return normalized, _PROCESSING_STEP_INDEX[normalized], _PROCESSING_STEP_TOTAL, _PROCESSING_STEP_LABELS[normalized]


_CONTROL_CHARACTER = re.compile(r"[\x00-\x1f\x7f]")


def normalize_uploaded_filename(filename: str, *, max_length: int = 255) -> str:
    """Normalize and validate a client filename for final-basename use only."""

    normalized = unicodedata.normalize("NFC", (filename or "").strip())
    if (
        not normalized
        or normalized in {".", ".."}
        or len(normalized) > max_length
        or "/" in normalized
        or "\\" in normalized
        or _CONTROL_CHARACTER.search(normalized)
    ):
        raise ValueError("invalid uploaded filename")
    return normalized


def normalize_relative_path(
    relative_path: str,
    *,
    max_depth: int = 16,
    max_length: int = 1024,
    max_basename_length: int = 255,
) -> str:
    """Return an NFC POSIX logical path while rejecting traversal and ambiguity."""

    raw = unicodedata.normalize("NFC", (relative_path or "").strip()).replace("\\", "/")
    if not raw or len(raw) > max_length or _CONTROL_CHARACTER.search(raw):
        raise ValueError("invalid relative path")
    if raw.startswith("/") or PureWindowsPath(raw).is_absolute() or PureWindowsPath(raw).drive:
        raise ValueError("absolute relative path is not allowed")

    parts = raw.split("/")
    if any(part in {"", ".", ".."} for part in parts):
        raise ValueError("relative path contains an unsafe segment")
    if len(parts) > max_depth:
        raise ValueError("relative path exceeds maximum depth")
    normalize_uploaded_filename(parts[-1], max_length=max_basename_length)
    normalized = PurePosixPath(*parts).as_posix()
    if normalized == "." or normalized.startswith("../"):
        raise ValueError("relative path is unsafe")
    return normalized


@dataclass(frozen=True)
class DepartmentRecord:
    """Describe one tenant-owned department used for catalog and graph scope."""

    department_id: str
    tenant_id: str
    company_id: str
    name: str
    normalized_key: str
    status: str = "active"


@dataclass(frozen=True)
class FolderRecord:
    """Describe one logical folder without mapping it to a physical directory."""

    folder_id: str
    tenant_id: str
    department_id: str
    parent_folder_id: str | None
    name: str
    normalized_path: str


@dataclass(frozen=True)
class DocumentRecord:
    """Persist stable document identity and provenance without file content."""

    doc_id: str
    tenant_id: str
    company_id: str
    department_id: str
    folder_path: str
    relative_path: str
    normalized_relative_path: str
    uploaded_filename: str
    display_name: str
    provenance_source_filename: str
    storage_reference: str
    content_sha256: str
    size_bytes: int
    mime_type: str
    document_type: str
    source_format: str
    ingest_status: str
    version: int
    authority: str = ""
    review_status: str = ""
    sensitivity: str = ""
    external_source_id: str = ""
    created_by: str = ""
    created_at: str = field(default_factory=utc_now_iso)
    updated_at: str = field(default_factory=utc_now_iso)
    chunks_count: int = 0
    entities_count: int = 0
    relations_count: int = 0
    error_code: str = ""
    legacy: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    def with_updates(self, **changes: Any) -> "DocumentRecord":
        """Return a replacement record with a refreshed update timestamp."""

        return replace(self, updated_at=utc_now_iso(), **changes)


@dataclass(frozen=True)
class FolderManifestEntry:
    """Describe one logical file entry supplied by a folder-upload manifest."""

    client_file_id: str
    relative_path: str
    department_id: str
    external_source_id: str = ""
    display_name: str = ""
    provenance_source_filename: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def normalized_relative_path(self) -> str:
        """Return the validated normalized path used for duplicate detection."""

        return normalize_relative_path(self.relative_path)


@dataclass(frozen=True)
class FolderUploadManifest:
    """Represent the logical context accompanying a multipart folder upload."""

    root_folder_name: str
    company_id: str
    department_mappings: dict[str, str]
    files: tuple[FolderManifestEntry, ...]


@dataclass(frozen=True)
class IngestDocumentInput:
    """Carry stable per-document context through synchronous or queued ingestion."""

    doc_id: str
    tenant_id: str
    company_id: str
    department_id: str
    file_path: str
    uploaded_filename: str
    display_name: str
    provenance_source_filename: str
    relative_path: str
    folder_path: str
    content_sha256: str
    version: int = 1
    authority: str = ""
    review_status: str = ""
    sensitivity: str = ""
    external_source_id: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def chunk_metadata(self) -> dict[str, Any]:
        """Build the catalog metadata that every persisted chunk must retain."""

        return {
            "doc_id": self.doc_id,
            "tenant_id": self.tenant_id,
            "company_id": self.company_id,
            "department_id": self.department_id,
            "relative_path": self.relative_path,
            "folder_path": self.folder_path,
            "uploaded_filename": self.uploaded_filename,
            "display_name": self.display_name,
            "provenance_source_filename": self.provenance_source_filename,
            "content_sha256": self.content_sha256,
            "version": self.version,
            "authority": self.authority,
            "review_status": self.review_status,
            "sensitivity": self.sensitivity,
            "external_source_id": self.external_source_id,
            **self.metadata,
        }


@dataclass
class DocumentChunk:
    """A document segment with its source and tenant metadata."""

    content: str
    doc_id: str
    chunk_index: int
    doc_type: DocType
    metadata: dict[str, Any] = field(default_factory=dict)
    embedding: list[float] | None = None
    tenant_id: str = ""

    @property
    def chunk_id(self) -> str:
        """Return the stable chunk identifier within one document."""

        return f"{self.doc_id}#chunk-{self.chunk_index}"


@runtime_checkable
class DocumentCatalogRepository(Protocol):
    """Define durable, tenant-filtered document catalog persistence."""

    def upsert_department(self, record: DepartmentRecord) -> DepartmentRecord:
        """Create or update a tenant-owned department."""

    def get_department(
        self,
        department_id: str,
        *,
        tenant_id: str,
        company_id: str | None = None,
    ) -> DepartmentRecord | None:
        """Return an authorized department or no result."""

    def list_departments(self, *, tenant_id: str, company_id: str | None = None) -> list[DepartmentRecord]:
        """List departments visible inside the supplied tenant/company scope."""

    def create_document(self, record: DocumentRecord) -> DocumentRecord:
        """Persist a new metadata-only catalog record."""

    def get_document(self, doc_id: str, *, tenant_id: str) -> DocumentRecord | None:
        """Return one tenant-owned document record."""

    def list_documents(
        self,
        *,
        tenant_id: str,
        company_id: str | None = None,
        department_id: str | None = None,
        include_deleted: bool = False,
    ) -> list[DocumentRecord]:
        """List catalog records filtered by tenant and optional organization scope."""

    def update_document(self, record: DocumentRecord) -> DocumentRecord:
        """Replace a tenant-owned record while preserving uniqueness rules."""

    def update_document_if_status(
        self,
        record: DocumentRecord,
        *,
        allowed_statuses: set[str],
    ) -> bool:
        """Replace one exact generation only while its current status remains allowed."""

    def remove_document(self, doc_id: str, *, tenant_id: str) -> bool:
        """Remove one incomplete catalog record during compensation."""
