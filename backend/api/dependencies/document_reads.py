"""Explicit read-side providers for tenant-scoped document browsing."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from fastapi import Request

from domain.documents import DepartmentRecord, DocumentRecord
from infrastructure.documents.local_uploads import LocalUploadStorage, UploadPolicyError
from shared.config import settings


class DocumentReadError(RuntimeError):
    """Represent a bounded failure while resolving an authenticated source file."""

    def __init__(self, code: str, message: str, *, status_code: int = 409) -> None:
        self.code = code
        self.message = message
        self.status_code = status_code
        super().__init__(message)


@dataclass(frozen=True, slots=True)
class DocumentReadProvider:
    """Provide tenant-scoped catalog, legacy-vector, and source-file reads."""

    catalog: Any
    vector_store: Any
    storage: LocalUploadStorage

    def list_catalog_documents(
        self,
        *,
        tenant_id: str,
        department_id: str | None = None,
    ) -> list[DocumentRecord]:
        """List catalog records within the authenticated tenant boundary."""

        if self.catalog is None:
            return []
        return self.catalog.list_documents(
            tenant_id=tenant_id,
            department_id=department_id,
        )

    def get_department(
        self,
        department_id: str,
        *,
        tenant_id: str,
    ) -> DepartmentRecord | None:
        """Read one tenant-owned department for folder validation."""

        if self.catalog is None:
            return None
        return self.catalog.get_department(department_id, tenant_id=tenant_id)

    def upsert_department(self, department: DepartmentRecord) -> None:
        """Persist one validated tenant-owned folder department."""

        if self.catalog is not None:
            self.catalog.upsert_department(department)

    def list_departments(
        self,
        *,
        tenant_id: str,
        company_id: str,
    ) -> list[DepartmentRecord]:
        """List departments within the authenticated tenant and company."""

        if self.catalog is None:
            return []
        return self.catalog.list_departments(
            tenant_id=tenant_id,
            company_id=company_id,
        )

    async def list_legacy_documents(self, *, tenant_id: str) -> list[dict[str, Any]]:
        """List vector-only legacy records for the authenticated tenant."""

        if self.vector_store is None:
            return []
        return await self.vector_store.list_documents(tenant_id=tenant_id)

    def accepted_reference(self, path: str, *, tenant_id: str) -> str:
        """Return an opaque reference or an empty value for an unavailable source."""

        try:
            return self.storage.accepted_reference(path, tenant_id)
        except UploadPolicyError:
            return ""

    async def get_chunks(self, doc_id: str, *, tenant_id: str | None) -> list[dict[str, Any]]:
        """Return only tenant-scoped chunks for one requested document."""

        if self.vector_store is None:
            return []
        return await self.vector_store.get_chunks_by_doc_id(doc_id, tenant_id=tenant_id)

    def get_document(self, doc_id: str, *, tenant_id: str) -> DocumentRecord | None:
        """Read one catalog record without permitting cross-tenant lookup."""

        if self.catalog is None:
            return None
        return self.catalog.get_document(doc_id, tenant_id=tenant_id)

    def resolve_source_path(self, record: DocumentRecord, *, tenant_id: str) -> str:
        """Resolve one catalog-owned source path without leaking storage details."""

        try:
            reference = self.storage.accepted_reference(
                record.storage_reference,
                tenant_id,
            )
            return self.storage.resolve_accepted(reference, tenant_id)
        except UploadPolicyError as error:
            raise DocumentReadError(
                "document_source_unavailable",
                "文档源文件不可用，请重新上传",
            ) from error


def get_document_read(request: Request) -> DocumentReadProvider:
    """Compose the read-side document provider from application-owned adapters."""

    return DocumentReadProvider(
        catalog=getattr(request.app.state, "document_catalog", None),
        vector_store=getattr(request.app.state, "vector_store", None),
        storage=LocalUploadStorage(settings.upload_dir),
    )


__all__ = ["DocumentReadError", "DocumentReadProvider", "get_document_read"]
