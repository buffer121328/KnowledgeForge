"""Conservative legacy vector-to-catalog metadata backfill helpers."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from dataclasses import dataclass
from typing import Any, Iterable

from domain.documents import DepartmentRecord, DocumentRecord


@dataclass(frozen=True)
class LegacyBackfillReport:
    """Summarize created, skipped, and unresolved legacy catalog records."""

    created: int
    skipped: int
    unresolved_names: int


@dataclass(frozen=True)
class CatalogMigrationReport:
    """Summarize one non-destructive SQLite-to-PostgreSQL catalog operation."""

    mode: str
    departments: int
    documents: int
    created: int
    skipped: int
    conflicts: tuple[str, ...]
    source_checksum: str
    target_checksum: str = ""


def _snapshot(catalog: Any, tenant_ids: Iterable[str]) -> tuple[list[Any], list[Any], str]:
    department_records: list[Any] = []
    document_records: list[Any] = []
    for tenant_id in sorted(set(tenant_ids)):
        department_records.extend(catalog.list_departments(tenant_id=tenant_id))
        document_records.extend(catalog.list_documents(tenant_id=tenant_id, include_deleted=True))
    payload = {
        "departments": [asdict(item) for item in department_records],
        "documents": [asdict(item) for item in document_records],
    }
    checksum = hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()
    return department_records, document_records, checksum


def migrate_sqlite_catalog(
    source: Any,
    target: Any,
    *,
    tenant_ids: Iterable[str],
    mode: str = "dry-run",
) -> CatalogMigrationReport:
    """Inventory, dry-run, apply or verify catalog metadata without deleting legacy rows."""
    if mode not in {"inventory", "dry-run", "apply", "verify"}:
        raise ValueError("mode must be inventory, dry-run, apply, or verify")
    tenant_scope = tuple(sorted(set(tenant_ids)))
    source_departments, source_documents, source_checksum = _snapshot(source, tenant_scope)
    target_departments, target_documents, target_checksum = _snapshot(target, tenant_scope)
    target_department_map = {
        (item.tenant_id, item.department_id): item for item in target_departments
    }
    target_document_map = {(item.tenant_id, item.doc_id): item for item in target_documents}
    conflicts: list[str] = []
    skipped = 0
    for item in source_departments:
        current = target_department_map.get((item.tenant_id, item.department_id))
        if current is not None and current != item:
            conflicts.append(f"department:{item.tenant_id}:{item.department_id}")
        elif current is not None:
            skipped += 1
    for item in source_documents:
        current = target_document_map.get((item.tenant_id, item.doc_id))
        if current is not None and current != item:
            conflicts.append(f"document:{item.tenant_id}:{item.doc_id}")
        elif current is not None:
            skipped += 1

    if mode == "verify":
        if source_checksum != target_checksum:
            conflicts.append("checksum:mismatch")
    created = 0
    if mode == "apply":
        if conflicts:
            raise ValueError("catalog migration conflicts require resolution")
        for item in source_departments:
            if (item.tenant_id, item.department_id) not in target_department_map:
                target.upsert_department(item)
                created += 1
        for item in source_documents:
            if (item.tenant_id, item.doc_id) not in target_document_map:
                target.create_document(item)
                created += 1
        _, _, target_checksum = _snapshot(target, tenant_scope)

    return CatalogMigrationReport(
        mode=mode,
        departments=len(source_departments),
        documents=len(source_documents),
        created=created,
        skipped=skipped,
        conflicts=tuple(conflicts),
        source_checksum=source_checksum,
        target_checksum=target_checksum,
    )


def backfill_legacy_documents(
    catalog: Any,
    documents: Iterable[dict[str, Any]],
    *,
    tenant_id: str,
    company_id: str | None = None,
    created_by: str = "legacy-catalog-backfill",
) -> LegacyBackfillReport:
    """Create explicit legacy records without guessing names or overwriting catalog metadata."""

    effective_company = company_id or tenant_id
    catalog.upsert_department(
        DepartmentRecord(
            department_id="legacy",
            tenant_id=tenant_id,
            company_id=effective_company,
            name="历史文档",
            normalized_key="legacy",
        )
    )
    created = skipped = unresolved = 0
    for item in documents:
        doc_id = str(item.get("doc_id") or "").strip()
        if not doc_id:
            continue
        if catalog.get_document(doc_id, tenant_id=tenant_id) is not None:
            skipped += 1
            continue
        explicit_name = str(
            item.get("original_filename")
            or item.get("uploaded_filename")
            or item.get("provenance_source_filename")
            or ""
        ).strip()
        name_unresolved = not explicit_name
        if name_unresolved:
            unresolved += 1
        internal_key = hashlib.sha256(doc_id.encode("utf-8")).hexdigest()[:32]
        catalog.create_document(
            DocumentRecord(
                doc_id=doc_id,
                tenant_id=tenant_id,
                company_id=effective_company,
                department_id="legacy",
                folder_path="",
                relative_path="",
                normalized_relative_path=f"legacy/{internal_key}",
                uploaded_filename=explicit_name,
                display_name=explicit_name,
                provenance_source_filename=explicit_name,
                storage_reference=str(item.get("source") or ""),
                content_sha256=str(item.get("content_sha256") or ""),
                size_bytes=max(0, int(item.get("size_bytes") or 0)),
                mime_type=str(item.get("mime_type") or "application/octet-stream"),
                document_type=str(item.get("doc_type") or "legacy"),
                source_format=str(item.get("source_format") or ""),
                ingest_status="legacy",
                version=max(1, int(item.get("version") or 1)),
                created_by=created_by,
                chunks_count=max(0, int(item.get("chunks_count") or 0)),
                entities_count=max(0, int(item.get("entities_count") or 0)),
                relations_count=max(0, int(item.get("relations_count") or 0)),
                legacy=True,
                metadata={"legacy_name_unresolved": name_unresolved},
            )
        )
        created += 1
    return LegacyBackfillReport(created=created, skipped=skipped, unresolved_names=unresolved)


__all__ = [
    "CatalogMigrationReport",
    "LegacyBackfillReport",
    "backfill_legacy_documents",
    "migrate_sqlite_catalog",
]
