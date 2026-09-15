"""SQLite metadata adapter for stable document catalog and provenance records."""

from __future__ import annotations

import json
import sqlite3
import threading
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from sqlalchemy import delete, insert, select, update

from infrastructure.postgres.database import DatabaseConflictError, DatabaseService
from infrastructure.postgres.models import departments, documents

from domain.documents import (
    CatalogConflictError,
    DepartmentRecord,
    DocumentRecord,
)


_DOCUMENT_COLUMNS = (
    "doc_id",
    "tenant_id",
    "company_id",
    "department_id",
    "folder_path",
    "relative_path",
    "normalized_relative_path",
    "uploaded_filename",
    "display_name",
    "provenance_source_filename",
    "storage_reference",
    "content_sha256",
    "size_bytes",
    "mime_type",
    "document_type",
    "source_format",
    "ingest_status",
    "version",
    "authority",
    "review_status",
    "sensitivity",
    "external_source_id",
    "created_by",
    "created_at",
    "updated_at",
    "chunks_count",
    "entities_count",
    "relations_count",
    "error_code",
    "legacy",
    "metadata_json",
)


class SQLiteDocumentCatalogRepository:
    """Persist metadata-only catalog records with SQLite transactions and indexes."""

    def __init__(self, path: str | Path) -> None:
        """Initialize the catalog database and its idempotent schema."""

        self.path = Path(path).expanduser().resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        self._lock = threading.RLock()
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        """Open one configured SQLite connection for a short transaction."""

        connection = sqlite3.connect(self.path, timeout=10.0)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA busy_timeout = 10000")
        return connection

    def _initialize(self) -> None:
        """Create metadata tables and transactional uniqueness constraints."""

        with self._lock, self._connect() as connection:
            connection.execute("PRAGMA journal_mode = WAL")
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS departments (
                    tenant_id TEXT NOT NULL,
                    department_id TEXT NOT NULL,
                    company_id TEXT NOT NULL,
                    name TEXT NOT NULL,
                    normalized_key TEXT NOT NULL,
                    status TEXT NOT NULL DEFAULT 'active',
                    PRIMARY KEY (tenant_id, department_id),
                    UNIQUE (tenant_id, company_id, normalized_key)
                );

                CREATE TABLE IF NOT EXISTS documents (
                    tenant_id TEXT NOT NULL,
                    doc_id TEXT NOT NULL,
                    company_id TEXT NOT NULL,
                    department_id TEXT NOT NULL,
                    folder_path TEXT NOT NULL,
                    relative_path TEXT NOT NULL,
                    normalized_relative_path TEXT NOT NULL,
                    uploaded_filename TEXT NOT NULL,
                    display_name TEXT NOT NULL,
                    provenance_source_filename TEXT NOT NULL,
                    storage_reference TEXT NOT NULL,
                    content_sha256 TEXT NOT NULL,
                    size_bytes INTEGER NOT NULL CHECK (size_bytes >= 0),
                    mime_type TEXT NOT NULL,
                    document_type TEXT NOT NULL,
                    source_format TEXT NOT NULL,
                    ingest_status TEXT NOT NULL,
                    version INTEGER NOT NULL CHECK (version > 0),
                    authority TEXT NOT NULL DEFAULT '',
                    review_status TEXT NOT NULL DEFAULT '',
                    sensitivity TEXT NOT NULL DEFAULT '',
                    external_source_id TEXT NOT NULL DEFAULT '',
                    created_by TEXT NOT NULL DEFAULT '',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    chunks_count INTEGER NOT NULL DEFAULT 0,
                    entities_count INTEGER NOT NULL DEFAULT 0,
                    relations_count INTEGER NOT NULL DEFAULT 0,
                    error_code TEXT NOT NULL DEFAULT '',
                    legacy INTEGER NOT NULL DEFAULT 0,
                    metadata_json TEXT NOT NULL DEFAULT '{}',
                    PRIMARY KEY (tenant_id, doc_id),
                    UNIQUE (tenant_id, department_id, normalized_relative_path, version),
                    FOREIGN KEY (tenant_id, department_id)
                        REFERENCES departments (tenant_id, department_id)
                );

                CREATE INDEX IF NOT EXISTS idx_documents_tenant_company
                    ON documents (tenant_id, company_id, ingest_status);
                CREATE INDEX IF NOT EXISTS idx_documents_tenant_department
                    ON documents (tenant_id, department_id, ingest_status);
                CREATE INDEX IF NOT EXISTS idx_documents_external_source
                    ON documents (tenant_id, external_source_id);
                """
            )

    @staticmethod
    def _document_values(record: DocumentRecord) -> tuple[Any, ...]:
        """Serialize one domain record into the ordered SQL column values."""

        values = asdict(record)
        values["legacy"] = 1 if record.legacy else 0
        values["metadata_json"] = json.dumps(record.metadata, ensure_ascii=False, sort_keys=True)
        values.pop("metadata", None)
        return tuple(values[column] for column in _DOCUMENT_COLUMNS)

    @staticmethod
    def _document_from_row(row: sqlite3.Row) -> DocumentRecord:
        """Hydrate a domain record from one SQLite row."""

        values = dict(row)
        metadata = json.loads(values.pop("metadata_json") or "{}")
        values["legacy"] = bool(values["legacy"])
        values["metadata"] = metadata if isinstance(metadata, dict) else {}
        return DocumentRecord(**values)

    @staticmethod
    def _department_from_row(row: sqlite3.Row) -> DepartmentRecord:
        """Hydrate one department record from a SQLite row."""

        return DepartmentRecord(**dict(row))

    def upsert_department(self, record: DepartmentRecord) -> DepartmentRecord:
        """Create or update a department inside its tenant/company scope."""

        try:
            with self._lock, self._connect() as connection:
                connection.execute(
                    """
                    INSERT INTO departments (
                        tenant_id, department_id, company_id, name, normalized_key, status
                    ) VALUES (?, ?, ?, ?, ?, ?)
                    ON CONFLICT(tenant_id, department_id) DO UPDATE SET
                        company_id = excluded.company_id,
                        name = excluded.name,
                        normalized_key = excluded.normalized_key,
                        status = excluded.status
                    """,
                    (
                        record.tenant_id,
                        record.department_id,
                        record.company_id,
                        record.name,
                        record.normalized_key,
                        record.status,
                    ),
                )
        except sqlite3.IntegrityError as error:
            raise CatalogConflictError("department catalog conflict") from error
        return record

    def get_department(
        self,
        department_id: str,
        *,
        tenant_id: str,
        company_id: str | None = None,
    ) -> DepartmentRecord | None:
        """Return a department only when it belongs to the requested scope."""

        query = "SELECT * FROM departments WHERE tenant_id = ? AND department_id = ?"
        params: list[Any] = [tenant_id, department_id]
        if company_id is not None:
            query += " AND company_id = ?"
            params.append(company_id)
        with self._connect() as connection:
            row = connection.execute(query, params).fetchone()
        return self._department_from_row(row) if row else None

    def list_departments(
        self,
        *,
        tenant_id: str,
        company_id: str | None = None,
    ) -> list[DepartmentRecord]:
        """List active and inactive departments without crossing tenant scope."""

        query = "SELECT * FROM departments WHERE tenant_id = ?"
        params: list[Any] = [tenant_id]
        if company_id is not None:
            query += " AND company_id = ?"
            params.append(company_id)
        query += " ORDER BY name, department_id"
        with self._connect() as connection:
            rows = connection.execute(query, params).fetchall()
        return [self._department_from_row(row) for row in rows]

    def create_document(self, record: DocumentRecord) -> DocumentRecord:
        """Insert one document record and convert uniqueness failures to domain errors."""

        placeholders = ", ".join("?" for _ in _DOCUMENT_COLUMNS)
        columns = ", ".join(_DOCUMENT_COLUMNS)
        try:
            with self._lock, self._connect() as connection:
                connection.execute(
                    f"INSERT INTO documents ({columns}) VALUES ({placeholders})",
                    self._document_values(record),
                )
        except sqlite3.IntegrityError as error:
            raise CatalogConflictError("document catalog conflict") from error
        return record

    def get_document(self, doc_id: str, *, tenant_id: str) -> DocumentRecord | None:
        """Return one document only inside its tenant scope."""

        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM documents WHERE tenant_id = ? AND doc_id = ?",
                (tenant_id, doc_id),
            ).fetchone()
        return self._document_from_row(row) if row else None

    def list_documents(
        self,
        *,
        tenant_id: str,
        company_id: str | None = None,
        department_id: str | None = None,
        include_deleted: bool = False,
    ) -> list[DocumentRecord]:
        """List tenant-owned records with optional company and department filters."""

        clauses = ["tenant_id = ?"]
        params: list[Any] = [tenant_id]
        if company_id is not None:
            clauses.append("company_id = ?")
            params.append(company_id)
        if department_id is not None:
            clauses.append("department_id = ?")
            params.append(department_id)
        if not include_deleted:
            clauses.append("ingest_status != 'deleted'")
        query = f"SELECT * FROM documents WHERE {' AND '.join(clauses)} ORDER BY created_at DESC, doc_id"
        with self._connect() as connection:
            rows = connection.execute(query, params).fetchall()
        return [self._document_from_row(row) for row in rows]

    def update_document(self, record: DocumentRecord) -> DocumentRecord:
        """Replace mutable catalog metadata for an existing tenant-owned document."""

        assignments = ", ".join(
            f"{column} = ?" for column in _DOCUMENT_COLUMNS if column not in {"tenant_id", "doc_id"}
        )
        values = dict(zip(_DOCUMENT_COLUMNS, self._document_values(record), strict=True))
        params = [values[column] for column in _DOCUMENT_COLUMNS if column not in {"tenant_id", "doc_id"}]
        params.extend([record.tenant_id, record.doc_id])
        try:
            with self._lock, self._connect() as connection:
                cursor = connection.execute(
                    f"UPDATE documents SET {assignments} WHERE tenant_id = ? AND doc_id = ?",
                    params,
                )
                if cursor.rowcount != 1:
                    raise LookupError("document catalog record not found")
        except sqlite3.IntegrityError as error:
            raise CatalogConflictError("document catalog conflict") from error
        return record

    def update_document_if_status(
        self,
        record: DocumentRecord,
        *,
        allowed_statuses: set[str],
    ) -> bool:
        """Conditionally replace one exact catalog generation and lifecycle state."""

        if not allowed_statuses:
            return False
        assignments = ", ".join(
            f"{column} = ?"
            for column in _DOCUMENT_COLUMNS
            if column not in {"tenant_id", "doc_id"}
        )
        values = dict(zip(_DOCUMENT_COLUMNS, self._document_values(record), strict=True))
        params = [
            values[column]
            for column in _DOCUMENT_COLUMNS
            if column not in {"tenant_id", "doc_id"}
        ]
        ordered_statuses = sorted(allowed_statuses)
        placeholders = ", ".join("?" for _ in ordered_statuses)
        params.extend(
            [record.tenant_id, record.doc_id, record.version, *ordered_statuses]
        )
        try:
            with self._lock, self._connect() as connection:
                cursor = connection.execute(
                    f"""
                    UPDATE documents SET {assignments}
                    WHERE tenant_id = ? AND doc_id = ? AND version = ?
                      AND ingest_status IN ({placeholders})
                    """,
                    params,
                )
        except sqlite3.IntegrityError as error:
            raise CatalogConflictError("document catalog conflict") from error
        return cursor.rowcount == 1

    def remove_document(self, doc_id: str, *, tenant_id: str) -> bool:
        """Hard-delete one incomplete record for transaction compensation."""

        with self._lock, self._connect() as connection:
            cursor = connection.execute(
                "DELETE FROM documents WHERE tenant_id = ? AND doc_id = ?",
                (tenant_id, doc_id),
            )
        return cursor.rowcount == 1

    def table_columns(self, table: str) -> list[dict[str, Any]]:
        """Return schema metadata for tests and operational verification."""

        if table not in {"departments", "documents"}:
            raise ValueError("unknown catalog table")
        with self._connect() as connection:
            rows = connection.execute(f"PRAGMA table_info({table})").fetchall()
        return [dict(row) for row in rows]


def _as_datetime(value: str) -> datetime:
    """Convert stable ISO catalog timestamps for relational DateTime columns."""
    return datetime.fromisoformat(value)


class PostgreSQLDocumentCatalogRepository:
    """Persist metadata-only document facts in tenant-scoped PostgreSQL rows."""

    def __init__(self, database: DatabaseService) -> None:
        self.database = database

    @staticmethod
    def _document_values(record: DocumentRecord) -> dict[str, Any]:
        values = asdict(record)
        values["metadata_json"] = values.pop("metadata")
        values["created_at"] = _as_datetime(record.created_at)
        values["updated_at"] = _as_datetime(record.updated_at)
        return values

    @staticmethod
    def _document_from_row(row: Any) -> DocumentRecord:
        values = dict(row)
        values["metadata"] = values.pop("metadata_json") or {}
        for field in ("created_at", "updated_at"):
            timestamp = values[field]
            if timestamp.tzinfo is None:
                timestamp = timestamp.replace(tzinfo=UTC)
            values[field] = timestamp.isoformat()
        return DocumentRecord(**values)

    @staticmethod
    def _department_from_row(row: Any) -> DepartmentRecord:
        return DepartmentRecord(**dict(row))

    def upsert_department(self, record: DepartmentRecord) -> DepartmentRecord:
        """Create or update a department without leaving its tenant scope."""
        try:
            with self.database.session() as session:
                existing = session.execute(
                    select(departments).where(
                        departments.c.tenant_id == record.tenant_id,
                        departments.c.department_id == record.department_id,
                    )
                ).mappings().one_or_none()
                values = asdict(record)
                if existing is None:
                    session.execute(insert(departments).values(**values))
                else:
                    session.execute(
                        update(departments)
                        .where(
                            departments.c.tenant_id == record.tenant_id,
                            departments.c.department_id == record.department_id,
                        )
                        .values(**values)
                    )
        except DatabaseConflictError as error:
            raise CatalogConflictError("department catalog conflict") from error
        return record

    def get_department(
        self,
        department_id: str,
        *,
        tenant_id: str,
        company_id: str | None = None,
    ) -> DepartmentRecord | None:
        statement = select(departments).where(
            departments.c.tenant_id == tenant_id,
            departments.c.department_id == department_id,
        )
        if company_id is not None:
            statement = statement.where(departments.c.company_id == company_id)
        with self.database.session() as session:
            row = session.execute(statement).mappings().one_or_none()
        return self._department_from_row(row) if row else None

    def list_departments(
        self,
        *,
        tenant_id: str,
        company_id: str | None = None,
    ) -> list[DepartmentRecord]:
        statement = select(departments).where(departments.c.tenant_id == tenant_id)
        if company_id is not None:
            statement = statement.where(departments.c.company_id == company_id)
        statement = statement.order_by(departments.c.name, departments.c.department_id)
        with self.database.session() as session:
            rows = session.execute(statement).mappings().all()
        return [self._department_from_row(row) for row in rows]

    def create_document(self, record: DocumentRecord) -> DocumentRecord:
        """Insert one metadata record and expose a stable catalog conflict."""
        try:
            with self.database.session() as session:
                session.execute(insert(documents).values(**self._document_values(record)))
        except DatabaseConflictError as error:
            raise CatalogConflictError("document catalog conflict") from error
        return record

    def get_document(self, doc_id: str, *, tenant_id: str) -> DocumentRecord | None:
        with self.database.session() as session:
            row = session.execute(
                select(documents).where(
                    documents.c.tenant_id == tenant_id,
                    documents.c.doc_id == doc_id,
                )
            ).mappings().one_or_none()
        return self._document_from_row(row) if row else None

    def list_documents(
        self,
        *,
        tenant_id: str,
        company_id: str | None = None,
        department_id: str | None = None,
        include_deleted: bool = False,
    ) -> list[DocumentRecord]:
        statement = select(documents).where(documents.c.tenant_id == tenant_id)
        if company_id is not None:
            statement = statement.where(documents.c.company_id == company_id)
        if department_id is not None:
            statement = statement.where(documents.c.department_id == department_id)
        if not include_deleted:
            statement = statement.where(documents.c.ingest_status != "deleted")
        statement = statement.order_by(documents.c.created_at.desc(), documents.c.doc_id)
        with self.database.session() as session:
            rows = session.execute(statement).mappings().all()
        return [self._document_from_row(row) for row in rows]

    def update_document(self, record: DocumentRecord) -> DocumentRecord:
        """Replace mutable metadata for one exact tenant/document identity."""
        values = self._document_values(record)
        values.pop("tenant_id")
        values.pop("doc_id")
        try:
            with self.database.session() as session:
                result = session.execute(
                    update(documents)
                    .where(
                        documents.c.tenant_id == record.tenant_id,
                        documents.c.doc_id == record.doc_id,
                    )
                    .values(**values)
                )
                if result.rowcount != 1:
                    raise LookupError("document catalog record not found")
        except DatabaseConflictError as error:
            raise CatalogConflictError("document catalog conflict") from error
        return record

    def update_document_if_status(
        self,
        record: DocumentRecord,
        *,
        allowed_statuses: set[str],
    ) -> bool:
        """Conditionally replace one exact PostgreSQL catalog generation."""

        if not allowed_statuses:
            return False
        values = self._document_values(record)
        values.pop("tenant_id")
        values.pop("doc_id")
        try:
            with self.database.session() as session:
                result = session.execute(
                    update(documents)
                    .where(
                        documents.c.tenant_id == record.tenant_id,
                        documents.c.doc_id == record.doc_id,
                        documents.c.version == record.version,
                        documents.c.ingest_status.in_(sorted(allowed_statuses)),
                    )
                    .values(**values)
                )
        except DatabaseConflictError as error:
            raise CatalogConflictError("document catalog conflict") from error
        return result.rowcount == 1

    def remove_document(self, doc_id: str, *, tenant_id: str) -> bool:
        """Hard-delete one incomplete tenant-owned row for compensation."""
        with self.database.session() as session:
            result = session.execute(
                delete(documents).where(
                    documents.c.tenant_id == tenant_id,
                    documents.c.doc_id == doc_id,
                )
            )
        return result.rowcount == 1

    @staticmethod
    def table_columns(table: str) -> list[dict[str, Any]]:
        """Expose schema names for metadata-only contract verification."""
        selected = {"departments": departments, "documents": documents}.get(table)
        if selected is None:
            raise ValueError("unknown catalog table")
        return [{"name": column.name} for column in selected.columns]


__all__ = ["PostgreSQLDocumentCatalogRepository", "SQLiteDocumentCatalogRepository"]
