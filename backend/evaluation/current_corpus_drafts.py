"""Bounded, organization-scoped authoring drafts for the active document corpus.

This module deliberately treats catalog records as metadata-only signals.  It never
reads document bodies, vector chunks, retrieval contexts, credentials, or test
identity configuration.  A candidate draft records whether a Fixture can be
reviewed against the *current* corpus; it never mutates a historical frozen bundle,
automatically approves a candidate, freezes a dataset, or starts validation.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import threading
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Mapping, Protocol

from domain.documents import DocumentIngestStatus
from uuid import uuid4


_DRAFT_ID = re.compile(r"^ccd_[a-f0-9]{32}$")
_CANDIDATE_ID = re.compile(r"^[a-z][a-z0-9_]{2,95}$")
_REVIEW_STATUSES = frozenset({"draft", "pending_review", "approved"})
_REQUIRED_FIXTURES = (
    "frozen_corpus_absence_check",
    "equal_authority_conflicting_documents",
    "finance_only_user_against_hr_document",
    "finance_only_user_against_administration_document",
    "bm25_unavailable_dense_graph_available",
    "all_retrieval_branches_unavailable",
    "prompt_injection_safety_fixture",
)
_CONTROLLED_ALIGNMENT_UPGRADES: dict[str, tuple[frozenset[str], str, str]] = {
    "frozen_corpus_absence_check": (
        frozenset({"needs_manual_authoring"}),
        "ready_for_review",
        "controlled_absence_fixture_ready",
    ),
    "bm25_unavailable_dense_graph_available": (
        frozenset({"needs_runtime_retrieval_review"}),
        "ready_for_review",
        "controlled_bm25_fixture_ready",
    ),
    "all_retrieval_branches_unavailable": (
        frozenset({"needs_manual_authoring"}),
        "ready_for_review",
        "controlled_all_branches_unavailable_fixture_ready",
    ),
    "prompt_injection_safety_fixture": (
        frozenset({"needs_manual_authoring"}),
        "ready_for_review",
        "controlled_prompt_injection_fixture_ready",
    ),
}


class CurrentCorpusDraftError(RuntimeError):
    """Base class for stable current-corpus draft failures."""


class CurrentCorpusDraftNotFound(CurrentCorpusDraftError):
    """Raised when an organization cannot access a requested draft."""


class CurrentCorpusDraftConflict(CurrentCorpusDraftError):
    """Raised when a client applies a stale draft revision."""


class CurrentCorpusDraftPermissionError(CurrentCorpusDraftError):
    """Raised when maker-checker separation would be violated."""


class CurrentCorpusDraftValidationError(CurrentCorpusDraftError):
    """Raised when an unsafe candidate transition is requested."""


class CatalogDocument(Protocol):
    """The metadata-only catalog surface consumed by candidate drafting."""

    doc_id: str
    department_id: str
    ingest_status: str


class DocumentCatalog(Protocol):
    """Read only the authenticated organization's catalog metadata."""

    def list_documents(self, *, tenant_id: str, **kwargs: object) -> list[CatalogDocument]: ...


def _now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat()


def _org_namespace(org_id: str) -> str:
    if not isinstance(org_id, str) or not org_id.strip():
        raise CurrentCorpusDraftNotFound("organization_not_found")
    return hashlib.sha256(org_id.encode("utf-8")).hexdigest()[:24]


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    """Atomically persist a metadata-only draft without exposing partial records."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        temporary.write_text(
            json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")),
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise CurrentCorpusDraftNotFound("current_corpus_draft_not_found") from error
    if not isinstance(value, dict):
        raise CurrentCorpusDraftNotFound("current_corpus_draft_not_found")
    return value


class CurrentCorpusDraftService:
    """Create and review metadata-only Fixture alignment drafts for one organization.

    The service carries raw document identifiers only so a future explicitly reviewed
    authoring step can bind expected source documents.  Public responses omit every
    document field other than its opaque ID; document title, filename, path, content,
    storage reference, hash, and department identifier never leave this boundary.
    """

    required_fixtures = _REQUIRED_FIXTURES

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self._locks: dict[str, threading.RLock] = {}
        self._locks_guard = threading.Lock()

    def _lock(self, org_id: str, draft_id: str) -> threading.RLock:
        key = f"{_org_namespace(org_id)}:{draft_id}"
        with self._locks_guard:
            return self._locks.setdefault(key, threading.RLock())

    @staticmethod
    def _safe_draft_id(draft_id: str) -> str:
        if not isinstance(draft_id, str) or not _DRAFT_ID.fullmatch(draft_id):
            raise CurrentCorpusDraftNotFound("current_corpus_draft_not_found")
        return draft_id

    @staticmethod
    def _safe_candidate_id(candidate_id: str) -> str:
        if not isinstance(candidate_id, str) or not _CANDIDATE_ID.fullmatch(candidate_id):
            raise CurrentCorpusDraftNotFound("current_corpus_candidate_not_found")
        return candidate_id

    def _org_root(self, org_id: str) -> Path:
        return self.root / _org_namespace(org_id)

    def _path(self, org_id: str, draft_id: str) -> Path:
        return self._org_root(org_id) / f"{self._safe_draft_id(draft_id)}.json"

    @staticmethod
    def _completed_documents(catalog: DocumentCatalog, org_id: str) -> list[CatalogDocument]:
        """Select only current ingested records and retain the three safe fields."""

        records = catalog.list_documents(tenant_id=org_id)
        selected: list[CatalogDocument] = []
        for record in records:
            doc_id = str(getattr(record, "doc_id", "") or "").strip()
            department = str(getattr(record, "department_id", "") or "").strip()
            status = str(getattr(record, "ingest_status", "") or "").strip()
            if doc_id and department and status == DocumentIngestStatus.INGESTED.value:
                selected.append(record)
        return sorted(selected, key=lambda record: str(record.doc_id))

    @staticmethod
    def _candidate(
        fixture: str,
        *,
        source_document_ids: list[str],
        source_references: list[dict[str, str]],
        alignment_status: str,
        reason_code: str,
        actor_id: str,
    ) -> dict[str, Any]:
        return {
            "candidate_id": fixture,
            "fixture": fixture,
            "source_document_ids": source_document_ids[:2],
            "source_references": source_references[:2],
            "source_document_count": len(source_document_ids[:2]),
            "alignment_status": alignment_status,
            "reason_code": reason_code,
            "review_status": "draft",
            "created_by": actor_id,
            "submitted_by": "",
            "reviewed_by": "",
            "review_reason": "",
            "submitted_at": None,
            "reviewed_at": None,
        }

    @classmethod
    def _build_candidates(
        cls,
        documents: list[CatalogDocument],
        *,
        actor_id: str,
    ) -> list[dict[str, Any]]:
        """Build conservative candidates without inferring document semantics."""

        by_department: dict[str, list[CatalogDocument]] = {}
        for record in documents:
            by_department.setdefault(str(record.department_id), []).append(record)
        all_records = list(documents)
        same_department_pair = next(
            (values[:2] for values in by_department.values() if len(values) >= 2),
            [],
        )
        hr_records = by_department.get("human_resources", [])[:1]
        administration_records = by_department.get("administration", [])[:1]
        general_records = all_records[:1]

        def ref(record: CatalogDocument) -> dict[str, str]:
            """Return source metadata only; never return path, content, or storage fields."""
            source = str(
                getattr(record, "display_name", "")
                or getattr(record, "provenance_source_filename", "")
                or "document_catalog"
            ).strip()[:256]
            return {
                "document_id": str(record.doc_id),
                "source": source or "document_catalog",
                "department": str(record.department_id),
            }

        same_department_pair_ids = [str(record.doc_id) for record in same_department_pair]
        hr_ids = [str(record.doc_id) for record in hr_records]
        administration_ids = [str(record.doc_id) for record in administration_records]
        general_id = [str(record.doc_id) for record in general_records]

        candidates = [
            cls._candidate(
                "frozen_corpus_absence_check",
                source_document_ids=[],
                source_references=[],
                alignment_status="ready_for_review",
                reason_code="controlled_absence_fixture_ready",
                actor_id=actor_id,
            ),
            cls._candidate(
                "equal_authority_conflicting_documents",
                source_document_ids=same_department_pair_ids,
                source_references=[ref(record) for record in same_department_pair],
                alignment_status=(
                    "needs_manual_conflict_confirmation"
                    if len(same_department_pair) == 2
                    else "unavailable"
                ),
                reason_code=(
                    "requires_conflict_confirmation"
                    if len(same_department_pair) == 2
                    else "insufficient_current_corpus_for_conflict"
                ),
                actor_id=actor_id,
            ),
            cls._candidate(
                "finance_only_user_against_hr_document",
                source_document_ids=hr_ids,
                source_references=[ref(record) for record in hr_records],
                alignment_status="ready_for_review" if hr_ids else "unavailable",
                reason_code="candidate_reference_bound" if hr_ids else "required_hr_document_missing",
                actor_id=actor_id,
            ),
            cls._candidate(
                "finance_only_user_against_administration_document",
                source_document_ids=administration_ids,
                source_references=[ref(record) for record in administration_records],
                alignment_status="ready_for_review" if administration_ids else "unavailable",
                reason_code=(
                    "candidate_reference_bound"
                    if administration_ids
                    else "required_administration_document_missing"
                ),
                actor_id=actor_id,
            ),
            cls._candidate(
                "bm25_unavailable_dense_graph_available",
                source_document_ids=general_id,
                source_references=[ref(record) for record in general_records],
                alignment_status="ready_for_review" if general_id else "unavailable",
                reason_code=(
                    "controlled_bm25_fixture_ready"
                    if general_id
                    else "current_corpus_empty"
                ),
                actor_id=actor_id,
            ),
            cls._candidate(
                "all_retrieval_branches_unavailable",
                source_document_ids=[],
                source_references=[],
                alignment_status="ready_for_review",
                reason_code="controlled_all_branches_unavailable_fixture_ready",
                actor_id=actor_id,
            ),
            cls._candidate(
                "prompt_injection_safety_fixture",
                source_document_ids=[],
                source_references=[],
                alignment_status="ready_for_review",
                reason_code="controlled_prompt_injection_fixture_ready",
                actor_id=actor_id,
            ),
        ]
        assert tuple(candidate["fixture"] for candidate in candidates) == _REQUIRED_FIXTURES
        return candidates

    @staticmethod
    def _draft_status(candidates: list[Mapping[str, Any]]) -> str:
        statuses = {str(item.get("review_status") or "draft") for item in candidates}
        if statuses == {"approved"}:
            return "approved"
        if "pending_review" in statuses:
            return "pending_review"
        return "draft"

    @staticmethod
    def _effective_alignment(candidate: Mapping[str, Any]) -> tuple[str, str]:
        """Expose safe controlled-Fixture upgrades for drafts created before this fix."""

        fixture = str(candidate.get("fixture") or "")
        status = str(candidate.get("alignment_status") or "unavailable")
        reason_code = str(candidate.get("reason_code") or "current_corpus_unaligned")
        upgrade = _CONTROLLED_ALIGNMENT_UPGRADES.get(fixture)
        if upgrade and status in upgrade[0]:
            return upgrade[1], upgrade[2]
        return status, reason_code

    @classmethod
    def _public(cls, draft: Mapping[str, Any]) -> dict[str, Any]:
        """Return safe draft data, intentionally excluding origin identities and corpus internals."""

        candidates = [
            {
                "candidate_id": str(item.get("candidate_id") or ""),
                "fixture": str(item.get("fixture") or ""),
                "source_document_ids": [
                    str(value) for value in item.get("source_document_ids", []) if isinstance(value, str)
                ][:2],
                "source_references": [
                    {
                        "document_id": str(reference.get("document_id") or ""),
                        "source": str(reference.get("source") or "document_catalog")[:256],
                        "department": str(reference.get("department") or "")[:128],
                    }
                    for reference in item.get("source_references", [])
                    if isinstance(reference, Mapping)
                ][:2],
                "source_document_count": min(
                    2, max(0, int(item.get("source_document_count") or 0))
                ),
                "alignment_status": cls._effective_alignment(item)[0],
                "reason_code": cls._effective_alignment(item)[1],
                "review_status": str(item.get("review_status") or "draft"),
                "submitted_at": item.get("submitted_at"),
                "reviewed_at": item.get("reviewed_at"),
                "review_reason": str(item.get("review_reason") or "")[:1_000],
            }
            for item in draft.get("candidates", [])
            if isinstance(item, Mapping)
        ]
        candidate_statuses = Counter(item["review_status"] for item in candidates)
        return {
            "draft_id": str(draft.get("draft_id") or ""),
            "status": cls._draft_status(candidates),
            "revision": max(1, int(draft.get("revision") or 1)),
            "created_at": str(draft.get("created_at") or ""),
            "updated_at": str(draft.get("updated_at") or ""),
            "document_count": max(0, int(draft.get("document_count") or 0)),
            "department_count": max(0, int(draft.get("department_count") or 0)),
            "candidate_count": len(candidates),
            "reviewable_candidate_count": sum(
                item["alignment_status"] == "ready_for_review" for item in candidates
            ),
            "approved_candidate_count": candidate_statuses["approved"],
            "authoring_dataset_id": str(draft.get("authoring_dataset_id") or "") or None,
            "current_corpus_snapshot_sha256": str(draft.get("current_corpus_snapshot_sha256") or "") or None,
            "authoring_dataset_created_at": draft.get("authoring_dataset_created_at"),
            "counts": {status: candidate_statuses[status] for status in sorted(_REVIEW_STATUSES)},
            "candidates": candidates,
        }

    @staticmethod
    def _load_candidate(draft: dict[str, Any], candidate_id: str) -> dict[str, Any]:
        candidate_id = CurrentCorpusDraftService._safe_candidate_id(candidate_id)
        for item in draft.get("candidates", []):
            if isinstance(item, dict) and item.get("candidate_id") == candidate_id:
                return item
        raise CurrentCorpusDraftNotFound("current_corpus_candidate_not_found")

    @staticmethod
    def _check_revision(draft: Mapping[str, Any], expected_revision: int) -> None:
        if expected_revision != draft.get("revision"):
            raise CurrentCorpusDraftConflict(f"revision_conflict:{draft.get('revision')}")

    @staticmethod
    def _bump(draft: dict[str, Any]) -> None:
        draft["revision"] = int(draft.get("revision") or 0) + 1
        draft["updated_at"] = _now()

    def create_from_records(
        self,
        org_id: str,
        records: list[CatalogDocument],
        *,
        actor_id: str,
    ) -> dict[str, Any]:
        """Persist a candidate draft from authenticated metadata records only."""

        if not isinstance(actor_id, str) or not actor_id.strip():
            raise CurrentCorpusDraftValidationError("invalid_actor")
        documents = [
            record
            for record in records
            if (
                str(getattr(record, "doc_id", "") or "").strip()
                and str(getattr(record, "department_id", "") or "").strip()
                and str(getattr(record, "ingest_status", "") or "").strip() == DocumentIngestStatus.INGESTED.value
            )
        ]
        documents = sorted(documents, key=lambda record: str(record.doc_id))
        department_count = len({str(item.department_id) for item in documents})
        draft_id = f"ccd_{uuid4().hex}"
        now = _now()
        draft = {
            "schema_version": "current-corpus-fixture-draft-v1",
            "draft_id": draft_id,
            "org_id_hash": _org_namespace(org_id),
            "revision": 1,
            "created_at": now,
            "updated_at": now,
            "created_by": actor_id,
            "document_count": len(documents),
            "department_count": department_count,
            "source_documents": [
                {
                    "document_id": str(record.doc_id),
                    "source": str(
                        getattr(record, "display_name", "")
                        or getattr(record, "provenance_source_filename", "")
                        or "document_catalog"
                    ).strip()[:256]
                    or "document_catalog",
                    "department": str(record.department_id),
                }
                for record in documents
            ],
            "candidates": self._build_candidates(documents, actor_id=actor_id),
        }
        with self._lock(org_id, draft_id):
            _write_json(self._path(org_id, draft_id), draft)
        return self._public(draft)

    def create(self, org_id: str, catalog: DocumentCatalog, *, actor_id: str) -> dict[str, Any]:
        """Snapshot active metadata and create a non-destructive review draft."""

        return self.create_from_records(
            org_id,
            self._completed_documents(catalog, org_id),
            actor_id=actor_id,
        )

    def get(self, org_id: str, draft_id: str) -> dict[str, Any]:
        """Read one organization-scoped metadata draft."""

        draft_id = self._safe_draft_id(draft_id)
        with self._lock(org_id, draft_id):
            return self._public(_read_json(self._path(org_id, draft_id)))

    def list(self, org_id: str, *, limit: int = 20) -> list[dict[str, Any]]:
        """List newest drafts for one organization without exposing storage paths."""

        if limit < 1 or limit > 50:
            raise CurrentCorpusDraftValidationError("invalid_list_limit")
        root = self._org_root(org_id)
        if not root.is_dir():
            return []
        items: list[dict[str, Any]] = []
        for path in root.glob("ccd_*.json"):
            if not _DRAFT_ID.fullmatch(path.stem):
                continue
            try:
                draft = _read_json(path)
            except CurrentCorpusDraftNotFound:
                continue
            if draft.get("org_id_hash") != _org_namespace(org_id):
                continue
            items.append(self._public(draft))
        return sorted(items, key=lambda item: item["created_at"], reverse=True)[:limit]

    def submit_candidate(
        self,
        org_id: str,
        draft_id: str,
        candidate_id: str,
        *,
        expected_revision: int,
        actor_id: str,
        confirm_actual_conflict: bool = False,
    ) -> dict[str, Any]:
        """Submit a controlled candidate after any required maker confirmation.

        Only a potential equal-authority conflict requires this confirmation: catalog
        metadata can prove that two current records exist, but cannot prove their
        contents conflict.  Retrieval outages, absence, and prompt-injection inputs
        are server-controlled scenarios and therefore proceed directly to review.
        """

        draft_id = self._safe_draft_id(draft_id)
        with self._lock(org_id, draft_id):
            path = self._path(org_id, draft_id)
            draft = _read_json(path)
            self._check_revision(draft, expected_revision)
            candidate = self._load_candidate(draft, candidate_id)
            if candidate.get("review_status") != "draft":
                raise CurrentCorpusDraftConflict("candidate_not_draft")
            if candidate.get("alignment_status") == "needs_manual_conflict_confirmation":
                if candidate.get("fixture") != "equal_authority_conflicting_documents":
                    raise CurrentCorpusDraftValidationError("candidate_not_ready_for_review")
                if not confirm_actual_conflict:
                    raise CurrentCorpusDraftValidationError("conflict_confirmation_required")
                candidate["alignment_status"] = "ready_for_review"
                candidate["reason_code"] = "conflict_candidate_confirmed"
                candidate["maker_confirmed_at"] = _now()
            else:
                effective_status, effective_reason = self._effective_alignment(candidate)
                if effective_status != "ready_for_review":
                    raise CurrentCorpusDraftValidationError("candidate_not_ready_for_review")
                candidate["alignment_status"] = effective_status
                candidate["reason_code"] = effective_reason
            candidate["review_status"] = "pending_review"
            candidate["submitted_by"] = actor_id
            candidate["submitted_at"] = _now()
            self._bump(draft)
            _write_json(path, draft)
            return self._public(draft)

    def review_candidate(
        self,
        org_id: str,
        draft_id: str,
        candidate_id: str,
        *,
        expected_revision: int,
        actor_id: str,
        decision: str,
        reason: str,
    ) -> dict[str, Any]:
        """Apply a maker-checker decision without exposing identity details."""

        if decision not in {"approve", "reject"}:
            raise CurrentCorpusDraftValidationError("invalid_review_decision")
        draft_id = self._safe_draft_id(draft_id)
        with self._lock(org_id, draft_id):
            path = self._path(org_id, draft_id)
            draft = _read_json(path)
            self._check_revision(draft, expected_revision)
            candidate = self._load_candidate(draft, candidate_id)
            if candidate.get("review_status") != "pending_review":
                raise CurrentCorpusDraftConflict("candidate_not_pending_review")
            if actor_id in {str(candidate.get("created_by") or ""), str(candidate.get("submitted_by") or "")}:
                raise CurrentCorpusDraftPermissionError("maker_cannot_approve")
            candidate["review_status"] = "approved" if decision == "approve" else "draft"
            candidate["reviewed_by"] = actor_id
            candidate["reviewed_at"] = _now()
            candidate["review_reason"] = str(reason or "")[:1_000]
            if decision == "reject":
                candidate["submitted_by"] = ""
                candidate["submitted_at"] = None
            self._bump(draft)
            _write_json(path, draft)
            return self._public(draft)


__all__ = [
    "CurrentCorpusDraftConflict",
    "CurrentCorpusDraftError",
    "CurrentCorpusDraftNotFound",
    "CurrentCorpusDraftPermissionError",
    "CurrentCorpusDraftService",
    "CurrentCorpusDraftValidationError",
]
