"""Tests for bounded, organization-scoped current-corpus Fixture draft authoring."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest

from domain.documents import DocumentRecord
from evaluation.current_corpus_drafts import (
    CurrentCorpusDraftPermissionError,
    CurrentCorpusDraftService,
    CurrentCorpusDraftValidationError,
)


@dataclass
class FakeCatalog:
    """Small metadata-only catalog fake used by the draft service tests."""

    records: list[DocumentRecord]

    def list_documents(self, *, tenant_id: str, **_kwargs: object) -> list[DocumentRecord]:
        return [record for record in self.records if record.tenant_id == tenant_id]


def _record(doc_id: str, department_id: str, *, status: str = "ingested") -> DocumentRecord:
    return DocumentRecord(
        doc_id=doc_id,
        tenant_id="org-a",
        company_id="company-a",
        department_id=department_id,
        folder_path="",
        relative_path="",
        normalized_relative_path="",
        uploaded_filename="reference-title.pdf",
        display_name="reference-title",
        provenance_source_filename="reference-title.pdf",
        storage_reference="not-returned-storage-reference",
        content_sha256="a" * 64,
        size_bytes=1,
        mime_type="application/pdf",
        document_type="policy",
        source_format="pdf",
        ingest_status=status,
        version=1,
    )


def _service(tmp_path: Path) -> CurrentCorpusDraftService:
    return CurrentCorpusDraftService(tmp_path / "current-corpus-drafts")


def test_creates_a_bounded_current_corpus_fixture_draft_without_document_content(tmp_path: Path) -> None:
    service = _service(tmp_path)
    catalog = FakeCatalog(
        [
            _record("hr-1", "human_resources"),
            _record("admin-1", "administration"),
            _record("finance-1", "finance"),
            _record("finance-2", "finance"),
            _record("ignored", "finance", status="failed"),
        ]
    )

    draft = service.create("org-a", catalog, actor_id="admin-a")

    assert draft["status"] == "draft"
    assert draft["document_count"] == 4
    assert draft["department_count"] == 3
    assert draft["candidate_count"] == 7
    assert len(draft["candidates"]) == 7
    assert {item["fixture"] for item in draft["candidates"]} == set(service.required_fixtures)

    hr_candidate = next(item for item in draft["candidates"] if item["fixture"].endswith("hr_document"))
    assert hr_candidate["alignment_status"] == "ready_for_review"
    assert hr_candidate["source_document_ids"] == ["hr-1"]
    controlled = {
        item["fixture"]: item
        for item in draft["candidates"]
        if item["fixture"]
        in {
            "frozen_corpus_absence_check",
            "bm25_unavailable_dense_graph_available",
            "all_retrieval_branches_unavailable",
            "prompt_injection_safety_fixture",
        }
    }
    assert set(controlled) == {
        "frozen_corpus_absence_check",
        "bm25_unavailable_dense_graph_available",
        "all_retrieval_branches_unavailable",
        "prompt_injection_safety_fixture",
    }
    assert all(item["alignment_status"] == "ready_for_review" for item in controlled.values())
    assert controlled["prompt_injection_safety_fixture"]["source_document_ids"] == []

    rendered = repr(draft)
    assert "reference-title" in rendered
    assert "not-returned-storage-reference" not in rendered
    assert "storage_reference" not in rendered
    assert "content_sha256" not in rendered
    assert "department_id" not in rendered


def test_current_corpus_draft_requires_distinct_admin_for_candidate_approval(tmp_path: Path) -> None:
    service = _service(tmp_path)
    catalog = FakeCatalog([_record("hr-1", "human_resources"), _record("admin-1", "administration")])
    created = service.create("org-a", catalog, actor_id="admin-a")
    candidate = next(item for item in created["candidates"] if item["alignment_status"] == "ready_for_review")

    submitted = service.submit_candidate(
        "org-a",
        created["draft_id"],
        candidate["candidate_id"],
        expected_revision=created["revision"],
        actor_id="admin-a",
    )
    assert submitted["status"] == "pending_review"

    with pytest.raises(CurrentCorpusDraftPermissionError, match="maker_cannot_approve"):
        service.review_candidate(
            "org-a",
            created["draft_id"],
            candidate["candidate_id"],
            expected_revision=submitted["revision"],
            actor_id="admin-a",
            decision="approve",
            reason="",
        )

    approved = service.review_candidate(
        "org-a",
        created["draft_id"],
        candidate["candidate_id"],
        expected_revision=submitted["revision"],
        actor_id="admin-b",
        decision="approve",
        reason="候选引用范围已复核",
    )
    approved_candidate = next(
        item for item in approved["candidates"] if item["candidate_id"] == candidate["candidate_id"]
    )
    assert approved_candidate["review_status"] == "approved"
    assert "reviewer_id" not in approved_candidate


def test_controlled_fixture_is_reviewable_without_manual_scenario_authoring(tmp_path: Path) -> None:
    service = _service(tmp_path)
    created = service.create("org-a", FakeCatalog([]), actor_id="admin-a")
    candidate = next(item for item in created["candidates"] if item["fixture"] == "prompt_injection_safety_fixture")

    assert candidate["alignment_status"] == "ready_for_review"
    assert candidate["source_document_ids"] == []

    submitted = service.submit_candidate(
        "org-a",
        created["draft_id"],
        candidate["candidate_id"],
        expected_revision=created["revision"],
        actor_id="admin-a",
    )
    assert next(item for item in submitted["candidates"] if item["candidate_id"] == candidate["candidate_id"])["review_status"] == "pending_review"


def test_legacy_controlled_candidate_is_presented_and_submitted_as_reviewable(tmp_path: Path) -> None:
    service = _service(tmp_path)
    created = service.create("org-a", FakeCatalog([]), actor_id="admin-a")
    path = service._path("org-a", created["draft_id"])
    raw = __import__("json").loads(path.read_text(encoding="utf-8"))
    candidate = next(item for item in raw["candidates"] if item["fixture"] == "prompt_injection_safety_fixture")
    candidate["alignment_status"] = "needs_manual_authoring"
    candidate["reason_code"] = "requires_safety_case_authoring"
    path.write_text(__import__("json").dumps(raw), encoding="utf-8")

    upgraded = service.get("org-a", created["draft_id"])
    public_candidate = next(item for item in upgraded["candidates"] if item["candidate_id"] == candidate["candidate_id"])
    assert public_candidate["alignment_status"] == "ready_for_review"
    assert public_candidate["reason_code"] == "controlled_prompt_injection_fixture_ready"

    submitted = service.submit_candidate(
        "org-a",
        created["draft_id"],
        candidate["candidate_id"],
        expected_revision=created["revision"],
        actor_id="admin-a",
    )
    assert next(item for item in submitted["candidates"] if item["candidate_id"] == candidate["candidate_id"])["review_status"] == "pending_review"


def test_conflict_candidate_requires_explicit_maker_confirmation_before_review(tmp_path: Path) -> None:
    service = _service(tmp_path)
    created = service.create(
        "org-a",
        FakeCatalog([_record("finance-1", "finance"), _record("finance-2", "finance")]),
        actor_id="admin-a",
    )
    candidate = next(item for item in created["candidates"] if item["fixture"] == "equal_authority_conflicting_documents")

    assert candidate["alignment_status"] == "needs_manual_conflict_confirmation"
    with pytest.raises(CurrentCorpusDraftValidationError, match="conflict_confirmation_required"):
        service.submit_candidate(
            "org-a",
            created["draft_id"],
            candidate["candidate_id"],
            expected_revision=created["revision"],
            actor_id="admin-a",
        )

    submitted = service.submit_candidate(
        "org-a",
        created["draft_id"],
        candidate["candidate_id"],
        expected_revision=created["revision"],
        actor_id="admin-a",
        confirm_actual_conflict=True,
    )
    confirmed = next(item for item in submitted["candidates"] if item["candidate_id"] == candidate["candidate_id"])
    assert confirmed["alignment_status"] == "ready_for_review"
    assert confirmed["review_status"] == "pending_review"


def test_current_corpus_drafts_are_organization_scoped(tmp_path: Path) -> None:
    service = _service(tmp_path)
    created = service.create("org-a", FakeCatalog([_record("hr-1", "human_resources")]), actor_id="admin-a")

    assert service.list("org-b") == []
    with pytest.raises(Exception, match="current_corpus_draft_not_found"):
        service.get("org-b", created["draft_id"])
