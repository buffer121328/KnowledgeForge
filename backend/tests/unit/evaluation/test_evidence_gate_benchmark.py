"""Acceptance tests for reviewed evidence-gate benchmark bundles."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path

import pytest

from evaluation.evidence_gate.benchmark import (
    REQUIRED_EVIDENCE_GATE_CATEGORIES,
    EvidenceGateBenchmarkError,
    load_evidence_gate_benchmark,
    reviewed_case_contract_errors,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_bundle(root: Path, *, reviewed: bool = True) -> Path:
    root.mkdir()
    statuses = {
        "fully_answerable": "answered",
        "completely_unanswerable": "insufficient_evidence",
        "background_only": "insufficient_evidence",
        "partially_answerable": "partially_answered",
        "conflicting": "conflicting_evidence",
        "missing_version_or_date": "needs_clarification",
        "missing_business_record": "insufficient_evidence",
        "authorization_filtered": "insufficient_evidence",
        "single_branch_unavailable": "answered",
        "all_branches_unavailable": "source_unavailable",
        "prompt_injection": "human_review_required",
    }
    states = {
        "fully_answerable": ["direct_evidence"],
        "partially_answerable": ["partial_evidence"],
        "conflicting": ["conflicting_evidence"],
        "background_only": ["relevant_background"],
        "authorization_filtered": ["insufficient_evidence"],
        "single_branch_unavailable": ["direct_evidence"],
        "all_branches_unavailable": ["insufficient_evidence"],
        "prompt_injection": ["invalid_provenance"],
    }
    records = []
    for index, category in enumerate(sorted(REQUIRED_EVIDENCE_GATE_CATEGORIES), 1):
        records.append(
            {
                "id": f"eg-{index:02d}",
                "question": f"reviewed question {index}",
                "category": category,
                "expected_response_status": statuses[category],
                "expected_evidence_states": states.get(
                    category, ["insufficient_evidence"]
                ),
                "expected_reason_codes": ["direct_support"],
                "expected_citation_context_ids": (
                    [f"ctx-{index}"]
                    if statuses[category] in {"answered", "partially_answered"}
                    else []
                ),
                "expected_branch_availability": {
                    "dense": "available",
                    "bm25": "available",
                    "graph": "available",
                },
                "notes": "review fixture",
            }
        )
    cases_path = root / "cases.jsonl"
    cases_path.write_text(
        "".join(json.dumps(item, sort_keys=True) + "\n" for item in records),
        encoding="utf-8",
    )
    review = {
        "schema_version": "evidence-gate-review-v1",
        "dataset_version": "evidence-gates-test-v1",
        "review_status": "approved" if reviewed else "pending_human_review",
        "reviewed_at": "2026-08-03T00:00:00Z" if reviewed else None,
        "reviewer_ids": ["reviewer-1"] if reviewed else [],
        "reviewed_case_ids": [item["id"] for item in records] if reviewed else [],
        "decision_notes": "fixture approval" if reviewed else "awaiting review",
    }
    review_path = root / "review-record.json"
    review_path.write_text(json.dumps(review, sort_keys=True) + "\n", encoding="utf-8")
    manifest = {
        "schema_version": "evidence-gate-benchmark-v1",
        "dataset_version": "evidence-gates-test-v1",
        "review_status": review["review_status"],
        "answerability_labels": sorted(REQUIRED_EVIDENCE_GATE_CATEGORIES),
        "expected_response_statuses": [
            "answered",
            "partially_answered",
            "insufficient_evidence",
            "needs_clarification",
            "conflicting_evidence",
            "human_review_required",
            "source_unavailable",
        ],
        "cases_file": cases_path.name,
        "cases_sha256": _sha256(cases_path),
        "review_record_file": review_path.name,
        "review_record_sha256": _sha256(review_path),
    }
    manifest_path = root / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest_path


def test_reviewed_bundle_validates_manifest_hash_review_and_coverage(tmp_path: Path) -> None:
    manifest_path = _write_bundle(tmp_path / "evidence")

    dataset = load_evidence_gate_benchmark(
        manifest_path,
        expected_manifest_sha256=_sha256(manifest_path),
    )

    assert dataset.dataset_version == "evidence-gates-test-v1"
    assert dataset.manifest_sha256 == _sha256(manifest_path)
    assert {case.category for case in dataset} == REQUIRED_EVIDENCE_GATE_CATEGORIES
    assert all(case.expected_evidence_states for case in dataset)
    assert all(case.expected_response_status for case in dataset)


def test_reviewed_routine_bundle_has_coherent_semantic_contracts() -> None:
    root = Path(__file__).resolve().parents[3] / "evaluation/data/evidence-gates/evidence-gates-v1-routine"
    manifest = root / "manifest.json"

    dataset = load_evidence_gate_benchmark(
        manifest,
        expected_manifest_sha256=_sha256(manifest),
        require_reviewed=False,
    )

    assert len(dataset) == 50
    assert {case.category for case in dataset} == REQUIRED_EVIDENCE_GATE_CATEGORIES
    assert all(reviewed_case_contract_errors(case) == () for case in dataset)


def test_routine_bundle_is_complete_for_category_diagnostic_planning() -> None:
    """The checked-in 50 cases expose every bounded diagnostic policy input."""
    root = (
        Path(__file__).resolve().parents[3]
        / "evaluation/data/evidence-gates/evidence-gates-v1-routine"
    )
    manifest = root / "manifest.json"
    dataset = load_evidence_gate_benchmark(
        manifest,
        expected_manifest_sha256=_sha256(manifest),
        require_reviewed=False,
    )

    assert Counter(case.category for case in dataset) == {
        "fully_answerable": 10,
        "all_branches_unavailable": 4,
        "authorization_filtered": 4,
        "background_only": 4,
        "completely_unanswerable": 4,
        "conflicting": 4,
        "missing_business_record": 4,
        "missing_version_or_date": 4,
        "partially_answerable": 4,
        "prompt_injection": 4,
        "single_branch_unavailable": 4,
    }
    assert Counter(case.required_fixture for case in dataset) == {
        "standard": 26,
        "all_retrieval_branches_unavailable": 4,
        "bm25_unavailable_dense_graph_available": 4,
        "equal_authority_conflicting_documents": 4,
        "finance_only_user_against_hr_document": 4,
        "frozen_corpus_absence_check": 4,
        "prompt_injection_safety_fixture": 4,
    }

    answer_categories = {
        "fully_answerable",
        "partially_answerable",
        "single_branch_unavailable",
    }
    missing_information_categories = {
        "background_only",
        "missing_business_record",
        "missing_version_or_date",
        "partially_answerable",
    }
    non_citable_control_categories = {
        "all_branches_unavailable",
        "authorization_filtered",
        "completely_unanswerable",
        "prompt_injection",
    }
    for case in dataset:
        assert case.id
        assert case.department_id
        assert case.source_benchmark_ids
        assert set(case.expected_branch_availability) == {"dense", "bm25", "graph"}
        assert case.expected_response_status
        assert case.expected_evidence_states
        assert case.expected_reason_codes
        assert reviewed_case_contract_errors(case) == ()

        if case.category == "all_branches_unavailable":
            assert case.expected_source_document_ids == ()
            assert case.expected_evidence_context_ids == ()
            assert case.expected_evidence_sections == ()
        else:
            # Refusal/safety cases can carry reviewed anchor evidence without
            # making that evidence citable as an ordinary answer.
            assert case.expected_source_document_ids
            assert case.expected_evidence_context_ids
            assert case.expected_evidence_sections

        if case.category in answer_categories:
            assert case.expected_citation_context_ids
        else:
            assert case.expected_citation_context_ids == ()
        if case.category in missing_information_categories:
            assert case.expected_missing_information_fields
        else:
            assert case.expected_missing_information_fields == ()
        if case.category in non_citable_control_categories:
            assert case.expected_refusal is True


def test_manifest_or_review_hash_mismatch_fails_closed(tmp_path: Path) -> None:
    manifest_path = _write_bundle(tmp_path / "evidence")
    expected_manifest_hash = _sha256(manifest_path)
    (manifest_path.parent / "cases.jsonl").write_text("{}\n", encoding="utf-8")

    with pytest.raises(EvidenceGateBenchmarkError, match="cases_sha256"):
        load_evidence_gate_benchmark(
            manifest_path,
            expected_manifest_sha256=expected_manifest_hash,
        )

    with pytest.raises(EvidenceGateBenchmarkError, match="manifest_sha256"):
        load_evidence_gate_benchmark(
            manifest_path,
            expected_manifest_sha256="0" * 64,
        )


def test_pending_human_review_can_be_linted_but_not_scored(tmp_path: Path) -> None:
    manifest_path = _write_bundle(tmp_path / "evidence", reviewed=False)

    candidate = load_evidence_gate_benchmark(
        manifest_path,
        expected_manifest_sha256=_sha256(manifest_path),
        require_reviewed=False,
    )
    assert len(candidate) == len(REQUIRED_EVIDENCE_GATE_CATEGORIES)

    with pytest.raises(EvidenceGateBenchmarkError, match="human review is not approved"):
        load_evidence_gate_benchmark(
            manifest_path,
            expected_manifest_sha256=_sha256(manifest_path),
        )
