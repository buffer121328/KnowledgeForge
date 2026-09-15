"""Acceptance tests for deterministic evidence-gate candidate preparation."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import pytest

from evaluation.evidence_gate.benchmark import (
    EvidenceGateBenchmarkError,
    REQUIRED_EVIDENCE_GATE_CATEGORIES,
    load_evidence_gate_benchmark,
    sha256_file,
)
from evaluation.evidence_gate.preparation import prepare_evidence_gate_candidates


PROJECT_ROOT = Path(__file__).resolve().parents[4]
COMPANY_DEMO_ROOT = PROJECT_ROOT / "backend/evaluation/data/company-demo"


def test_preparation_expands_and_binds_review_candidates(tmp_path: Path) -> None:
    output = tmp_path / "evidence-gates"

    report = prepare_evidence_gate_candidates(COMPANY_DEMO_ROOT, output)

    manifest_path = output / "manifest.json"
    dataset = load_evidence_gate_benchmark(
        manifest_path,
        expected_manifest_sha256=sha256_file(manifest_path),
        require_reviewed=False,
    )
    category_counts = Counter(case.category for case in dataset)
    assert len(dataset) == 100
    assert set(category_counts) == REQUIRED_EVIDENCE_GATE_CATEGORIES
    assert category_counts["fully_answerable"] == 20
    assert all(
        count == 8
        for category, count in category_counts.items()
        if category != "fully_answerable"
    )
    assert report["review_status"] == "pending_human_review"
    assert report["case_count"] == len(dataset)
    assert all(
        not context_id.startswith("candidate-")
        for case in dataset
        for context_id in case.expected_evidence_context_ids
    )
    assert all(
        "#chunk-" in context_id
        for case in dataset
        for context_id in case.expected_citation_context_ids
    )
    assert any(case.expected_missing_information_fields for case in dataset)
    raw_cases = [
        json.loads(line)
        for line in (output / "cases.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert all(case.get("department_id") for case in raw_cases)
    assert any(case.get("department_id") == "administration" for case in raw_cases)

    review = json.loads((output / "review-record.json").read_text())
    assert review["review_status"] == "pending_human_review"
    assert review["reviewed_case_ids"] == []
    assert (output / "pre-review-report.json").is_file()
    assert (output / "context-catalog.jsonl").is_file()
    automated = json.loads((output / "automated-pre-review.json").read_text())
    assert automated["status"] == "passed_pending_human_review"
    assert automated["human_approval"] is False
    assert all(automated["checks"].values())
    assert automated["checks"]["case_count_exactly_100"] is True
    assert automated["checks"]["fully_answerable_at_most_20"] is True
    assert automated["checks"]["other_categories_at_least_8"] is True
    assert automated["checks"]["all_cases_have_department_ownership"] is True


def test_preparation_freezes_supplemental_artifact_hashes(tmp_path: Path) -> None:
    output = tmp_path / "evidence-gates"
    report = prepare_evidence_gate_candidates(COMPANY_DEMO_ROOT, output)
    (output / "automated-pre-review.json").write_text("{}\n", encoding="utf-8")

    with pytest.raises(
        EvidenceGateBenchmarkError, match="automated_pre_review_sha256"
    ):
        load_evidence_gate_benchmark(
            output / "manifest.json",
            expected_manifest_sha256=report["manifest_sha256"],
            require_reviewed=False,
        )
