"""Deterministic and privacy-bounded four-layer diagnostic report tests."""

from __future__ import annotations

import json

from evaluation.diagnostic.report import build_diagnostic_report
from evaluation.release.quality_gates import aggregate_quality_report


def _outcome(case: str, category: str, score: float) -> dict[str, object]:
    return {
        "benchmark_id": case,
        "category": category,
        "metric": "response_route_correctness",
        "metric_family": "deterministic",
        "status": "scored",
        "score": score,
        "direction": "minimum",
        "hard_gate": False,
    }


def test_category_macro_has_equal_weight_and_route_confusion_is_bounded() -> None:
    records = [
        {
            "benchmark_id": f"answer-{index}",
            "category": "fully_answerable",
            "status": "succeeded",
            "expected_response_status": "answered",
            "response_status": "answered",
        }
        for index in range(10)
    ]
    records.append(
        {
            "benchmark_id": "conflict-1",
            "category": "conflicting",
            "status": "succeeded",
            "expected_response_status": "conflicting_evidence",
            "response_status": "answered",
        }
    )
    outcomes = [
        *[_outcome(f"answer-{index}", "fully_answerable", 1.0) for index in range(10)],
        _outcome("conflict-1", "conflicting", 0.0),
    ]

    report = build_diagnostic_report(
        records=records,
        metric_outcomes=outcomes,
        retrieval_summary={"schema_version": "stage-retrieval-metrics-v1", "stages": {}},
        isolation_evidence={
            "cross_snapshot_contamination": False,
            "unauthorized_oracle_access": False,
        },
    )

    metric = report["answer"]["metrics"]["response_route_correctness"]
    assert metric["micro_mean"] == 10 / 11
    assert metric["category_macro_mean"] == 0.5
    assert report["routing"]["by_category"]["conflicting"]["confusion"] == {
        "conflicting_evidence->answered": 1
    }
    assert report["routing"]["by_category"]["conflicting"]["failing_case_ids"] == [
        "conflict-1"
    ]


def test_safety_hard_gates_cannot_be_compensated_by_answer_quality() -> None:
    safety = {
        "benchmark_id": "auth-1",
        "category": "authorization_filtered",
        "metric": "cross_scope_leakage_rate",
        "metric_family": "safety",
        "status": "scored",
        "score": 1.0,
        "direction": "maximum",
        "hard_gate": True,
    }
    report = build_diagnostic_report(
        records=[
            {
                "benchmark_id": "auth-1",
                "category": "authorization_filtered",
                "status": "succeeded",
                "expected_response_status": "insufficient_evidence",
                "response_status": "insufficient_evidence",
            }
        ],
        metric_outcomes=[safety],
        retrieval_summary={"schema_version": "stage-retrieval-metrics-v1", "stages": {}},
        isolation_evidence={
            "cross_snapshot_contamination": False,
            "unauthorized_oracle_access": False,
        },
    )

    assert report["safety"]["hard_gates"]["cross_scope_leakage_rate"]["passed"] is False
    assert report["decision"] == "blocked"
    assert "overall_quality_score" not in report


def test_report_is_replay_deterministic_private_and_additive_to_legacy_quality() -> None:
    records = [
        {
            "benchmark_id": "case-1",
            "retrieval_mode": "vector",
            "category": "fully_answerable",
            "status": "succeeded",
            "expected_response_status": "answered",
            "response_status": "answered",
            "question": "private question",
            "response": "private answer",
            "contexts": [{"content": "private context", "source": "/private/a.pdf"}],
            "retrieved_context_ids": ["doc-a"],
            "reference_context_ids": ["doc-a"],
            "expected_refusal": False,
            "refused": False,
            "latency_ms": 1.0,
        }
    ]
    inputs = {
        "metric_outcomes": [_outcome("case-1", "fully_answerable", 1.0)],
        "retrieval_summary": {
            "schema_version": "stage-retrieval-metrics-v1",
            "credentials": "private token",
            "stages": {
                "dense": {
                    "exact_evidence": {
                        "status_counts": {"scored": 1},
                        "scored": 1,
                        "mean_recall_at_k": {"1": 1.0},
                        "candidate_text": "private candidate text",
                        "absolute_path": "/private/candidate.json",
                    }
                }
            },
        },
        "isolation_evidence": {
            "cross_snapshot_contamination": False,
            "unauthorized_oracle_access": False,
        },
    }
    first = build_diagnostic_report(records=records, **inputs)
    second = build_diagnostic_report(records=records, **inputs)
    assert first == second
    assert len(first["report_sha256"]) == 64
    serialized = json.dumps(first, sort_keys=True)
    assert "private question" not in serialized
    assert "private answer" not in serialized
    assert "private context" not in serialized
    assert "/private/a.pdf" not in serialized
    assert "private token" not in serialized
    assert "private candidate text" not in serialized
    assert "/private/candidate.json" not in serialized

    metadata = {
        "run_id": "run-1",
        "benchmark_sha256": "a" * 64,
        "benchmark_source": "benchmark.jsonl",
        "code_revision": "b" * 40,
        "retrieval_modes": ["vector"],
        "started_at": "2026-08-24T00:00:00Z",
    }
    legacy = aggregate_quality_report(metadata, records, smoke_threshold=1)
    enriched = aggregate_quality_report(
        metadata, records, smoke_threshold=1, diagnostic_inputs=inputs
    )
    assert "diagnostic" not in legacy
    assert enriched["diagnostic"] == first
