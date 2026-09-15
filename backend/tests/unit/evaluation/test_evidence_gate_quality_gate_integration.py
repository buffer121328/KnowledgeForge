"""Acceptance tests for evidence metrics in the shared offline release gate."""

from __future__ import annotations

import json
from pathlib import Path

from evaluation.evidence_gate.metrics import REQUIRED_RUN_IDENTITY_FIELDS
from evaluation.release.quality_gates import aggregate_quality_report, evaluate_release_gate
from evaluation.scripts.run_quality_gate import main as run_quality_gate_main


def _identity() -> dict[str, object]:
    return {
        field: ["vector"] if field == "retrieval_modes" else "identity-v1"
        for field in REQUIRED_RUN_IDENTITY_FIELDS
    }


def test_quality_report_aggregates_evidence_metrics_and_blocks_failed_threshold() -> None:
    metadata = {
        "run_id": "run-1",
        "benchmark_sha256": "a" * 64,
        "benchmark_source": "cases.jsonl",
        "started_at": "2026-08-03T00:00:00Z",
        "retrieval_modes": ["vector"],
        "run_identity": _identity(),
    }
    records = [
        {
            "benchmark_id": "eg-no-answer",
            "retrieval_mode": "vector",
            "category": "completely_unanswerable",
            "status": "succeeded",
            "expected_refusal": True,
            "refused": False,
            "expected_response_status": "insufficient_evidence",
            "response_status": "answered",
            "expected_citation_context_ids": [],
            "claims": [
                {"claim_id": "claim-1", "material": True, "citation_ids": []}
            ],
            "citations": [],
            "grounding_result": {"passed": False},
            "retrieved_context_ids": [],
            "reference_context_ids": [],
            "latency_ms": 10,
        }
    ]

    report = aggregate_quality_report(metadata, records, smoke_threshold=1)
    decision = evaluate_release_gate(
        report,
        evidence_thresholds={
            "no_answer_hallucination_rate": 0.0,
            "groundedness_pass_rate": 1.0,
        },
    )

    assert report["evidence_gate"]["run_identity"] == _identity()
    assert report["evidence_gate"]["metrics"]["no_answer_hallucination_rate"] == 1.0
    assert decision["decision"] == "blocked"
    assert {reason["field"] for reason in decision["reasons"]} >= {
        "evidence_gate.metrics.no_answer_hallucination_rate",
        "evidence_gate.metrics.groundedness_pass_rate",
    }


def test_quality_gate_cli_enforces_evidence_threshold_file(tmp_path: Path) -> None:
    metadata_path = tmp_path / "run_metadata.json"
    responses_path = tmp_path / "responses.jsonl"
    thresholds_path = tmp_path / "thresholds.json"
    output_path = tmp_path / "quality_report.json"
    metadata_path.write_text(
        json.dumps(
            {
                "run_id": "run-cli",
                "benchmark_sha256": "b" * 64,
                "benchmark_source": "cases.jsonl",
                "started_at": "2026-08-03T00:00:00Z",
                "retrieval_modes": ["vector"],
                "run_identity": _identity(),
            }
        ),
        encoding="utf-8",
    )
    responses_path.write_text(
        json.dumps(
            {
                "benchmark_id": "eg-cli",
                "retrieval_mode": "vector",
                "category": "completely_unanswerable",
                "status": "succeeded",
                "expected_refusal": True,
                "refused": False,
                "expected_response_status": "insufficient_evidence",
                "response_status": "answered",
                "expected_citation_context_ids": [],
                "claims": [],
                "citations": [],
                "grounding_result": {"passed": False},
                "retrieved_context_ids": [],
                "reference_context_ids": [],
                "latency_ms": 10,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    thresholds_path.write_text(
        json.dumps({"no_answer_hallucination_rate": 0.0}), encoding="utf-8"
    )

    exit_code = run_quality_gate_main(
        [
            "--metadata-json",
            str(metadata_path),
            "--responses-jsonl",
            str(responses_path),
            "--output-json",
            str(output_path),
            "--smoke-threshold",
            "1",
            "--release-gate",
            "--evidence-thresholds-json",
            str(thresholds_path),
        ]
    )

    assert exit_code == 3
    assert json.loads(output_path.read_text())["evidence_gate"]["metrics"][
        "no_answer_hallucination_rate"
    ] == 1.0
