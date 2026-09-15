"""Acceptance tests for draft evidence-gate calibration artifacts."""

from __future__ import annotations

import json
from pathlib import Path

from evaluation.evidence_gate.calibration import build_calibration_draft


_CITATION_METRICS = ("claim_citation_coverage", "citation_correctness")


def _quality_report(*, metric_overrides: dict[str, object] | None = None) -> dict[str, object]:
    metrics: dict[str, object] = {
        "refusal_precision": 1.0,
        "refusal_recall": 1.0,
        "no_answer_hallucination_rate": 0.0,
        "answerable_false_refusal_rate": 0.025,
        "partial_answer_recognition_rate": 1.0,
        "conflict_recognition_rate": 1.0,
        "claim_citation_coverage": 1.0,
        "citation_correctness": 1.0,
        "groundedness_pass_rate": 1.0,
    }
    metrics.update(metric_overrides or {})
    return {
        "evidence_gate": {
            "counts": {"completed": 40},
            "metrics": metrics,
            "run_identity": {"run_id": "run-1", "dataset_version": "v1"},
        }
    }


def _write_quality_report(path: Path, report: dict[str, object]) -> Path:
    path.write_text(json.dumps(report), encoding="utf-8")
    return path


def test_calibration_accepts_baseline_off_null_citation_metrics_when_shadow_measures_them(
    tmp_path: Path,
) -> None:
    baseline = _write_quality_report(
        tmp_path / "baseline.json",
        _quality_report(metric_overrides={name: None for name in _CITATION_METRICS}),
    )
    shadow = _write_quality_report(tmp_path / "shadow.json", _quality_report())

    result = build_calibration_draft(
        {"baseline": baseline, "shadow": shadow},
        output_dir=tmp_path / "calibration",
        calibration_version="evidence-calibration-2026-08-v1",
    )

    assert result["status"] == "completed"
    assert result["measured"] is True
    assert result["calibration_version"] == "evidence-calibration-2026-08-v1"
    thresholds = json.loads(result["thresholds_path"].read_text())
    assert thresholds["no_answer_hallucination_rate"] == 0.01
    assert thresholds["claim_citation_coverage"] == 1.0
    record = json.loads(result["record_path"].read_text())
    assert record["missing_metrics"] == []
    assert record["stage_metric_requirements"] == {
        "baseline": [
            "answerable_false_refusal_rate",
            "conflict_recognition_rate",
            "groundedness_pass_rate",
            "no_answer_hallucination_rate",
            "partial_answer_recognition_rate",
            "refusal_precision",
            "refusal_recall",
        ],
        "shadow": [
            "answerable_false_refusal_rate",
            "citation_correctness",
            "claim_citation_coverage",
            "conflict_recognition_rate",
            "groundedness_pass_rate",
            "no_answer_hallucination_rate",
            "partial_answer_recognition_rate",
            "refusal_precision",
            "refusal_recall",
        ],
    }
    assert [measurement["stage"] for measurement in record["measurements"]] == [
        "baseline",
        "shadow",
    ]


def test_calibration_blocks_when_shadow_citation_metric_is_missing(tmp_path: Path) -> None:
    baseline = _write_quality_report(
        tmp_path / "baseline.json",
        _quality_report(metric_overrides={name: None for name in _CITATION_METRICS}),
    )
    shadow = _write_quality_report(
        tmp_path / "shadow.json",
        _quality_report(metric_overrides={"citation_correctness": None}),
    )

    result = build_calibration_draft(
        {"baseline": baseline, "shadow": shadow},
        output_dir=tmp_path / "calibration",
        calibration_version="evidence-calibration-2026-08-v1",
    )

    assert result["status"] == "blocked_missing_measurements"
    assert result["measured"] is False
    assert result["missing_metrics"] == ["citation_correctness"]
    record = json.loads(result["record_path"].read_text())
    assert record["missing_measurements"] == [
        {"metric": "citation_correctness", "stage": "shadow"}
    ]
    assert not result["thresholds_path"].exists()


def test_calibration_blocks_when_baseline_metric_remains_applicable_but_missing(
    tmp_path: Path,
) -> None:
    baseline = _write_quality_report(
        tmp_path / "baseline.json",
        _quality_report(
            metric_overrides={
                "refusal_recall": None,
                **{name: None for name in _CITATION_METRICS},
            }
        ),
    )
    shadow = _write_quality_report(tmp_path / "shadow.json", _quality_report())

    result = build_calibration_draft(
        {"baseline": baseline, "shadow": shadow},
        output_dir=tmp_path / "calibration",
        calibration_version="evidence-calibration-2026-08-v1",
    )

    assert result["status"] == "blocked_missing_measurements"
    assert result["missing_metrics"] == ["refusal_recall"]
    record = json.loads(result["record_path"].read_text())
    assert record["missing_measurements"] == [
        {"metric": "refusal_recall", "stage": "baseline"}
    ]


def test_unnamed_reports_remain_strict_when_citation_metrics_are_missing(
    tmp_path: Path,
) -> None:
    quality_path = _write_quality_report(
        tmp_path / "quality.json",
        _quality_report(metric_overrides={name: None for name in _CITATION_METRICS}),
    )

    result = build_calibration_draft(
        [quality_path],
        output_dir=tmp_path / "calibration",
        calibration_version="evidence-calibration-2026-08-v1",
    )

    assert result["status"] == "blocked_missing_measurements"
    assert result["missing_metrics"] == sorted(_CITATION_METRICS)


def test_calibration_stays_blocked_when_real_metrics_are_missing(tmp_path: Path) -> None:
    quality_path = _write_quality_report(tmp_path / "quality.json", {"evidence_gate": {}})

    result = build_calibration_draft(
        [quality_path],
        output_dir=tmp_path / "calibration",
        calibration_version="evidence-calibration-2026-08-v1",
    )

    assert result["status"] == "blocked_missing_measurements"
    assert result["measured"] is False
    assert not result["thresholds_path"].exists()


def _source_evidence(*, version: str = "v1", report_hash: str = "a" * 64) -> dict[str, object]:
    ragas = {
        name: {
            "total": 100,
            "scored": 100,
            "not_applicable": 0,
            "skipped": 0,
            "failed": 0,
            "mean": 0.9,
        }
        for name in ("faithfulness", "factual_correctness", "context_precision", "context_recall")
    }
    return {
        "quality_report_sha256": report_hash,
        "responses_sha256": "b" * 64,
        "manifest_sha256": "c" * 64,
        "expected_manifest_sha256": "c" * 64,
        "dataset_id": "evidence-gates-v1",
        "version": version,
        "case_count": 100,
        "gate_mode": "off",
        "run_id": "run-1",
        "ragas_coverage": ragas,
        "report_failed_records": 0,
        "report_invalid_provenance": 0,
        "run_classification": "baseline",
    }


def test_calibration_ignores_diagnostic_ragas_metrics_outside_release_contract(
    tmp_path: Path,
) -> None:
    baseline = _write_quality_report(
        tmp_path / "baseline.json",
        _quality_report(metric_overrides={name: None for name in _CITATION_METRICS}),
    )
    shadow = _write_quality_report(tmp_path / "shadow.json", _quality_report())
    baseline_source = _source_evidence(report_hash="d" * 64)
    shadow_source = {**_source_evidence(report_hash="e" * 64), "gate_mode": "shadow"}
    for source in (baseline_source, shadow_source):
        source["ragas_coverage"]["answer_relevancy"] = {
            "total": 100,
            "scored": 40,
            "not_applicable": 60,
            "skipped": 0,
            "failed": 0,
            "mean": 0.8,
        }

    result = build_calibration_draft(
        {"baseline": baseline, "shadow": shadow},
        output_dir=tmp_path / "calibration",
        calibration_version="cal-v1",
        source_evidence={"baseline": baseline_source, "shadow": shadow_source},
    )

    assert result["status"] == "completed"
    record = json.loads(result["record_path"].read_text())
    assert set(record["source_identities"]["baseline"]["ragas"]) == {
        "faithfulness",
        "factual_correctness",
        "context_precision",
        "context_recall",
    }


def test_calibration_records_hash_bound_sources_ragas_and_shadow_deltas(tmp_path: Path) -> None:
    baseline = _write_quality_report(tmp_path / "baseline.json", _quality_report(
        metric_overrides={"refusal_precision": 0.9, **{name: None for name in _CITATION_METRICS}},
    ))
    shadow = _write_quality_report(tmp_path / "shadow.json", _quality_report(
        metric_overrides={"refusal_precision": 0.95},
    ))
    result = build_calibration_draft(
        {"baseline": baseline, "shadow": shadow}, output_dir=tmp_path / "calibration",
        calibration_version="cal-v1",
        source_evidence={
            "baseline": _source_evidence(report_hash="d" * 64),
            "shadow": {**_source_evidence(report_hash="e" * 64), "gate_mode": "shadow", "run_id": "run-2"},
        },
    )
    assert result["status"] == "completed"
    record = json.loads(result["record_path"].read_text())
    assert record["source_identities"]["baseline"]["quality_report_sha256"] == "d" * 64
    assert record["source_identities"]["shadow"]["ragas"]["faithfulness"]["mean"] == 0.9
    assert record["metric_deltas"]["refusal_precision"] == {
        "baseline": 0.9, "shadow": 0.95, "delta": 0.04999999999999993,
    }
    assert record["metric_deltas"]["citation_correctness"]["delta"] is None


def test_calibration_rejects_unbound_source_evidence(tmp_path: Path) -> None:
    baseline = _write_quality_report(tmp_path / "baseline.json", _quality_report(
        metric_overrides={name: None for name in _CITATION_METRICS},
    ))
    shadow = _write_quality_report(tmp_path / "shadow.json", _quality_report())
    result = build_calibration_draft(
        {"baseline": baseline, "shadow": shadow}, output_dir=tmp_path / "calibration",
        calibration_version="cal-v1",
        source_evidence={
            "baseline": _source_evidence(report_hash="not-a-hash"),
            "shadow": _source_evidence(version="v2"),
        },
    )
    assert result["status"] == "blocked_invalid_evidence"
    assert {item["code"] for item in result["integrity_errors"]} == {"calibration_source_unbound"}
    assert not result["thresholds_path"].exists()


def test_calibration_rejects_incomparable_source_identities(tmp_path: Path) -> None:
    baseline = _write_quality_report(tmp_path / "baseline.json", _quality_report(
        metric_overrides={name: None for name in _CITATION_METRICS},
    ))
    shadow = _write_quality_report(tmp_path / "shadow.json", _quality_report())
    result = build_calibration_draft(
        {"baseline": baseline, "shadow": shadow}, output_dir=tmp_path / "calibration",
        calibration_version="cal-v1",
        source_evidence={
            "baseline": _source_evidence(),
            "shadow": _source_evidence(version="v2"),
        },
    )
    assert result["status"] == "blocked_invalid_evidence"
    assert {item["code"] for item in result["integrity_errors"]} == {"calibration_source_incomparable"}
