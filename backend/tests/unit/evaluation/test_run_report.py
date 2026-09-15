"""评测报告汇总的单元测试：分层指标、百分比转换与注意事项生成。"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from evaluation.benchmarks.run_report import build_run_report
from evaluation.benchmarks.result_reader import EvaluationResultError


def _write_run(root: Path, run_id: str, *, report: dict, ragas: dict) -> None:
    """构造最小可读的 run 产物目录。"""
    run_dir = root / run_id
    run_dir.mkdir(parents=True)
    (run_dir / "run_metadata.json").write_text(
        json.dumps({"started_at": "2026-01-01T00:00:00Z"}), encoding="utf-8"
    )
    (run_dir / "quality-report.json").write_text(
        json.dumps(report, ensure_ascii=False), encoding="utf-8"
    )
    (run_dir / "ragas_scores.summary.json").write_text(
        json.dumps(ragas, ensure_ascii=False), encoding="utf-8"
    )


def _base_report() -> dict:
    return {
        "run_classification": "smoke_only",
        "counts": {"total": 12, "succeeded": 12, "failed": 0, "invalid_provenance": 0},
        "metrics": {"id_based_context_precision": 0.2408, "id_based_context_recall": 0.7778},
        "evidence_gate": {
            "metrics": {
                "groundedness_pass_rate": 0.0,
                "conflict_recognition_rate": 1.0,
                "refusal_precision": 1.0,
                "refusal_recall": 1.0,
            },
            "failure_sample_ids": {
                "groundedness_pass_rate": ["case-a", "case-b", "case-c", "case-d"]
            },
        },
        "category_metrics": {
            "metrics": {
                "unauthorized_citation_rate": {"mean": 0.0, "scored": 2, "total": 2},
                "cross_scope_leakage_rate": {"mean": 0.0, "scored": 2, "total": 2},
            }
        },
    }


def _base_ragas() -> dict:
    return {
        "families": {"ragas": {"failed": 0, "scored": 27, "not_applicable": 81}},
        "metrics": {
            "faithfulness": {"mean": 0.8664, "scored": 3, "total": 12, "failed": 0},
            "factual_correctness": {"mean": 0.3467, "scored": 3, "total": 12, "failed": 0},
            "answer_relevancy": {"mean": 0.8641, "scored": 3, "total": 12, "failed": 0},
            "context_recall": {"mean": 0.725, "scored": 4, "total": 12, "failed": 0},
            "rubrics_score_with_reference": {"mean": 4.6, "scored": 5, "total": 12, "failed": 0},
        },
    }


def test_report_sections_and_percent_conversion(tmp_path: Path) -> None:
    run_id = "20260101T000000Z-00000000"
    _write_run(tmp_path, run_id, report=_base_report(), ragas=_base_ragas())

    result = build_run_report(tmp_path, run_id)

    assert result["run_id"] == run_id
    assert result["counts"]["succeeded"] == 12
    ragas_keys = [entry["key"] for entry in result["sections"]["ragas"]]
    assert ragas_keys == [
        "faithfulness",
        "factual_correctness",
        "answer_relevancy",
        "context_recall",
        "rubrics_score_with_reference",
    ]
    faithfulness = result["sections"]["ragas"][0]
    assert faithfulness["value"] == 0.8664
    assert faithfulness["scored"] == 3

    by_key = {entry["key"]: entry for entry in result["sections"]["evidence"]}
    assert by_key["conflict_recognition_rate"]["value"] == 100.0
    assert by_key["conflict_recognition_rate"]["unit"] == "percent"

    retrieval = {entry["key"]: entry for entry in result["sections"]["retrieval"]}
    assert retrieval["id_based_context_recall"]["value"] == 77.78
    assert retrieval["id_based_context_precision"]["value"] == 24.08

    safety = {entry["key"]: entry for entry in result["sections"]["safety"]}
    assert safety["unauthorized_citation_rate"]["value"] == 0.0
    assert safety["unauthorized_citation_rate"]["direction"] == "lower_is_better"


def test_missing_evidence_metric_reported_as_uncovered(tmp_path: Path) -> None:
    report = _base_report()
    report["evidence_gate"]["metrics"]["groundedness_pass_rate"] = None
    run_id = "20260101T000000Z-00000000"
    _write_run(tmp_path, run_id, report=report, ragas=_base_ragas())

    result = build_run_report(tmp_path, run_id)

    advisory_texts = [item["message"] for item in result["advisories"]]
    assert any("未覆盖" in text for text in advisory_texts)


def test_low_score_and_failures_produce_advisories(tmp_path: Path) -> None:
    report = _base_report()
    report["counts"]["failed"] = 2
    report["counts"]["succeeded"] = 10
    report["counts"]["total"] = 12
    run_id = "20260101T000000Z-00000000"
    ragas = _base_ragas()
    ragas["metrics"]["factual_correctness"]["mean"] = 0.3
    _write_run(tmp_path, run_id, report=report, ragas=ragas)

    result = build_run_report(tmp_path, run_id)

    levels = {item["level"] for item in result["advisories"]}
    assert "critical" in levels
    messages = [item["message"] for item in result["advisories"]]
    assert any("2/12" in text and "失败" in text for text in messages)
    assert any("低于 0.7" in text for text in messages)
    assert any("groundedness_pass_rate" in text and "4 条未通过" in text for text in messages)


def test_safety_metric_above_zero_is_critical(tmp_path: Path) -> None:
    report = _base_report()
    report["category_metrics"]["metrics"]["unauthorized_citation_rate"] = {
        "mean": 0.5,
        "scored": 2,
        "total": 2,
    }
    run_id = "20260101T000000Z-00000000"
    _write_run(tmp_path, run_id, report=report, ragas=_base_ragas())

    result = build_run_report(tmp_path, run_id)

    assert any(
        item["level"] == "critical" and "unauthorized_citation_rate" in item["message"]
        for item in result["advisories"]
    )


def test_invalid_run_id_rejected(tmp_path: Path) -> None:
    with pytest.raises(EvaluationResultError, match="invalid run_id"):
        build_run_report(tmp_path, "../escape")
