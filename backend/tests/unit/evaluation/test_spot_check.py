"""评测异常人工抽检的单元测试：分类启发式、持久化与有界输出。"""

from __future__ import annotations

import json
import os
import stat
from pathlib import Path

import pytest

from evaluation.benchmarks.spot_check import (
    SPOT_CHECK_ANOMALY_CODES,
    list_anomalies,
    submit_spot_check,
)


def _write_run(
    root: Path,
    run_id: str,
    *,
    records: list[dict],
    scores: list[dict] | None = None,
) -> None:
    """构造一个最小可读的 run 产物目录。"""
    run_dir = root / run_id
    run_dir.mkdir(parents=True)
    lines = "\n".join(json.dumps(r, ensure_ascii=False) for r in records)
    (run_dir / "responses.jsonl").write_text(lines + "\n", encoding="utf-8")
    if scores is not None:
        lines = "\n".join(json.dumps(s, ensure_ascii=False) for s in scores)
        (run_dir / "ragas_scores.jsonl").write_text(lines + "\n", encoding="utf-8")


def _record(benchmark_id: str, **overrides: object) -> dict:
    base = {
        "benchmark_id": benchmark_id,
        "retrieval_mode": "dense_bm25_graph",
        "category": "fully_answerable",
        "status": "succeeded",
        "question": "问题",
        "response": "回答",
        "expected_response_status": "answered",
        "response_status": "answered",
        "expected_evidence_states": ["direct_evidence"],
        "evidence_state": "direct_evidence",
        "expected_evidence_context_ids": ["doc-a#chunk-0"],
        "retrieved_context_ids": ["doc-a#chunk-0"],
    }
    base.update(overrides)
    return base


def test_clean_run_yields_no_anomalies(tmp_path: Path) -> None:
    _write_run(tmp_path, "20260101T000000Z-00000000", records=[_record("case-1")])
    result = list_anomalies(tmp_path, "20260101T000000Z-00000000")
    assert result["anomaly_count"] == 0
    assert result["total_records"] == 1
    assert result["anomalies"] == []


def test_classifies_status_and_state_and_recall_gaps(tmp_path: Path) -> None:
    _write_run(
        tmp_path,
        "20260101T000000Z-00000000",
        records=[
            _record(
                "case-fail",
                status="failed",
                exception="boom",
                response_status="insufficient_evidence",
                expected_evidence_context_ids=["doc-a#chunk-0"],
                retrieved_context_ids=[],
                evidence_state="insufficient_evidence",
            ),
        ],
    )
    result = list_anomalies(tmp_path, "20260101T000000Z-00000000")
    assert result["anomaly_count"] == 1
    codes = result["anomalies"][0]["anomaly_codes"]
    assert "scoring_failed" in codes
    assert "contract_mismatch" in codes
    assert "evidence_recall_gap" in codes


def test_recall_gap_is_judged_at_document_level(tmp_path: Path) -> None:
    """期望 chunk 未命中但同文档其他 chunk 已检回时，不记召回缺口。"""
    _write_run(
        tmp_path,
        "20260101T000000Z-00000000",
        records=[
            _record(
                "case-other-chunk-hit",
                expected_evidence_context_ids=["doc-a#chunk-0"],
                retrieved_context_ids=["doc-a", "doc-b"],
            ),
        ],
    )
    result = list_anomalies(tmp_path, "20260101T000000Z-00000000")
    assert result["anomaly_count"] == 0


def test_recall_gap_flags_missing_document_entirely(tmp_path: Path) -> None:
    """整份期望文档都未检回时，仍必须记召回缺口。"""
    _write_run(
        tmp_path,
        "20260101T000000Z-00000000",
        records=[
            _record(
                "case-doc-missing",
                expected_evidence_context_ids=["doc-a#chunk-0", "doc-c#chunk-2"],
                retrieved_context_ids=["doc-a", "doc-b"],
            ),
        ],
    )
    result = list_anomalies(tmp_path, "20260101T000000Z-00000000")
    assert result["anomaly_count"] == 1
    assert "evidence_recall_gap" in result["anomalies"][0]["anomaly_codes"]


def test_low_score_metric_triggers_anomaly(tmp_path: Path) -> None:
    _write_run(
        tmp_path,
        "20260101T000000Z-00000000",
        records=[_record("case-low")],
        scores=[
            {
                "benchmark_id": "case-low",
                "retrieval_mode": "dense_bm25_graph",
                "metric": "factual_correctness",
                "score": 0.13,
                "status": "scored",
            },
            {
                "benchmark_id": "case-low",
                "retrieval_mode": "dense_bm25_graph",
                "metric": "faithfulness",
                "score": 0.97,
                "status": "scored",
            },
        ],
    )
    result = list_anomalies(tmp_path, "20260101T000000Z-00000000")
    assert result["anomaly_count"] == 1
    item = result["anomalies"][0]
    assert "ragas_low_score" in item["anomaly_codes"]
    assert item["ragas_scores"]["factual_correctness"] == 0.13


def test_submit_spot_check_persists_owner_only_and_overwrites(tmp_path: Path) -> None:
    run_id = "20260101T000000Z-00000000"
    _write_run(tmp_path, run_id, records=[_record("case-1", status="failed")])

    first = submit_spot_check(
        tmp_path,
        run_id,
        benchmark_id="case-1",
        verdict="judge_error",
        note="回答覆盖完整,评分器误判",
        reviewer_id="user-1",
    )
    assert first["spot_checked"] == 1
    item = first["anomalies"][0]["spot_check"]
    assert item["verdict"] == "judge_error"
    assert item["reviewer_id"] == "user-1"
    assert item["note"] == "回答覆盖完整,评分器误判"

    path = tmp_path / run_id / "spot-checks.json"
    assert stat.S_IMODE(path.stat().st_mode) == 0o600

    # 重复提交覆盖旧结论
    second = submit_spot_check(
        tmp_path,
        run_id,
        benchmark_id="case-1",
        verdict="confirmed_ok",
        reviewer_id="user-2",
    )
    assert second["spot_checked"] == 1
    assert second["anomalies"][0]["spot_check"]["verdict"] == "confirmed_ok"


def test_submit_rejects_unknown_benchmark_and_verdict(tmp_path: Path) -> None:
    run_id = "20260101T000000Z-00000000"
    _write_run(tmp_path, run_id, records=[_record("case-1", status="failed")])
    from evaluation.benchmarks.result_reader import EvaluationResultError

    with pytest.raises(EvaluationResultError, match="benchmark_id"):
        submit_spot_check(
            tmp_path, run_id, benchmark_id="missing", verdict="judge_error"
        )
    with pytest.raises(EvaluationResultError, match="verdict"):
        submit_spot_check(
            tmp_path, run_id, benchmark_id="case-1", verdict="not-a-verdict"
        )


def test_corrupt_spot_check_file_is_tolerated(tmp_path: Path) -> None:
    run_id = "20260101T000000Z-00000000"
    _write_run(tmp_path, run_id, records=[_record("case-1", status="failed")])
    (tmp_path / run_id / "spot-checks.json").write_text("{broken", encoding="utf-8")

    result = list_anomalies(tmp_path, run_id)
    assert result["anomaly_count"] == 1
    assert result["anomalies"][0]["spot_check"] is None


def test_anomaly_codes_enum_is_bounded() -> None:
    assert set(SPOT_CHECK_ANOMALY_CODES) == {
        "scoring_failed",
        "ragas_metric_failed",
        "ragas_low_score",
        "contract_mismatch",
        "evidence_recall_gap",
    }
    assert os.environ.get("SPOT_CHECK_NO_ENV_USE", "") == ""
