"""评测异常记录的人工抽检（spot check）支持。

两个职责：
1. 异常自动分类——从已落盘的评测产物（responses.jsonl + ragas_scores.jsonl）
   识别值得人工复核的样本，给出有界的异常类型清单；分类是只读启发式，
   绝不修改任何评测产物。
2. 人工抽检标记的持久化——把管理员对单条样本的复核结论写入运行产物目录
   旁边的 ``spot-checks.json``（owner-only 0600，原子写入），作为离线证据
   与前端抽检面板的数据源。产物与 run 一一对应，不引入额外存储依赖。
"""

from __future__ import annotations

import json
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from evaluation.benchmarks.result_reader import (
    EvaluationResultError,
    RUN_ID_PATTERN,
    _load_ragas_scores,
    _read_jsonl,
    _resolve_run_dir,
)

# 异常类型的有界枚举（稳定值，前端据此渲染标签）
SPOT_CHECK_ANOMALY_CODES = (
    "scoring_failed",          # 评测执行失败（status != succeeded）
    "ragas_metric_failed",     # RAGAS 指标评分失败（status=failed）
    "ragas_low_score",         # 任一已评指标低于低分阈值
    "contract_mismatch",       # 行为契约（响应状态/证据状态）与预期不符
    "evidence_recall_gap",     # 期望证据上下文未全部命中
)  # noqa: E123
_LOW_SCORE_THRESHOLD = 0.5  # 任一 RAGAS 指标低于该值视为低分异常
# 抽检结论的有界枚举
SPOT_CHECK_VERDICTS = ("judge_error", "system_issue", "confirmed_ok", "needs_data_fix")
_NOTE_LIMIT = 600  # 抽检备注的最大长度（字符）
_ID_LIMIT = 200  # benchmark_id 的最大长度（字符）
_REVIEWER_LIMIT = 128  # 复核人标识的最大长度（字符）


def _anomaly_codes(
    record: dict[str, Any],
    metric_scores: dict[str, float],
) -> list[str]:
    """按稳定顺序返回单条记录命中的异常类型。"""
    codes: list[str] = []
    if str(record.get("status") or "succeeded") != "succeeded":
        codes.append("scoring_failed")
    if (
        str(record.get("expected_response_status") or "")
        and str(record.get("response_status") or "")
        != str(record.get("expected_response_status"))
    ):
        codes.append("contract_mismatch")
    expected_states = record.get("expected_evidence_states")
    if isinstance(expected_states, list) and expected_states:
        observed_state = record.get("evidence_state")
        if isinstance(observed_state, str) and observed_state not in expected_states:
            codes.append("contract_mismatch")
    expected_ids = record.get("expected_evidence_context_ids")
    if isinstance(expected_ids, list) and expected_ids:
        # 期望绑定是 chunk 级（如 "doc#chunk-7"），但同文档内其他 chunk 也能
        # 支撑同一论断；召回缺口按文档级判定，避免把"检回同文档其他块"误报
        # 为漏检。chunk 级明细仍可从 responses.jsonl 的 retrieved_chunk_ids 查看。
        expected_documents = {
            context_id.split("#", 1)[0]
            for context_id in expected_ids
            if isinstance(context_id, str) and context_id
        }
        retrieved = {
            context_id.split("#", 1)[0]
            for context_id in record.get("retrieved_context_ids") or []
            if isinstance(context_id, str) and context_id
        }
        if not expected_documents <= retrieved:
            codes.append("evidence_recall_gap")
    if metric_scores and any(score < _LOW_SCORE_THRESHOLD for score in metric_scores.values()):
        codes.append("ragas_low_score")
    return codes


def _spot_check_path(run_dir: Path) -> Path:
    """返回运行产物目录内的抽检标记文件路径。"""
    return run_dir / "spot-checks.json"


def _read_spot_checks(run_dir: Path) -> dict[str, Any]:
    """读取抽检标记文件；缺失或损坏时返回空结构（不抛错）。"""
    path = _spot_check_path(run_dir)
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, UnicodeDecodeError):
        return {"version": 1, "reviews": {}}
    if not isinstance(value, dict) or not isinstance(value.get("reviews"), dict):
        return {"version": 1, "reviews": {}}
    return value


def _write_spot_checks(run_dir: Path, payload: dict[str, Any]) -> None:
    """以 owner-only 权限原子写入抽检标记文件。"""
    path = _spot_check_path(run_dir)
    temp_path = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    descriptor = os.open(temp_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_path, path)
        os.chmod(path, 0o600)
    except Exception:
        try:
            temp_path.unlink(missing_ok=True)
        finally:
            raise


def _bounded_review(entry: Any) -> dict[str, Any] | None:
    """把已存的抽检记录修剪为有界输出；不合规条目返回 None。"""
    if not isinstance(entry, dict):
        return None
    verdict = entry.get("verdict")
    if verdict not in SPOT_CHECK_VERDICTS:
        return None
    note = entry.get("note")
    return {
        "verdict": verdict,
        "note": note if isinstance(note, str) and note else "",
        "reviewer_id": str(entry.get("reviewer_id") or "")[:_REVIEWER_LIMIT],
        "reviewed_at": str(entry.get("reviewed_at") or "")[:40],
    }


def list_anomalies(
    results_root: str | Path,
    run_id: str,
) -> dict[str, Any]:
    """返回单次运行中值得人工抽检的样本清单（只读启发式分类）。

    Args:
        results_root: 评测结果根目录。
        run_id: run 标识。
    """
    if not RUN_ID_PATTERN.fullmatch(run_id):
        raise EvaluationResultError("invalid run_id")
    run_dir = _resolve_run_dir(Path(results_root), run_id)
    records, skipped = _read_jsonl(run_dir / "responses.jsonl")
    scores = _load_ragas_scores(run_dir)
    checks = _read_spot_checks(run_dir)
    reviews = checks.get("reviews") if isinstance(checks.get("reviews"), dict) else {}

    anomalies: list[dict[str, Any]] = []
    for record in records:
        benchmark_id = str(record.get("benchmark_id") or "")
        if not benchmark_id:
            continue
        mode = str(record.get("retrieval_mode") or "")
        metric_scores = scores.get((benchmark_id, mode), {})
        codes = _anomaly_codes(record, metric_scores)
        if not codes:
            continue
        review = _bounded_review(reviews.get(benchmark_id))
        response = record.get("response")
        if isinstance(response, str) and len(response) > 200:
            response = response[:200] + "…"
        anomalies.append(
            {
                "benchmark_id": benchmark_id[:_ID_LIMIT],
                "retrieval_mode": mode,
                "category": record.get("category"),
                "anomaly_codes": codes,
                "status": record.get("status"),
                "exception": record.get("exception"),
                "expected_response_status": record.get("expected_response_status"),
                "observed_response_status": record.get("response_status"),
                "expected_evidence_states": record.get("expected_evidence_states"),
                "observed_evidence_state": record.get("evidence_state"),
                "expected_evidence_context_ids": list(
                    record.get("expected_evidence_context_ids") or []
                ),
                "retrieved_context_ids": list(
                    record.get("retrieved_context_ids") or []
                )[:20],
                "question": str(record.get("question") or "")[:300],
                "response": response,
                "ragas_scores": {k: round(v, 4) for k, v in sorted(metric_scores.items())},
                "spot_check": review,
            }
        )
    anomalies.sort(key=lambda item: item["benchmark_id"])
    return {
        "run_id": run_id,
        "total_records": len(records),
        "skipped_lines": skipped,
        "anomaly_count": len(anomalies),
        "spot_checked": sum(1 for item in anomalies if item["spot_check"] is not None),
        "low_score_threshold": _LOW_SCORE_THRESHOLD,
        "anomalies": anomalies,
    }


def submit_spot_check(
    results_root: str | Path,
    run_id: str,
    *,
    benchmark_id: str,
    verdict: str,
    note: str = "",
    reviewer_id: str = "",
) -> dict[str, Any]:
    """写入/更新单条样本的人工抽检结论，返回更新后的清单。

    Args:
        results_root: 评测结果根目录。
        run_id: run 标识。
        benchmark_id: 样本的基准条目 ID。
        verdict: 有界结论枚举之一。
        note: 有限长度的复核备注。
        reviewer_id: 复核人标识（服务端从认证上下文注入，不信任前端）。
    """
    if not RUN_ID_PATTERN.fullmatch(run_id):
        raise EvaluationResultError("invalid run_id")
    if not benchmark_id or len(benchmark_id) > _ID_LIMIT:
        raise EvaluationResultError("invalid benchmark_id")
    if verdict not in SPOT_CHECK_VERDICTS:
        raise EvaluationResultError("invalid verdict")
    run_dir = _resolve_run_dir(Path(results_root), run_id)
    records, _ = _read_jsonl(run_dir / "responses.jsonl")
    if not any(str(r.get("benchmark_id") or "") == benchmark_id for r in records):
        raise EvaluationResultError("benchmark_id not in run")

    checks = _read_spot_checks(run_dir)
    reviews = checks.setdefault("reviews", {})
    reviews[benchmark_id] = {
        "verdict": verdict,
        "note": (note or "")[:_NOTE_LIMIT],
        "reviewer_id": (reviewer_id or "")[:_REVIEWER_LIMIT],
        "reviewed_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    checks["version"] = 1
    _write_spot_checks(run_dir, checks)
    return list_anomalies(results_root, run_id)


__all__ = [
    "SPOT_CHECK_ANOMALY_CODES",
    "SPOT_CHECK_VERDICTS",
    "list_anomalies",
    "submit_spot_check",
]
