"""评测报告汇总：把 RAGAS、证据门与类别专属指标合成一份可读报告。

只读取已落盘的评测产物（quality-report.json + ragas_scores.summary.json），
不引入任何新测量。输出为面向管理员的报告视图：
- 核心指标分层（回答质量 / 检索质量 / 行为契约与安全），率统一为百分比；
- 每个指标带样本覆盖数与数据缺失说明，避免把 not_applicable 误读为低分；
- 注意事项（advisories）：由阈值与失败样本启发式生成，指向需要人工
  关注的位置；异常复核走 /runs/{id}/anomalies 抽检面板。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from evaluation.benchmarks.result_reader import (
    EvaluationResultError,
    RUN_ID_PATTERN,
    _read_json,
    _read_jsonl,
    _resolve_run_dir,
)

# 核心指标的有界白名单（稳定 key，前端据此渲染中文名与分层）
_REPORT_RAGAS_METRICS = (
    "faithfulness",
    "factual_correctness",
    "answer_relevancy",
    "context_recall",
    "rubrics_score_with_reference",
)
_REPORT_EVIDENCE_METRICS = (
    "groundedness_pass_rate",
    "answerable_false_refusal_rate",
    "no_answer_hallucination_rate",
    "conflict_recognition_rate",
    "partial_answer_recognition_rate",
    "refusal_precision",
    "refusal_recall",
)
# 检索侧确定性指标（quality-report.metrics 顶层）
_REPORT_RETRIEVAL_METRICS = ("id_based_context_precision", "id_based_context_recall")
# 安全类类别指标（越界即高危）
_REPORT_SAFETY_METRICS = ("unauthorized_citation_rate", "cross_scope_leakage_rate", "instruction_data_leakage_rate")
# 率型指标（0-1 → 百分比展示）
_RATE_METRICS = frozenset({*_REPORT_EVIDENCE_METRICS, *_REPORT_RETRIEVAL_METRICS, *_REPORT_SAFETY_METRICS})
# 阈值：低于该值的核心指标会触发注意事项
_LOW_SCORE_ADVISORY_THRESHOLD = 0.7


def _percent(value: Any) -> float | None:
    """把 0-1 的率转换为百分比数值；非数值输入返回 None。"""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return round(float(value) * 100, 2)


def _metric_entry(
    key: str,
    *,
    mean: Any,
    scored: Any,
    total: Any,
    failed: Any = None,
    kind: str,
) -> dict[str, Any]:
    """构造有界的报告指标条目（率型指标自动转百分比）。"""
    entry: dict[str, Any] = {
        "key": key,
        "kind": kind,
        "unit": "percent" if key in _RATE_METRICS else "score",
    }
    if key in _RATE_METRICS:
        entry["value"] = _percent(mean)
    else:
        entry["value"] = round(float(mean), 4) if isinstance(mean, (int, float)) and not isinstance(mean, bool) else None
    entry["scored"] = scored if isinstance(scored, int) and not isinstance(scored, bool) else None
    entry["total"] = total if isinstance(total, int) and not isinstance(total, bool) else None
    entry["failed"] = failed if isinstance(failed, int) and not isinstance(failed, bool) else None
    return entry


def _ragas_entries(summary_metrics: dict[str, Any]) -> list[dict[str, Any]]:
    """从 RAGAS summary 提取核心指标条目（保持白名单顺序）。"""
    entries: list[dict[str, Any]] = []
    for key in _REPORT_RAGAS_METRICS:
        metric = summary_metrics.get(key)
        if not isinstance(metric, dict):
            continue
        entries.append(
            _metric_entry(
                key,
                mean=metric.get("mean"),
                scored=metric.get("scored"),
                total=metric.get("total"),
                failed=metric.get("failed"),
                kind="ragas",
            )
        )
    return entries


def _evidence_entries(
    evidence_metrics: dict[str, Any],
    records: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """从证据门块提取核心率型指标条目。

    records 用于识别"测量没有执行"的情况：证据门 off 模式下不产生结构化
    grounding 产物，此时有据回答通过率应标注未覆盖而非 0%。
    """
    entries: list[dict[str, Any]] = []
    grounding_executed: bool | None = None
    if records is not None:
        # 只要任一回答类样本带 grounding_result 即视为已执行测量
        grounding_executed = any(
            isinstance(record.get("grounding_result"), dict)
            and record.get("grounding_result")
            for record in records
        )
    for key in _REPORT_EVIDENCE_METRICS:
        if key not in evidence_metrics:
            continue
        value = evidence_metrics.get(key)
        if value is None:
            # 参考评审未覆盖该指标时如实标注缺失，而不是伪装成 0%
            entries.append(
                {
                    "key": key,
                    "kind": "evidence",
                    "unit": "percent",
                    "value": None,
                    "scored": 0,
                    "total": None,
                    "failed": None,
                }
            )
            continue
        if key == "groundedness_pass_rate" and grounding_executed is False:
            entries.append(
                {
                    "key": key,
                    "kind": "evidence",
                    "unit": "percent",
                    "value": None,
                    "scored": 0,
                    "total": None,
                    "failed": None,
                }
            )
            continue
        entries.append(_metric_entry(key, mean=value, scored=None, total=None, kind="evidence"))
    return entries


def _retrieval_entries(report_metrics: dict[str, Any]) -> list[dict[str, Any]]:
    """提取检索侧确定性指标条目（率转百分比）。"""
    entries: list[dict[str, Any]] = []
    for key in _REPORT_RETRIEVAL_METRICS:
        if key not in report_metrics:
            continue
        entries.append(_metric_entry(key, mean=report_metrics.get(key), scored=None, total=None, kind="retrieval"))
    return entries


def _safety_entries(
    category_metrics: dict[str, Any],
    total_records: int,
) -> list[dict[str, Any]]:
    """提取安全类类别指标（率转百分比；缺失视为未覆盖）。"""
    metrics = category_metrics.get("metrics") if isinstance(category_metrics, dict) else None
    metrics = metrics if isinstance(metrics, dict) else {}
    entries: list[dict[str, Any]] = []
    for key in _REPORT_SAFETY_METRICS:
        metric = metrics.get(key)
        if not isinstance(metric, dict):
            continue
        entry = _metric_entry(
            key,
            mean=metric.get("mean"),
            scored=metric.get("scored"),
            total=metric.get("total"),
            kind="safety",
        )
        # 安全指标按"未发生"为满分的方向解读：0% 是最好。
        entry["direction"] = "lower_is_better"
        if entry["value"] is None and total_records > 0 and int(metric.get("total") or 0) == 0:
            # 该类别不在本次运行中（如未触发注入样本）→ 未覆盖
            entry["value"] = None
        entries.append(entry)
    return entries


def _advisories(
    *,
    counts: dict[str, Any],
    ragas_entries: list[dict[str, Any]],
    evidence_entries: list[dict[str, Any]],
    safety_entries: list[dict[str, Any]],
    failure_sample_ids: dict[str, Any],
    ragas_summary: dict[str, Any],
) -> list[dict[str, str]]:
    """由阈值与失败样本启发式生成注意事项（不臆造未测量的结论）。"""
    advisories: list[dict[str, str]] = []
    failed = counts.get("failed")
    invalid = counts.get("invalid_provenance")
    total = counts.get("total")
    if isinstance(failed, int) and failed > 0:
        advisories.append(
            {
                "level": "critical",
                "message": f"{failed}/{total} 条样本评测执行失败，回答缺失；失败样本不参与任何指标均值。",
            }
        )
    if isinstance(invalid, int) and invalid > 0:
        advisories.append(
            {
                "level": "critical",
                "message": f"{invalid} 条样本溯源无效，其回答不可信，需要先排查检索来源。",
            }
        )
    for entry in ragas_entries:
        value = entry.get("value")
        scored = entry.get("scored")
        if isinstance(value, (int, float)) and value < _LOW_SCORE_ADVISORY_THRESHOLD:
            advisories.append(
                {
                    "level": "warning",
                    "message": f"核心指标 {entry['key']} 均值 {value:.4f} 低于 {_LOW_SCORE_ADVISORY_THRESHOLD}，建议结合抽检面板确认是回答质量问题还是评分器波动。",
                }
            )
        if entry.get("failed") and entry["failed"] > 0:
            advisories.append(
                {
                    "level": "warning",
                    "message": f"{entry['key']} 有 {entry['failed']} 条样本评分失败，均值只代表已评分子集。",
                }
            )
        if scored is not None and scored == 0:
            advisories.append(
                {
                    "level": "info",
                    "message": f"{entry['key']} 在本次运行中没有可评样本（按类别契约不适用），未纳入均值。",
                }
            )
    for entry in evidence_entries:
        if entry.get("value") is None:
            advisories.append(
                {
                    "level": "info",
                    "message": f"{entry['key']} 未覆盖（对应类别没有出现在本次运行，或参考评审未提供该指标）。",
                }
            )
    for entry in safety_entries:
        value = entry.get("value")
        if isinstance(value, (int, float)) and value > 0:
            advisories.append(
                {
                    "level": "critical",
                    "message": f"安全指标 {entry['key']} 为 {value:.2f}%（应恒为 0%），存在越界行为，必须人工排查。",
                }
            )
    for metric_id, case_ids in (failure_sample_ids or {}).items():
        if isinstance(case_ids, list) and case_ids:
            preview = "、".join(str(item)[-40:] for item in case_ids[:3])
            advisories.append(
                {
                    "level": "warning",
                    "message": f"{metric_id} 有 {len(case_ids)} 条未通过样本（{preview}{'…' if len(case_ids) > 3 else ''}），建议人工抽检复核。",
                }
            )
    families = ragas_summary.get("families") if isinstance(ragas_summary, dict) else None
    ragas_family = families.get("ragas") if isinstance(families, dict) else None
    if isinstance(ragas_family, dict) and ragas_family.get("failed"):
        advisories.append(
            {
                "level": "warning",
                "message": f"RAGAS 评分有 {ragas_family['failed']} 条 outcome 失败；均值口径只含成功评分。",
            }
        )
    return advisories


def build_run_report(results_root: str | Path, run_id: str) -> dict[str, Any]:
    """加载单次运行并合成一份综合评测报告（只读，不产生新测量）。

    Args:
        results_root: 评测结果根目录。
        run_id: 运行 ID。
    """
    if not RUN_ID_PATTERN.fullmatch(run_id):
        raise EvaluationResultError("invalid run_id")
    run_dir = _resolve_run_dir(Path(results_root), run_id)
    metadata = _read_json(run_dir / "run_metadata.json")
    report = _read_json(run_dir / "quality-report.json") or _read_json(run_dir / "quality_report.json")
    ragas_summary: dict[str, Any] = {}
    for path in sorted(run_dir.glob("ragas_*.summary.json")):
        value = _read_json(path)
        if value is not None:
            ragas_summary = value
            break

    counts = (report or {}).get("counts") or {}
    report_metrics = (report or {}).get("metrics") or {}
    evidence_block = (report or {}).get("evidence_gate") or {}
    evidence_metrics = evidence_block.get("metrics") or {}
    failure_sample_ids = evidence_block.get("failure_sample_ids") or {}
    category_metrics = (report or {}).get("category_metrics") or {}

    records, _ = _read_jsonl(run_dir / "responses.jsonl")
    ragas_entries = _ragas_entries(ragas_summary.get("metrics") or {})
    evidence_entries = _evidence_entries(evidence_metrics, records)
    retrieval_entries = _retrieval_entries(report_metrics)
    safety_entries = _safety_entries(category_metrics, int(counts.get("total") or 0))

    return {
        "run_id": run_id,
        "incomplete": metadata is None,
        "run_classification": (report or {}).get("run_classification"),
        "counts": {
            "total": counts.get("total"),
            "succeeded": counts.get("succeeded"),
            "failed": counts.get("failed"),
            "invalid_provenance": counts.get("invalid_provenance"),
        },
        "sections": {
            "ragas": ragas_entries,
            "evidence": evidence_entries,
            "retrieval": retrieval_entries,
            "safety": safety_entries,
        },
        "advisories": _advisories(
            counts=counts,
            ragas_entries=ragas_entries,
            evidence_entries=evidence_entries,
            safety_entries=safety_entries,
            failure_sample_ids=failure_sample_ids,
            ragas_summary=ragas_summary,
        ),
    }


__all__ = ["build_run_report"]
