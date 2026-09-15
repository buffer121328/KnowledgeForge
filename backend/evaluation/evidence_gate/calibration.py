"""从已测量的质量报告构建受约束的校准草稿产物。"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from evaluation.evidence_gate.metrics import SUPPORTED_EVIDENCE_GATE_METRICS

CANDIDATE_THRESHOLDS = {  # 候选安全阈值（仅为草稿，不构成生效发布证据）
    "refusal_precision": 0.98,  # 拒答精确率阈值
    "refusal_recall": 0.98,  # 拒答召回率阈值
    "no_answer_hallucination_rate": 0.01,  # 无据作答（幻觉）率上限
    "answerable_false_refusal_rate": 0.02,  # 可答问题被错误拒答的比率上限
    "partial_answer_recognition_rate": 0.95,  # 部分可答识别率阈值
    "conflict_recognition_rate": 0.98,  # 证据冲突识别率阈值
    "claim_citation_coverage": 1.0,  # 论断引用覆盖率阈值
    "citation_correctness": 0.99,  # 引用正确率阈值
    "groundedness_pass_rate": 0.99,  # 有据性通过率阈值
}

_CITATION_DERIVED_METRICS = frozenset(  # 依赖引用系统派生的指标（baseline 阶段允许缺失）
    {"claim_citation_coverage", "citation_correctness"}
)
_REQUIRED_RAGAS_METRICS = frozenset({"faithfulness", "factual_correctness", "context_precision", "context_recall"})
_SUPPORTED_EVALUATION_CASE_COUNTS = frozenset({50, 100})
_RELEASE_STAGE_METRIC_REQUIREMENTS = {  # 各发布阶段必须测量的指标：baseline 可缺引用派生指标，shadow 必须全量
    "baseline": SUPPORTED_EVIDENCE_GATE_METRICS - _CITATION_DERIVED_METRICS,
    "shadow": SUPPORTED_EVIDENCE_GATE_METRICS,
}


def _load_report(path: Path) -> dict[str, Any]:
    """读取质量报告 JSON 文件并要求其顶层为 JSON 对象。

    Args:
        path: 质量报告文件路径。
    """
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("quality report must be a JSON object")
    return value


def _report_sources(
    quality_report_paths: Mapping[str, str | Path] | Sequence[str | Path],
) -> tuple[list[tuple[str, Path]], bool]:
    """把报告来源规范化为 (阶段, 路径) 列表，同时不削弱旧式序列调用方。

    Args:
        quality_report_paths: 阶段名到报告路径的映射；或旧式按顺序给出的报告路径序列。

    返回值为 ((阶段, 路径) 列表, 是否显式声明了阶段)。
    """
    # 显式映射：阶段名即键
    if isinstance(quality_report_paths, Mapping):
        return [
            (str(stage), Path(path))
            for stage, path in quality_report_paths.items()
        ], True
    # 旧式序列：自动编号为 report-1、report-2 …，不套用阶段指标差异
    return [
        (f"report-{index}", Path(path))
        for index, path in enumerate(quality_report_paths, start=1)
    ], False


def _required_metrics(*, stage: str, explicit_stages: bool) -> frozenset[str]:
    """返回对单个报告来源适用的受限必需指标集合。

    Args:
        stage: 发布阶段名称（如 baseline / shadow）。
        explicit_stages: 调用方是否显式声明了阶段；未显式时一律要求全量指标。
    """
    if explicit_stages:
        return _RELEASE_STAGE_METRIC_REQUIREMENTS.get(
            stage, SUPPORTED_EVIDENCE_GATE_METRICS
        )
    return SUPPORTED_EVIDENCE_GATE_METRICS


def _is_measured_metric(value: Any) -> bool:
    """判断取值是否为真实测得的可比数值：有限的非布尔数字。

    Args:
        value: 待检查的指标取值。
    """
    # bool 是 int 的子类，必须显式排除；NaN/inf 等非有限值也不算已测量
    return (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(float(value))
    )


def _is_sha256(value: Any) -> bool:
    """Return whether ``value`` is a bounded lowercase SHA-256 digest."""
    return isinstance(value, str) and len(value) == 64 and all(char in "0123456789abcdef" for char in value)


def _bounded_ragas_summary(value: Any, *, case_count: int) -> tuple[dict[str, dict[str, Any]], list[str]]:
    """Validate and retain only review-safe RAGAS coverage/means."""
    if not isinstance(value, Mapping):
        return {}, ["ragas_coverage_missing"]
    safe: dict[str, dict[str, Any]] = {}
    errors: list[str] = []
    for metric in sorted(_REQUIRED_RAGAS_METRICS):
        item = value.get(metric)
        if not isinstance(item, Mapping):
            errors.append("ragas_coverage_missing")
            continue
        counts = {
            name: item.get(name, 0 if name == "not_applicable" else None)
            for name in ("total", "scored", "not_applicable", "skipped", "failed")
        }
        if any(isinstance(raw, bool) or not isinstance(raw, int) or raw < 0 for raw in counts.values()):
            errors.append("ragas_coverage_invalid")
            continue
        mean = item.get("mean")
        if (
            counts["total"] != case_count
            or (
                counts["scored"]
                + counts["not_applicable"]
                + counts["skipped"]
                + counts["failed"]
                != case_count
            )
            or counts["skipped"] != 0
            or counts["failed"] != 0
            or counts["scored"] == 0
            or isinstance(mean, bool)
            or not isinstance(mean, (int, float))
            or not math.isfinite(float(mean))
        ):
            errors.append("ragas_coverage_invalid")
            continue
        safe[metric] = {**counts, "mean": float(mean)}
    if set(safe) != _REQUIRED_RAGAS_METRICS:
        errors.append("ragas_coverage_missing")
    return safe, sorted(set(errors))


def _validate_source_evidence(
    source_evidence: Mapping[str, Mapping[str, Any]] | None,
    *,
    explicit_stages: bool,
) -> tuple[dict[str, dict[str, Any]], list[dict[str, str]]]:
    """Validate report bindings and return bounded identities for calibration records."""
    if source_evidence is None:
        return {}, []  # Compatibility for the standalone calibration CLI.
    errors: list[dict[str, str]] = []
    safe: dict[str, dict[str, Any]] = {}
    required = ("baseline", "shadow") if explicit_stages else tuple(source_evidence)
    manifests: set[str] = set()
    identities: set[tuple[Any, Any, Any]] = set()
    for stage in required:
        item = source_evidence.get(stage)
        if not isinstance(item, Mapping):
            errors.append({"stage": stage, "code": "calibration_source_missing"})
            continue
        report_hash = item.get("quality_report_sha256")
        responses_hash = item.get("responses_sha256")
        manifest = item.get("manifest_sha256")
        expected_manifest = item.get("expected_manifest_sha256")
        case_count = item.get("case_count")
        if not _is_sha256(report_hash) or not _is_sha256(responses_hash):
            errors.append({"stage": stage, "code": "calibration_source_unbound"})
            continue
        if not _is_sha256(manifest) or manifest != expected_manifest:
            errors.append({"stage": stage, "code": "calibration_source_identity_invalid"})
            continue
        if (
            isinstance(case_count, bool)
            or not isinstance(case_count, int)
            or case_count not in _SUPPORTED_EVALUATION_CASE_COUNTS
        ):
            errors.append({"stage": stage, "code": "calibration_source_smoke_only"})
            continue
        if (
            item.get("run_classification") != "baseline"
            or item.get("report_failed_records") != 0
            or item.get("report_invalid_provenance") != 0
        ):
            errors.append({"stage": stage, "code": "calibration_source_report_incomplete"})
            continue
        ragas, ragas_errors = _bounded_ragas_summary(item.get("ragas_coverage"), case_count=case_count)
        if ragas_errors:
            errors.extend({"stage": stage, "code": code} for code in ragas_errors)
            continue
        dataset_id = item.get("dataset_id")
        version = item.get("version")
        gate_mode = item.get("gate_mode")
        run_id = item.get("run_id")
        if not all(isinstance(raw, str) and raw for raw in (dataset_id, version, gate_mode, run_id)):
            errors.append({"stage": stage, "code": "calibration_source_identity_invalid"})
            continue
        manifests.add(manifest)
        identities.add((dataset_id, version, case_count))
        safe[stage] = {
            "quality_report_sha256": report_hash,
            "responses_sha256": responses_hash,
            "manifest_sha256": manifest,
            "dataset_id": dataset_id,
            "version": version,
            "case_count": case_count,
            "gate_mode": gate_mode,
            "run_id": run_id,
            "ragas": ragas,
        }
    if len(manifests) > 1 or len(identities) > 1:
        errors.append({"stage": "calibration", "code": "calibration_source_incomparable"})
    return safe, sorted(errors, key=lambda item: (item["stage"], item["code"]))


def _metric_deltas(measurements: Sequence[dict[str, Any]]) -> dict[str, dict[str, float | None]]:
    """Record signed shadow-minus-baseline differences without inventing zeroes."""
    by_stage = {item.get("stage"): item.get("metrics") for item in measurements}
    baseline = by_stage.get("baseline") if isinstance(by_stage.get("baseline"), Mapping) else {}
    shadow = by_stage.get("shadow") if isinstance(by_stage.get("shadow"), Mapping) else {}
    names = sorted({*baseline.keys(), *shadow.keys()})
    result: dict[str, dict[str, float | None]] = {}
    for name in names:
        left, right = baseline.get(name), shadow.get(name)
        result[name] = {
            "baseline": float(left) if _is_measured_metric(left) else None,
            "shadow": float(right) if _is_measured_metric(right) else None,
            "delta": float(right) - float(left) if _is_measured_metric(left) and _is_measured_metric(right) else None,
        }
    return result


def build_calibration_draft(
    quality_report_paths: Mapping[str, str | Path] | Sequence[str | Path],
    *,
    output_dir: str | Path,
    calibration_version: str,
    source_evidence: Mapping[str, Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """仅当所有适用指标都已被真实测量时才写出候选阈值，否则输出阻断状态的校准记录。

    Args:
        quality_report_paths: 各阶段质量报告路径映射，或按顺序给出的报告路径序列。
        output_dir: 校准草稿产物的输出目录。
        calibration_version: 本次校准的版本标识。
    """
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    thresholds_path = output / "evidence-thresholds.candidate.json"  # 候选阈值文件
    record_path = output / "calibration-record.json"  # 校准记录文件
    # ① 解析报告来源与各阶段必需指标
    sources, explicit_stages = _report_sources(quality_report_paths)
    measurements: list[dict[str, Any]] = []
    missing: set[str] = set()
    missing_measurements: list[dict[str, str]] = []
    stage_metric_requirements: dict[str, list[str]] = {}

    # ② 先校验调用方传入的哈希绑定、冻结身份与 RAGAS 覆盖证据。
    source_identities, integrity_errors = _validate_source_evidence(
        source_evidence, explicit_stages=explicit_stages,
    )

    # ③ 逐报告校验每个必需指标都已测量（缺失即记录，不得伪造测量值）
    for stage, path in sources:
        report = _load_report(path)
        required_metrics = _required_metrics(
            stage=stage,
            explicit_stages=explicit_stages,
        )
        stage_metric_requirements[stage] = sorted(required_metrics)
        evidence = report.get("evidence_gate")
        metrics = evidence.get("metrics") if isinstance(evidence, dict) else None
        identity = evidence.get("run_identity") if isinstance(evidence, dict) else None
        counts = evidence.get("counts") if isinstance(evidence, dict) else None
        metric_values = metrics if isinstance(metrics, Mapping) else {}
        for name in sorted(required_metrics):
            if not _is_measured_metric(metric_values.get(name)):
                missing.add(name)
                missing_measurements.append({"stage": stage, "metric": name})
        measurements.append(
            {
                "stage": stage,
                "run_id": identity.get("run_id") if isinstance(identity, Mapping) else None,
                "dataset_version": (
                    identity.get("dataset_version") if isinstance(identity, Mapping) else None
                ),
                "completed": counts.get("completed") if isinstance(counts, Mapping) else None,
                "metrics": {
                    name: metric_values.get(name)
                    for name in sorted(metric_values)
                    if isinstance(name, str)
                },
            }
        )

    # ④ 安全关卡：完整度、来源身份和 RAGAS 覆盖任一不满足均不得进入审批。
    measured = bool(sources) and not missing and not integrity_errors
    status = (
        "completed" if measured
        else "blocked_invalid_evidence" if integrity_errors
        else "blocked_missing_measurements"
    )
    if measured:
        thresholds_path.write_text(
            json.dumps(CANDIDATE_THRESHOLDS, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    else:
        thresholds_path.unlink(missing_ok=True)
    # ⑤ 写出校准记录：候选阈值不是生效发布证据，必须先在冻结 holdout 上验证
    record = {
        "schema_version": "evidence-gate-calibration-v1",
        "calibration_version": calibration_version,
        "status": status,
        "measured": measured,
        "quality_report_count": len(sources),
        "stage_metric_requirements": stage_metric_requirements,
        "missing_metrics": sorted(missing),
        "missing_measurements": missing_measurements,
        "measurements": measurements,
        "metric_deltas": _metric_deltas(measurements),
        "source_identities": source_identities,
        "integrity_errors": integrity_errors,
        "candidate_thresholds": CANDIDATE_THRESHOLDS if measured else None,
        "notes": (
            "Candidate safety thresholds are non-effective release evidence. "
            "Validate them on a frozen holdout run before any separate operations change."
        ),
    }
    record_path.write_text(
        json.dumps(record, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    # ⑤ 返回摘要（含阈值与记录文件路径）
    return {
        "status": status,
        "measured": measured,
        "calibration_version": calibration_version,
        "missing_metrics": sorted(missing),
        "integrity_errors": integrity_errors,
        "thresholds_path": thresholds_path,
        "record_path": record_path,
    }


__all__ = ["CANDIDATE_THRESHOLDS", "build_calibration_draft"]
