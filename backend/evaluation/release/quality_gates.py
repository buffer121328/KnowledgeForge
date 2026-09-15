"""Deterministic, offline quality reports and fail-closed release gates."""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

from ..evidence_gate.metrics import (
    REQUIRED_RUN_IDENTITY_FIELDS,
    aggregate_category_custom_metrics,
    aggregate_evidence_gate_metrics,
    evaluate_evidence_gate_promotion,
)
from ..benchmarks.category_metric_policy import get_category_metric_contract
from ..diagnostic.report import build_diagnostic_report

QUALITY_REPORT_VERSION = 1
DEFAULT_SMOKE_THRESHOLD = 30
_COMPLETED_STATUSES = frozenset({"succeeded", "invalid_provenance"})
_NATIVE_STRATEGIES = frozenset(
    {"dense", "dense_graph", "dense_bm25", "dense_bm25_graph"}
)
_BM25_STRATEGIES = frozenset({"dense_bm25", "dense_bm25_graph"})
_FROZEN_METADATA_FIELDS = (
    "evidence_class",
    "corpus_revision",
    "knowledge_revision",
    "embedding_model",
    "answer_model",
    "prompt_id",
    "retrieval_config_id",
    "top_k",
    "cache_policy",
    "authorization_scope_fixture",
)
_BM25_METADATA_FIELDS = (
    "schema_version",
    "tokenizer_version",
    "k1",
    "b",
    "index_generation",
)


def _finite_number(value: Any) -> float | None:
    """Return a finite float, excluding booleans and invalid numeric values."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _unique_strings(value: Any) -> list[str] | None:
    """Return unique non-empty strings while preserving input order."""
    if not isinstance(value, list) or any(
        not isinstance(item, str) or not item for item in value
    ):
        return None
    return list(dict.fromkeys(value))


def _mean(values: Iterable[float]) -> float | None:
    """Return a stable arithmetic mean when values exist."""
    values = list(values)
    if not values:
        return None
    return round(sum(values) / len(values), 6)


def _percentile(values: Iterable[float], percentile: float) -> float | None:
    """Return a linearly interpolated percentile."""
    ordered = sorted(values)
    if not ordered:
        return None
    position = (len(ordered) - 1) * percentile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return round(ordered[lower], 3)
    return round(
        ordered[lower]
        + (ordered[upper] - ordered[lower]) * (position - lower),
        3,
    )


def _refusal_metrics(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate refusal classification for completed benchmark records."""
    tp = fp = fn = tn = 0
    for record in records:
        if record.get("status") not in _COMPLETED_STATUSES:
            continue
        expected = record.get("expected_refusal")
        refused = record.get("refused")
        if not isinstance(expected, bool) or not isinstance(refused, bool):
            continue
        if expected and refused:
            tp += 1
        elif not expected and refused:
            fp += 1
        elif expected and not refused:
            fn += 1
        else:
            tn += 1
    precision = tp / (tp + fp) if tp + fp else None
    recall = tp / (tp + fn) if tp + fn else None
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision is not None
        and recall is not None
        and precision + recall
        else None
    )
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": None if precision is None else round(precision, 6),
        "recall": None if recall is None else round(recall, 6),
        "f1": None if f1 is None else round(f1, 6),
    }


def _summarize_records(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Summarize provenance, ID-based retrieval quality, refusal, and latency."""
    counts = {
        "total": len(records),
        "succeeded": sum(record.get("status") == "succeeded" for record in records),
        "failed": sum(record.get("status") == "failed" for record in records),
        "invalid_provenance": sum(
            record.get("status") == "invalid_provenance" for record in records
        ),
    }
    completed = [
        record for record in records if record.get("status") in _COMPLETED_STATUSES
    ]
    scored: list[tuple[set[str], set[str]]] = []
    for record in records:
        if record.get("status") != "succeeded":
            continue
        retrieved = _unique_strings(record.get("retrieved_source_document_ids"))
        reference = _unique_strings(record.get("reference_source_document_ids"))
        if retrieved is None and reference is None:
            namespaces = record.get("id_namespaces")
            legacy_compatible = not isinstance(namespaces, dict) or (
                namespaces.get("retrieved_context_ids")
                == namespaces.get("reference_context_ids")
            )
            if legacy_compatible:
                retrieved = _unique_strings(record.get("retrieved_context_ids"))
                reference = _unique_strings(record.get("reference_context_ids"))
        if retrieved is None or reference is None:
            continue
        scored.append((set(retrieved), set(reference)))
    precision_values = [
        len(retrieved & reference) / len(retrieved)
        for retrieved, reference in scored
        if retrieved
    ]
    recall_values = [
        len(retrieved & reference) / len(reference)
        for retrieved, reference in scored
        if reference
    ]
    latencies = [
        latency
        for record in completed
        if (latency := _finite_number(record.get("latency_ms"))) is not None
        and latency >= 0
    ]
    counts["completed"] = len(completed)
    counts["id_scored"] = len(scored)
    provenance_denominator = counts["succeeded"] + counts["invalid_provenance"]
    metrics = {
        "provenance_completeness": (
            round(counts["succeeded"] / provenance_denominator, 6)
            if provenance_denominator
            else None
        ),
        "id_based_context_precision": _mean(precision_values),
        "id_based_context_recall": _mean(recall_values),
        "refusal": _refusal_metrics(records),
    }
    return {
        "counts": counts,
        "metrics": metrics,
        "latency_ms": {
            "p50": _percentile(latencies, 0.50),
            "p95": _percentile(latencies, 0.95),
        },
    }


def _bounded_mapping(value: Any, allowed_fields: Sequence[str]) -> dict[str, Any]:
    """Copy only explicitly allowed scalar or bounded nested metadata fields."""
    if not isinstance(value, dict):
        return {}
    return {field: value.get(field) for field in allowed_fields if field in value}


def _safe_release_evidence(value: Any) -> dict[str, Any]:
    """Retain only bounded boolean/numeric release evidence, never raw corpus text."""
    if not isinstance(value, dict):
        return {}
    return {
        "tenant_isolation": value.get("tenant_isolation"),
        "department_isolation": value.get("department_isolation"),
        "sparse_lifecycle": _bounded_mapping(
            value.get("sparse_lifecycle"), ("add", "update", "delete", "rebuild")
        ),
        "degradation": _bounded_mapping(
            value.get("degradation"),
            ("unavailable_snapshot", "corrupt_snapshot"),
        ),
        "rollback": _bounded_mapping(
            value.get("rollback"), ("configuration_only", "target_strategy")
        ),
        "cost": _bounded_mapping(value.get("cost"), ("observed", "budget", "unit")),
    }


def _safe_run_identity(value: Any) -> dict[str, Any]:
    """Keep only bounded replay identifiers and retrieval mode names."""
    if not isinstance(value, dict):
        return {}
    safe: dict[str, Any] = {}
    for field in REQUIRED_RUN_IDENTITY_FIELDS:
        item = value.get(field)
        if field == "retrieval_modes":
            if isinstance(item, list):
                safe[field] = [
                    mode
                    for mode in item[:16]
                    if isinstance(mode, str) and 0 < len(mode) <= 128
                ]
            continue
        if isinstance(item, str) and 0 < len(item) <= 200:
            safe[field] = item
    return safe


def _safe_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    """Retain reproducibility and gate metadata without raw prompts or credentials."""
    source = metadata.get("benchmark_source")
    safe = {
        "run_id": metadata.get("run_id"),
        "benchmark_sha256": metadata.get("benchmark_sha256"),
        "benchmark_source": (
            Path(source).name if isinstance(source, str) and source else None
        ),
        "code_revision": metadata.get("code_revision"),
        "started_at": metadata.get("started_at"),
        "retrieval_modes": list(metadata.get("retrieval_modes") or []),
    }
    for field in _FROZEN_METADATA_FIELDS:
        if field in metadata:
            safe[field] = metadata.get(field)
    if "bm25" in metadata:
        safe["bm25"] = _bounded_mapping(metadata.get("bm25"), _BM25_METADATA_FIELDS)
    if "release_evidence" in metadata:
        safe["release_evidence"] = _safe_release_evidence(
            metadata.get("release_evidence")
        )
    if "run_identity" in metadata:
        safe["run_identity"] = _safe_run_identity(metadata.get("run_identity"))
    return safe


def aggregate_quality_report(
    metadata: dict[str, Any],
    records: Sequence[dict[str, Any]],
    *,
    smoke_threshold: int = DEFAULT_SMOKE_THRESHOLD,
    diagnostic_inputs: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a deterministic report from runner artifacts without retaining raw text."""
    if not isinstance(metadata, dict):
        raise ValueError("metadata must be a JSON object")
    if smoke_threshold <= 0:
        raise ValueError("smoke_threshold must be positive")
    safe = _safe_metadata(metadata)
    modes = [mode for mode in safe["retrieval_modes"] if isinstance(mode, str)]
    summary = _summarize_records(records)
    by_mode = {
        mode: _summarize_records(
            [record for record in records if record.get("retrieval_mode") == mode]
        )
        for mode in modes
    }
    categories = sorted(
        {
            record.get("category")
            for record in records
            if isinstance(record.get("category"), str)
        }
    )
    by_category = {
        category: _summarize_records(
            [record for record in records if record.get("category") == category]
        )
        for category in categories
    }
    smoke_only = any(
        by_mode.get(mode, {}).get("counts", {}).get("completed", 0)
        < smoke_threshold
        for mode in modes
    )
    report = {
        "schema_version": QUALITY_REPORT_VERSION,
        "metadata": safe,
        "smoke_threshold": smoke_threshold,
        "run_classification": "smoke_only" if smoke_only else "baseline",
        "retrieval_modes": modes,
        "counts": summary["counts"],
        "metrics": summary["metrics"],
        "latency_ms": summary["latency_ms"],
        "by_retrieval_mode": by_mode,
        "by_category": by_category,
    }
    # Keep the policy denominator separate from the legacy 9 x case matrix.
    # The latter is useful for an applicability audit, but is misleading as a
    # headline because most category/metric pairs are intentionally N/A.
    if categories:
        case_categories = {
            str(record.get("benchmark_id")): str(record.get("category"))
            for record in records
            if record.get("benchmark_id") and record.get("category")
        }
        try:
            contracts = [
                get_category_metric_contract(category)
                for category in case_categories.values()
            ]
        except Exception:
            contracts = []
        if contracts:
            report["evaluation_coverage"] = {
                "case_count": len(case_categories),
                "ragas_planned_outcomes": sum(
                    sum(item.applicable for item in contract.ragas)
                    for contract in contracts
                ),
                "custom_planned_outcomes": sum(
                    len(contract.custom) for contract in contracts
                ),
                "ragas_matrix_outcomes": len(case_categories) * 9,
            }
    if any(record.get("expected_response_status") is not None for record in records):
        report["evidence_gate"] = aggregate_evidence_gate_metrics(
            records,
            run_identity=safe.get("run_identity"),
        )
        report["category_metrics"] = aggregate_category_custom_metrics(records)
    if diagnostic_inputs is not None:
        report["diagnostic"] = build_diagnostic_report(
            records=records,
            metric_outcomes=diagnostic_inputs.get("metric_outcomes", ()),
            retrieval_summary=diagnostic_inputs.get("retrieval_summary", {}),
            isolation_evidence=diagnostic_inputs.get("isolation_evidence", {}),
        )
    return report


def _missing_metadata_error(field: str) -> dict[str, str]:
    """Build one structured missing-metadata validation error."""
    return {
        "field": field,
        "code": "missing_metadata",
        "message": f"{field} is required",
    }


def validate_quality_report(
    report: dict[str, Any],
    *,
    required_modes: Sequence[str] | None = None,
) -> list[dict[str, str]]:
    """Return structured validation errors; missing values never become zero."""
    errors: list[dict[str, str]] = []
    if not isinstance(report, dict):
        return [
            {
                "field": "report",
                "code": "invalid_type",
                "message": "report must be an object",
            }
        ]
    if report.get("schema_version") != QUALITY_REPORT_VERSION:
        errors.append(
            {
                "field": "schema_version",
                "code": "unsupported_schema",
                "message": "unsupported quality report schema",
            }
        )
    metadata = report.get("metadata")
    modes = list(required_modes or report.get("retrieval_modes") or [])
    if not isinstance(metadata, dict):
        errors.append(
            {
                "field": "metadata",
                "code": "missing_metadata",
                "message": "metadata is required",
            }
        )
    else:
        for field in ("run_id", "benchmark_sha256", "retrieval_modes", "started_at"):
            value = metadata.get(field)
            if value in (None, "") or (
                field == "retrieval_modes" and not isinstance(value, list)
            ):
                errors.append(_missing_metadata_error(f"metadata.{field}"))
        if any(mode in _NATIVE_STRATEGIES for mode in modes):
            for field in _FROZEN_METADATA_FIELDS:
                if metadata.get(field) in (None, ""):
                    errors.append(_missing_metadata_error(f"metadata.{field}"))
        if any(mode in _BM25_STRATEGIES for mode in modes):
            bm25 = metadata.get("bm25")
            if not isinstance(bm25, dict):
                errors.append(_missing_metadata_error("metadata.bm25"))
            else:
                for field in _BM25_METADATA_FIELDS:
                    if bm25.get(field) in (None, ""):
                        errors.append(_missing_metadata_error(f"metadata.bm25.{field}"))
    counts = report.get("counts")
    if not isinstance(counts, dict):
        errors.append(
            {
                "field": "counts",
                "code": "missing_counts",
                "message": "counts are required",
            }
        )
    else:
        for field in ("total", "succeeded", "failed", "invalid_provenance"):
            if not isinstance(counts.get(field), int) or counts[field] < 0:
                errors.append(
                    {
                        "field": f"counts.{field}",
                        "code": "invalid_counts",
                        "message": f"counts.{field} is required",
                    }
                )
    if not isinstance(report.get("metrics"), dict) or not isinstance(
        report.get("latency_ms"), dict
    ):
        errors.append(
            {
                "field": "metrics",
                "code": "missing_metrics",
                "message": "metrics and latency_ms are required",
            }
        )
    by_mode = report.get("by_retrieval_mode")
    if not isinstance(by_mode, dict):
        errors.append(
            {
                "field": "by_retrieval_mode",
                "code": "missing_strata",
                "message": "retrieval mode strata are required",
            }
        )
    else:
        for mode in modes:
            if mode not in by_mode:
                errors.append(
                    {
                        "field": f"by_retrieval_mode.{mode}",
                        "code": "missing_strata",
                        "message": f"missing retrieval mode: {mode}",
                    }
                )
    if report.get("run_classification") not in {"smoke_only", "baseline"}:
        errors.append(
            {
                "field": "run_classification",
                "code": "invalid_classification",
                "message": "invalid run classification",
            }
        )
    return errors


def _bm25_release_reasons(metadata: Any) -> list[dict[str, str]]:
    """Require complete isolation, lifecycle, degradation, rollback, and cost evidence."""
    evidence = metadata.get("release_evidence") if isinstance(metadata, dict) else None
    if not isinstance(evidence, dict):
        return [{"code": "missing_release_evidence", "field": "metadata.release_evidence"}]
    reasons: list[dict[str, str]] = []
    if metadata.get("evidence_class") != "production_service_benchmark":
        reasons.append(
            {"code": "non_production_evidence", "field": "metadata.evidence_class"}
        )
    if evidence.get("tenant_isolation") is not True:
        reasons.append(
            {"code": "tenant_isolation_failed", "field": "metadata.release_evidence.tenant_isolation"}
        )
    if evidence.get("department_isolation") is not True:
        reasons.append(
            {"code": "department_isolation_failed", "field": "metadata.release_evidence.department_isolation"}
        )
    lifecycle = evidence.get("sparse_lifecycle")
    if not isinstance(lifecycle, dict) or any(
        lifecycle.get(action) is not True for action in ("add", "update", "delete", "rebuild")
    ):
        reasons.append(
            {"code": "missing_sparse_lifecycle_evidence", "field": "metadata.release_evidence.sparse_lifecycle"}
        )
    degradation = evidence.get("degradation")
    if not isinstance(degradation, dict) or any(
        degradation.get(check) is not True
        for check in ("unavailable_snapshot", "corrupt_snapshot")
    ):
        reasons.append(
            {"code": "missing_degradation_evidence", "field": "metadata.release_evidence.degradation"}
        )
    rollback = evidence.get("rollback")
    rollback_target = rollback.get("target_strategy") if isinstance(rollback, dict) else None
    if (
        not isinstance(rollback, dict)
        or rollback.get("configuration_only") is not True
        or rollback_target not in {"dense", "dense_graph"}
    ):
        reasons.append(
            {"code": "missing_rollback_evidence", "field": "metadata.release_evidence.rollback"}
        )
    cost = evidence.get("cost")
    observed = _finite_number(cost.get("observed")) if isinstance(cost, dict) else None
    budget = _finite_number(cost.get("budget")) if isinstance(cost, dict) else None
    unit = cost.get("unit") if isinstance(cost, dict) else None
    if observed is None or budget is None or budget < 0 or not isinstance(unit, str) or not unit:
        reasons.append(
            {"code": "missing_cost_evidence", "field": "metadata.release_evidence.cost"}
        )
    elif observed < 0 or observed > budget:
        reasons.append(
            {"code": "cost_budget_exceeded", "field": "metadata.release_evidence.cost.observed"}
        )
    return reasons


def evaluate_release_gate(
    report: dict[str, Any],
    *,
    required_modes: Sequence[str] | None = None,
    thresholds: dict[str, float] | None = None,
    evidence_thresholds: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Evaluate a report conservatively; every missing release input blocks."""
    errors = validate_quality_report(report, required_modes=required_modes)
    reasons = [{"code": error["code"], "field": error["field"]} for error in errors]
    counts = report.get("counts") if isinstance(report, dict) else {}
    if isinstance(counts, dict) and counts.get("failed", 0) > 0:
        reasons.append({"code": "failed_records", "field": "counts.failed"})
    if isinstance(counts, dict) and counts.get("invalid_provenance", 0) > 0:
        reasons.append(
            {"code": "invalid_provenance", "field": "counts.invalid_provenance"}
        )
    if report.get("run_classification") == "smoke_only":
        reasons.append({"code": "smoke_only", "field": "run_classification"})
    modes = list(required_modes or report.get("retrieval_modes") or [])
    if any(mode in _BM25_STRATEGIES for mode in modes):
        reasons.extend(_bm25_release_reasons(report.get("metadata")))
    if thresholds:
        metrics = report.get("metrics") if isinstance(report, dict) else {}
        latency = report.get("latency_ms") if isinstance(report, dict) else {}
        for name, threshold in thresholds.items():
            if name == "p95_latency_ms":
                value = latency.get("p95") if isinstance(latency, dict) else None
                if not isinstance(value, (int, float)) or value > threshold:
                    reasons.append(
                        {"code": "latency_budget_exceeded", "field": "latency_ms.p95"}
                    )
                continue
            value = metrics.get(name) if isinstance(metrics, dict) else None
            if not isinstance(value, (int, float)) or value < threshold:
                reasons.append(
                    {"code": "threshold_failed", "field": f"metrics.{name}"}
                )
    if evidence_thresholds is not None:
        evidence_report = report.get("evidence_gate")
        if not isinstance(evidence_report, dict):
            reasons.append(
                {"code": "missing_evidence_gate_report", "field": "evidence_gate"}
            )
        else:
            evidence_decision = evaluate_evidence_gate_promotion(
                evidence_report, thresholds=evidence_thresholds
            )
            for reason in evidence_decision["reasons"]:
                reasons.append(
                    {
                        **reason,
                        "field": f"evidence_gate.{reason['field']}",
                    }
                )
    unique_reasons = list(
        {(item["code"], item["field"]): item for item in reasons}.values()
    )
    return {
        "decision": "blocked" if unique_reasons else "passed",
        "reasons": unique_reasons,
    }


__all__ = [
    "DEFAULT_SMOKE_THRESHOLD",
    "QUALITY_REPORT_VERSION",
    "aggregate_evidence_gate_metrics",
    "aggregate_quality_report",
    "evaluate_evidence_gate_promotion",
    "evaluate_release_gate",
    "validate_quality_report",
]
