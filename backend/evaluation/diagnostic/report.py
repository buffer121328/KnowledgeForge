"""Deterministic four-layer category diagnostic summaries."""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from typing import Any

from evaluation.diagnostic.preregistration import diagnostic_run_plan_identity

DIAGNOSTIC_REPORT_VERSION = "category-diagnostic-report-v1"
_MAX_CASE_IDS = 20
_TERMINAL_STATUSES = frozenset(
    {"scored", "not_applicable", "failed", "unsupported"}
)


class DiagnosticReportError(ValueError):
    """Diagnostic inputs are incomplete or unsafe to summarize."""


def _bounded_id(value: Any) -> str | None:
    return value if isinstance(value, str) and 0 < len(value) <= 128 else None


def _routing_summary(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for record in records:
        category = _bounded_id(record.get("category"))
        if category is not None:
            grouped[category].append(record)
    by_category: dict[str, Any] = {}
    for category, rows in sorted(grouped.items()):
        confusion: Counter[str] = Counter()
        failures: list[str] = []
        for row in rows:
            expected = _bounded_id(row.get("expected_response_status"))
            observed = _bounded_id(row.get("response_status"))
            if expected is None or observed is None:
                continue
            confusion[f"{expected}->{observed}"] += 1
            case_id = _bounded_id(row.get("benchmark_id"))
            if expected != observed and case_id and len(failures) < _MAX_CASE_IDS:
                failures.append(case_id)
        by_category[category] = {
            "count": len(rows),
            "confusion": dict(sorted(confusion.items())),
            "failing_case_ids": failures,
        }
    return {"by_category": by_category}


def _metric_summary(
    outcomes: Sequence[Mapping[str, Any]], *, families: set[str]
) -> dict[str, Any]:
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for outcome in outcomes:
        metric = _bounded_id(outcome.get("metric"))
        family = outcome.get("metric_family")
        status = outcome.get("status")
        if metric and family in families and status in _TERMINAL_STATUSES:
            grouped[metric].append(outcome)
    metrics: dict[str, Any] = {}
    for metric, rows in sorted(grouped.items()):
        statuses = Counter(str(row["status"]) for row in rows)
        category_scores: dict[str, list[float]] = defaultdict(list)
        all_scores: list[float] = []
        failing_ids: list[str] = []
        direction = next(
            (
                row.get("direction")
                for row in rows
                if row.get("direction") in {"minimum", "maximum"}
            ),
            "minimum",
        )
        for row in rows:
            score = row.get("score")
            category = _bounded_id(row.get("category"))
            if (
                row.get("status") == "scored"
                and isinstance(score, (int, float))
                and not isinstance(score, bool)
                and math.isfinite(float(score))
                and category
            ):
                value = float(score)
                all_scores.append(value)
                category_scores[category].append(value)
                failed = value < 1.0 if direction == "minimum" else value > 0.0
                case_id = _bounded_id(row.get("benchmark_id"))
                if failed and case_id and len(failing_ids) < _MAX_CASE_IDS:
                    failing_ids.append(case_id)
        category_means = {
            category: sum(values) / len(values)
            for category, values in sorted(category_scores.items())
        }
        metrics[metric] = {
            "direction": direction,
            "coverage": {
                status: statuses.get(status, 0) for status in sorted(_TERMINAL_STATUSES)
            },
            "micro_mean": sum(all_scores) / len(all_scores) if all_scores else None,
            "category_means": category_means,
            "category_macro_mean": (
                sum(category_means.values()) / len(category_means)
                if category_means
                else None
            ),
            "failing_case_ids": failing_ids,
        }
    return {"metrics": metrics}


def _safe_retrieval_summary(value: Mapping[str, Any]) -> dict[str, Any]:
    schema = _bounded_id(value.get("schema_version"))
    if schema is None:
        raise DiagnosticReportError("retrieval_summary_schema_invalid")
    safe: dict[str, Any] = {"schema_version": schema}
    for field in ("status", "reason_code"):
        field_value = value.get(field)
        if isinstance(field_value, str) and 0 < len(field_value) <= 96:
            safe[field] = field_value
    k_values = value.get("k_values")
    if isinstance(k_values, list) and all(
        isinstance(item, int) and not isinstance(item, bool) and 0 < item <= 100
        for item in k_values
    ):
        safe["k_values"] = list(k_values[:16])
    def safe_namespace(item: Any) -> dict[str, Any]:
        if not isinstance(item, Mapping):
            return {}
        bounded: dict[str, Any] = {}
        statuses = item.get("status_counts")
        if isinstance(statuses, Mapping):
            bounded["status_counts"] = {
                str(key): int(count)
                for key, count in statuses.items()
                if _bounded_id(key)
                and isinstance(count, int)
                and not isinstance(count, bool)
                and 0 <= count <= 100_000
            }
        scored = item.get("scored")
        if isinstance(scored, int) and not isinstance(scored, bool) and 0 <= scored <= 100_000:
            bounded["scored"] = scored
        for field in ("mean_recall_at_k", "mean_ndcg_at_k"):
            rates = item.get(field)
            if isinstance(rates, Mapping):
                bounded[field] = {
                    str(key): (float(rate) if isinstance(rate, (int, float)) else None)
                    for key, rate in rates.items()
                    if _bounded_id(str(key))
                    and (
                        rate is None
                        or (
                            isinstance(rate, (int, float))
                            and not isinstance(rate, bool)
                            and math.isfinite(float(rate))
                        )
                    )
                }
        mean_mrr = item.get("mean_mrr")
        if mean_mrr is None or (
            isinstance(mean_mrr, (int, float)) and not isinstance(mean_mrr, bool)
            and math.isfinite(float(mean_mrr))
        ):
            bounded["mean_mrr"] = None if mean_mrr is None else float(mean_mrr)
        return bounded

    def safe_stages(item: Any) -> dict[str, Any]:
        if not isinstance(item, Mapping):
            return {}
        return {
            stage: {
                namespace: safe_namespace(namespace_value)
                for namespace, namespace_value in stage_value.items()
                if namespace in {"source_document", "exact_evidence"}
            }
            for stage, stage_value in item.items()
            if stage in {"dense", "bm25", "graph", "fused", "reranked"}
            and isinstance(stage_value, Mapping)
        }

    safe["stages"] = safe_stages(value.get("stages"))
    categories = value.get("categories")
    if isinstance(categories, Mapping):
        safe["categories"] = {
            category: safe_stages(stage_values)
            for category, stage_values in categories.items()
            if _bounded_id(category) and isinstance(stage_values, Mapping)
        }
    return safe


def _safety_summary(
    records: Sequence[Mapping[str, Any]],
    outcomes: Sequence[Mapping[str, Any]],
    isolation_evidence: Mapping[str, Any],
) -> dict[str, Any]:
    hard_gates: dict[str, Any] = {}
    for outcome in outcomes:
        if outcome.get("hard_gate") is not True:
            continue
        metric = _bounded_id(outcome.get("metric"))
        case_id = _bounded_id(outcome.get("benchmark_id"))
        score = outcome.get("score")
        direction = outcome.get("direction")
        passed = bool(
            outcome.get("status") == "scored"
            and isinstance(score, (int, float))
            and not isinstance(score, bool)
            and math.isfinite(float(score))
            and (float(score) >= 1.0 if direction == "minimum" else float(score) <= 0.0)
        )
        if metric is None:
            continue
        current = hard_gates.setdefault(metric, {"passed": True, "failing_case_ids": []})
        current["passed"] = current["passed"] and passed
        if not passed and case_id and len(current["failing_case_ids"]) < _MAX_CASE_IDS:
            current["failing_case_ids"].append(case_id)
    invalid_ids = [
        case_id
        for record in records
        if record.get("status") == "invalid_provenance"
        and (case_id := _bounded_id(record.get("benchmark_id"))) is not None
    ][:_MAX_CASE_IDS]
    hard_gates["invalid_provenance"] = {
        "passed": not invalid_ids,
        "failing_case_ids": invalid_ids,
    }
    for field in ("cross_snapshot_contamination", "unauthorized_oracle_access"):
        value = isolation_evidence.get(field)
        if not isinstance(value, bool):
            raise DiagnosticReportError("isolation_evidence_incomplete")
        hard_gates[field] = {"passed": not value, "failing_case_ids": []}
    return {
        **_metric_summary(outcomes, families={"safety"}),
        "hard_gates": dict(sorted(hard_gates.items())),
        "passed": all(item["passed"] for item in hard_gates.values()),
    }


def build_diagnostic_report(
    *,
    records: Sequence[Mapping[str, Any]],
    metric_outcomes: Sequence[Mapping[str, Any]],
    retrieval_summary: Mapping[str, Any],
    isolation_evidence: Mapping[str, Any],
) -> dict[str, Any]:
    """Build the four non-compensating layers and a content-addressed hash."""

    retrieval = _safe_retrieval_summary(retrieval_summary)
    safety = _safety_summary(records, metric_outcomes, isolation_evidence)
    report: dict[str, Any] = {
        "schema_version": DIAGNOSTIC_REPORT_VERSION,
        "run_plan": diagnostic_run_plan_identity(),
        "retrieval": retrieval,
        "routing": _routing_summary(records),
        "answer": _metric_summary(
            metric_outcomes, families={"ragas", "deterministic"}
        ),
        "safety": safety,
        "decision": "passed" if safety["passed"] else "blocked",
    }
    canonical = json.dumps(
        report, ensure_ascii=True, separators=(",", ":"), sort_keys=True
    ).encode("utf-8")
    report["report_sha256"] = hashlib.sha256(canonical).hexdigest()
    return report


__all__ = [
    "DIAGNOSTIC_REPORT_VERSION",
    "DiagnosticReportError",
    "build_diagnostic_report",
]
