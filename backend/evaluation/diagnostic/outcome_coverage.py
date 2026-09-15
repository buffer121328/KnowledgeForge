"""Fail-closed terminal coverage for category-aware diagnostic outcomes."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any, Iterable, Mapping

from evaluation.benchmarks.category_metric_policy import (
    category_metric_policy_identity,
    get_category_metric_contract,
)

_TERMINAL_STATUSES = frozenset(
    {"scored", "not_applicable", "failed", "unsupported"}
)


class DiagnosticCoverageError(ValueError):
    """Resolved case/metric coverage is incomplete or identity-incompatible."""


@dataclass(frozen=True, slots=True)
class ExpectedPolicyOutcome:
    benchmark_id: str
    category: str
    metric_id: str
    outcome_family: str
    applicable: bool
    reason_code: str


def expected_policy_outcomes(
    cases: Iterable[Mapping[str, Any]],
) -> tuple[ExpectedPolicyOutcome, ...]:
    """Resolve the exact independent case/metric pairs before scoring."""

    expected: list[ExpectedPolicyOutcome] = []
    seen_cases: set[str] = set()
    for case in cases:
        benchmark_id = case.get("benchmark_id")
        category = case.get("category")
        if not isinstance(benchmark_id, str) or not benchmark_id or benchmark_id in seen_cases:
            raise DiagnosticCoverageError("diagnostic_case_identity_invalid")
        if not isinstance(category, str):
            raise DiagnosticCoverageError("diagnostic_case_category_invalid")
        seen_cases.add(benchmark_id)
        contract = get_category_metric_contract(category)
        expected.extend(
            ExpectedPolicyOutcome(
                benchmark_id=benchmark_id,
                category=category,
                metric_id=item.metric_id,
                outcome_family=item.outcome_family,
                applicable=item.applicable,
                reason_code=item.reason_code,
            )
            for item in contract.metrics
        )
    if not expected:
        raise DiagnosticCoverageError("diagnostic_cases_empty")
    return tuple(expected)


def validate_diagnostic_outcome_coverage(
    cases: Iterable[Mapping[str, Any]],
    outcomes: Iterable[Mapping[str, Any]],
    *,
    variant_id: str,
    response_snapshot_sha256: str,
) -> dict[str, Any]:
    """Validate exact terminal coverage and return a bounded deterministic summary."""

    expected = expected_policy_outcomes(cases)
    expected_by_key = {
        (item.benchmark_id, item.metric_id): item for item in expected
    }
    identity = category_metric_policy_identity()
    records = list(outcomes)
    keys = [(item.get("benchmark_id"), item.get("metric")) for item in records]
    if len(keys) != len(set(keys)):
        raise DiagnosticCoverageError("diagnostic_outcome_duplicate")
    if set(keys) != set(expected_by_key):
        raise DiagnosticCoverageError("diagnostic_outcome_coverage_mismatch")

    statuses: Counter[str] = Counter()
    reasons: Counter[str] = Counter()
    by_category: dict[str, Counter[str]] = defaultdict(Counter)
    for outcome, key in zip(records, keys, strict=True):
        planned = expected_by_key[key]
        status = outcome.get("status")
        reason = outcome.get("reason_code")
        if status not in _TERMINAL_STATUSES:
            raise DiagnosticCoverageError("diagnostic_outcome_status_invalid")
        if outcome.get("category") != planned.category:
            raise DiagnosticCoverageError("diagnostic_outcome_category_mismatch")
        if outcome.get("metric_family") != planned.outcome_family:
            raise DiagnosticCoverageError("diagnostic_outcome_family_mismatch")
        if (
            outcome.get("category_policy_version")
            != identity["category_policy_version"]
            or outcome.get("category_policy_sha256")
            != identity["category_policy_sha256"]
            or outcome.get("variant_id") != variant_id
            or outcome.get("response_snapshot_sha256") != response_snapshot_sha256
        ):
            raise DiagnosticCoverageError("diagnostic_outcome_identity_drift")
        if planned.applicable:
            if status == "not_applicable":
                raise DiagnosticCoverageError("diagnostic_outcome_applicability_mismatch")
        elif status != "not_applicable" or reason != planned.reason_code:
            raise DiagnosticCoverageError("diagnostic_outcome_applicability_mismatch")
        if status in {"failed", "unsupported", "not_applicable"}:
            if not isinstance(reason, str) or not reason or len(reason) > 96:
                raise DiagnosticCoverageError("diagnostic_outcome_reason_invalid")
            reasons[reason] += 1
        statuses[status] += 1
        by_category[planned.category][status] += 1

    if len(reasons) > 64:
        raise DiagnosticCoverageError("diagnostic_outcome_reasons_unbounded")
    return {
        "expected": len(expected),
        "statuses": {key: statuses.get(key, 0) for key in sorted(_TERMINAL_STATUSES)},
        "reasons": dict(sorted(reasons.items())),
        "by_category": {
            category: {
                key: counts.get(key, 0) for key in sorted(_TERMINAL_STATUSES)
            }
            for category, counts in sorted(by_category.items())
        },
        **identity,
        "variant_id": variant_id,
        "response_snapshot_sha256": response_snapshot_sha256,
    }


__all__ = [
    "DiagnosticCoverageError",
    "ExpectedPolicyOutcome",
    "expected_policy_outcomes",
    "validate_diagnostic_outcome_coverage",
]
