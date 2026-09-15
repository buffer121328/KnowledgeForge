"""Exact terminal coverage tests for resolved category metric plans."""

from __future__ import annotations

import pytest

from evaluation.benchmarks.category_metric_policy import category_metric_policy_identity
from evaluation.diagnostic.outcome_coverage import (
    DiagnosticCoverageError,
    expected_policy_outcomes,
    validate_diagnostic_outcome_coverage,
)


CASES = (
    {"benchmark_id": "answer-01", "category": "fully_answerable"},
    {"benchmark_id": "auth-01", "category": "authorization_filtered"},
)
SNAPSHOT = "b" * 64


def _complete_outcomes() -> list[dict[str, object]]:
    identity = category_metric_policy_identity()
    outcomes: list[dict[str, object]] = []
    for expected in expected_policy_outcomes(CASES):
        outcomes.append(
            {
                "benchmark_id": expected.benchmark_id,
                "category": expected.category,
                "metric": expected.metric_id,
                "metric_family": expected.outcome_family,
                "status": "scored" if expected.applicable else "not_applicable",
                "reason_code": None if expected.applicable else expected.reason_code,
                **identity,
                "variant_id": "observed",
                "response_snapshot_sha256": SNAPSHOT,
            }
        )
    return outcomes


def test_complete_plan_counts_all_terminal_statuses_and_bounded_reasons() -> None:
    outcomes = _complete_outcomes()
    applicable = [item for item in outcomes if item["status"] == "scored"]
    applicable[0].update(status="failed", reason_code="judge_timeout")
    applicable[1].update(
        status="unsupported", reason_code="metric_implementation_unavailable"
    )

    summary = validate_diagnostic_outcome_coverage(
        CASES,
        outcomes,
        variant_id="observed",
        response_snapshot_sha256=SNAPSHOT,
    )

    assert summary["expected"] == len(outcomes)
    assert sum(summary["statuses"].values()) == len(outcomes)
    assert summary["statuses"]["failed"] == 1
    assert summary["statuses"]["unsupported"] == 1
    assert summary["statuses"]["not_applicable"] > 0
    assert summary["reasons"]["judge_timeout"] == 1
    assert set(summary["by_category"]) == {"fully_answerable", "authorization_filtered"}


@pytest.mark.parametrize("mutation", ["missing", "duplicate", "legacy_skip", "policy_drift"])
def test_incomplete_duplicate_legacy_or_drifted_coverage_fails_closed(
    mutation: str,
) -> None:
    outcomes = _complete_outcomes()
    if mutation == "missing":
        outcomes.pop()
    elif mutation == "duplicate":
        outcomes.append(dict(outcomes[0]))
    elif mutation == "legacy_skip":
        outcomes[0]["status"] = "skipped"
    else:
        outcomes[0]["category_policy_sha256"] = "c" * 64

    with pytest.raises(DiagnosticCoverageError):
        validate_diagnostic_outcome_coverage(
            CASES,
            outcomes,
            variant_id="observed",
            response_snapshot_sha256=SNAPSHOT,
        )

