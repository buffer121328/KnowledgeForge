"""Regression tests for category diagnostic outcome/checkpoint identities."""

from __future__ import annotations

import pytest

from evaluation.benchmarks.evaluation_outcomes import (
    EvaluationOutcomeError,
    attach_diagnostic_identity,
    outcome_checkpoint_key,
)


IDENTITY = {
    "category_policy_version": "evidence-gate-category-metrics-v1",
    "category_policy_sha256": "a" * 64,
    "variant_id": "observed",
    "response_snapshot_sha256": "b" * 64,
}


@pytest.mark.parametrize("family", ["ragas", "deterministic", "retrieval", "safety"])
def test_all_metric_families_receive_frozen_diagnostic_identity(family: str) -> None:
    outcome = {
        "benchmark_id": "case-01",
        "retrieval_mode": "dense_bm25_graph",
        "metric": "metric-01",
        "category": "fully_answerable",
    }

    attach_diagnostic_identity(
        outcome,
        metric_family=family,
        metric_version="metric-v1",
        identity=IDENTITY,
        judge_identity={"provider": "eval_openai", "model": "mimo-v2.5-pro"}
        if family == "ragas"
        else None,
    )

    assert outcome["metric_family"] == family
    assert outcome["metric_version"] == "metric-v1"
    assert outcome["category_policy_version"] == IDENTITY["category_policy_version"]
    assert outcome["category_policy_sha256"] == "a" * 64
    assert outcome["variant_id"] == "observed"
    assert outcome["response_snapshot_sha256"] == "b" * 64
    assert ("judge_identity" in outcome) is (family == "ragas")


def test_checkpoint_key_separates_variant_snapshot_family_and_policy() -> None:
    base = {
        "benchmark_id": "case-01",
        "retrieval_mode": "dense_bm25_graph",
        "metric": "faithfulness",
        "metric_family": "ragas",
        **IDENTITY,
    }
    keys = {
        outcome_checkpoint_key(base),
        outcome_checkpoint_key({**base, "variant_id": "oracle_context"}),
        outcome_checkpoint_key({**base, "response_snapshot_sha256": "c" * 64}),
        outcome_checkpoint_key({**base, "metric_family": "deterministic"}),
        outcome_checkpoint_key({**base, "category_policy_sha256": "d" * 64}),
    }
    assert len(keys) == 5


def test_legacy_checkpoint_key_remains_readable() -> None:
    assert outcome_checkpoint_key(
        {
            "benchmark_id": "case-01",
            "retrieval_mode": "vector",
            "metric": "faithfulness",
        }
    ) == ("case-01", "vector", "faithfulness")


def test_malformed_or_secret_bearing_identity_fails_closed() -> None:
    outcome = {
        "benchmark_id": "case-01",
        "retrieval_mode": "vector",
        "metric": "faithfulness",
        "category": "fully_answerable",
    }
    with pytest.raises(EvaluationOutcomeError, match="identity_fields_invalid"):
        attach_diagnostic_identity(
            outcome,
            metric_family="ragas",
            metric_version="metric-v1",
            identity={**IDENTITY, "credential": "secret"},
        )

