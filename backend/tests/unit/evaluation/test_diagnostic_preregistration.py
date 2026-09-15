"""Acceptance coverage for the routine diagnostic preregistration."""

from __future__ import annotations

import pytest

from evaluation.diagnostic.preregistration import (
    DIAGNOSTIC_BOOTSTRAP_SEED,
    DIAGNOSTIC_MIN_ABSOLUTE_EFFECT,
    DIAGNOSTIC_PREREGISTRATION_VERSION,
    DIAGNOSTIC_RETRIEVAL_K_VALUES,
    classify_paired_effect,
    diagnostic_run_plan_identity,
)
from evaluation.benchmarks.category_metric_policy import category_metric_policy_identity
from evaluation.evidence_gate.benchmark import REQUIRED_EVIDENCE_GATE_CATEGORIES


def test_one_fixture_per_category_is_explicitly_inconclusive() -> None:
    """Eleven apparently improved cases are too weak for a causal recommendation."""
    categories = sorted(REQUIRED_EVIDENCE_GATE_CATEGORIES)
    result = classify_paired_effect(
        [0.0 for _category in categories],
        [1.0 for _category in categories],
    )

    assert len(categories) == 11
    assert result == {
        "schema_version": "routine-diagnostic-preregistration-v1",
        "decision": "inconclusive",
        "reason_code": "insufficient_paired_samples",
        "pairs": 11,
        "required_pairs": 30,
        "minimum_absolute_effect": 0.15,
    }


def test_stable_large_paired_lift_is_material_under_frozen_rule() -> None:
    result = classify_paired_effect([0.4] * 30, [0.6] * 30)

    assert result["decision"] == "material_improvement"
    assert result["reason_code"] == "paired_material_improvement"
    assert result["observed_mean_difference"] == pytest.approx(0.2)
    assert result["confidence_interval_95"] == [0.2, 0.2]
    assert result["seed"] == 20260824


def test_stable_but_small_lift_does_not_trigger_a_treatment() -> None:
    result = classify_paired_effect([0.4] * 30, [0.5] * 30)

    assert result["decision"] == "inconclusive"
    assert result["reason_code"] == "effect_below_material_threshold"
    assert result["observed_mean_difference"] == pytest.approx(0.1)


def test_preregistered_identity_and_retrieval_depths_are_frozen() -> None:
    assert DIAGNOSTIC_PREREGISTRATION_VERSION == (
        "routine-diagnostic-preregistration-v1"
    )
    assert DIAGNOSTIC_RETRIEVAL_K_VALUES == (1, 3, 5, 8)
    assert DIAGNOSTIC_BOOTSTRAP_SEED == 20260824
    assert DIAGNOSTIC_MIN_ABSOLUTE_EFFECT == 0.15


def test_diagnostic_run_plan_binds_category_policy_identity() -> None:
    identity = diagnostic_run_plan_identity()

    assert identity["category_policy"] == category_metric_policy_identity()
    assert identity["preregistration_version"] == DIAGNOSTIC_PREREGISTRATION_VERSION
    assert identity["retrieval_k_values"] == [1, 3, 5, 8]
    assert identity["bootstrap_seed"] == DIAGNOSTIC_BOOTSTRAP_SEED
    assert identity["minimum_absolute_effect"] == DIAGNOSTIC_MIN_ABSOLUTE_EFFECT
