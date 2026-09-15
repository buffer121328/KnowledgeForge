"""Pre-registered rules for the 50-case routine diagnostic evaluation."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Literal

from ..bootstrap import (
    DEFAULT_BOOTSTRAP_RESAMPLES,
    MIN_BOOTSTRAP_PAIRS,
    paired_bootstrap_mean_difference,
)
from ..benchmarks.category_metric_policy import category_metric_policy_identity

DIAGNOSTIC_PREREGISTRATION_VERSION = "routine-diagnostic-preregistration-v1"
DIAGNOSTIC_RETRIEVAL_K_VALUES = (1, 3, 5, 8)
DIAGNOSTIC_BOOTSTRAP_SEED = 20260824
DIAGNOSTIC_MIN_ABSOLUTE_EFFECT = 0.15

EffectDecision = Literal[
    "material_improvement",
    "material_regression",
    "inconclusive",
]


def diagnostic_run_plan_identity() -> dict[str, Any]:
    """Bind the frozen policy and decision rules into a bounded run-plan identity."""

    return {
        "preregistration_version": DIAGNOSTIC_PREREGISTRATION_VERSION,
        "category_policy": category_metric_policy_identity(),
        "retrieval_k_values": list(DIAGNOSTIC_RETRIEVAL_K_VALUES),
        "bootstrap_seed": DIAGNOSTIC_BOOTSTRAP_SEED,
        "minimum_absolute_effect": DIAGNOSTIC_MIN_ABSOLUTE_EFFECT,
    }


def classify_paired_effect(
    baseline_scores: Sequence[float],
    candidate_scores: Sequence[float],
) -> dict[str, Any]:
    """Classify one frozen paired comparison using pre-registered evidence rules.

    This wrapper intentionally converts an undersized routine comparison into an
    explicit inconclusive result. Malformed or unpaired vectors still fail closed
    through the shared bootstrap validator.
    """

    if len(baseline_scores) != len(candidate_scores):
        raise ValueError("diagnostic score vectors must be paired")
    if len(baseline_scores) < MIN_BOOTSTRAP_PAIRS:
        return {
            "schema_version": DIAGNOSTIC_PREREGISTRATION_VERSION,
            "decision": "inconclusive",
            "reason_code": "insufficient_paired_samples",
            "pairs": len(baseline_scores),
            "required_pairs": MIN_BOOTSTRAP_PAIRS,
            "minimum_absolute_effect": DIAGNOSTIC_MIN_ABSOLUTE_EFFECT,
        }

    interval = paired_bootstrap_mean_difference(
        baseline_scores,
        candidate_scores,
        resamples=DEFAULT_BOOTSTRAP_RESAMPLES,
        seed=DIAGNOSTIC_BOOTSTRAP_SEED,
        min_pairs=MIN_BOOTSTRAP_PAIRS,
    )
    observed = interval["observed_mean_difference"]
    decision: EffectDecision = "inconclusive"
    reason_code = "confidence_interval_crosses_zero"
    if abs(observed) < DIAGNOSTIC_MIN_ABSOLUTE_EFFECT:
        reason_code = "effect_below_material_threshold"
    elif interval["stable_positive_gain"]:
        decision = "material_improvement"
        reason_code = "paired_material_improvement"
    elif interval["stable_negative_gain"]:
        decision = "material_regression"
        reason_code = "paired_material_regression"
    return {
        "schema_version": DIAGNOSTIC_PREREGISTRATION_VERSION,
        "decision": decision,
        "reason_code": reason_code,
        "minimum_absolute_effect": DIAGNOSTIC_MIN_ABSOLUTE_EFFECT,
        **interval,
    }


__all__ = [
    "DIAGNOSTIC_BOOTSTRAP_SEED",
    "DIAGNOSTIC_MIN_ABSOLUTE_EFFECT",
    "DIAGNOSTIC_PREREGISTRATION_VERSION",
    "DIAGNOSTIC_RETRIEVAL_K_VALUES",
    "classify_paired_effect",
    "diagnostic_run_plan_identity",
]
