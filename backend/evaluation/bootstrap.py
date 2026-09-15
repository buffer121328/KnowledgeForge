"""Deterministic paired bootstrap utilities for approved offline evaluations.

The helper intentionally operates on already-reviewed, paired score vectors. It
does not call a judge, load application context, or infer scores from answers.
Callers must enforce the benchmark's sample-size and approval policy before
publishing the resulting interval.
"""

from __future__ import annotations

import math
import random
from collections.abc import Sequence
from typing import Any


MIN_BOOTSTRAP_PAIRS = 30
DEFAULT_BOOTSTRAP_RESAMPLES = 1_000


def _finite(value: Any) -> float:
    """Return the finite."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("bootstrap scores must be numeric")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError("bootstrap scores must be finite")
    return number


def _percentile(values: Sequence[float], percentile: float) -> float:
    """Return the percentile."""
    ordered = sorted(values)
    if not ordered:
        raise ValueError("bootstrap distribution is empty")
    position = (len(ordered) - 1) * percentile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def paired_bootstrap_mean_difference(
    vector_scores: Sequence[float],
    hybrid_scores: Sequence[float],
    *,
    resamples: int = DEFAULT_BOOTSTRAP_RESAMPLES,
    seed: int = 0,
    min_pairs: int = MIN_BOOTSTRAP_PAIRS,
) -> dict[str, Any]:
    """Return a reproducible percentile CI for hybrid-minus-vector scores.

    The function fails closed for unpaired, non-finite, undersized, or invalid
    input. It is suitable for offline reports only; no claim of significance is
    made when the interval crosses zero.
    """

    if len(vector_scores) != len(hybrid_scores):
        raise ValueError("bootstrap score vectors must be paired")
    if len(vector_scores) < min_pairs:
        raise ValueError(f"at least {min_pairs} paired scores are required")
    if resamples < 1:
        raise ValueError("resamples must be positive")
    vector = [_finite(value) for value in vector_scores]
    hybrid = [_finite(value) for value in hybrid_scores]
    differences = [right - left for left, right in zip(vector, hybrid, strict=True)]
    observed = sum(differences) / len(differences)
    rng = random.Random(seed)
    distribution: list[float] = []
    for _ in range(resamples):
        sample = [differences[rng.randrange(len(differences))] for _ in differences]
        distribution.append(sum(sample) / len(sample))
    lower = _percentile(distribution, 0.025)
    upper = _percentile(distribution, 0.975)
    return {
        "pairs": len(differences),
        "resamples": resamples,
        "seed": seed,
        "observed_mean_difference": round(observed, 6),
        "confidence_interval_95": [round(lower, 6), round(upper, 6)],
        "stable_positive_gain": lower > 0,
        "stable_negative_gain": upper < 0,
    }


__all__ = [
    "DEFAULT_BOOTSTRAP_RESAMPLES",
    "MIN_BOOTSTRAP_PAIRS",
    "paired_bootstrap_mean_difference",
]
