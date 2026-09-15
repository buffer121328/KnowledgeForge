from __future__ import annotations

import pytest

from evaluation.bootstrap import paired_bootstrap_mean_difference


def test_paired_bootstrap_is_reproducible_and_reports_stable_gain() -> None:
    vector = [0.4] * 30
    hybrid = [0.6] * 30

    first = paired_bootstrap_mean_difference(vector, hybrid, seed=7)
    second = paired_bootstrap_mean_difference(vector, hybrid, seed=7)

    assert first == second
    assert first["pairs"] == 30
    assert first["resamples"] == 1000
    assert first["stable_positive_gain"] is True
    assert first["confidence_interval_95"] == [0.2, 0.2]


@pytest.mark.parametrize(
    "vector,hybrid,error",
    [
        ([0.1] * 29, [0.2] * 29, "at least"),
        ([0.1] * 30, [0.2] * 29, "paired"),
        ([float("nan")] * 30, [0.2] * 30, "finite"),
    ],
)
def test_paired_bootstrap_fails_closed(vector, hybrid, error: str) -> None:
    with pytest.raises(ValueError, match=error):
        paired_bootstrap_mean_difference(vector, hybrid)
