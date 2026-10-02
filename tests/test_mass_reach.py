"""Measured crossings, censoring and adaptive evaluation contracts."""

import numpy as np
import pytest

from hwoslaps.modeling.mass_reach import adaptive_mass_reach, mass_reach


@pytest.mark.parametrize("target,expected", [(0.25, 10**7.25), (0.75, 10**7.75)])
def test_arbitrary_fraction_crossing_has_an_independent_log_mass_oracle(target, expected):
    reach = mass_reach([1e7, 1e8], [0, 1], target)
    assert reach.status == "bracketed"
    assert reach.mass_msun == pytest.approx(expected)
    assert reach.lower_mass_msun == 1e7
    assert reach.upper_mass_msun == 1e8


def test_log_statistic_interpolation_and_exact_samples():
    reach = mass_reach([1e7, 1e8], [1, 100], 10, interpolation="log")
    assert reach.mass_msun == pytest.approx(np.sqrt(1e15))
    sampled = mass_reach([1e7, 1e8, 1e9], [1, 10, 20], 10)
    assert sampled.status == "sampled"
    assert sampled.mass_msun == 1e8


@pytest.mark.parametrize(
    "values,status,low,high", [([2, 3], "below_range", None, 1e7), ([0, 0.5], "above_range", 1e8, None)]
)
def test_unbracketed_reach_is_a_bound_and_never_an_extrapolation(values, status, low, high):
    result = mass_reach([1e7, 1e8], values, 1)
    assert result.status == status
    assert result.mass_msun is None
    assert result.lower_mass_msun == low
    assert result.upper_mass_msun == high


def test_nonmonotone_curve_does_not_report_a_unique_reach():
    result = mass_reach([1e7, 1e8, 1e9], [0.2, 0.8, 0.3], 0.5)
    assert result.status == "non_monotonic"
    assert result.mass_msun is None


def test_log_interpolation_with_a_zero_lower_bracket_fails_closed():
    with pytest.raises(ValueError, match="positive bracket"):
        mass_reach([1e7, 1e8], [0, 50], 10, interpolation="log")


@pytest.mark.parametrize(
    "masses,values",
    [([1e7, 1e7], [0, 1]), ([1e8, 1e7], [0, 1]), ([1e7, np.nan], [0, 1]), ([1e7, 1e8], [0, np.nan])],
)
def test_invalid_measured_axes_are_rejected(masses, values):
    with pytest.raises(ValueError):
        mass_reach(masses, values, 0.5)


def test_adaptive_reach_is_bounded_finite_and_never_repeats_an_evaluation():
    visited = []

    def evaluate(mass):
        visited.append(mass)
        return np.log10(mass) - 7

    result = adaptive_mass_reach(evaluate, 1e7, 1e9, 0.35, tolerance_dex=0.01, max_evaluations=12)
    assert len(visited) <= 12
    assert len(set(visited)) == len(visited)
    assert all(1e7 <= mass <= 1e9 for mass in visited)
    assert result.reach.mass_msun == pytest.approx(10**7.35)
    assert np.log10(result.reach.upper_mass_msun / result.reach.lower_mass_msun) <= 0.01
