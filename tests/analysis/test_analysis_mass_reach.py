"""Mass reach of analysis.reach: crossings, censoring, summary chaining and adaptive refinement.

Oracles: log-mass interpolation by hand, the analytic crossing of a power law,
and the measured bracket itself (no extrapolation).
"""

from dataclasses import replace

import numpy as np
import pytest

from hwoslaps.analysis.reach import adaptive_mass_reach, crossing, mass_reach
from hwoslaps.analysis.reductions import ForecastSummary


@pytest.mark.parametrize("target,expected", [(0.25, 10**7.25), (0.75, 10**7.75)])
def test_arbitrary_fraction_crossing_has_an_independent_log_mass_oracle(target, expected):
    reach = crossing([1e7, 1e8], [0, 1], target=target, interpolation="linear")
    assert reach.status == "bracketed"
    assert reach.mass_msun == pytest.approx(expected)
    assert reach.lower_mass_msun == 1e7
    assert reach.upper_mass_msun == 1e8


def test_log_statistic_interpolation_and_exact_samples():
    reach = crossing([1e7, 1e8], [1, 100], target=10, interpolation="log")
    assert reach.mass_msun == pytest.approx(np.sqrt(1e15))
    sampled = crossing([1e7, 1e8, 1e9], [1, 10, 20], target=10, interpolation="linear")
    assert sampled.status == "sampled"
    assert sampled.mass_msun == 1e8


@pytest.mark.parametrize(
    "values,status,low,high", [([2, 3], "below_range", None, 1e7), ([0, 0.5], "above_range", 1e8, None)]
)
def test_unbracketed_reach_is_a_bound_and_never_an_extrapolation(values, status, low, high):
    result = crossing([1e7, 1e8], values, target=1, interpolation="linear")
    assert result.status == status
    assert result.mass_msun is None
    assert result.lower_mass_msun == low
    assert result.upper_mass_msun == high


def test_nonmonotone_curve_does_not_report_a_unique_reach():
    result = crossing([1e7, 1e8, 1e9], [0.2, 0.8, 0.3], target=0.5, interpolation="linear")
    assert result.status == "non_monotonic"
    assert result.mass_msun is None


def test_log_interpolation_with_a_zero_lower_bracket_fails_closed():
    with pytest.raises(ValueError, match="positive bracket"):
        crossing([1e7, 1e8], [0, 50], target=10, interpolation="log")


@pytest.mark.parametrize(
    "masses,values,interpolation",
    [([1e7, 1e7], [0, 1], "linear"), ([1e8, 1e7], [0, 1], "linear"), ([1e7, np.nan], [0, 1], "linear"),
     ([1e7, 1e8], [0, np.nan], "linear"), ([1e7, 1e8], [0, 1], "cubic")],
)
def test_invalid_measured_axes_are_rejected(masses, values, interpolation):
    with pytest.raises(ValueError):
        crossing(masses, values, target=0.5, interpolation=interpolation)


def test_adaptive_reach_is_bounded_finite_and_never_repeats_an_evaluation():
    visited = []

    def evaluate(mass):
        visited.append(mass)
        return np.log10(mass) - 7

    result = adaptive_mass_reach(evaluate, lower_mass_msun=1e7, upper_mass_msun=1e9, target=0.35,
                                 interpolation="linear", tolerance_dex=0.01, max_evaluations=12)
    assert len(visited) <= 12
    assert len(set(visited)) == len(visited)
    assert all(1e7 <= mass <= 1e9 for mass in visited)
    assert result.reach.mass_msun == pytest.approx(10**7.35)
    assert np.log10(result.reach.upper_mass_msun / result.reach.lower_mass_msun) <= 0.01
    np.testing.assert_array_equal(result.masses_msun, sorted(visited))


def test_adaptive_reach_stops_when_the_bracket_reaches_float_resolution():
    visited = []

    def evaluate(mass):
        visited.append(mass)
        if len(visited) > 200:
            raise RuntimeError("refinement keeps evaluating a mass it has already measured")
        return np.log10(mass) - 7

    result = adaptive_mass_reach(evaluate, lower_mass_msun=1e7, upper_mass_msun=1e9, target=0.35,
                                 interpolation="linear", tolerance_dex=1e-300, max_evaluations=1000)
    assert len(set(visited)) == len(visited)
    lower, upper = result.reach.lower_mass_msun, result.reach.upper_mass_msun
    assert not lower < np.sqrt(lower * upper) < upper
    assert result.reach.mass_msun == pytest.approx(10**7.35, rel=1e-12)


def test_power_law_q_crosses_at_the_analytic_mass_with_log_interpolation():
    masses = np.array([1e7, 1e8, 1e9])
    slope, target, analytic = 0.8, 10.0, 10**7.4
    q = (target / analytic**slope) * masses**slope
    logarithmic = crossing(masses, q, target=target, interpolation="log")
    assert logarithmic.status == "bracketed" and logarithmic.interpolation == "log"
    assert logarithmic.mass_msun == pytest.approx(analytic, rel=1e-12)
    linear = crossing(masses, q, target=target, interpolation="linear")
    assert linear.interpolation == "linear"
    assert linear.mass_msun < 0.9 * analytic
    with pytest.raises(TypeError, match="interpolation"):
        crossing(masses, q, target=target)


def test_mass_reach_reads_the_named_summary_quantity():
    summary = ForecastSummary(
        masses_msun=np.array([1e7, 1e8, 1e9]), q_threshold=10.0, metric="q_asimov", selected_count=4,
        q_max=np.array([2.0, 20.0, 200.0]), detectable_count=np.array([0, 1, 4]),
        detectable_fraction=np.array([0.0, 0.25, 1.0]), detectable_area_arcsec2=np.array([0.0, 0.01, 0.04]),
        boundary_detectable=np.array([False, False, True]))
    for quantity, target, interpolation in (("q_max", 10.0, "log"), ("detectable_fraction", 0.5, "linear"),
                                            ("detectable_area_arcsec2", 0.02, "log")):
        reach = mass_reach(summary, quantity=quantity, target=target, interpolation=interpolation)
        direct = crossing(summary.masses_msun, getattr(summary, quantity), target=target,
                          interpolation=interpolation)
        assert reach.mass_msun == direct.mass_msun and reach.status == direct.status == "bracketed"
        assert reach.quantity == quantity and direct.quantity is None
        assert reach.interpolation == interpolation and reach.target == target
    without_areas = replace(summary, detectable_area_arcsec2=None)
    with pytest.raises(ValueError, match="no detectable_area_arcsec2"):
        mass_reach(without_areas, quantity="detectable_area_arcsec2", target=0.02, interpolation="log")
    with pytest.raises(ValueError, match="quantity must be one of"):
        mass_reach(summary, quantity="detectable_count", target=1.0, interpolation="linear")
