"""Artist data from actual forecast/reduction products, with asymmetric transport oracles."""

from dataclasses import replace

import numpy as np
import pytest

from hwoslaps.analysis.reach import mass_reach
from hwoslaps.analysis.reductions import summarize
from hwoslaps.fisher.positions import explicit_positions, grid_positions
from hwoslaps.fisher.result import ForecastResult
from hwoslaps.plotting import plot_detection_map, plot_knowledge_error, plot_mass_curve, plot_statistic_map

pytestmark = pytest.mark.backend


def test_forecast_plots_draw_the_result_values(plt, forecast_product):
    result = forecast_product
    _, ax = plt.subplots()
    assert plot_statistic_map(result, "q_asimov", mass_index=1, ax=ax) is ax
    np.testing.assert_array_equal(ax.images[0].get_array(), result.q_asimov[1].reshape(3, 3))
    assert ax.images[0].origin == "lower"
    assert ax.images[0].get_extent() == pytest.approx((-.6, .6, -.6, .6))
    assert ax.get_xlabel() == "x (arcsec)" and ax.get_ylabel() == "y (arcsec)"

    # Sparse rows must retain an unevaluated hole, not paint it as a nondetection.
    keep = np.array([True, True, True, True, False, True, True, True, True])
    sparse = replace(result, positions=result.positions.select(keep), fisher_raw=result.fisher_raw[:, keep],
                     fisher_profiled=result.fisher_profiled[:, keep])
    sparse_ax = plot_statistic_map(sparse, "q_asimov", mass_index=1)
    expected = result.q_asimov[1].reshape(3, 3).copy()
    expected[1, 1] = np.nan
    np.testing.assert_array_equal(sparse_ax.images[0].get_array().filled(np.nan), expected)

    summary = summarize(result, q_threshold=float(np.min(result.q_asimov)) / 2)
    for quantity in ("q_max", "detectable_fraction", "detectable_area_arcsec2"):
        curve_ax = plot_mass_curve(summary, quantity)
        np.testing.assert_array_equal(curve_ax.lines[0].get_xdata(), [1e8, 2e8])
        np.testing.assert_array_equal(curve_ax.lines[0].get_ydata(), getattr(summary, quantity))
        assert curve_ax.get_xscale() == "log"
    target = float(np.mean(summary.q_max))
    reach = mass_reach(summary, quantity="q_max", target=target, interpolation="linear")
    reached = plot_mass_curve(summary, "q_max", reach=reach)
    np.testing.assert_array_equal(reached.lines[1].get_ydata(), [target, target])
    np.testing.assert_array_equal(reached.lines[2].get_xdata(), [reach.mass_msun, reach.mass_msun])
    bound = mass_reach(summary, quantity="q_max", target=float(np.max(summary.q_max)) * 2, interpolation="linear")
    bounded = plot_mass_curve(summary, "q_max", reach=bound)
    assert len(bounded.lines) == 2  # no invented crossing for an above-range bound


def test_detection_map_uses_caller_threshold_metric_and_amplitude_sign(plt):
    positions = grid_positions((0, 0), spacing_arcsec=1, half_width_arcsec=1, annulus=None)
    information = np.full((2, 9), 4.)
    amplitudes = np.array([[1., 1, 1, 1, 1, 1, 1, 1, 1], [-2., 2, 1, 2, 1, 2, 1, 2, 1]])
    result = ForecastResult(np.array([1e8, 2e8]), positions, information + 1, information,
                            amplitudes, np.zeros((2, 9)), "kernel", {}, {})
    ax = plot_detection_map(result, q_threshold=16, mass_index=1)
    np.testing.assert_array_equal(ax.images[0].get_array(), [[0, 1, 0], [1, 0, 1], [0, 1, 0]])
    diagnostic = plot_detection_map(result, q_threshold=3, mass_index=1, metric="q_asimov")
    np.testing.assert_array_equal(diagnostic.images[0].get_array(), np.ones((3, 3)))
    stricter = plot_detection_map(result, q_threshold=17, mass_index=1)
    np.testing.assert_array_equal(stricter.images[0].get_array(), np.zeros((3, 3)))


def test_knowledge_error_plot_preserves_distinct_ratios_and_floor_gaps(plt, area_product):
    ax = plot_knowledge_error(area_product)
    for line, expected in zip(ax.lines, ([np.nan, .5], [np.nan, 1.], [np.nan, .5])):
        np.testing.assert_array_equal(line.get_xdata(), [1e8, 2e8])
        np.testing.assert_array_equal(line.get_ydata(), expected)
    assert ax.lines[0].get_label() == "Retention"
    assert "R" in ax.lines[1].get_label() and "selection" in ax.lines[2].get_label()
    assert ax.get_xscale() == "log"
    # Full-domain spurious/reference =3/4, distinct from in-selection F=2/4.
    assert area_product.spurious_area_arcsec2[1] / area_product.reference_area_arcsec2[1] == .75


@pytest.mark.parametrize("failure", ["lattice", "missing_statistic", "unknown_statistic", "missing_area",
                                    "unknown_quantity", "negative_index", "bool_index", "large_index", "foreign_reach"])
def test_map_plots_refuse_missing_lattice_or_statistic(plt, forecast_product, failure):
    result = forecast_product
    if failure == "lattice":
        result = replace(result, positions=explicit_positions(result.positions_yx, (0, 0)))
        action = lambda: plot_statistic_map(result, "q_asimov", mass_index=0)
    elif failure in ("missing_statistic", "unknown_statistic"):
        action = lambda: plot_statistic_map(result, "q_mismatch" if failure == "missing_statistic" else "absent", mass_index=0)
    elif failure in ("negative_index", "bool_index", "large_index"):
        index = {"negative_index": -1, "bool_index": True, "large_index": 2}[failure]
        action = lambda: plot_statistic_map(result, "q_asimov", mass_index=index)
    elif failure == "missing_area":
        result = replace(result, positions=explicit_positions(result.positions_yx, (0, 0)))
        summary = summarize(result, q_threshold=1)
        action = lambda: plot_mass_curve(summary, "detectable_area_arcsec2")
    elif failure == "foreign_reach":
        summary = summarize(result, q_threshold=1)
        reached = mass_reach(summary, quantity="detectable_fraction", target=.5, interpolation="linear")
        action = lambda: plot_mass_curve(summary, "q_max", reach=reached)
    else:
        summary = summarize(result, q_threshold=1)
        action = lambda: plot_mass_curve(summary, "absent")
    with pytest.raises(ValueError):
        action()
