"""Spatial reductions of analysis.reductions.

Oracles: the 41621de ladder reduction (``scripts/run_ladder.py::_rung_metrics``:
q_max over the closed aperture, detected cell area and fraction inside it, and
the perimeter clipping flag), evaluated by hand on the maps below, and hand
arrays for the mismatch rules.
"""

import numpy as np
import pytest

from hwoslaps.analysis.reductions import aperture_selection, summarize
from hwoslaps.fisher.positions import explicit_positions, grid_positions
from hwoslaps.fisher.result import ForecastResult

nan = np.nan


def forecast(positions, fisher_profiled, amplitude_hat=None, amplitude_spurious=None):
    profiled = np.asarray(fisher_profiled, dtype=float)
    return ForecastResult(
        masses_msun=np.geomspace(1e7, 1e8, profiled.shape[0]), positions=positions,
        fisher_raw=profiled + 1.0, fisher_profiled=profiled,
        amplitude_hat=amplitude_hat, amplitude_spurious=amplitude_spurious,
        psf_relation="matched" if amplitude_hat is None else "kernel", config={}, provenance={})


def lattice(spacing, half_width):
    return grid_positions((0.0, 0.0), spacing_arcsec=spacing, half_width_arcsec=half_width, annulus=None)


def test_summary_reproduces_paper_rung_metrics():
    below = np.ones((5, 5))
    below[0, 0], below[1, 1], below[2, 2] = 50.0, 999.0, 7.0
    inside = below.copy()
    inside[2, 3], inside[1, 2] = 12.0, 30.0
    values = np.stack((below.reshape(-1), inside.reshape(-1)))
    ladder = forecast(lattice(1.0, 2.0), values)
    selection = aperture_selection(ladder, centre_yx=(0.0, 0.0), radius_arcsec=1.0)
    summary = summarize(ladder, q_threshold=10.0, selection=selection)
    assert summary.selected_count == 5 and summary.metric == "q_asimov" and summary.q_threshold == 10.0
    np.testing.assert_array_equal(summary.q_max, [7.0, 30.0])
    np.testing.assert_array_equal(summary.detectable_count, [0, 2])
    np.testing.assert_array_equal(summary.detectable_area_arcsec2, [0.0, 2.0])
    np.testing.assert_array_equal(summary.detectable_fraction, [0.0, 2 / 5])
    np.testing.assert_array_equal(summary.boundary_detectable, [True, True])
    np.testing.assert_array_equal(summary.masses_msun, ladder.masses_msun)

    disc = ladder.positions.within((0.0, 0.0), 1.0)
    sparse = forecast(ladder.positions.aperture((0.0, 0.0), 1.0, include_boundary=True),
                      values[:, disc | ladder.boundary])
    sparse_summary = summarize(sparse, q_threshold=10.0,
                               selection=aperture_selection(sparse, centre_yx=(0.0, 0.0), radius_arcsec=1.0))
    for name in ("q_max", "detectable_count", "detectable_area_arcsec2", "detectable_fraction",
                 "boundary_detectable"):
        np.testing.assert_array_equal(getattr(sparse_summary, name), getattr(summary, name), err_msg=name)
    disc_only = forecast(ladder.positions.aperture((0.0, 0.0), 1.0, include_boundary=False), values[:, disc])
    assert summarize(disc_only, q_threshold=10.0).boundary_detectable is None

    small = forecast(lattice(0.5, 0.5), np.stack((np.arange(9.0), np.arange(9.0) + 5.0)))
    everything = summarize(small, q_threshold=10.0,
                           selection=aperture_selection(small, centre_yx=(0.0, 0.0), radius_arcsec=10.0))
    np.testing.assert_array_equal(everything.q_max, [8.0, 13.0])
    np.testing.assert_array_equal(everything.detectable_area_arcsec2, [0.0, 4 * 0.25])
    np.testing.assert_array_equal(everything.detectable_fraction, [0.0, 4 / 9])
    np.testing.assert_array_equal(everything.boundary_detectable, [False, True])

    counts = range(10, 15)
    paper_scale = forecast(lattice(0.05, 0.1), np.stack([np.arange(25.0) + count - 10 for count in counts]))
    rung_areas = summarize(paper_scale, q_threshold=15.0).detectable_area_arcsec2
    assert rung_areas.tobytes() == np.array([count * float(0.05) ** 2 for count in counts]).tobytes()


def test_mismatch_q_max_zeroes_non_positive_amplitudes():
    positions = explicit_positions([[0.0, 0.1], [0.1, 0.0], [0.1, 0.1]], (0.0, 0.0))
    mismatched = forecast(positions, np.full((1, 3), 4.0), amplitude_hat=np.array([[2.0, -3.0, 1.0]]),
                          amplitude_spurious=np.array([[-1.0, 0.5, 2.0]]))
    data = summarize(mismatched, q_threshold=10.0)
    assert data.metric == "q_mismatch"
    np.testing.assert_array_equal(data.q_max, [16.0])
    np.testing.assert_array_equal(data.detectable_fraction, [1 / 3])
    assert data.detectable_area_arcsec2 is None and data.boundary_detectable is None
    null = summarize(mismatched, q_threshold=10.0, metric="q_spurious")
    np.testing.assert_array_equal(null.q_max, [16.0])
    np.testing.assert_array_equal(null.detectable_count, [1])
    matched_diagnostic = summarize(mismatched, q_threshold=3.0, metric="q_asimov")
    np.testing.assert_array_equal(matched_diagnostic.q_max, [4.0])
    np.testing.assert_array_equal(matched_diagnostic.detectable_fraction, [1.0])


def test_non_finite_values_at_consumed_positions_raise():
    layout = lattice(1.0, 2.0)
    selection = layout.within((0.0, 0.0), 1.0)
    information = np.full((1, 25), 4.0)
    amplitude = np.full((1, 25), 2.0)
    interior = amplitude.copy()
    interior[0, 6] = np.nan  # Failed interior row outside selection and boundary.
    skipped = forecast(layout, information, amplitude_hat=interior, amplitude_spurious=amplitude)
    np.testing.assert_array_equal(summarize(skipped, q_threshold=10.0, selection=selection).q_max, [16.0])
    with pytest.raises(ValueError, match="selected positions contains failed_numeric"):
        summarize(skipped, q_threshold=10.0)
    edge = amplitude.copy()
    edge[0, 0] = np.nan
    clipped = forecast(layout, information, amplitude_hat=edge, amplitude_spurious=amplitude)
    with pytest.raises(ValueError, match="boundary positions contains failed_numeric"):
        summarize(clipped, q_threshold=10.0, selection=selection)
    report = summarize(clipped, q_threshold=10.0, selection=selection, invalid_policy="report")
    np.testing.assert_array_equal(report.validity.failed_numeric_count, [0])
    np.testing.assert_array_equal(report.boundary_validity.failed_numeric_count, [1])
    assert report.boundary_detectable is None
    np.testing.assert_array_equal(report.q_max, [16.0])


def test_empty_aperture_and_empty_selection_raise():
    small = forecast(lattice(1.0, 1.0), np.ones((1, 9)))
    with pytest.raises(ValueError, match="no position lies inside"):
        aperture_selection(small, centre_yx=(5.0, 5.0), radius_arcsec=0.5)
    with pytest.raises(ValueError, match="at least one position"):
        summarize(small, q_threshold=1.0, selection=np.zeros(9, dtype=bool))


@pytest.mark.parametrize("metric", ["q_asimov", "q_mismatch", "q_spurious"])
def test_zero_information_is_usable_but_unidentified_in_every_summary_metric(metric):
    positions = explicit_positions([[0.0, 0.0], [0.1, 0.0], [0.2, 0.0]], (0.0, 0.0))
    result = forecast(positions, [[0.0, 4.0, 4.0]], amplitude_hat=[[np.nan, 0.0, 2.0]],
                      amplitude_spurious=[[np.nan, 0.0, 1.0]])
    summary = summarize(result, q_threshold=3.0, metric=metric)
    np.testing.assert_array_equal(summary.validity.valid_count, [2])
    np.testing.assert_array_equal(summary.validity.zero_information_count, [1])
    np.testing.assert_array_equal(summary.validity.failed_numeric_count, [0])
    np.testing.assert_array_equal(summary.validity.complete, [True])
    assert summary.validity.evaluated_count == summary.selected_count == 3
    assert summary.detectable_count[0] == (2 if metric == "q_asimov" else 1)
    assert np.isfinite(summary.q_max[0]) and np.isfinite(summary.detectable_fraction[0])


@pytest.mark.parametrize("failure", ["nonfinite_amplitude", "nonfinite_information", "finite_provisional"])
def test_report_retains_fixed_selection_and_explicit_unknown_aggregate(failure):
    from dataclasses import replace
    positions = explicit_positions([[0.0, 0.0], [0.1, 0.0], [0.2, 0.0]], (0.0, 0.0))
    information = np.full((2, 3), 4.0)
    fitted = np.ones((2, 3))
    if failure == "nonfinite_amplitude":
        fitted[0, 1] = np.nan
    elif failure == "nonfinite_information":
        information[0, 1] = np.inf
    result = forecast(positions, information, amplitude_hat=fitted, amplitude_spurious=np.zeros((2, 3)))
    if failure == "finite_provisional":
        status = result.template_status.copy()
        status[0, 1] = "failed_numeric"
        result = replace(result, template_status=status)
    with pytest.raises(ValueError, match="failed_numeric"):
        summarize(result, q_threshold=3.0)
    summary = summarize(result, q_threshold=3.0, invalid_policy="report")
    assert summary.selected_count == 3 and summary.invalid_policy == "report"
    np.testing.assert_array_equal(summary.validity.failed_numeric_count, [1, 0])
    np.testing.assert_array_equal(summary.validity.valid_count, [2, 3])
    np.testing.assert_array_equal(summary.known_detectable_count, [2, 3])
    np.testing.assert_array_equal(summary.detectable_count, [np.nan, 3])
    np.testing.assert_array_equal(summary.detectable_fraction, [np.nan, 1.0])
    np.testing.assert_array_equal(summary.q_max, [np.nan, 4.0])
    assert summary.detectable_area_arcsec2 is None


def test_report_does_not_make_an_entirely_failed_grid_a_zero_area():
    layout = lattice(1.0, 1.0)
    information = np.full((1, 9), 4.0)
    result = forecast(layout, information, amplitude_hat=np.full((1, 9), np.nan),
                      amplitude_spurious=np.zeros((1, 9)))
    summary = summarize(result, q_threshold=3.0, invalid_policy="report")
    assert summary.validity.failed_numeric_count[0] == summary.selected_count == 9
    assert summary.known_detectable_count[0] == 0
    for field in ("q_max", "detectable_count", "detectable_fraction", "detectable_area_arcsec2"):
        assert np.isnan(getattr(summary, field)[0]), field
    assert summary.boundary_detectable is None


@pytest.mark.parametrize("policy", ["exclude", "ignore", None, True])
def test_summary_requires_a_declared_validity_policy(policy):
    result = forecast(lattice(1.0, 1.0), np.ones((1, 9)))
    with pytest.raises(ValueError, match="invalid_policy"):
        summarize(result, q_threshold=1.0, invalid_policy=policy)



def test_mass_reach_rejects_reported_missing_values_instead_of_creating_a_nondetection_bound():
    from hwoslaps.analysis import mass_reach
    positions = explicit_positions([[0.0, 0.0], [0.1, 0.0]], (0.0, 0.0))
    result = forecast(positions, [[1.0, 1.0], [4.0, 4.0]], amplitude_hat=[[1.0, 1.0], [np.nan, 1.0]],
                      amplitude_spurious=np.zeros((2, 2)))
    report = summarize(result, q_threshold=3.0, invalid_policy="report")
    for quantity in ("q_max", "detectable_fraction"):
        with pytest.raises(ValueError, match="values and target must be finite"):
            mass_reach(report, quantity=quantity, target=0.5, interpolation="linear")


def test_report_json_uses_null_and_retains_validity_through_strict_public_writer(tmp_path):
    import json
    from hwoslaps.artifacts import write_json
    positions = explicit_positions([[0.0, 0.0], [0.1, 0.0]], (0.0, 0.0))
    result = forecast(positions, [[4.0, 4.0]], amplitude_hat=[[1.0, np.nan]], amplitude_spurious=[[0.0, 0.0]])
    report = summarize(result, q_threshold=3.0, invalid_policy="report")
    mapping = report.to_mapping()
    assert mapping["q_max"] == [None] and mapping["detectable_fraction"] == [None]
    assert mapping["validity"]["failed_numeric_count"] == [1]
    assert mapping["validity"]["complete"] == [False]
    assert mapping["selected_count"] == 2 and mapping["known_detectable_count"] == [1]
    path = write_json(tmp_path / "summary.json", mapping)
    assert json.loads(path.read_text()) == mapping
    assert "NaN" not in path.read_text() and "Infinity" not in path.read_text()
