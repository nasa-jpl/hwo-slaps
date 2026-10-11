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
    interior = information.copy()
    interior[0, 6] = 0.0
    skipped = forecast(layout, interior, amplitude_hat=np.where(interior > 0, amplitude, nan),
                       amplitude_spurious=amplitude)
    np.testing.assert_array_equal(summarize(skipped, q_threshold=10.0, selection=selection).q_max, [16.0])
    with pytest.raises(ValueError, match="q_mismatch is not finite at a selected or boundary position"):
        summarize(skipped, q_threshold=10.0)
    edge = information.copy()
    edge[0, 0] = 0.0
    clipped = forecast(layout, edge, amplitude_hat=np.where(edge > 0, amplitude, nan), amplitude_spurious=amplitude)
    with pytest.raises(ValueError, match="not finite at a selected or boundary position"):
        summarize(clipped, q_threshold=10.0, selection=selection)


def test_empty_aperture_and_empty_selection_raise():
    small = forecast(lattice(1.0, 1.0), np.ones((1, 9)))
    with pytest.raises(ValueError, match="no position lies inside"):
        aperture_selection(small, centre_yx=(5.0, 5.0), radius_arcsec=0.5)
    with pytest.raises(ValueError, match="at least one position"):
        summarize(small, q_threshold=1.0, selection=np.zeros(9, dtype=bool))
