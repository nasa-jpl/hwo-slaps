"""Real renderer/backend proofs for the reusable position/mass API."""

import copy

import numpy as np
import pytest

pytest.importorskip("autolens")
pytest.importorskip("hcipy")

from test_fisher_grid_map import _make_detector, grid_setup as grid_setup
from hwoslaps.modeling.fisher_detector import FisherDetector
from hwoslaps.modeling.fisher_geometry import select_aperture_and_perimeter
from hwoslaps.modeling.forecast_results import summarize_forecast
from hwoslaps.psf.utils import DetectorPSF, pyauto_kernel_native


@pytest.mark.parametrize("engine,batch_size", [("reference", 16), ("jax", 7), ("jax", 8)])
def test_sparse_mass_bank_matches_dense_reference_and_reuses_accelerated_products(
    grid_setup, engine, batch_size
):
    if engine == "jax":
        pytest.importorskip("jax")
    detector = _make_detector(
        grid_setup,
        {**grid_setup["config"]["modeling"]["fisher"]["map"], "engine": engine, "batch_size": batch_size},
    )
    layout = detector._grid_layout()
    selection = select_aperture_and_perimeter(layout, (0.0, 0.0), 0.1)
    result = detector.evaluate_masses(
        [8e7, 1e8], selection.positions_yx, domain_positions_yx=layout.positions_yx
    )
    indices = np.asarray(selection.node_indices)
    expected = grid_setup["grid_map"].q_asimov_2d[indices[:, 0], indices[:, 1]]
    np.testing.assert_allclose(result.q_asimov[-1], expected, rtol=1e-6, atol=1e-12)
    summary = summarize_forecast(result, 10, cell_areas_arcsec2=layout.spacing_arcsec**2)
    assert summary.evaluated_position_count == len(selection.positions_yx)
    if engine == "jax":
        original = detector._jax_grid_engine
        detector.evaluate_positions(
            selection.positions_yx, mass_msun=1e8, domain_positions_yx=layout.positions_yx
        )
        assert detector._jax_grid_engine is original
        rebuilt = _make_detector(grid_setup, {**detector.map_config, "engine": "jax"})
        oracle = rebuilt.evaluate_positions(
            selection.positions_yx, mass_msun=8e7, domain_positions_yx=layout.positions_yx
        )
        np.testing.assert_allclose(result.q_asimov[0], oracle.q_asimov[0], rtol=1e-9, atol=1e-12)


def test_external_truth_and_fit_kernels_use_real_mismatch_statistics_without_optical_regeneration(
    grid_setup, monkeypatch
):
    config = copy.deepcopy(grid_setup["config"])
    truth = DetectorPSF.from_array(pyauto_kernel_native(grid_setup["psf_data"].kernel), 0.1)
    fit = DetectorPSF.from_array(np.ones((3, 3)), 0.1)

    def unexpected_optical_generation(*args, **kwargs):
        raise AssertionError("external detector kernels must not regenerate an optical pupil")

    monkeypatch.setattr(FisherDetector, "_quiet_generate_psf_system", unexpected_optical_generation)
    detector = FisherDetector(
        observation_baseline=grid_setup["observation_baseline"],
        lensing_baseline=grid_setup["lensing_baseline"],
        psf_data=truth,
        fit_psf_data=fit,
        full_config=config,
        fisher_config=config["modeling"]["fisher"],
    )
    result = detector.evaluate_positions([[0.1, 0.1]])
    assert result.amplitude_hat is not None
    assert result.amplitude_spurious is not None
    assert np.all(np.isfinite(result.q_mismatch))
    assert np.any(result.q_spurious > 0)
