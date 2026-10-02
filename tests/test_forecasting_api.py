"""Real public workflow contracts for prepared subhalo forecasts."""
from copy import deepcopy

import numpy as np
import pytest

pytest.importorskip("autolens")

from hwoslaps import (
    ForecastResult, forecast, load_forecast_result, prepare_forecast,
    simulate,
)
from hwoslaps.psf import DetectorPSF
from test_fisher_grid_map import grid_setup


@pytest.fixture
def kernel_config(grid_setup, tmp_path):
    config = deepcopy(grid_setup["config"])
    config.pop("plotting", None)
    config.pop("run_name", None)
    truth = np.zeros((3, 3)); truth[1, 1] = 1
    fit = np.array([[0, 0.05, 0], [0.05, 0.8, 0.05], [0, 0.05, 0]])
    np.save(tmp_path / "truth.npy", truth)
    np.save(tmp_path / "fit.npy", fit)
    config["psf"] = {"provider": "kernel", "kernel": {
        "path": str(tmp_path / "truth.npy"), "pixel_scale_arcsec": 0.1,
    }, "fit_kernel": {"path": str(tmp_path / "fit.npy"), "pixel_scale_arcsec": 0.1}}
    return config


def test_file_kernel_forecast_preserves_declared_mismatch_and_real_output(
    kernel_config, tmp_path, monkeypatch,
):
    before = deepcopy(kernel_config)
    monkeypatch.chdir(tmp_path)
    from hwoslaps.observation import generator
    monkeypatch.setattr(generator, "apply_detector_noise", lambda **kwargs: pytest.fail("forecast sampled noise"))
    prepared = prepare_forecast(kernel_config)
    assert isinstance(prepared.truth_psf, DetectorPSF)
    assert prepared.scene.has_subhalo is False
    assert prepared.observation.metadata["sample_noise"] is False
    assert prepared.config["psf"]["kernel"]["path"] == kernel_config["psf"]["kernel"]["path"]
    assert prepared.config["modeling"]["fit_psf"]["mode"] == "kernel"
    result = forecast(prepared, masses=[8e7, 1e8], positions=[[0.1, 0.1]])
    assert isinstance(result, ForecastResult)
    assert result.q_asimov.shape == (2, 1)
    assert result.q_mismatch is not None and result.q_spurious is not None
    assert np.all(np.isfinite(result.q_mismatch))
    assert result.runtime_provenance["truth_kernel"] != result.runtime_provenance["fit_kernel"]
    assert not (tmp_path / "outputs").exists()
    assert kernel_config == before
    path = result.save_npz(tmp_path / "result.npz")
    loaded = load_forecast_result(path)
    np.testing.assert_array_equal(loaded.q_mismatch, result.q_mismatch)
    assert loaded.runtime_provenance == result.runtime_provenance
    with pytest.raises(FileExistsError):
        result.save_npz(path)


def test_prepared_trial_simulation_is_an_explicit_injection(kernel_config):
    kernel_config["psf"].pop("fit_kernel")
    prepared = prepare_forecast(kernel_config)
    trial = prepared.trial(1e8, (0.1, 0.1))
    injected = simulate(prepared, trial=trial, sample_noise=False)
    assert injected.metadata["subhalo"]["enabled"] is True
    assert injected.metadata["subhalo"]["mass"] == trial.mass_msun
    assert injected.metadata["subhalo"]["position"]["centre"] == list(trial.position_yx_arcsec)
    assert not np.array_equal(injected.data.native, prepared.observation.data.native)
    assert prepared.scene.has_subhalo is False
    assert injected.metadata["sample_noise"] is False
    control = simulate(prepared, trial=trial, injected=False, sample_noise=False)
    assert control.metadata["subhalo"]["enabled"] is False
    np.testing.assert_array_equal(control.data.native, prepared.observation.data.native)


def test_supplied_kernel_sampling_fails_before_preparation(kernel_config):
    wrong = DetectorPSF.from_array(np.ones((3, 3)), 0.2)
    with pytest.raises(ValueError, match="angular sampling"):
        prepare_forecast(kernel_config, psf=wrong)


@pytest.mark.parametrize("provider", ["kernel", "fit_kernel"])
def test_effective_file_identity_rejects_changed_calibration(kernel_config, provider):
    prepared = prepare_forecast(kernel_config)
    np.save(kernel_config["psf"][provider]["path"], np.ones((3, 3)))
    with pytest.raises(ValueError, match="detector-response identity"):
        prepare_forecast(prepared.config)


def test_external_kernel_does_not_invent_optical_nuisance_basis(kernel_config):
    kernel_config["modeling"]["fisher"]["include_psf_nuisance"] = True
    kernel_config["modeling"]["fisher"]["psf_basis"] = {"global_zernikes": [4]}
    with pytest.raises(ValueError, match="no declared wavefront basis"):
        prepare_forecast(kernel_config)


def test_prepared_configuration_cannot_relabel_cached_science(kernel_config):
    prepared = prepare_forecast(kernel_config)
    before = forecast(prepared, masses=[1e8], positions=[[0.1, 0.1]])
    original = prepared.config["observation"]["exposure_time"]
    external = prepared.config
    external["observation"]["exposure_time"] = original * 2
    assert prepared.config["observation"]["exposure_time"] == original
    after = forecast(prepared, masses=[1e8], positions=[[0.1, 0.1]])
    np.testing.assert_array_equal(after.q_asimov, before.q_asimov)
    assert after.runtime_provenance["config_hash"] == before.runtime_provenance["config_hash"]


def test_explicit_fit_psf_cannot_write_artifacts_during_preparation(
    kernel_config, grid_setup, tmp_path, monkeypatch,
):
    """An optical model export flag must fail before any generated output."""
    kernel_config["psf"].pop("fit_kernel")
    model_psf = deepcopy(grid_setup["config"]["psf"])
    model_psf["hres_psf"]["save_highres_psf_npy"] = True
    kernel_config["modeling"]["fit_psf"] = {"mode": "explicit", "psf": model_psf}
    monkeypatch.chdir(tmp_path)
    before = set(tmp_path.rglob("*"))
    with pytest.raises(ValueError, match="Model PSF export is an explicit output operation"):
        prepare_forecast(kernel_config)
    assert set(tmp_path.rglob("*")) == before
