"""Runtime smoke tests for the Fisher detector.

These tests use the real AutoLens + HCIPy stack with a deliberately tiny
configuration to ensure the integrated detector executes end to end.
"""

from __future__ import annotations

import copy
import os
from pathlib import Path
import sys

import numpy as np
import pytest
import yaml

pytest.importorskip("autolens")
pytest.importorskip("hcipy")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from hwoslaps.lensing import generate_lensing_system
from hwoslaps.modeling.fisher_detector import FisherDetector
from hwoslaps.observation import generate_observation
from hwoslaps.psf.generator import generate_psf_system


def _load_master_config() -> dict:
    config_path = PROJECT_ROOT / "configs" / "master_config.yaml"
    with config_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _build_runtime_config(tmp_dir: Path) -> dict:
    config = _load_master_config()
    config["run_name"] = "fisher_runtime_test"
    config["plotting"]["enabled"] = False
    config["plotting"]["output_dir"] = str(tmp_dir)

    config["lensing"]["grid"] = {"shape": [31, 31], "pixel_scale": 0.1}
    config["lensing"]["lens_galaxy"]["mass"]["einstein_radius"] = 0.5
    config["lensing"]["lens_galaxy"]["mass"]["centre"] = [0.0, 0.0]
    config["lensing"]["source_galaxy"]["light"]["centre"] = [0.0, 0.0]
    config["lensing"]["source_galaxy"]["light"]["ell_comps"] = [0.03, -0.01]
    config["lensing"]["source_galaxy"]["light"]["intensity"] = 5.0
    config["lensing"]["source_galaxy"]["light"]["effective_radius"] = 0.2
    config["lensing"]["subhalo"] = {
        "enabled": True,
        "model": "PointMass",
        "mass": 1.0e8,
        "position": {"type": "direct", "centre": [0.2, -0.1]},
    }

    config["psf"]["kernel"]["shape_native"] = [11, 11]
    config["psf"]["hres_psf"]["num_pix"] = 128
    config["psf"]["hres_psf"]["num_airy"] = 6
    config["psf"]["hres_psf"]["save_highres_psf_npy"] = False

    aberr = config["psf"]["aberrations"]
    aberr["enable_segment_pistons"] = False
    aberr["enable_segment_tiptilts"] = False
    aberr["enable_segment_hexikes"] = False
    aberr["enable_global_zernikes"] = False
    aberr["segment_pistons"] = {}
    aberr["segment_tiptilts"] = {}
    aberr["segment_hexikes"] = {}
    aberr["global_zernikes"] = {}

    config["observation"]["exposure_time"] = 200.0
    config["observation"]["detector"]["gain"] = 1.0
    config["observation"]["detector"]["read_noise"] = 0.2
    config["observation"]["detector"]["dark_current"] = 0.002
    config["observation"]["detector"]["sky_background"] = 0.5

    config["modeling"]["detection"] = "fisher"
    config["modeling"]["fisher"]["snr_threshold"] = 3.0
    config["modeling"]["fisher"]["mode"] = "local"
    config["modeling"]["fisher"]["map"] = {
        "type": "explicit",
        "ring": {
            "num_angles": 4,
            "offset_pixels": 0.0,
        },
        "explicit_positions_yx": [[0.2, -0.1], [0.0, 0.3]],
    }
    config["modeling"]["fisher"]["mask_mode"] = "all_pixels"
    config["modeling"]["fisher"]["include_psf_nuisance"] = False
    config["modeling"]["fisher"]["compute_psf_mode_scan"] = False
    return config


@pytest.fixture(scope="module")
def runtime_setup(tmp_path_factory):
    tmp_dir = tmp_path_factory.mktemp("fisher-runtime")
    os.environ["NUMBA_CACHE_DIR"] = str(tmp_dir / "numba-cache")
    os.environ["MPLCONFIGDIR"] = str(tmp_dir / "mplconfig")
    os.environ["XDG_CACHE_HOME"] = str(tmp_dir / "xdg-cache")

    config = _build_runtime_config(tmp_dir)
    config_baseline = copy.deepcopy(config)
    config_baseline["lensing"]["subhalo"]["enabled"] = False

    psf_data = generate_psf_system(config["psf"], full_config=config)
    lensing_baseline = generate_lensing_system(
        config_baseline["lensing"],
        full_config=config_baseline,
    )
    lensing_test = generate_lensing_system(config["lensing"], full_config=config)
    observation_baseline = generate_observation(
        lensing_baseline,
        psf_data,
        observation_config=config_baseline["observation"],
        full_config=config_baseline,
    )
    observation_test = generate_observation(
        lensing_test,
        psf_data,
        observation_config=config["observation"],
        full_config=config,
    )

    return {
        "config": config,
        "psf_data": psf_data,
        "lensing_baseline": lensing_baseline,
        "lensing_test": lensing_test,
        "observation_baseline": observation_baseline,
        "observation_test": observation_test,
    }


def _make_detector(runtime_setup, *, mode: str, fisher_overrides: dict | None = None):
    config = copy.deepcopy(runtime_setup["config"])
    fisher_config = copy.deepcopy(config["modeling"]["fisher"])
    fisher_config["mode"] = mode
    if fisher_overrides:
        fisher_config.update(fisher_overrides)
    return FisherDetector(
        observation_baseline=runtime_setup["observation_baseline"],
        lensing_baseline=runtime_setup["lensing_baseline"],
        psf_data=runtime_setup["psf_data"],
        full_config=config,
        fisher_config=fisher_config,
    )


def _detector_stub_with_map_config(map_config: dict):
    detector = FisherDetector.__new__(FisherDetector)
    detector.map_config = map_config
    detector.map_type = str(map_config.get("type", "")).lower()
    detector._candidate_positions_cache = None
    detector._grid_layout_cache = None
    return detector


def test_fisher_detector_rejects_empty_explicit_map_positions():
    detector = _detector_stub_with_map_config(
        {
            "type": "explicit",
            "explicit_positions_yx": [],
        }
    )

    with pytest.raises(ValueError, match="explicit_positions_yx must be non-empty"):
        detector._candidate_positions()


@pytest.mark.parametrize(
    "explicit_positions",
    [
        [[0.1]],
        [["0.1", 0.2]],
        [[np.nan, 0.2]],
    ],
)
def test_fisher_detector_rejects_malformed_explicit_map_positions(explicit_positions):
    detector = _detector_stub_with_map_config(
        {
            "type": "explicit",
            "explicit_positions_yx": explicit_positions,
        }
    )

    with pytest.raises(ValueError, match="explicit_positions_yx"):
        detector._candidate_positions()


def test_fisher_detector_runtime_local_executes(runtime_setup):
    detector = _make_detector(runtime_setup, mode="local")
    local = detector.compute_local(
        observation_test=runtime_setup["observation_test"],
        lensing_test=runtime_setup["lensing_test"],
    )

    assert detector.n_psf_fit_modes == 0
    assert detector.n_psf_scan_modes == 0
    assert np.isfinite(local.snr_asimov)
    assert np.isfinite(local.delta_chi2_profiled)
    assert local.psf_mode_scan is None


def test_fisher_detector_runtime_map_executes(runtime_setup):
    detector = _make_detector(runtime_setup, mode="map")
    result = detector.compute_map()

    assert result.num_positions == 2
    assert result.positions_yx.shape == (2, 2)
    assert np.all(np.isfinite(result.snr_asimov_by_position))
    assert np.all(np.isfinite(result.delta_chi2_profiled_by_position))


def test_fisher_detector_runtime_psf_fit_executes(runtime_setup):
    detector = _make_detector(
        runtime_setup,
        mode="local",
        fisher_overrides={
            "include_psf_nuisance": True,
            "psf_basis": {"global_zernikes": {"mode_nolls": [4]}},
            "fit_psf_mode_selection": {"global_zernikes": {"mode_nolls": [4]}},
            "psf_mode_prior_sigmas": {"global_zernikes": 5.0},
        },
    )
    local = detector.compute_local(
        observation_test=runtime_setup["observation_test"],
        lensing_test=runtime_setup["lensing_test"],
    )

    assert detector.n_psf_fit_modes == 1
    assert detector.psf_fit_mode_names == ["psf.global_zernikes[4]"]
    assert np.isfinite(local.snr_asimov)
    assert local.psf_mode_scan is None


def test_fisher_detector_runtime_psf_scan_executes(runtime_setup):
    detector = _make_detector(
        runtime_setup,
        mode="local",
        fisher_overrides={
            "compute_psf_mode_scan": True,
            "psf_basis": {"global_zernikes": {"mode_nolls": [4]}},
            "scan_psf_mode_selection": {"global_zernikes": {"mode_nolls": [4]}},
            "psf_mode_prior_sigmas": {"global_zernikes": 5.0},
        },
    )
    local = detector.compute_local(
        observation_test=runtime_setup["observation_test"],
        lensing_test=runtime_setup["lensing_test"],
    )

    assert detector.n_psf_scan_modes == 1
    assert detector.psf_scan_mode_names == ["psf.global_zernikes[4]"]
    assert local.psf_mode_scan is not None
    assert len(local.psf_mode_scan.couplings) == 1
    coupling = local.psf_mode_scan.couplings[0]
    assert np.isfinite(coupling.z_per_unit)
    assert coupling.one_sigma_z is not None
    assert np.isfinite(coupling.one_sigma_z)


def test_fisher_detector_runtime_psf_scan_executes_with_signed_derivative_kernel(runtime_setup):
    config = copy.deepcopy(runtime_setup["config"])
    aberr = config["psf"]["aberrations"]
    aberr["enable_segment_hexikes"] = True
    aberr["segment_hexikes"] = {
        0: {4: 35.0},
        1: {5: -20.0},
    }
    aberr["enable_global_zernikes"] = False
    aberr["global_zernikes"] = {}

    config_baseline = copy.deepcopy(config)
    config_baseline["lensing"]["subhalo"]["enabled"] = False

    psf_data = generate_psf_system(config["psf"], full_config=config)
    lensing_baseline = generate_lensing_system(
        config_baseline["lensing"],
        full_config=config_baseline,
    )
    lensing_test = generate_lensing_system(config["lensing"], full_config=config)
    observation_baseline = generate_observation(
        lensing_baseline,
        psf_data,
        observation_config=config_baseline["observation"],
        full_config=config_baseline,
    )
    observation_test = generate_observation(
        lensing_test,
        psf_data,
        observation_config=config["observation"],
        full_config=config,
    )

    fisher_config = copy.deepcopy(config["modeling"]["fisher"])
    fisher_config.update({
        "compute_psf_mode_scan": True,
        "include_psf_nuisance": False,
        "psf_basis": {"global_zernikes": {"mode_nolls": [4]}},
        "scan_psf_mode_selection": {"global_zernikes": {"mode_nolls": [4]}},
        "psf_mode_prior_sigmas": {"global_zernikes": 5.0},
    })

    detector = FisherDetector(
        observation_baseline=observation_baseline,
        lensing_baseline=lensing_baseline,
        psf_data=psf_data,
        full_config=config,
        fisher_config=fisher_config,
    )
    local = detector.compute_local(
        observation_test=observation_test,
        lensing_test=lensing_test,
    )

    assert detector.n_psf_scan_modes == 1
    assert local.psf_mode_scan is not None
    coupling = local.psf_mode_scan.couplings[0]
    assert np.isfinite(coupling.z_per_unit)
