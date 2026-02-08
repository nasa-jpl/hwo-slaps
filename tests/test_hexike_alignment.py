"""Hexike API alignment tests for hwoslaps.

These tests validate that hwoslaps' segment-hexike integration matches the
mainstream HCIPy API semantics in the local hcipy checkout.
"""

from __future__ import annotations

import copy
import importlib.util
from pathlib import Path

import numpy as np
import pytest
import yaml

try:
    import hcipy  # noqa: F401
    HCIPY_AVAILABLE = True
except Exception:
    HCIPY_AVAILABLE = False

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src" / "hwoslaps"


def _load_module(relative_path: str, module_name: str):
    module_path = SRC_ROOT / relative_path
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


validation = _load_module("config/validation.py", "hwoslaps_config_validation")

if HCIPY_AVAILABLE:
    aberration_models = _load_module("psf/aberration_models.py", "hwoslaps_psf_aberration_models")
    telescope_models = _load_module("psf/telescope_models.py", "hwoslaps_psf_telescope_models")
else:
    aberration_models = None
    telescope_models = None


@pytest.fixture(scope="module")
def minimal_psf_config():
    return {
        "telescope": {
            "gap_size": 0.006,
            "segment_point_to_point": 1.65,
            "pupil_diameter": 7.225765,
            "num_rings": 0,
            "focal_length": 144.0,
            "supersampling_factor": 1,
        },
        "hres_psf": {
            "num_pix": 64,
            "wavelength": 5.0e-7,
            "num_airy": 8,
            "sampling": 4,
        },
    }


@pytest.fixture(scope="module")
def telescope_data(minimal_psf_config):
    if not HCIPY_AVAILABLE:
        pytest.skip("hcipy is not installed")
    return telescope_models.create_hcipy_telescope(minimal_psf_config)


def test_segment_hexike_api_matches_manual_for_noll_indexing(telescope_data):
    wavelength = telescope_data["wavelength"]
    segments = telescope_data["segments"]
    segment_hexikes_noll = {0: {1: 50.0, 3: -20.0}}

    phase_manual = aberration_models.apply_segment_zernikes_manual(
        segment_hexikes_noll, segments, telescope_data, wavelength
    )
    phase_api, hexike_surface = aberration_models.apply_segment_zernikes_api(
        segment_hexikes_noll, telescope_data, wavelength
    )

    # API and manual implementation should agree for the same HCIPy-style Noll indices.
    assert np.allclose(np.asarray(phase_manual), np.asarray(phase_api), rtol=1e-8, atol=1e-10)
    # Returned phase screen should be exactly what the HCIPy optic reports.
    assert np.allclose(np.asarray(phase_api), np.asarray(hexike_surface.phase_for(wavelength)))


def test_segment_hexike_api_rejects_zero_based_mode_indices(telescope_data):
    wavelength = telescope_data["wavelength"]

    with pytest.raises(ValueError, match="1-based Noll"):
        aberration_models.apply_segment_zernikes_api(
            {0: {0: 40.0, 1: 10.0}}, telescope_data, wavelength
        )


def test_config_validation_rejects_zero_based_segment_hexike_modes():
    config_path = PROJECT_ROOT / "configs" / "master_config.yaml"
    with config_path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    bad_config = copy.deepcopy(config)
    bad_config["psf"]["aberrations"]["enable_segment_hexikes"] = True
    bad_config["psf"]["aberrations"]["segment_hexikes"] = {0: {0: 100.0}}

    with pytest.raises(ValueError, match="1-based Noll"):
        validation.validate_or_raise(bad_config)


def test_config_validation_accepts_one_based_segment_hexike_modes():
    config_path = PROJECT_ROOT / "configs" / "master_config.yaml"
    with config_path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    good_config = copy.deepcopy(config)
    good_config["psf"]["aberrations"]["enable_segment_hexikes"] = True
    good_config["psf"]["aberrations"]["segment_hexikes"] = {0: {1: 100.0}}

    validation.validate_or_raise(good_config)
