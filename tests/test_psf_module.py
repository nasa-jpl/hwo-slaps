"""PyTest suite for hwoslaps.psf module.

This suite validates scientific and mathematical invariants of the PSF pipeline
using configuration from `hwo-slaps/configs/master_config.yaml`.

Each test documents its purpose, the config inputs used, and asserts
numerical correctness of key quantities.
"""

import copy
import math
import os
from types import SimpleNamespace
from typing import Any, Dict, Tuple

import numpy as np
import pytest
import yaml

from hwoslaps.psf import (
    generate_psf_system,
)
from hwoslaps.psf.generator import _total_aperture_rms_nm
from hwoslaps.psf.psf_metrics import calculate_strehl_ratio

# ---- Fixtures ----


@pytest.fixture(scope="session")
def config_path() -> str:
    """Absolute path to master_config.yaml used by all tests."""
    here = os.path.abspath(os.path.dirname(__file__))
    return os.path.abspath(os.path.join(here, "..", "configs", "master_config.yaml"))


@pytest.fixture(scope="session")
def master_config(config_path: str) -> Dict[str, Any]:
    """Load and parse the master configuration used for PSF generation.

    Source: `hwo-slaps/configs/master_config.yaml`.
    """
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
    assert isinstance(cfg, dict)
    return cfg


@pytest.fixture(scope="session")
def psf_config(master_config: Dict[str, Any]) -> Dict[str, Any]:
    """Return the `psf` section of the configuration (deep copy)."""
    cfg = copy.deepcopy(master_config)
    assert "psf" in cfg
    return cfg["psf"]


@pytest.fixture(scope="session")
def psf_data(master_config: Dict[str, Any], psf_config: Dict[str, Any]):
    """Generate a PSF system using the configuration from master_config.yaml.

    Source of inputs: all taken from `master_config` (PSF, lensing, and
    telescope).
    """
    return generate_psf_system(psf_config, full_config=master_config)


@pytest.fixture(scope="session")
def perfect_psf_data(master_config: Dict[str, Any]) -> Any:
    """Generate a near-perfect PSF by disabling all aberrations.

    Source: same `master_config.yaml` with only
    `psf.aberrations.enable_* = False`.
    """
    cfg_perfect = copy.deepcopy(master_config)
    ab = cfg_perfect["psf"]["aberrations"]
    ab["enable_segment_pistons"] = False
    ab["enable_segment_tiptilts"] = False
    ab["enable_segment_hexikes"] = False
    ab["enable_global_zernikes"] = False
    return generate_psf_system(cfg_perfect["psf"], full_config=cfg_perfect)


# ---- Helper utilities for expectations ----

def _expected_integer_and_used_sampling(full_cfg: Dict[str, Any]) -> Tuple[int, float]:
    """Compute expected integer subsampling factor N and used sampling.

    Mirrors the math in `hwoslaps.psf.generator.generate_psf_system`.

    Returns
    -------
    (N, used_sampling)
    """
    lensing = full_cfg["lensing"]
    psf = full_cfg["psf"]
    sim = psf["hres_psf"]
    tel = psf["telescope"]

    target_pixel_scale = float(lensing["grid"]["pixel_scale"])  # arcsec/pix
    wavelength = float(sim["wavelength"])  # meters
    pupil_diameter = float(tel["pupil_diameter"])  # meters
    requested_sampling = float(sim["sampling"])  # px per lambda/D

    res_element_arcsec = (wavelength / pupil_diameter) * 206264.8062471
    hres_pixel_scale_initial = res_element_arcsec / requested_sampling
    non_integer_factor = target_pixel_scale / hres_pixel_scale_initial
    N = int(round(non_integer_factor))
    used_sampling = (N * res_element_arcsec) / target_pixel_scale
    return N, used_sampling


def _odd_enforced(shape_native):
    shape = list(shape_native)
    for i in range(len(shape)):
        if shape[i] % 2 == 0:
            shape[i] += 1
    return shape


# ---- Tests ----

def test_config_schema_and_types(master_config: Dict[str, Any]):
    """Validate config structure and types for PSF generation.

    Source: `hwo-slaps/configs/master_config.yaml`.
    """
    # Top-level keys
    assert "psf" in master_config and "lensing" in master_config

    # Lensing grid
    assert "grid" in master_config["lensing"]
    assert isinstance(master_config["lensing"]["grid"]["pixel_scale"], (int, float))

    psf = master_config["psf"]
    # Telescope
    tel = psf["telescope"]
    for key in [
        "gap_size",
        "segment_point_to_point",
        "pupil_diameter",
        "num_rings",
        "focal_length",
    ]:
        assert key in tel
    assert isinstance(tel["pupil_diameter"], (int, float)) and tel["pupil_diameter"] > 0

    # High-res PSF block
    hres = psf["hres_psf"]
    for key in ["num_pix", "wavelength", "num_airy", "sampling"]:
        assert key in hres
    assert isinstance(hres["num_pix"], int) and hres["num_pix"] > 0
    assert isinstance(hres["wavelength"], (int, float)) and hres["wavelength"] > 0
    assert isinstance(hres["num_airy"], (int, float)) and hres["num_airy"] > 0
    assert isinstance(hres["sampling"], (int, float)) and hres["sampling"] > 0

    # Kernel parameters
    kernel = psf["kernel"]
    assert "shape_native" in kernel
    assert isinstance(kernel["shape_native"], list) and len(kernel["shape_native"]) == 2
    assert all(isinstance(v, int) and v > 0 for v in kernel["shape_native"])  # type: ignore

    # Aberrations section exists
    assert "aberrations" in psf and isinstance(psf["aberrations"], dict)


def test_generate_psf_core_invariants(master_config: Dict[str, Any], psf_data):
    """Core invariants: kernel shape, kernel pixel scale, and sampling math.

    Inputs: full config from `master_config.yaml`.
    """
    psf_cfg = master_config["psf"]
    expected_shape = _odd_enforced(list(psf_cfg["kernel"]["shape_native"]))
    assert list(psf_data.kernel.shape_native) == expected_shape

    # Kernel pixel scale must match lensing.grid.pixel_scale exactly
    cfg_pixel_scale = float(master_config["lensing"]["grid"]["pixel_scale"])
    assert math.isclose(float(psf_data.kernel_pixel_scale), cfg_pixel_scale, rel_tol=1e-12, abs_tol=0.0)

    # Integer subsampling and used sampling must match theory
    N_exp, used_sampling_exp = _expected_integer_and_used_sampling(master_config)
    assert psf_data.integer_subsampling_factor == int(N_exp)
    assert math.isclose(float(psf_data.used_sampling_factor), float(used_sampling_exp), rel_tol=1e-12)


def test_highres_psf_dimensions_from_sampling(master_config: Dict[str, Any], psf_data):
    """High-res PSF dimensions equal floor(2 * num_airy * used_sampling).

    Inputs: `num_airy` and auto-adjusted sampling from `master_config.yaml`.
    """
    num_airy = float(master_config["psf"]["hres_psf"]["num_airy"])
    q_used = float(psf_data.used_sampling_factor)
    dim_expected = int(2 * num_airy * q_used)
    shape = psf_data.psf.intensity.shaped.shape
    assert shape[0] == dim_expected and shape[1] == dim_expected


def test_kernel_energy_conservation_and_centering(psf_data):
    """Downsampled kernel must conserve energy and peak near the array center.

    Inputs: kernel generated from config-defined detector pixel scale and size.
    """
    kernel = psf_data.kernel.native
    # Energy conservation
    assert np.isclose(kernel.sum(), 1.0, rtol=1e-12, atol=1e-12)

    # Peak at center (odd-sized kernel ⇒ unique center index)
    ny, nx = kernel.shape
    cy, cx = ny // 2, nx // 2
    py, px = np.unravel_index(np.argmax(kernel), kernel.shape)
    assert abs(py - cy) <= 1 and abs(px - cx) <= 1


def test_aberration_flags_match_config(master_config: Dict[str, Any], psf_data):
    """Flags in PSFData reflect enable_* toggles and coefficient presence.

    Inputs: `psf.aberrations.*` from `master_config.yaml`.
    """
    ab = master_config["psf"]["aberrations"]

    assert psf_data.has_segment_pistons == bool(
        ab.get("enable_segment_pistons", True) and ab.get("segment_pistons"))
    assert psf_data.has_segment_tiptilts == bool(
        ab.get("enable_segment_tiptilts", True) and ab.get("segment_tiptilts"))
    assert psf_data.has_segment_hexikes == bool(
        ab.get("enable_segment_hexikes", True) and ab.get("segment_hexikes"))
    assert psf_data.has_global_zernikes == bool(
        ab.get("enable_global_zernikes", True) and ab.get("global_zernikes"))

    # If enabled, the corresponding RMS summaries should be non-trivial
    if psf_data.has_segment_pistons:
        assert psf_data.segment_piston_rms_nm > 0
    if psf_data.has_segment_tiptilts:
        assert psf_data.segment_tiptilt_rms_urad >= 0
    if psf_data.has_global_zernikes:
        assert psf_data.global_zernike_rms_nm >= 0


def test_fwhm_physical_scale_sanity(psf_data):
    """FWHM must be positive and within a plausible multiple of λ/D.

    Inputs: generated PSF and telescope parameters embedded in PSFData.
    """
    assert psf_data.fwhm_arcsec is not None and psf_data.fwhm_arcsec > 0

    # Sanity window: allow generous bounds to remain robust across configs.
    dl = psf_data.diffraction_limit_arcsec
    ratio = float(psf_data.fwhm_arcsec) / float(dl)
    assert 0.3 <= ratio <= 5.0


def test_strehl_ratio_degrades_with_aberrations(psf_data, perfect_psf_data):
    """Strehl ratio against a perfect PSF must lie strictly below unity.

    Inputs: aberrated PSF from `master_config.yaml`; perfect PSF from the same
    config with all `enable_*` toggles set to False.
    """
    strehl = float(calculate_strehl_ratio(psf_data.psf, perfect_psf_data.psf))
    assert 0.0 < strehl <= 1.0
    # With any non-zero aberration, ratio should be strictly below 1
    assert strehl < 1.0


def test_even_kernel_shape_is_enforced_to_odd(master_config: Dict[str, Any]):
    """If an even kernel shape is requested, implementation must bump to odd.

    Inputs: cloned `master_config.yaml` with an even kernel.shape_native.
    """
    cfg = copy.deepcopy(master_config)
    # Make both dimensions even
    cfg["psf"]["kernel"]["shape_native"] = [
        int(cfg["psf"]["kernel"]["shape_native"][0]) + 1,
        int(cfg["psf"]["kernel"]["shape_native"][1]) + 1,
    ]
    # If they were already even, adding 1 makes them odd; force even here
    cfg["psf"]["kernel"]["shape_native"] = [
        v if v % 2 == 0 else v + 1 for v in cfg["psf"]["kernel"]["shape_native"]
    ]

    data = generate_psf_system(cfg["psf"], full_config=cfg)
    sy, sx = data.kernel.shape_native
    assert sy % 2 == 1 and sx % 2 == 1


def test_single_global_zernike_rms_approx_matches_input(master_config: Dict[str, Any]):
    """A single 10 nm RMS global Zernike yields ~10 nm wavefront RMS.

    Source: `master_config.yaml` cloned with all segment terms disabled and
    `global_zernikes = {4: 10}` (focus). We assert that:
    - `psf_data.global_zernike_rms_nm ≈ 10 nm` (exact by construction), and
    - `psf_data.total_rms_nm` measured over the hex pupil is approximately
      10 nm within a generous tolerance to allow for pupil-geometry and
      discretization effects.
    """
    cfg = copy.deepcopy(master_config)
    ab = cfg["psf"]["aberrations"]
    ab["enable_segment_pistons"] = False
    ab["enable_segment_tiptilts"] = False
    ab["enable_segment_hexikes"] = False
    ab["enable_global_zernikes"] = True
    ab["global_zernikes"] = {4: 10.0}

    data = generate_psf_system(cfg["psf"], full_config=cfg)

    # By construction, this summary equals the RMS of provided nm coefficients.
    assert data.has_global_zernikes is True
    assert data.global_zernike_rms_nm == pytest.approx(10.0, rel=1e-12, abs=0.0)

    # Measured total RMS across the hex pupil should be close to 10 nm.
    # Allow moderate tolerance for geometry/normalization and discretization.
    assert data.total_rms_nm == pytest.approx(10.0, rel=0.3)


def test_total_aperture_rms_helper_exact_hand_value():
    """Helper reproduces an exact hand-derived piston-removed OPD RMS.

    Inputs: synthetic 5-pixel telescope data with one dark pixel. The
    surface [5, 5, 15, 15, 500] nm doubles on reflection to
    [10, 10, 30, 30, 1000] nm OPD, and the phase screen
    [0, 0, 0.1, 0.1, 0] rad converts at λ/(2π) = 100 nm/rad to add
    [0, 0, 10, 10, 0] nm. Over the four illuminated pixels the OPD is
    [10, 10, 40, 40] nm with mean 25 nm, so deviations are
    [-15, -15, 15, 15] nm and the RMS is exactly 15 nm.
    """
    telescope_data = {
        "hsm": SimpleNamespace(
            surface=np.array([5.0e-9, 5.0e-9, 15.0e-9, 15.0e-9, 5.0e-7])
        ),
        "wavelength": 2.0 * np.pi * 1.0e-7,
        "aper": np.array([1.0, 1.0, 1.0, 1.0, 0.0]),
    }
    phase_screens = {"global_zernikes": np.array([0.0, 0.0, 0.1, 0.1, 0.0])}

    rms_nm = _total_aperture_rms_nm(telescope_data, phase_screens)
    assert rms_nm == pytest.approx(15.0, rel=1e-12)


def test_total_aperture_rms_helper_raises_on_malformed_screen():
    """A phase screen whose shape mismatches the pupil must raise loudly.

    Inputs: 4-pixel surface and aperture with a 3-pixel phase screen. The
    OPD addition genuinely fails to broadcast, and the helper must raise
    RuntimeError chained from the original error with a message naming
    the operation. No 0.0 fallback is permitted.
    """
    telescope_data = {
        "hsm": SimpleNamespace(surface=np.zeros(4)),
        "wavelength": 5.0e-7,
        "aper": np.ones(4),
    }
    phase_screens = {"global_zernikes": np.zeros(3)}

    with pytest.raises(
        RuntimeError, match="Total aperture OPD RMS calculation failed"
    ) as excinfo:
        _total_aperture_rms_nm(telescope_data, phase_screens)
    assert isinstance(excinfo.value.__cause__, ValueError)
