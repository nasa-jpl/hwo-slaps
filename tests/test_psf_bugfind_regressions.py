"""Regression tests for PSF physics/math bugfind review.

These tests intentionally encode correctness expectations that are stricter
than the existing smoke tests. They should fail until the PSF implementation
is fixed; they do not modify production code.
"""

from __future__ import annotations

import copy
import io
import contextlib
from pathlib import Path

import hcipy
import numpy as np
import pytest
import yaml

from hwoslaps.config.validation import validate_or_raise
from hwoslaps.psf.aberration_models import (
    apply_global_zernikes,
    apply_segment_pistons,
    apply_segment_tiptilts,
    apply_segment_zernikes,
    generate_random_segment_aberrations,
)
from hwoslaps.psf.generator import generate_psf_system
from hwoslaps.psf.telescope_models import create_hcipy_telescope


PROJECT_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture()
def master_config() -> dict:
    with (PROJECT_ROOT / "configs" / "master_config.yaml").open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _compact_full_config(config: dict) -> dict:
    """Return a small full config that still exercises the real PSF path."""
    cfg = copy.deepcopy(config)
    cfg["plotting"]["enabled"] = False
    cfg["psf"]["hres_psf"]["num_pix"] = 64
    cfg["psf"]["hres_psf"]["num_airy"] = 4
    cfg["psf"]["hres_psf"]["save_highres_psf_npy"] = False
    cfg["psf"]["kernel"]["shape_native"] = [9, 9]

    aberr = cfg["psf"]["aberrations"]
    aberr["enable_segment_pistons"] = False
    aberr["enable_segment_tiptilts"] = False
    aberr["enable_segment_hexikes"] = False
    aberr["enable_global_zernikes"] = False
    aberr["segment_pistons"] = {}
    aberr["segment_tiptilts"] = {}
    aberr["segment_hexikes"] = {}
    aberr["global_zernikes"] = {}
    return cfg


@pytest.fixture()
def compact_config(master_config: dict) -> dict:
    return _compact_full_config(master_config)


@pytest.fixture()
def compact_telescope(compact_config: dict) -> dict:
    return create_hcipy_telescope(compact_config["psf"])


def _quiet_generate(config: dict):
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        return generate_psf_system(config["psf"], full_config=config)


def test_global_zernike_config_keys_are_one_based_noll_indices(compact_telescope: dict):
    """Config key 4 should apply HCIPy's Noll Z4, not zero-based basis index 4."""
    wavelength = compact_telescope["wavelength"]
    pupil_grid = compact_telescope["pupil_grid"]
    phase_screen = np.asarray(apply_global_zernikes({4: 1.0}, compact_telescope, wavelength))

    zernike_basis = hcipy.make_zernike_basis(
        8,
        D=pupil_grid.x.max() - pupil_grid.x.min(),
        grid=pupil_grid,
    )
    expected_noll_4 = np.asarray(zernike_basis[3]) * (2 * np.pi * 1e-9 / wavelength)

    assert np.allclose(phase_screen, expected_noll_4, rtol=1e-12, atol=1e-14)


def test_segment_hexike_uses_hcipy_surface_with_telescope_segments(compact_telescope: dict):
    """Segment hexikes should be represented by HCIPy's surface on the telescope masks."""
    wavelength = compact_telescope["wavelength"]
    segment_hexikes = {
        0: {1: 50.0, 3: -20.0},
        7: {2: 15.0},
        18: {4: -10.0},
    }

    phase_screen, hexike_surface = apply_segment_zernikes(segment_hexikes, compact_telescope, wavelength)

    assert isinstance(hexike_surface, hcipy.SegmentedHexikeSurface)
    assert np.allclose(np.asarray(phase_screen), np.asarray(hexike_surface.phase_for(wavelength)))
    assert hexike_surface.input_grid is compact_telescope["pupil_grid"]
    assert hexike_surface.coefficients.shape[0] == len(compact_telescope["segments"])
    assert np.allclose(hexike_surface.coefficients[0, 0], 25e-9)
    assert np.allclose(hexike_surface.coefficients[0, 2], -10e-9)


@pytest.mark.parametrize(
    "bad_segment",
    [-1, 19],
)
def test_segment_pistons_reject_invalid_segment_ids(compact_telescope: dict, bad_segment: int):
    """Segment pistons should fail fast instead of silently ignoring or wrapping IDs."""
    with pytest.raises(ValueError, match="segment"):
        apply_segment_pistons(
            compact_telescope["hsm"],
            {bad_segment: 10.0},
            compact_telescope["wavelength"],
            len(compact_telescope["segments"]),
        )


@pytest.mark.parametrize(
    "bad_segment",
    [-1, 19],
)
def test_segment_tiptilts_reject_invalid_segment_ids(compact_telescope: dict, bad_segment: int):
    """Segment tip/tilts should fail fast instead of silently ignoring or wrapping IDs."""
    with pytest.raises(ValueError, match="segment"):
        apply_segment_tiptilts(
            compact_telescope["hsm"],
            {bad_segment: (1.0, -2.0)},
            len(compact_telescope["segments"]),
        )


@pytest.mark.parametrize(
    ("flag", "coeff_key"),
    [
        ("enable_segment_pistons", "segment_pistons"),
        ("enable_segment_tiptilts", "segment_tiptilts"),
        ("enable_segment_hexikes", "segment_hexikes"),
        ("enable_global_zernikes", "global_zernikes"),
    ],
)
def test_enabled_aberration_families_must_have_coefficients(
    master_config: dict,
    flag: str,
    coeff_key: str,
):
    """Strict configs should not allow enabled aberration families with empty coefficient dicts."""
    cfg = _compact_full_config(master_config)
    cfg["psf"]["aberrations"][flag] = True
    cfg["psf"]["aberrations"][coeff_key] = {}

    with pytest.raises(ValueError, match=coeff_key):
        validate_or_raise(cfg)


def test_generate_psf_system_does_not_mutate_input_config(compact_config: dict):
    """PSF generation should not rewrite caller-owned config/provenance in place."""
    sampling_before = compact_config["psf"]["hres_psf"]["sampling"]

    _quiet_generate(compact_config)

    assert compact_config["psf"]["hres_psf"]["sampling"] == sampling_before


def test_perfect_psf_reports_unity_strehl(compact_config: dict):
    """A generated perfect PSF should carry a Strehl ratio of one in PSFData."""
    psf_data = _quiet_generate(compact_config)

    assert psf_data.strehl_ratio == pytest.approx(1.0, rel=1e-12, abs=1e-12)
    assert psf_data.is_diffraction_limited is True


def test_generate_random_segment_aberrations_rejects_degenerate_segment_count():
    """Random segment aberrations should not return NaN pistons for one segment."""
    with pytest.raises(ValueError, match="num_segments"):
        generate_random_segment_aberrations(10.0, num_segments=1, seed=1)


def test_psfdata_wavefront_is_not_the_same_object_as_focal_plane_psf(compact_config: dict):
    """PSFData.wavefront should describe the pupil-plane wavefront, not duplicate PSFData.psf."""
    psf_data = _quiet_generate(compact_config)

    assert psf_data.wavefront is not psf_data.psf
    assert psf_data.wavefront.electric_field.grid is psf_data.telescope_data["pupil_grid"]
