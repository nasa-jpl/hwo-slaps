"""Physics validation tests for PSF generation.

These tests check the PSF module against external optical expectations rather
than only against internal implementation contracts.
"""

from __future__ import annotations

import contextlib
import copy
import io
from pathlib import Path

import hcipy
import numpy as np
import pytest
import yaml

from hwoslaps.constants import ARCSEC_PER_RAD
from hwoslaps.psf.generator import generate_psf_system
from hwoslaps.psf.psf_metrics import measure_fwhm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
AIRY_FWHM_DIAMETER_LAMBDA_OVER_D = 1.028993969
AIRY_FIRST_NULL_RADIUS_LAMBDA_OVER_D = 1.219669891
AIRY_ENCIRCLED_ENERGY_FIRST_NULL = 0.837784869


@pytest.fixture()
def compact_no_aberration_config() -> dict:
    with (PROJECT_ROOT / "configs" / "master_config.yaml").open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    cfg = copy.deepcopy(cfg)
    cfg["plotting"]["enabled"] = False
    cfg["psf"]["hres_psf"]["num_pix"] = 128
    cfg["psf"]["hres_psf"]["num_airy"] = 6
    cfg["psf"]["hres_psf"]["sampling"] = 5
    cfg["psf"]["hres_psf"]["save_highres_psf_npy"] = False
    cfg["psf"]["kernel"]["shape_native"] = [15, 15]

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


def _quiet_generate(config: dict):
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        return generate_psf_system(config["psf"], full_config=config)


def _kernel_second_moment(kernel_native: np.ndarray) -> float:
    y, x = np.indices(kernel_native.shape)
    cy, cx = np.array(kernel_native.shape) // 2
    return float(np.sum(kernel_native * ((x - cx) ** 2 + (y - cy) ** 2)))


def test_circular_airy_matches_analytic_fwhm_null_and_encircled_energy():
    """A plain circular aperture should reproduce standard Airy optics."""
    pupil_diameter = 1.0
    wavelength = 500e-9
    focal_length = 10.0
    focal_sampling = 16
    num_airy = 40

    pupil_grid = hcipy.make_pupil_grid(512, pupil_diameter)
    aperture = hcipy.make_circular_aperture(pupil_diameter)(pupil_grid)
    focal_grid = hcipy.make_focal_grid(
        q=focal_sampling,
        num_airy=num_airy,
        pupil_diameter=pupil_diameter,
        focal_length=focal_length,
        reference_wavelength=wavelength,
    )
    propagator = hcipy.FraunhoferPropagator(pupil_grid, focal_grid, focal_length)
    psf = propagator(hcipy.Wavefront(aperture, wavelength))

    pixel_scale_arcsec = wavelength / pupil_diameter * ARCSEC_PER_RAD / focal_sampling
    fwhm_arcsec = float(measure_fwhm(psf, pixel_scale_arcsec))
    expected_fwhm_arcsec = (
        AIRY_FWHM_DIAMETER_LAMBDA_OVER_D
        * wavelength
        / pupil_diameter
        * ARCSEC_PER_RAD
    )
    assert fwhm_arcsec == pytest.approx(expected_fwhm_arcsec, rel=2e-3)

    intensity = np.asarray(psf.power)
    rho = (
        np.sqrt(np.asarray(focal_grid.x) ** 2 + np.asarray(focal_grid.y) ** 2)
        / focal_length
        / (wavelength / pupil_diameter)
    )

    first_null_window = (rho > 0.9) & (rho < 1.5)
    first_null_radius = float(rho[first_null_window][np.argmin(intensity[first_null_window])])
    assert first_null_radius == pytest.approx(AIRY_FIRST_NULL_RADIUS_LAMBDA_OVER_D, rel=2e-2)

    encircled_energy = float(
        np.sum(intensity[rho <= AIRY_FIRST_NULL_RADIUS_LAMBDA_OVER_D])
        / np.sum(intensity)
    )
    assert encircled_energy == pytest.approx(AIRY_ENCIRCLED_ENERGY_FIRST_NULL, abs=5e-3)


@pytest.mark.parametrize("aberration_nm", [5.0, 20.0])
def test_global_aberration_strehl_matches_marechal_limit(
    compact_no_aberration_config: dict,
    aberration_nm: float,
):
    """Small OPD aberrations follow the Marechal Strehl approximation."""
    cfg = copy.deepcopy(compact_no_aberration_config)
    aberr = cfg["psf"]["aberrations"]
    aberr["enable_global_zernikes"] = True
    aberr["global_zernikes"] = {4: aberration_nm}

    psf_data = _quiet_generate(cfg)
    sigma_m = psf_data.total_rms_nm * 1e-9
    wavelength = cfg["psf"]["hres_psf"]["wavelength"]
    marechal_strehl = np.exp(-((2 * np.pi * sigma_m / wavelength) ** 2))

    assert psf_data.total_rms_nm > 0
    assert psf_data.strehl_ratio == pytest.approx(marechal_strehl, rel=1e-3, abs=5e-4)


def test_segmented_no_aberration_psf_converges_with_pupil_sampling(
    compact_no_aberration_config: dict,
):
    """Unaberrated PSF metrics stay stable as pupil sampling increases."""
    results = []
    for num_pix in [96, 128, 192]:
        cfg = copy.deepcopy(compact_no_aberration_config)
        cfg["psf"]["hres_psf"]["num_pix"] = num_pix
        psf_data = _quiet_generate(cfg)

        kernel = np.asarray(psf_data.kernel.native)
        center = tuple(np.array(kernel.shape) // 2)
        results.append(
            {
                "fwhm": float(psf_data.fwhm_arcsec),
                "strehl": float(psf_data.strehl_ratio),
                "kernel_peak": float(kernel[center]),
                "kernel_second_moment": _kernel_second_moment(kernel),
                "total_rms": float(psf_data.total_rms_nm),
            }
        )

    reference = results[-1]
    for result in results:
        assert result["strehl"] == pytest.approx(1.0, rel=1e-12, abs=1e-12)
        assert result["total_rms"] == pytest.approx(0.0, rel=0.0, abs=1e-12)
        assert result["fwhm"] == pytest.approx(reference["fwhm"], rel=2e-3)
        assert result["kernel_peak"] == pytest.approx(reference["kernel_peak"], rel=5e-3)
        assert result["kernel_second_moment"] == pytest.approx(
            reference["kernel_second_moment"],
            rel=1e-2,
        )
