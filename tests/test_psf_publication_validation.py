"""Publication-grade validation tests for the PSF module.

These tests are intentionally independent of the main study pipeline. They
exercise the optical scale, branch consistency, PyAutoLens convolution
centering, and parameter robustness expected before production sweeps.
"""

from __future__ import annotations

import contextlib
import copy
import io
from pathlib import Path

import autolens as al
import numpy as np
import pytest
import yaml
from scipy.special import j1

from hwoslaps.constants import ARCSEC_PER_RAD
from hwoslaps.psf.generator import generate_psf_system
from hwoslaps.psf.utils import make_pyauto_convolver

PROJECT_ROOT = Path(__file__).resolve().parents[1]
AIRY_FWHM_DIAMETER_LAMBDA_OVER_D = 1.028993969
AIRY_FIRST_NULL_RADIUS_LAMBDA_OVER_D = 1.219669891
AIRY_ENCIRCLED_ENERGY_FIRST_NULL = 0.837784869


@pytest.fixture()
def compact_no_aberration_config() -> dict:
    """Load master_config.yaml shrunk to a fast, unaberrated PSF."""
    with (PROJECT_ROOT / "configs" / "master_config.yaml").open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    cfg = copy.deepcopy(cfg)
    cfg["plotting"]["enabled"] = False
    cfg["psf"]["hres_psf"]["num_pix"] = 96
    cfg["psf"]["hres_psf"]["num_airy"] = 5
    cfg["psf"]["hres_psf"]["sampling"] = 5
    cfg["psf"]["hres_psf"]["save_highres_psf_npy"] = False
    cfg["psf"]["kernel"]["shape_native"] = [9, 9]
    _disable_aberrations(cfg)
    return cfg


def _disable_aberrations(config: dict) -> None:
    aberr = config["psf"]["aberrations"]
    aberr["enable_segment_pistons"] = False
    aberr["enable_segment_tiptilts"] = False
    aberr["enable_segment_hexikes"] = False
    aberr["enable_global_zernikes"] = False
    aberr["segment_pistons"] = {}
    aberr["segment_tiptilts"] = {}
    aberr["segment_hexikes"] = {}
    aberr["global_zernikes"] = {}


def _quiet_generate(config: dict):
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        return generate_psf_system(config["psf"], full_config=config)


def _center_crop(array_2d: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    y0 = (array_2d.shape[0] - shape[0]) // 2
    x0 = (array_2d.shape[1] - shape[1]) // 2
    return array_2d[y0:y0 + shape[0], x0:x0 + shape[1]]


def _binned_highres_kernel(psf_data) -> np.ndarray:
    subsampling = int(psf_data.integer_subsampling_factor)
    kernel = np.asarray(psf_data.kernel.native)
    highres_power = np.asarray(psf_data.psf.power.shaped)
    crop_shape = (
        kernel.shape[0] * subsampling,
        kernel.shape[1] * subsampling,
    )

    highres_crop = _center_crop(highres_power, crop_shape)
    binned = highres_crop.reshape(
        kernel.shape[0],
        subsampling,
        kernel.shape[1],
        subsampling,
    ).sum(axis=(1, 3))
    return binned / np.sum(binned)


def _expected_integer_and_sampling(config: dict) -> tuple[int, float]:
    psf = config["psf"]
    wavelength = float(psf["hres_psf"]["wavelength"])
    pupil_diameter = float(psf["telescope"]["pupil_diameter"])
    requested_sampling = float(psf["hres_psf"]["sampling"])
    target_pixel_scale = float(config["lensing"]["grid"]["pixel_scale"])

    resolution_element_arcsec = wavelength / pupil_diameter * ARCSEC_PER_RAD
    initial_hres_pixel_scale = resolution_element_arcsec / requested_sampling
    integer_subsampling = int(round(target_pixel_scale / initial_hres_pixel_scale))
    used_sampling = integer_subsampling * resolution_element_arcsec / target_pixel_scale
    return integer_subsampling, used_sampling


def _fft_circular_aperture_psf(
    *,
    n_pixels: int = 1024,
    computational_width_diameters: float = 16.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return a pure-NumPy Fraunhofer PSF for a circular aperture.

    The image-plane coordinate is rho = theta / (lambda / D), so analytic
    Airy constants can be tested without invoking HCIPy or project PSF code.
    """
    diameter = 1.0
    dx = computational_width_diameters / n_pixels
    axis = (np.arange(n_pixels) - n_pixels // 2) * dx
    yy, xx = np.meshgrid(axis, axis, indexing="ij")
    pupil = ((xx**2 + yy**2) <= (diameter / 2) ** 2).astype(float)

    psf = np.abs(np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(pupil)))) ** 2
    psf /= np.max(psf)

    frequency_axis = np.fft.fftshift(np.fft.fftfreq(n_pixels, d=dx))
    fy, fx = np.meshgrid(frequency_axis, frequency_axis, indexing="ij")
    rho = np.sqrt(fx**2 + fy**2) * diameter
    return psf, rho, frequency_axis * diameter


def test_numpy_fft_circular_aperture_matches_analytic_airy_metrics():
    """A non-HCIPy FFT oracle reproduces Airy FWHM, first null, and energy."""
    psf, rho, rho_axis = _fft_circular_aperture_psf()
    center = psf.shape[0] // 2
    rho_positive = rho_axis[center:]
    central_cut = psf[center, center:]

    half_max_idx = int(np.flatnonzero(central_cut <= 0.5)[0])
    r1, r2 = rho_positive[half_max_idx - 1], rho_positive[half_max_idx]
    i1, i2 = central_cut[half_max_idx - 1], central_cut[half_max_idx]
    half_max_radius = r1 + (0.5 - i1) * (r2 - r1) / (i2 - i1)
    fwhm_diameter = 2 * half_max_radius

    null_window = (rho_positive > 0.9) & (rho_positive < 1.5)
    first_null_radius = float(rho_positive[null_window][np.argmin(central_cut[null_window])])
    encircled_energy = float(
        np.sum(psf[rho <= AIRY_FIRST_NULL_RADIUS_LAMBDA_OVER_D])
        / np.sum(psf)
    )

    sample_radii = np.array([0.25, 0.5, 0.75, 1.0])
    fft_profile = np.interp(sample_radii, rho_positive, central_cut)
    analytic_profile = (2 * j1(np.pi * sample_radii) / (np.pi * sample_radii)) ** 2

    assert fwhm_diameter == pytest.approx(AIRY_FWHM_DIAMETER_LAMBDA_OVER_D, rel=2e-3)
    assert first_null_radius == pytest.approx(AIRY_FIRST_NULL_RADIUS_LAMBDA_OVER_D, rel=3e-2)
    assert encircled_energy == pytest.approx(AIRY_ENCIRCLED_ENERGY_FIRST_NULL, abs=2e-3)
    np.testing.assert_allclose(fft_profile, analytic_profile, rtol=5e-3, atol=1e-4)


@pytest.mark.parametrize(
    "aberration_update",
    [
        pytest.param({}, id="no_aberration"),
        pytest.param(
            {
                "enable_global_zernikes": True,
                "global_zernikes": {4: 10.0, 5: -5.0},
            },
            id="global_zernikes",
        ),
        pytest.param(
            {
                "enable_segment_hexikes": True,
                "segment_hexikes": {0: {2: 5.0, 3: -2.5}, 7: {4: 3.0}},
            },
            id="segment_hexikes",
        ),
    ],
)
def test_detector_kernel_matches_binned_highres_branch(
    compact_no_aberration_config: dict,
    aberration_update: dict,
):
    """Detector kernels equal flux-conserving binning of high-res PSFs."""
    cfg = copy.deepcopy(compact_no_aberration_config)
    cfg["psf"]["aberrations"].update(aberration_update)

    psf_data = _quiet_generate(cfg)
    kernel = np.asarray(psf_data.kernel.native)
    binned_highres = _binned_highres_kernel(psf_data)

    assert psf_data.integer_subsampling_factor >= 1
    assert psf_data.pixel_scale_arcsec * psf_data.integer_subsampling_factor == pytest.approx(
        psf_data.kernel_pixel_scale,
        rel=1e-12,
        abs=1e-15,
    )
    assert np.sum(kernel) == pytest.approx(1.0, rel=1e-12, abs=1e-15)
    assert np.allclose(kernel, binned_highres, rtol=1e-12, atol=1e-14)


def test_generated_kernel_is_centered_under_pyautolens_delta_convolution(
    compact_no_aberration_config: dict,
):
    """Check that PyAuto convolution centers the kernel footprint."""
    psf_data = _quiet_generate(compact_no_aberration_config)
    kernel = np.asarray(psf_data.kernel.native)
    image_shape = (kernel.shape[0] + 16, kernel.shape[1] + 16)
    image_center = np.array(image_shape) // 2
    delta_image = np.zeros(image_shape)
    delta_image[tuple(image_center)] = 1.0

    mask = al.Mask2D.all_false(
        shape_native=image_shape,
        pixel_scales=psf_data.kernel_pixel_scale,
    )
    array = al.Array2D(values=delta_image, mask=mask)
    convolver = make_pyauto_convolver(psf_data.kernel)
    convolved = np.asarray(
        convolver.convolved_image_via_real_space_from(
            image=array,
            blurring_image=None,
        ).native
    )

    y0 = image_center[0] - kernel.shape[0] // 2
    x0 = image_center[1] - kernel.shape[1] // 2
    response_footprint = convolved[
        y0:y0 + kernel.shape[0],
        x0:x0 + kernel.shape[1],
    ]
    expected = np.zeros_like(convolved)
    expected[
        y0:y0 + kernel.shape[0],
        x0:x0 + kernel.shape[1],
    ] = kernel

    assert np.sum(convolved) == pytest.approx(1.0, rel=1e-12, abs=1e-15)
    assert tuple(np.unravel_index(np.argmax(convolved), convolved.shape)) == tuple(image_center)
    np.testing.assert_allclose(response_footprint, kernel, rtol=1e-12, atol=1e-14)
    np.testing.assert_allclose(convolved, expected, rtol=1e-12, atol=1e-14)


@pytest.mark.parametrize(
    ("wavelength_m", "pixel_scale_arcsec", "kernel_shape"),
    [
        (450e-9, 0.0060, [7, 7]),
        (500e-9, 0.00716, [9, 9]),
        (700e-9, 0.0120, [11, 11]),
    ],
)
def test_generation_parameter_sweep_preserves_sampling_flux_and_diffraction_scale(
    compact_no_aberration_config: dict,
    wavelength_m: float,
    pixel_scale_arcsec: float,
    kernel_shape: list[int],
):
    """Representative wavelength/pixel-scale/kernel choices stay physical."""
    cfg = copy.deepcopy(compact_no_aberration_config)
    cfg["psf"]["hres_psf"]["wavelength"] = wavelength_m
    cfg["psf"]["hres_psf"]["num_airy"] = 5
    cfg["psf"]["kernel"]["shape_native"] = kernel_shape
    cfg["lensing"]["grid"]["pixel_scale"] = pixel_scale_arcsec

    psf_data = _quiet_generate(cfg)
    kernel = np.asarray(psf_data.kernel.native)
    expected_subsampling, expected_sampling = _expected_integer_and_sampling(cfg)
    fwhm_over_lambda_d = psf_data.fwhm_arcsec / psf_data.diffraction_limit_arcsec

    assert expected_subsampling >= 1
    assert psf_data.integer_subsampling_factor == expected_subsampling
    assert psf_data.used_sampling_factor == pytest.approx(expected_sampling, rel=1e-12)
    assert psf_data.pixel_scale_arcsec * expected_subsampling == pytest.approx(
        psf_data.kernel_pixel_scale,
        rel=1e-12,
        abs=1e-15,
    )
    assert psf_data.kernel_pixel_scale == pytest.approx(pixel_scale_arcsec, rel=0.0, abs=1e-15)
    assert np.sum(kernel) == pytest.approx(1.0, rel=1e-12, abs=1e-15)
    assert list(kernel.shape) == kernel_shape
    assert tuple(np.unravel_index(np.argmax(kernel), kernel.shape)) == (
        kernel.shape[0] // 2,
        kernel.shape[1] // 2,
    )
    assert 0.8 <= fwhm_over_lambda_d <= 1.4
