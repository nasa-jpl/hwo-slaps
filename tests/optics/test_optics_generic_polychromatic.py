"""Optical wavelength stacks against independent Airy integration and scaling laws."""

import math

import numpy as np
import pytest
from scipy.special import j1

from hwoslaps.constants import ARCSEC_PER_RAD
from hwoslaps.optics.metrics import captured_power_fraction, fwhm_arcsec
from hwoslaps.optics.optical_psf import FocalField
from hwoslaps.optics.providers import build_psf_provider, parse_psf
from hwoslaps.optics.wavefront import WavefrontCoefficients

pytestmark = pytest.mark.backend

# Carried from the validated mono EE owner, not loosened for this stack.
# XTX must justify this bound at both new pixel scales before acceptance.
_STACK_EE_ATOL_PENDING_XTX = 3e-3


def _optics(circular_pupil, *, pixel, nodes=(4e-7, 6e-7), coefficients=None):
    spec = parse_psf({"truth": {"kind": "optical", "pupil": circular_pupil, "focal_length_m": 10.0,
                               "wavelength_samples": len(nodes), "kernel_shape": [31, 31],
                               "detector_oversampling": 15, "wavefront": coefficients or {}}}).truth
    return build_psf_provider(spec, pixel_scale_arcsec=pixel, wavelengths_m=nodes)


def _kernel_field(values, pixel, wavelength):
    axis = (np.arange(values.shape[0]) - values.shape[0] // 2) * pixel
    x, y = np.meshgrid(axis, axis)
    return FocalField(values, x, y, pixel, wavelength)


def _integrated_airy(shape, pixel, wavelength, diameter=1.0):
    # Gaussian quadrature of [2 J1(pi D theta/lambda)/(pi D theta/lambda)]^2,
    # independent of the Fraunhofer propagator and its detector midpoint rule.
    abscissae, weights = np.polynomial.legendre.leggauss(20)
    coordinates = (np.arange(shape)[:, None] - shape // 2 + abscissae[None, :] / 2) * pixel / ARCSEC_PER_RAD
    y = coordinates[:, :, None, None]
    x = coordinates[None, None, :, :]
    v = math.pi * diameter * np.hypot(y, x) / wavelength
    amplitude = np.divide(2 * j1(v), v, out=np.ones_like(v), where=v != 0.0)
    image = np.einsum("iajb,a,b->ij", amplitude**2, weights, weights)
    return image / image.sum()


@pytest.mark.parametrize("fine", [False, True], ids=["nyquist-500nm", "fine-400nm"])
def test_wavelength_stack_shares_pixels_and_follows_integrated_airy(circular_pupil, fine):
    pixel = (4e-7 / 8 if fine else 5e-7 / 2) * ARCSEC_PER_RAD
    psf = _optics(circular_pupil, pixel=pixel)
    stack = psf.kernels()
    widths, reference_widths = [], []
    for wavelength, kernel in zip(psf.wavelengths_m, stack):
        assert kernel.pixel_scale_arcsec == pixel and kernel.shape == (31, 31)
        assert kernel.source["wavelength_m"] == wavelength
        assert kernel.source["captured_power_fraction"] == captured_power_fraction(psf, wavelength)
        direct = psf.kernel(wavelength)
        (one,) = psf.kernels([wavelength])
        np.testing.assert_array_equal(one.kernel, direct.kernel)
        assert dict(one.source) == dict(direct.source)
        widths.append(fwhm_arcsec(_kernel_field(kernel.kernel, pixel, wavelength)))
        reference = _integrated_airy(31, pixel, wavelength)
        reference_widths.append(fwhm_arcsec(_kernel_field(reference, pixel, wavelength)))
        # A circular zero-OPD image is invariant under D4 and one-ULP evaluation noise.
        for values, width in ((kernel.kernel, widths[-1]), (reference, reference_widths[-1])):
            for turns in range(4):
                for reflected in (False, True):
                    equivalent = np.rot90(values, turns)
                    if reflected:
                        equivalent = equivalent[:, ::-1]
                    equivalent = equivalent.copy()
                    equivalent[15, 16] = np.nextafter(equivalent[15, 16], np.inf)
                    actual = fwhm_arcsec(_kernel_field(equivalent, pixel, wavelength))
                    assert actual == pytest.approx(width, rel=64 * np.finfo(float).eps), "FWHM changes under symmetry and roundoff"
        axis = (np.arange(31) - 15) * pixel
        x, y = np.meshgrid(axis, axis)
        distance = np.hypot(y, x)
        radii = pixel * np.array([2.0, 4.0, 8.0, 12.0])
        observed_energy = np.array([np.sum(kernel.kernel[distance <= radius]) for radius in radii])
        reference_energy = np.array([np.sum(reference[distance <= radius]) for radius in radii])
        np.testing.assert_allclose(observed_energy, reference_energy, rtol=0.0, atol=_STACK_EE_ATOL_PENDING_XTX)
    ratio, expected_ratio = widths[1] / widths[0], reference_widths[1] / reference_widths[0]
    assert ratio == pytest.approx(expected_ratio, rel=1e-3)
    if fine:
        assert ratio == pytest.approx(1.5, rel=0.01)
    else:
        # Independent 20/40-point Airy integration with stable radial bins agrees to 1e-15.
        assert expected_ratio == pytest.approx(1.569258068303233, rel=1e-4)
    with pytest.raises(ValueError, match="name one"):
        psf.kernel()


def test_joint_wavelength_opd_and_pixel_scaling_preserves_the_kernel(circular_pupil):
    pixel = 4e-7 / 8 * ARCSEC_PER_RAD
    coefficients = {"zernikes": {4: 12.0, 7: 5.0}}
    first = _optics(circular_pupil, pixel=pixel, nodes=(4e-7,), coefficients=coefficients)
    second = _optics(circular_pupil, pixel=2 * pixel, nodes=(8e-7,),
                     coefficients={"zernikes": {4: 24.0, 7: 10.0}})
    np.testing.assert_allclose(first.kernel().kernel, second.kernel().kernel, rtol=0.0, atol=1e-14)


def test_coefficient_variant_retains_stack_and_each_node_identity(circular_pupil):
    psf = _optics(circular_pupil, pixel=0.03)
    coefficients = WavefrontCoefficients.from_mapping({"zernikes": {4: 8.0}}, "wavefront")
    variant = psf.with_coefficients(coefficients)
    assert variant.wavelengths_m == psf.wavelengths_m and variant.basis is psf.basis
    assert variant.pupil is psf.pupil
    assert variant.reference_wavelength_m == 4e-7
    for wavelength, kernel in zip(variant.wavelengths_m, variant.kernels()):
        assert kernel.source["wavelength_m"] == wavelength
        assert kernel.source["coefficients_digest"] == coefficients.digest()
        assert kernel.source["captured_power_fraction"] == captured_power_fraction(variant, wavelength)


def test_drawn_stack_exposes_actual_prior_bytes_without_moving_the_mono_limit(circular_pupil, tmp_path):
    from importlib import resources
    import hashlib

    prior = tmp_path / "prior.yaml"
    content = resources.files("hwoslaps.optics").joinpath("priors/jwst_wss_static_v1.yaml").read_bytes()
    prior.write_bytes(content)
    optical = {"kind": "optical", "pupil": circular_pupil, "focal_length_m": 10.0,
               "detector_oversampling": 3, "kernel_shape": [11, 11],
               "draw": {"prior": {"path": str(prior)}, "family": "global", "amplitude_rms_nm": 10.0, "seed": 11}}
    mono = build_psf_provider(parse_psf({"truth": {**optical, "wavelength_nm": 500.0}}).truth,
                              pixel_scale_arcsec=0.03)
    sampled = build_psf_provider(parse_psf({"truth": {**optical, "wavelength_samples": 1}}).truth,
                                 pixel_scale_arcsec=0.03, wavelengths_m=(500.0 / 1e9,))
    assert sampled.file_digests == {str(prior): hashlib.sha256(content).hexdigest()}
    assert sampled.draw.to_mapping() == mono.draw.to_mapping()
    assert sampled.coefficients == mono.coefficients
    assert sampled.basis.aperture_rms_nm(sampled.coefficients) == pytest.approx(10.0, abs=1e-10)
    np.testing.assert_array_equal(sampled.kernel().kernel, mono.kernel().kernel)
    assert dict(sampled.kernel().source) == dict(mono.kernel().source)
