"""Circular pupils against analytic Airy optics: scale, obscuration, area, Strehl, pixel integration, layout."""

import math

import numpy as np
import pytest
from scipy import integrate, optimize, special

from hwoslaps.constants import ARCSEC_PER_RAD
from hwoslaps.optics.metrics import captured_power_fraction, encircled_energy, fwhm_arcsec, strehl_ratio
from hwoslaps.optics.optical_psf import FocalField
from hwoslaps.optics.providers import build_psf_provider, parse_psf
from hwoslaps.optics.pupils import build_pupil, parse_pupil

pytestmark = pytest.mark.backend

WAVELENGTH_NM = 500.0


def _lambda_over_d_arcsec(diameter_m):
    return WAVELENGTH_NM * 1e-9 / diameter_m * ARCSEC_PER_RAD


def _circular_psf(diameter_m=1.0, pixels=512, *, obscuration=0.0, oversampling=3, shape=(11, 11),
                  wavefront=None):
    pupil = {"kind": "circular", "diameter_m": diameter_m, "pixels": pixels, "supersampling": 4,
             "obscuration_ratio": obscuration}
    truth = {"kind": "optical", "pupil": pupil, "focal_length_m": 10.0, "wavelength_nm": WAVELENGTH_NM,
             "detector_oversampling": oversampling, "kernel_shape": list(shape), "wavefront": wavefront or {}}
    return build_psf_provider(parse_psf({"truth": truth}).truth,
                              pixel_scale_arcsec=_lambda_over_d_arcsec(diameter_m) / 2)


def _airy_encircled_energy(v):
    return 1.0 - special.j0(v) ** 2 - special.j1(v) ** 2


def test_airy_fwhm_and_encircled_energy():
    psf = _circular_psf()
    field = psf.focal_field(samples_per_lambda_over_d=16, radius_lambda_over_d=6)
    lambda_over_d = _lambda_over_d_arcsec(1.0)
    half_v = optimize.brentq(lambda v: (2 * special.j1(v) / v) ** 2 - 0.5, 1.0, 2.0)
    assert fwhm_arcsec(field) == pytest.approx(2 * half_v / math.pi * lambda_over_d, rel=2e-3)
    radii = np.array([1.219670, 2.233]) * lambda_over_d
    np.testing.assert_allclose(encircled_energy(field, radii), _airy_encircled_energy(math.pi * radii / lambda_over_d),
                               rtol=0.0, atol=3e-3)
    with pytest.raises(ValueError, match="field radius"):
        encircled_energy(field, [7 * lambda_over_d])


@pytest.mark.parametrize("coordinate_ulps", [-8, -4, 0, 4, 8])
def test_fwhm_integer_bin_edges_keep_actual_mean_radii(coordinate_ulps):
    coordinate = 1.0 + coordinate_ulps*np.spacing(1.0)
    x, y = np.meshgrid([-coordinate, 0.0, coordinate], [-coordinate, 0.0, coordinate])
    image = np.array([[1., 2., 1.], [2., 4., 2.], [1., 2., 1.]])/16
    field = FocalField(image, x, y, 1.0, 5e-7)
    # Independently enumerated bins: centre alone, then four axes and four diagonals.
    # Half-height interpolation is (4-2)/(4-1.5)=0.8; use the actual coordinate radius.
    expected = 0.8*coordinate*(1+math.sqrt(2))
    np.testing.assert_array_max_ulp(fwhm_arcsec(field), expected, maxulp=4)


def test_fwhm_retains_a_real_fractional_peak():
    sigma = 1.1
    centre = np.array([0.27, -0.13])
    y, x = np.mgrid[-15:16, -15:16].astype(float)
    image = np.exp(-((y-centre[0])**2 + (x-centre[1])**2) / (2*sigma**2))
    field = FocalField(image/image.sum(), x, y, 1.0, 5e-7)
    # Closed Gaussian three-sample peak, without calling the metric's peak helper.
    fitted = np.sinh(centre/sigma**2) / (2*(np.exp(1/(2*sigma**2))-np.cosh(centre/sigma**2)))
    # Hand-enumerated memberships of [0,1) and [1,2) around this fractional peak.
    points = (np.array([(0, 0), (1, 0), (0, -1)]),
              np.array([(-1, -1), (-1, 0), (-1, 1), (0, -2), (0, 1),
                        (1, -1), (1, 1), (2, -1), (2, 0)]))
    mean_radius, mean_intensity = [], []
    for members in points:
        mean_radius.append(np.mean(np.hypot(members[:, 0]-fitted[0], members[:, 1]-fitted[1])))
        mean_intensity.append(np.mean(np.exp(-np.sum((members-centre)**2, axis=1)/(2*sigma**2))))
    half = 0.5*np.exp(-np.sum(centre**2)/(2*sigma**2))
    fraction = (mean_intensity[0]-half)/(mean_intensity[0]-mean_intensity[1])
    expected = 2*(mean_radius[0] + fraction*(mean_radius[1]-mean_radius[0]))
    assert fwhm_arcsec(field) == pytest.approx(expected, rel=64*np.finfo(float).eps)


def test_obscured_encircled_energy_matches_annular_form():
    obscuration = 0.3
    psf = _circular_psf(obscuration=obscuration)
    field = psf.focal_field(samples_per_lambda_over_d=16, radius_lambda_over_d=6)
    lambda_over_d = _lambda_over_d_arcsec(1.0)

    def annular(v):
        e = obscuration
        cross = integrate.quad(lambda x: special.j1(x) * special.j1(e * x) / x, 0.0, v, limit=200)[0]
        return (_airy_encircled_energy(v) + e ** 2 * _airy_encircled_energy(e * v) - 4 * e * cross) / (1 - e ** 2)

    v = np.array([2.0, 4.0, 8.0])
    np.testing.assert_allclose(encircled_energy(field, v / math.pi * lambda_over_d), [annular(x) for x in v],
                               rtol=0.0, atol=3e-3)


def test_obscuration_and_spiders_set_area_and_orientation():
    pupil = build_pupil(parse_pupil({"kind": "circular", "diameter_m": 1.0, "pixels": 512, "supersampling": 4,
                                     "obscuration_ratio": 0.3,
                                     "spiders": {"count": 4, "width_m": 0.01, "angle_deg": 45.0}}, "pupil"))
    expected = math.pi / 4 * (1 - 0.3 ** 2) - 4 * 0.01 * 0.5 * (1 - 0.3)
    assert pupil.collecting_area_m2 == pytest.approx(expected, rel=2e-3)
    transmission = np.asarray(pupil.transmission)
    x, y = np.asarray(pupil.grid.x), np.asarray(pupil.grid.y)

    def at(x0, y0):
        return transmission[np.argmin(np.hypot(x - x0, y - y0))]

    radius = 0.35
    for angle in (45.0, 135.0, 225.0, 315.0):
        assert at(radius * math.cos(math.radians(angle)), radius * math.sin(math.radians(angle))) == 0.0
    for angle in (0.0, 90.0, 180.0, 270.0):
        assert at(radius * math.cos(math.radians(angle)), radius * math.sin(math.radians(angle))) == 1.0
    assert at(0.0, 0.0) == 0.0
    unobstructed = build_pupil(parse_pupil({"kind": "circular", "diameter_m": 1.0, "pixels": 512,
                                            "supersampling": 4}, "pupil"))
    assert unobstructed.collecting_area_m2 == pytest.approx(math.pi / 4, rel=2e-3)


@pytest.mark.parametrize("defocus_nm", [0.0, 20.0, 60.0])
def test_strehl_of_pure_defocus(defocus_nm):
    psf = _circular_psf(pixels=256, wavefront={"zernikes": {4: defocus_nm}})
    a = 2 * math.pi * math.sqrt(3) * defocus_nm / WAVELENGTH_NM
    expected = 1.0 if a == 0.0 else (math.sin(a) / a) ** 2
    if defocus_nm == 0.0:
        assert strehl_ratio(psf) == 1.0
    assert strehl_ratio(psf) == pytest.approx(expected, abs=2e-3)


def test_pixel_integrated_kernel_matches_analytic_airy():
    psf = _circular_psf(diameter_m=2.0, pixels=256, oversampling=9)
    kernel = psf.kernel().kernel

    def pixel_power(dy, dx):
        def intensity(y, x):
            v = math.pi * math.hypot(x, y)
            return 1.0 if v == 0.0 else (2 * special.j1(v) / v) ** 2

        return integrate.dblquad(intensity, (dx - 0.5) / 2, (dx + 0.5) / 2, (dy - 0.5) / 2, (dy + 0.5) / 2,
                                 epsabs=1e-12, epsrel=1e-10)[0]

    centre = pixel_power(0, 0)
    for dy, dx in [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]:
        assert kernel[5 + dy, 5 + dx] / kernel[5, 5] == pytest.approx(pixel_power(dy, dx) / centre, rel=2e-3)


def test_unaberrated_kernel_is_centred():
    kernel = _circular_psf(pixels=256, shape=(15, 15)).kernel().kernel
    np.testing.assert_allclose(kernel, kernel[::-1, ::-1], rtol=0.0, atol=1e-14)
    rows, columns = np.indices(kernel.shape)
    assert np.sum(rows * kernel) == pytest.approx(7.0, abs=1e-12)
    assert np.sum(columns * kernel) == pytest.approx(7.0, abs=1e-12)


def test_rectangular_kernel_keeps_row_column_order():
    coma = {"zernikes": {7: 50.0}}
    square = _circular_psf(diameter_m=2.0, pixels=256, wavefront=coma, shape=(31, 31)).kernel().kernel
    peak = square.max()
    assert np.max(np.abs(square - square.T)) > 1e-3 * peak
    assert np.max(np.abs(square - square[::-1, :])) > 1e-3 * peak
    wide = _circular_psf(diameter_m=2.0, pixels=256, wavefront=coma, shape=(17, 31)).kernel().kernel
    assert wide.shape == (17, 31)
    band = square[7:24, :]
    np.testing.assert_allclose(wide, band / band.sum(), rtol=1e-10, atol=0.0)


def test_captured_power_fraction_is_bracketed_by_airy_encircled_energy():
    psf = _circular_psf(diameter_m=2.0, pixels=256)
    half_width = 5.5 / 2
    fraction = captured_power_fraction(psf)
    assert _airy_encircled_energy(math.pi * half_width) <= fraction <= _airy_encircled_energy(
        math.pi * math.sqrt(2) * half_width)
