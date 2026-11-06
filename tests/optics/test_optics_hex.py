"""Hexagonally segmented pupils: area convention, containment, read-only arrays, phase units, captured power."""

import dataclasses
import math

import numpy as np
import pytest

from hwoslaps.config.checks import ConfigError
from hwoslaps.optics.metrics import captured_power_fraction, fwhm_arcsec, strehl_ratio
from hwoslaps.optics.providers import build_psf_provider, parse_psf
from hwoslaps.optics.pupils import Pupil, build_pupil, parse_pupil
from hwoslaps.optics.wavefront import WavefrontMode

pytestmark = pytest.mark.backend


def _hexagon_area(point_to_point_m):
    return 3 * math.sqrt(3) / 8 * point_to_point_m ** 2


def test_collecting_area_matches_analytic_geometry(paper_pupil):
    paper = build_pupil(parse_pupil(paper_pupil, "pupil"))
    assert paper.collecting_area_m2 == 33.606448937520405
    assert 0.95 < paper.collecting_area_m2 / (19 * _hexagon_area(1.65)) < 1.005

    single = build_pupil(parse_pupil({"kind": "hex_segmented", "diameter_m": 1.0, "pixels": 256,
                                      "supersampling": 4, "rings": 0, "segment_point_to_point_m": 1.0,
                                      "gap_m": 0.0}, "pupil"))
    assert single.collecting_area_m2 == pytest.approx(_hexagon_area(1.0), rel=1e-2)

    ring = build_pupil(parse_pupil({**paper_pupil, "gap_m": 0.0, "central_segment": False}, "pupil"))
    assert ring.segment_count == 18
    assert ring.collecting_area_m2 == pytest.approx(18 * _hexagon_area(1.65), rel=1e-3)


def test_pupil_grid_must_contain_the_aperture(p1_pupil, paper_pupil):
    for mapping in (p1_pupil, paper_pupil):
        assert build_pupil(parse_pupil(mapping, "pupil")).active_segments == tuple(range(19))
    flat_to_flat = 1.65 * math.sqrt(3) / 2
    reach = 2 * (flat_to_flat + 0.006) + flat_to_flat / 2
    narrow = {**p1_pupil, "diameter_m": 7.0}
    with pytest.raises(ConfigError, match=f"at least {2 * reach:.6g}") as caught:
        parse_pupil(narrow, "psf.truth.pupil")
    assert caught.value.path == "psf.truth.pupil.diameter_m"
    spec = parse_pupil(p1_pupil, "pupil")
    with pytest.raises(ValueError, match=f"at least {2 * reach:.6g}"):
        build_pupil(dataclasses.replace(spec, diameter_m=7.0))


def test_pupil_cannot_be_edited_through_its_arrays(p1_pupil):
    pupil = build_pupil(parse_pupil(p1_pupil, "pupil"))
    caller = pupil.transmission.copy()
    direct = Pupil(pupil.spec, pupil.grid, caller, pupil.segments, pupil.segment_centres)
    caller[:] = 0.0
    assert np.array_equal(direct.transmission, pupil.transmission)
    for array in (pupil.transmission, pupil.illuminated_mask, direct.transmission):
        with pytest.raises(ValueError, match="read-only"):
            array[0] = 0
    mask = pupil.segments[3]
    mask[:] = 0.0
    assert np.max(pupil.segments[3]) > 0.5


@pytest.mark.parametrize("defocus_nm", [5.0, 20.0])
def test_strehl_follows_marechal(paper_pupil, defocus_nm):
    truth = {"kind": "optical", "pupil": paper_pupil, "focal_length_m": 144.0, "wavelength_nm": 500.0,
             "detector_oversampling": 3, "kernel_shape": [51, 51], "wavefront": {"zernikes": {4: defocus_nm}}}
    psf = build_psf_provider(parse_psf({"truth": truth}).truth, pixel_scale_arcsec=0.00716)
    sigma_nm = psf.basis.aperture_rms_nm(psf.coefficients)
    marechal = math.exp(-(2 * math.pi * sigma_nm / 500.0) ** 2)
    assert strehl_ratio(psf) == pytest.approx(marechal, rel=1e-3, abs=5e-4)


def test_recorded_captured_power_equals_the_metric(p1_truth):
    psf = build_psf_provider(parse_psf({"truth": p1_truth}).truth, pixel_scale_arcsec=0.03)
    recorded = psf.kernel().source["captured_power_fraction"]
    assert recorded == captured_power_fraction(psf, 5e-7)
    assert 0.0 < recorded < 1.0
    mode = WavefrontMode("segment_hexikes", 1, 0)
    shifted = psf.coefficients.replace(mode, 1.0)
    derivative_side = psf.kernel(coefficients=shifted).source["captured_power_fraction"]
    assert derivative_side == captured_power_fraction(psf, coefficients=shifted)
    assert derivative_side != recorded


def test_segmented_kernel_converges_with_pupil_sampling(paper_pupil):
    results = []
    for pixels in (96, 128, 192):
        truth = {"kind": "optical", "pupil": {**paper_pupil, "pixels": pixels},
                 "focal_length_m": 144.0, "wavelength_nm": 500.0,
                 "detector_oversampling": 3, "kernel_shape": [15, 15], "wavefront": {}}
        provider = build_psf_provider(parse_psf({"truth": truth}).truth, pixel_scale_arcsec=0.00716)
        kernel = provider.kernel().kernel
        rows, columns = np.indices(kernel.shape)
        centre = np.array(kernel.shape) // 2
        moment = np.sum(kernel * ((rows - centre[0]) ** 2 + (columns - centre[1]) ** 2))
        field = provider.focal_field(samples_per_lambda_over_d=6, radius_lambda_over_d=6)
        assert strehl_ratio(provider) == pytest.approx(1.0, rel=1e-12, abs=1e-12)
        assert provider.basis.aperture_rms_nm(provider.coefficients) == pytest.approx(0.0, abs=1e-12)
        results.append((fwhm_arcsec(field), kernel[tuple(centre)], moment))
    reference = results[-1]
    for width, peak, moment in results:
        assert width == pytest.approx(reference[0], rel=2e-3)
        assert peak == pytest.approx(reference[1], rel=5e-3)
        assert moment == pytest.approx(reference[2], rel=1e-2)
