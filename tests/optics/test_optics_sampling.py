"""The aliasing and sub-pixel Nyquist guards of optical kernels (optics.optical_psf.check_sampling)."""

import pytest

from hwoslaps.optics.optical_psf import check_sampling
from hwoslaps.optics.providers import parse_psf


def _spec(pupil, oversampling, shape):
    return parse_psf({"truth": {"kind": "optical", "pupil": pupil, "focal_length_m": 144.0, "wavelength_nm": 500.0,
                                "detector_oversampling": oversampling, "kernel_shape": [shape, shape]}}).truth


# Paper geometry: the 512-pixel pupil over 7.225765 m repeats the focal plane every
# 512 lambda / D, 7.308" at 500 nm and 6.577" at 450 nm, against a 999 x 0.00716" = 7.1528" kernel.
# P1: 1.827" against 17 x 0.03" = 0.51"; 0.03" / 11 against lambda / 2D = 0.007136".
# A 0.1" pixel with D = 2.4 m at 550 nm: lambda / 2D = 0.023635", so oversampling 5 is the least.
@pytest.mark.parametrize("pupil, oversampling, shape, scale, wavelength_nm, message", [
    ("paper", 3, 999, 0.00716, 500.0, None),
    ("paper", 3, 999, 0.00716, 450.0, "aliased kernel at 450 nm.*psf.truth.pupil.pixels"),
    ("p1", 11, 17, 0.03, 500.0, None),
    ("small", 1, 21, 0.1, 550.0, "under-resolved kernel at 550 nm.*detector_oversampling to at least 5"),
    ("small", 4, 21, 0.1, 550.0, "under-resolved kernel at 550 nm"),
    ("small", 5, 21, 0.1, 550.0, None),
], ids=["paper-500nm", "paper-450nm", "p1", "small-n1", "small-n4", "small-n5"])
def test_sampling_guards(p1_pupil, paper_pupil, pupil, oversampling, shape, scale, wavelength_nm, message):
    pupils = {"paper": paper_pupil, "p1": p1_pupil,
              "small": {"kind": "circular", "diameter_m": 2.4, "pixels": 512, "supersampling": 2}}
    spec = _spec(pupils[pupil], oversampling, shape)
    if message is None:
        check_sampling(spec, pixel_scale_arcsec=scale, wavelength_m=wavelength_nm * 1e-9)
    else:
        with pytest.raises(ValueError, match=message):
            check_sampling(spec, pixel_scale_arcsec=scale, wavelength_m=wavelength_nm * 1e-9)
