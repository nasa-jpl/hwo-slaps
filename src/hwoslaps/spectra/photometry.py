"""Photon-counting AB photometry on the bandpass's shared logarithmic measure."""

from __future__ import annotations

import math
from numbers import Real

import numpy as np

from ..constants import AB_ZERO_POINT_JY, JANSKY_SI, PLANCK_J_S
from .bandpass import Bandpass, integrate_dlnlambda
from .sed import SED

__all__ = ["ab_scale_jy", "ab_to_fnu_jy", "band_mean_throughput", "detected_flux_per_m2",
           "effective_wavelength_m", "fnu_jy_to_ab", "rate_from_ab", "sky_rate_e_per_s_per_pixel", "synthetic_ab_mag"]


def _number(value: float, name: str, *, positive: bool = False) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite real number")
    if positive and value <= 0.0:
        raise ValueError(f"{name} must be positive")
    return float(value)


def ab_to_fnu_jy(magnitude: float) -> float:
    magnitude = _number(magnitude, "magnitude")
    flux = AB_ZERO_POINT_JY * 10.0 ** (-0.4 * magnitude)
    if not math.isfinite(flux) or flux <= 0.0:
        raise ValueError("magnitude has no positive finite representable flux density")
    return flux


def fnu_jy_to_ab(fnu_jy: float) -> float:
    return -2.5 * math.log10(_number(fnu_jy, "fnu_jy", positive=True) / AB_ZERO_POINT_JY)


def _photon_integral(sed: SED | None, band: Bandpass) -> float:
    wavelengths, throughput = band.integration_grid(sed)
    shape = 1.0 if sed is None else sed.fnu(wavelengths)
    return integrate_dlnlambda(throughput * shape, wavelengths)


def _positive_integral(sed: SED | None, band: Bandpass) -> float:
    integral = _photon_integral(sed, band)
    if not math.isfinite(integral) or integral <= 0.0:
        raise ValueError("the spectrum has no photons through the bandpass")
    return integral


def detected_flux_per_m2(sed: SED, scale_jy: float, bandpass: Bandpass) -> float:
    scale = _number(scale_jy, "scale_jy", positive=True)
    return scale * JANSKY_SI / PLANCK_J_S * _photon_integral(sed, bandpass)


def ab_scale_jy(sed: SED, magnitude: float, band: Bandpass) -> float:
    return ab_to_fnu_jy(magnitude) * _positive_integral(None, band) / _positive_integral(sed, band)


def synthetic_ab_mag(sed: SED, scale_jy: float, band: Bandpass) -> float:
    scale = _number(scale_jy, "scale_jy", positive=True)
    mean_fnu = scale * _positive_integral(sed, band) / _positive_integral(None, band)
    return fnu_jy_to_ab(mean_fnu)


def rate_from_ab(magnitude: float, bandpass: Bandpass, area_m2: float, *,
                 sed: SED | None = None, reference_band: Bandpass | None = None) -> float:
    area = _number(area_m2, "area_m2", positive=True)
    if reference_band is None:
        return area * ab_to_fnu_jy(magnitude) * JANSKY_SI / PLANCK_J_S * _positive_integral(None, bandpass)
    if sed is None:
        raise ValueError("a reference-band magnitude requires an SED")
    return area * detected_flux_per_m2(sed, ab_scale_jy(sed, magnitude, reference_band), bandpass)


def sky_rate_e_per_s_per_pixel(ab_mag_per_arcsec2: float, bandpass: Bandpass, area_m2: float,
                               pixel_scale_arcsec: float, *, sed: SED | None = None,
                               reference_band: Bandpass | None = None) -> float:
    scale = _number(pixel_scale_arcsec, "pixel_scale_arcsec", positive=True)
    return rate_from_ab(ab_mag_per_arcsec2, bandpass, area_m2, sed=sed, reference_band=reference_band) * scale**2


def effective_wavelength_m(sed: SED, bandpass: Bandpass) -> float:
    wavelengths, throughput = bandpass.integration_grid(sed)
    photon_shape = throughput * sed.fnu(wavelengths)
    denominator = integrate_dlnlambda(photon_shape, wavelengths)
    if denominator <= 0.0:
        raise ValueError("the spectrum has no photons through the bandpass")
    return integrate_dlnlambda(photon_shape * wavelengths, wavelengths) / denominator


def band_mean_throughput(bandpass: Bandpass, sed: SED | None = None) -> float:
    wavelengths, throughput = bandpass.integration_grid(sed)
    shape = np.ones_like(wavelengths) if sed is None else sed.fnu(wavelengths)
    denominator = integrate_dlnlambda(shape, wavelengths)
    if denominator <= 0.0:
        raise ValueError("the spectrum has no photons on the bandpass support")
    return integrate_dlnlambda(shape * throughput, wavelengths) / denominator
