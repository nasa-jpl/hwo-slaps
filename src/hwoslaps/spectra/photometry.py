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


def _scaled_measure(sed: SED | None, band: Bandpass, *, response: bool = True) -> tuple[np.ndarray, np.ndarray, float]:
    """One bounded integrand and its logarithmic scale, on the prescribed spectral grid."""
    wavelengths, throughput = band.integration_grid(sed)
    shape = np.ones_like(wavelengths) if sed is None else sed.fnu(wavelengths)
    weights = throughput if response else np.ones_like(throughput)
    positive = (shape > 0.0) & (weights > 0.0)
    normalized = np.zeros_like(wavelengths)
    if not np.any(positive):
        return wavelengths, normalized, -math.inf
    logarithms = np.log(shape[positive]) + np.log(weights[positive])
    scale = float(np.max(logarithms))
    normalized[positive] = np.exp(logarithms - scale)
    return wavelengths, normalized, scale


def _log_integral(sed: SED | None, band: Bandpass, *, response: bool = True) -> float:
    wavelengths, normalized, scale = _scaled_measure(sed, band, response=response)
    integral = integrate_dlnlambda(normalized, wavelengths)
    return -math.inf if integral == 0.0 else scale + math.log(integral)


def _positive_log_integral(sed: SED | None, band: Bandpass, *, response: bool = True) -> float:
    integral = _log_integral(sed, band, response=response)
    if not math.isfinite(integral):
        raise ValueError("the spectrum has no photons through the bandpass")
    return integral


def _positive_from_log(value: float, name: str) -> float:
    try:
        result = math.exp(value)
    except OverflowError as error:
        raise ValueError(f"{name} has no finite representable value") from error
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} has no positive finite representable value")
    return result


def detected_flux_per_m2(sed: SED, scale_jy: float, bandpass: Bandpass) -> float:
    scale = _number(scale_jy, "scale_jy", positive=True)
    integral = _log_integral(sed, bandpass)
    if integral == -math.inf:
        return 0.0
    return _positive_from_log(math.log(scale) + math.log(JANSKY_SI) - math.log(PLANCK_J_S) + integral,
                              "detected flux per square metre")


def ab_scale_jy(sed: SED, magnitude: float, band: Bandpass) -> float:
    scale = math.log(ab_to_fnu_jy(magnitude)) + _positive_log_integral(None, band) - _positive_log_integral(sed, band)
    return _positive_from_log(scale, "SED scale_jy")


def synthetic_ab_mag(sed: SED, scale_jy: float, band: Bandpass) -> float:
    scale = _number(scale_jy, "scale_jy", positive=True)
    log_mean = math.log(scale) + _positive_log_integral(sed, band) - _positive_log_integral(None, band)
    return -2.5 / math.log(10.0) * (log_mean - math.log(AB_ZERO_POINT_JY))


def rate_from_ab(magnitude: float, bandpass: Bandpass, area_m2: float, *,
                 sed: SED | None = None, reference_band: Bandpass | None = None) -> float:
    area = _number(area_m2, "area_m2", positive=True)
    if reference_band is None:
        return area * ab_to_fnu_jy(magnitude) * JANSKY_SI / PLANCK_J_S * _positive_integral(None, bandpass)
    if sed is None:
        raise ValueError("a reference-band magnitude requires an SED")
    instrument = _log_integral(sed, bandpass)
    if instrument == -math.inf:
        return 0.0
    reference = _positive_log_integral(sed, reference_band)
    rate = (math.log(area) + math.log(ab_to_fnu_jy(magnitude)) + math.log(JANSKY_SI) - math.log(PLANCK_J_S)
            + _positive_log_integral(None, reference_band) + instrument - reference)
    return _positive_from_log(rate, "reference-band detected rate")


def sky_rate_e_per_s_per_pixel(ab_mag_per_arcsec2: float, bandpass: Bandpass, area_m2: float,
                               pixel_scale_arcsec: float, *, sed: SED | None = None,
                               reference_band: Bandpass | None = None) -> float:
    scale = _number(pixel_scale_arcsec, "pixel_scale_arcsec", positive=True)
    return rate_from_ab(ab_mag_per_arcsec2, bandpass, area_m2, sed=sed, reference_band=reference_band) * scale**2


def effective_wavelength_m(sed: SED, bandpass: Bandpass) -> float:
    wavelengths, photon_shape, _ = _scaled_measure(sed, bandpass)
    denominator = integrate_dlnlambda(photon_shape, wavelengths)
    if denominator <= 0.0:
        raise ValueError("the spectrum has no photons through the bandpass")
    maximum = float(wavelengths[-1])
    return maximum * (integrate_dlnlambda(photon_shape * (wavelengths / maximum), wavelengths) / denominator)


def band_mean_throughput(bandpass: Bandpass, sed: SED | None = None) -> float:
    numerator = _positive_log_integral(sed, bandpass)
    denominator = _positive_log_integral(sed, bandpass, response=False)
    return _positive_from_log(numerator - denominator, "band mean throughput")
