"""PSF quality measures, as calls separate from kernel evaluation.

``strehl_ratio`` is the on-axis intensity of the aberrated pupil relative to the
unaberrated one, exact for the sampled pupil (no focal-plane sampling); in the
small-aberration limit it is ``exp(-(2 pi sigma / lambda)^2)``. The wavefront RMS
``sigma`` is ``WavefrontBasis.aperture_rms_nm``.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike

from .optical_psf import FocalField, OpticalPSF
from .wavefront import WavefrontCoefficients

__all__ = ["captured_power_fraction", "encircled_energy", "fwhm_arcsec", "strehl_ratio"]


def strehl_ratio(psf: OpticalPSF, wavelength_m: float | None = None, *,
                 coefficients: WavefrontCoefficients | None = None) -> float:
    """``|sum E w|^2 / |sum t w|^2`` for the aberrated pupil field ``E`` and transmission ``t``."""
    field = np.asarray(psf.pupil_wavefront(wavelength_m, coefficients=coefficients).electric_field)
    weights = psf.pupil.grid.weights
    transmission = np.asarray(psf.pupil.transmission)
    return float(np.abs(np.sum(field * weights)) ** 2 / np.abs(np.sum(transmission * weights)) ** 2)


def _field_radius_arcsec(field: FocalField) -> float:
    return float(min(np.max(np.abs(field.x_arcsec)), np.max(np.abs(field.y_arcsec))))


def encircled_energy(field: FocalField, radii_arcsec: ArrayLike) -> np.ndarray:
    """Fraction of the pupil power within each radius of the optical axis.

    The intensity is in units of the pupil power, so light outside the evaluated field still
    counts in the denominator. A radius beyond the field raises.
    """
    radii = np.asarray(radii_arcsec, dtype=float)
    limit = _field_radius_arcsec(field)
    if np.any(radii > limit) or np.any(radii < 0.0):
        raise ValueError(f"radii must lie in [0, {limit:.6g}] arcsec, the field radius")
    distance = np.hypot(field.x_arcsec, field.y_arcsec)
    return np.array([float(np.sum(field.intensity[distance <= radius])) for radius in radii.ravel()]).reshape(
        radii.shape)


def _parabola_offset(left: float, centre: float, right: float) -> float:
    curvature = left - 2.0 * centre + right
    return 0.0 if curvature == 0.0 else (left - right) / (2.0 * curvature)


def fwhm_arcsec(field: FocalField) -> float:
    """Full width at half maximum of the azimuthally averaged intensity about its peak.

    The peak is refined by three-point parabolas along each axis; samples are averaged in
    radial bins one sample pitch wide; the half-maximum crossing is interpolated linearly
    between the mean radii of the two bins that bracket it. Raises when the peak sample lies
    on the field edge or the profile never falls below half maximum.
    """
    intensity = field.intensity
    row, column = np.unravel_index(int(np.argmax(intensity)), intensity.shape)
    if row in (0, intensity.shape[0] - 1) or column in (0, intensity.shape[1] - 1):
        raise ValueError("the intensity peak lies on the edge of the field")
    pitch = field.sample_arcsec
    centre_x = field.x_arcsec[row, column] + pitch * _parabola_offset(
        intensity[row, column - 1], intensity[row, column], intensity[row, column + 1])
    centre_y = field.y_arcsec[row, column] + pitch * _parabola_offset(
        intensity[row - 1, column], intensity[row, column], intensity[row + 1, column])
    radius = np.hypot(field.x_arcsec - centre_x, field.y_arcsec - centre_y).ravel()
    values = intensity.ravel()
    scaled_radius = radius / pitch
    integer_radius = np.rint(scaled_radius)
    # Stabilize bin membership at a rounded integer; keep true radii for the means.
    roundoff = 8 * np.spacing(np.maximum(integer_radius, 1.0))
    classified_radius = np.where(np.abs(scaled_radius - integer_radius) <= roundoff,
                                 integer_radius, scaled_radius)
    bins = np.floor(classified_radius).astype(int)
    counts = np.bincount(bins)
    filled = counts > 0
    mean_radius = np.bincount(bins, weights=radius)[filled] / counts[filled]
    mean_value = np.bincount(bins, weights=values)[filled] / counts[filled]
    half = 0.5 * float(np.max(values))
    below = np.flatnonzero(mean_value < half)
    if below.size == 0 or below[0] == 0:
        raise ValueError("the azimuthal profile never falls below half maximum within the field")
    outer = int(below[0])
    inner = outer - 1
    fraction = (mean_value[inner] - half) / (mean_value[inner] - mean_value[outer])
    return float(2.0 * (mean_radius[inner] + fraction * (mean_radius[outer] - mean_radius[inner])))


def captured_power_fraction(psf: OpticalPSF, wavelength_m: float | None = None, *,
                            coefficients: WavefrontCoefficients | None = None) -> float:
    """Share of the pupil power that lands inside the kernel support, before normalization."""
    power = psf.detector_power(wavelength_m, coefficients=coefficients)
    return float(np.sum(power.ravel())) / psf.pupil.total_power
