"""Named PSF perturbation families for random ensemble draws.

This module defines the sampled mode families used by study ensembles and
the random-draw functions that realize them at a target wavefront RMS. Each
draw returns config-ready aberration dictionaries in the units of
``psf.aberrations``: nanometers of wavefront OPD, keyed by 1-based Noll
indices.

Families
--------
- ``segment_piston``: per-segment piston, hexike Noll mode 1.
- ``segment_tiptilt``: per-segment tip and tilt, hexike Noll modes 2-3.
- ``segment_hexike``: per-segment hexikes, SPIE default Noll modes 2-6.
- ``global_zernike``: global Zernikes, SPIE default Noll modes 4-11.

Segment piston and tip/tilt are expressed in the hexike basis rather than
through the segmented-mirror actuator interface so that every family shares
one amplitude axis: nanometers RMS of wavefront over the aperture.

Notes
-----
The normalization semantics are load-bearing for reproducing study
ensembles and must not change silently:

- Segment-family draws normalize the flattened coefficient vector to
  ``target_rms*sqrt(n_segments)`` so the aperture RMS matches the target
  for equal-area segments; global-Zernike draws normalize the coefficient
  vector to the target RMS directly.
- Segment-piston draws are mean-subtracted across segments before
  normalization: a common piston is an unobservable global phase, and
  keeping it would overstate the stated amplitude relative to the
  piston-removed pupil RMS that the PSF generator measures.
- A zero target RMS returns an empty dictionary, meaning no perturbation.
- Combined-family draws are composed at the ensemble level by splitting the
  RMS budget across families (the SPIE convention is equal variance,
  ``target/sqrt(2)`` per family) and drawing each family at its budget.
- Coefficient-space normalization is exact for segment families but not for
  global Zernikes, whose disk-normalized modes lose mode-dependent RMS when
  restricted to the segmented aperture (SPIE ensembles measured 0.87 +/- 0.05
  of nominal). RASTI ensembles must therefore realize draws through
  `renormalize_to_aperture_rms`, which rescales the drawn coefficients so
  the measured piston-removed aperture RMS equals the target exactly for
  every family and for joint (combined) draws.
"""

import numpy as np

from .aberration_models import apply_global_zernikes, apply_segment_zernikes

SEGMENT_PISTON_NOLLS = (1,)
"""Hexike Noll modes of the segment-piston family (`tuple` of `int`)."""

SEGMENT_TIPTILT_NOLLS = (2, 3)
"""Hexike Noll modes of the segment tip/tilt family (`tuple` of `int`)."""

SPIE_SEGMENT_HEXIKE_NOLLS = (2, 3, 4, 5, 6)
"""SPIE-default hexike Noll modes of the segment-hexike family
(`tuple` of `int`)."""

SPIE_GLOBAL_ZERNIKE_NOLLS = (4, 5, 6, 7, 8, 9, 10, 11)
"""SPIE-default Noll modes of the global-Zernike family (`tuple` of `int`)."""


def _validate_draw_inputs(mode_nolls, target_rms_nm, name):
    """Validate shared mode-list and amplitude arguments for family draws.

    Parameters
    ----------
    mode_nolls : sequence of `int`
        1-based Noll mode indices.
    target_rms_nm : `float`
        Target wavefront RMS in nanometers.
    name : `str`
        Argument name used in error messages.

    Returns
    -------
    modes : `list` of `int`
        Validated mode indices.
    target : `float`
        Validated target RMS.

    Raises
    ------
    ValueError
        Raised if the mode list is empty or contains indices below 1, or if
        the target RMS is negative or nonfinite.
    """
    modes = [int(mode) for mode in mode_nolls]
    if not modes:
        raise ValueError(f'{name} mode list must not be empty.')
    if any(mode < 1 for mode in modes):
        raise ValueError(f'{name} mode indices must be 1-based Noll integers (>= 1).')
    target = float(target_rms_nm)
    if not np.isfinite(target) or target < 0.0:
        raise ValueError(f'{name} target RMS must be a finite non-negative number.')
    return modes, target


def _normalize_vector(values, target_norm):
    """Rescale a vector to the requested Euclidean norm.

    Parameters
    ----------
    values : `numpy.ndarray`
        Vector to rescale.
    target_norm : `float`
        Requested Euclidean norm.

    Returns
    -------
    scaled : `numpy.ndarray`
        Rescaled vector.

    Raises
    ------
    ValueError
        Raised if the input vector has zero norm and the target is nonzero.
    """
    norm = float(np.linalg.norm(values))
    if norm == 0.0:
        raise ValueError('Cannot normalize a zero random PSF coefficient vector.')
    return np.asarray(values, dtype=float) * (float(target_norm) / norm)


def draw_segment_hexike_family(rng, segments, mode_nolls, target_aperture_rms_nm,
                               subtract_segment_mean=False):
    """Draw random segment-hexike coefficients at a target aperture RMS.

    Parameters
    ----------
    rng : `numpy.random.Generator`
        Random generator supplying the draw.
    segments : sequence of `int`
        Zero-based segment identifiers to perturb.
    mode_nolls : sequence of `int`
        1-based hexike Noll modes drawn on every segment.
    target_aperture_rms_nm : `float`
        Target wavefront RMS over the aperture in nanometers OPD.
    subtract_segment_mean : `bool`, optional
        Whether to subtract the across-segment mean per mode before
        normalization. Used by the piston family to remove the
        unobservable common piston.

    Returns
    -------
    segment_hexikes : `dict`
        Mapping from segment identifier to ``{mode_noll: amplitude_nm}``,
        ready for ``psf.aberrations.segment_hexikes``. Empty when the
        target RMS is zero.

    Notes
    -----
    For equal-area segment modes the aperture RMS is approximately
    ``sqrt(sum(coeff**2)/n_segments)``, so the flattened coefficient vector
    is normalized to ``target*sqrt(n_segments)``. The generated PSF records
    the measured pupil RMS.
    """
    modes, target = _validate_draw_inputs(mode_nolls, target_aperture_rms_nm,
                                          'segment hexike family')
    segment_ids = [int(segment) for segment in segments]
    if not segment_ids:
        raise ValueError('segment hexike family segment list must not be empty.')
    if target == 0.0:
        return {}

    raw = rng.standard_normal((len(segment_ids), len(modes)))
    if subtract_segment_mean:
        if len(segment_ids) < 2:
            raise ValueError('Mean subtraction requires at least two segments.')
        raw = raw - raw.mean(axis=0, keepdims=True)
    scaled = _normalize_vector(raw.ravel(), target * np.sqrt(len(segment_ids)))
    matrix = scaled.reshape((len(segment_ids), len(modes)))
    return {
        segment: {
            mode: float(matrix[seg_idx, mode_idx]) for mode_idx, mode in enumerate(modes)
        }
        for seg_idx, segment in enumerate(segment_ids)
    }


def draw_segment_piston_family(rng, segments, target_aperture_rms_nm):
    """Draw a random segment-piston perturbation at a target aperture RMS.

    Pistons are drawn as hexike Noll mode 1 and mean-subtracted across
    segments before normalization, so the target refers to the
    piston-removed wavefront RMS that actually shapes the PSF.

    Parameters
    ----------
    rng : `numpy.random.Generator`
        Random generator supplying the draw.
    segments : sequence of `int`
        Zero-based segment identifiers to perturb.
    target_aperture_rms_nm : `float`
        Target wavefront RMS over the aperture in nanometers OPD.

    Returns
    -------
    segment_hexikes : `dict`
        Mapping from segment identifier to ``{1: amplitude_nm}``, ready
        for ``psf.aberrations.segment_hexikes``. Empty when the target
        RMS is zero.
    """
    return draw_segment_hexike_family(
        rng,
        segments,
        SEGMENT_PISTON_NOLLS,
        target_aperture_rms_nm,
        subtract_segment_mean=True,
    )


def draw_segment_tiptilt_family(rng, segments, target_aperture_rms_nm):
    """Draw a random segment tip/tilt perturbation at a target aperture RMS.

    Tips and tilts are drawn as hexike Noll modes 2 and 3, keeping the
    family on the same nanometers-RMS wavefront axis as the other
    families. A unit-RMS linear ramp over a regular hexagon of circumradius
    ``a`` has slope ``sqrt(24/5)/a``, so a hexike wavefront-OPD coefficient
    ``c`` corresponds to a mirror surface tilt of
    ``sqrt(24/5)*c/segment_point_to_point`` radians.

    Parameters
    ----------
    rng : `numpy.random.Generator`
        Random generator supplying the draw.
    segments : sequence of `int`
        Zero-based segment identifiers to perturb.
    target_aperture_rms_nm : `float`
        Target wavefront RMS over the aperture in nanometers OPD.

    Returns
    -------
    segment_hexikes : `dict`
        Mapping from segment identifier to ``{2: tip_nm, 3: tilt_nm}``,
        ready for ``psf.aberrations.segment_hexikes``. Empty when the
        target RMS is zero.
    """
    return draw_segment_hexike_family(
        rng,
        segments,
        SEGMENT_TIPTILT_NOLLS,
        target_aperture_rms_nm,
    )


def draw_global_zernike_family(rng, mode_nolls, target_rms_nm):
    """Draw random global-Zernike coefficients at a target RMS.

    Parameters
    ----------
    rng : `numpy.random.Generator`
        Random generator supplying the draw.
    mode_nolls : sequence of `int`
        1-based Noll modes to draw.
    target_rms_nm : `float`
        Target wavefront RMS over the pupil in nanometers OPD.

    Returns
    -------
    global_zernikes : `dict`
        Mapping from Noll mode to amplitude in nanometers, ready for
        ``psf.aberrations.global_zernikes``. Empty when the target RMS
        is zero.
    """
    modes, target = _validate_draw_inputs(mode_nolls, target_rms_nm,
                                          'global Zernike family')
    if target == 0.0:
        return {}
    coeffs = _normalize_vector(rng.standard_normal(len(modes)), target)
    return {mode: float(coeffs[idx]) for idx, mode in enumerate(modes)}


def measure_aperture_rms_nm(telescope_data, segment_hexikes=None,
                            global_zernikes=None):
    """Measure the piston-removed wavefront RMS of drawn aberrations.

    The measurement reproduces the PSF generator's ``total_rms_nm``
    definition: the OPD is evaluated on the pupil grid and its RMS is taken
    over the illuminated aperture only, after removing the mean.

    Parameters
    ----------
    telescope_data : `dict`
        Pupil-side telescope dictionary returned by
        :func:`hwoslaps.psf.telescope_models.create_hcipy_telescope`.
    segment_hexikes : `dict`, optional
        Mapping from segment identifier to ``{mode_noll: amplitude_nm}``.
    global_zernikes : `dict`, optional
        Mapping from Noll mode to amplitude in nanometers.

    Returns
    -------
    measured_rms_nm : `float`
        Piston-removed wavefront RMS over the illuminated aperture in
        nanometers OPD. Zero when both aberration dictionaries are empty.
    """
    wavelength = telescope_data['wavelength']
    opd = np.zeros_like(np.asarray(telescope_data['pupil_grid'].zeros()))
    rad_to_opd = wavelength / (2.0 * np.pi)
    if segment_hexikes:
        phase_screen, _ = apply_segment_zernikes(segment_hexikes,
                                                 telescope_data, wavelength)
        opd = opd + np.asarray(phase_screen) * rad_to_opd
    if global_zernikes:
        phase_screen = apply_global_zernikes(global_zernikes, telescope_data,
                                             wavelength)
        opd = opd + np.asarray(phase_screen) * rad_to_opd
    valid = np.asarray(telescope_data['aper']) > 0.5
    opd_valid = opd[valid]
    opd_valid = opd_valid - np.mean(opd_valid)
    return float(np.sqrt(np.mean(opd_valid**2)) * 1e9)


def renormalize_to_aperture_rms(telescope_data, target_rms_nm,
                                segment_hexikes=None, global_zernikes=None):
    """Rescale drawn aberrations to an exact measured aperture RMS.

    The OPD is linear in the coefficients, so multiplying every coefficient
    by ``target/measured`` makes the measured piston-removed aperture RMS
    equal the target exactly. This closes the gap between coefficient-space
    normalization and physical wavefront amplitude, which is significant
    for global Zernikes on the segmented aperture. Passing both aberration
    dictionaries rescales their summed wavefront, which is the correct
    joint treatment for combined-family draws.

    Parameters
    ----------
    telescope_data : `dict`
        Pupil-side telescope dictionary returned by
        :func:`hwoslaps.psf.telescope_models.create_hcipy_telescope`. Must
        match the telescope configuration used to generate the PSF.
    target_rms_nm : `float`
        Target piston-removed aperture wavefront RMS in nanometers OPD.
    segment_hexikes : `dict`, optional
        Mapping from segment identifier to ``{mode_noll: amplitude_nm}``.
    global_zernikes : `dict`, optional
        Mapping from Noll mode to amplitude in nanometers.

    Returns
    -------
    segment_hexikes : `dict`
        Rescaled segment-hexike dictionary; empty when the target RMS is
        zero or no segment aberrations were given.
    global_zernikes : `dict`
        Rescaled global-Zernike dictionary; empty when the target RMS is
        zero or no global aberrations were given.

    Raises
    ------
    ValueError
        Raised if the target RMS is negative or nonfinite, or if the
        target is nonzero while the drawn aberrations measure zero RMS.
    """
    target = float(target_rms_nm)
    if not np.isfinite(target) or target < 0.0:
        raise ValueError('Renormalization target RMS must be a finite '
                         'non-negative number.')
    if target == 0.0:
        return {}, {}
    measured = measure_aperture_rms_nm(telescope_data,
                                       segment_hexikes=segment_hexikes,
                                       global_zernikes=global_zernikes)
    if measured == 0.0:
        raise ValueError('Cannot renormalize aberrations with zero measured '
                         'aperture RMS to a nonzero target.')
    scale = target / measured
    scaled_segment = {
        segment: {mode: float(coeff * scale) for mode, coeff in modes.items()}
        for segment, modes in (segment_hexikes or {}).items()
    }
    scaled_global = {
        mode: float(coeff * scale)
        for mode, coeff in (global_zernikes or {}).items()
    }
    return scaled_segment, scaled_global
