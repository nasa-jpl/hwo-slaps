"""Named PSF perturbation families for random ensemble draws.

This module defines the sampled mode families used by study ensembles and
the random-draw functions that realize them at a target wavefront RMS.
Non-weighted draws return config-ready aberration dictionaries in the units
of ``psf.aberrations``: nanometers of wavefront OPD, keyed by 1-based Noll
indices. Weighted draws return coefficients in the prior's sequentially
orthonormalized aperture basis and must pass through
:func:`realize_weighted_draw` before being handed to HCIPy.

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

from dataclasses import dataclass, field
import math
from numbers import Integral

import numpy as np
import yaml

from .aberration_models import apply_global_zernikes, apply_segment_zernikes
from .opd_basis import ApertureBasisTransform

SEGMENT_PISTON_NOLLS = (1,)
"""Hexike Noll modes of the segment-piston family (`tuple` of `int`)."""

SEGMENT_TIPTILT_NOLLS = (2, 3)
"""Hexike Noll modes of the segment tip/tilt family (`tuple` of `int`)."""

SPIE_SEGMENT_HEXIKE_NOLLS = (2, 3, 4, 5, 6)
"""SPIE-default hexike Noll modes of the segment-hexike family
(`tuple` of `int`)."""

SPIE_GLOBAL_ZERNIKE_NOLLS = (4, 5, 6, 7, 8, 9, 10, 11)
"""SPIE-default Noll modes of the global-Zernike family (`tuple` of `int`)."""


def noll_to_radial_order(noll):
    """Return the radial order of a 1-based Noll mode index.

    Parameters
    ----------
    noll : `int`
        1-based Noll mode index.

    Returns
    -------
    radial_order : `int`
        Radial order corresponding to ``noll``.

    Raises
    ------
    ValueError
        Raised if ``noll`` is not an integer or is below one.
    """
    if isinstance(noll, (bool, np.bool_)) or not isinstance(noll, Integral):
        raise ValueError('noll must be a 1-based integer.')
    noll = int(noll)
    if noll < 1:
        raise ValueError('noll must be a 1-based integer.')
    return (math.isqrt(8 * noll - 7) - 1) // 2


def _normalize_weight_dict(weights, field_name, minimum_mode):
    """Validate and unit-normalize one side of a mode-weight prior."""
    if not isinstance(weights, dict):
        raise ValueError(f'{field_name} must be a dictionary.')

    validated = {}
    for mode, weight in weights.items():
        if isinstance(mode, (bool, np.bool_)) or not isinstance(mode, Integral):
            raise ValueError(f'{field_name} mode keys must be 1-based integers.')
        mode = int(mode)
        if mode < minimum_mode:
            raise ValueError(
                f'{field_name} mode keys must be >= {minimum_mode}.'
            )
        if isinstance(weight, (bool, np.bool_)):
            raise ValueError(f'{field_name} weights must be finite and non-negative.')
        try:
            weight = float(weight)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f'{field_name} weights must be finite and non-negative.'
            ) from exc
        if not np.isfinite(weight) or weight < 0.0:
            raise ValueError(f'{field_name} weights must be finite and non-negative.')
        validated[mode] = weight

    if not validated:
        return {}
    norm = float(np.linalg.norm(list(validated.values())))
    if norm == 0.0:
        raise ValueError(f'{field_name} must have a positive sum of squared weights.')
    return {mode: validated[mode] / norm for mode in sorted(validated)}


@dataclass(frozen=True)
class ModeWeightPrior:
    """Shape-only prior in sequential orthonormal aperture bases.

    Each non-empty weight dictionary is normalized independently to unit
    sum of squared weights. Consequently, absolute input scales are
    discarded and the prior describes only the relative mode mix. The
    weights scale coefficients in sequentially QR-orthonormalized aperture
    bases, not raw HCIPy Noll modes. Exact-RMS conditioning of every draw
    means the realized marginal variance fractions differ from the squared
    table weights; the measured difference reaches 15% per mode for the
    committed JWST drift table.

    Parameters
    ----------
    name : `str`
        Non-empty prior name.
    global_weights : `dict` [`int`, `float`]
        Global Zernike Noll indices and non-negative weights. Global modes
        1--3 are excluded by design.
    segment_weights : `dict` [`int`, `float`]
        Segment hexike Noll indices and non-negative weights.
    segment_variance_fraction : `float`
        Fraction of combined-draw coefficient variance assigned to the
        segment side.
    metadata : `dict`, optional
        Free-form provenance metadata. It is stored but never interpreted.
    """

    name: str
    global_weights: dict[int, float]
    segment_weights: dict[int, float]
    segment_variance_fraction: float
    metadata: dict = field(default_factory=dict)

    def __post_init__(self):
        """Validate the prior and normalize both populated weight sides."""
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError('name must be a non-empty string.')
        global_weights = _normalize_weight_dict(
            self.global_weights, 'global_weights', 4
        )
        segment_weights = _normalize_weight_dict(
            self.segment_weights, 'segment_weights', 1
        )
        if not global_weights and not segment_weights:
            raise ValueError(
                'global_weights and segment_weights must not both be empty.'
            )
        if isinstance(self.segment_variance_fraction, (bool, np.bool_)):
            raise ValueError(
                'segment_variance_fraction must be finite and in [0, 1].'
            )
        try:
            fraction = float(self.segment_variance_fraction)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                'segment_variance_fraction must be finite and in [0, 1].'
            ) from exc
        if not np.isfinite(fraction) or not 0.0 <= fraction <= 1.0:
            raise ValueError(
                'segment_variance_fraction must be finite and in [0, 1].'
            )
        if not isinstance(self.metadata, dict):
            raise ValueError('metadata must be a dictionary.')

        object.__setattr__(self, 'global_weights', global_weights)
        object.__setattr__(self, 'segment_weights', segment_weights)
        object.__setattr__(self, 'segment_variance_fraction', fraction)
        object.__setattr__(self, 'metadata', dict(self.metadata))


def _validate_mode_range(mode_range, field_name, minimum_mode):
    """Validate an inclusive mode range and return integer bounds."""
    if mode_range is None:
        return None
    if not isinstance(mode_range, (tuple, list)) or len(mode_range) != 2:
        raise ValueError(f'{field_name} must be an inclusive (lo, hi) pair.')
    lo, hi = mode_range
    if any(
        isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral)
        for value in (lo, hi)
    ):
        raise ValueError(f'{field_name} bounds must be integers.')
    lo = int(lo)
    hi = int(hi)
    if lo < minimum_mode:
        raise ValueError(f'{field_name} lower bound must be >= {minimum_mode}.')
    if hi < lo:
        raise ValueError(f'{field_name} upper bound must be >= its lower bound.')
    return lo, hi


def make_power_law_prior(alpha, global_mode_range=(4, 55),
                         segment_mode_range=(1, 10),
                         segment_variance_fraction=0.5, name=None):
    """Construct a radial-order power-law mode-weight prior.

    Global weights follow ``n**(-alpha)``. Segment weights follow
    ``(n + 1)**(-alpha)``; the added one is a placeholder convention that
    avoids a singular weight for segment piston at radial order zero.

    Parameters
    ----------
    alpha : `float`
        Finite, non-negative power-law index.
    global_mode_range : (`int`, `int`) or `None`, optional
        Inclusive global-Zernike Noll range, or `None` to omit that side.
    segment_mode_range : (`int`, `int`) or `None`, optional
        Inclusive segment-hexike Noll range, or `None` to omit that side.
    segment_variance_fraction : `float`, optional
        Fraction of combined-draw variance assigned to segment modes.
    name : `str`, optional
        Prior name. Defaults to ``power_law_alpha_{alpha:g}``.

    Returns
    -------
    prior : `ModeWeightPrior`
        Normalized shape-only mode-weight prior.

    Raises
    ------
    ValueError
        Raised for invalid alpha, ranges, or variance fraction.
    """
    if isinstance(alpha, (bool, np.bool_)):
        raise ValueError('alpha must be a finite non-negative number.')
    try:
        alpha = float(alpha)
    except (TypeError, ValueError) as exc:
        raise ValueError('alpha must be a finite non-negative number.') from exc
    if not np.isfinite(alpha) or alpha < 0.0:
        raise ValueError('alpha must be a finite non-negative number.')

    global_range = _validate_mode_range(global_mode_range,
                                        'global_mode_range', 4)
    segment_range = _validate_mode_range(segment_mode_range,
                                         'segment_mode_range', 1)
    if global_range is None and segment_range is None:
        raise ValueError(
            'global_mode_range and segment_mode_range must not both be None.'
        )

    global_weights = {}
    if global_range is not None:
        global_weights = {
            mode: noll_to_radial_order(mode)**(-alpha)
            for mode in range(global_range[0], global_range[1] + 1)
        }
    segment_weights = {}
    if segment_range is not None:
        segment_weights = {
            mode: (noll_to_radial_order(mode) + 1)**(-alpha)
            for mode in range(segment_range[0], segment_range[1] + 1)
        }
    if name is None:
        name = f'power_law_alpha_{alpha:g}'
    metadata = {
        'kind': 'power_law',
        'alpha': alpha,
        'global_mode_range': global_range,
        'segment_mode_range': segment_range,
    }
    return ModeWeightPrior(
        name=name,
        global_weights=global_weights,
        segment_weights=segment_weights,
        segment_variance_fraction=segment_variance_fraction,
        metadata=metadata,
    )


def load_mode_weight_prior(path):
    """Load an orthonormal-aperture mode-weight prior from a YAML table.

    Table weights are directional scales in sequentially
    QR-orthonormalized aperture bases. They are not realized marginal
    variance fractions: exact-RMS conditioning changes the second moments
    by up to 15% per mode for the committed JWST drift table.

    Parameters
    ----------
    path : path-like
        YAML table path.

    Returns
    -------
    prior : `ModeWeightPrior`
        Validated and normalized mode-weight prior.

    Raises
    ------
    ValueError
        Raised if the document structure or prior values are invalid.
    """
    with open(path, 'r', encoding='utf-8') as stream:
        try:
            document = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            raise ValueError(f'Invalid mode-weight prior YAML: {exc}') from exc
    if not isinstance(document, dict):
        raise ValueError('Mode-weight prior document must be a mapping.')

    allowed = {
        'name',
        'segment_variance_fraction',
        'global_weights',
        'segment_weights',
        'metadata',
    }
    unknown = sorted(set(document) - allowed, key=str)
    if unknown:
        raise ValueError(f'Unknown top-level key: {unknown[0]}')
    for required in ('name', 'segment_variance_fraction'):
        if required not in document:
            raise ValueError(f'Missing required field: {required}')
    if 'global_weights' not in document and 'segment_weights' not in document:
        raise ValueError(
            'Missing required field: global_weights or segment_weights.'
        )
    return ModeWeightPrior(
        name=document['name'],
        global_weights=document.get('global_weights', {}),
        segment_weights=document.get('segment_weights', {}),
        segment_variance_fraction=document['segment_variance_fraction'],
        metadata=document.get('metadata', {}),
    )


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


def _validate_weighted_target(target_rms_nm, family_name):
    """Validate a weighted-family target RMS."""
    if isinstance(target_rms_nm, (bool, np.bool_)):
        raise ValueError(
            f'{family_name} target RMS must be a finite non-negative number.'
        )
    try:
        target = float(target_rms_nm)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f'{family_name} target RMS must be a finite non-negative number.'
        ) from exc
    if not np.isfinite(target) or target < 0.0:
        raise ValueError(
            f'{family_name} target RMS must be a finite non-negative number.'
        )
    return target


def draw_weighted_global_zernike_family(rng, prior, target_rms_nm):
    """Draw orthonormal-aperture global coefficients at a target norm.

    This function performs coefficient-space normalization only. The caller
    must convert the result with :class:`ApertureBasisTransform` before raw
    HCIPy application. Exact-RMS conditioning changes the realized marginal
    variance fractions from the squared directional weights by up to 15%
    per mode for the committed JWST drift table. Use
    :func:`realize_weighted_draw` to convert and realize the exact
    piston-removed aperture wavefront RMS on a telescope pupil.

    Parameters
    ----------
    rng : `numpy.random.Generator`
        Random generator supplying the draw.
    prior : `ModeWeightPrior`
        Shape-only mode-weight prior with a populated global side.
    target_rms_nm : `float`
        Target coefficient-vector norm in nanometers OPD.

    Returns
    -------
    global_zernikes : `dict`
        Ascending mapping from global Noll mode to coefficient in
        nanometers. Empty when the target is zero.

    Raises
    ------
    ValueError
        Raised for a bad target or an empty global side.
    """
    target = _validate_weighted_target(
        target_rms_nm, 'weighted global Zernike family'
    )
    if target == 0.0:
        return {}
    if not prior.global_weights:
        raise ValueError('prior.global_weights must not be empty.')

    modes = sorted(prior.global_weights)
    raw = rng.standard_normal(len(modes))
    weights = np.array([prior.global_weights[mode] for mode in modes])
    coeffs = _normalize_vector(raw * weights, target)
    return {mode: float(coeffs[index]) for index, mode in enumerate(modes)}


def draw_weighted_segment_hexike_family(
    rng, segments, prior, target_rms_nm
):
    """Draw orthonormal-aperture segment modes at a target coefficient RMS.

    When mode 1 is present, only its across-segment mean is removed. Common
    segment tip or tilt is retained because it is a physical sawtooth rather
    than an unobservable global ramp. This function performs coefficient-
    space normalization only; the caller must convert with
    :class:`ApertureBasisTransform`. Exact-RMS conditioning changes realized
    marginal variance fractions from the squared directional weights by up
    to 15% per mode for the committed JWST drift table. Use
    :func:`realize_weighted_draw` for conversion and exact aperture RMS.

    Parameters
    ----------
    rng : `numpy.random.Generator`
        Random generator supplying the draw.
    segments : sequence of `int`
        Segment identifiers in the desired draw order.
    prior : `ModeWeightPrior`
        Shape-only mode-weight prior with a populated segment side.
    target_rms_nm : `float`
        Target equal-area segment coefficient RMS in nanometers OPD.

    Returns
    -------
    segment_hexikes : `dict`
        Mapping from segment to ascending ``{mode: coefficient_nm}``.
        Empty when the target is zero.

    Raises
    ------
    ValueError
        Raised for a bad target, empty segment list or segment side, or a
        mode-1 draw with fewer than two segments.
    """
    target = _validate_weighted_target(
        target_rms_nm, 'weighted segment hexike family'
    )
    segment_ids = [int(segment) for segment in segments]
    if not segment_ids:
        raise ValueError('weighted segment hexike family segments must not be empty.')
    if target == 0.0:
        return {}
    if not prior.segment_weights:
        raise ValueError('prior.segment_weights must not be empty.')

    modes = sorted(prior.segment_weights)
    if 1 in modes and len(segment_ids) < 2:
        raise ValueError('Segment mode 1 requires at least two segments.')
    raw = rng.standard_normal((len(segment_ids), len(modes)))
    weights = np.array([prior.segment_weights[mode] for mode in modes])
    weighted = raw * weights[np.newaxis, :]
    if 1 in modes:
        piston_column = modes.index(1)
        weighted[:, piston_column] -= np.mean(weighted[:, piston_column])
    scaled = _normalize_vector(
        weighted.ravel(), target * np.sqrt(len(segment_ids))
    )
    matrix = scaled.reshape((len(segment_ids), len(modes)))
    return {
        segment: {
            mode: float(matrix[segment_index, mode_index])
            for mode_index, mode in enumerate(modes)
        }
        for segment_index, segment in enumerate(segment_ids)
    }


def draw_weighted_combined_family(rng, segments, prior, target_rms_nm):
    """Draw an orthonormal-aperture segment-plus-global family.

    The segment side is drawn first. A side with exactly zero variance
    budget is skipped without consuming random numbers. This function only
    normalizes orthonormal-basis coefficients. Exact-RMS conditioning makes
    realized marginal variance fractions differ from squared directional
    weights by up to 15% per mode for the committed JWST drift table. For
    exact combined amplitude, convert and jointly renormalize with::

        realize_weighted_draw(
            telescope_data, basis_transform, target,
            segment_coefficients=segment_hexikes,
            global_coefficients=global_zernikes)

    Parameters
    ----------
    rng : `numpy.random.Generator`
        Random generator supplying the draw.
    segments : sequence of `int`
        Segment identifiers in the desired draw order.
    prior : `ModeWeightPrior`
        Shape-only mode-weight prior and variance split.
    target_rms_nm : `float`
        Total coefficient-space RMS budget in nanometers OPD.

    Returns
    -------
    segment_hexikes : `dict`
        Segment-hexike coefficient mapping, or empty when skipped.
    global_zernikes : `dict`
        Global-Zernike coefficient mapping, or empty when skipped.

    Raises
    ------
    ValueError
        Raised for a bad target or a nonzero side with no weights.
    """
    target = _validate_weighted_target(target_rms_nm,
                                       'weighted combined family')
    if target == 0.0:
        return {}, {}

    fraction = prior.segment_variance_fraction
    segment_budget = target * np.sqrt(fraction)
    global_budget = target * np.sqrt(1.0 - fraction)
    segment_hexikes = {}
    global_zernikes = {}
    if segment_budget != 0.0:
        segment_hexikes = draw_weighted_segment_hexike_family(
            rng, segments, prior, segment_budget
        )
    if global_budget != 0.0:
        global_zernikes = draw_weighted_global_zernike_family(
            rng, prior, global_budget
        )
    return segment_hexikes, global_zernikes


def realize_weighted_draw(telescope_data, basis_transform, target_rms_nm,
                          segment_coefficients=None,
                          global_coefficients=None):
    """Convert one weighted draw to raw HCIPy coefficients and normalize it.

    This is the canonical consumption path for all weighted families:
    orthonormal-basis draw, cached change of basis, then exact physical RMS
    renormalization. Reuse one transform for every draw made with the same
    telescope configuration and mode ranges.

    Parameters
    ----------
    telescope_data : `dict`
        Pupil-side telescope data used to construct ``basis_transform``.
    basis_transform : `ApertureBasisTransform`
        Cached transform for exactly the modes present in the draw.
    target_rms_nm : `float`
        Target piston-removed aperture wavefront RMS in nanometers OPD.
    segment_coefficients : `dict`, optional
        Orthonormal-basis per-segment coefficient dictionaries.
    global_coefficients : `dict`, optional
        Orthonormal-basis global coefficient dictionary.

    Returns
    -------
    segment_raw : `dict`
        Renormalized raw per-segment HCIPy coefficients.
    global_raw : `dict`
        Renormalized raw global HCIPy coefficients.
    """
    if not isinstance(basis_transform, ApertureBasisTransform):
        raise TypeError('basis_transform must be an ApertureBasisTransform.')
    segment_raw, global_raw = basis_transform.to_raw(
        segment_coefficients=segment_coefficients,
        global_coefficients=global_coefficients,
    )
    return renormalize_to_aperture_rms(
        telescope_data,
        target_rms_nm,
        segment_hexikes=segment_raw,
        global_zernikes=global_raw,
    )


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
