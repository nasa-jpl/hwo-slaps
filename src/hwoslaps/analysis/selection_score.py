"""Frozen D-F4 selection statistics, score, ranking and rank stability.

The pre-registered rule that turns a Stage 0 pool of no-subhalo
observations into the selected follow-up tier. The signed definition is
``scratch/q1_observing_conditions/selection_rule.md`` (v2, 2026-08-23);
this module is its executable form and the two must agree exactly.

Every function here is pure over declared array inputs. Nothing imports
or touches an active Fisher object, an engine dataset or a
configuration: the caller supplies the electron maps, the noise scalars
and the grid, and the statistics follow.

Definitions
-----------
Signal and noise are electrons and electrons squared. The blank-pixel
variance is ``B = (sky + dark) * t + read_noise ** 2`` in ``e^2``, the
source-free limit of the engine noise map, and the expected per-pixel
variance of a source map is ``sigma_i^2 = s_i + B``. Both match
``scripts/derive_hwo_eac1_hri_reference.py``.

============================  ================================
Quantity                      Unit
============================  ================================
``s`` (signal)                ``e-`` in the exposure
``sigma^2``, ``B``            ``e-^2``
``S`` (arc S/N)               dimensionless
``|grad s|``                  ``e- arcsec^-1``
``G`` (gradient power)        ``arcsec^-2``
``theta_res``, angles         arcsec
``C`` (complexity)            dimensionless
============================  ================================

The arc signal-to-noise follows the engine convention exactly,

``S = sqrt( sum_i s_i^2 / sigma_i^2 )``,

which recovers 303.94 for the committed reference. The gradient power
is

``G = sum_i |grad s|_i^2 / sigma_i^2``,

with central-difference gradients divided by the pixel scale, so the
gradient is an angular derivative in ``e- arcsec^-1`` rather than a bare
neighbouring-pixel difference and ``G`` carries ``arcsec^-2``. Both sums
run over the same declared aperture.

The complexity statistic is

``C = theta_res^2 * G / S^2``,   ``theta_res = lambda / D``,

which is dimensionless and, in the background-dominated limit where
``sigma`` does not follow the source, invariant under a uniform flux
rescaling of the arc: ``S`` scales linearly and ``G`` quadratically, so
the brightness cancels. That is the whole point of the statistic. It
removes the brightness double-counting the earlier ``z(log S) +
z(log G)`` score carried, because ``G`` scales approximately as ``S^2``
under flux scaling.

The frozen score is

``score = z(log S) + z(log C)``,

with ``z`` standardizing over the post-floor-cut Stage 0 pool that is
actually being ranked (population standard deviation, ``ddof=0``). The
pre-registered comparison also evaluates the ``s_only`` score
``z(log S)`` and, post-campaign only, the oracle ranking by measured
sensitivity.

Notes
-----
Deterministic rules, each covered by a unit test:

- Floor cuts are strict: ``theta_E > 0.5`` arcsec and ``S > 20``. A
  member exactly on a floor fails it.
- Standardization is the population z-score. A pool with exactly zero
  spread standardizes to all zeros rather than dividing by zero.
- ``log S`` and ``log C`` require strictly positive finite inputs. A
  ranked member with zero or non-finite ``S`` or ``C`` raises: a flat
  arc carries no complexity and the score is undefined there, so the
  pool is rejected loudly instead of being silently reordered.
- Ranking ties break on the ascending sha256 hex digest of the system
  id, which is independent of pool membership, pool order and floating
  point.
- Rank stability compares two rankings by Spearman correlation of their
  positions, top-K Jaccard, and the recovered fraction of an oracle
  top-K.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from typing import Iterable, Sequence

import numpy as np


__all__ = [
    "APERTURE_THETA_E_MULTIPLE",
    "FLOOR_ARC_SNR",
    "FLOOR_THETA_E_ARCSEC",
    "GOLDEN_TIER_SIZE",
    "RADIAN_TO_ARCSEC",
    "SCORE_VARIANTS",
    "SELECTED_TIER_SIZE",
    "SelectionResult",
    "aperture_mask",
    "apply_floor_cuts",
    "arc_snr",
    "blank_variance_e2",
    "complexity",
    "diffraction_scale_arcsec",
    "expected_variance_e2",
    "gradient_power",
    "oracle_recovered_fraction",
    "rank_by_score",
    "rank_by_sensitivity",
    "rank_pool",
    "ranking_positions",
    "selection_scores",
    "spearman_rank_correlation",
    "standardize",
    "top_k_jaccard",
]


FLOOR_THETA_E_ARCSEC = 0.5
"""Einstein-radius floor cut in arcseconds (`float`).

Collett 2015 as imposed by O'Riordan et al. 2023: ``theta_E > 0.5``.
"""

FLOOR_ARC_SNR = 20.0
"""Integrated arc signal-to-noise floor cut (`float`).

Collett 2015 as imposed by O'Riordan et al. 2023: ``S/N > 20``.
"""

APERTURE_THETA_E_MULTIPLE = 2.0
"""Aperture radius in units of ``theta_E`` (`float`), the D-F7 ruling."""

SELECTED_TIER_SIZE = 12
"""Size of the selected follow-up tier (`int`)."""

GOLDEN_TIER_SIZE = 5
"""Size of the golden subset drawn from the selected tier (`int`)."""

RADIAN_TO_ARCSEC = 180.0 * 3600.0 / math.pi
"""Arcseconds in one radian (`float`)."""

SCORE_VARIANTS = ("s_only", "s_plus_c")
"""Pre-registered operational score variants (`tuple` of `str`).

``s_only`` is ``z(log S)`` and ``s_plus_c`` is the frozen
``z(log S) + z(log C)``. The third pre-registered curve, the oracle
ranking by measured sensitivity, is not a score: see
`rank_by_sensitivity`.
"""


@dataclass(frozen=True)
class SelectionResult:
    """One pool ranked under one score variant.

    Attributes
    ----------
    variant : `str`
        Score variant, a member of `SCORE_VARIANTS`.
    system_ids : `tuple` [`str`]
        Every input system id, in input order.
    passed_floor : `tuple` [`bool`]
        Floor-cut outcome aligned with ``system_ids``.
    survivor_ids : `tuple` [`str`]
        Post-cut ids in input order, the standardization pool.
    scores : `tuple` [`float`]
        Scores aligned with ``survivor_ids``.
    ranking : `tuple` [`str`]
        Survivor ids best first, ties broken by id digest.
    selected_ids : `tuple` [`str`]
        Leading ``selected_size`` entries of ``ranking``.
    golden_ids : `tuple` [`str`]
        Leading ``golden_size`` entries of ``ranking``.
    """

    variant: str
    system_ids: tuple[str, ...]
    passed_floor: tuple[bool, ...]
    survivor_ids: tuple[str, ...]
    scores: tuple[float, ...]
    ranking: tuple[str, ...]
    selected_ids: tuple[str, ...]
    golden_ids: tuple[str, ...]


def _require_finite_array(values, name: str, ndim: int | None = None) -> np.ndarray:
    """Return one finite float array or raise naming the offender."""
    array = np.asarray(values, dtype=float)
    if ndim is not None and array.ndim != ndim:
        raise ValueError(f"{name} must be {ndim}-dimensional, got shape {array.shape}.")
    if array.size == 0:
        raise ValueError(f"{name} must not be empty.")
    if not np.all(np.isfinite(array)):
        bad = int(np.count_nonzero(~np.isfinite(array)))
        raise ValueError(f"{name} must be finite; {bad} of {array.size} entries are not.")
    return array


def _require_positive(value, name: str) -> float:
    """Return one strictly positive finite scalar or raise."""
    if isinstance(value, bool) or not isinstance(value, (int, float, np.integer, np.floating)):
        raise ValueError(f"{name} must be a real number, got {value!r}.")
    scalar = float(value)
    if not math.isfinite(scalar) or scalar <= 0.0:
        raise ValueError(f"{name} must be positive and finite, got {scalar}.")
    return scalar


def _require_non_negative(value, name: str) -> float:
    """Return one non-negative finite scalar or raise."""
    if isinstance(value, bool) or not isinstance(value, (int, float, np.integer, np.floating)):
        raise ValueError(f"{name} must be a real number, got {value!r}.")
    scalar = float(value)
    if not math.isfinite(scalar) or scalar < 0.0:
        raise ValueError(f"{name} must be non-negative and finite, got {scalar}.")
    return scalar


def _resolve_mask(mask, shape: tuple[int, ...], name: str) -> np.ndarray:
    """Return a boolean selection mask covering at least one pixel."""
    if mask is None:
        return np.ones(shape, dtype=bool)
    resolved = np.asarray(mask)
    if resolved.dtype != np.bool_:
        raise ValueError(f"{name} must be a boolean array, got dtype {resolved.dtype}.")
    if resolved.shape != shape:
        raise ValueError(f"{name} shape {resolved.shape} does not match the image shape {shape}.")
    if not np.any(resolved):
        raise ValueError(f"{name} selects no pixels.")
    return resolved


def _require_ids(system_ids: Sequence[str], name: str) -> tuple[str, ...]:
    """Return one tuple of unique non-empty string ids or raise."""
    ids = tuple(system_ids)
    for entry in ids:
        if not isinstance(entry, str) or entry == "":
            raise ValueError(f"{name} entries must be non-empty strings, got {entry!r}.")
    if len(set(ids)) != len(ids):
        duplicates = sorted({entry for entry in ids if ids.count(entry) > 1})
        raise ValueError(f"{name} must be unique; duplicated: {', '.join(duplicates)}.")
    return ids


def _id_digest(system_id: str) -> str:
    """Return the sha256 hex digest that breaks ranking ties."""
    return hashlib.sha256(system_id.encode("utf-8")).hexdigest()


def blank_variance_e2(sky_background_e_s, dark_current_e_s, read_noise_e, exposure_time_s):
    """Return the blank-pixel variance of one observing setup.

    A pixel carrying no source light collects sky and dark electrons and
    is read once, so its variance is
    ``(sky + dark) * t + read_noise ** 2`` in electrons squared. The read
    noise is the effective combined-image value, because the engine noise
    model applies exactly one squared read-noise term.

    Parameters
    ----------
    sky_background_e_s : `float`
        Sky background in electrons per pixel per second.
    dark_current_e_s : `float`
        Dark current in electrons per pixel per second.
    read_noise_e : `float`
        Effective read noise in electrons per pixel.
    exposure_time_s : `float`
        Exposure time in seconds.

    Returns
    -------
    variance : `float`
        Blank-pixel variance in electrons squared.

    Raises
    ------
    ValueError
        Raised for a non-finite or negative input, or when the setup
        leaves a blank pixel with zero variance.
    """
    sky = _require_non_negative(sky_background_e_s, "sky_background_e_s")
    dark = _require_non_negative(dark_current_e_s, "dark_current_e_s")
    read_noise = _require_non_negative(read_noise_e, "read_noise_e")
    exposure = _require_positive(exposure_time_s, "exposure_time_s")
    variance = (sky + dark) * exposure + read_noise ** 2
    if variance <= 0.0:
        raise ValueError(
            "The declared detector leaves a blank pixel with zero variance, "
            "so no arc signal-to-noise is defined."
        )
    return float(variance)


def expected_variance_e2(source_electrons, blank_variance):
    """Return the expected per-pixel variance of one electron map.

    The expected variance is ``sigma_i^2 = s_i + B``: source shot noise
    on top of the blank-pixel floor, with no noise realization anywhere.

    Parameters
    ----------
    source_electrons : array-like
        Source electrons per pixel in the exposure. Values must be
        non-negative for the expected-variance reading to hold.
    blank_variance : `float`
        Blank-pixel variance in electrons squared.

    Returns
    -------
    variance : `numpy.ndarray`
        Per-pixel variance in electrons squared.

    Raises
    ------
    ValueError
        Raised for non-finite input, negative source electrons, or a
        non-positive blank variance.
    """
    electrons = _require_finite_array(source_electrons, "source_electrons")
    variance = _require_positive(blank_variance, "blank_variance")
    if np.any(electrons < 0.0):
        raise ValueError(
            "source_electrons must be non-negative for an expected-variance "
            f"map; minimum is {float(np.min(electrons))}."
        )
    return electrons + variance


def arc_snr(signal_e, variance_e2, mask=None):
    """Integrate one electron map into an arc signal-to-noise.

    The engine reports a per-pixel source signal-to-noise of
    ``s_p / sigma_p``, so the integrated value over the aperture is
    ``S = sqrt( sum_p s_p^2 / sigma_p^2 )``. Passing the noiseless source
    map with its expected variance reproduces the committed reference
    value of 303.94; passing a background-subtracted realization with an
    observed-side variance map gives the noisy estimator of the same
    statistic.

    Parameters
    ----------
    signal_e : array-like
        Signal electrons per pixel. May be negative on the noisy path.
    variance_e2 : array-like
        Per-pixel variance in electrons squared, strictly positive.
    mask : array-like, optional
        Boolean aperture selecting the summed pixels. Defaults to every
        pixel.

    Returns
    -------
    arc_snr : `float`
        Integrated signal-to-noise over the aperture, dimensionless.

    Raises
    ------
    ValueError
        Raised on shape mismatch, non-finite entries, a non-positive
        variance, or an empty aperture.
    """
    signal = _require_finite_array(signal_e, "signal_e")
    variance = _require_finite_array(variance_e2, "variance_e2")
    if signal.shape != variance.shape:
        raise ValueError(
            f"signal_e shape {signal.shape} does not match variance_e2 shape {variance.shape}."
        )
    if np.any(variance <= 0.0):
        raise ValueError(f"variance_e2 must be positive; minimum is {float(np.min(variance))}.")
    selection = _resolve_mask(mask, signal.shape, "mask")
    return float(np.sqrt(np.sum(signal[selection] ** 2 / variance[selection])))


def gradient_power(signal_e, variance_e2, pixel_scale_arcsec, mask=None):
    """Integrate one electron map into a noise-weighted gradient power.

    ``G = sum_i (|grad s|_i^2) / sigma_i^2`` over the aperture, with
    ``grad s`` in electrons per arcsecond. The gradient is the
    second-order central difference in the array interior and the
    first-order one-sided difference on the array border, exactly
    `numpy.gradient` with the pixel scale as the spacing on both axes.
    Dividing by the pixel scale is the whole angular normalization: it
    makes the gradient an angular derivative in ``e- arcsec^-1``, so the
    statistic is stated on a declared angular scale instead of on the
    sampling of the grid it happened to be measured on. Holding the
    pixel values fixed and halving the pixel scale therefore quadruples
    ``G``, which is the intended behaviour of a per-arcsecond gradient.

    Parameters
    ----------
    signal_e : array-like
        Two-dimensional signal electrons per pixel.
    variance_e2 : array-like
        Per-pixel variance in electrons squared, strictly positive.
    pixel_scale_arcsec : `float`
        Pixel scale in arcseconds per pixel, identical on both axes.
    mask : array-like, optional
        Boolean aperture selecting the summed pixels. Defaults to every
        pixel.

    Returns
    -------
    gradient_power : `float`
        Noise-weighted gradient power in inverse square arcseconds.

    Raises
    ------
    ValueError
        Raised on shape mismatch, non-finite entries, a non-positive
        variance or pixel scale, an image thinner than three pixels on
        either axis, or an empty aperture.
    """
    signal = _require_finite_array(signal_e, "signal_e", ndim=2)
    variance = _require_finite_array(variance_e2, "variance_e2", ndim=2)
    if signal.shape != variance.shape:
        raise ValueError(
            f"signal_e shape {signal.shape} does not match variance_e2 shape {variance.shape}."
        )
    if min(signal.shape) < 3:
        raise ValueError(
            "signal_e must be at least three pixels on both axes for a central "
            f"difference, got shape {signal.shape}."
        )
    if np.any(variance <= 0.0):
        raise ValueError(f"variance_e2 must be positive; minimum is {float(np.min(variance))}.")
    scale = _require_positive(pixel_scale_arcsec, "pixel_scale_arcsec")
    selection = _resolve_mask(mask, signal.shape, "mask")
    grad_y, grad_x = np.gradient(signal, scale, scale)
    power = grad_y ** 2 + grad_x ** 2
    return float(np.sum(power[selection] / variance[selection]))


def diffraction_scale_arcsec(wavelength_m, diameter_m):
    """Return the diffraction scale ``theta_res = lambda / D``.

    Parameters
    ----------
    wavelength_m : `float`
        Observing wavelength in metres.
    diameter_m : `float`
        Telescope aperture diameter in metres.

    Returns
    -------
    theta_res : `float`
        Diffraction scale in arcseconds.

    Raises
    ------
    ValueError
        Raised for a non-positive or non-finite input.
    """
    wavelength = _require_positive(wavelength_m, "wavelength_m")
    diameter = _require_positive(diameter_m, "diameter_m")
    return float(wavelength / diameter * RADIAN_TO_ARCSEC)


def complexity(gradient_power_value, arc_snr_value, theta_res_arcsec):
    """Return the brightness-normalized complexity statistic.

    ``C = theta_res^2 * G / S^2``. The angular factor makes ``C``
    dimensionless and puts it on the resolution element the instrument
    actually delivers, and dividing by ``S^2`` removes the brightness
    that ``G`` and ``S`` otherwise both carry.

    Parameters
    ----------
    gradient_power_value : `float`
        Noise-weighted gradient power in inverse square arcseconds.
    arc_snr_value : `float`
        Integrated arc signal-to-noise.
    theta_res_arcsec : `float`
        Diffraction scale in arcseconds.

    Returns
    -------
    complexity : `float`
        Dimensionless complexity statistic.

    Raises
    ------
    ValueError
        Raised for a non-positive or non-finite input.
    """
    power = _require_positive(gradient_power_value, "gradient_power_value")
    snr = _require_positive(arc_snr_value, "arc_snr_value")
    theta_res = _require_positive(theta_res_arcsec, "theta_res_arcsec")
    return float(theta_res ** 2 * power / snr ** 2)


def aperture_mask(y_arcsec, x_arcsec, radius_arcsec, centre_arcsec=(0.0, 0.0)):
    """Select the pixels inside one circular aperture.

    Coordinates come from the grid the scene was ray-traced on, so the
    aperture uses the engine's own convention rather than a reconstructed
    one. The radius is a closed interval, matching the engine's fixed
    annulus mask.

    Parameters
    ----------
    y_arcsec, x_arcsec : array-like
        Native-shaped pixel coordinates in arcseconds.
    radius_arcsec : `float`
        Aperture radius in arcseconds, normally
        ``APERTURE_THETA_E_MULTIPLE * theta_E``.
    centre_arcsec : `tuple` [`float`], optional
        Aperture centre as ``(y, x)`` in arcseconds. Defaults to the
        origin, which is the lens centre in the production convention.

    Returns
    -------
    mask : `numpy.ndarray`
        Boolean array, true inside the aperture.

    Raises
    ------
    ValueError
        Raised on shape mismatch, non-finite coordinates, a non-positive
        radius, a malformed centre, or an aperture holding no pixels.
    """
    y_coords = _require_finite_array(y_arcsec, "y_arcsec")
    x_coords = _require_finite_array(x_arcsec, "x_arcsec")
    if y_coords.shape != x_coords.shape:
        raise ValueError(
            f"y_arcsec shape {y_coords.shape} does not match x_arcsec shape {x_coords.shape}."
        )
    radius = _require_positive(radius_arcsec, "radius_arcsec")
    centre = tuple(centre_arcsec)
    if len(centre) != 2:
        raise ValueError(f"centre_arcsec must hold two entries, got {centre_arcsec!r}.")
    centre_y = float(centre[0])
    centre_x = float(centre[1])
    if not (math.isfinite(centre_y) and math.isfinite(centre_x)):
        raise ValueError(f"centre_arcsec must be finite, got {centre_arcsec!r}.")
    mask = np.hypot(y_coords - centre_y, x_coords - centre_x) <= radius
    if not np.any(mask):
        raise ValueError(
            f"The {radius} arcsec aperture at {centre} holds no pixels of the declared grid."
        )
    return mask


def apply_floor_cuts(theta_e_arcsec, arc_snr_values):
    """Apply the Collett 2015 floor cuts to one pool.

    ``theta_E > 0.5`` arcsec and ``S > 20``, both strict, so a member
    exactly on a floor fails it.

    Parameters
    ----------
    theta_e_arcsec : array-like
        Einstein radii in arcseconds, one per pool member.
    arc_snr_values : array-like
        Integrated arc signal-to-noise, one per pool member.

    Returns
    -------
    passed : `numpy.ndarray`
        Boolean array, true for members surviving both cuts.

    Raises
    ------
    ValueError
        Raised on length mismatch or non-finite input.
    """
    theta_e = _require_finite_array(theta_e_arcsec, "theta_e_arcsec", ndim=1)
    snr = _require_finite_array(arc_snr_values, "arc_snr_values", ndim=1)
    if theta_e.shape != snr.shape:
        raise ValueError(
            f"theta_e_arcsec holds {theta_e.size} entries and arc_snr_values holds {snr.size}."
        )
    return (theta_e > FLOOR_THETA_E_ARCSEC) & (snr > FLOOR_ARC_SNR)


def standardize(values):
    """Standardize one pool to zero mean and unit spread.

    The population standard deviation (``ddof=0``) is used, so the
    z-scores depend only on the pool and not on a sample correction. A
    pool with exactly zero spread standardizes to zeros: every member is
    identical, so no member can rank above another on this statistic.

    Parameters
    ----------
    values : array-like
        One statistic over the pool.

    Returns
    -------
    z : `numpy.ndarray`
        Standardized statistic.

    Raises
    ------
    ValueError
        Raised for an empty or non-finite input.
    """
    array = _require_finite_array(values, "values", ndim=1)
    spread = float(np.std(array))
    if spread == 0.0:
        return np.zeros_like(array)
    return (array - float(np.mean(array))) / spread


def _require_log_ready(values, name: str) -> np.ndarray:
    """Return one array whose logarithm is defined, or raise loudly."""
    array = np.asarray(values, dtype=float)
    if array.ndim != 1 or array.size == 0:
        raise ValueError(f"{name} must be a non-empty one-dimensional array.")
    bad = ~(np.isfinite(array) & (array > 0.0))
    if np.any(bad):
        offenders = ", ".join(
            f"index {index}: {array[index]}" for index in np.flatnonzero(bad)[:8]
        )
        raise ValueError(
            f"{name} must be strictly positive and finite for the log score; {offenders}."
        )
    return array


def selection_scores(arc_snr_values, complexity_values, variant="s_plus_c"):
    """Score one post-cut pool under a pre-registered variant.

    ``s_plus_c`` is the frozen score ``z(log S) + z(log C)`` and
    ``s_only`` is the ``z(log S)`` comparison curve. Standardization runs
    over exactly the pool passed in, which is the post-floor-cut pool
    being ranked.

    Parameters
    ----------
    arc_snr_values : array-like
        Integrated arc signal-to-noise per pool member.
    complexity_values : array-like
        Complexity statistic per pool member. Read but unused under
        ``s_only``, and validated either way so the two curves share one
        admissible pool.
    variant : `str`, optional
        Member of `SCORE_VARIANTS`.

    Returns
    -------
    scores : `numpy.ndarray`
        Score per pool member, higher ranking first.

    Raises
    ------
    ValueError
        Raised for an unknown variant, a length mismatch, or a member
        whose ``S`` or ``C`` is zero, negative or non-finite.
    """
    if variant not in SCORE_VARIANTS:
        raise ValueError(f"variant must be one of {SCORE_VARIANTS}, got {variant!r}.")
    snr = _require_log_ready(arc_snr_values, "arc_snr_values")
    complexity_array = _require_log_ready(complexity_values, "complexity_values")
    if snr.shape != complexity_array.shape:
        raise ValueError(
            f"arc_snr_values holds {snr.size} entries and complexity_values holds "
            f"{complexity_array.size}."
        )
    score = standardize(np.log(snr))
    if variant == "s_plus_c":
        score = score + standardize(np.log(complexity_array))
    return score


def _rank(system_ids: Sequence[str], keys: np.ndarray, descending: bool) -> tuple[str, ...]:
    """Order ids by one key, breaking ties on the ascending id digest."""
    ids = _require_ids(system_ids, "system_ids")
    if keys.shape != (len(ids),):
        raise ValueError(f"system_ids holds {len(ids)} entries and the ranking key holds {keys.size}.")
    sign = -1.0 if descending else 1.0
    order = sorted(range(len(ids)), key=lambda index: (sign * float(keys[index]), _id_digest(ids[index])))
    return tuple(ids[index] for index in order)


def rank_by_score(system_ids, scores):
    """Rank one pool by descending score with a deterministic tiebreak.

    Equal scores break on the ascending sha256 hex digest of the system
    id, which depends on neither the pool membership nor the input order,
    so a re-run or a re-ordered pool produces the identical ranking.

    Parameters
    ----------
    system_ids : sequence of `str`
        Unique non-empty system identifiers.
    scores : array-like
        Score per member, higher ranking first.

    Returns
    -------
    ranking : `tuple` [`str`]
        System ids, best first.

    Raises
    ------
    ValueError
        Raised for duplicate or malformed ids, a length mismatch, or a
        non-finite score.
    """
    keys = _require_finite_array(scores, "scores", ndim=1)
    return _rank(system_ids, keys, descending=True)


def rank_by_sensitivity(system_ids, m_lim_log10_msun):
    """Rank one pool by measured sensitivity, best first.

    The oracle curve of the pre-registered comparison. A lower detectable
    subhalo mass is a more sensitive system, so the ranking is ascending
    in ``log10(M_lim)``. This ranking may only be formed after the
    injected-subhalo ladders are measured; the operational score is
    frozen before any of these values exist.

    Parameters
    ----------
    system_ids : sequence of `str`
        Unique non-empty system identifiers.
    m_lim_log10_msun : array-like
        Measured detectable subhalo mass per member, ``log10`` solar
        masses.

    Returns
    -------
    ranking : `tuple` [`str`]
        System ids, most sensitive first.

    Raises
    ------
    ValueError
        Raised for duplicate or malformed ids, a length mismatch, or a
        non-finite mass.
    """
    keys = _require_finite_array(m_lim_log10_msun, "m_lim_log10_msun", ndim=1)
    return _rank(system_ids, keys, descending=False)


def rank_pool(
    system_ids,
    theta_e_arcsec,
    arc_snr_values,
    complexity_values,
    variant="s_plus_c",
    selected_size=SELECTED_TIER_SIZE,
    golden_size=GOLDEN_TIER_SIZE,
):
    """Run the whole frozen rule over one Stage 0 pool.

    Floor cuts, then standardization over the survivors, then the score,
    then the ranking and its tiers.

    Parameters
    ----------
    system_ids : sequence of `str`
        Unique non-empty system identifiers.
    theta_e_arcsec : array-like
        Einstein radii in arcseconds.
    arc_snr_values : array-like
        Integrated arc signal-to-noise.
    complexity_values : array-like
        Complexity statistic.
    variant : `str`, optional
        Member of `SCORE_VARIANTS`.
    selected_size : `int`, optional
        Selected-tier size, the frozen `SELECTED_TIER_SIZE`.
    golden_size : `int`, optional
        Golden-tier size, the frozen `GOLDEN_TIER_SIZE`.

    Returns
    -------
    result : `SelectionResult`
        Cuts, scores, ranking and tiers.

    Raises
    ------
    ValueError
        Raised when the inputs disagree in length, when a survivor has an
        inadmissible statistic, when the tier sizes are not ordered
        positive integers, or when too few members survive the cuts to
        fill the selected tier.
    """
    ids = _require_ids(system_ids, "system_ids")
    passed = apply_floor_cuts(theta_e_arcsec, arc_snr_values)
    if passed.size != len(ids):
        raise ValueError(f"system_ids holds {len(ids)} entries and the statistics hold {passed.size}.")
    for name, size in (("selected_size", selected_size), ("golden_size", golden_size)):
        if isinstance(size, bool) or not isinstance(size, (int, np.integer)) or int(size) < 1:
            raise ValueError(f"{name} must be a positive integer, got {size!r}.")
    if int(golden_size) > int(selected_size):
        raise ValueError(
            f"golden_size {int(golden_size)} exceeds selected_size {int(selected_size)}."
        )
    survivors = np.flatnonzero(passed)
    if survivors.size < int(selected_size):
        raise ValueError(
            f"{survivors.size} of {len(ids)} pool members survive the floor cuts, too few to "
            f"fill a selected tier of {int(selected_size)}."
        )
    survivor_ids = tuple(ids[index] for index in survivors)
    snr = np.asarray(arc_snr_values, dtype=float)[survivors]
    complexity_array = np.asarray(complexity_values, dtype=float)[survivors]
    scores = selection_scores(snr, complexity_array, variant=variant)
    ranking = rank_by_score(survivor_ids, scores)
    return SelectionResult(
        variant=variant,
        system_ids=ids,
        passed_floor=tuple(bool(entry) for entry in passed),
        survivor_ids=survivor_ids,
        scores=tuple(float(entry) for entry in scores),
        ranking=ranking,
        selected_ids=ranking[: int(selected_size)],
        golden_ids=ranking[: int(golden_size)],
    )


def ranking_positions(ranking):
    """Map each system id to its zero-based position in one ranking.

    Parameters
    ----------
    ranking : sequence of `str`
        System ids, best first.

    Returns
    -------
    positions : `dict` [`str`, `int`]
        Position of every id, zero for the best-ranked member.

    Raises
    ------
    ValueError
        Raised for duplicate or malformed ids.
    """
    ids = _require_ids(ranking, "ranking")
    return {system_id: index for index, system_id in enumerate(ids)}


def _average_ranks(values: np.ndarray) -> np.ndarray:
    """Return one-based ranks, ties sharing their average rank."""
    order = np.argsort(values, kind="stable")
    ordered = values[order]
    ranks = np.empty(values.size, dtype=float)
    start = 0
    while start < values.size:
        stop = start
        while stop + 1 < values.size and ordered[stop + 1] == ordered[start]:
            stop += 1
        ranks[order[start:stop + 1]] = 0.5 * (start + stop) + 1.0
        start = stop + 1
    return ranks


def spearman_rank_correlation(x, y):
    """Return the tie-aware Spearman rank correlation of two vectors.

    Ranks are one-based with tied values sharing their average rank, and
    the correlation is the Pearson coefficient of those ranks, which is
    the standard tie-corrected Spearman statistic.

    Parameters
    ----------
    x, y : array-like
        Paired observations, at least two of them.

    Returns
    -------
    rho : `float`
        Spearman rank correlation in ``[-1, 1]``.

    Raises
    ------
    ValueError
        Raised on a length mismatch, fewer than two pairs, non-finite
        input, or a vector whose entries are all tied, which leaves the
        correlation undefined.
    """
    first = _require_finite_array(x, "x", ndim=1)
    second = _require_finite_array(y, "y", ndim=1)
    if first.shape != second.shape:
        raise ValueError(f"x holds {first.size} entries and y holds {second.size}.")
    if first.size < 2:
        raise ValueError("Spearman correlation needs at least two pairs.")
    rank_x = _average_ranks(first)
    rank_y = _average_ranks(second)
    dx = rank_x - float(np.mean(rank_x))
    dy = rank_y - float(np.mean(rank_y))
    denominator = math.sqrt(float(np.sum(dx ** 2)) * float(np.sum(dy ** 2)))
    if denominator == 0.0:
        raise ValueError(
            "Spearman correlation is undefined when every entry of a vector is tied."
        )
    return float(np.sum(dx * dy) / denominator)


def _top_k(ranking: Iterable[str], k: int, name: str) -> set[str]:
    """Return the leading ``k`` ids of one ranking as a set."""
    ids = _require_ids(tuple(ranking), name)
    if isinstance(k, bool) or not isinstance(k, (int, np.integer)) or int(k) < 1:
        raise ValueError(f"k must be a positive integer, got {k!r}.")
    if int(k) > len(ids):
        raise ValueError(f"{name} holds {len(ids)} entries, fewer than the requested top {int(k)}.")
    return set(ids[: int(k)])


def top_k_jaccard(ranking_a, ranking_b, k):
    """Return the Jaccard index of two top-``k`` sets.

    Parameters
    ----------
    ranking_a, ranking_b : sequence of `str`
        System ids, best first.
    k : `int`
        Tier size, normally `SELECTED_TIER_SIZE`.

    Returns
    -------
    jaccard : `float`
        Intersection over union of the two top-``k`` sets.

    Raises
    ------
    ValueError
        Raised for a non-positive ``k``, a ranking shorter than ``k``, or
        malformed ids.
    """
    top_a = _top_k(ranking_a, k, "ranking_a")
    top_b = _top_k(ranking_b, k, "ranking_b")
    return float(len(top_a & top_b) / len(top_a | top_b))


def oracle_recovered_fraction(ranking, oracle_ranking, k):
    """Return the fraction of an oracle top-``k`` a ranking recovers.

    Parameters
    ----------
    ranking : sequence of `str`
        Operational ranking, best first.
    oracle_ranking : sequence of `str`
        Ranking by measured sensitivity, best first.
    k : `int`
        Tier size, normally `SELECTED_TIER_SIZE`.

    Returns
    -------
    fraction : `float`
        Recovered share of the oracle tier, in ``[0, 1]``.

    Raises
    ------
    ValueError
        Raised for a non-positive ``k``, a ranking shorter than ``k``, or
        malformed ids.
    """
    top_ranking = _top_k(ranking, k, "ranking")
    top_oracle = _top_k(oracle_ranking, k, "oracle_ranking")
    return float(len(top_ranking & top_oracle) / len(top_oracle))
