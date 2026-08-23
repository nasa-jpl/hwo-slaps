"""The adaptive mass-ladder walk and its estimands, free of the engine.

One ladder campaign job is one member's complete adaptive mass ladder, so
the decision of which subhalo mass to measure next is taken between two
Fisher grid maps rather than between two campaigns. This module holds
that decision and the estimands read off the finished ladder.

The walk is expressed as a function of the measurements so far: given the
rungs already measured it returns the rung that comes next, or nothing
and the reason the walk stopped. That shape keeps `scripts/run_ladder.py`
a thin loop and leaves the whole adaptive policy exercisable on a machine
that cannot import the modelling engine, so nothing here imports one.

The estimand conventions are the validated panel ones. ``M_best`` is the
log-linear interpolation of the aperture ``q_max`` through the detection
threshold on the refined ladder, and ``M10`` and ``M50`` are linear
interpolations in log-mass of the aperture detected-area fraction through
0.10 and 0.50 on the coarse rungs, each taken at the first upward
crossing. A ladder that never crosses yields nothing: an unbracketed
crossing is a finding, never an extrapolation.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping, Optional, Sequence


Q_THRESHOLD = 10.0
"""Detection threshold the ladder crosses (`float`)."""

THRESHOLD_DECLARATION = "q_F >= 10"
"""Text the frozen mass-ladder policy states its threshold as (`str`)."""

M10_LEVEL = 0.10
"""Aperture detected-area fraction the ``M10`` estimand crosses (`float`)."""

M50_LEVEL = 0.50
"""Aperture detected-area fraction the ``M50`` estimand crosses (`float`)."""

RUNG_DECIMALS = 2
"""Decimal places a rung's log-mass is identified at (`int`).

The panel machinery identifies every rung as ``round(logm, 2)``, so the
0.25 dex coarse lattice, its downward extension and the 0.1 dex
refinement rungs all land on exact two-decimal values and a rung has one
canonical identity no matter which arithmetic produced it.
"""

PHASE_COARSE = "coarse_ascent"
"""Walk phase climbing the coarse lattice from ``coarse.low`` (`str`)."""

PHASE_EXTEND_DOWN = "extend_down"
"""Walk phase closing the curve on measured zeros (`str`)."""

PHASE_REFINE = "refine"
"""Walk phase filling the bracketing coarse pair (`str`)."""

PHASE_COMPLETE = "complete"
"""Walk phase of a ladder with no rung left to measure (`str`)."""

STOP_SATURATED = "aperture_saturated"
"""Ascent stopped on a saturated aperture (`str`)."""

STOP_M50 = "m50_reached"
"""Ascent stopped with ``M50`` bracketed (`str`)."""

STOP_CEILING = "coarse_ceiling"
"""Ascent stopped at ``coarse.high`` (`str`)."""

_EPS = 1.0e-9

_EXTEND_DOWN_MAX_DEX = 2.0
"""Hard sanity bound on the downward extension (`float`).

The frozen extend_down rule terminates physically when the detected
area reaches zero, ordinarily within a rung or two of the floor. A
descent that passes this many dex below ``coarse.low`` without closing
means the physics is broken, and the walk fails loudly instead of
descending forever.
"""
"""Absolute tolerance for comparing rounded log-mass rungs (`float`)."""


@dataclass(frozen=True)
class LadderPolicy:
    """The frozen mass-ladder policy one walk is executed under.

    Parameters
    ----------
    coarse_low : `float`
        Log-mass the coarse ascent starts at.
    coarse_high : `float`
        Log-mass the coarse ascent may not climb beyond.
    coarse_step : `float`
        Coarse ascent step in dex, also the extension step downward.
    refine_step : `float`
        Refinement step in dex inside the bracketing coarse pair.
    extend_down_zero_rungs : `int`
        Consecutive zero-area rungs required below the lowest rung with
        detected area before the curve counts as closed.
    saturation_fraction : `float`
        Aperture detected-area fraction counted as saturation.
    q_threshold : `float`
        Detection threshold the ``M_best`` crossing is taken through.
    """

    coarse_low: float
    coarse_high: float
    coarse_step: float
    refine_step: float
    extend_down_zero_rungs: int
    saturation_fraction: float
    q_threshold: float

    def __post_init__(self):
        for name in ("coarse_step", "refine_step", "q_threshold"):
            value = float(getattr(self, name))
            if not value > 0.0:
                raise ValueError(
                    f"LadderPolicy.{name} must be positive, got {value}"
                )
        if self.coarse_high <= self.coarse_low:
            raise ValueError(
                f"LadderPolicy.coarse_high {self.coarse_high} must lie above "
                f"coarse_low {self.coarse_low}"
            )
        steps = (self.coarse_high - self.coarse_low)/self.coarse_step
        if abs(steps - round(steps)) > _EPS:
            raise ValueError(
                f"The coarse ladder from {self.coarse_low} to "
                f"{self.coarse_high} is not a whole number of "
                f"{self.coarse_step} dex steps"
            )
        if self.refine_step >= self.coarse_step:
            raise ValueError(
                f"LadderPolicy.refine_step {self.refine_step} must be finer "
                f"than coarse_step {self.coarse_step}"
            )
        if int(self.extend_down_zero_rungs) < 1:
            raise ValueError(
                "LadderPolicy.extend_down_zero_rungs must be at least one, "
                f"got {self.extend_down_zero_rungs}"
            )
        if not 0.0 < self.saturation_fraction <= 1.0:
            raise ValueError(
                "LadderPolicy.saturation_fraction must lie in (0, 1], got "
                f"{self.saturation_fraction}"
            )


@dataclass(frozen=True)
class RungMeasurement:
    """One measured rung of a mass ladder.

    Parameters
    ----------
    logm : `float`
        Rung log-mass, rounded to `RUNG_DECIMALS`.
    q_max : `float`
        Largest ``q_F`` inside the D-F7 aperture at this rung.
    detectable_area_arcsec2 : `float`
        Detected area inside the D-F7 aperture at this rung.
    aperture_fraction : `float`
        Detected fraction of the aperture's evaluated nodes.
    """

    logm: float
    q_max: float
    detectable_area_arcsec2: float
    aperture_fraction: float


@dataclass(frozen=True)
class WalkStep:
    """The walk's decision given the rungs measured so far.

    Parameters
    ----------
    logm : `float` or `None`
        Log-mass of the rung to measure next, or `None` when the walk is
        finished.
    phase : `str`
        Phase the returned rung belongs to, or `PHASE_COMPLETE`.
    stop_reason : `str` or `None`
        Why the coarse ascent stopped, once it has. `None` while the
        ascent is still climbing.
    """

    logm: Optional[float]
    phase: str
    stop_reason: Optional[str]


@dataclass(frozen=True)
class Crossing:
    """One interpolated crossing and the rungs that bracket it.

    Parameters
    ----------
    logm : `float`
        Interpolated log-mass of the crossing.
    lower_logm : `float`
        Log-mass of the bracketing rung below the crossing.
    upper_logm : `float`
        Log-mass of the bracketing rung at or above the crossing.
    lower_value : `float`
        Measured value at ``lower_logm``.
    upper_value : `float`
        Measured value at ``upper_logm``.
    """

    logm: float
    lower_logm: float
    upper_logm: float
    lower_value: float
    upper_value: float


def rung_key(logm) -> float:
    """Return the canonical identity of one rung's log-mass.

    Parameters
    ----------
    logm : `float`
        Rung log-mass in any arithmetic form.

    Returns
    -------
    key : `float`
        The log-mass rounded to `RUNG_DECIMALS`.
    """
    value = float(logm)
    if not math.isfinite(value):
        raise ValueError(f"A ladder rung must be finite, got {logm!r}")
    return round(value, RUNG_DECIMALS)


def policy_from_mass_ladder(block: Mapping) -> LadderPolicy:
    """Build the walk policy from the staged mass-ladder block.

    Parameters
    ----------
    block : `Mapping`
        The staged ``ladder.mass_ladder`` block: the design freeze's
        `mass_ladder` policy echoed verbatim plus the ladder spec's two
        implementation constants.

    Returns
    -------
    policy : `LadderPolicy`
        Policy the walk is executed under.
    """
    declared = str(block["threshold"]).strip()
    if declared != THRESHOLD_DECLARATION:
        raise ValueError(
            f"The staged mass-ladder policy declares the detection threshold "
            f"{declared!r} but this checkout implements "
            f"{THRESHOLD_DECLARATION!r}"
        )
    coarse = block["coarse"]
    return LadderPolicy(
        coarse_low=float(coarse["low"]),
        coarse_high=float(coarse["high"]),
        coarse_step=float(coarse["step_dex"]),
        refine_step=float(block["refine"]["step_dex"]),
        extend_down_zero_rungs=int(block["extend_down"]["zero_rungs"]),
        saturation_fraction=float(block["saturation_fraction"]),
        q_threshold=Q_THRESHOLD,
    )


def measurement(entry: Mapping) -> RungMeasurement:
    """Read one per-rung table row into a measurement.

    Parameters
    ----------
    entry : `Mapping` or `RungMeasurement`
        Row carrying ``logm``, ``q_max``, ``detectable_area_arcsec2``
        and ``aperture_fraction``, or an already validated measurement.

    Returns
    -------
    rung : `RungMeasurement`
        The validated measurement.
    """
    if isinstance(entry, RungMeasurement):
        return entry
    rung = RungMeasurement(
        logm=rung_key(entry["logm"]),
        q_max=float(entry["q_max"]),
        detectable_area_arcsec2=float(entry["detectable_area_arcsec2"]),
        aperture_fraction=float(entry["aperture_fraction"]),
    )
    for name, value in (
        ("q_max", rung.q_max),
        ("detectable_area_arcsec2", rung.detectable_area_arcsec2),
        ("aperture_fraction", rung.aperture_fraction),
    ):
        if not math.isfinite(value) or value < 0.0:
            raise ValueError(
                f"Rung {rung.logm} carries {name} {value}, which is not a "
                "non-negative finite measurement"
            )
    if rung.aperture_fraction > 1.0:
        raise ValueError(
            f"Rung {rung.logm} carries aperture_fraction "
            f"{rung.aperture_fraction}, above the whole aperture"
        )
    return rung


def measurements(measured: Sequence[Mapping]) -> tuple:
    """Read a per-rung table into ascending, unique measurements.

    Parameters
    ----------
    measured : `Sequence`
        Per-rung table rows in any order.

    Returns
    -------
    rungs : `tuple`
        `RungMeasurement` instances sorted by ascending log-mass.
    """
    rungs = [measurement(entry) for entry in measured]
    keys = [rung.logm for rung in rungs]
    if len(set(keys)) != len(keys):
        duplicates = sorted({key for key in keys if keys.count(key) > 1})
        raise ValueError(
            f"The ladder measures these rungs more than once: {duplicates}"
        )
    return tuple(sorted(rungs, key=lambda rung: rung.logm))


def is_coarse_rung(logm, policy: LadderPolicy) -> bool:
    """Return whether one rung sits on the coarse lattice.

    The lattice is anchored at ``coarse.low`` and runs in both
    directions, because `extend_down` continues it below ``coarse.low``
    whenever the curve does not close above it.

    Parameters
    ----------
    logm : `float`
        Rung log-mass.
    policy : `LadderPolicy`
        Policy the walk is executed under.

    Returns
    -------
    coarse : `bool`
        Whether the rung is a coarse-lattice rung.
    """
    steps = (rung_key(logm) - policy.coarse_low)/policy.coarse_step
    return abs(steps - round(steps)) <= _EPS


def coarse_measurements(measured: Sequence[Mapping], policy: LadderPolicy):
    """Return the measured coarse rungs, ascending.

    Parameters
    ----------
    measured : `Sequence`
        Per-rung table rows.
    policy : `LadderPolicy`
        Policy the walk is executed under.

    Returns
    -------
    rungs : `tuple`
        Coarse-lattice `RungMeasurement` instances, ascending.
    """
    return tuple(
        rung
        for rung in measurements(measured)
        if is_coarse_rung(rung.logm, policy)
    )


def _ascent_top(coarse, policy: LadderPolicy):
    """Return the highest rung of the contiguous chain from ``coarse.low``."""
    by_key = {rung.logm: rung for rung in coarse}
    logm = rung_key(policy.coarse_low)
    top = None
    while logm in by_key:
        top = by_key[logm]
        logm = rung_key(logm + policy.coarse_step)
    return top


def ascent_stop(measured: Sequence[Mapping], policy: LadderPolicy):
    """Return why the coarse ascent has stopped, or `None`.

    The ascent stops at the first rung that saturates the aperture,
    reaches the ``M50`` level, or sits at ``coarse.high``. Saturation is
    reported ahead of ``M50`` because it is the stronger statement about
    the same monotone quantity: a saturated aperture has necessarily
    passed ``M50`` as well, and the artifact carries one stop reason.

    Parameters
    ----------
    measured : `Sequence`
        Per-rung table rows.
    policy : `LadderPolicy`
        Policy the walk is executed under.

    Returns
    -------
    reason : `str` or `None`
        `STOP_SATURATED`, `STOP_M50`, `STOP_CEILING`, or `None` while the
        ascent is still climbing.
    """
    top = _ascent_top(coarse_measurements(measured, policy), policy)
    if top is None:
        return None
    if top.aperture_fraction >= policy.saturation_fraction:
        return STOP_SATURATED
    if top.aperture_fraction >= M50_LEVEL:
        return STOP_M50
    if top.logm >= rung_key(policy.coarse_high) - _EPS:
        return STOP_CEILING
    return None


def extend_down_rung(measured: Sequence[Mapping], policy: LadderPolicy):
    """Return the next downward rung needed to close the curve, or `None`.

    The curve closes on measured zeros: the lowest rung carrying any
    detected area must sit above `LadderPolicy.extend_down_zero_rungs`
    consecutive zero-area coarse rungs. A ladder whose lowest measured
    rung still detects extends below ``coarse.low`` until it does not.

    Parameters
    ----------
    measured : `Sequence`
        Per-rung table rows.
    policy : `LadderPolicy`
        Policy the walk is executed under.

    Returns
    -------
    logm : `float` or `None`
        Log-mass of the rung to add, or `None` when the curve is closed
        or has no detected area anywhere to close under.
    """
    coarse = coarse_measurements(measured, policy)
    positive = [
        rung for rung in coarse if rung.detectable_area_arcsec2 > 0.0
    ]
    if not positive:
        return None
    lowest_positive = positive[0].logm
    zeros_below = [rung for rung in coarse if rung.logm < lowest_positive]
    if len(zeros_below) >= policy.extend_down_zero_rungs:
        return None
    next_rung = rung_key(coarse[0].logm - policy.coarse_step)
    if next_rung < rung_key(policy.coarse_low - _EXTEND_DOWN_MAX_DEX):
        raise ValueError(
            f"The downward extension reached logm {next_rung}, more than "
            f"{_EXTEND_DOWN_MAX_DEX} dex below the coarse floor "
            f"{policy.coarse_low}, without closing on "
            f"{policy.extend_down_zero_rungs} zero-area rungs; detected "
            "area persisting this far below the declared ladder is not a "
            "curve to extend but a physics failure to investigate"
        )
    return next_rung


def threshold_bracket(measured: Sequence[Mapping], policy: LadderPolicy):
    """Return the tightest measured pair spanning the detection threshold.

    This is the ``t9`` bracketing convention: the lowest rung at or above
    the threshold, paired with the highest rung below it that still lies
    beneath that rung in log-mass.

    Parameters
    ----------
    measured : `Sequence`
        Per-rung table rows.
    policy : `LadderPolicy`
        Policy the walk is executed under.

    Returns
    -------
    bracket : `tuple` or `None`
        ``(below, above)`` `RungMeasurement` pair, or `None` when the
        ladder does not bracket the threshold.
    """
    rungs = measurements(measured)
    above = [rung for rung in rungs if rung.q_max >= policy.q_threshold]
    if not above:
        return None
    upper = above[0]
    below = [
        rung
        for rung in rungs
        if rung.q_max < policy.q_threshold and rung.logm < upper.logm
    ]
    if not below:
        return None
    return below[-1], upper


def refinement_rungs(measured: Sequence[Mapping], policy: LadderPolicy):
    """Return the refinement rungs inside the bracketing coarse pair.

    The freeze declares exactly one refinement: the interior of the
    coarse pair that brackets the detection-threshold crossing, filled at
    the refinement step.

    Parameters
    ----------
    measured : `Sequence`
        Per-rung table rows.
    policy : `LadderPolicy`
        Policy the walk is executed under.

    Returns
    -------
    rungs : `tuple`
        Interior log-masses in ascending order, empty when the coarse
        ladder does not bracket the crossing.
    """
    bracket = threshold_bracket(coarse_measurements(measured, policy), policy)
    if bracket is None:
        return ()
    lower, upper = bracket
    interior = []
    logm = rung_key(lower.logm + policy.refine_step)
    while logm < upper.logm - _EPS:
        interior.append(logm)
        logm = rung_key(logm + policy.refine_step)
    return tuple(interior)


def next_rung(measured: Sequence[Mapping], policy: LadderPolicy) -> WalkStep:
    """Return the rung to measure next, or the finished walk's stop reason.

    The walk climbs the coarse lattice, closes the curve downward on
    measured zeros, then refines the coarse pair bracketing the
    detection-threshold crossing.

    Parameters
    ----------
    measured : `Sequence`
        Per-rung table rows measured so far.
    policy : `LadderPolicy`
        Policy the walk is executed under.

    Returns
    -------
    step : `WalkStep`
        The next rung and its phase, or `None` with the ascent's stop
        reason once nothing is left to measure.
    """
    coarse = coarse_measurements(measured, policy)
    top = _ascent_top(coarse, policy)
    if top is None:
        return WalkStep(rung_key(policy.coarse_low), PHASE_COARSE, None)

    reason = ascent_stop(measured, policy)
    if reason is None:
        return WalkStep(
            rung_key(top.logm + policy.coarse_step), PHASE_COARSE, None
        )

    downward = extend_down_rung(measured, policy)
    if downward is not None:
        return WalkStep(downward, PHASE_EXTEND_DOWN, reason)

    known = {rung.logm for rung in measurements(measured)}
    for logm in refinement_rungs(measured, policy):
        if logm not in known:
            return WalkStep(logm, PHASE_REFINE, reason)
    return WalkStep(None, PHASE_COMPLETE, reason)


def log_linear_crossing(measured: Sequence[Mapping], policy: LadderPolicy):
    """Interpolate the log-mass where ``q_max`` crosses the threshold.

    The ``t9`` ``M_best`` convention: log-linear interpolation of the
    aperture ``q_max`` through the threshold across the tightest
    bracketing pair of the refined ladder.

    Parameters
    ----------
    measured : `Sequence`
        Per-rung table rows of the refined ladder.
    policy : `LadderPolicy`
        Policy the walk is executed under.

    Returns
    -------
    crossing : `Crossing` or `None`
        The interpolated crossing with its bracketing rungs, or `None`
        when the ladder never crosses the threshold.
    """
    bracket = threshold_bracket(measured, policy)
    if bracket is None:
        return None
    lower, upper = bracket
    if lower.q_max <= 0.0:
        raise ValueError(
            f"Rung {lower.logm} brackets the detection threshold from below "
            f"with q_max {lower.q_max}, which has no logarithm; the log-linear "
            "crossing convention cannot be applied to a ladder that reaches "
            "the threshold from exactly zero"
        )
    span = math.log10(upper.q_max) - math.log10(lower.q_max)
    if span <= 0.0:
        raise ValueError(
            f"Rungs {lower.logm} and {upper.logm} bracket the detection "
            f"threshold with q_max {lower.q_max} and {upper.q_max}, which do "
            "not increase across the bracket"
        )
    fraction = (
        math.log10(policy.q_threshold) - math.log10(lower.q_max)
    )/span
    return Crossing(
        logm=float(lower.logm + (upper.logm - lower.logm)*fraction),
        lower_logm=lower.logm,
        upper_logm=upper.logm,
        lower_value=lower.q_max,
        upper_value=upper.q_max,
    )


def aperture_fraction_crossing(
    measured: Sequence[Mapping], policy: LadderPolicy, level: float
):
    """Interpolate the log-mass where the aperture fraction crosses a level.

    The panel ``_crossing`` convention: linear interpolation in log-mass
    across the first consecutive coarse pair that crosses the level
    upward. ``M10`` and ``M50`` are this crossing at 0.10 and 0.50.

    Parameters
    ----------
    measured : `Sequence`
        Per-rung table rows.
    policy : `LadderPolicy`
        Policy the walk is executed under.
    level : `float`
        Aperture detected-area fraction to cross.

    Returns
    -------
    crossing : `Crossing` or `None`
        The interpolated crossing with its bracketing rungs, or `None`
        when the coarse ladder does not bracket the level.
    """
    coarse = coarse_measurements(measured, policy)
    for lower, upper in zip(coarse, coarse[1:]):
        if lower.aperture_fraction < level <= upper.aperture_fraction:
            span = upper.aperture_fraction - lower.aperture_fraction
            return Crossing(
                logm=float(
                    lower.logm
                    + (level - lower.aperture_fraction)
                    * (upper.logm - lower.logm)/span
                ),
                lower_logm=lower.logm,
                upper_logm=upper.logm,
                lower_value=lower.aperture_fraction,
                upper_value=upper.aperture_fraction,
            )
    return None


__all__ = [
    "Crossing",
    "LadderPolicy",
    "M10_LEVEL",
    "M50_LEVEL",
    "PHASE_COARSE",
    "PHASE_COMPLETE",
    "PHASE_EXTEND_DOWN",
    "PHASE_REFINE",
    "Q_THRESHOLD",
    "RUNG_DECIMALS",
    "RungMeasurement",
    "STOP_CEILING",
    "STOP_M50",
    "STOP_SATURATED",
    "THRESHOLD_DECLARATION",
    "WalkStep",
    "aperture_fraction_crossing",
    "ascent_stop",
    "coarse_measurements",
    "extend_down_rung",
    "is_coarse_rung",
    "log_linear_crossing",
    "measurement",
    "measurements",
    "next_rung",
    "policy_from_mass_ladder",
    "refinement_rungs",
    "rung_key",
    "threshold_bracket",
]
