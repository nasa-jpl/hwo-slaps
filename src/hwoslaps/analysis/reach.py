"""Mass reach: where a measured curve over subhalo mass crosses a target.

The crossing interpolates between the two measured masses that bracket the
target, always in log mass. ``interpolation`` sets how the value is
interpolated: ``linear`` in the value or ``log`` in its logarithm. A power law
``q = A M**alpha`` is a straight line in (log M, log q), so ``log`` recovers its
crossing exactly while ``linear`` places it low for ``alpha > 0``; ``log`` suits
``q_max`` and areas, ``linear`` suits fractions. A curve that never reaches the
target inside the measured range is a bound, never an extrapolation, and a
non-monotonic curve has no unique reach. Every result records the target, the
interpolation and the summary quantity it came from.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace
from numbers import Integral
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike

from .reductions import ForecastSummary

__all__ = ["AdaptiveMassResult", "Interpolation", "MassReach", "adaptive_mass_reach", "crossing", "mass_reach"]

Interpolation = Literal["linear", "log"]
"""How the value is interpolated between bracketing masses; mass is always interpolated in log."""

_QUANTITIES = ("q_max", "detectable_fraction", "detectable_area_arcsec2")


@dataclass(frozen=True)
class MassReach:
    """A crossing mass, or the bound the measurements give.

    ``sampled``: a measured value equals the target. ``bracketed``: interpolated
    between ``lower_mass_msun`` and ``upper_mass_msun``. ``below_range``: the
    lowest measured value already exceeds the target, so the reach is below
    ``upper_mass_msun``. ``above_range``: no measured value reaches it, so the
    reach is above ``lower_mass_msun``. ``non_monotonic``: no unique reach.
    """

    status: Literal["sampled", "bracketed", "below_range", "above_range", "non_monotonic"]
    mass_msun: float | None
    lower_mass_msun: float | None
    upper_mass_msun: float | None
    target: float
    quantity: str | None
    interpolation: Interpolation


def crossing(masses_msun: ArrayLike, values: ArrayLike, *, target: float,
             interpolation: Interpolation) -> MassReach:
    """The mass where a measured curve over strictly increasing masses reaches ``target``."""
    masses = np.asarray(masses_msun, dtype=float)
    curve = np.asarray(values, dtype=float)
    level = float(target)
    if masses.ndim != 1 or not masses.size or curve.shape != masses.shape:
        raise ValueError("masses and values must be non-empty vectors of the same length")
    if not np.all(np.isfinite(masses)) or np.any(masses <= 0) or np.any(np.diff(masses) <= 0):
        raise ValueError("masses must be positive, finite and strictly increasing")
    if not np.all(np.isfinite(curve)) or not np.isfinite(level):
        raise ValueError("values and target must be finite")
    if interpolation not in ("linear", "log"):
        raise ValueError(f"interpolation must be 'linear' or 'log', got {interpolation!r}")

    def reach(status, mass, lower, upper) -> MassReach:
        return MassReach(status, mass, lower, upper, level, None, interpolation)

    if np.any(np.diff(curve) < 0):
        return reach("non_monotonic", None, None, None)
    if curve[0] > level:
        return reach("below_range", None, None, float(masses[0]))
    if curve[-1] < level:
        return reach("above_range", None, float(masses[-1]), None)
    above = int(np.flatnonzero(curve >= level)[0])
    if curve[above] == level:
        mass = float(masses[above])
        return reach("sampled", mass, mass, mass)
    low = above - 1
    lower, upper = curve[low], curve[above]
    if interpolation == "log":
        if lower <= 0 or level <= 0:
            raise ValueError("log interpolation requires positive bracket values and target")
        lower, upper, level_value = np.log([lower, upper, level])
    else:
        level_value = level
    fraction = (level_value - lower) / (upper - lower)
    log_mass = np.log(masses[low]) + fraction * np.log(masses[above] / masses[low])
    return reach("bracketed", float(np.exp(log_mass)), float(masses[low]), float(masses[above]))


def mass_reach(summary: ForecastSummary, *,
               quantity: Literal["q_max", "detectable_fraction", "detectable_area_arcsec2"],
               target: float, interpolation: Interpolation) -> MassReach:
    """The crossing of one summary quantity over the summary's masses."""
    if quantity not in _QUANTITIES:
        raise ValueError(f"quantity must be one of {_QUANTITIES}, got {quantity!r}")
    values = getattr(summary, quantity)
    if values is None:
        raise ValueError(f"the summary has no {quantity}: its result has no cell areas")
    found = crossing(summary.masses_msun, values, target=target, interpolation=interpolation)
    return replace(found, quantity=quantity)


@dataclass(frozen=True, eq=False)
class AdaptiveMassResult:
    """The measured samples, in increasing mass, and their crossing; no unmeasured mass is asserted."""

    masses_msun: np.ndarray
    values: np.ndarray
    reach: MassReach


def adaptive_mass_reach(evaluate: Callable[[float], float], *, lower_mass_msun: float,
                        upper_mass_msun: float, target: float, interpolation: Interpolation,
                        tolerance_dex: float = 0.05, max_evaluations: int = 32) -> AdaptiveMassResult:
    """Refine the crossing of ``evaluate(mass)`` by bisection in log mass inside ``[lower, upper]``.

    The bracket is never extended and no mass is evaluated twice. Refinement
    stops when the bracket is within ``tolerance_dex``, after
    ``max_evaluations`` evaluations, when the bracket is too narrow to split at
    double precision, or when the samples stop being bracketed (a bound or a
    non-monotonic curve is reported as found).
    """
    bounds = np.asarray([lower_mass_msun, upper_mass_msun], dtype=float)
    if not np.all(np.isfinite(bounds)) or bounds[0] <= 0 or bounds[1] <= bounds[0]:
        raise ValueError("mass bounds must be positive finite values with lower < upper")
    if not np.isfinite(tolerance_dex) or tolerance_dex <= 0:
        raise ValueError("tolerance_dex must be positive and finite")
    if isinstance(max_evaluations, bool) or not isinstance(max_evaluations, Integral) or max_evaluations < 2:
        raise ValueError("max_evaluations must be an integer of at least two")
    samples = {float(mass): float(evaluate(float(mass))) for mass in bounds}
    while True:
        masses = np.asarray(sorted(samples), dtype=float)
        values = np.asarray([samples[mass] for mass in masses], dtype=float)
        found = crossing(masses, values, target=target, interpolation=interpolation)
        if found.status != "bracketed" or len(samples) >= max_evaluations:
            break
        if np.log10(found.upper_mass_msun / found.lower_mass_msun) <= tolerance_dex:
            break
        middle = float(np.sqrt(found.lower_mass_msun * found.upper_mass_msun))
        if not found.lower_mass_msun < middle < found.upper_mass_msun:
            break
        samples[middle] = float(evaluate(middle))
    return AdaptiveMassResult(masses, values, found)
