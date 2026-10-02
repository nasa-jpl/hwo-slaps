"""Generic mass-threshold crossings and bounded adaptive refinement."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np


@dataclass(frozen=True)
class MassReach:
    """A crossing or an explicit bound from measured masses.

    ``below_range`` means the crossing lies below the lowest measured mass;
    ``above_range`` means it lies above the highest. Neither is extrapolated.
    Nonmonotone measurements do not establish a unique mass reach.
    """

    status: str
    mass_msun: Optional[float]
    lower_mass_msun: Optional[float]
    upper_mass_msun: Optional[float]
    target: float


def mass_reach(masses_msun, values, target, *, interpolation="linear") -> MassReach:
    """Interpolate a monotone measured curve in log mass with honest censoring.

    ``linear`` interpolates the measured values; ``log`` interpolates their
    logarithms and requires positive bracket values. Either accepts arbitrary
    detection fractions or statistics rather than fixed study estimands.
    """
    masses = np.asarray(masses_msun, dtype=float)
    curve = np.asarray(values, dtype=float)
    level = float(target)
    if masses.ndim != 1 or not masses.size or curve.shape != masses.shape:
        raise ValueError("masses and values must be non-empty vectors of the same length")
    if not np.all(np.isfinite(masses)) or np.any(masses <= 0) or np.any(np.diff(masses) <= 0):
        raise ValueError("masses must be positive, finite, and strictly increasing")
    if not np.all(np.isfinite(curve)) or not np.isfinite(level):
        raise ValueError("values and target must be finite")
    if interpolation not in {"linear", "log"}:
        raise ValueError("interpolation must be linear or log")
    if np.any(np.diff(curve) < 0):
        return MassReach("non_monotonic", None, None, None, level)
    if curve[0] > level:
        return MassReach("below_range", None, None, float(masses[0]), level)
    if curve[-1] < level:
        return MassReach("above_range", None, float(masses[-1]), None, level)
    above = int(np.flatnonzero(curve >= level)[0])
    if curve[above] == level:
        mass = float(masses[above])
        return MassReach("sampled", mass, mass, mass, level)
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
    return MassReach("bracketed", float(np.exp(log_mass)), float(masses[low]), float(masses[above]), level)


@dataclass(frozen=True)
class AdaptiveMassResult:
    """Measured samples and their crossing; no unmeasured mass is asserted."""

    masses_msun: np.ndarray
    values: np.ndarray
    reach: MassReach


def adaptive_mass_reach(
    evaluate: Callable[[float], float],
    lower_mass_msun: float,
    upper_mass_msun: float,
    target: float,
    *,
    tolerance_dex: float = 0.05,
    max_evaluations: int = 32,
    interpolation: str = "linear",
) -> AdaptiveMassResult:
    """Refine a measured bracket by bisection in log mass.

    The caller supplies the scalar statistic (maximum q or any detection
    fraction). The declared finite mass range is never extended implicitly.
    Nonmonotone samples stop refinement and are reported without interpolation.
    """
    bounds = np.asarray([lower_mass_msun, upper_mass_msun], dtype=float)
    if not np.all(np.isfinite(bounds)) or bounds[0] <= 0 or bounds[1] <= bounds[0]:
        raise ValueError("mass bounds must be positive finite values with lower < upper")
    if not np.isfinite(tolerance_dex) or tolerance_dex <= 0:
        raise ValueError("tolerance_dex must be positive and finite")
    if (
        isinstance(max_evaluations, bool)
        or not isinstance(max_evaluations, (int, np.integer))
        or max_evaluations < 2
    ):
        raise ValueError("max_evaluations must be an integer of at least two")
    samples = {float(mass): float(evaluate(float(mass))) for mass in bounds}
    while True:
        masses = np.asarray(sorted(samples), dtype=float)
        values = np.asarray([samples[mass] for mass in masses], dtype=float)
        reach = mass_reach(masses, values, target, interpolation=interpolation)
        if reach.status != "bracketed" or len(samples) >= max_evaluations:
            break
        lower, upper = reach.lower_mass_msun, reach.upper_mass_msun
        if np.log10(upper / lower) <= tolerance_dex:
            break
        middle = float(np.sqrt(lower * upper))
        samples[middle] = float(evaluate(middle))
    return AdaptiveMassResult(masses, values, reach)
