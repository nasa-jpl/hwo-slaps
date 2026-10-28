"""Deterministic likelihood strategies in the model's declared prior coordinates.

One-dimensional models have a normalized uniform prior in their *listed coordinate*,
which can itself be log mass. Both the global scan and the adaptive quadrature run in
the unit interval, so evidence never acquires a spurious physical box-width factor.
The scalar maximum checks every sampled basin and both endpoints on successive meshes;
it does not assume that the full likelihood is unimodal. Like any finite search, it
cannot certify absence of features narrower than its recorded mesh resolution.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.integrate import quad
from scipy.optimize import brentq, minimize_scalar

__all__ = ["LowDimensionOutcome", "exact_likelihood", "scalar_likelihood"]

_VERSION = "prior_unit_interval_v1"
_PROBABILITIES = (0.16, 0.5, 0.84)
_MESHES = (128, 256, 512)
_LOG_LIKELIHOOD_TOLERANCE = 1.e-7
_QUADRATURE_RTOL = 1.e-8
_QUADRATURE_ATOL = 1.e-11


@dataclass(frozen=True)
class LowDimensionOutcome:
    best_vector: tuple[float, ...]
    best_log_likelihood: float
    record: dict[str, Any]


def _finite_likelihood(value: float) -> float:
    value = float(value)
    if not math.isfinite(value):
        raise ValueError("deterministic strategy encountered a non-finite likelihood")
    return value


def exact_likelihood(log_likelihood: float) -> LowDimensionOutcome:
    """The sole likelihood and normalized evidence of a model with no free parameters."""
    value = _finite_likelihood(log_likelihood)
    return LowDimensionOutcome((), value, {
        "procedure_version": _VERSION, "optimization_accepted": True, "best_vector": [],
        "best_log_likelihood": value,
        "log_evidence": value, "integration_complete": True, "posterior_quantiles": {},
        "likelihood_calls": 1, "prior_measure": "point_mass", "retained_state": {
            "route": "inline", "complete": True, "required_files": []}})


def scalar_likelihood(log_likelihood: Callable[[tuple[float]], float], lower: float, upper: float,
                      *, parameter_name: str, integrate: bool = True) -> LowDimensionOutcome:
    """Search every resolved basin, then integrate the normalized declared uniform prior.

    Optimization and integration have separate acceptance flags. Integration failure keeps
    the maximum and provisional evidence; it cannot invalidate an otherwise accepted q.
    ``integrate=False`` requests maximum likelihood alone, without posterior/evidence work.
    """
    lower, upper = float(lower), float(upper)
    if not (math.isfinite(lower) and math.isfinite(upper) and lower < upper):
        raise ValueError("scalar strategy needs finite lower < upper")
    width = upper - lower
    if not math.isfinite(width):
        raise ValueError("scalar prior width must be finite")
    cache: dict[float, float] = {}

    def evaluate(unit: float) -> float:
        unit = float(unit)
        if not 0. <= unit <= 1.:
            raise ValueError("scalar strategy evaluated outside its prior support")
        if unit not in cache:
            cache[unit] = _finite_likelihood(log_likelihood((lower + width * unit,)))
        return cache[unit]

    mesh_records, candidates = [], []
    previous = None
    stable = False
    optimizer_success = True
    best_unit = 0.
    best_value = evaluate(best_unit)
    for intervals in _MESHES:
        grid = np.linspace(0., 1., intervals + 1)
        values = np.asarray([evaluate(unit) for unit in grid])
        # Strict on one side avoids launching hundreds of minimizers on a flat plateau.
        peaks = np.flatnonzero(((values[1:-1] >= values[:-2]) & (values[1:-1] > values[2:])) |
                               ((values[1:-1] > values[:-2]) & (values[1:-1] >= values[2:]))) + 1
        level = [(float(grid[index]), float(values[index])) for index in (0, intervals, int(np.argmax(values)))]
        for index in peaks:
            result = minimize_scalar(lambda unit: -evaluate(unit), method="bounded",
                                     bounds=(float(grid[index - 1]), float(grid[index + 1])),
                                     options={"xatol": 2.e-12, "maxiter": 200})
            optimizer_success = optimizer_success and bool(result.success)
            level.append((float(result.x), evaluate(result.x)))
        unit, value = max(level, key=lambda item: item[1])
        if value > best_value:
            best_unit, best_value = unit, value
        stable = previous is not None and abs(best_value - previous) <= _LOG_LIKELIHOOD_TOLERANCE
        mesh_records.append({"intervals": intervals, "best_unit": best_unit,
                             "best_log_likelihood": best_value, "sampled_basins": len(peaks)})
        candidates.extend(level)
        previous = best_value
    accepted = stable and optimizer_success
    best_vector = (lower + width * best_unit,)
    record = {
        "procedure_version": _VERSION, "optimization_accepted": accepted,
        "best_vector": list(best_vector), "best_log_likelihood": best_value,
        "mesh_maximum_stable": stable, "local_optimizers_succeeded": optimizer_success,
        "log_evidence": None, "integration_complete": False,
        "posterior_quantiles": None, "prior_measure": "normalized_uniform_in_declared_coordinate",
        "parameter_name": parameter_name, "support": [lower, upper], "meshes": mesh_records,
        "candidate_maxima": [{"unit": unit, "log_likelihood": value} for unit, value in sorted(set(candidates))],
        "log_likelihood_tolerance": _LOG_LIKELIHOOD_TOLERANCE,
        "resolution_limit_unit": 1. / _MESHES[-1],
        "global_search_limit": "finite meshes cannot exclude unresolved narrower modes",
        "retained_state": {"route": "inline", "complete": True, "required_files": []},
    }
    if integrate:
        try:
            # Use a log shift for underflow safety; the shift cancels in posterior probabilities.
            def density(unit: float) -> float:
                return math.exp(evaluate(unit) - best_value)

            quadrature_ok = True

            def integral(left: float, right: float) -> tuple[float, float]:
                nonlocal quadrature_ok
                output = quad(density, left, right, epsabs=_QUADRATURE_ATOL,
                              epsrel=_QUADRATURE_RTOL, limit=200, full_output=1)
                quadrature_ok = quadrature_ok and len(output) == 3
                return float(output[0]), float(output[1])

            # Split the integral across the full support and all discovered maxima. Quadrature
            # therefore sees separated peaks even when a full-interval call would miss one.
            knots = sorted(set(np.linspace(0., 1., 65).tolist() + [unit for unit, _ in candidates]))
            pieces = [integral(left, right) for left, right in zip(knots[:-1], knots[1:], strict=True)]
            masses = np.asarray([piece[0] for piece in pieces])
            total, error = math.fsum(masses), math.fsum(piece[1] for piece in pieces)
            if not (math.isfinite(total) and total > 0.):
                raise ValueError("scalar quadrature produced no finite positive evidence")
            # A second partition checks the numerical integral independently of its local error
            # estimates. This is separate from agreement between the maximum-search meshes.
            checks = [integral(float(left), float(right)) for left, right in
                      zip(np.linspace(0., 1., 34)[:-1], np.linspace(0., 1., 34)[1:], strict=True)]
            check_total = math.fsum(piece[0] for piece in checks)
            discrepancy = abs(total - check_total) / total
            complete = quadrature_ok and error / total <= _QUADRATURE_RTOL and discrepancy <= _QUADRATURE_RTOL
            cumulative = np.concatenate(([0.], np.cumsum(masses)))
            quantiles = []
            for probability in _PROBABILITIES:
                target = probability * total
                index = min(int(np.searchsorted(cumulative, target, side="right")) - 1, len(masses) - 1)
                left, right = knots[index:index + 2]
                remainder = target - cumulative[index]
                unit = brentq(lambda coordinate: integral(left, coordinate)[0] - remainder,
                              left, right, xtol=2.e-12, rtol=1.e-12)
                quantiles.append(lower + width * unit)
            complete = complete and quadrature_ok
            record.update(log_evidence=best_value + math.log(total), integration_complete=complete,
                          posterior_quantiles={parameter_name: quantiles},
                          quadrature={"relative_error_estimate": error / total,
                                      "partition_relative_difference": discrepancy,
                                      "relative_tolerance": _QUADRATURE_RTOL,
                                      "absolute_tolerance": _QUADRATURE_ATOL,
                                      "log_shift": best_value, "scaled_integral": total,
                                      "intervals": [{"left_unit": left, "right_unit": right,
                                                     "scaled_mass": mass, "error_estimate": err}
                                                    for left, right, (mass, err) in
                                                    zip(knots[:-1], knots[1:], pieces, strict=True)]})
        except (ValueError, RuntimeError, OverflowError) as error:
            record["integration_complete"] = False
            record["integration_error"] = f"{type(error).__name__}: {error}"
    # Quadrature is also an independent challenge to the scanned maximum. Retain a better
    # evaluated point, but do not call the unresolved scan accepted or hide it behind q.
    checked_unit, checked_value = max(cache.items(), key=lambda item: item[1])
    if checked_value > best_value + _LOG_LIKELIHOOD_TOLERANCE:
        record["optimization_accepted"] = False
        record["maximum_check_error"] = "quadrature found a better maximum than the global scan"
        best_vector = (lower + width * checked_unit,)
        best_value = checked_value
        record["best_vector"] = list(best_vector)
        record["best_log_likelihood"] = best_value
    record["likelihood_calls"] = len(cache)
    return LowDimensionOutcome(best_vector, best_value, record)
