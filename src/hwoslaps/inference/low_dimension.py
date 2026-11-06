"""Deterministic likelihood strategies in the model's declared prior coordinates.

One-dimensional models have a normalized uniform prior in their listed coordinate,
which can itself be log mass. Search and integration use the unit interval. Every
resolved basin and both edge intervals are refined; a finite mesh cannot certify
absence of narrower unseen modes. Already evaluated incumbents are always retained.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from numbers import Integral
from typing import Any

import numpy as np
from scipy.integrate import quad
from scipy.optimize import brentq, minimize_scalar

__all__ = ["LowDimensionOutcome", "exact_likelihood", "scalar_likelihood"]

_VERSION = "prior_unit_interval_v2"
_PROBABILITIES = (0.16, 0.5, 0.84)
_MESHES = (128, 256, 512)
_LOG_LIKELIHOOD_TOLERANCE = 1.e-7
_QUADRATURE_RTOL = 1.e-8
_QUADRATURE_ATOL = 1.e-11


class _LikelihoodBudgetExhausted(RuntimeError):
    pass


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
        "best_log_likelihood": value, "log_evidence": value, "integration_complete": True,
        "posterior_quantiles": {}, "likelihood_calls": 1, "precomputed_likelihood_calls": 1,
        "likelihood_function_calls": 0, "budget_exhausted": False, "stop_reason": "exact_evaluation",
        "prior_measure": "point_mass", "retained_state": {
            "route": "inline", "complete": True, "required_files": []}})


def scalar_likelihood(log_likelihood: Callable[[tuple[float]], float], lower: float, upper: float,
                      *, parameter_name: str, integrate: bool = True,
                      max_likelihood_calls: int | None = None,
                      initial_evaluation: tuple[float, float] | None = None) -> LowDimensionOutcome:
    """Bounded maximum and normalized quadrature with one hard likelihood-call budget.

    ``initial_evaluation=(physical_coordinate, log_likelihood)`` is an already evaluated
    incumbent. It counts against the budget and is reused. Cached evaluations cost no
    further calls. Optimization, quadrature and any polishing share the same cap; no
    refinement receives an additional budget. Exhaustion retains the best evaluated point
    and records which quality checks remain incomplete. ``None`` imposes no call cap.
    """
    lower, upper = float(lower), float(upper)
    if not (math.isfinite(lower) and math.isfinite(upper) and lower < upper):
        raise ValueError("scalar strategy needs finite lower < upper")
    width = upper - lower
    if not math.isfinite(width):
        raise ValueError("scalar prior width must be finite")
    if max_likelihood_calls is not None and (isinstance(max_likelihood_calls, bool)
            or not isinstance(max_likelihood_calls, Integral) or max_likelihood_calls < 1):
        raise ValueError("max_likelihood_calls must be a positive integer or None")
    cache: dict[float, float] = {}
    physical: dict[float, float] = {}
    best_unit, best_value = 0., -math.inf
    initial_value = None
    if initial_evaluation is not None:
        coordinate, value = initial_evaluation
        coordinate, value = float(coordinate), _finite_likelihood(value)
        if not math.isfinite(coordinate) or not lower <= coordinate <= upper:
            raise ValueError("the initial scalar evaluation must lie inside prior support")
        best_unit, best_value = (coordinate - lower) / width, value
        cache[best_unit], physical[best_unit] = value, coordinate
        initial_value = value
    precomputed = len(cache)
    budget_exhausted = False

    def evaluate(unit: float) -> float:
        nonlocal best_unit, best_value, budget_exhausted
        unit = float(unit)
        if not 0. <= unit <= 1.:
            raise ValueError("scalar strategy evaluated outside its prior support")
        if unit not in cache:
            if max_likelihood_calls is not None and len(cache) >= max_likelihood_calls:
                budget_exhausted = True
                raise _LikelihoodBudgetExhausted("scalar likelihood-call budget exhausted")
            coordinate = lower + width * unit
            value = _finite_likelihood(log_likelihood((coordinate,)))
            cache[unit], physical[unit] = value, coordinate
            if value > best_value:
                best_unit, best_value = unit, value
        return cache[unit]

    mesh_records, candidates = [], []
    previous, mesh_best = None, -math.inf
    stable, optimizer_success, scan_complete = False, True, False
    exhausted_stage = None
    try:
        for intervals in _MESHES:
            grid = np.linspace(0., 1., intervals + 1)
            values = np.asarray([evaluate(unit) for unit in grid])
            peaks = np.flatnonzero(((values[1:-1] >= values[:-2]) & (values[1:-1] > values[2:])) |
                                   ((values[1:-1] > values[:-2]) & (values[1:-1] >= values[2:]))) + 1
            level = [(float(grid[index]), float(values[index])) for index in (0, intervals, int(np.argmax(values)))]
            # The endpoint itself can beat every mesh node even when a broad interior
            # maximum lies between that endpoint and its first neighbour.
            brackets = {(float(grid[0]), float(grid[1])), (float(grid[-2]), float(grid[-1]))}
            brackets.update((float(grid[index - 1]), float(grid[index + 1])) for index in peaks)
            for left, right in sorted(brackets):
                result = minimize_scalar(lambda unit: -evaluate(unit), method="bounded", bounds=(left, right),
                                         options={"xatol": 2.e-12, "maxiter": 200})
                optimizer_success = optimizer_success and bool(result.success)
                level.append((float(result.x), evaluate(result.x)))
            unit, value = max(level, key=lambda item: item[1])
            mesh_best = max(mesh_best, value)
            stable = previous is not None and abs(mesh_best - previous) <= _LOG_LIKELIHOOD_TOLERANCE
            mesh_records.append({"intervals": intervals, "best_unit": unit, "best_log_likelihood": mesh_best,
                                 "sampled_basins": len(peaks), "edge_intervals_refined": 2})
            candidates.extend(level)
            previous = mesh_best
        scan_complete = True
    except _LikelihoodBudgetExhausted:
        exhausted_stage = "optimization"
    known_structure = (scan_complete and initial_value is not None
                       and initial_value > mesh_best + _LOG_LIKELIHOOD_TOLERANCE)
    accepted = scan_complete and stable and optimizer_success and not known_structure
    record = {
        "procedure_version": _VERSION, "optimization_accepted": accepted,
        "best_vector": [physical[best_unit]], "best_log_likelihood": best_value,
        "optimization_complete": scan_complete, "mesh_maximum_stable": stable,
        "local_optimizers_succeeded": optimizer_success, "log_evidence": None,
        "integration_complete": False, "posterior_quantiles": None,
        "prior_measure": "normalized_uniform_in_declared_coordinate", "parameter_name": parameter_name,
        "support": [lower, upper], "meshes": mesh_records,
        "candidate_maxima": [{"unit": unit, "log_likelihood": value} for unit, value in sorted(set(candidates))],
        "log_likelihood_tolerance": _LOG_LIKELIHOOD_TOLERANCE,
        "resolution_limit_unit": 1. / _MESHES[-1],
        "global_search_limit": "finite meshes cannot exclude unresolved narrower modes",
        "requested_likelihood_calls": None if max_likelihood_calls is None else int(max_likelihood_calls),
        "precomputed_likelihood_calls": precomputed,
        "retained_state": {"route": "inline", "complete": True, "required_files": []},
    }
    if initial_evaluation is not None:
        record["initial_evaluation"] = {"physical": float(initial_evaluation[0]), "log_likelihood": initial_value}
    if known_structure:
        # A known high point cannot be dismissed as an unseen-mode caveat. Preserve it,
        # but do not certify a mesh/integral that failed to resolve its neighbourhood.
        record["known_incumbent_exceeds_resolved_mesh"] = True
        record["maximum_check_error"] = "known incumbent exposes structure unresolved by the scalar meshes"
    scan_best = best_value
    if integrate and scan_complete and not known_structure:
        stage = "quadrature"
        try:
            def density(unit: float) -> float:
                return math.exp(evaluate(unit) - scan_best)

            quadrature_ok = True

            def integral(left: float, right: float) -> tuple[float, float]:
                nonlocal quadrature_ok
                output = quad(density, left, right, epsabs=_QUADRATURE_ATOL,
                              epsrel=_QUADRATURE_RTOL, limit=200, full_output=1)
                quadrature_ok = quadrature_ok and len(output) == 3
                return float(output[0]), float(output[1])

            knots = sorted(set(np.linspace(0., 1., 65).tolist() + [unit for unit, _ in candidates]))
            pieces = []
            quadrature = {"relative_tolerance": _QUADRATURE_RTOL, "absolute_tolerance": _QUADRATURE_ATOL,
                          "log_shift": scan_best, "full_prior_coverage": False, "intervals": []}
            record["quadrature"] = quadrature
            for left, right in zip(knots[:-1], knots[1:], strict=True):
                mass, error = integral(left, right)
                pieces.append((mass, error))
                quadrature["intervals"].append({"left_unit": left, "right_unit": right,
                                                 "scaled_mass": mass, "error_estimate": error})
            masses = np.asarray([piece[0] for piece in pieces])
            total, error = math.fsum(masses), math.fsum(piece[1] for piece in pieces)
            if not (math.isfinite(total) and total > 0.):
                raise ValueError("scalar quadrature produced no finite positive evidence")
            quadrature.update(full_prior_coverage=True, scaled_integral=total, relative_error_estimate=error / total)
            # Retain a whole-support evidence estimate even if the remaining checks or
            # posterior work exhaust the budget; its quality remains provisional.
            record["log_evidence"] = scan_best + math.log(total)
            stage = "quadrature_check"
            checks = [integral(float(left), float(right)) for left, right in
                      zip(np.linspace(0., 1., 34)[:-1], np.linspace(0., 1., 34)[1:], strict=True)]
            discrepancy = abs(total - math.fsum(piece[0] for piece in checks)) / total
            quadrature["partition_relative_difference"] = discrepancy
            complete = quadrature_ok and error / total <= _QUADRATURE_RTOL and discrepancy <= _QUADRATURE_RTOL
            cumulative = np.concatenate(([0.], np.cumsum(masses)))
            quantiles = []
            stage = "posterior_quantiles"
            for probability in _PROBABILITIES:
                target = probability * total
                index = min(int(np.searchsorted(cumulative, target, side="right")) - 1, len(masses) - 1)
                left, right = knots[index:index + 2]
                remainder = target - cumulative[index]
                unit = brentq(lambda coordinate: integral(left, coordinate)[0] - remainder,
                              left, right, xtol=2.e-12, rtol=1.e-12)
                quantiles.append(lower + width * unit)
            record.update(integration_complete=complete and quadrature_ok,
                          posterior_quantiles={parameter_name: quantiles})
        except _LikelihoodBudgetExhausted:
            exhausted_stage = stage
            record["integration_error"] = f"scalar likelihood-call budget exhausted during {stage}"
        except (ValueError, RuntimeError, OverflowError) as error:
            record["integration_error"] = f"{type(error).__name__}: {error}"
    if best_value > scan_best + _LOG_LIKELIHOOD_TOLERANCE:
        # Polish an already discovered point rather than retaining an arbitrary quadrature
        # node. Its discovery also invalidates the mesh's mode-resolution claim.
        record["optimization_accepted"] = False
        record["integration_complete"] = False
        record["maximum_check_error"] = "quadrature exposed structure unresolved by the scalar meshes"
        centre = best_unit
        left, right = max(0., centre - 1. / _MESHES[-1]), min(1., centre + 1. / _MESHES[-1])
        try:
            lc, rc = evaluate(left), evaluate(right)
            if left < centre < right and best_value > max(lc, rc):
                result = minimize_scalar(lambda unit: -evaluate(unit), method="brent", bracket=(left, centre, right),
                                         options={"xtol": 1.e-12, "maxiter": 200})
            else:
                result = minimize_scalar(lambda unit: -evaluate(unit), method="bounded", bounds=(left, right),
                                         options={"xatol": 2.e-12, "maxiter": 200})
            record["post_integration_polish"] = {"success": bool(result.success), "bracket_unit": [left, right]}
        except _LikelihoodBudgetExhausted:
            exhausted_stage = "maximum_polish"
        except (ValueError, RuntimeError, OverflowError) as error:
            record["post_integration_polish"] = {"success": False, "error": f"{type(error).__name__}: {error}"}
    record.update(best_vector=[physical[best_unit]], best_log_likelihood=best_value,
                  likelihood_calls=len(cache), likelihood_function_calls=len(cache) - precomputed,
                  budget_exhausted=budget_exhausted, budget_exhausted_stage=exhausted_stage)
    record["stop_reason"] = ("likelihood_budget_exhausted" if budget_exhausted else
                             "unresolved_maximum" if not record["optimization_accepted"] else
                             "maximum_only" if not integrate else
                             "criteria_satisfied" if record["integration_complete"] else "integration_incomplete")
    return LowDimensionOutcome((physical[best_unit],), best_value, record)
