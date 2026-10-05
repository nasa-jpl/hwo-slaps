"""Refinement of one role maximum: multistart L-BFGS-B on the unit box and six acceptance gates.

Every start runs a bounded L-BFGS-B from its unit-box vector; every evaluation is tracked and
the best finite evaluation inside the box is retained even when its gradient is not finite. A
tighter repeat starts from the global best and replaces it when lower. The retained best is
accepted (``accepted_repeatable_profile``) only when all six gates hold:

- ``support``: at least ``minimum_distinct_original_start_support`` original starts end within
  ``support_log_likelihood_tolerance`` of the best;
- ``repeat``: the tighter repeat completed and moved the best by at most
  ``repeat_log_likelihood_tolerance``;
- ``finite_gradient``: the gradient at the best is finite;
- ``scalar_residual``: ``|2 f - r . r| <= scalar_residual_tolerance`` at the best;
- ``direct_log_likelihood``: the direct likelihood at the best agrees with ``-f - N / 2``;
- ``incumbent``: the sampler maximum evaluates, the best is not worse than it, its direct value
  is consistent and equals the sampler's saved likelihood within the same tolerance.

The gates certify a repeatable maximum, not stationarity; the projected gradient at the best is
recorded beside them. numpy and scipy only.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Callable, Iterable, Mapping, Sequence
from typing import Any

import numpy as np
import scipy.optimize

from .objective import BoxObjective, ScalarCheck
from .result import RefineOutcome, RoleStatus
from .settings import PROCEDURE_VERSION, RefineSettings
from .starts import RefineStart

__all__ = ["refine"]

logger = logging.getLogger(__name__)

_BOUND_TOLERANCE = 1.0e-10


def _empty_state() -> dict[str, Any]:
    return {"best_half_chi2": None, "best_chi2": None, "best_z": None, "best_x": None, "best_gradient": None,
            "best_gradient_valid": False, "best_gradient_error": None, "evaluation_count": 0,
            "failed_evaluation_count": 0}


def _record_best_finite(state: dict[str, Any], z: Sequence[float], value: float, gradient: Sequence[float] | None,
                        physical_x: Sequence[float]) -> bool:
    """Retain a strictly better finite evaluation inside the box, even with a bad gradient."""
    z_array = np.asarray(z, dtype=float)
    x_array = np.asarray(physical_x, dtype=float)
    scalar = float(value)
    if (not np.isfinite(scalar) or not np.all(np.isfinite(z_array)) or not np.all(np.isfinite(x_array))
            or np.any(z_array < 0.0) or np.any(z_array > 1.0)):
        return False
    old = state.get("best_half_chi2")
    if old is not None and scalar >= float(old):
        return False
    state.update({"best_half_chi2": scalar, "best_chi2": 2.0 * scalar, "best_z": z_array.tolist(),
                  "best_x": x_array.tolist()})
    gradient_array = None if gradient is None else np.asarray(gradient, dtype=float)
    valid = (gradient_array is not None and gradient_array.shape == z_array.shape
             and np.all(np.isfinite(gradient_array)))
    state["best_gradient_valid"] = bool(valid)
    state["best_gradient"] = gradient_array.tolist() if valid else None
    state["best_gradient_error"] = None if valid else "nonfinite_or_incompatible_gradient"
    return True


def _projected_gradient(gradient: Sequence[float] | None, z: Sequence[float]) -> dict[str, Any]:
    """The unit-box gradient with outward components at an active bound zeroed."""
    z_array = np.asarray(z, dtype=float)
    gradient_array = None if gradient is None else np.asarray(gradient, dtype=float)
    if gradient_array is None or gradient_array.shape != z_array.shape or not np.all(np.isfinite(gradient_array)):
        return {"valid": False, "linf": None, "l2": None, "values": None}
    value = gradient_array.copy()
    value[(z_array <= _BOUND_TOLERANCE) & (value > 0.0)] = 0.0
    value[(z_array >= 1.0 - _BOUND_TOLERANCE) & (value < 0.0)] = 0.0
    return {"valid": True, "linf": float(np.max(np.abs(value))), "l2": float(np.linalg.norm(value)),
            "values": value.tolist()}


def _support_summary(runs: Iterable[Mapping[str, Any]], best_half_chi2: float,
                     settings: RefineSettings) -> dict[str, Any]:
    """Count the distinct original starts whose best ends within tolerance of the retained best."""
    finite = [row for row in runs if row.get("observed_best_half_chi2") is not None
              and np.isfinite(float(row["observed_best_half_chi2"]))]
    seen: set[int] = set()
    duplicate: list[int] = []
    original: list[Mapping[str, Any]] = []
    for row in finite:
        if not row["start_provenance"]["original"]:
            continue
        index = int(row["start_index"])
        if index in seen:
            duplicate.append(index)
            continue
        seen.add(index)
        original.append(row)
    supporting = [int(row["start_index"]) for row in original
                  if float(row["observed_best_half_chi2"]) - float(best_half_chi2)
                  <= settings.support_log_likelihood_tolerance]
    return {"finite_start_count": len(finite), "original_start_count": len(original),
            "supporting_original_start_indices": supporting,
            "duplicate_original_start_indices": sorted(set(duplicate)),
            "minimum_supporting_original_starts": settings.minimum_distinct_original_start_support,
            "support_tolerance_log_likelihood": settings.support_log_likelihood_tolerance,
            "support_passed": len(supporting) >= settings.minimum_distinct_original_start_support}


def _check_start(start: RefineStart, objective: BoxObjective) -> None:
    z0 = np.asarray(start.normalized, dtype=float)
    if z0.shape != objective.lower.shape:
        raise ValueError(f"start {start.index} has {z0.size} parameters, the objective "
                         f"{objective.lower.size}; the objective was not evaluated")
    if not np.allclose(objective.to_physical(z0), np.asarray(start.physical, dtype=float), rtol=1.0e-9,
                       atol=1.0e-12):
        raise ValueError(f"start {start.index} does not map back to its physical origin in the objective's box; "
                         "the objective was not evaluated")


def _run_start(start: RefineStart, objective: BoxObjective, settings: RefineSettings) -> dict[str, Any]:
    z0 = np.asarray(start.normalized, dtype=float)
    x0 = np.asarray(start.physical, dtype=float)
    state = _empty_state()

    def tracked(z: np.ndarray) -> tuple[float, np.ndarray]:
        state["evaluation_count"] += 1
        try:
            value, gradient = objective.value_and_gradient(np.asarray(z, dtype=float))
        except Exception:
            state["failed_evaluation_count"] += 1
            raise
        _record_best_finite(state, z, value, gradient, objective.to_physical(z))
        return value, gradient

    solver_result = None
    exception = None
    endpoint_z = None
    endpoint_value = None
    endpoint_gradient = None
    start_value = None
    try:
        start_value, _ = tracked(z0)
        start_value = float(start_value) if np.isfinite(start_value) else None
        solver_result = scipy.optimize.minimize(
            tracked, z0, method="L-BFGS-B", jac=True, bounds=[(0.0, 1.0)] * z0.size,
            options={"maxiter": int(settings.maxiter), "ftol": float(settings.ftol), "gtol": float(settings.gtol),
                     "maxls": int(settings.maxls)})
        endpoint_z = np.asarray(solver_result.x, dtype=float)
        endpoint_value, endpoint_gradient = tracked(endpoint_z)
    except Exception as error:  # one start's numerical failure leaves that run without an endpoint
        exception = f"{type(error).__name__}: {error}"
    endpoint_gradient_finite = endpoint_gradient is not None and bool(np.all(np.isfinite(endpoint_gradient)))
    logger.info("refinement start %d: best half chi-square %r after %d evaluations", start.index,
                state["best_half_chi2"], state["evaluation_count"])
    return {
        "start_index": start.index,
        "start_provenance": start.to_mapping(),
        "start_z": z0.tolist(),
        "start_x": x0.tolist(),
        "start_half_chi2": start_value,
        "solver_endpoint_z": None if endpoint_z is None else endpoint_z.tolist(),
        "solver_endpoint_x": None if endpoint_z is None else objective.to_physical(endpoint_z).tolist(),
        "solver_endpoint_half_chi2": endpoint_value,
        "solver_endpoint_chi2": None if endpoint_value is None else 2.0 * endpoint_value,
        "solver_endpoint_gradient_z": np.asarray(endpoint_gradient).tolist() if endpoint_gradient_finite else None,
        "solver_endpoint_gradient_valid": endpoint_gradient_finite,
        "solver_endpoint_projected_gradient": None if endpoint_z is None
        else _projected_gradient(endpoint_gradient, endpoint_z),
        "observed_best_z": state["best_z"],
        "observed_best_x": state["best_x"],
        "observed_best_half_chi2": state["best_half_chi2"],
        "observed_best_chi2": state["best_chi2"],
        "observed_best_gradient_z": state["best_gradient"],
        "observed_best_gradient_valid": state["best_gradient_valid"],
        "observed_best_gradient_error": state["best_gradient_error"],
        "observed_best_projected_gradient": None if state["best_z"] is None
        else _projected_gradient(state["best_gradient"], state["best_z"]),
        "evaluation_count": state["evaluation_count"],
        "failed_evaluation_count": state["failed_evaluation_count"],
        "success": None if solver_result is None else bool(solver_result.success),
        "status": None if solver_result is None else int(solver_result.status),
        "message": exception if solver_result is None else str(solver_result.message),
        "nit": 0 if solver_result is None else int(solver_result.nit),
        "nfev": state["evaluation_count"],
        "njev": 0 if solver_result is None else int(solver_result.njev),
    }


def _incumbent_record(incumbent: RefineStart, incumbent_run: Mapping[str, Any], best_half_chi2: float | None,
                      direct_check: Callable[[np.ndarray, float], ScalarCheck] | None,
                      settings: RefineSettings) -> dict[str, Any]:
    """The sampler maximum against the retained best: evaluated, not better than the best,
    directly consistent, and equal to the sampler's saved likelihood."""
    tolerance = settings.scalar_residual_tolerance
    half = incumbent_run.get("start_half_chi2")
    evaluated = half is not None and np.isfinite(float(half))
    record: dict[str, Any] = {
        "physical_vector": list(incumbent.physical), "normalized_vector": list(incumbent.normalized),
        "half_chi2": None if not evaluated else float(half), "chi2": None if not evaluated else 2.0 * float(half),
        "evaluated": bool(evaluated), "saved_log_likelihood": incumbent.origin.get("saved_log_likelihood"),
        "direct_log_likelihood": None, "direct_log_likelihood_error": None, "saved_log_likelihood_error": None,
        "candidate_not_worse": False, "scalar_consistent": False, "matches_sampler": False,
        "tolerance": tolerance, "passed": False,
    }
    if not evaluated:
        return record
    if best_half_chi2 is not None and np.isfinite(best_half_chi2):
        record["candidate_not_worse"] = bool(float(best_half_chi2) <= float(half) + tolerance)
    if direct_check is not None:
        scalar = direct_check(np.asarray(incumbent.normalized, dtype=float), float(half))
        direct = scalar.direct_log_likelihood
        direct_error = scalar.direct_log_likelihood_error
        record["direct_log_likelihood"] = direct
        record["direct_log_likelihood_error"] = direct_error
        record["scalar_consistent"] = bool(np.isfinite(float(direct_error)) and float(direct_error) <= tolerance)
        saved = record["saved_log_likelihood"]
        if (isinstance(saved, (int, float)) and not isinstance(saved, bool) and np.isfinite(float(saved))
                and np.isfinite(float(direct))):
            saved_error = abs(float(saved) - float(direct))
            record["saved_log_likelihood_error"] = saved_error
            record["matches_sampler"] = bool(saved_error <= tolerance)
    record["passed"] = bool(record["candidate_not_worse"] and record["scalar_consistent"]
                            and record["matches_sampler"])
    return record


def _procedure(settings: RefineSettings) -> dict[str, Any]:
    return {"procedure": settings.to_mapping(), "procedure_version": PROCEDURE_VERSION}


def refine(starts: Sequence[RefineStart], objective: BoxObjective, settings: RefineSettings) -> RefineOutcome:
    """Refine a role maximum from the sampler incumbent and the separated starts.

    ``starts[0]`` must be the sampler maximum-likelihood incumbent, followed by exactly
    ``settings.original_start_count`` original starts. Every start is checked against the
    objective's box before the objective is evaluated.
    """
    if len(starts) != settings.original_start_count + 1:
        raise ValueError(f"refinement needs the incumbent and {settings.original_start_count} starts, "
                         f"got {len(starts)} starts")
    incumbent = starts[0]
    if incumbent.source != "sampler_ml" or incumbent.original:
        raise ValueError("the first start must be the sampler maximum-likelihood incumbent")
    for start in starts:
        _check_start(start, objective)
    runs = [_run_start(start, objective, settings) for start in starts]
    start_records = [start.to_mapping() for start in starts]
    finite = [row for row in runs if row["observed_best_half_chi2"] is not None]
    if not finite:
        record = {"incumbent": _incumbent_record(incumbent, runs[0], None, None, settings),
                  "start_provenance": start_records, "runs": runs,
                  "candidate_start_agreement": _support_summary([], np.inf, settings),
                  "candidate_acceptance_status": RoleStatus.UNRESOLVED.value, **_procedure(settings)}
        gates = dict.fromkeys(("support", "repeat", "finite_gradient", "scalar_residual",
                               "direct_log_likelihood", "incumbent"), False)
        return RefineOutcome(acceptance_status=RoleStatus.UNRESOLVED, best_log_likelihood=None, best_vector=None,
                             gates=gates, record=record, projected_gradient_linf=None,
                             repeat_converged=None)
    best_run = min(finite, key=lambda row: float(row["observed_best_half_chi2"]))
    best_z = np.asarray(best_run["observed_best_z"], dtype=float)
    best_half = float(best_run["observed_best_half_chi2"])
    repeat_state = _empty_state()

    def tracked_repeat(z: np.ndarray) -> tuple[float, np.ndarray]:
        repeat_state["evaluation_count"] += 1
        value, gradient = objective.value_and_gradient(np.asarray(z, dtype=float))
        _record_best_finite(repeat_state, z, value, gradient, objective.to_physical(z))
        return value, gradient

    repeat_result = None
    repeat_exception = None
    repeat_endpoint_z = None
    repeat_endpoint_value = None
    repeat_endpoint_gradient = None
    try:
        tracked_repeat(best_z)
        repeat_result = scipy.optimize.minimize(
            tracked_repeat, best_z, method="L-BFGS-B", jac=True, bounds=[(0.0, 1.0)] * best_z.size,
            options={"maxiter": int(settings.repeat_maxiter), "ftol": float(settings.repeat_ftol),
                     "gtol": float(settings.repeat_gtol), "maxls": int(settings.maxls)})
        repeat_endpoint_z = np.asarray(repeat_result.x, dtype=float)
        repeat_endpoint_value, repeat_endpoint_gradient = tracked_repeat(repeat_endpoint_z)
    except Exception as error:  # a failed repeat fails the repeat gate
        repeat_exception = f"{type(error).__name__}: {error}"
    repeat_best = repeat_state["best_half_chi2"]
    if repeat_best is not None and float(repeat_best) < best_half:
        best_half = float(repeat_best)
        best_z = np.asarray(repeat_state["best_z"], dtype=float)
        best_gradient = repeat_state["best_gradient"]
    else:
        best_gradient = best_run["observed_best_gradient_z"]
    best_x = np.asarray(objective.to_physical(best_z), dtype=float)
    residual_value = np.asarray(objective.residual(best_z), dtype=float)
    residual_error = abs(2.0 * best_half - float(residual_value @ residual_value))
    scalar = objective.direct_check(best_z, best_half)
    direct_error = scalar.direct_log_likelihood_error
    support = _support_summary(runs, best_half, settings)
    repeat_change = None if repeat_best is None else abs(float(repeat_best) - float(best_run["observed_best_half_chi2"]))
    gradient_valid = best_gradient is not None and bool(np.all(np.isfinite(np.asarray(best_gradient))))
    incumbent_record = _incumbent_record(incumbent, runs[0], best_half, objective.direct_check, settings)
    gates = {
        "support": bool(support["support_passed"]),
        "repeat": bool(repeat_result is not None and repeat_change is not None
                       and repeat_change <= settings.repeat_log_likelihood_tolerance),
        "finite_gradient": gradient_valid,
        "scalar_residual": bool(residual_error <= settings.scalar_residual_tolerance),
        "direct_log_likelihood": bool(float(direct_error) <= settings.scalar_residual_tolerance),
        "incumbent": bool(incumbent_record["passed"]),
    }
    status = RoleStatus.ACCEPTED if all(gates.values()) else RoleStatus.UNRESOLVED
    projected = _projected_gradient(best_gradient, best_z)
    repeat_gradient_finite = (repeat_endpoint_gradient is not None
                              and bool(np.all(np.isfinite(np.asarray(repeat_endpoint_gradient)))))
    repeat_record = {
        "start_z": best_run["observed_best_z"],
        "solver_endpoint_z": None if repeat_endpoint_z is None else repeat_endpoint_z.tolist(),
        "solver_endpoint_half_chi2": repeat_endpoint_value,
        "solver_endpoint_gradient_valid": repeat_gradient_finite,
        "observed_best_half_chi2": repeat_best,
        "observed_best_z": repeat_state["best_z"],
        "observed_best_gradient_valid": repeat_state["best_gradient_valid"],
        "evaluation_count": repeat_state["evaluation_count"],
        "failed_evaluation_count": repeat_state["failed_evaluation_count"],
        "success": None if repeat_result is None else bool(repeat_result.success),
        "status": None if repeat_result is None else int(repeat_result.status),
        "message": repeat_exception if repeat_result is None else str(repeat_result.message),
    }
    record = {
        "incumbent": incumbent_record,
        "start_provenance": start_records,
        "runs": runs,
        "tighter_repeat": repeat_record,
        "candidate_best_vector": best_x.tolist(),
        "candidate_best_z": best_z.tolist(),
        "candidate_best_half_chi2": best_half,
        "candidate_best_chi2": 2.0 * best_half,
        "candidate_best_log_likelihood": scalar.direct_log_likelihood,
        "candidate_best_gradient_z": None if not gradient_valid else np.asarray(best_gradient).tolist(),
        "candidate_best_gradient_valid": gradient_valid,
        "candidate_best_projected_gradient": projected,
        "candidate_scalar_residual_error": float(residual_error),
        "candidate_direct_log_likelihood": scalar.direct_log_likelihood,
        "candidate_implied_log_likelihood": scalar.implied_log_likelihood,
        "candidate_direct_log_likelihood_error": direct_error,
        "candidate_start_agreement": support,
        "candidate_not_worse_than_incumbent": incumbent_record["candidate_not_worse"],
        "candidate_acceptance_status": status.value,
        **_procedure(settings),
    }
    direct = float(scalar.direct_log_likelihood)
    return RefineOutcome(acceptance_status=status, best_log_likelihood=direct if math.isfinite(direct) else None,
                         best_vector=tuple(float(value) for value in best_x), gates=gates,
                         record=record, projected_gradient_linf=projected["linf"],
                         repeat_converged=None if repeat_result is None else bool(repeat_result.success))
