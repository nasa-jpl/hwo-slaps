"""Bounded local optimization initialized by a current search.

The module is import-safe: JAX, AutoLens, AutoFit and SciPy are imported only
inside the execution seams that need them.  It has no campaign globals and
does not monkeypatch process state.  A caller supplies a fresh search output
directory and receives an explicit record of every start, endpoint, best
finite evaluation, repeat, and scalar consistency check.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, replace
import json
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np

from .autolens_runner import AutoLensFitRunner
from .output_schema import NonlinearFitSummary


from .profile_settings import FreshProfileSettings, OBJECTIVE_VERSION, PROCEDURE_VERSION
from .profile_calibration import (
    ZeroResidualAnchorRunner,
    likelihood_matched_tangent,
)

__all__ = [
    "PROCEDURE_VERSION",
    "OBJECTIVE_VERSION",
    "FreshProfileSettings",
    "CurrentSearchStart",
    "prior_box",
    "normalize_physical_vector",
    "denormalize_vector",
    "select_current_search_starts",
    "record_best_finite",
    "projected_gradient",
    "support_summary",
    "make_jax_objective",
    "optimize_current_search_profile",
    "FreshProfileRunner",
    "ZeroResidualAnchorRunner",
    "FreshProfileValidator",
    "likelihood_matched_tangent",
]


def _residual_array(value: Any, xp: Any) -> Any:
    """Unwrap an AutoArray residual without invoking host NumPy conversion."""
    return xp.asarray(getattr(value, "array", value)).reshape(-1)


@dataclass(frozen=True)
class CurrentSearchStart:
    """One vector retained from the current fresh search.

    ``physical_vector`` is the parameter vector in model units exactly as the
    sampler wrote it.  ``normalized_vector`` is its image in the unit box the
    optimizer works in.  The two are related by
    :func:`normalize_physical_vector` and are never interchangeable.
    """

    start_index: int
    physical_vector: tuple[float, ...]
    normalized_vector: tuple[float, ...]
    source_kind: str
    original_start: bool
    origin: Mapping[str, Any]

    @classmethod
    def from_physical(
        cls,
        *,
        start_index: int,
        physical_vector: Sequence[float],
        lower: Sequence[float],
        upper: Sequence[float],
        source_kind: str,
        original_start: bool,
        origin: Mapping[str, Any],
    ) -> "CurrentSearchStart":
        """Build a start from a physical vector, normalizing exactly once."""
        lower_array, upper_array, _ = prior_box(lower, upper)
        x = _finite_vector(physical_vector, lower_array, upper_array)
        z = normalize_physical_vector(x, lower_array, upper_array)
        return cls(
            start_index=int(start_index),
            physical_vector=tuple(float(item) for item in x),
            normalized_vector=tuple(float(item) for item in z),
            source_kind=source_kind,
            original_start=bool(original_start),
            origin=origin,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "start_index": self.start_index,
            "physical_vector": list(self.physical_vector),
            "normalized_vector": list(self.normalized_vector),
            "source_kind": self.source_kind,
            "original_start": self.original_start,
            "origin": dict(self.origin),
        }


def prior_box(
    lower: Sequence[float],
    upper: Sequence[float],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return finite prior bounds and their strictly positive widths."""
    lower_array = np.asarray(lower, dtype=float)
    upper_array = np.asarray(upper, dtype=float)
    if (
        lower_array.ndim != 1
        or lower_array.shape != upper_array.shape
        or not np.all(np.isfinite(lower_array))
        or not np.all(np.isfinite(upper_array))
        or np.any(upper_array <= lower_array)
    ):
        raise ValueError("invalid finite prior box")
    return lower_array, upper_array, upper_array - lower_array


def normalize_physical_vector(
    physical_vector: Sequence[float],
    lower: Sequence[float],
    upper: Sequence[float],
) -> np.ndarray:
    """Map a physical vector inside the prior box to unit-box coordinates.

    The mapping is ``z = (x - lower) / (upper - lower)``.  The round trip
    through :func:`denormalize_vector` is checked so that a start can never
    be evaluated at a point other than its physical origin.
    """
    lower_array, upper_array, widths = prior_box(lower, upper)
    x = _finite_vector(physical_vector, lower_array, upper_array)
    z = (x - lower_array) / widths
    if np.any(z < 0.0) or np.any(z > 1.0) or not np.all(np.isfinite(z)):
        raise ValueError("normalized start left the unit box")
    round_trip = lower_array + z * widths
    if not np.allclose(round_trip, x, rtol=1.0e-12, atol=1.0e-12 * np.max(widths)):
        raise ValueError("normalized start does not map back to its physical origin")
    return z


def denormalize_vector(
    normalized_vector: Sequence[float],
    lower: Sequence[float],
    upper: Sequence[float],
) -> np.ndarray:
    """Map unit-box coordinates back to physical parameters."""
    lower_array, _, widths = prior_box(lower, upper)
    z = np.asarray(normalized_vector, dtype=float)
    if (
        z.shape != lower_array.shape
        or not np.all(np.isfinite(z))
        or np.any(z < 0.0)
        or np.any(z > 1.0)
    ):
        raise ValueError("normalized vector has the wrong shape or lies outside the unit box")
    return lower_array + z * widths


def _finite_vector(vector: Sequence[float], lower: np.ndarray, upper: np.ndarray) -> np.ndarray:
    value = np.asarray(vector, dtype=float)
    if value.shape != lower.shape or not np.all(np.isfinite(value)):
        raise ValueError("current-search vector has the wrong shape or non-finite values")
    if np.any(value < lower) or np.any(value > upper):
        raise ValueError("current-search vector lies outside the prior support")
    return value


def _summary_ml(summary_path: Path, names: Sequence[str]) -> tuple[np.ndarray, Mapping[str, Any]]:
    payload = json.loads(Path(summary_path).read_text(encoding="utf-8"))
    try:
        sample = payload["arguments"]["max_log_likelihood_sample"]["arguments"]
        values = sample["kwargs"]["arguments"]
    except (KeyError, TypeError) as exc:
        raise ValueError("current search summary has no max-log-likelihood sample") from exc
    if not isinstance(values, Mapping) or set(values) != set(names):
        raise ValueError("current-search ML parameter names do not match the model")
    return (
        np.asarray([float(values[name]) for name in names], dtype=float),
        {"origin": "current_search_ml", "saved_log_likelihood": sample.get("log_likelihood")},
    )


def posterior_sigma_normalized(
    rows: Sequence[Mapping[str, Any]],
    names: Sequence[str],
    lower: np.ndarray,
    widths: np.ndarray,
) -> np.ndarray:
    """Weighted posterior standard deviation per parameter in unit-box units."""
    if not rows or "weight" not in rows[0]:
        raise ValueError("current search samples must carry posterior weights")
    vectors = np.asarray([[float(row[name]) for name in names] for row in rows], dtype=float)
    weights = np.asarray([float(row["weight"]) for row in rows], dtype=float)
    if vectors.ndim != 2 or vectors.shape[1] != lower.shape[0] or not np.all(np.isfinite(vectors)):
        raise ValueError("current search samples must be finite vectors of the model dimension")
    if not np.all(np.isfinite(weights)) or np.any(weights < 0) or weights.sum() <= 0:
        raise ValueError("current search sample weights must be finite, non-negative and not all zero")
    weights = weights / weights.sum()
    z = (vectors - lower) / widths
    mean = weights @ z
    sigma = np.sqrt(weights @ (z - mean) ** 2)
    if not np.all(np.isfinite(sigma)) or np.any(sigma <= 0):
        raise ValueError("posterior sigma must be positive and finite for every parameter")
    return sigma


def select_current_search_starts(
    summary_path: Path,
    samples_path: Path,
    names: Sequence[str],
    lower: Sequence[float],
    upper: Sequence[float],
    settings: FreshProfileSettings,
) -> tuple[list[CurrentSearchStart], np.ndarray, np.ndarray]:
    """Select the ML incumbent and separated samples from one fresh search.

    Two starts are distinct when they are at least
    ``start_separation_normalized_l2`` prior widths apart, the rule for broad
    posteriors, or at least ``start_separation_posterior_sigma`` posterior
    standard deviations apart, the scale a sharp (noiseless) posterior
    actually has. The posterior sigma is the weighted spread of the search's
    own samples in unit-box coordinates and is recorded with the incumbent.

    The returned starts carry the sampler's physical vectors and their
    unit-box images; the optimizer consumes only the latter.
    """
    lower_array, upper_array, widths = prior_box(lower, upper)
    ml, ml_origin = _summary_ml(Path(summary_path), names)
    selected = [_finite_vector(ml, lower_array, upper_array)]
    with Path(samples_path).open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream, skipinitialspace=True))
    posterior_sigma = posterior_sigma_normalized(rows, names, lower_array, widths)
    origins: list[Mapping[str, Any]] = [
        {
            **dict(ml_origin),
            "selection_rule": {
                "distinct_if": "prior_normalized_l2 >= start_separation_normalized_l2 "
                "or posterior_sigma_l2 >= start_separation_posterior_sigma",
                "start_separation_normalized_l2": float(settings.start_separation_normalized_l2),
                "start_separation_posterior_sigma": float(settings.start_separation_posterior_sigma),
                "posterior_sigma_normalized": posterior_sigma.tolist(),
            },
        }
    ]
    ranked = sorted(
        enumerate(rows),
        key=lambda pair: (-float(pair[1]["log_likelihood"]), pair[0]),
    )
    for row_index, row in ranked:
        vector = np.asarray([float(row[name]) for name in names], dtype=float)
        if vector.shape != lower_array.shape or not np.all(np.isfinite(vector)):
            continue
        if np.any(vector < lower_array) or np.any(vector > upper_array):
            continue
        deltas = [(vector - old) / widths for old in selected]
        prior_l2 = [float(np.linalg.norm(delta)) for delta in deltas]
        sigma_l2 = [float(np.linalg.norm(delta / posterior_sigma)) for delta in deltas]
        distinct = all(
            prior >= settings.start_separation_normalized_l2
            or sigma >= settings.start_separation_posterior_sigma
            for prior, sigma in zip(prior_l2, sigma_l2)
        )
        if not distinct:
            continue
        selected.append(vector)
        origins.append(
            {
                "origin": "current_search_sample",
                "row": int(row_index),
                "saved_log_likelihood": float(row["log_likelihood"]),
                "separation_prior_normalized_l2": min(prior_l2),
                "separation_posterior_sigma": min(sigma_l2),
            }
        )
        if len(selected) == settings.original_start_count + 1:
            break
    if len(selected) != settings.original_start_count + 1:
        raise ValueError(
            "current search did not supply the ML incumbent plus "
            f"{settings.original_start_count} separated samples"
        )
    starts = [
        CurrentSearchStart.from_physical(
            start_index=index,
            physical_vector=vector,
            lower=lower_array,
            upper=upper_array,
            source_kind="current_search_ml" if index == 0 else "current_search_sample",
            original_start=index != 0,
            origin=origin,
        )
        for index, (vector, origin) in enumerate(zip(selected, origins))
    ]
    return starts, lower_array, upper_array


def record_best_finite(
    state: dict[str, Any],
    z: Sequence[float],
    value: float,
    gradient: Sequence[float] | None,
    physical_x: Sequence[float],
) -> bool:
    """Retain a better finite admissible point, even with bad derivatives."""
    z_array = np.asarray(z, dtype=float)
    x_array = np.asarray(physical_x, dtype=float)
    scalar = float(value)
    if (
        not np.isfinite(scalar)
        or not np.all(np.isfinite(z_array))
        or not np.all(np.isfinite(x_array))
        or np.any(z_array < 0.0)
        or np.any(z_array > 1.0)
    ):
        return False
    old = state.get("best_half_chi2")
    if old is not None and scalar >= float(old):
        return False
    state.update(
        {
            "best_half_chi2": scalar,
            "best_chi2": 2.0 * scalar,
            "best_z": z_array.tolist(),
            "best_x": x_array.tolist(),
        }
    )
    gradient_array = None if gradient is None else np.asarray(gradient, dtype=float)
    valid = (
        gradient_array is not None
        and gradient_array.shape == z_array.shape
        and np.all(np.isfinite(gradient_array))
    )
    state["best_gradient_valid"] = bool(valid)
    state["best_gradient"] = gradient_array.tolist() if valid else None
    state["best_gradient_error"] = None if valid else "nonfinite_or_incompatible_gradient"
    return True


def projected_gradient(
    gradient: Sequence[float] | None,
    z: Sequence[float],
    tolerance: float = 1.0e-10,
) -> dict[str, Any]:
    """Return a finite normalized-box projected-gradient diagnostic."""
    z_array = np.asarray(z, dtype=float)
    gradient_array = None if gradient is None else np.asarray(gradient, dtype=float)
    if (
        gradient_array is None
        or gradient_array.shape != z_array.shape
        or not np.all(np.isfinite(gradient_array))
    ):
        return {"valid": False, "linf": None, "l2": None, "values": None}
    value = gradient_array.copy()
    value[(z_array <= tolerance) & (value > 0.0)] = 0.0
    value[(z_array >= 1.0 - tolerance) & (value < 0.0)] = 0.0
    return {
        "valid": True,
        "linf": float(np.max(np.abs(value))),
        "l2": float(np.linalg.norm(value)),
        "values": value.tolist(),
    }


def support_summary(
    runs: Iterable[Mapping[str, Any]],
    best_half_chi2: float,
    settings: FreshProfileSettings,
) -> dict[str, Any]:
    """Count unique original sample starts supporting the retained best."""
    finite = [
        row for row in runs
        if row.get("observed_best_half_chi2") is not None
        and np.isfinite(float(row["observed_best_half_chi2"]))
    ]
    seen: set[int] = set()
    duplicate: list[int] = []
    original: list[Mapping[str, Any]] = []
    for row in finite:
        provenance = row.get("start_provenance") or {}
        if not provenance.get("original_start", False):
            continue
        index = int(row["start_index"])
        if index in seen:
            duplicate.append(index)
            continue
        seen.add(index)
        original.append(row)
    supporting = [
        int(row["start_index"])
        for row in original
        if float(row["observed_best_half_chi2"]) - float(best_half_chi2)
        <= settings.support_log_likelihood_tolerance
    ]
    return {
        "finite_start_count": len(finite),
        "original_start_count": len(original),
        "supporting_original_start_indices": supporting,
        "duplicate_original_start_indices": sorted(set(duplicate)),
        "minimum_supporting_original_starts": settings.minimum_distinct_original_start_support,
        "support_tolerance_log_likelihood": settings.support_log_likelihood_tolerance,
        "support_passed": len(supporting) >= settings.minimum_distinct_original_start_support,
    }


def make_jax_objective(
    analysis: Any,
    model: Any,
    lower: Sequence[float],
    upper: Sequence[float],
) -> tuple[
    Callable[[np.ndarray], tuple[float, np.ndarray]],
    Callable[[np.ndarray], np.ndarray],
    Callable[[np.ndarray], np.ndarray],
    Callable[[np.ndarray, float], dict[str, Any]],
]:
    """Build the JAX objective and direct scalar consistency callback."""
    import jax
    import jax.numpy as jnp

    lower_array = np.asarray(lower, dtype=float)
    widths = np.asarray(upper, dtype=float) - lower_array

    def to_x(z: np.ndarray) -> np.ndarray:
        return lower_array + np.asarray(z, dtype=float) * widths

    def residual_jax(x: Any) -> Any:
        instance = model.instance_from_vector(vector=x, xp=jnp)
        fit = analysis.fit_from(instance=instance)
        return _residual_array(fit.normalized_residual_map, jnp)

    def half_chi2(x: Any) -> Any:
        residual = residual_jax(x)
        return 0.5 * jnp.vdot(residual, residual)

    value_and_grad = jax.jit(jax.value_and_grad(half_chi2))
    residual_compiled = jax.jit(residual_jax)

    def objective(z: np.ndarray) -> tuple[float, np.ndarray]:
        value, gradient_x = value_and_grad(to_x(z))
        return float(np.asarray(value)), np.asarray(gradient_x, dtype=float) * widths

    def residual(z: np.ndarray) -> np.ndarray:
        return np.asarray(residual_compiled(to_x(z)), dtype=float)

    def direct_check(z: np.ndarray, half_value: float) -> dict[str, Any]:
        x = to_x(z)
        residual_value = residual(z)
        residual_error = abs(2.0 * float(half_value) - float(residual_value @ residual_value))
        instance = model.instance_from_vector(vector=x.tolist())
        fit = analysis.fit_from(instance=instance)
        direct_log_likelihood = float(analysis.log_likelihood_function(instance))
        implied_log_likelihood = -float(half_value) - 0.5 * float(fit.noise_normalization)
        return {
            "physical_vector": x.tolist(),
            "scalar_residual_error": float(residual_error),
            "direct_log_likelihood": direct_log_likelihood,
            "implied_log_likelihood": implied_log_likelihood,
            "direct_log_likelihood_error": abs(direct_log_likelihood - implied_log_likelihood),
        }

    return objective, residual, to_x, direct_check


def _run_one_start(
    start: CurrentSearchStart,
    objective: Callable[[np.ndarray], tuple[float, np.ndarray]],
    to_x: Callable[[np.ndarray], np.ndarray],
    settings: FreshProfileSettings,
    progress: Callable[[dict[str, Any]], None] | None,
) -> dict[str, Any]:
    import scipy.optimize

    z0 = np.asarray(start.normalized_vector, dtype=float)
    x0 = np.asarray(start.physical_vector, dtype=float)
    if (
        z0.ndim != 1
        or z0.shape != x0.shape
        or not np.all(np.isfinite(z0))
        or np.any(z0 < 0.0)
        or np.any(z0 > 1.0)
    ):
        raise ValueError(
            f"start {start.start_index} is not a finite unit-box vector; "
            "the objective was not evaluated"
        )
    mapped = np.asarray(to_x(z0), dtype=float)
    if not np.allclose(mapped, x0, rtol=1.0e-9, atol=1.0e-12):
        raise ValueError(
            f"start {start.start_index} does not map back to its physical origin; "
            "the objective was not evaluated"
        )
    state: dict[str, Any] = {
        "best_half_chi2": None,
        "best_chi2": None,
        "best_z": None,
        "best_x": None,
        "best_gradient": None,
        "best_gradient_valid": False,
        "best_gradient_error": None,
        "evaluation_count": 0,
        "failed_evaluation_count": 0,
    }

    def tracked(z: np.ndarray) -> tuple[float, np.ndarray]:
        state["evaluation_count"] += 1
        try:
            value, gradient = objective(np.asarray(z, dtype=float))
        except Exception:
            state["failed_evaluation_count"] += 1
            raise
        if record_best_finite(state, z, value, gradient, to_x(z)) and progress is not None:
            progress({"phase": "multistart", "start_index": start.start_index, **state})
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
            tracked,
            z0,
            method="L-BFGS-B",
            jac=True,
            bounds=[(0.0, 1.0)] * z0.size,
            options={
                "maxiter": int(settings.maxiter),
                "ftol": float(settings.ftol),
                "gtol": float(settings.gtol),
                "maxls": int(settings.maxls),
            },
        )
        endpoint_z = np.asarray(solver_result.x, dtype=float)
        endpoint_value, endpoint_gradient = tracked(endpoint_z)
    except Exception as exc:
        exception = f"{type(exc).__name__}: {exc}"
    return {
        "start_index": start.start_index,
        "start_provenance": start.to_dict(),
        "start_z": z0.tolist(),
        "start_x": x0.tolist(),
        "start_half_chi2": start_value,
        "solver_endpoint_z": None if endpoint_z is None else endpoint_z.tolist(),
        "solver_endpoint_x": None if endpoint_z is None else to_x(endpoint_z).tolist(),
        "solver_endpoint_half_chi2": endpoint_value,
        "solver_endpoint_chi2": None if endpoint_value is None else 2.0 * endpoint_value,
        "solver_endpoint_gradient_z": None
        if endpoint_gradient is None or not np.all(np.isfinite(endpoint_gradient))
        else np.asarray(endpoint_gradient).tolist(),
        "solver_endpoint_gradient_valid": bool(
            endpoint_gradient is not None and np.all(np.isfinite(endpoint_gradient))
        ),
        "solver_endpoint_projected_gradient": None
        if endpoint_z is None
        else projected_gradient(endpoint_gradient, endpoint_z),
        "observed_best_z": state["best_z"],
        "observed_best_x": state["best_x"],
        "observed_best_half_chi2": state["best_half_chi2"],
        "observed_best_chi2": state["best_chi2"],
        "observed_best_gradient_z": state["best_gradient"],
        "observed_best_gradient_valid": state["best_gradient_valid"],
        "observed_best_gradient_error": state["best_gradient_error"],
        "observed_best_projected_gradient": None
        if state["best_z"] is None
        else projected_gradient(state["best_gradient"], state["best_z"]),
        "evaluation_count": state["evaluation_count"],
        "failed_evaluation_count": state["failed_evaluation_count"],
        "success": None if solver_result is None else bool(solver_result.success),
        "status": None if solver_result is None else int(solver_result.status),
        "message": exception if solver_result is None else str(solver_result.message),
        "nit": 0 if solver_result is None else int(getattr(solver_result, "nit", 0)),
        "nfev": state["evaluation_count"],
        "njev": 0 if solver_result is None else int(getattr(solver_result, "njev", 0)),
    }


def _incumbent_record(
    incumbent: CurrentSearchStart,
    incumbent_run: Mapping[str, Any],
    best_half_chi2: float | None,
    direct_check: Callable[[np.ndarray, float], Mapping[str, Any]] | None,
    settings: FreshProfileSettings,
) -> dict[str, Any]:
    """Check the current-search ML incumbent against the retained maximum.

    Three conditions must hold at the direct-evaluation tolerance: the
    incumbent evaluates finitely on the objective, the retained maximum is
    not worse than it, and the sampler's saved likelihood at the incumbent
    agrees with the direct evaluation there.
    """
    tolerance = settings.scalar_residual_tolerance
    half = incumbent_run.get("start_half_chi2")
    evaluated = half is not None and np.isfinite(float(half))
    record: dict[str, Any] = {
        "physical_vector": list(incumbent.physical_vector),
        "normalized_vector": list(incumbent.normalized_vector),
        "half_chi2": None if not evaluated else float(half),
        "chi2": None if not evaluated else 2.0 * float(half),
        "evaluated": bool(evaluated),
        "saved_log_likelihood": incumbent.origin.get("saved_log_likelihood"),
        "direct_log_likelihood": None,
        "direct_log_likelihood_error": None,
        "saved_log_likelihood_error": None,
        "candidate_not_worse": False,
        "scalar_consistent": False,
        "matches_current_search": False,
        "tolerance": tolerance,
        "passed": False,
    }
    if not evaluated:
        return record
    if best_half_chi2 is not None and np.isfinite(best_half_chi2):
        record["candidate_not_worse"] = bool(float(best_half_chi2) <= float(half) + tolerance)
    if direct_check is not None:
        scalar = dict(direct_check(np.asarray(incumbent.normalized_vector, dtype=float), float(half)))
        direct = scalar.get("direct_log_likelihood")
        direct_error = scalar.get("direct_log_likelihood_error")
        record["direct_log_likelihood"] = direct
        record["direct_log_likelihood_error"] = direct_error
        record["scalar_consistent"] = bool(
            direct_error is not None
            and np.isfinite(float(direct_error))
            and float(direct_error) <= tolerance
        )
        saved = record["saved_log_likelihood"]
        if (
            direct is not None
            and isinstance(saved, (int, float))
            and not isinstance(saved, bool)
            and np.isfinite(float(saved))
            and np.isfinite(float(direct))
        ):
            saved_error = abs(float(saved) - float(direct))
            record["saved_log_likelihood_error"] = saved_error
            record["matches_current_search"] = bool(saved_error <= tolerance)
    record["passed"] = bool(
        record["candidate_not_worse"]
        and record["scalar_consistent"]
        and record["matches_current_search"]
    )
    return record


def optimize_current_search_profile(
    starts: Sequence[CurrentSearchStart],
    objective: Callable[[np.ndarray], tuple[float, np.ndarray]],
    residual: Callable[[np.ndarray], np.ndarray],
    to_x: Callable[[np.ndarray], np.ndarray],
    direct_check: Callable[[np.ndarray, float], Mapping[str, Any]] | None,
    settings: FreshProfileSettings,
    *,
    progress: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """Run fresh bounded L-BFGS-B starts and a tighter repeat.

    The first start must be the current-search maximum-likelihood incumbent.
    Its objective value is a floor on the returned maximum: a repeatable
    endpoint worse than the incumbent never replaces it, and an incumbent
    that no original start supports leaves the profile unresolved.
    """
    if len(starts) != settings.original_start_count + 1:
        raise ValueError("optimizer received an incomplete current-search start set")
    incumbent = starts[0]
    if incumbent.source_kind != "current_search_ml" or incumbent.original_start:
        raise ValueError("the first start must be the current-search ML incumbent")
    runs = [_run_one_start(start, objective, to_x, settings, progress) for start in starts]
    finite = [row for row in runs if row["observed_best_half_chi2"] is not None]
    if not finite:
        return {
            "procedure_version": settings.version,
            "candidate_acceptance_status": "unresolved_optimization",
            "fresh_start_count": len(starts),
            "fresh_original_start_count": settings.original_start_count,
            "current_ml_incumbent": incumbent.to_dict(),
            "incumbent": _incumbent_record(incumbent, runs[0], None, None, settings),
            "start_provenance": [start.to_dict() for start in starts],
            "runs": runs,
            "candidate_start_agreement": support_summary([], np.inf, settings),
            "old_convergence_transferred": False,
            "fallback_solver": False,
        }
    best_run = min(finite, key=lambda row: float(row["observed_best_half_chi2"]))
    best_z = np.asarray(best_run["observed_best_z"], dtype=float)
    best_half = float(best_run["observed_best_half_chi2"])
    repeat_state: dict[str, Any] = {
        "best_half_chi2": None,
        "best_chi2": None,
        "best_z": None,
        "best_x": None,
        "best_gradient": None,
        "best_gradient_valid": False,
        "best_gradient_error": None,
        "evaluation_count": 0,
        "failed_evaluation_count": 0,
    }

    def tracked_repeat(z: np.ndarray) -> tuple[float, np.ndarray]:
        repeat_state["evaluation_count"] += 1
        value, gradient = objective(np.asarray(z, dtype=float))
        if record_best_finite(repeat_state, z, value, gradient, to_x(z)) and progress is not None:
            progress({"phase": "tighter_repeat", **repeat_state})
        return value, gradient

    repeat_result = None
    repeat_exception = None
    repeat_endpoint_z = None
    repeat_endpoint_value = None
    repeat_endpoint_gradient = None
    try:
        tracked_repeat(best_z)
        import scipy.optimize

        repeat_result = scipy.optimize.minimize(
            tracked_repeat,
            best_z,
            method="L-BFGS-B",
            jac=True,
            bounds=[(0.0, 1.0)] * best_z.size,
            options={
                "maxiter": int(settings.repeat_maxiter),
                "ftol": float(settings.repeat_ftol),
                "gtol": float(settings.repeat_gtol),
                "maxls": int(settings.maxls),
            },
        )
        repeat_endpoint_z = np.asarray(repeat_result.x, dtype=float)
        repeat_endpoint_value, repeat_endpoint_gradient = tracked_repeat(repeat_endpoint_z)
    except Exception as exc:
        repeat_exception = f"{type(exc).__name__}: {exc}"
    repeat_best = repeat_state["best_half_chi2"]
    if repeat_best is not None and float(repeat_best) < best_half:
        best_half = float(repeat_best)
        best_z = np.asarray(repeat_state["best_z"], dtype=float)
        best_gradient = repeat_state["best_gradient"]
    else:
        best_gradient = best_run["observed_best_gradient_z"]
    best_x = np.asarray(to_x(best_z), dtype=float)
    residual_value = np.asarray(residual(best_z), dtype=float)
    residual_error = abs(2.0 * best_half - float(residual_value @ residual_value))
    scalar = {} if direct_check is None else dict(direct_check(best_z, best_half))
    direct_error = scalar.get("direct_log_likelihood_error")
    support = support_summary(runs, best_half, settings)
    repeat_change = (
        None
        if repeat_best is None
        else abs(float(repeat_best) - float(best_run["observed_best_half_chi2"]))
    )
    gradient_valid = best_gradient is not None and np.all(np.isfinite(np.asarray(best_gradient)))
    incumbent_record = _incumbent_record(
        incumbent, runs[0], best_half, direct_check, settings
    )
    accepted = bool(
        support["support_passed"]
        and repeat_result is not None
        and repeat_change is not None
        and repeat_change <= settings.repeat_log_likelihood_tolerance
        and gradient_valid
        and residual_error <= settings.scalar_residual_tolerance
        and direct_error is not None
        and float(direct_error) <= settings.scalar_residual_tolerance
        and incumbent_record["passed"]
    )
    repeat_record = {
        "start_provenance": {"origin": "tighter_repeat_of_current_search_best", "original_start": False},
        "start_z": best_run["observed_best_z"],
        "solver_endpoint_z": None if repeat_endpoint_z is None else repeat_endpoint_z.tolist(),
        "solver_endpoint_half_chi2": repeat_endpoint_value,
        "solver_endpoint_gradient_valid": bool(
            repeat_endpoint_gradient is not None and np.all(np.isfinite(np.asarray(repeat_endpoint_gradient)))
        ),
        "observed_best_half_chi2": repeat_best,
        "observed_best_z": repeat_state["best_z"],
        "observed_best_gradient_valid": repeat_state["best_gradient_valid"],
        "evaluation_count": repeat_state["evaluation_count"],
        "failed_evaluation_count": repeat_state["failed_evaluation_count"],
        "success": None if repeat_result is None else bool(repeat_result.success),
        "status": None if repeat_result is None else int(repeat_result.status),
        "message": repeat_exception if repeat_result is None else str(repeat_result.message),
    }
    return {
        "procedure_version": settings.version,
        "optimizer": "L-BFGS-B",
        "parameterization": "normalized_box_0_1",
        "fresh_start_count": len(starts),
        "fresh_original_start_count": settings.original_start_count,
        "current_ml_incumbent": incumbent.to_dict(),
        "incumbent": incumbent_record,
        "candidate_not_worse_than_incumbent": incumbent_record["candidate_not_worse"],
        "start_provenance": [start.to_dict() for start in starts],
        "runs": runs,
        "tighter_repeat": repeat_record,
        "candidate_best_vector": best_x.tolist(),
        "candidate_best_z": best_z.tolist(),
        "candidate_best_half_chi2": best_half,
        "candidate_best_chi2": 2.0 * best_half,
        "candidate_best_log_likelihood": scalar.get("direct_log_likelihood"),
        "candidate_best_gradient_z": (
            None if not gradient_valid else np.asarray(best_gradient).tolist()
        ),
        "candidate_best_gradient_valid": bool(gradient_valid),
        "candidate_best_projected_gradient": projected_gradient(best_gradient, best_z),
        "candidate_scalar_residual_error": float(residual_error),
        "candidate_direct_log_likelihood": scalar.get("direct_log_likelihood"),
        "candidate_implied_log_likelihood": scalar.get("implied_log_likelihood"),
        "candidate_direct_log_likelihood_error": direct_error,
        "candidate_start_agreement": support,
        "candidate_acceptance_status": (
            "accepted_repeatable_profile" if accepted else "unresolved_optimization"
        ),
        "old_convergence_transferred": False,
        "fallback_solver": False,
        "fresh_corrected_objective_only": True,
    }


class FreshProfileRunner(AutoLensFitRunner):
    """Replace fresh sampler maxima with verified local maxima."""

    def __init__(
        self,
        settings: Any,
        output_dir: str,
        profile_settings: FreshProfileSettings | None = None,
    ):
        super().__init__(settings, output_dir)
        self.profile_settings = profile_settings or FreshProfileSettings()
        self.profile_records: dict[str, dict[str, Any]] = {}

    def run_model(
        self,
        *,
        model: Any,
        analysis: Any,
        role: str,
        **kwargs: Any,
    ) -> NonlinearFitSummary:
        summary = super().run_model(model=model, analysis=analysis, role=role, **kwargs)
        callback_errors = [
            warning
            for warning in summary.warnings
            if str(warning).startswith("result_callback failed:")
        ]
        if callback_errors:
            error = "fresh profile blocked by result callback failure: " + "; ".join(
                str(item) for item in callback_errors
            )
            self.profile_records[role] = {
                "procedure_version": self.profile_settings.version,
                "candidate_acceptance_status": "unresolved_callback_error",
                "fresh_search_status": summary.status,
                "fresh_search_error": error,
                "old_convergence_transferred": False,
            }
            return replace(summary, status="failed", error=error)
        if summary.status != "success" or summary.result_path is None:
            self.profile_records[role] = {
                "procedure_version": self.profile_settings.version,
                "candidate_acceptance_status": "failed_fresh_search",
                "fresh_search_status": summary.status,
                "fresh_search_error": summary.error,
                "old_convergence_transferred": False,
            }
            return summary
        result_path = Path(summary.result_path).resolve()
        output_root = Path(self.output_dir).resolve()
        if not result_path.is_relative_to(output_root):
            raise RuntimeError("fresh search result escaped the case output namespace")
        names = [".".join(path) for path in model.unique_prior_paths]
        lower = np.asarray([prior.lower_limit for prior in model.priors_ordered_by_id], dtype=float)
        upper = np.asarray([prior.upper_limit for prior in model.priors_ordered_by_id], dtype=float)
        starts, lower, upper = select_current_search_starts(
            result_path / "files" / "samples_summary.json",
            result_path / "files" / "samples.csv",
            names,
            lower,
            upper,
            self.profile_settings,
        )
        objective, residual, to_x, direct_check = make_jax_objective(
            analysis, model, lower, upper
        )
        progress_path = output_root / "fresh_profile_progress.json"

        def progress(record: dict[str, Any]) -> None:
            progress_path.write_text(
                json.dumps({"role": role, "active": record}, indent=2, allow_nan=False) + "\n",
                encoding="utf-8",
            )

        profile = optimize_current_search_profile(
            starts,
            objective,
            residual,
            to_x,
            direct_check,
            self.profile_settings,
            progress=progress,
        )
        profile["procedure"] = self.profile_settings.to_dict()
        profile["fresh_search_result_path"] = str(result_path)
        profile["fresh_search_analysis_key"] = summary.analysis_key
        profile["fresh_search_summary"] = summary.to_dict()
        self.profile_records[role] = profile
        best_log_likelihood = profile.get("candidate_best_log_likelihood")
        if best_log_likelihood is None:
            return replace(
                summary,
                status="failed",
                log_likelihood_max=None,
                figure_of_merit_max=None,
                error="fresh local profile produced no finite retained evaluation",
            )
        warnings = list(summary.warnings)
        warnings.append("local_profile_replaced_sampler_likelihood_max")
        return replace(
            summary,
            log_likelihood_max=float(best_log_likelihood),
            figure_of_merit_max=float(best_log_likelihood),
            log_likelihood_extraction_method="fresh_profile_best_observed",
            warnings=warnings,
        )


class FreshProfileValidator:
    """Delegate the pair fit and optionally attach the tangent comparator."""

    def __init__(self, runner: Any, *, compute_comparator: bool = False):
        from .validator import NonlinearMetricValidator

        self.runner = runner
        self._delegate = NonlinearMetricValidator(runner)
        self.compute_comparator = bool(compute_comparator)

    def validate_case(
        self,
        dataset: Any,
        dataset_metadata: Any,
        full_config: Any,
        trial: Any,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        preflight = getattr(self.runner, "preflight_anchor", None)
        if callable(preflight):
            preflight(
                dataset,
                full_config,
                trial,
                fit_mode=str(kwargs.get("fit_mode", "fixed_template")),
                mass_context=kwargs.get("mass_context"),
            )
        result = self._delegate.validate_case(
            dataset,
            dataset_metadata,
            full_config,
            trial,
            *args,
            **kwargs,
        )
        if not self.compute_comparator:
            return result
        comparator = likelihood_matched_tangent(
            self.runner,
            dataset,
            full_config,
            trial,
            fit_mode=str(kwargs.get("fit_mode", "fixed_template")),
            mass_context=kwargs.get("mass_context"),
        )
        self.runner.profile_records["likelihood_matched_tangent"] = comparator
        result.diagnostics["likelihood_matched_tangent"] = comparator
        return result
