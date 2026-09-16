"""Local nonlinear profile-likelihood optimization utilities."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Callable, List, Optional, Sequence

import numpy as np
from scipy.optimize import least_squares

ResidualFunction = Callable[[np.ndarray], np.ndarray]


@dataclass(frozen=True)
class LocalFitAttempt:
    """One local least-squares attempt from one initialization."""

    label: str
    success: bool
    status: int
    message: str
    chi2: float
    x: List[float]
    nfev: int
    optimality: float
    residual_calls: int = 0
    jacobian_calls: int = 0
    endpoint_chi2: Optional[float] = None
    active_mask: Optional[List[int]] = None

    def to_dict(self) -> dict:
        """Return this attempt as a plain dictionary."""
        return asdict(self)


@dataclass(frozen=True)
class LocalProfileFitResult:
    """Best local fit and all attempts used to establish it."""

    model_name: str
    best: LocalFitAttempt
    attempts: List[LocalFitAttempt]
    convergence_abs_spread: Optional[float]
    convergence_rel_spread: Optional[float]
    reliability_note: str

    @property
    def chi2_min(self) -> float:
        """Chi-squared of the best attempt (`float`, read-only)."""
        return float(self.best.chi2)

    def to_dict(self) -> dict:
        """Return this fit result as a plain dictionary."""
        return asdict(self)


def _coerce_initial_points(initial_points: Sequence[Sequence[float]]) -> List[np.ndarray]:
    points = [np.asarray(point, dtype=float) for point in initial_points]
    if not points:
        raise ValueError("At least one initial point is required.")
    size = points[0].size
    for point in points:
        if point.ndim != 1:
            raise ValueError("Initial points must be one-dimensional arrays.")
        if point.size != size:
            raise ValueError("All initial points must have the same length.")
        if not np.all(np.isfinite(point)):
            raise ValueError("Initial points must be finite.")
    return points


def fit_local_least_squares_profile(
    *,
    model_name: str,
    residual_fn: ResidualFunction,
    initial_points: Sequence[Sequence[float]],
    labels: Optional[Sequence[str]] = None,
    lower_bounds: Optional[Sequence[float]] = None,
    upper_bounds: Optional[Sequence[float]] = None,
    max_nfev: int = 60,
    ftol: Optional[float] = 1.0e-5,
    xtol: float = 1.0e-5,
    gtol: float = 1.0e-5,
    x_scale: str | Sequence[float] = "jac",
    reliability_note: str = "",
    selection_rel_tolerance: float = 1.0e-6,
    progress_callback: Optional[Callable[[dict], None]] = None,
    jacobian_fn: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    attempt_callback: Optional[Callable[[LocalFitAttempt], None]] = None,
) -> LocalProfileFitResult:
    """Run multistart local least-squares profiling and return the best fit."""
    points = _coerce_initial_points(initial_points)
    n_params = points[0].size
    if labels is None:
        labels = [f"start_{idx}" for idx in range(len(points))]
    if len(labels) != len(points):
        raise ValueError("labels must match the number of initial points.")

    if lower_bounds is None:
        lower = np.full(n_params, -np.inf, dtype=float)
    else:
        lower = np.asarray(lower_bounds, dtype=float)
    if upper_bounds is None:
        upper = np.full(n_params, np.inf, dtype=float)
    else:
        upper = np.asarray(upper_bounds, dtype=float)
    if lower.shape != (n_params,) or upper.shape != (n_params,):
        raise ValueError("Bounds must match the initial-point dimensionality.")

    # Retain this legacy argument for callers, but never exchange objective
    # quality for a successful solver flag, at any relative tolerance.
    if selection_rel_tolerance < 0:
        raise ValueError("selection_rel_tolerance must be nonnegative")
    if np.any(lower >= upper):
        raise ValueError("Every lower bound must be below its upper bound")
    attempts: List[LocalFitAttempt] = []
    for label, point in zip(labels, points):
        calls = 0
        jacobian_calls = 0
        best_chi2 = np.inf
        best_x = point.copy()

        def tracked(x):
            nonlocal calls, best_chi2, best_x
            calls += 1
            x = np.asarray(x, dtype=float)
            if not np.all(np.isfinite(x)) or np.any(x < lower) or np.any(x > upper):
                raise ValueError("Objective point is outside admissible bounds")
            residual = np.asarray(residual_fn(x), dtype=float)
            if residual.ndim != 1 or not np.all(np.isfinite(residual)):
                raise ValueError("Residual must be a finite one-dimensional vector")
            chi2 = float(residual @ residual)
            if not np.isfinite(chi2):
                raise ValueError("Nonfinite chi-squared")
            if chi2 < best_chi2:
                best_chi2, best_x = chi2, x.copy()
                if progress_callback is not None:
                    progress_callback(
                        {"label": str(label), "chi2": chi2, "x": x.tolist(), "residual_calls": calls}
                    )
            return residual

        def tracked_jacobian(x):
            nonlocal jacobian_calls
            jacobian_calls += 1
            matrix = np.asarray(jacobian_fn(x), dtype=float)
            if matrix.ndim != 2 or matrix.shape[1] != n_params or not np.all(np.isfinite(matrix)):
                raise ValueError("Jacobian must be a finite matrix with one column per parameter")
            return matrix

        result = None
        error = None
        try:
            tracked(point)  # Preserve the initial point even if the solver fails.
            result = least_squares(
                tracked,
                point,
                bounds=(lower, upper),
                method="trf",
                max_nfev=int(max_nfev),
                ftol=None if ftol is None else float(ftol),
                xtol=float(xtol),
                gtol=float(gtol),
                x_scale=x_scale,
                jac=tracked_jacobian if jacobian_fn is not None else "2-point",
            )
            tracked(result.x)
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
        attempt = LocalFitAttempt(
            label=str(label),
            success=bool(result is not None and result.success and error is None),
            status=int(result.status) if result is not None else -1,
            message=error or str(result.message),
            chi2=float(best_chi2),
            x=[float(value) for value in best_x],
            nfev=int(result.nfev) if result is not None else 0,
            optimality=float(result.optimality) if result is not None else float("inf"),
            residual_calls=calls,
            jacobian_calls=jacobian_calls,
            endpoint_chi2=float(np.asarray(result.fun) @ np.asarray(result.fun))
            if result is not None
            else None,
            active_mask=np.asarray(result.active_mask, dtype=int).tolist() if result is not None else None,
        )
        attempts.append(attempt)
        if attempt_callback is not None:
            attempt_callback(attempt)
    finite_attempts = [attempt for attempt in attempts if np.isfinite(attempt.chi2)]
    if not finite_attempts:
        raise ValueError(
            "No finite admissible objective in any start: "
            + "; ".join(attempt.message for attempt in attempts)
        )
    best = min(finite_attempts, key=lambda attempt: attempt.chi2)
    if len(attempts) >= 2:
        chi2_values = np.asarray([attempt.chi2 for attempt in attempts], dtype=float)
        spread_abs = float(np.max(chi2_values) - np.min(chi2_values))
        spread_rel = float(spread_abs / max(abs(best.chi2), 1.0))
    else:
        spread_abs = None
        spread_rel = None

    return LocalProfileFitResult(
        model_name=str(model_name),
        best=best,
        attempts=attempts,
        convergence_abs_spread=spread_abs,
        convergence_rel_spread=spread_rel,
        reliability_note=str(reliability_note),
    )


def profile_likelihood_q(
    *,
    smooth_chi2_min: float,
    subhalo_chi2_min: float,
    clip_negative: bool = True,
) -> float:
    """Return the smooth-vs-subhalo profile statistic.

    Parameters
    ----------
    smooth_chi2_min : `float`
        Best smooth-model chi-squared.
    subhalo_chi2_min : `float`
        Best subhalo-model chi-squared.
    clip_negative : `bool`, optional
        Whether to clip the statistic at zero. ``False`` returns the
        signed contrast, preserving cases where the subhalo model fits
        worse than the smooth model.

    Returns
    -------
    q : `float`
        ``smooth_chi2_min - subhalo_chi2_min``, clipped at zero when
        ``clip_negative`` is True.
    """
    signed = float(smooth_chi2_min) - float(subhalo_chi2_min)
    if clip_negative:
        return float(max(0.0, signed))
    return signed
