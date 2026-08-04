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
    ftol: float = 1.0e-5,
    xtol: float = 1.0e-5,
    gtol: float = 1.0e-5,
    x_scale: str | Sequence[float] = "jac",
    reliability_note: str = "",
    selection_rel_tolerance: float = 1.0e-6,
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

    attempts: List[LocalFitAttempt] = []
    for label, point in zip(labels, points):
        result = least_squares(
            residual_fn,
            point,
            bounds=(lower, upper),
            method="trf",
            max_nfev=int(max_nfev),
            ftol=float(ftol),
            xtol=float(xtol),
            gtol=float(gtol),
            x_scale=x_scale,
        )
        residual = np.asarray(result.fun, dtype=float)
        attempts.append(
            LocalFitAttempt(
                label=str(label),
                success=bool(result.success),
                status=int(result.status),
                message=str(result.message),
                chi2=float(residual @ residual),
                x=[float(value) for value in np.asarray(result.x, dtype=float)],
                nfev=int(result.nfev),
                optimality=float(result.optimality),
            )
        )

    attempts.sort(key=lambda attempt: attempt.chi2)
    min_chi2 = float(attempts[0].chi2)
    selection_tol = float(selection_rel_tolerance) * max(abs(min_chi2), 1.0)
    successful_near_best = [
        attempt for attempt in attempts
        if attempt.success and attempt.chi2 <= min_chi2 + selection_tol
    ]
    best = successful_near_best[0] if successful_near_best else attempts[0]
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
) -> float:
    """Return the non-negative smooth-vs-subhalo profile statistic."""
    return float(max(0.0, float(smooth_chi2_min) - float(subhalo_chi2_min)))
