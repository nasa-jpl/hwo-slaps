"""The half chi-square of one role on the unit box, its gradient and the scalar consistency check.

With ``x = lower + z (upper - lower)`` and ``r(x)`` the flattened normalized residual map of
``analysis.fit_from(model.instance_from_vector(x))``, the objective is ``f(z) = r . r / 2``, and
its gradient in ``z`` is the gradient in ``x`` times the box widths. ``BoxObjective`` is the only
input type of the optimiser; ``jax_objective`` builds it for an AutoLens analysis.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from .starts import prior_box

__all__ = ["BoxObjective", "ScalarCheck", "jax_objective"]


@dataclass(frozen=True)
class ScalarCheck:
    """The likelihood at a point evaluated directly, against the one the objective implies.

    ``implied_log_likelihood`` is ``-f - noise_normalization / 2``.
    """

    direct_log_likelihood: float
    implied_log_likelihood: float
    direct_log_likelihood_error: float


@dataclass(frozen=True, eq=False)
class BoxObjective:
    """``f(z)`` and ``df/dz`` on the unit box of the prior box ``[lower, upper]``, the residual
    vector whose half squared norm is ``f``, and the direct likelihood check at a point."""

    lower: np.ndarray
    upper: np.ndarray
    value_and_gradient: Callable[[np.ndarray], tuple[float, np.ndarray]]
    residual: Callable[[np.ndarray], np.ndarray]
    direct_check: Callable[[np.ndarray, float], ScalarCheck]

    def __post_init__(self) -> None:
        lower, upper, _ = prior_box(self.lower, self.upper)
        for name, array in (("lower", lower.copy()), ("upper", upper.copy())):
            array.setflags(write=False)
            object.__setattr__(self, name, array)

    def to_physical(self, z: np.ndarray) -> np.ndarray:
        """``lower + z (upper - lower)``."""
        return self.lower + np.asarray(z, dtype=float) * (self.upper - self.lower)


def _residual_array(value: Any, xp: Any) -> Any:
    """An AutoArray residual as a flat array without a host conversion."""
    return xp.asarray(getattr(value, "array", value)).reshape(-1)


def jax_objective(analysis: Any, model: Any, lower: Sequence[float], upper: Sequence[float], *,
                  check_gradient_domain: bool = False) -> BoxObjective:
    """The compiled objective of an AutoLens ``analysis`` and its AutoFit ``model``.

    ``jax.jit(jax.value_and_grad(f))`` and the compiled residual are built once here; the direct
    check evaluates ``analysis.log_likelihood_function`` on the numpy path.
    """
    import jax
    import jax.numpy as jnp

    lower_array = np.asarray(lower, dtype=float)
    upper_array = np.asarray(upper, dtype=float)
    widths = upper_array - lower_array

    def to_x(z: np.ndarray) -> np.ndarray:
        return lower_array + np.asarray(z, dtype=float) * widths

    def residual_jax(x: Any) -> Any:
        instance = model.instance_from_vector(vector=x, xp=jnp)
        fit = analysis.fit_from(instance=instance)
        return _residual_array(fit.normalized_residual_map, jnp)

    def half_chi2(x: Any) -> Any:
        residual = residual_jax(x)
        return 0.5 * jnp.vdot(residual, residual)

    if check_gradient_domain:
        from jax.experimental import checkify

        checked_gradient = jax.jit(checkify.checkify(jax.value_and_grad(half_chi2)))

        def value_and_grad(x):
            error, result = checked_gradient(x)
            error.throw()
            return result
    else:
        value_and_grad = jax.jit(jax.value_and_grad(half_chi2))
    residual_compiled = jax.jit(residual_jax)

    def value_and_gradient(z: np.ndarray) -> tuple[float, np.ndarray]:
        value, gradient_x = value_and_grad(to_x(z))
        return float(np.asarray(value)), np.asarray(gradient_x, dtype=float) * widths

    def residual(z: np.ndarray) -> np.ndarray:
        return np.asarray(residual_compiled(to_x(z)), dtype=float)

    def direct_check(z: np.ndarray, half_value: float) -> ScalarCheck:
        instance = model.instance_from_vector(vector=to_x(z).tolist())
        fit = analysis.fit_from(instance=instance)
        direct = float(analysis.log_likelihood_function(instance))
        implied = -float(half_value) - 0.5 * float(fit.noise_normalization)
        return ScalarCheck(direct_log_likelihood=direct, implied_log_likelihood=implied,
                           direct_log_likelihood_error=abs(direct - implied))

    return BoxObjective(lower=lower_array, upper=upper_array, value_and_gradient=value_and_gradient,
                        residual=residual, direct_check=direct_check)
