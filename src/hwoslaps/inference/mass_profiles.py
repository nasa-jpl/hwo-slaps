"""Inference-only mass-profile differentials; all backend primals unchanged.

PowerLaw: at slope2 the normalization cusp cancels, leaving the linear
Cartesian m=2 harmonic. At other slopes the public objective refuses a free
exact-circle shape gradient. Ordinary noncircular derivatives and every value
call remain the pinned parent operations.

Isothermal: the parent's near-circular series with its unused closed form
evaluated below one, so the reverse-mode gradient stays finite.
"""

import functools

import autoarray as aa
import autolens as al
import jax
import jax.numpy as jnp
import numpy as np
from autogalaxy.profiles.mass.total.isothermal import psi_from
from jax.custom_derivatives import SymbolicZero
from jax.experimental import checkify

from .hypotheses import ISOTHERMAL_SERIES_THRESHOLD

__all__ = ["PowerLaw", "Isothermal"]


def _parent_value(parameters, coordinates, *, exact_circle=False):
    cy, cx, e1, e2, theta, slope = parameters
    if exact_circle:
        # ell_comps are exactly zero here. As constants they stay on the circle;
        # AutoGalaxy nudges a traced exact-circle ell_comps off the origin.
        e1, e2 = 0.0, 0.0
    profile = al.mp.PowerLaw(centre=(cy, cx), ell_comps=(e1, e2), einstein_radius=theta, slope=slope)
    grid = al.Grid2DIrregular(values=coordinates, xp=jnp)
    return profile.deflections_yx_2d_from(grid=grid, xp=jnp).array


@functools.partial(jax.custom_jvp, nondiff_argnums=(0,))
def _deflections(constant_circle, parameters, coordinates):
    return _parent_value(parameters, coordinates, exact_circle=constant_circle)


def _deflections_jvp(constant_circle, primals, tangents):
    parameters, coordinates = primals
    dp, dg = tangents
    geometry_active = (any(not isinstance(dp[i], SymbolicZero) for i in (0, 1, 2, 3))
                       or not isinstance(dg, SymbolicZero))
    dp = tuple(jnp.zeros_like(p) if isinstance(d, SymbolicZero) else d
               for p, d in zip(parameters, dp, strict=True))
    if isinstance(dg, SymbolicZero):
        dg = jnp.zeros_like(coordinates)
    primal = _parent_value(parameters, coordinates, exact_circle=constant_circle)
    cy, cx, e1, e2, theta, slope = parameters

    def regular_circle(_):
        relative = coordinates - jnp.stack((cy, cx))
        at_centre = jnp.all(relative == 0.0, axis=1)
        if geometry_active:
            checkify.check(jnp.all(~at_centre),
                           "PowerLaw circle+slope2 gradient is undefined at an exactly coincident mass-centre "
                           "sample with active geometry; use value-only sampling or noncoincident samples")
        # Differentiate all other directions through the original parent at the
        # exact circle, with the ellipse held constant inside the constructor.
        _, original = jax.jvp(lambda p, g: _parent_value(p, g, exact_circle=True),
                              (parameters, coordinates), (dp, dg))
        phi = jnp.arctan2(relative[:, 0], relative[:, 1])
        cosine = dp[3] * jnp.cos(2.0 * phi) + dp[2] * jnp.sin(2.0 * phi)
        sine = dp[3] * jnp.sin(2.0 * phi) - dp[2] * jnp.cos(2.0 * phi)
        a = -theta / 3.0
        radial, angular = a * cosine, -2.0 * a * sine
        harmonic = jnp.stack((radial * jnp.sin(phi) + angular * jnp.cos(phi),
                              radial * jnp.cos(phi) - angular * jnp.sin(phi)), axis=-1)
        return original + harmonic

    def parent(_):
        return jax.jvp(lambda p, g: _parent_value(p, g, exact_circle=constant_circle),
                       (parameters, coordinates), (dp, dg))[1]

    differential = jax.lax.cond((e1 == 0.0) & (e2 == 0.0) & (slope == 2.0),
                                regular_circle, parent, operand=None)
    return primal, differential


_deflections.defjvp(_deflections_jvp, symbolic_zeros=True)


class PowerLaw(al.mp.PowerLaw):
    """The public parent primal with its regular circle+slope2 Cartesian differential."""

    @aa.decorators.to_vector_yx
    def deflections_yx_2d_from(self, grid, xp=np, **kwargs):
        if xp is np or getattr(grid, "is_transformed", False):
            return super().deflections_yx_2d_from(grid=grid, xp=xp, **kwargs).array
        # The parent primal of a concrete exact circle is evaluated on the circle, as the parent does.
        constant_circle = all(not isinstance(value, jax.core.Tracer) and float(value) == 0.0
                              for value in self.ell_comps)
        parameters = tuple(jnp.asarray(value) for value in (*self.centre, *self.ell_comps,
                                                           self.einstein_radius, self.slope))
        return _deflections(constant_circle, parameters, jnp.asarray(grid.array))


class Isothermal(al.mp.Isothermal):
    """The public parent primal with a finite near-circular reverse-mode gradient.

    AutoGalaxy evaluates the closed form that its series branch discards at s = 1,
    where arctanh is infinite on the minor axis; 0.5 keeps that branch finite.
    Remove this adapter once upstream evaluates it below one.
    """

    def deflections_yx_2d_from(self, grid, xp=np, **kwargs):
        if xp is np:
            return super().deflections_yx_2d_from(grid=grid, xp=xp, **kwargs)
        return self._deflections_yx_2d_from(grid=grid, xp=xp, **kwargs)

    @aa.decorators.to_vector_yx
    @aa.decorators.transform(rotate_back=True)
    def _deflections_yx_2d_from(self, grid, xp=np, **kwargs):
        axis_ratio = self.axis_ratio(xp)
        one_minus_q2 = 1 - axis_ratio**2
        small = one_minus_q2 < ISOTHERMAL_SERIES_THRESHOLD
        sqrt_one_minus_q2 = xp.sqrt(xp.where(small, 0.5, one_minus_q2))
        factor = 2.0 * self.einstein_radius_rescaled(xp) * axis_ratio / sqrt_one_minus_q2
        psi = psi_from(grid=grid, axis_ratio=axis_ratio, core_radius=0.0, xp=xp)
        deflection_y = xp.arctanh(xp.divide(xp.multiply(sqrt_one_minus_q2, grid.array[:, 0]), psi))
        deflection_x = xp.arctan(xp.divide(xp.multiply(sqrt_one_minus_q2, grid.array[:, 1]), psi))
        deflections = xp.multiply(factor, xp.vstack((deflection_y, deflection_x)).T)

        t_y = grid.array[:, 0] / psi
        t_x = grid.array[:, 1] / psi
        u_y = one_minus_q2 * t_y**2
        u_x = -one_minus_q2 * t_x**2
        series_y = t_y * (1.0 + u_y * (1.0 / 3.0 + u_y * (1.0 / 5.0 + u_y / 7.0)))
        series_x = t_x * (1.0 + u_x * (1.0 / 3.0 + u_x * (1.0 / 5.0 + u_x / 7.0)))
        deflections_series = xp.multiply(2.0 * self.einstein_radius_rescaled(xp) * axis_ratio,
                                         xp.vstack((series_y, series_x)).T)
        return xp.where(small, deflections_series, deflections)
