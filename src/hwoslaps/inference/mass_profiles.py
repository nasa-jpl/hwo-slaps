"""Inference-only regular circular slope-two EPL differential; all backend primals unchanged.

At slope2 the normalization cusp cancels, leaving the linear Cartesian m=2
harmonic. At other slopes the public objective refuses a free exact-circle
shape gradient. Ordinary noncircular derivatives and every value call remain
the pinned parent operations. Isothermal is not routed through this adapter.
"""

import autoarray as aa
import autolens as al
import jax
import jax.numpy as jnp
import numpy as np
from jax.custom_derivatives import SymbolicZero
from jax.experimental import checkify

__all__ = ["PowerLaw"]


def _parent_value(parameters, coordinates, *, stop_ellipse=False):
    cy, cx, e1, e2, theta, slope = parameters
    if stop_ellipse:
        e1, e2 = jax.lax.stop_gradient(e1), jax.lax.stop_gradient(e2)
    profile = al.mp.PowerLaw(centre=(cy, cx), ell_comps=(e1, e2), einstein_radius=theta, slope=slope)
    grid = al.Grid2DIrregular(values=coordinates, xp=jnp)
    return profile.deflections_yx_2d_from(grid=grid, xp=jnp).array


@jax.custom_jvp
def _deflections(parameters, coordinates):
    return _parent_value(parameters, coordinates)


def _deflections_jvp(primals, tangents):
    parameters, coordinates = primals
    dp, dg = tangents
    geometry_active = (any(not isinstance(dp[i], SymbolicZero) for i in (0, 1, 2, 3))
                       or not isinstance(dg, SymbolicZero))
    dp = tuple(jnp.zeros_like(p) if isinstance(d, SymbolicZero) else d
               for p, d in zip(parameters, dp, strict=True))
    if isinstance(dg, SymbolicZero):
        dg = jnp.zeros_like(coordinates)
    primal = _parent_value(parameters, coordinates)
    cy, cx, e1, e2, theta, slope = parameters

    def regular_circle(_):
        relative = coordinates - jnp.stack((cy, cx))
        at_centre = jnp.all(relative == 0.0, axis=1)
        if geometry_active:
            checkify.check(jnp.all(~at_centre),
                           "PowerLaw circle+slope2 gradient is undefined at an exactly coincident mass-centre "
                           "sample with active geometry; use value-only sampling or noncoincident samples")
        # Differentiate all other directions through the original parent with
        # only ellipse stopped inside the constructor (not a zero tangent value).
        _, original = jax.jvp(lambda p, g: _parent_value(p, g, stop_ellipse=True),
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
        return jax.jvp(_parent_value, (parameters, coordinates), (dp, dg))[1]

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
        parameters = tuple(jnp.asarray(value) for value in (*self.centre, *self.ell_comps,
                                                           self.einstein_radius, self.slope))
        return _deflections(parameters, jnp.asarray(grid.array))
