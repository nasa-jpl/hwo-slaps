"""Inference-only Exponential ellipticity derivatives at a circular profile.

The backend's radius primal and nonround differential are preserved. At ell_comps=(0,0),
the Cartesian differential removes the polar-coordinate singularity. Exact source-centre
samples have a translation cusp and are refused for gradients; value-only sampling is finite.
"""

from __future__ import annotations

import autoarray as aa
import autogalaxy as ag
import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental import checkify

__all__ = ["Exponential"]


def _parent_radius(ell_comps, grid):
    q = ag.convert.axis_ratio_from((ell_comps[0], ell_comps[1]), xp=jnp)
    elliptical = jnp.sqrt(jnp.add(jnp.square(grid[:, 1]), jnp.square(jnp.divide(grid[:, 0], q))))
    outer_q = ag.convert.axis_ratio_from((ell_comps[0], ell_comps[1]), xp=jnp)
    return jnp.multiply(jnp.sqrt(outer_q), elliptical)


@jax.custom_jvp
def _circular_radius(ell_comps, grid):
    return _parent_radius(ell_comps, grid)


@_circular_radius.defjvp
def _circular_radius_jvp(primals, tangents):
    ell_comps, grid = primals
    de, dg = tangents
    radius = _parent_radius(ell_comps, grid)
    y, x, dy, dx = grid[:, 0], grid[:, 1], dg[:, 0], dg[:, 1]
    differential = (y * dy + x * dx + (y ** 2 - x ** 2) * de[1] - 2.0 * x * y * de[0]) / radius
    return radius, differential


@jax.custom_jvp
def _supported_gradient(radius):
    return radius


@_supported_gradient.defjvp
def _supported_gradient_jvp(primals, tangents):
    radius, = primals
    derivative, = tangents
    checkify.check(jnp.all(radius != 0.0),
                   "Exponential gradient is undefined at an exactly zero-radius source-centre sample; "
                   "use value-only sampling or a sampling with no coincident source-centre point")
    return radius, derivative


class Exponential(ag.lp.Exponential):
    @aa.decorators.to_array
    def eccentric_radii_grid_from(self, grid, xp=np, **kwargs):
        parent = super().eccentric_radii_grid_from
        if xp is np:
            return parent(grid=grid, xp=xp, **kwargs)
        ell = jnp.asarray(self.ell_comps)
        def original(_):
            return jnp.asarray(parent(grid=grid, xp=xp, **kwargs).array)
        radius = jax.lax.cond(jnp.all(ell == 0.0), lambda _: _circular_radius(ell, jnp.asarray(grid.array)),
                              original, operand=None)
        return _supported_gradient(radius)
