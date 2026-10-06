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
from jax.custom_derivatives import SymbolicZero

__all__ = ["Exponential", "Sersic"]


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


@jax.custom_jvp
def _sersic_radius(ell_comps, grid, sersic_index):
    return _parent_radius(ell_comps, grid)


def _sersic_radius_jvp(primals, tangents):
    ell, grid, n = primals
    de, dg, _ = tangents
    radius = _parent_radius(ell, grid)
    if isinstance(de, SymbolicZero) and isinstance(dg, SymbolicZero):
        # Index-only differentiation cannot move a radius. In particular,
        # I(0)=I_e exp(b_n) has a finite index partial for every supported n.
        return radius, jnp.zeros_like(radius)
    if isinstance(de, SymbolicZero):
        de = jnp.zeros_like(ell)
    if isinstance(dg, SymbolicZero):
        dg = jnp.zeros_like(grid)
    centre = jnp.all(grid == 0.0, axis=1)
    checkify.check(jnp.all(~centre | (n < 1.0)),
                   "Sersic gradient is undefined at an exactly zero-radius source-centre sample for n >= 1; "
                   "use value-only sampling or a sampling with no coincident source-centre point")

    def circular(_):
        y, x, dy, dx = grid[:, 0], grid[:, 1], dg[:, 0], dg[:, 1]
        denominator = jnp.where(centre, 1.0, radius)
        return (y * dy + x * dx + (y ** 2 - x ** 2) * de[1] - 2.0 * x * y * de[0]) / denominator

    def nonround(_):
        # Radius has no derivative at zero. The composed brightness does for n<1;
        # evaluate the nonround radius JVP only at safe nonzero points and set its
        # centre tangent to zero, where the brightness differential is zero.
        safe = jnp.where(centre[:, None], jnp.array([1.0, 0.0]), grid)
        safe_dg = jnp.where(centre[:, None], 0.0, dg)
        _, differential = jax.jvp(_parent_radius, (ell, safe), (de, safe_dg))
        return jnp.where(centre, 0.0, differential)

    differential = jax.lax.cond(jnp.all(ell == 0.0), circular, nonround, operand=None)
    return radius, differential


_sersic_radius.defjvp(_sersic_radius_jvp, symbolic_zeros=True)


@jax.custom_jvp
def _sersic_power(radius_over_effective, n):
    return jnp.power(radius_over_effective, 1.0 / n)


@_sersic_power.defjvp
def _sersic_power_jvp(primals, tangents):
    u, n = primals
    du, dn = tangents
    power = jnp.power(u, 1.0 / n)
    safe = jnp.where(u == 0.0, 1.0, u)
    # For n<1, d(u^(1/n))/du and d(u^(1/n))/dn are zero at u=0.
    derivative_u = jnp.where(u == 0.0, 0.0, (1.0 / n) * jnp.power(safe, 1.0 / n - 1.0))
    derivative_n = -power * jnp.log(safe) / n ** 2
    return power, derivative_u * du + derivative_n * dn


@jax.custom_jvp
def _sersic_transform(grid, centre, angle):
    return aa.util.geometry.transform_grid_2d_to_reference_frame(
        grid_2d=grid, centre=centre, angle=angle, xp=jnp)


@_sersic_transform.defjvp
def _sersic_transform_jvp(primals, tangents):
    grid, centre, angle = primals
    dg, dc, da = tangents
    primal = _sersic_transform(grid, centre, angle)
    coincident = jnp.all(grid == centre, axis=1)
    # AutoArray's polar transform has an undefined norm/atan2 derivative at
    # a coincident point. Keep its primal and its noncoincident differential.
    safe = jnp.where(coincident[:, None], centre + jnp.array([1.0, 0.0]), grid)
    _, original = jax.jvp(lambda g, c, a: aa.util.geometry.transform_grid_2d_to_reference_frame(
        grid_2d=g, centre=c, angle=a, xp=jnp), (safe, centre, angle), (dg, dc, da))
    radians = jnp.radians(angle)
    translated = dg - dc
    dy, dx = translated[:, 0], translated[:, 1]
    at_centre = jnp.stack((dy * jnp.cos(radians) - dx * jnp.sin(radians),
                          dx * jnp.cos(radians) + dy * jnp.sin(radians)), axis=-1)
    return primal, jnp.where(coincident[:, None], at_centre, original)


class Sersic(ag.lp.Sersic):
    """Exact backend values with Cartesian circular and n-dependent centre differentials."""

    @aa.decorators.to_grid
    def transformed_to_reference_frame_grid_from(self, grid, xp=np, **kwargs):
        if xp is np:
            return super().transformed_to_reference_frame_grid_from(grid=grid, xp=xp, **kwargs)
        return _sersic_transform(jnp.asarray(grid.array), jnp.asarray(self.centre), self.angle(xp))

    @aa.decorators.to_array
    def eccentric_radii_grid_from(self, grid, xp=np, **kwargs):
        if xp is np:
            return super().eccentric_radii_grid_from(grid=grid, xp=xp, **kwargs)
        return _sersic_radius(jnp.asarray(self.ell_comps), jnp.asarray(grid.array), self.sersic_index)

    def image_2d_via_radii_from(self, grid_radii, xp=np, **kwargs):
        if xp is np:
            return super().image_2d_via_radii_from(grid_radii=grid_radii, xp=xp, **kwargs)
        return xp.multiply(self._intensity, xp.exp(xp.multiply(-self.sersic_constant,
            xp.add(_sersic_power(xp.divide(grid_radii.array, self.effective_radius), self.sersic_index), -1))))
