"""TNFW backend profile shared by truth scenes and fixed/freed fits."""

from __future__ import annotations

import autoarray as aa
import autolens as al
import numpy as np

__all__ = ["TruncatedNFWSph"]


_REGULAR_WIDTH = 0.02
_F_REGULAR = (1.0, -2 / 3, 7 / 15, -12 / 35, 83 / 315, -146 / 693, 523 / 3003,
              -952 / 6435, 14051 / 109395)
_G_REGULAR = (1 / 3, -2 / 5, 13 / 35, -20 / 63, 61 / 231, -94 / 429, 1181 / 6435,
              -1896 / 12155, 6223 / 46189)


def _regular_polynomial(offset, coefficients):
    value = coefficients[-1]
    for coefficient in reversed(coefficients[:-1]):
        value = coefficient + offset * value
    return value


class TruncatedNFWSph(al.mp.NFWTruncatedSph):
    """BMO profile with accurate regular-neighborhood coordinates and traced array math."""

    def _parent_g(self, radius, xp):
        # Call the parent's F directly so its outside-neighborhood scalar precision stays intact.
        f = al.mp.NFWTruncatedSph.coord_func_f(self, grid_radius=radius, xp=xp)
        real_radius = xp.real(radius)
        squared = real_radius**2
        return xp.where(real_radius > 1.0, (1.0 - f) / (squared - 1.0),
                        xp.where(real_radius < 1.0, (f - 1.0) / (1.0 - squared), 1.0 / 3.0))

    def coord_func_f(self, grid_radius, xp=np):
        scalar = isinstance(grid_radius, (float, complex))
        if scalar and abs(grid_radius - 1.0) > _REGULAR_WIDTH:
            return super().coord_func_f(grid_radius=grid_radius, xp=xp)
        radius = xp.array([grid_radius]) if isinstance(grid_radius, (float, complex)) else xp.asarray(grid_radius)
        regular = xp.abs(radius - 1.0) <= _REGULAR_WIDTH
        if xp is np and not np.any(regular):
            return super().coord_func_f(grid_radius=grid_radius, xp=xp)
        safe_radius = xp.where(regular, 2.0, radius)
        value = super().coord_func_f(grid_radius=safe_radius, xp=xp)
        # The regular series avoids cancellation near one and retains its limiting derivatives.
        return xp.where(regular, _regular_polynomial(radius.astype(xp.complex128) - 1.0, _F_REGULAR), value)

    def coord_func_g(self, grid_radius, xp=np):
        scalar = isinstance(grid_radius, (float, complex))
        if scalar and abs(grid_radius - 1.0) > _REGULAR_WIDTH:
            return self._parent_g(xp.array([grid_radius], dtype=xp.complex64), xp)
        accurate_radius = xp.array([grid_radius], dtype=xp.complex128) if scalar else xp.asarray(grid_radius, dtype=xp.complex128)
        radius = xp.array([grid_radius], dtype=xp.complex64) if scalar else xp.asarray(grid_radius)
        regular = xp.abs(accurate_radius - 1.0) <= _REGULAR_WIDTH
        if xp is np and not np.any(regular):
            return self._parent_g(radius, xp)
        safe_radius = xp.where(regular, 2.0, radius)
        value = self._parent_g(safe_radius, xp)
        return xp.where(regular, _regular_polynomial(accurate_radius - 1.0, _G_REGULAR), value)

    @aa.decorators.to_vector_yx
    @aa.decorators.transform
    def deflections_yx_2d_from(self, grid, xp=np, **kwargs):
        eta = xp.multiply(1.0 / self.scale_radius, self.radial_grid_from(grid=grid, xp=xp, **kwargs).array)
        deflection_grid = xp.multiply((4.0 * self.kappa_s * self.scale_radius / eta),
                                     self.deflection_func_sph(grid_radius=eta, xp=xp))
        return self._cartesian_grid_via_radial_from(grid=grid, radius=deflection_grid, xp=xp)

    def convergence_func(self, grid_radius, xp=np):
        radius = grid_radius.array if hasattr(grid_radius, "array") else grid_radius
        radius = ((1.0 / self.scale_radius) * radius) + 0j
        return xp.real(2.0 * self.kappa_s * self.coord_func_l(grid_radius=radius, xp=xp))
