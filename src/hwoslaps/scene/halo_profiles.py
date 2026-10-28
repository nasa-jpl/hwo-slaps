"""Spherical halo adapters shared by truth scenes, templates and fitted models."""
from __future__ import annotations

import autoarray as aa
import autolens as al
import numpy as np

from .halo_numerics import nfw_f, nfw_g, spherical_deflections

__all__ = ["NFWSph", "PointMass", "IsothermalSph", "TruncatedNFWSph"]


class _SphericalValues:
    @aa.decorators.to_vector_yx
    @aa.decorators.transform
    def deflections_yx_2d_from(self, grid, xp=np, **kwargs):
        values = grid.array if hasattr(grid, "array") else grid
        parameters = ({"einstein_radius": self.einstein_radius} if self._halo_kind in ("PointMass", "SIS") else
                      {"kappa_s": self.kappa_s, "scale_radius": self.scale_radius})
        if self._halo_kind == "TNFW":
            parameters["truncation_radius"] = self.truncation_radius
        return spherical_deflections(self._halo_kind, values, parameters, xp)


class PointMass(_SphericalValues, al.mp.PointMass):
    """Unsoftened point lens; exact central samples require image integration."""
    _halo_kind = "PointMass"


class IsothermalSph(_SphericalValues, al.mp.IsothermalSph):
    """Unsoftened SIS; exact central samples require image integration."""
    _halo_kind = "SIS"


class NFWSph(_SphericalValues, al.mp.NFWSph):
    """NFW with stable positive radii and the removable central vector value."""
    _halo_kind = "NFW"


class TruncatedNFWSph(_SphericalValues, al.mp.NFWTruncatedSph):
    """BMO n=1 profile with stable central and scale-radius neighborhoods."""
    _halo_kind = "TNFW"

    def coord_func_f(self, grid_radius, xp=np):
        return nfw_f(grid_radius, xp)

    def coord_func_g(self, grid_radius, xp=np):
        return nfw_g(grid_radius, xp)

    def convergence_func(self, grid_radius, xp=np):
        radius = grid_radius.array if hasattr(grid_radius, "array") else grid_radius
        radius = ((1.0 / self.scale_radius) * radius) + 0j
        return xp.real(2.0 * self.kappa_s * self.coord_func_l(grid_radius=radius, xp=xp))
