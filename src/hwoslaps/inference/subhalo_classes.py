"""Freed AutoLens halo profiles with a log10_m200 prior, in the paper's traced operation order.

This backend module is imported only when a freed fit is constructed. Profile classes live at
module scope so spawned sampler workers can reconstruct them by name.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Literal

import autolens as al
import autoarray as aa
import numpy as np

from ..identity import mapping_digest
from ..scene.cosmology import Cosmology, LensingGeometry
from ..scene.halos import (Halo, HaloModel, ConcentrationSpec, TruncationSpec,
                           MOLINE2017_MASS_RANGE_MSUN, Moline2017, halo_lensing_traced)
from .settings import MassSupport

__all__ = ["NFWM200SubhaloSph", "PointMassM200Subhalo", "SISM200Subhalo", "SubhaloMassMapping",
           "TNFWM200SubhaloSph", "TruncatedNFWSph",
           "freed_profile_class", "mass_mapping"]


def _xp_for(*values: Any) -> Any:
    import jax
    import jax.numpy as jnp

    pending = list(values)
    while pending:
        value = pending.pop()
        if isinstance(value, (jax.Array, jax.core.Tracer)):
            return jnp
        if isinstance(value, dict):
            pending.extend(value.keys())
            pending.extend(value.values())
        elif isinstance(value, (tuple, list)):
            pending.extend(value)
    return np


def _scalar(value: Any) -> Any:
    return float(value) if isinstance(value, (int, float, np.generic)) else value


@dataclass(frozen=True)
class SubhaloMassMapping:
    model: Literal["PointMass", "SIS", "NFW", "TNFW"]
    concentration: ConcentrationSpec | None
    h: float
    geometry: LensingGeometry
    support: MassSupport
    truncation: TruncationSpec | None = None

    def digest(self) -> str:
        record = {"model": self.model, "concentration": None if self.concentration is None else asdict(self.concentration), "h": self.h,
                  "geometry": asdict(self.geometry), "support": self.support.to_mapping()}
        if self.truncation is not None:
            record["truncation"] = asdict(self.truncation)
        return mapping_digest(record)

    def profile_scales(self, log10_mass: float) -> dict[str, float]:
        if not self.support.contains(log10_mass):
            raise ValueError(f"log10_m200 {log10_mass} lies outside support {self.support.to_mapping()}")
        return {key: float(value) for key, value in self.traced_scales(log10_mass, np).items()}

    def traced_scales(self, log10_mass: Any, xp: Any) -> dict[str, Any]:
        return dict(halo_lensing_traced(HaloModel(self.model, self.concentration, self.truncation), 10.0 ** log10_mass, self.geometry,
                                       reduced_h=self.h, xp=xp))


def mass_mapping(hypothesis: Halo, cosmology: Cosmology, support: MassSupport) -> SubhaloMassMapping:
    if hypothesis.cosmology != cosmology:
        raise ValueError("the mass mapping cosmology differs from the hypothesis")
    if isinstance(hypothesis.model.concentration, Moline2017):
        low, high = MOLINE2017_MASS_RANGE_MSUN
        if 10.0 ** support.log10_mass_min < low or 10.0 ** support.log10_mass_max > high:
            raise ValueError(f"moline2017_eq7 support must lie in [{low:g}, {high:g}] Msun")
    return SubhaloMassMapping(model=hypothesis.model.type, concentration=hypothesis.model.concentration,
                              h=(hypothesis.model.concentration.h if isinstance(hypothesis.model.concentration, Moline2017)
                                 and hypothesis.model.concentration.h is not None else cosmology.reduced_h),
                              geometry=cosmology.geometry(hypothesis.redshift, hypothesis.source_redshift),
                              support=support, truncation=hypothesis.model.truncation)


def _scales(mapping: SubhaloMassMapping | None, kind: str, centre: Any, log10_m200: Any) -> dict[str, Any]:
    if mapping is None or mapping.model != kind:
        raise ValueError(f"mass_mapping for {kind} is required")
    return mapping.traced_scales(log10_m200, _xp_for(centre, log10_m200))


class TruncatedNFWSph(al.mp.NFWTruncatedSph):
    """Pinned BMO profile with array-namespace propagation through its radial functions."""

    def coord_func_f(self, grid_radius, xp=np):
        if xp is np:
            return super().coord_func_f(grid_radius=grid_radius, xp=xp)
        radius = xp.array([grid_radius]) if isinstance(grid_radius, (float, complex)) else xp.asarray(grid_radius)
        regular = radius == 1.0
        safe_radius = xp.where(regular, 2.0, radius)
        value = super().coord_func_f(grid_radius=safe_radius, xp=xp)
        # F is smooth at one; the parent's equality branch loses F'(1)=-2/3.
        return xp.where(regular, 1.0 - (2.0 / 3.0) * (radius - 1.0), value)

    def coord_func_g(self, grid_radius, xp=np):
        if xp is np:
            return super().coord_func_g(grid_radius=grid_radius, xp=xp)
        radius = (xp.array([grid_radius], dtype=xp.complex64) if isinstance(grid_radius, (float, complex))
                  else xp.asarray(grid_radius))
        regular = radius == 1.0
        safe_radius = xp.where(regular, 2.0, radius)
        value = super().coord_func_g(grid_radius=safe_radius, xp=xp)
        # G=(1-F)/(r**2-1) has the regular limits G(1)=1/3 and G'(1)=-2/5.
        return xp.where(regular, 1.0 / 3.0 - (2.0 / 5.0) * (radius - 1.0), value)

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


class TNFWM200SubhaloSph(TruncatedNFWSph):
    def __init__(self, centre=(0.0, 0.0), log10_m200=7.0, mass_mapping=None):
        scales = _scales(mass_mapping, "TNFW", centre, log10_m200)
        super().__init__(centre=centre, **scales)
        self.log10_m200, self.mass_mapping = _scalar(log10_m200), mass_mapping


class NFWM200SubhaloSph(al.mp.NFWSph):
    def __init__(self, centre=(0.0, 0.0), log10_m200=7.0, mass_mapping=None):
        scales = _scales(mass_mapping, "NFW", centre, log10_m200)
        super().__init__(centre=centre, **scales)
        self.log10_m200, self.mass_mapping = _scalar(log10_m200), mass_mapping


class SISM200Subhalo(al.mp.IsothermalSph):
    def __init__(self, centre=(0.0, 0.0), log10_m200=7.0, mass_mapping=None):
        scales = _scales(mass_mapping, "SIS", centre, log10_m200)
        super().__init__(centre=centre, **scales)
        self.log10_m200, self.mass_mapping = _scalar(log10_m200), mass_mapping


class PointMassM200Subhalo(al.mp.PointMass):
    def __init__(self, centre=(0.0, 0.0), log10_m200=7.0, mass_mapping=None):
        scales = _scales(mass_mapping, "PointMass", centre, log10_m200)
        super().__init__(centre=centre, **scales)
        self.log10_m200, self.mass_mapping = _scalar(log10_m200), mass_mapping

    def _cartesian_grid_via_radial_from(self, grid, radius, xp=np, **kwargs):
        """Unwrap AutoArray radius values only on the JAX path, as the pinned paper adapter does."""
        if xp is np:
            return super()._cartesian_grid_via_radial_from(grid=grid, radius=radius, xp=xp, **kwargs)
        grid_values = grid.array if hasattr(grid, "array") else grid
        radius_values = radius.array if hasattr(radius, "array") else radius
        angles = xp.arctan2(grid_values[:, 0], grid_values[:, 1])
        directions = xp.stack((xp.sin(angles), xp.cos(angles)), axis=-1)
        return xp.multiply(radius_values[:, None], directions)


def freed_profile_class(model: str) -> type:
    classes = {"NFW": NFWM200SubhaloSph, "SIS": SISM200Subhalo, "PointMass": PointMassM200Subhalo,
               "TNFW": TNFWM200SubhaloSph}
    if model not in classes:
        raise ValueError(f"unknown freed halo model {model!r}")
    return classes[model]
