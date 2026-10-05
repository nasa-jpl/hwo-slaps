"""Effective Einstein radius: sqrt(A / pi) of the tangential critical curve of the lens mass.

The tangential eigenvalue ``1 - kappa - |gamma|`` of the lens mass components (no light,
perturbers or subhalo), scaled to the scene's source plane, is evaluated by AutoGalaxy's
``LensCalc`` on a uniform grid about the origin with 0.01" pixels and a half width of four
times the largest Einstein-radius parameter plus the offset of the lens centre. Zero
contours come from marching squares. Among the closed contours that enclose the lens centre
(even-odd rule) the one of largest area is the critical curve. For an SIE this radius is the
intermediate-axis radius ``2 sqrt(q) theta_E / (1 + q)``.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np

from .cosmology import Cosmology
from .profiles import PROFILE_TYPES, instantiate

if TYPE_CHECKING:
    from .spec import SceneSpec

__all__ = ["effective_einstein_radius"]

_PIXEL_SCALE_ARCSEC = 0.01
_HALF_WIDTH_FACTOR = 4.0
_CLOSURE_TOLERANCE_PIXELS = 0.5
_BORDER_MARGIN_PIXELS = 2.0
_MIN_CONTOUR_VERTICES = 32


def _polygon_area(polygon: np.ndarray) -> float:
    y, x = polygon[:-1, 0], polygon[:-1, 1]
    return float(abs(0.5 * np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y)))


def _encloses(polygon: np.ndarray, point: tuple[float, float]) -> bool:
    vertices = polygon[:-1]
    y_start, x_start = vertices[:, 0], vertices[:, 1]
    y_end, x_end = np.roll(y_start, -1), np.roll(x_start, -1)
    straddles = (y_start > point[0]) != (y_end > point[0])
    with np.errstate(divide="ignore", invalid="ignore"):
        x_cross = x_start + (point[0] - y_start) * (x_end - x_start) / (y_end - y_start)
    return bool(np.count_nonzero(straddles & (point[1] < x_cross)) % 2 == 1)


def effective_einstein_radius(spec: SceneSpec, cosmology: Cosmology) -> float:
    """sqrt(area / pi) of the outer tangential critical curve of the lens mass around the lens centre."""
    import autoarray as aa
    import autolens as al
    from autogalaxy.operate.lens_calc import LensCalc

    radii = [component.values[definition.key] for component in spec.lens.mass
             for definition in PROFILE_TYPES[component.type].parameters(component.values)
             if definition.kind == "einstein_radius"]
    if not radii:
        raise ValueError("no lens mass component has an Einstein radius to size the critical-curve grid")
    centre = spec.lens_centre
    profiles = {}
    for component in spec.lens.mass:
        profiles.update(instantiate(component))
    tracer = al.Tracer(galaxies=[al.Galaxy(redshift=spec.lens.redshift, **profiles),
                                 al.Galaxy(redshift=spec.source.redshift)], cosmology=cosmology.autogalaxy())

    scale = _PIXEL_SCALE_ARCSEC
    pixels = 2 * int(math.ceil((_HALF_WIDTH_FACTOR * max(radii) + max(abs(centre[0]), abs(centre[1]))) / scale))
    half_width = 0.5 * pixels * scale
    grid = aa.Grid2D.uniform(shape_native=(pixels, pixels), pixel_scales=(scale, scale))
    grid.is_evaluation_grid = True
    curves = [np.asarray(curve, dtype=float) for curve in
              LensCalc.from_mass_obj(tracer).tangential_critical_curve_list_from(grid=grid, pixel_scale=scale)]
    closed = [curve for curve in curves
              if curve.shape[0] >= 4 and math.hypot(*(curve[0] - curve[-1])) <= _CLOSURE_TOLERANCE_PIXELS * scale]
    enclosing = [curve for curve in closed if _encloses(curve, centre)]
    if not enclosing:
        raise ValueError(f"no closed tangential critical curve encloses the lens centre {centre} ({len(curves)} "
                         f"contours, {len(closed)} closed) on the {pixels} x {pixels} grid of {scale} arcsec pixels")
    chosen = max(enclosing, key=_polygon_area)
    if float(np.max(np.abs(chosen))) > half_width - _BORDER_MARGIN_PIXELS * scale:
        raise ValueError(f"the tangential critical curve reaches the edge of the {2 * half_width} arcsec grid")
    if chosen.shape[0] < _MIN_CONTOUR_VERTICES:
        raise ValueError(f"the tangential critical curve has {chosen.shape[0]} vertices on {scale} arcsec pixels, "
                         f"fewer than {_MIN_CONTOUR_VERTICES}: the lens is too small to measure")
    return math.sqrt(_polygon_area(chosen) / math.pi)
