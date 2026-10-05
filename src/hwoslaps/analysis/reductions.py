"""Spatial reductions of a forecast at the caller's detection threshold.

:func:`summarize` reduces a result over a selection of its positions, per mass:

- ``q_max``: the largest statistic in the selection; for ``q_mismatch`` and
  ``q_spurious`` a value whose fitted amplitude is not positive counts as 0, so
  ``q_max`` is below the raw maximum wherever the largest value comes from a
  negative fit.
- ``detectable_count`` and ``detectable_fraction``: the detections in the
  selection and their share of the selected positions.
- ``detectable_area_arcsec2``: the number of those detections times the cell
  area ``spacing**2`` (grid layouts only, whose cells are uniform).
- ``boundary_detectable``: a detection on any boundary node of the layout,
  selected or not. The detectable region then reaches the lattice edge, so the
  area of a larger domain could be larger. It exists only when the result
  holds every boundary node of its layout; otherwise clipping is unknown and
  it is None.

Statistics, and the fitted amplitudes of mismatch metrics, must be finite at
every selected and boundary position. An aperture estimand on a grid is
``summarize(result, q_threshold=T, selection=aperture_selection(result,
centre_yx=c, radius_arcsec=r))`` on the full layout or on its
``aperture(..., include_boundary=True)`` subset; the boundary flag then
reports whether the lattice edge clipped the detections.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike

from ..fisher.result import ForecastResult, Metric

__all__ = ["ForecastSummary", "Metric", "aperture_selection", "summarize"]


@dataclass(frozen=True, eq=False)
class ForecastSummary:
    """Per-mass reductions of a forecast over a selection of its positions."""

    masses_msun: np.ndarray
    q_threshold: float
    metric: str
    selected_count: int
    q_max: np.ndarray
    detectable_count: np.ndarray
    detectable_fraction: np.ndarray
    detectable_area_arcsec2: np.ndarray | None
    boundary_detectable: np.ndarray | None


def summarize(result: ForecastResult, *, q_threshold: float, metric: Metric | None = None,
              selection: ArrayLike | None = None) -> ForecastSummary:
    """Reduce ``result`` over ``selection`` (a boolean vector over its positions; all by default)."""
    chosen = result.detection_metric if metric is None else metric
    detected = result.detections(q_threshold=q_threshold, metric=chosen)
    size = len(result.positions)
    if selection is None:
        selected = np.ones(size, dtype=bool)
    else:
        selected = np.asarray(selection)
        if selected.dtype != bool or selected.shape != (size,):
            raise ValueError(f"selection must be a boolean vector of length {size}")
        if not np.any(selected):
            raise ValueError("selection must contain at least one position")
    boundary = result.boundary
    consumed = selected if boundary is None else selected | boundary
    values = getattr(result, chosen)
    if not np.all(np.isfinite(values[:, consumed])):
        raise ValueError(f"{chosen} is not finite at a selected or boundary position")
    effective = values
    if chosen != "q_asimov":
        amplitude = result.amplitude_hat if chosen == "q_mismatch" else result.amplitude_spurious
        if not np.all(np.isfinite(amplitude[:, consumed])):
            raise ValueError(f"the fitted amplitude of {chosen} is not finite at a selected or boundary position")
        effective = np.where(amplitude > 0, values, 0.0)
    count = np.count_nonzero(detected[:, selected], axis=1)
    grid = result.positions.grid
    return ForecastSummary(
        masses_msun=result.masses_msun,
        q_threshold=float(q_threshold),
        metric=chosen,
        selected_count=int(np.count_nonzero(selected)),
        q_max=np.max(effective[:, selected], axis=1),
        detectable_count=count,
        detectable_fraction=np.mean(detected[:, selected], axis=1),
        detectable_area_arcsec2=None if grid is None else count * grid.spacing_arcsec ** 2,
        boundary_detectable=None if boundary is None else np.any(detected[:, boundary], axis=1),
    )


def aperture_selection(result: ForecastResult, *, centre_yx: tuple[float, float],
                       radius_arcsec: float) -> np.ndarray:
    """Positions in the closed disc ``dy**2 + dx**2 <= r**2``; an empty aperture raises."""
    inside = result.positions.within(centre_yx, radius_arcsec)
    if not np.any(inside):
        raise ValueError(f"no position lies inside the aperture of radius {radius_arcsec} arcsec "
                         f"about {tuple(centre_yx)}")
    return inside
