"""Forecast maps and mass curves on caller-owned Axes, with no file output."""

from __future__ import annotations

from numbers import Integral
from typing import TYPE_CHECKING

import numpy as np

from .axes import axes_or_new

if TYPE_CHECKING:
    from ..analysis.knowledge_error import KnowledgeErrorAreas
    from ..analysis.reach import MassReach
    from ..analysis.reductions import ForecastSummary
    from ..fisher.result import ForecastResult, Metric

__all__ = ["plot_statistic_map", "plot_detection_map", "plot_mass_curve", "plot_knowledge_error"]

_STATISTICS = ("fisher_raw", "fisher_profiled", "q_asimov", "z_asimov", "sigma_amplitude", "degradation",
               "q_mismatch", "z_mismatch", "q_spurious", "z_spurious")
_QUANTITIES = {"q_max": "Maximum q", "detectable_fraction": "Detected fraction",
               "detectable_area_arcsec2": "Detected area (arcsec²)"}


def _mass_index(result: ForecastResult, mass_index: int) -> int:
    if isinstance(mass_index, (bool, np.bool_)) or not isinstance(mass_index, Integral) \
            or not 0 <= mass_index < len(result.masses_msun):
        raise ValueError(f"mass_index must be an integer in [0, {len(result.masses_msun)}), got {mass_index!r}")
    return int(mass_index)


def _map(result: ForecastResult, row: np.ndarray, ax):
    grid = result.positions.grid
    if grid is None:
        raise ValueError("a map plot needs a result with a grid lattice")
    image = grid.to_image(row)
    half = grid.spacing_arcsec / 2
    extent = (grid.x_coords[0] - half, grid.x_coords[-1] + half,
              grid.y_coords[0] - half, grid.y_coords[-1] + half)
    ax = axes_or_new(ax)
    ax.imshow(image, origin="lower", extent=extent, interpolation="nearest")
    ax.set_xlabel("x (arcsec)")
    ax.set_ylabel("y (arcsec)")
    return ax


def plot_statistic_map(result: ForecastResult, statistic: str, *, mass_index: int, ax=None) -> "Axes":
    """Show one statistic row; absent statistics or a non-grid result refuse."""
    index = _mass_index(result, mass_index)
    if statistic not in _STATISTICS:
        raise ValueError(f"statistic must be one of {_STATISTICS}, got {statistic!r}")
    values = getattr(result, statistic)
    if values is None:
        raise ValueError(f"the result has no {statistic}")
    ax = _map(result, values[index], ax)
    ax.set_title(f"{statistic}; M = {result.masses_msun[index]:g} M☉")
    return ax


def plot_detection_map(result: ForecastResult, *, q_threshold: float, mass_index: int,
                       metric: Metric | None = None, ax=None) -> "Axes":
    """Show signed-amplitude detections at the required caller threshold."""
    index = _mass_index(result, mass_index)
    detections = result.detections(q_threshold=q_threshold, metric=metric)
    ax = _map(result, detections[index], ax)
    chosen = result.detection_metric if metric is None else metric
    ax.set_title(f"{chosen} detections; q ≥ {q_threshold:g}; M = {result.masses_msun[index]:g} M☉")
    return ax


def plot_mass_curve(summary: ForecastSummary, quantity: str, *, reach: MassReach | None = None, ax=None) -> "Axes":
    """Show a summary against mass, with an optional measured reach and target.

    A bound or nonmonotonic reach has no measured crossing mass and adds no
    vertical crossing marker. Its target remains visible as a horizontal line.
    """
    if quantity not in _QUANTITIES:
        raise ValueError(f"quantity must be one of {tuple(_QUANTITIES)}, got {quantity!r}")
    values = getattr(summary, quantity)
    if values is None:
        raise ValueError(f"the summary has no {quantity}")
    if reach is not None and reach.quantity is not None and reach.quantity != quantity:
        raise ValueError(f"reach quantity {reach.quantity!r} differs from plotted quantity {quantity!r}")
    ax = axes_or_new(ax)
    ax.plot(summary.masses_msun, values, label=quantity)
    if reach is not None:
        ax.axhline(reach.target, linestyle=":", label="Reach target")
        if reach.mass_msun is not None:
            ax.axvline(reach.mass_msun, linestyle="--", label="Mass reach")
    ax.set_xscale("log")
    ax.set_xlabel("Subhalo mass (M☉)")
    ax.set_ylabel(_QUANTITIES[quantity])
    ax.legend()
    return ax


def plot_knowledge_error(areas: KnowledgeErrorAreas, *, ax=None) -> "Axes":
    """Show retained/reference, detected/reference R and in-selection spurious/reference F.

    Floor-excluded NaNs remain gaps; full-domain spurious area has different
    units and is not shown as the dimensionless in-selection F.
    """
    ax = axes_or_new(ax)
    for values, label in ((areas.retention, "Retention"), (areas.detected_area_ratio, "Detected area ratio R"),
                          (areas.spurious_ratio, "Spurious area ratio F (selection)")):
        ax.plot(areas.masses_msun, values, label=label)
    ax.set_xscale("log")
    ax.set_xlabel("Subhalo mass (M☉)")
    ax.set_ylabel("Area / reference detected area")
    ax.set_title(f"q ≥ {areas.q_threshold:g}; reference count ≥ {areas.min_reference_count}")
    ax.legend()
    return ax
