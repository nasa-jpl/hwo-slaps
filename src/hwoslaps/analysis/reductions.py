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

Usable templates include identifiable fits and zero-information q=0 trials.
Numerical failures at selected or boundary positions raise by default. The
explicit report policy retains status counts and marks incomplete aggregates
unavailable; it does not turn failures into nondetections. An aperture estimand on a grid is
``summarize(result, q_threshold=T, selection=aperture_selection(result,
centre_yx=c, radius_arcsec=r))`` on the full layout or on its
``aperture(..., include_boundary=True)`` subset; the boundary flag then
reports whether the lattice edge clipped the detections.
"""

from __future__ import annotations

from dataclasses import dataclass, fields, is_dataclass
from typing import Any, Literal

import numpy as np
from numpy.typing import ArrayLike

from ..fisher.result import ForecastResult, Metric

__all__ = ["ForecastSummary", "InvalidPolicy", "Metric", "TemplateValidity", "aperture_selection", "summarize"]


InvalidPolicy = Literal["raise", "report"]
"""Reject consumed numerical failures, or retain explicit incomplete diagnostics."""


def _read_only(values: np.ndarray) -> np.ndarray:
    copied = np.array(values)
    copied.setflags(write=False)
    return copied


def _report_mapping(value: Any) -> Any:
    """Encode aggregate diagnostics with explicit nulls for unavailable numbers.

    This is a report format, not the finite-only configuration/digest policy.
    Status counts and fixed cohorts accompany missing scientific aggregates.
    """
    if is_dataclass(value):
        return {field.name: _report_mapping(getattr(value, field.name)) for field in fields(value)}
    if isinstance(value, np.ndarray):
        return _report_mapping(value.tolist())
    if isinstance(value, (list, tuple)):
        return [_report_mapping(item) for item in value]
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.bool_):
        return bool(value)
    return value


@dataclass(frozen=True, eq=False)
class TemplateValidity:
    """Per-mass status counts over one fixed spatial selection.

    ``valid_count`` counts identifiable templates. Zero-information templates
    are usable q=0 trials with unidentified amplitudes. All three counts sum
    to ``evaluated_count``; no failed position silently leaves the cohort.
    """

    evaluated_count: int
    valid_count: np.ndarray
    zero_information_count: np.ndarray
    failed_numeric_count: np.ndarray

    @classmethod
    def from_result(cls, result: ForecastResult, selection: np.ndarray) -> TemplateValidity:
        status = result.template_status[:, selection]
        return cls(int(np.count_nonzero(selection)), *(_read_only(np.count_nonzero(status == value, axis=1))
                   for value in ("valid", "zero_information", "failed_numeric")))

    def to_mapping(self) -> dict[str, Any]:
        return _report_mapping(self) | {"complete": self.complete.tolist()}

    @property
    def complete(self) -> np.ndarray:
        """Whether every selected template is usable at each mass."""
        return _read_only(self.failed_numeric_count == 0)

    def require_complete(self, name: str) -> None:
        if not np.all(self.complete):
            raise ValueError(f"{name} contains failed_numeric templates by mass "
                             f"{self.failed_numeric_count.tolist()}; choose invalid_policy='report' "
                             "to retain incomplete diagnostics without a detection estimate")


def _invalid_policy(value: str) -> None:
    if value not in ("raise", "report"):
        raise ValueError("invalid_policy must be 'raise' or 'report'")


def _reported(values: np.ndarray, complete: np.ndarray) -> np.ndarray:
    """A full-cohort aggregate is unavailable when any consumed row failed."""
    if np.all(complete):
        return values
    return np.where(complete, values, np.nan)


@dataclass(frozen=True, eq=False)
class ForecastSummary:
    """Per-mass spatial reductions and explicit validity/cohort diagnostics.

    With ``invalid_policy='report'``, full-selection quantities are NaN at
    incomplete masses. ``known_detectable_count`` remains an integer diagnostic
    of healthy detections, not an estimate for the unevaluated positions.
    """

    masses_msun: np.ndarray
    q_threshold: float
    metric: str
    selected_count: int
    q_max: np.ndarray
    detectable_count: np.ndarray
    detectable_fraction: np.ndarray
    detectable_area_arcsec2: np.ndarray | None
    boundary_detectable: np.ndarray | None
    validity: TemplateValidity
    boundary_validity: TemplateValidity | None
    known_detectable_count: np.ndarray
    invalid_policy: InvalidPolicy

    def to_mapping(self) -> dict[str, Any]:
        """JSON-ready report, preserving unavailable aggregates as nulls."""
        return _report_mapping(self) | {
            "validity": self.validity.to_mapping(),
            "boundary_validity": None if self.boundary_validity is None else self.boundary_validity.to_mapping(),
        }


def summarize(result: ForecastResult, *, q_threshold: float, metric: Metric | None = None,
              selection: ArrayLike | None = None, invalid_policy: InvalidPolicy = "raise") -> ForecastSummary:
    """Reduce over a fixed selection; reject failures or explicitly report incompleteness.

    Zero-information templates count as usable q=0 nondetections. Under
    ``report``, failed rows remain in the cohort: affected full-cohort counts,
    fractions, areas and q maxima are NaN, with known detections and status
    counts retained separately. Boundary completeness is checked independently.
    """
    _invalid_policy(invalid_policy)
    chosen = result.detection_metric if metric is None else metric
    detected = result.detections(q_threshold=q_threshold, metric=chosen, invalid_policy="exclude")
    size = len(result.positions)
    if selection is None:
        selected = np.ones(size, dtype=bool)
    else:
        selected = np.asarray(selection)
        if selected.dtype != bool or selected.shape != (size,):
            raise ValueError(f"selection must be a boolean vector of length {size}")
        if not np.any(selected):
            raise ValueError("selection must contain at least one position")
    validity = TemplateValidity.from_result(result, selected)
    boundary = result.boundary
    boundary_validity = None if boundary is None else TemplateValidity.from_result(result, boundary)
    if invalid_policy == "raise":
        validity.require_complete("selected positions")
        if boundary_validity is not None:
            boundary_validity.require_complete("boundary positions")
    values = getattr(result, chosen)
    effective = values
    if chosen != "q_asimov":
        amplitude = result.amplitude_hat if chosen == "q_mismatch" else result.amplitude_spurious
        effective = np.where(amplitude > 0.0, values, 0.0)
    # A failed value may be finite/provisional or nonfinite; status controls it.
    effective = np.where(result.template_status == "failed_numeric", -np.inf, effective)
    known_count = np.count_nonzero(detected[:, selected], axis=1)
    count = _reported(known_count, validity.complete)
    selected_count = int(np.count_nonzero(selected))
    grid = result.positions.grid
    boundary_detectable = None
    if boundary_validity is not None and np.all(boundary_validity.complete):
        boundary_detectable = np.any(detected[:, boundary], axis=1)
    return ForecastSummary(
        masses_msun=result.masses_msun,
        q_threshold=float(q_threshold),
        metric=chosen,
        selected_count=selected_count,
        q_max=_reported(np.max(effective[:, selected], axis=1), validity.complete),
        detectable_count=count,
        detectable_fraction=count / selected_count,
        detectable_area_arcsec2=None if grid is None else count * grid.spacing_arcsec ** 2,
        boundary_detectable=boundary_detectable,
        validity=validity,
        boundary_validity=boundary_validity,
        known_detectable_count=known_count,
        invalid_policy=invalid_policy,
    )


def aperture_selection(result: ForecastResult, *, centre_yx: tuple[float, float],
                       radius_arcsec: float) -> np.ndarray:
    """Positions in the closed disc ``dy**2 + dx**2 <= r**2``; an empty aperture raises."""
    inside = result.positions.within(centre_yx, radius_arcsec)
    if not np.any(inside):
        raise ValueError(f"no position lies inside the aperture of radius {radius_arcsec} arcsec "
                         f"about {tuple(centre_yx)}")
    return inside
