"""Array results and explicit spatial reductions for subhalo forecasts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Optional

import numpy as np


@dataclass(frozen=True)
class ForecastResult:
    """Forecasts indexed by mass and position, preserving undefined diagnostics.

    Masses are in solar masses and sky positions are ``(y, x)`` arcseconds.
    Every statistic has shape ``(number of masses, number of positions)``.
    Optional mismatch fields are absent for a matched observation/model PSF.
    An undefined mismatch statistic remains NaN rather than becoming a detection.
    """

    masses_msun: np.ndarray
    positions_yx: np.ndarray
    q_asimov: np.ndarray
    fisher_raw: np.ndarray
    fisher_profiled: np.ndarray
    sigma_amplitude: np.ndarray
    degradation: np.ndarray
    amplitude_hat: Optional[np.ndarray] = None
    q_mismatch: Optional[np.ndarray] = None
    z_mismatch: Optional[np.ndarray] = None
    amplitude_spurious: Optional[np.ndarray] = None
    q_spurious: Optional[np.ndarray] = None
    z_spurious: Optional[np.ndarray] = None
    runtime_provenance: Optional[Mapping[str, Any]] = None

    def __post_init__(self):
        masses = np.asarray(self.masses_msun, dtype=float)
        positions = np.asarray(self.positions_yx, dtype=float)
        if masses.ndim != 1 or not masses.size or not np.all(np.isfinite(masses)) or np.any(masses <= 0):
            raise ValueError("masses_msun must be a non-empty vector of positive finite masses")
        if (
            positions.ndim != 2
            or positions.shape[1] != 2
            or not positions.shape[0]
            or not np.all(np.isfinite(positions))
        ):
            raise ValueError("positions_yx must be a non-empty finite (N, 2) array")
        object.__setattr__(self, "masses_msun", masses.copy())
        object.__setattr__(self, "positions_yx", positions.copy())
        shape = (masses.size, positions.shape[0])
        for name in (
            "q_asimov",
            "fisher_raw",
            "fisher_profiled",
            "sigma_amplitude",
            "degradation",
            "amplitude_hat",
            "q_mismatch",
            "z_mismatch",
            "amplitude_spurious",
            "q_spurious",
            "z_spurious",
        ):
            value = getattr(self, name)
            if value is None:
                if name in {"q_asimov", "fisher_raw", "fisher_profiled", "sigma_amplitude", "degradation"}:
                    raise ValueError(f"{name} is a required forecast statistic")
                continue
            values = np.asarray(value, dtype=float)
            if values.shape != shape:
                raise ValueError(f"{name} must have shape {shape}, got {values.shape}")
            object.__setattr__(self, name, values.copy())

    @property
    def z_asimov(self) -> np.ndarray:
        """Local Fisher-equivalent significance of the matched template."""
        return np.sqrt(self.q_asimov)

    @property
    def detection_metric(self) -> str:
        """Statistic for the actual data/model pair, including knowledge error."""
        return "q_mismatch" if self.q_mismatch is not None else "q_asimov"

    def detections(self, q_threshold, *, metric=None) -> np.ndarray:
        """Apply the declared threshold and positive-amplitude detection rule.

        Matched-template q remains available as an explicit diagnostic override.
        Undefined statistics or amplitudes do not count as detections.
        """
        threshold = _detection_threshold(q_threshold)
        selected_metric = self.detection_metric if metric is None else metric
        if selected_metric not in {"q_asimov", "q_mismatch", "q_spurious"}:
            raise ValueError("metric must be q_asimov, q_mismatch, or q_spurious")
        values = getattr(self, selected_metric)
        if values is None:
            raise ValueError(f"{selected_metric} was not evaluated")
        detected = np.isfinite(values) & (values >= threshold)
        if selected_metric != "q_asimov":
            amplitude = self.amplitude_hat if selected_metric == "q_mismatch" else self.amplitude_spurious
            if amplitude is None:
                raise ValueError(f"{selected_metric} requires its signed amplitude diagnostics")
            detected &= np.isfinite(amplitude) & (amplitude > 0)
        return detected

    def save_npz(self, path):
        """Save through the package's artifact owner."""
        from ..forecast_artifacts import save_forecast_result

        return save_forecast_result(self, path)

    @classmethod
    def load_npz(cls, path):
        """Load a forecast artifact without importing a renderer."""
        from ..forecast_artifacts import load_forecast_result

        return load_forecast_result(path)


@dataclass(frozen=True)
class ForecastSummary:
    """Per-mass diagnostics for an explicitly selected spatial sample.

    Fractions count selected positions. Areas exist only when callers supply
    quadrature cell areas. Boundary detection is absent unless a boundary is
    explicitly supplied, so arbitrary sparse samples do not imply containment.
    """

    masses_msun: np.ndarray
    q_max: np.ndarray
    detectable_fraction: np.ndarray
    detectable_area_arcsec2: Optional[np.ndarray]
    boundary_detectable: Optional[np.ndarray]
    evaluated_position_count: int
    q_threshold: float
    metric: str


def _detection_threshold(value) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError("q_threshold must be positive and finite, not boolean")
    threshold = float(value)
    if not np.isfinite(threshold) or threshold <= 0:
        raise ValueError("q_threshold must be positive and finite")
    return threshold


def summarize_forecast(
    result: ForecastResult,
    q_threshold: float,
    *,
    selection=None,
    cell_areas_arcsec2=None,
    boundary=None,
    metric: Optional[str] = None,
) -> ForecastSummary:
    """Reduce an arbitrary candidate set without imposing a study aperture.

    ``selection`` and ``boundary`` are boolean vectors over result positions.
    Mismatch and spurious detections require both positive fitted amplitude and
    a statistic at or above the supplied threshold. By default, a result with
    a model mismatch uses its actual data statistic, rather than the matched
    template forecast; explicit ``metric='q_asimov'`` selects that diagnostic.
    """
    threshold = _detection_threshold(q_threshold)
    metric = result.detection_metric if metric is None else metric
    detected = result.detections(threshold, metric=metric)
    values = getattr(result, metric)
    size = result.positions_yx.shape[0]

    def mask(value, name, *, default):
        if value is None:
            return np.full(size, default, dtype=bool)
        array = np.asarray(value)
        if array.shape != (size,) or array.dtype != bool:
            raise ValueError(f"{name} must be a boolean vector of length {size}")
        return array

    selected = mask(selection, "selection", default=True)
    if not np.any(selected):
        raise ValueError("selection must contain at least one evaluated position")
    edges = None if boundary is None else mask(boundary, "boundary", default=False)
    consumed = selected if edges is None else selected | edges
    # Undefined zero-information mismatch diagnostics are permitted, but their
    # presence must not silently lower the inferred fraction of usable area.
    if not np.all(np.isfinite(values[:, consumed])):
        raise ValueError("cannot summarize non-finite statistics at consumed positions")
    effective_values = values
    if metric != "q_asimov":
        amplitude = result.amplitude_hat if metric == "q_mismatch" else result.amplitude_spurious
        if not np.all(np.isfinite(amplitude[:, consumed])):
            raise ValueError("cannot summarize non-finite amplitudes at consumed positions")
        effective_values = np.where(amplitude > 0, values, 0.0)
    area = None
    if cell_areas_arcsec2 is not None:
        weights = np.asarray(cell_areas_arcsec2, dtype=float)
        if weights.ndim == 0:
            weights = np.full(size, float(weights))
        if weights.shape != (size,) or not np.all(np.isfinite(weights)) or np.any(weights <= 0):
            raise ValueError("cell_areas_arcsec2 must be positive finite areas per position")
        area = np.sum(detected[:, selected] * weights[selected], axis=1)
    return ForecastSummary(
        masses_msun=result.masses_msun.copy(),
        q_max=np.max(effective_values[:, selected], axis=1),
        detectable_fraction=np.mean(detected[:, selected], axis=1),
        detectable_area_arcsec2=area,
        boundary_detectable=None if edges is None else np.any(detected[:, edges], axis=1),
        evaluated_position_count=int(np.count_nonzero(selected)),
        q_threshold=threshold,
        metric=metric,
    )
