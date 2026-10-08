"""PSF knowledge-error areas, common-cohort tolerances and nominal separation.

Retention measures the intersection with correct-PSF detections. The paper's R
is instead the ratio of all mismatch detections to reference detections, and F
uses spurious detections inside the same selection. Full-domain spurious area
is reported separately. Thresholds, count floors and tolerance limits are the
caller's; eligible member/direction keys come from the reference experiment.
"""

from __future__ import annotations

from collections.abc import Collection, Hashable, Mapping
from dataclasses import dataclass
from numbers import Integral, Real
from typing import Any

import numpy as np

from ..fisher.result import ForecastResult
from .binomial import BinomialCount, clopper_pearson

__all__ = ["KnowledgeErrorAreas", "ToleranceCriterion", "ToleranceResult", "SeparationResult",
           "knowledge_error_areas", "knowledge_error_tolerance", "first_separating_amplitude"]


@dataclass(frozen=True, eq=False)
class KnowledgeErrorAreas:
    """Per-mass counts and areas; ratios are NaN below the reference count floor."""

    masses_msun: np.ndarray
    q_threshold: float
    min_reference_count: int
    reference_count: np.ndarray
    reference_area_arcsec2: np.ndarray
    mismatch_count: np.ndarray
    mismatch_area_arcsec2: np.ndarray
    retained_count: np.ndarray
    retained_area_arcsec2: np.ndarray
    spurious_count: np.ndarray
    spurious_area_arcsec2: np.ndarray
    spurious_in_selection_count: np.ndarray
    spurious_in_selection_area_arcsec2: np.ndarray
    retention: np.ndarray
    detected_area_ratio: np.ndarray
    spurious_ratio: np.ndarray


def _same_array(first: np.ndarray, second: np.ndarray, name: str) -> None:
    if first.shape != second.shape or first.dtype != second.dtype or first.tobytes() != second.tobytes():
        raise ValueError(f"{name} differs between the reference and mismatched forecasts")


def _pair_identity(reference: ForecastResult, mismatched: ForecastResult) -> None:
    _same_array(reference.masses_msun, mismatched.masses_msun, "masses_msun")
    _same_array(reference.positions_yx, mismatched.positions_yx, "positions_yx")
    if reference.cell_areas_arcsec2 is None or mismatched.cell_areas_arcsec2 is None:
        raise ValueError("both forecasts need cell_areas_arcsec2")
    _same_array(reference.cell_areas_arcsec2, mismatched.cell_areas_arcsec2, "cell_areas_arcsec2")
    if reference.psf_relation != "matched" or reference.provenance.get("psf_relation") != "matched":
        raise ValueError("the reference psf_relation must be matched")
    if mismatched.amplitude_hat is None or mismatched.amplitude_spurious is None:
        raise ValueError("the mismatched forecast needs all mismatch statistics")
    for key in ("comparison_digest", "nuisance_names", "truth_kernels"):
        if key not in reference.provenance or key not in mismatched.provenance:
            raise ValueError(f"both forecasts need provenance {key}")
        if reference.provenance[key] != mismatched.provenance[key]:
            raise ValueError(f"provenance {key} differs between the forecasts")
    for result in (reference, mismatched):
        if not isinstance(result.provenance["comparison_digest"], str) or not result.provenance["comparison_digest"]:
            raise ValueError("provenance comparison_digest must be a non-empty string")
        if not isinstance(result.provenance.get("mask"), Mapping) or not result.provenance["mask"].get("digest"):
            raise ValueError("both forecasts need the provenance mask digest")
    if reference.provenance["mask"]["digest"] != mismatched.provenance["mask"]["digest"]:
        raise ValueError("provenance mask digest differs between the forecasts")


def knowledge_error_areas(reference: ForecastResult, mismatched: ForecastResult, *, q_threshold: float,
                          min_reference_count: int, selection: np.ndarray | None = None) -> KnowledgeErrorAreas:
    """Reduce comparable forecasts, whose configurations may differ only in ``psf.model``.

    Both results must have identical mass/position bytes, uniform cell areas,
    comparison digest, mask digest, nuisance order and complete truth bindings.
    An aperture selection gives the paper's R and F; spurious area also covers
    every evaluated position. Mismatch detections require positive amplitudes.
    """
    _pair_identity(reference, mismatched)
    if isinstance(min_reference_count, (bool, np.bool_)) or not isinstance(min_reference_count, Integral) \
            or min_reference_count < 1:
        raise ValueError("min_reference_count must be an integer >= 1, not boolean")
    size = len(reference.positions)
    selected = np.ones(size, dtype=bool) if selection is None else np.asarray(selection)
    if selected.dtype != bool or selected.shape != (size,):
        raise ValueError(f"selection must be a boolean vector of length {size}")
    correct = reference.detections(q_threshold=q_threshold, metric="q_asimov")
    detected = mismatched.detections(q_threshold=q_threshold, metric="q_mismatch")
    spurious = mismatched.detections(q_threshold=q_threshold, metric="q_spurious")
    counts = [np.count_nonzero(values[:, chosen], axis=1) for values, chosen in (
        (correct, selected), (detected, selected), (correct & detected, selected),
        (spurious, np.ones(size, dtype=bool)), (spurious, selected))]
    # PositionSet guarantees uniform spacing**2; retain the paper count-times-area arithmetic.
    areas = [count * reference.cell_areas_arcsec2[0] for count in counts]
    eligible = counts[0] >= min_reference_count
    ratios = []
    for numerator in (areas[2], areas[1], areas[4]):
        ratio = np.full(reference.masses_msun.shape, np.nan)
        np.divide(numerator, areas[0], out=ratio, where=eligible)
        ratios.append(ratio)
    return KnowledgeErrorAreas(reference.masses_msun, float(q_threshold), int(min_reference_count),
                               counts[0], areas[0], counts[1], areas[1], counts[2], areas[2],
                               counts[3], areas[3], counts[4], areas[4], *ratios)


def _finite_number(value: float, name: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real) or not np.isfinite(value):
        raise ValueError(f"{name} must be a finite number, not boolean, got {value!r}")
    return float(value)


def _amplitudes(values: Mapping[float, Any], name: str) -> tuple[float, ...]:
    for amplitude in values:
        if _finite_number(amplitude, f"{name} amplitude") < 0:
            raise ValueError(f"{name} amplitude must be non-negative, got {amplitude!r}")
    return tuple(sorted(values))


@dataclass(frozen=True)
class ToleranceCriterion:
    """Linear quantiles and inclusive limits for the same eligible cohort."""

    retention_quantile: float
    retention_min: float
    spurious_quantile: float
    spurious_max: float

    def __post_init__(self) -> None:
        for name in ("retention_quantile", "retention_min", "spurious_quantile", "spurious_max"):
            number = _finite_number(getattr(self, name), name)
            if name.endswith("quantile") and not 0 <= number <= 1:
                raise ValueError(f"{name} must lie in [0, 1]")


@dataclass(frozen=True)
class ToleranceResult:
    """Largest passing amplitude, all passing amplitudes and first evaluated failure."""

    amplitude: float | None
    passing: tuple[float, ...]
    first_failing: float | None
    criterion: ToleranceCriterion
    eligible: int
    excluded: int


def knowledge_error_tolerance(retention: Mapping[float, Mapping[Hashable, float]],
                              spurious: Mapping[float, Mapping[Hashable, float]], *,
                              eligible: Collection[Hashable], criterion: ToleranceCriterion) -> ToleranceResult:
    """Apply both gates to one fixed cohort of eligible member/direction keys.

    Every amplitude and both maps must contain the same keys, including the
    ineligible ones. An eligible non-finite value raises ValueError instead of changing
    the cohort. The caller omits any endpoint anchor from these maps.
    """
    amplitudes = _amplitudes(retention, "retention")
    _amplitudes(spurious, "spurious")
    if not amplitudes or set(retention) != set(spurious):
        raise ValueError("retention and spurious must contain the same non-empty amplitude set")
    keys = set(retention[amplitudes[0]])
    for name, values in (("retention", retention), ("spurious", spurious)):
        for amplitude in amplitudes:
            actual = set(values[amplitude])
            if actual != keys:
                raise ValueError(f"{name} at amplitude {amplitude} has differing keys: "
                                 f"missing {sorted(keys - actual, key=repr)!r}, extra {sorted(actual - keys, key=repr)!r}")
    cohort = set(eligible)
    if not cohort or not cohort <= keys:
        raise ValueError(f"eligible must be a non-empty subset of the cohort; missing keys "
                         f"{sorted(cohort - keys, key=repr)!r}")
    ordered = sorted(cohort, key=repr)
    passing, failing = [], []
    for amplitude in amplitudes:
        samples = []
        for name, values in (("retention", retention), ("spurious", spurious)):
            sample = [_finite_number(values[amplitude][key], f"{name} at amplitude {amplitude}, key {key!r}")
                      for key in ordered]
            samples.append(sample)
        passes = np.quantile(samples[0], criterion.retention_quantile, method="linear") >= criterion.retention_min \
            and np.quantile(samples[1], criterion.spurious_quantile, method="linear") <= criterion.spurious_max
        (passing if passes else failing).append(float(amplitude))
    return ToleranceResult(max(passing) if passing else None, tuple(passing), min(failing) if failing else None,
                           criterion, len(cohort), len(keys) - len(cohort))


@dataclass(frozen=True)
class SeparationResult:
    """CP separation of controls from a null, nominal for pooled clustered directions."""

    amplitude: float | None
    confidence: float
    null_interval: tuple[float, float]
    control_intervals: Mapping[float, tuple[float, float]]
    separates_all_larger: bool | None

    def to_mapping(self) -> dict[str, Any]:
        return {"amplitude": self.amplitude, "confidence": self.confidence,
                "null_interval": list(self.null_interval),
                "control_intervals": {amplitude: list(interval) for amplitude, interval in self.control_intervals.items()},
                "separates_all_larger": self.separates_all_larger, "interval": "clopper_pearson_nominal"}


def first_separating_amplitude(controls: Mapping[float, BinomialCount], null: BinomialCount, *,
                               confidence: float) -> SeparationResult:
    """Smallest amplitude with CP lower bound strictly above the null upper bound.

    Larger controls are checked separately: separation need not be monotonic.
    CP is exact for independent binomial trials, nominal for clustered directions.
    """
    amplitudes = _amplitudes(controls, "control")
    null_interval = clopper_pearson(null.count, null.trials, confidence=confidence)
    intervals = {float(amplitude): clopper_pearson(controls[amplitude].count, controls[amplitude].trials,
                                                 confidence=confidence) for amplitude in amplitudes}
    separating = [amplitude for amplitude, bounds in intervals.items() if bounds[0] > null_interval[1]]
    first = min(separating) if separating else None
    all_larger = None if first is None else all(bounds[0] > null_interval[1]
                                              for amplitude, bounds in intervals.items() if amplitude > first)
    return SeparationResult(first, float(confidence), null_interval, intervals, all_larger)
