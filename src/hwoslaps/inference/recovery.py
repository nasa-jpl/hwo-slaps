"""Freed-subhalo recovery: the sampler and refined estimates and the posterior quantiles.

Raw values only. The margins to the ends of the mass support are reported in dex; a boundary
flag is a threshold the caller applies. Posterior band fractions require the raw sampler
draws and weights, not just these quantiles. The sampler estimate is the
sampler's maximum-likelihood instance (the values the paper reported); the refined estimate
is decoded from the refined best vector, the solution whose likelihood enters ``q_signed``.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from .result import check_record_keys
from .settings import MassSupport

if TYPE_CHECKING:
    from .fit_model import FitModel
    from .result import RefineOutcome, SamplerDiagnostics
    from .subhalo_classes import SubhaloMassMapping

__all__ = ["SubhaloEstimate", "SubhaloRecovery", "extract_recovery", "weighted_quantiles"]


@dataclass(frozen=True)
class SubhaloEstimate:
    """One estimate of the freed subhalo: mass, centre (y, x), derived profile scales, margins."""

    log10_mass: float
    centre_yx: tuple[float, float]
    profile_scales: Mapping[str, float]
    margin_to_lower_dex: float
    margin_to_upper_dex: float

    def to_mapping(self) -> dict[str, Any]:
        return {"log10_mass": self.log10_mass, "centre_yx": list(self.centre_yx),
                "profile_scales": dict(self.profile_scales), "margin_to_lower_dex": self.margin_to_lower_dex,
                "margin_to_upper_dex": self.margin_to_upper_dex}

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, Any]) -> SubhaloEstimate:
        check_record_keys(mapping, cls)
        return cls(log10_mass=float(mapping["log10_mass"]),
                   centre_yx=tuple(float(value) for value in mapping["centre_yx"]),
                   profile_scales={name: float(value) for name, value in mapping["profile_scales"].items()},
                   margin_to_lower_dex=float(mapping["margin_to_lower_dex"]),
                   margin_to_upper_dex=float(mapping["margin_to_upper_dex"]))


def _quantiles(value: Any) -> tuple[float, float, float] | None:
    return None if value is None else tuple(float(item) for item in value)


@dataclass(frozen=True)
class SubhaloRecovery:
    """Freed-H1 values, with provisional quantiles retained and explicitly labelled.

    ``pdf_converged`` is the historical backend weight-usability heuristic only.
    Use ``posterior_quality`` for default scientific aggregation.
    """

    sampler: SubhaloEstimate
    refined: SubhaloEstimate | None
    log10_mass_quantiles: tuple[float, float, float] | None
    centre_y_quantiles: tuple[float, float, float] | None
    centre_x_quantiles: tuple[float, float, float] | None
    support: MassSupport
    pdf_converged: bool
    sample_count: int
    posterior_quality: str = "unknown"

    @property
    def pdf_usable(self) -> bool:
        return self.pdf_converged

    def to_mapping(self) -> dict[str, Any]:
        return {"sampler": self.sampler.to_mapping(),
                "refined": None if self.refined is None else self.refined.to_mapping(),
                "log10_mass_quantiles": _list(self.log10_mass_quantiles),
                "centre_y_quantiles": _list(self.centre_y_quantiles),
                "centre_x_quantiles": _list(self.centre_x_quantiles),
                "support": self.support.to_mapping(), "pdf_converged": self.pdf_converged,
                "sample_count": self.sample_count, "posterior_quality": self.posterior_quality}

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, Any]) -> SubhaloRecovery:
        mapping = {"posterior_quality": "unknown", **mapping}
        check_record_keys(mapping, cls)
        refined = mapping["refined"]
        return cls(sampler=SubhaloEstimate.from_mapping(mapping["sampler"]),
                   refined=None if refined is None else SubhaloEstimate.from_mapping(refined),
                   log10_mass_quantiles=_quantiles(mapping["log10_mass_quantiles"]),
                   centre_y_quantiles=_quantiles(mapping["centre_y_quantiles"]),
                   centre_x_quantiles=_quantiles(mapping["centre_x_quantiles"]),
                   support=MassSupport.from_mapping(mapping["support"], path="support"),
                   pdf_converged=bool(mapping["pdf_converged"]), sample_count=int(mapping["sample_count"]),
                   posterior_quality=mapping["posterior_quality"])


def _list(values: tuple[float, ...] | None) -> list[float] | None:
    return None if values is None else list(values)


def weighted_quantiles(values: np.ndarray, weights: np.ndarray | None,
                       probabilities: Sequence[float] = (0.16, 0.5, 0.84)) -> tuple[float, ...]:
    """Quantiles of weighted samples by the midpoint empirical CDF.

    Sorted samples get cumulative probability ``(cumsum(w) - w / 2) / sum(w)`` and quantiles
    interpolate the sorted values against it, clamped to the smallest and largest sample.
    Zero-weight entries do not contribute knots; equal values share one probability atom.
    All positive weights, including equal
    weights, use this convention. Weights must be finite, nonnegative and have positive
    support; contributing values must be finite. Without weights the separate contract is
    NumPy's default linear quantile estimator on a nonempty finite one-dimensional array.
    """
    values = np.asarray(values, dtype=float)
    probabilities = np.asarray(list(probabilities), dtype=float)
    if probabilities.ndim != 1 or not np.all(np.isfinite(probabilities)) or np.any(
            (probabilities < 0.0) | (probabilities > 1.0)):
        raise ValueError("probabilities must be finite and between zero and one")
    if values.ndim != 1 or values.size == 0:
        raise ValueError("values must be a nonempty one-dimensional array")
    if weights is None:
        if not np.all(np.isfinite(values)):
            raise ValueError("unweighted values must be finite")
        return tuple(float(value) for value in np.quantile(values, probabilities))
    weights = np.asarray(weights, dtype=float)
    if weights.shape != values.shape:
        raise ValueError(f"weights of shape {weights.shape} do not match values of shape {values.shape}")
    if not np.all(np.isfinite(weights)) or np.any(weights < 0.0):
        raise ValueError("weights must be finite and nonnegative")
    positive = weights > 0.0
    if not np.any(positive):
        raise ValueError("weights must have positive support")
    values, weights = values[positive], weights[positive]
    if not np.all(np.isfinite(values)):
        raise ValueError("positive-weight values must be finite")
    order = np.argsort(values, kind="stable")
    sorted_values = values[order]
    # Scaling first prevents overflow of the sum for otherwise valid finite measures.
    sorted_weights = weights[order] / np.max(weights)
    sorted_values, first = np.unique(sorted_values, return_index=True)
    sorted_weights = np.add.reduceat(sorted_weights, first)
    normalization = float(np.sum(sorted_weights))
    cumulative = (np.cumsum(sorted_weights) - 0.5 * sorted_weights) / normalization
    return tuple(float(np.interp(probability, cumulative, sorted_values)) for probability in probabilities)


def _estimate(log10_mass: float, centre_yx: tuple[float, float], mapping: SubhaloMassMapping) -> SubhaloEstimate:
    return SubhaloEstimate(log10_mass=log10_mass, centre_yx=centre_yx,
                           profile_scales=dict(mapping.profile_scales(log10_mass)),
                           margin_to_lower_dex=log10_mass - mapping.support.log10_mass_min,
                           margin_to_upper_dex=mapping.support.log10_mass_max - log10_mass)


def extract_recovery(result: Any, mapping: SubhaloMassMapping, model: FitModel,
                     refinement: RefineOutcome | None, diagnostics: SamplerDiagnostics | None = None) -> SubhaloRecovery:
    """Recovery from a finished freed H1 search and, when it ran, its refinement.

    ``result`` is the AutoFit result of the search of ``model``; the subhalo is read at
    ``model.subhalo_path``. Centre component 0 is y.
    """
    path = model.subhalo_path
    if path is None:
        raise ValueError("recovery reads the subhalo role's model; this model has no subhalo")
    subhalo = result.max_log_likelihood_instance
    for name in path:
        subhalo = getattr(subhalo, name)
    centre_y, centre_x = (float(value) for value in subhalo.centre)
    sampler = _estimate(float(subhalo.log10_m200), (centre_y, centre_x), mapping)
    refined = None
    if refinement is not None and refinement.best_vector is not None:
        prefix = ".".join(path)
        names = list(model.parameter_names)

        def element(suffix: str) -> float:
            return float(refinement.best_vector[names.index(f"{prefix}.{suffix}")])

        refined = _estimate(element("log10_m200"), (element("centre.centre_0"), element("centre.centre_1")), mapping)
    samples = result.samples
    pdf_converged = bool(samples.pdf_converged)
    mass = np.asarray(samples.values_for_path(path + ("log10_m200",)), dtype=float)
    centre_y_values = np.asarray(samples.values_for_path(path + ("centre", "centre_0")), dtype=float)
    centre_x_values = np.asarray(samples.values_for_path(path + ("centre", "centre_1")), dtype=float)
    weights = np.asarray(samples.weight_list, dtype=float)
    usable = pdf_converged and mass.size > 0
    quality = ("unavailable" if not usable else "complete" if diagnostics is not None
               and diagnostics.criteria_satisfied is True else "provisional")
    return SubhaloRecovery(
        sampler=sampler, refined=refined,
        log10_mass_quantiles=weighted_quantiles(mass, weights) if usable else None,
        centre_y_quantiles=weighted_quantiles(centre_y_values, weights) if usable else None,
        centre_x_quantiles=weighted_quantiles(centre_x_values, weights) if usable else None,
        support=mapping.support, pdf_converged=pdf_converged, sample_count=int(mass.size),
        posterior_quality=quality)
