"""Freed-subhalo recovery: the sampler and refined estimates and the posterior quantiles.

Raw values only. The margins to the ends of the mass support are reported in dex; a boundary
flag or a band fraction is a threshold the caller applies. The sampler estimate is the
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
    from .result import RefineOutcome
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
    """Freed-subhalo summary of the H1 search. Quantiles are (p16, p50, p84) of the sampler
    samples, None when the sampler reports an unconverged posterior."""

    sampler: SubhaloEstimate
    refined: SubhaloEstimate | None
    log10_mass_quantiles: tuple[float, float, float] | None
    centre_y_quantiles: tuple[float, float, float] | None
    centre_x_quantiles: tuple[float, float, float] | None
    support: MassSupport
    pdf_converged: bool
    sample_count: int

    def to_mapping(self) -> dict[str, Any]:
        return {"sampler": self.sampler.to_mapping(),
                "refined": None if self.refined is None else self.refined.to_mapping(),
                "log10_mass_quantiles": _list(self.log10_mass_quantiles),
                "centre_y_quantiles": _list(self.centre_y_quantiles),
                "centre_x_quantiles": _list(self.centre_x_quantiles),
                "support": self.support.to_mapping(), "pdf_converged": self.pdf_converged,
                "sample_count": self.sample_count}

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, Any]) -> SubhaloRecovery:
        check_record_keys(mapping, cls)
        refined = mapping["refined"]
        return cls(sampler=SubhaloEstimate.from_mapping(mapping["sampler"]),
                   refined=None if refined is None else SubhaloEstimate.from_mapping(refined),
                   log10_mass_quantiles=_quantiles(mapping["log10_mass_quantiles"]),
                   centre_y_quantiles=_quantiles(mapping["centre_y_quantiles"]),
                   centre_x_quantiles=_quantiles(mapping["centre_x_quantiles"]),
                   support=MassSupport.from_mapping(mapping["support"], path="support"),
                   pdf_converged=bool(mapping["pdf_converged"]), sample_count=int(mapping["sample_count"]))


def _list(values: tuple[float, ...] | None) -> list[float] | None:
    return None if values is None else list(values)


def weighted_quantiles(values: np.ndarray, weights: np.ndarray | None,
                       probabilities: Sequence[float] = (0.16, 0.5, 0.84)) -> tuple[float, ...]:
    """Quantiles of weighted samples by the midpoint empirical CDF.

    Sorted samples get cumulative probability ``(cumsum(w) - w / 2) / sum(w)`` and quantiles
    interpolate the sorted values against it, clamped to the smallest and largest sample.
    Without weights, with fewer than two samples, with equal weights, or with a non-finite or
    non-positive weight sum, the result is ``np.quantile`` of the values.
    """
    values = np.asarray(values, dtype=float)
    if weights is None:
        return tuple(float(value) for value in np.quantile(values, list(probabilities)))
    weights = np.asarray(weights, dtype=float)
    if weights.shape != values.shape:
        raise ValueError(f"weights of shape {weights.shape} do not match values of shape {values.shape}")
    order = np.argsort(values)
    sorted_values = values[order]
    sorted_weights = weights[order]
    if sorted_weights.size < 2 or np.all(sorted_weights == sorted_weights[0]):
        return tuple(float(value) for value in np.quantile(values, list(probabilities)))
    normalization = float(np.sum(sorted_weights))
    if not np.isfinite(normalization) or normalization <= 0.0:
        return tuple(float(value) for value in np.quantile(values, list(probabilities)))
    cumulative = (np.cumsum(sorted_weights) - 0.5 * sorted_weights) / normalization
    return tuple(float(np.interp(probability, cumulative, sorted_values)) for probability in probabilities)


def _estimate(log10_mass: float, centre_yx: tuple[float, float], mapping: SubhaloMassMapping) -> SubhaloEstimate:
    return SubhaloEstimate(log10_mass=log10_mass, centre_yx=centre_yx,
                           profile_scales=dict(mapping.profile_scales(log10_mass)),
                           margin_to_lower_dex=log10_mass - mapping.support.log10_mass_min,
                           margin_to_upper_dex=mapping.support.log10_mass_max - log10_mass)


def extract_recovery(result: Any, mapping: SubhaloMassMapping, model: FitModel,
                     refinement: RefineOutcome | None) -> SubhaloRecovery:
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
    converged = pdf_converged and mass.size > 0
    return SubhaloRecovery(
        sampler=sampler, refined=refined,
        log10_mass_quantiles=weighted_quantiles(mass, weights) if converged else None,
        centre_y_quantiles=weighted_quantiles(centre_y_values, weights) if converged else None,
        centre_x_quantiles=weighted_quantiles(centre_x_values, weights) if converged else None,
        support=mapping.support, pdf_converged=pdf_converged, sample_count=int(mass.size))
