"""Observation statistics and deterministic ranking under a caller-supplied policy."""

from __future__ import annotations

import math
import operator
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
from numpy.typing import ArrayLike

from ..config.checks import Boolean, ConfigError, Integer, Key, ListOf, Nullable, Real, Table, Text
from ..constants import ARCSEC_PER_RAD
from ..fisher.data_space import annulus_mask
from ..identity import text_digest
from ..scene.spec import pixel_centres_yx

if TYPE_CHECKING:
    from ..observation.observation import Observation

__all__ = ["ElectronMaps", "electron_maps", "aperture_mask", "arc_snr", "gradient_power",
           "diffraction_scale_arcsec", "complexity", "standardize", "rank", "spearman_rank_correlation",
           "top_k_jaccard", "top_k_recovery", "Cut", "ScoreTerm", "RankingPolicy", "RankingResult", "rank_pool"]


@dataclass(frozen=True, eq=False)
class ElectronMaps:
    signal_e: np.ndarray
    variance_e2: np.ndarray
    pixel_scale_arcsec: float


def electron_maps(observation: Observation) -> ElectronMaps:
    """Source-plane electrons and the observation's all-light detector variance."""
    exposure = observation.exposure
    return ElectronMaps(observation.light_rate_by_plane_e_per_s["source"] * exposure.exposure_time_s,
                        (observation.noise_map_adu * exposure.detector.gain_e_per_adu) ** 2,
                        observation.grid.pixel_scale_arcsec)


def aperture_mask(observation: Observation, *, centre_yx: tuple[float, float], radius_arcsec: float) -> np.ndarray:
    """Closed disc on the observation's pixel grid; an empty aperture raises."""
    radius = _positive(radius_arcsec, "radius_arcsec")
    y, x = pixel_centres_yx(observation.grid.shape, observation.grid.pixel_scale_arcsec)
    return annulus_mask(y, x, centre_yx=centre_yx, inner_arcsec=0.0, outer_arcsec=radius)


def _array(values: ArrayLike, name: str, ndim: int | None = None) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if ndim is not None and array.ndim != ndim:
        raise ValueError(f"{name} must be {ndim}-dimensional, got shape {array.shape}")
    if array.size == 0 or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be non-empty and finite")
    return array


def _positive(value: float, name: str) -> float:
    return Real(min=0.0, min_open=True)(value, name)


def _image_inputs(signal_e: ArrayLike, variance_e2: ArrayLike, mask: np.ndarray | None,
                  ndim: int | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    signal, variance = _array(signal_e, "signal_e", ndim), _array(variance_e2, "variance_e2", ndim)
    if signal.shape != variance.shape:
        raise ValueError(f"signal_e shape {signal.shape} does not match variance_e2 shape {variance.shape}")
    if np.any(variance <= 0):
        raise ValueError("variance_e2 must be positive")
    selected = np.ones(signal.shape, dtype=bool) if mask is None else np.asarray(mask)
    if selected.dtype != bool or selected.shape != signal.shape or not np.any(selected):
        raise ValueError("mask must be boolean, match the image shape and select at least one pixel")
    return signal, variance, selected


def arc_snr(signal_e: ArrayLike, variance_e2: ArrayLike, mask: np.ndarray | None = None) -> float:
    """Quadrature signal-to-noise over selected pixels; negative noisy signals are valid."""
    signal, variance, selected = _image_inputs(signal_e, variance_e2, mask)
    return float(np.sqrt(np.sum(signal[selected] ** 2 / variance[selected])))


def gradient_power(signal_e: ArrayLike, variance_e2: ArrayLike, pixel_scale_arcsec: float,
                   mask: np.ndarray | None = None) -> float:
    """Noise-weighted squared angular gradient, using numpy's two-axis differences."""
    signal, variance, selected = _image_inputs(signal_e, variance_e2, mask, ndim=2)
    if min(signal.shape) < 3:
        raise ValueError("signal_e needs at least three pixels on both axes")
    scale = _positive(pixel_scale_arcsec, "pixel_scale_arcsec")
    grad_y, grad_x = np.gradient(signal, scale, scale)
    power = grad_y ** 2 + grad_x ** 2
    return float(np.sum(power[selected] / variance[selected]))


def diffraction_scale_arcsec(wavelength_m: float, diameter_m: float) -> float:
    return float(_positive(wavelength_m, "wavelength_m") / _positive(diameter_m, "diameter_m") * ARCSEC_PER_RAD)


def complexity(gradient_power_value: float, arc_snr_value: float, theta_res_arcsec: float) -> float:
    """Brightness-normalized angular structure, theta_res**2 G / S**2."""
    power = _positive(gradient_power_value, "gradient_power_value")
    snr = _positive(arc_snr_value, "arc_snr_value")
    theta = _positive(theta_res_arcsec, "theta_res_arcsec")
    return float(theta ** 2 * power / snr ** 2)


def standardize(values: ArrayLike) -> np.ndarray:
    """Population z-scores with exact sums, bitwise invariant under permutation."""
    array = _array(values, "values", ndim=1)
    deviations = array - math.fsum(array.tolist()) / array.size
    spread = math.sqrt(math.fsum((deviations ** 2).tolist()) / array.size)
    return np.zeros_like(array) if spread == 0 else deviations / spread


def _ids(values: Sequence[str], name: str) -> tuple[str, ...]:
    ids = tuple(values)
    if not all(isinstance(value, str) and value for value in ids):
        raise ValueError(f"{name} must contain non-empty string ids")
    if len(set(ids)) != len(ids):
        raise ValueError(f"{name} ids must be unique")
    return ids


def rank(ids: Sequence[str], keys: ArrayLike, *, descending: bool) -> tuple[str, ...]:
    """Order by value, then ascending SHA256 of each id to break ties."""
    names = _ids(ids, "ids")
    values = _array(keys, "keys", ndim=1)
    Boolean()(descending, "descending")
    if values.shape != (len(names),):
        raise ValueError("ids and keys must have the same number of entries")
    sign = -1.0 if descending else 1.0
    order = sorted(range(len(names)), key=lambda i: (sign * float(values[i]), text_digest(names[i])))
    return tuple(names[i] for i in order)


def _average_ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="stable")
    ordered = values[order]
    ranks = np.empty(values.size, dtype=float)
    start = 0
    while start < values.size:
        stop = start
        while stop + 1 < values.size and ordered[stop + 1] == ordered[start]:
            stop += 1
        ranks[order[start:stop + 1]] = 0.5 * (start + stop) + 1.0
        start = stop + 1
    return ranks


def spearman_rank_correlation(x: ArrayLike, y: ArrayLike) -> float:
    """Pearson correlation of average ranks, including tied entries."""
    first, second = _array(x, "x", ndim=1), _array(y, "y", ndim=1)
    if first.shape != second.shape or first.size < 2:
        raise ValueError("x and y need the same number of entries, at least two")
    rank_x, rank_y = _average_ranks(first), _average_ranks(second)
    dx, dy = rank_x - float(np.mean(rank_x)), rank_y - float(np.mean(rank_y))
    denominator = math.sqrt(float(np.sum(dx ** 2)) * float(np.sum(dy ** 2)))
    if denominator == 0:
        raise ValueError("Spearman correlation is undefined for a fully tied vector")
    return float(np.sum(dx * dy) / denominator)


def _top_k(ranking: Sequence[str], k: int, name: str) -> set[str]:
    ids = _ids(ranking, name)
    size = Integer(min=1)(k, "k")
    if size > len(ids):
        raise ValueError(f"{name} has fewer than k entries")
    return set(ids[:size])


def top_k_jaccard(ranking_a: Sequence[str], ranking_b: Sequence[str], k: int) -> float:
    first, second = _top_k(ranking_a, k, "ranking_a"), _top_k(ranking_b, k, "ranking_b")
    return float(len(first & second) / len(first | second))


def top_k_recovery(ranking: Sequence[str], reference: Sequence[str], k: int) -> float:
    return float(len(_top_k(ranking, k, "ranking") & _top_k(reference, k, "reference")) / k)


_OPERATORS = {">": operator.gt, ">=": operator.ge, "<": operator.lt, "<=": operator.le}


@dataclass(frozen=True)
class Cut:
    feature: str
    operator: Literal[">", ">=", "<", "<="]
    threshold: float

    def __post_init__(self) -> None:
        Text()(self.feature, "cut.feature")
        Text(choices=tuple(_OPERATORS))(self.operator, "cut.operator")
        Real()(self.threshold, "cut.threshold")


@dataclass(frozen=True)
class ScoreTerm:
    feature: str
    weight: float
    log: bool = False

    def __post_init__(self) -> None:
        Text()(self.feature, "term.feature")
        Real()(self.weight, "term.weight")
        if self.weight == 0:
            raise ConfigError("term.weight", "must be non-zero")
        Boolean()(self.log, "term.log")


_CUT_TABLE = Table((Key("feature", Text(), "feature to compare"),
                    Key("operator", Text(choices=tuple(_OPERATORS)), "comparison operator"),
                    Key("threshold", Real(), "finite cut threshold")))
_TERM_TABLE = Table((Key("feature", Text(), "feature to score"),
                     Key("weight", Real(), "finite non-zero score weight"),
                     Key("log", Boolean(), "natural log before standardizing", False)))
_POLICY_TABLE = Table((Key("terms", ListOf(_TERM_TABLE, min_length=1), "score terms"),
                       Key("cuts", ListOf(_CUT_TABLE), "pool cuts", []),
                       Key("select", Nullable(Integer(min=1)), "top-k size, null ranks only", None)))


@dataclass(frozen=True)
class RankingPolicy:
    terms: tuple[ScoreTerm, ...]
    cuts: tuple[Cut, ...] = ()
    select: int | None = None

    def __post_init__(self) -> None:
        terms, cuts = tuple(self.terms), tuple(self.cuts)
        if not terms or not all(isinstance(term, ScoreTerm) for term in terms):
            raise ConfigError("policy.terms", "must contain at least one ScoreTerm")
        if not all(isinstance(cut, Cut) for cut in cuts):
            raise ConfigError("policy.cuts", "must contain Cut values")
        Nullable(Integer(min=1))(self.select, "policy.select")
        object.__setattr__(self, "terms", terms)
        object.__setattr__(self, "cuts", cuts)

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, Any]) -> RankingPolicy:
        values = _POLICY_TABLE.read(mapping, "policy")
        return cls(tuple(ScoreTerm(**term) for term in values["terms"]),
                   tuple(Cut(**cut) for cut in values["cuts"]), values["select"])

    def to_mapping(self) -> dict[str, Any]:
        return {"terms": [{"feature": t.feature, "weight": t.weight, "log": t.log} for t in self.terms],
                "cuts": [{"feature": c.feature, "operator": c.operator, "threshold": c.threshold} for c in self.cuts],
                "select": self.select}


@dataclass(frozen=True)
class RankingResult:
    ids: tuple[str, ...]
    passed: tuple[bool, ...]
    survivors: tuple[str, ...]
    scores: tuple[float, ...]
    ranking: tuple[str, ...]
    selected: tuple[str, ...]
    policy: RankingPolicy


def rank_pool(ids: Sequence[str], features: Mapping[str, ArrayLike], policy: RankingPolicy) -> RankingResult:
    """Cut, standardize over survivors, score and rank under the supplied policy."""
    names = _ids(ids, "ids")
    arrays = {name: _array(values, f"features.{name}", ndim=1) for name, values in features.items()}
    if any(values.shape != (len(names),) for values in arrays.values()):
        raise ValueError("every feature must have the same number of entries as ids")
    required = {term.feature for term in policy.terms} | {cut.feature for cut in policy.cuts}
    if not required <= arrays.keys():
        raise ValueError(f"missing features: {sorted(required - arrays.keys())}")
    passed = np.ones(len(names), dtype=bool)
    for cut in policy.cuts:
        passed &= _OPERATORS[cut.operator](arrays[cut.feature], cut.threshold)
    survivors = tuple(name for name, keep in zip(names, passed) if keep)
    if not survivors:
        raise ValueError("no pool member passes the policy cuts")
    if policy.select is not None and policy.select > len(survivors):
        raise ValueError("too few survivors for policy.select")
    score = np.zeros(len(survivors), dtype=float)
    for term in policy.terms:
        values = arrays[term.feature][passed]
        if term.log:
            if np.any(values <= 0):
                raise ValueError(f"features.{term.feature} must be strictly positive for a log score")
            values = np.log(values)
        score += term.weight * standardize(values)
    ranking = rank(survivors, score, descending=True)
    return RankingResult(names, tuple(bool(value) for value in passed), survivors,
                         tuple(float(value) for value in score), ranking,
                         () if policy.select is None else ranking[:policy.select], policy)
