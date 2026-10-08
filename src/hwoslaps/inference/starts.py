"""Refinement starts from a finished search: the sampler maximum and separated posterior samples.

Start 0 is the sampler's maximum-likelihood sample (the incumbent). Posterior samples follow in
order of decreasing saved log-likelihood; a sample becomes a start when, against every start
already chosen, it lies at least ``start_separation_normalized_l2`` apart in the unit box (the
rule for broad posteriors) or at least ``start_separation_posterior_sigma`` weighted posterior
standard deviations apart (the scale a sharp, noise-free posterior has). Each start carries its
physical vector and its unit-box image ``z = (x - lower) / (upper - lower)``; the optimiser
consumes only the latter.
"""

from __future__ import annotations

import csv
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np

from .settings import RefineSettings

__all__ = ["RefineStart", "normalize", "posterior_sigma_normalized", "prior_box", "select_starts"]

SOURCES = ("sampler_ml", "sampler_sample")


@dataclass(frozen=True)
class RefineStart:
    """One start: physical vector, unit-box vector, where it came from and whether it counts as
    an original start for the support gate (the separated samples do; the incumbent does not)."""

    index: int
    physical: tuple[float, ...]
    normalized: tuple[float, ...]
    source: Literal["sampler_ml", "sampler_sample"]
    original: bool
    origin: Mapping[str, Any]

    def __post_init__(self) -> None:
        physical = tuple(float(value) for value in self.physical)
        normalized = tuple(float(value) for value in self.normalized)
        if self.source not in SOURCES:
            raise ValueError(f"start source must be one of {SOURCES}, got {self.source!r}")
        if not physical or len(physical) != len(normalized) or not np.all(np.isfinite(physical + normalized)):
            raise ValueError(f"start {self.index} needs finite physical and unit-box vectors of one length")
        if not all(0.0 <= value <= 1.0 for value in normalized):
            raise ValueError(f"start {self.index} lies outside the unit box: {normalized}")
        object.__setattr__(self, "physical", physical)
        object.__setattr__(self, "normalized", normalized)

    @classmethod
    def from_physical(cls, *, index: int, physical: Sequence[float], lower: Sequence[float],
                      upper: Sequence[float], source: str, original: bool,
                      origin: Mapping[str, Any]) -> RefineStart:
        """A start from a physical vector inside the prior box, normalized exactly once."""
        lower_array, upper_array, _ = prior_box(lower, upper)
        x = _finite_vector(physical, lower_array, upper_array)
        z = normalize(x, lower_array, upper_array)
        return cls(index=int(index), physical=tuple(float(item) for item in x),
                   normalized=tuple(float(item) for item in z), source=source, original=bool(original),
                   origin=origin)

    def to_mapping(self) -> dict[str, Any]:
        return {"index": self.index, "physical": list(self.physical), "normalized": list(self.normalized),
                "source": self.source, "original": self.original, "origin": dict(self.origin)}


def prior_box(lower: Sequence[float], upper: Sequence[float]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Finite prior bounds and their strictly positive widths."""
    lower_array = np.asarray(lower, dtype=float)
    upper_array = np.asarray(upper, dtype=float)
    if (lower_array.ndim != 1 or lower_array.shape != upper_array.shape or not np.all(np.isfinite(lower_array))
            or not np.all(np.isfinite(upper_array)) or np.any(upper_array <= lower_array)):
        raise ValueError("invalid finite prior box")
    return lower_array, upper_array, upper_array - lower_array


def normalize(physical: Sequence[float], lower: Sequence[float], upper: Sequence[float]) -> np.ndarray:
    """The unit-box image ``(x - lower) / (upper - lower)`` of a physical vector in the box.

    The round trip is checked (1e-12 relative, 1e-12 of the largest width absolute), so a start
    is never evaluated at a point other than its physical origin.
    """
    lower_array, upper_array, widths = prior_box(lower, upper)
    x = _finite_vector(physical, lower_array, upper_array)
    z = (x - lower_array) / widths
    if np.any(z < 0.0) or np.any(z > 1.0) or not np.all(np.isfinite(z)):
        raise ValueError("normalized start left the unit box")
    round_trip = lower_array + z * widths
    if not np.allclose(round_trip, x, rtol=1.0e-12, atol=1.0e-12 * np.max(widths)):
        raise ValueError("normalized start does not map back to its physical origin")
    return z


def _finite_vector(vector: Sequence[float], lower: np.ndarray, upper: np.ndarray) -> np.ndarray:
    value = np.asarray(vector, dtype=float)
    if value.shape != lower.shape or not np.all(np.isfinite(value)):
        raise ValueError("start vector has the wrong shape or non-finite values")
    if np.any(value < lower) or np.any(value > upper):
        raise ValueError("start vector lies outside the prior support")
    return value


def _sampler_maximum(summary_path: Path, names: Sequence[str]) -> tuple[np.ndarray, dict[str, Any]]:
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    try:
        sample = payload["arguments"]["max_log_likelihood_sample"]["arguments"]
        values = sample["kwargs"]["arguments"]
    except (KeyError, TypeError) as error:
        raise ValueError(f"{summary_path} has no maximum-likelihood sample") from error
    if not isinstance(values, Mapping) or set(values) != set(names):
        raise ValueError(f"the maximum-likelihood sample of {summary_path} names other parameters than the model")
    return (np.asarray([float(values[name]) for name in names], dtype=float),
            {"saved_log_likelihood": sample.get("log_likelihood")})


def posterior_sigma_normalized(rows: Sequence[Mapping[str, Any]], names: Sequence[str], lower: np.ndarray,
                               widths: np.ndarray) -> np.ndarray:
    """Weighted posterior standard deviation per parameter, in unit-box units."""
    if not rows or "weight" not in rows[0]:
        raise ValueError("sampler samples must carry posterior weights")
    vectors = np.asarray([[float(row[name]) for name in names] for row in rows], dtype=float)
    weights = np.asarray([float(row["weight"]) for row in rows], dtype=float)
    if vectors.ndim != 2 or vectors.shape[1] != lower.shape[0] or not np.all(np.isfinite(vectors)):
        raise ValueError("sampler samples must be finite vectors of the model dimension")
    if not np.all(np.isfinite(weights)) or np.any(weights < 0) or weights.sum() <= 0:
        raise ValueError("sampler sample weights must be finite, non-negative and not all zero")
    weights = weights / weights.sum()
    z = (vectors - lower) / widths
    mean = weights @ z
    sigma = np.sqrt(weights @ (z - mean) ** 2)
    if not np.all(np.isfinite(sigma)) or np.any(sigma <= 0):
        raise ValueError("posterior sigma must be positive and finite for every parameter")
    return sigma


def select_starts(files_dir: Path, names: Sequence[str], lower: Sequence[float], upper: Sequence[float],
                  settings: RefineSettings) -> tuple[RefineStart, ...]:
    """The incumbent and ``settings.original_start_count`` separated samples of one search.

    ``files_dir`` is the search's AutoFit ``files`` directory, holding ``samples_summary.json``
    and ``samples.csv``. Raises ValueError when too few separated samples exist.
    """
    files_dir = Path(files_dir)
    lower_array, upper_array, widths = prior_box(lower, upper)
    maximum, maximum_origin = _sampler_maximum(files_dir / "samples_summary.json", names)
    selected = [_finite_vector(maximum, lower_array, upper_array)]
    with (files_dir / "samples.csv").open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream, skipinitialspace=True))
    posterior_sigma = posterior_sigma_normalized(rows, names, lower_array, widths)
    origins: list[Mapping[str, Any]] = [{
        **maximum_origin,
        "selection_rule": {
            "distinct_if": "prior_normalized_l2 >= start_separation_normalized_l2 "
                           "or posterior_sigma_l2 >= start_separation_posterior_sigma",
            "start_separation_normalized_l2": float(settings.start_separation_normalized_l2),
            "start_separation_posterior_sigma": float(settings.start_separation_posterior_sigma),
            "posterior_sigma_normalized": posterior_sigma.tolist(),
        },
    }]
    ranked = sorted(enumerate(rows), key=lambda pair: (-float(pair[1]["log_likelihood"]), pair[0]))
    for row_index, row in ranked:
        vector = np.asarray([float(row[name]) for name in names], dtype=float)
        if vector.shape != lower_array.shape or not np.all(np.isfinite(vector)):
            continue
        if np.any(vector < lower_array) or np.any(vector > upper_array):
            continue
        deltas = [(vector - old) / widths for old in selected]
        prior_l2 = [float(np.linalg.norm(delta)) for delta in deltas]
        sigma_l2 = [float(np.linalg.norm(delta / posterior_sigma)) for delta in deltas]
        distinct = all(prior >= settings.start_separation_normalized_l2
                       or sigma >= settings.start_separation_posterior_sigma
                       for prior, sigma in zip(prior_l2, sigma_l2))
        if not distinct:
            continue
        selected.append(vector)
        origins.append({"row": int(row_index), "saved_log_likelihood": float(row["log_likelihood"]),
                        "separation_prior_normalized_l2": min(prior_l2),
                        "separation_posterior_sigma": min(sigma_l2)})
        if len(selected) == settings.original_start_count + 1:
            break
    if len(selected) != settings.original_start_count + 1:
        raise ValueError(f"the search supplied {len(selected) - 1} separated samples besides the "
                         f"maximum-likelihood incumbent; refinement needs {settings.original_start_count}")
    return tuple(RefineStart.from_physical(index=index, physical=vector, lower=lower_array, upper=upper_array,
                                           source="sampler_ml" if index == 0 else "sampler_sample",
                                           original=index != 0, origin=origin)
                 for index, (vector, origin) in enumerate(zip(selected, origins)))
