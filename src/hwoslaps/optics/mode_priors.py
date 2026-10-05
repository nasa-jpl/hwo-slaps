"""Mode-weight priors and direction draws in orthonormal aperture bases.

A prior is shape only: per side (global Zernikes, Noll >= 4; segment hexikes, Noll >= 1)
the weights are normalized to unit sum of squares, and ``segment_variance_fraction``
splits a combined draw's budget between the sides. The weights scale coefficients of the
sequentially orthonormalized aperture bases of ``optics.aperture_basis``, not raw HCIPy
modes, and exact-RMS conditioning of each draw makes the realized variance fractions
differ from the squared weights (by up to 15% per mode for the JWST drift table).

Priors come packaged (``jwst_wss_static_v1``, ``jwst_wss_drift_v1``, derived from JWST
wavefront sensing by ``scripts/derive_jwst_mode_weight_tables.py``), from a YAML table
file, or as a radial-order power law for other telescopes. Draws keep the random-number
order of the prior tables' derivation: a side with no budget consumes no random numbers.
"""

from __future__ import annotations

import dataclasses
import hashlib
import importlib.resources
import math
import types
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from numbers import Integral
from pathlib import Path
from typing import Any, Literal

import numpy as np
import yaml

from ..config.checks import ConfigError, FilePath, Integer, Key, Nullable, Pair, Real, Rule, Table, Text
from ..identity import mapping_digest

__all__ = [
    "PACKAGED_PRIORS", "PRIOR_TABLE", "ModeWeightPrior", "ModeWeightPriorSpec", "PowerLawPriorSpec",
    "draw_combined_orthonormal", "draw_global_orthonormal", "draw_segment_orthonormal", "load_prior",
    "noll_radial_order", "parse_mode_weight_prior", "parse_prior", "power_law_prior",
]

PACKAGED_PRIORS: tuple[str, ...] = ("jwst_wss_static_v1", "jwst_wss_drift_v1")
"""Names of the prior tables shipped in ``hwoslaps/optics/priors``."""


@dataclass(frozen=True)
class PowerLawPriorSpec:
    """Weights ``n^-alpha`` (global) and ``(n + 1)^-alpha`` (segment) over inclusive Noll ranges."""

    alpha: float
    global_nolls: tuple[int, int] | None
    segment_nolls: tuple[int, int] | None
    segment_variance_fraction: float


@dataclass(frozen=True)
class ModeWeightPriorSpec:
    """Where a mode-weight prior comes from: a packaged name, an absolute file path, or a power law."""

    kind: Literal["packaged", "path", "power_law"]
    name: str | None
    path: Path | None
    power_law: PowerLawPriorSpec | None

    def __post_init__(self) -> None:
        present = {"packaged": self.name, "path": self.path, "power_law": self.power_law}
        if self.kind not in present:
            raise ValueError(f"prior kind must be one of {list(present)}, got {self.kind!r}")
        if any(value is not None for kind, value in present.items() if kind != self.kind):
            raise ValueError(f"a {self.kind} prior sets only its own field, got {self!r}")
        if self.kind == "packaged" and not isinstance(self.name, str):
            raise ValueError(f"a packaged prior needs a name, got {self.name!r}")
        if self.kind == "path" and not (isinstance(self.path, Path) and self.path.is_absolute()):
            raise ValueError(f"a prior file needs an absolute Path (relative paths resolve against the "
                             f"configuration file that names them), got {self.path!r}")
        if self.kind == "power_law" and not isinstance(self.power_law, PowerLawPriorSpec):
            raise ValueError(f"a power-law prior needs a PowerLawPriorSpec, got {self.power_law!r}")


def _check_power_law(values: Mapping[str, Any], path: str) -> None:
    if values["global_nolls"] is None and values["segment_nolls"] is None:
        raise ConfigError(path, "set global_nolls, segment_nolls or both")
    for name in ("global_nolls", "segment_nolls"):
        if values[name] is not None and values[name][0] > values[name][1]:
            raise ConfigError(f"{path}.{name}" if path else name, "the range (lo, hi) needs lo <= hi")


POWER_LAW_TABLE = Table((
    Key("alpha", Real(min=0.0), "power-law index of the weights in radial order"),
    Key("global_nolls", Nullable(Pair(Integer(min=4))), "inclusive global Zernike Noll range, or null for none"),
    Key("segment_nolls", Nullable(Pair(Integer(min=1))), "inclusive segment hexike Noll range, or null for none"),
    Key("segment_variance_fraction", Real(min=0.0, max=1.0), "share of a combined draw's variance on segments"),
), rules=(Rule("at least one side; each range has lo <= hi", _check_power_law),))

PRIOR_TABLE = Table((
    Key("packaged", Nullable(Text(choices=PACKAGED_PRIORS)), "a prior table shipped with hwoslaps", default=None),
    Key("path", Nullable(FilePath((".yaml",))), "a prior table file", default=None),
    Key("power_law", Nullable(POWER_LAW_TABLE), "a radial-order power-law prior", default=None),
), exactly_one=(("packaged", "path", "power_law"),))


def parse_prior(mapping: Mapping[str, Any], path: str) -> ModeWeightPriorSpec:
    """The prior specification of a ``prior`` mapping, read strictly through ``PRIOR_TABLE``."""
    values = PRIOR_TABLE.read(mapping, path)
    if values["packaged"] is not None:
        return ModeWeightPriorSpec("packaged", values["packaged"], None, None)
    if values["path"] is not None:
        return ModeWeightPriorSpec("path", None, Path(values["path"]), None)
    law = values["power_law"]
    return ModeWeightPriorSpec("power_law", None, None, PowerLawPriorSpec(
        law["alpha"], None if law["global_nolls"] is None else tuple(law["global_nolls"]),
        None if law["segment_nolls"] is None else tuple(law["segment_nolls"]), law["segment_variance_fraction"]))


def _unit_weights(weights: Any, name: str, minimum_noll: int) -> Mapping[int, float]:
    """One side of a prior, validated and normalized to unit sum of squares, ascending."""
    if not isinstance(weights, Mapping):
        raise ValueError(f"{name} must be a mapping of Noll index to weight")
    validated = {}
    for noll, weight in weights.items():
        if isinstance(noll, (bool, np.bool_)) or not isinstance(noll, Integral) or int(noll) < minimum_noll:
            raise ValueError(f"{name} keys must be integer Noll indices >= {minimum_noll}, got {noll!r}")
        if isinstance(weight, (bool, np.bool_)):
            raise ValueError(f"{name} weights must be finite and non-negative, got {weight!r}")
        try:
            value = float(weight)
        except (TypeError, ValueError) as error:
            raise ValueError(f"{name} weights must be finite and non-negative, got {weight!r}") from error
        if not (np.isfinite(value) and value >= 0.0):
            raise ValueError(f"{name} weights must be finite and non-negative, got {weight!r}")
        validated[int(noll)] = value
    if not validated:
        return types.MappingProxyType({})
    norm = float(np.linalg.norm(list(validated.values())))
    if norm == 0.0:
        raise ValueError(f"{name} must have a positive sum of squared weights")
    return types.MappingProxyType({noll: validated[noll] / norm for noll in sorted(validated)})


@dataclass(frozen=True)
class ModeWeightPrior:
    """A shape-only prior: unit-norm weights per side and the combined-draw variance split."""

    name: str
    global_weights: Mapping[int, float]
    segment_weights: Mapping[int, float]
    segment_variance_fraction: float
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("a prior needs a non-empty name")
        global_weights = _unit_weights(self.global_weights, "global_weights", 4)
        segment_weights = _unit_weights(self.segment_weights, "segment_weights", 1)
        if not global_weights and not segment_weights:
            raise ValueError("global_weights and segment_weights must not both be empty")
        fraction = self.segment_variance_fraction
        if isinstance(fraction, (bool, np.bool_)):
            raise ValueError("segment_variance_fraction must be finite and in [0, 1]")
        try:
            fraction = float(fraction)
        except (TypeError, ValueError) as error:
            raise ValueError("segment_variance_fraction must be finite and in [0, 1]") from error
        if not (np.isfinite(fraction) and 0.0 <= fraction <= 1.0):
            raise ValueError("segment_variance_fraction must be finite and in [0, 1]")
        if not isinstance(self.metadata, Mapping):
            raise ValueError("metadata must be a mapping")
        object.__setattr__(self, "global_weights", global_weights)
        object.__setattr__(self, "segment_weights", segment_weights)
        object.__setattr__(self, "segment_variance_fraction", fraction)
        object.__setattr__(self, "metadata", types.MappingProxyType(dict(self.metadata)))


_DOCUMENT_KEYS = ("name", "segment_variance_fraction", "global_weights", "segment_weights", "metadata")


def parse_mode_weight_prior(document_bytes: bytes | str) -> ModeWeightPrior:
    """A prior from the bytes of its YAML table; a caller recording a digest hashes these bytes."""
    try:
        document = yaml.safe_load(document_bytes)
    except yaml.YAMLError as error:
        raise ValueError(f"invalid mode-weight prior YAML: {error}") from error
    if not isinstance(document, dict):
        raise ValueError("a mode-weight prior document must be a mapping")
    unknown = sorted(set(document) - set(_DOCUMENT_KEYS), key=str)
    if unknown:
        raise ValueError(f"unknown mode-weight prior key {unknown[0]!r}; keys are {list(_DOCUMENT_KEYS)}")
    for required in ("name", "segment_variance_fraction"):
        if required not in document:
            raise ValueError(f"a mode-weight prior needs {required!r}")
    if "global_weights" not in document and "segment_weights" not in document:
        raise ValueError("a mode-weight prior needs global_weights, segment_weights or both")
    return ModeWeightPrior(document["name"], document.get("global_weights", {}), document.get("segment_weights", {}),
                           document["segment_variance_fraction"], document.get("metadata", {}))


def noll_radial_order(noll: int) -> int:
    """The radial order ``n`` of a Noll index."""
    if isinstance(noll, (bool, np.bool_)) or not isinstance(noll, Integral) or noll < 1:
        raise ValueError(f"a Noll index is an integer >= 1, got {noll!r}")
    return (math.isqrt(8 * int(noll) - 7) - 1) // 2


def _noll_range(bounds: Sequence[int] | None, name: str, minimum: int) -> tuple[int, int] | None:
    if bounds is None:
        return None
    if not isinstance(bounds, (tuple, list)) or len(bounds) != 2 or not all(
            isinstance(b, Integral) and not isinstance(b, (bool, np.bool_)) for b in bounds):
        raise ValueError(f"{name} must be an inclusive pair of integers (lo, hi), got {bounds!r}")
    low, high = int(bounds[0]), int(bounds[1])
    if low < minimum or high < low:
        raise ValueError(f"{name} needs {minimum} <= lo <= hi, got {bounds!r}")
    return low, high


def power_law_prior(alpha: float, *, global_nolls: tuple[int, int] | None, segment_nolls: tuple[int, int] | None,
                    segment_variance_fraction: float) -> ModeWeightPrior:
    """Global weights ``n^-alpha`` and segment weights ``(n + 1)^-alpha`` in radial order ``n``.

    The added one on the segment side gives segment piston (``n = 0``) a finite weight.
    """
    if isinstance(alpha, (bool, np.bool_)) or not isinstance(alpha, (Integral, float)) or not (
            math.isfinite(float(alpha)) and float(alpha) >= 0.0):
        raise ValueError(f"alpha must be a finite non-negative number, got {alpha!r}")
    alpha = float(alpha)
    global_range = _noll_range(global_nolls, "global_nolls", 4)
    segment_range = _noll_range(segment_nolls, "segment_nolls", 1)
    if global_range is None and segment_range is None:
        raise ValueError("global_nolls and segment_nolls must not both be None")
    global_weights = {} if global_range is None else {
        noll: noll_radial_order(noll) ** (-alpha) for noll in range(global_range[0], global_range[1] + 1)}
    segment_weights = {} if segment_range is None else {
        noll: (noll_radial_order(noll) + 1) ** (-alpha) for noll in range(segment_range[0], segment_range[1] + 1)}
    metadata = {"kind": "power_law", "alpha": alpha, "global_nolls": global_range, "segment_nolls": segment_range}
    return ModeWeightPrior(f"power_law_alpha_{alpha:g}", global_weights, segment_weights, segment_variance_fraction,
                           metadata)


def load_prior(spec: ModeWeightPriorSpec) -> tuple[ModeWeightPrior, str]:
    """The prior and its digest: SHA-256 of the exact table bytes parsed, or of the power-law record.

    Packaged names read the package data; a file is read at its absolute path. There is no
    fallback to the working directory or the repository.
    """
    if spec.kind == "power_law":
        law = spec.power_law
        prior = power_law_prior(law.alpha, global_nolls=law.global_nolls, segment_nolls=law.segment_nolls,
                                segment_variance_fraction=law.segment_variance_fraction)
        return prior, mapping_digest({"kind": "power_law", **dataclasses.asdict(law)})
    if spec.kind == "packaged":
        if spec.name not in PACKAGED_PRIORS:
            raise FileNotFoundError(f"no packaged prior {spec.name!r}; packaged priors are {list(PACKAGED_PRIORS)}")
        document = importlib.resources.files(__package__).joinpath("priors", f"{spec.name}.yaml").read_bytes()
    else:
        if not spec.path.is_file():
            raise FileNotFoundError(f"prior table {spec.path} does not exist")
        document = spec.path.read_bytes()
    return parse_mode_weight_prior(document), hashlib.sha256(document).hexdigest()


def _budget(value: Any, what: str) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{what} must be a finite non-negative number, got {value!r}")
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{what} must be a finite non-negative number, got {value!r}") from error
    if not (np.isfinite(number) and number >= 0.0):
        raise ValueError(f"{what} must be a finite non-negative number, got {value!r}")
    return number


def _scaled_to_norm(values: np.ndarray, norm_nm: float) -> np.ndarray:
    norm = float(np.linalg.norm(values))
    if norm == 0.0:
        raise ValueError("cannot scale a zero coefficient vector to a nonzero norm")
    return np.asarray(values, dtype=float) * (float(norm_nm) / norm)


def draw_global_orthonormal(rng: np.random.Generator, prior: ModeWeightPrior,
                            norm_nm: float) -> dict[int, float]:
    """Orthonormal global coefficients with Euclidean norm ``norm_nm``, ascending Noll; empty at 0."""
    target = _budget(norm_nm, "the global draw norm")
    if target == 0.0:
        return {}
    if not prior.global_weights:
        raise ValueError(f"prior {prior.name!r} has no global weights")
    nolls = sorted(prior.global_weights)
    raw = rng.standard_normal(len(nolls))
    weights = np.array([prior.global_weights[noll] for noll in nolls])
    coefficients = _scaled_to_norm(raw * weights, target)
    return {noll: float(coefficients[index]) for index, noll in enumerate(nolls)}


def draw_segment_orthonormal(rng: np.random.Generator, segments: Sequence[int], prior: ModeWeightPrior,
                             rms_nm: float) -> dict[int, dict[int, float]]:
    """Orthonormal per-segment coefficients with RMS ``rms_nm`` per segment; empty at 0.

    The flattened vector is scaled to ``rms_nm sqrt(n_segments)``. With segment piston
    (Noll 1) present only its across-segment mean is removed: a common piston is a global
    phase, while common tip and tilt are a physical sawtooth and are kept.
    """
    target = _budget(rms_nm, "the segment draw RMS")
    segment_ids = [int(segment) for segment in segments]
    if not segment_ids:
        raise ValueError("a segment draw needs at least one segment")
    if target == 0.0:
        return {}
    if not prior.segment_weights:
        raise ValueError(f"prior {prior.name!r} has no segment weights")
    nolls = sorted(prior.segment_weights)
    if 1 in nolls and len(segment_ids) < 2:
        raise ValueError("segment piston (Noll 1) needs at least two segments")
    raw = rng.standard_normal((len(segment_ids), len(nolls)))
    weights = np.array([prior.segment_weights[noll] for noll in nolls])
    weighted = raw * weights[np.newaxis, :]
    if 1 in nolls:
        piston = nolls.index(1)
        weighted[:, piston] -= np.mean(weighted[:, piston])
    matrix = _scaled_to_norm(weighted.ravel(), target * np.sqrt(len(segment_ids))).reshape(weighted.shape)
    return {segment: {noll: float(matrix[row, column]) for column, noll in enumerate(nolls)}
            for row, segment in enumerate(segment_ids)}


def draw_combined_orthonormal(rng: np.random.Generator, segments: Sequence[int], prior: ModeWeightPrior,
                              amplitude_nm: float) -> tuple[dict[int, dict[int, float]], dict[int, float]]:
    """Segment side at ``A sqrt(f)`` (drawn first), then global side at ``A sqrt(1 - f)``.

    A side whose budget is exactly zero is skipped without consuming random numbers.
    """
    target = _budget(amplitude_nm, "the combined draw amplitude")
    if target == 0.0:
        return {}, {}
    fraction = prior.segment_variance_fraction
    segment_budget = target * np.sqrt(fraction)
    global_budget = target * np.sqrt(1.0 - fraction)
    segment_side = draw_segment_orthonormal(rng, segments, prior, segment_budget) if segment_budget != 0.0 else {}
    global_side = draw_global_orthonormal(rng, prior, global_budget) if global_budget != 0.0 else {}
    return segment_side, global_side
