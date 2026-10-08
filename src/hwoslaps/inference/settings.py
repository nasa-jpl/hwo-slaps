"""Settings of one nonlinear case: fit boxes and mask, sampler and refinement.

Every settings class is a frozen value checked when it is built, so a value outside its
domain fails at construction with a ``ConfigError`` naming the field. ``from_mapping`` reads a
mapping strictly: an unknown or misspelled key is an error at its dotted path, and an absent
key takes the documented default. ``to_mapping`` writes the record a result stores;
``from_mapping`` reads it back to an equal value (a custom ``PixelMask`` is recorded by digest
only and exists only through the Python API).
"""

from __future__ import annotations

import base64
import binascii
import dataclasses
import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

import numpy as np

from ..config.checks import (
    REQUIRED, ConfigError, Integer, Key, Nullable, Real, Rule, Table, Text, dataclass_table,
)
from ..identity import array_digest

if TYPE_CHECKING:
    from ..scene.profiles import Interval

__all__ = [
    "FIT_TABLE", "SAMPLER_TABLE", "REFINE_TABLE", "DEFAULT_BOX_RULES", "FIT_MODES", "MASK_NAMES",
    "OBJECTIVE_VERSION", "PROCEDURE_VERSION", "BoxRule",
    "FitSpec", "MassSupport", "PixelMask", "PriorWidths", "RefineSettings", "SamplerSettings",
]

OBJECTIVE_VERSION = "consistent_sampling_v2"
"""Dataset construction of a fit: e-/s data, model-kernel copy, generation over-sampling."""

PROCEDURE_VERSION = "normalized_lbfgsb_v2"
"""Refinement procedure: multistart L-BFGS-B on the unit box with the six acceptance gates."""

FIT_MODES = ("fixed_template", "local_search", "freed")
MASK_NAMES = ("all_pixels_minus_psf_border", "forecast_mask_minus_psf_border")
H1_STRATEGIES = ("search", "truth_anchor")
ROLES = ("smooth", "subhalo")


def _path(prefix: str, name: str) -> str:
    return f"{prefix}.{name}" if prefix else name


def _fields(value: Any) -> dict[str, Any]:
    return {item.name: getattr(value, item.name) for item in dataclasses.fields(value)}


def _record(value: Any) -> dict[str, Any]:
    return {name: list(item) if isinstance(item, tuple) else item for name, item in _fields(value).items()}


def _default(cls: type, name: str) -> Any:
    return next(item.default for item in dataclasses.fields(cls) if item.name == name)


def _settings_table(cls: type, *, docs: Mapping[str, str], checks: Mapping[str, Any],
                    units: Mapping[str, str] | None = None, rules: tuple[Rule, ...] = ()) -> Table:
    """The key table of a settings dataclass, with the value domain of each named field."""
    base = dataclass_table(cls, docs=docs, units=units or {})
    unknown = set(checks) - {key.name for key in base.keys}
    if unknown:
        raise TypeError(f"{cls.__name__} has no fields {sorted(unknown)}")
    keys = tuple(dataclasses.replace(key, check=checks.get(key.name, key.check)) for key in base.keys)
    return Table(keys, rules=rules)


def _canonical(instance: Any, table: Table) -> None:
    """Check a frozen settings value against its table and store every field in read form."""
    for name, value in table.read(_fields(instance), "").items():
        object.__setattr__(instance, name, tuple(value) if isinstance(value, list) else value)


_POSITIVE = Real(min=0.0, min_open=True)
_NON_NEGATIVE = Real(min=0.0)
_COUNT = Integer(min=1)


# ------------------------------------------------------------------ mass support


def _check_mass_order(values: Mapping[str, Any], path: str) -> None:
    if not values["log10_mass_min"] < values["log10_mass_max"]:
        raise ConfigError(_path(path, "log10_mass_max"),
                          f"must exceed log10_mass_min ({values['log10_mass_min']!r}), "
                          f"got {values['log10_mass_max']!r}")


@dataclass(frozen=True)
class MassSupport:
    """The closed log-mass interval a freed subhalo mass may take."""

    log10_mass_min: float
    log10_mass_max: float

    def __post_init__(self) -> None:
        _canonical(self, _MASS_SUPPORT)

    def contains(self, log10_mass: float) -> bool:
        return self.log10_mass_min <= float(log10_mass) <= self.log10_mass_max

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, Any], *, path: str = "fit.mass_support") -> MassSupport:
        return cls(**_MASS_SUPPORT.read(mapping, path))

    def to_mapping(self) -> dict[str, float]:
        return _record(self)


_MASS_SUPPORT = _settings_table(
    MassSupport,
    docs={"log10_mass_min": "lower end of the freed mass prior, log10(M200 / Msun) (point mass: log10(M / Msun))",
          "log10_mass_max": "upper end of the freed mass prior; must exceed log10_mass_min"},
    checks={}, units={"log10_mass_min": "dex", "log10_mass_max": "dex"},
    rules=(Rule("log10_mass_min < log10_mass_max", _check_mass_order),),
)


# ------------------------------------------------------------------ box rules


def _check_clip(values: Mapping[str, Any], path: str) -> None:
    clip = values["clip"]
    if clip is not None and not clip[0] < clip[1]:
        raise ConfigError(_path(path, "clip"), f"must be an interval with lower < upper, got {clip!r}")


@dataclass(frozen=True)
class BoxRule:
    """How the uniform prior box of one fitted parameter is built around its truth value."""

    half_width: float
    fractional: bool = False
    clip: tuple[float, float] | None = None

    def __post_init__(self) -> None:
        _canonical(self, _BOX_RULE)

    def box(self, truth: float, domain: Interval) -> tuple[float, float]:
        """``(lower, upper)`` around ``truth``, clipped into ``clip`` and into the parameter domain.

        ``clip`` is an open interval: its ends enter as the adjacent floats inward. A domain
        end enters as itself when closed, as its inward neighbour when open, and not at all
        when infinite.
        """
        value = float(truth)
        if self.fractional:
            if value == 0.0:
                raise ConfigError("", f"a fractional box rule ({self.half_width!r}) has zero width at truth 0.0; "
                                      "use an absolute rule")
            half = abs(value) * float(self.half_width)
            lower, upper = value - half, value + half
        else:
            lower, upper = value - float(self.half_width), value + float(self.half_width)
        if self.clip is not None:
            lower = max(lower, float(np.nextafter(self.clip[0], self.clip[1])))
            upper = min(upper, float(np.nextafter(self.clip[1], self.clip[0])))
        if math.isfinite(domain.lower):
            lower = max(lower, float(np.nextafter(domain.lower, math.inf)) if domain.open_lower
                        else float(domain.lower))
        if math.isfinite(domain.upper):
            upper = min(upper, float(np.nextafter(domain.upper, -math.inf)) if domain.open_upper
                        else float(domain.upper))
        if not (lower <= value <= upper and lower < upper):
            raise ConfigError("", f"truth {value!r} does not lie inside the box [{lower!r}, {upper!r}] of rule "
                                  f"{self.to_mapping()} within the domain ({domain.lower!r}, {domain.upper!r})")
        return lower, upper

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, Any], *, path: str = "rule") -> BoxRule:
        return cls(**_BOX_RULE.read(mapping, path))

    def to_mapping(self) -> dict[str, Any]:
        return _record(self)


_BOX_RULE = _settings_table(
    BoxRule,
    docs={"half_width": "half width of the box: absolute, or a fraction of |truth| when fractional",
          "fractional": "the half width is a fraction of |truth|",
          "clip": "open interval (lower, upper) the box is clipped into"},
    checks={"half_width": _POSITIVE},
    rules=(Rule("clip lower < clip upper", _check_clip),),
)

DEFAULT_BOX_RULES: tuple[tuple[str, BoxRule], ...] = (
    ("lens.amplitude", BoxRule(0.5, fractional=True)),
    ("lens.einstein_radius", BoxRule(0.01)),
    ("lens.ellipticity", BoxRule(0.02, clip=(-0.9, 0.9))),
    ("lens.multipole", BoxRule(0.01)),
    ("lens.orientation", BoxRule(5.0)),
    ("lens.position", BoxRule(0.005)),
    ("lens.sersic_index", BoxRule(0.3, fractional=True)),
    ("lens.shear", BoxRule(0.01)),
    ("lens.size", BoxRule(0.3, fractional=True)),
    ("lens.slope", BoxRule(0.05)),
    ("source.amplitude", BoxRule(0.5, fractional=True)),
    ("source.ellipticity", BoxRule(0.05, clip=(-0.9, 0.9))),
    ("source.orientation", BoxRule(5.0)),
    ("source.position", BoxRule(0.01)),
    ("source.sersic_index", BoxRule(0.3, fractional=True)),
    ("source.size", BoxRule(0.3, fractional=True)),
)
"""Default box rules keyed ``<galaxy>.<parameter kind>``, sorted by key.

The widths are the defaults of the RASTI code. Lens-light rows reuse the source widths, and the
orientation rows (degrees, for an Image ``rotation_deg``) give the orientation freedom of the
analytic source ellipticity box in the paper's first test scene. The nonlinear runs reported in
the paper used wider boxes; the nonlinear-fits guide shows how to set them."""

_RULE_KEYS = tuple(name for name, _ in DEFAULT_BOX_RULES)


@dataclass(frozen=True)
class PriorWidths:
    """Box rules per parameter kind and the subhalo centre windows."""

    rules: tuple[tuple[str, BoxRule], ...] = DEFAULT_BOX_RULES
    subhalo_local_window_arcsec: float = 0.03
    subhalo_freed_window_arcsec: float = 0.15

    def __post_init__(self) -> None:
        rules = tuple(self.rules)
        names = [entry[0] for entry in rules if isinstance(entry, tuple) and len(entry) == 2]
        if len(names) != len(rules) or not all(isinstance(rule, BoxRule) for _, rule in rules):
            raise ConfigError("rules", "must be (key, BoxRule) pairs")
        if names != sorted(set(names)):
            raise ConfigError("rules", f"keys must be distinct and sorted, got {names}")
        unknown = sorted(set(names) - set(_RULE_KEYS))
        if unknown:
            raise ConfigError("rules", f"unknown keys {unknown}; known: {', '.join(_RULE_KEYS)}")
        object.__setattr__(self, "rules", rules)
        for name in ("subhalo_local_window_arcsec", "subhalo_freed_window_arcsec"):
            object.__setattr__(self, name, _POSITIVE(getattr(self, name), name))

    def rule(self, galaxy: Literal["lens", "source"], kind: str) -> BoxRule:
        key = f"{galaxy}.{kind}"
        for name, rule in self.rules:
            if name == key:
                return rule
        raise KeyError(f"no box rule for {key!r}; rules: {', '.join(name for name, _ in self.rules)}")

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, Any], *, path: str = "fit.prior_widths") -> PriorWidths:
        values = _PRIOR_WIDTHS.read(mapping, path)
        return cls(rules=tuple((name, BoxRule(**values["rules"][name])) for name in _RULE_KEYS),
                   subhalo_local_window_arcsec=values["subhalo_local_window_arcsec"],
                   subhalo_freed_window_arcsec=values["subhalo_freed_window_arcsec"])

    def to_mapping(self) -> dict[str, Any]:
        return {"rules": {name: rule.to_mapping() for name, rule in self.rules},
                "subhalo_local_window_arcsec": self.subhalo_local_window_arcsec,
                "subhalo_freed_window_arcsec": self.subhalo_freed_window_arcsec}


def _rule_override(name: str, rule: BoxRule) -> Key:
    """A rules entry: each field given overrides that field of the default rule."""
    keys = tuple(dataclasses.replace(key, default=getattr(rule, key.name)) for key in _BOX_RULE.keys)
    return Key(name, Table(keys, rules=_BOX_RULE.rules), f"box rule of `{name}` parameters", {})


_PRIOR_WIDTHS = Table((
    Key("rules", Table(tuple(_rule_override(name, rule) for name, rule in DEFAULT_BOX_RULES)),
        "box rules keyed `<galaxy>.<parameter kind>`", {}),
    Key("subhalo_local_window_arcsec", _POSITIVE, "half width of the subhalo centre box in local_search",
        _default(PriorWidths, "subhalo_local_window_arcsec"), "arcsec"),
    Key("subhalo_freed_window_arcsec", _POSITIVE, "half width of the subhalo centre box in freed",
        _default(PriorWidths, "subhalo_freed_window_arcsec"), "arcsec"),
))


# ------------------------------------------------------------------ fit specification


@dataclass(frozen=True, eq=False)
class PixelMask:
    """A fitted-pixel mask given through Python (True = fitted); equal by digest."""

    values: np.ndarray

    def __post_init__(self) -> None:
        values = np.array(self.values, copy=True)
        if values.dtype != np.bool_ or values.ndim != 2:
            raise ValueError(f"a PixelMask is a 2-D boolean array, got dtype {values.dtype} and shape {values.shape}")
        values.setflags(write=False)
        object.__setattr__(self, "values", values)

    @property
    def digest(self) -> str:
        return array_digest(self.values)

    def to_record(self) -> dict[str, Any]:
        packed = np.packbits(self.values.reshape(-1), bitorder="little")
        return {"name": "custom_minus_psf_border", "shape": list(self.values.shape),
                "encoding": "packbits-little-base64", "values": base64.b64encode(packed.tobytes()).decode("ascii"),
                "digest": self.digest}

    @classmethod
    def from_record(cls, mapping: Mapping[str, Any], *, path: str = "fit.mask") -> PixelMask:
        keys = {"name", "shape", "encoding", "values", "digest"}
        if not isinstance(mapping, Mapping) or set(mapping) != keys:
            raise ConfigError(path, f"mask record keys must be {sorted(keys)}")
        if mapping["name"] != "custom_minus_psf_border":
            raise ConfigError(_path(path, "name"), "must be custom_minus_psf_border")
        if mapping["encoding"] != "packbits-little-base64":
            raise ConfigError(_path(path, "encoding"), "must be packbits-little-base64")
        shape = mapping["shape"]
        if not isinstance(shape, (tuple, list)) or len(shape) != 2 or any(
                isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 0 for value in shape):
            raise ConfigError(_path(path, "shape"), "must be two nonnegative integer dimensions")
        if not isinstance(mapping["values"], str):
            raise ConfigError(_path(path, "values"), "must be base64 text")
        try:
            packed = base64.b64decode(mapping["values"], validate=True)
        except (ValueError, binascii.Error) as error:
            raise ConfigError(_path(path, "values"), "invalid base64 mask") from error
        count = math.prod(shape)
        if len(packed) != (count + 7) // 8:
            raise ConfigError(_path(path, "values"), "byte length differs from the recorded mask shape")
        bits = np.unpackbits(np.frombuffer(packed, dtype=np.uint8), bitorder="little")
        if np.any(bits[count:]):
            raise ConfigError(_path(path, "values"), "mask padding bits must be zero")
        mask = cls(bits[:count].astype(bool).reshape(tuple(shape)))
        if mask.digest != mapping["digest"]:
            raise ConfigError(_path(path, "digest"), "mask digest differs from the decoded raw boolean values")
        return mask

    def __eq__(self, other: object) -> bool:
        return isinstance(other, PixelMask) and self.digest == other.digest

    def __hash__(self) -> int:
        return hash(self.digest)


def _check_mass_support(values: Mapping[str, Any], path: str) -> None:
    freed = values["mode"] == "freed"
    if freed and values["mass_support"] is None:
        raise ConfigError(_path(path, "mass_support"), "required with mode freed")
    if not freed and values["mass_support"] is not None:
        raise ConfigError(_path(path, "mass_support"), f"applies to mode freed only, not {values['mode']}")


@dataclass(frozen=True)
class FitSpec:
    """What a case fits: the subhalo mode, the fitted pixels, the boxes and the H1 strategy."""

    mode: Literal["fixed_template", "local_search", "freed"]
    mask: Literal["all_pixels_minus_psf_border", "forecast_mask_minus_psf_border"] | PixelMask = \
        "all_pixels_minus_psf_border"
    prior_widths: PriorWidths = PriorWidths()
    mass_support: MassSupport | None = None
    h1: Literal["search", "truth_anchor"] = "search"
    anchor_chi2_tolerance: float = 1.0e-8

    def __post_init__(self) -> None:
        for name in ("mode", "h1", "anchor_chi2_tolerance"):
            object.__setattr__(self, name, _FIT_CHECKS[name](getattr(self, name), name))
        if not isinstance(self.mask, PixelMask):
            _FIT_CHECKS["mask"](self.mask, "mask")
        if not isinstance(self.prior_widths, PriorWidths):
            raise ConfigError("prior_widths", f"must be a PriorWidths, got {self.prior_widths!r}")
        if self.mass_support is not None and not isinstance(self.mass_support, MassSupport):
            raise ConfigError("mass_support", f"must be a MassSupport or None, got {self.mass_support!r}")
        _check_mass_support({"mode": self.mode, "mass_support": self.mass_support}, "")

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, Any], *, path: str = "fit") -> FitSpec:
        return cls._from_values(FIT_TABLE.read(mapping, path), path)

    @classmethod
    def _from_values(cls, values: Mapping[str, Any], path: str) -> FitSpec:
        support = values["mass_support"]
        return cls(mode=values["mode"], mask=values["mask"],
                   prior_widths=PriorWidths.from_mapping(values["prior_widths"], path=_path(path, "prior_widths")),
                   mass_support=None if support is None else MassSupport(**support),
                   h1=values["h1"], anchor_chi2_tolerance=values["anchor_chi2_tolerance"])

    def to_mapping(self) -> dict[str, Any]:
        mask = ({"name": "custom_minus_psf_border", "digest": self.mask.digest}
                if isinstance(self.mask, PixelMask) else self.mask)
        return {"mode": self.mode, "mask": mask, "prior_widths": self.prior_widths.to_mapping(),
                "mass_support": None if self.mass_support is None else self.mass_support.to_mapping(),
                "h1": self.h1, "anchor_chi2_tolerance": self.anchor_chi2_tolerance}


    def to_record(self) -> dict[str, Any]:
        record = self.to_mapping()
        if isinstance(self.mask, PixelMask):
            record["mask"] = self.mask.to_record()
        return record

    @classmethod
    def from_record(cls, mapping: Mapping[str, Any], *, path: str = "fit") -> FitSpec:
        expected = {key.name for key in _FIT_RECORD_SPEC.keys}
        if not isinstance(mapping, Mapping) or set(mapping) != expected:
            raise ConfigError(path, f"fit record keys must be {sorted(expected)}")
        return cls._from_values(_FIT_RECORD_SPEC.read(mapping, path), path)


FIT_TABLE = Table((
    Key("mode", Text(choices=FIT_MODES), "subhalo model of H1: fixed at the hypothesis, centre free, "
        "or centre and mass free", REQUIRED),
    Key("mask", Text(choices=MASK_NAMES), "fitted pixels before the PSF border is removed: all pixels, or "
        "the forecast mask", _default(FitSpec, "mask")),
    Key("prior_widths", _PRIOR_WIDTHS, "prior boxes of the fitted parameters", {}),
    Key("mass_support", Nullable(_MASS_SUPPORT), "freed mass prior; required with mode freed",
        _default(FitSpec, "mass_support")),
    Key("h1", Text(choices=H1_STRATEGIES), "H1 by a sampler search, or by the truth vector on expected data",
        _default(FitSpec, "h1")),
    Key("anchor_chi2_tolerance", _POSITIVE, "largest chi-square at which the truth vector is the H1 maximum",
        _default(FitSpec, "anchor_chi2_tolerance")),
), rules=(Rule("mass_support is required with mode freed and not allowed otherwise", _check_mass_support),))

_FIT_CHECKS = {key.name: key.check for key in FIT_TABLE.keys}


def _record_mask(value: Any, path: str) -> str | PixelMask:
    if isinstance(value, Mapping):
        return PixelMask.from_record(value, path=path)
    return Text(choices=MASK_NAMES)(value, path)


_FIT_RECORD_SPEC = dataclasses.replace(FIT_TABLE, keys=tuple(
    dataclasses.replace(key, check=_record_mask) if key.name == "mask" else key for key in FIT_TABLE.keys))


# ------------------------------------------------------------------ sampler


def _check_jax_cores(values: Mapping[str, Any], path: str) -> None:
    if values["use_jax"] and values["number_of_cores"] != 1:
        raise ConfigError(_path(path, "number_of_cores"),
                          f"must be 1 with use_jax (the JAX likelihood vectorizes batches), "
                          f"got {values['number_of_cores']}")


@dataclass(frozen=True)
class SamplerSettings:
    """Nautilus settings of the role searches. The sampler seed is a separate argument."""

    n_live_smooth: int = 100
    n_live_subhalo_fixed: int = 100
    n_live_subhalo_search: int = 200
    n_eff: float | None = None
    n_shell: int | None = None
    f_live: float | None = None
    discard_exploration: bool | None = None
    n_like_max: int | None = None
    number_of_cores: int = 1
    use_jax: bool = False
    jax_n_batch: int = 100
    retain_search_internal: bool = False

    def __post_init__(self) -> None:
        _canonical(self, SAMPLER_TABLE)

    def n_live(self, role: Literal["smooth", "subhalo"], mode: str) -> int:
        """Live points of a role search: H0, then H1 fixed at the hypothesis or searching."""
        if role not in ROLES:
            raise ValueError(f"role must be one of {ROLES}, got {role!r}")
        if mode not in FIT_MODES:
            raise ValueError(f"mode must be one of {FIT_MODES}, got {mode!r}")
        if role == "smooth":
            return self.n_live_smooth
        return self.n_live_subhalo_fixed if mode == "fixed_template" else self.n_live_subhalo_search

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, Any], *, path: str = "sampler") -> SamplerSettings:
        return cls(**SAMPLER_TABLE.read(mapping, path))

    def to_mapping(self) -> dict[str, Any]:
        return _record(self)


SAMPLER_TABLE = _settings_table(
    SamplerSettings,
    docs={"n_live_smooth": "live points of the H0 search",
          "n_live_subhalo_fixed": "live points of the H1 search in mode fixed_template",
          "n_live_subhalo_search": "live points of the H1 search in modes local_search and freed",
          "n_eff": "effective sample size at which Nautilus stops; null: the backend default",
          "n_shell": "minimum points per shell before Nautilus stops; null: the backend default",
          "f_live": "live-set evidence fraction at which exploration ends; null: the backend default",
          "discard_exploration": "drop exploration-phase points from the posterior; null: the backend default",
          "n_like_max": "largest number of likelihood calls; null: no limit",
          "number_of_cores": "sampler processes; 1 with use_jax",
          "use_jax": "JAX likelihood, vectorized over batches of vectors",
          "jax_n_batch": "vectors per JAX likelihood batch",
          "retain_search_internal": "keep the raw Nautilus state on disk after the fit"},
    checks={"n_live_smooth": _COUNT, "n_live_subhalo_fixed": _COUNT, "n_live_subhalo_search": _COUNT,
            "n_eff": Nullable(_POSITIVE), "n_shell": Nullable(_COUNT),
            "f_live": Nullable(Real(min=0.0, max=1.0, min_open=True)), "n_like_max": Nullable(_COUNT),
            "number_of_cores": _COUNT, "jax_n_batch": _COUNT},
    rules=(Rule("use_jax requires number_of_cores 1", _check_jax_cores),),
)


# ------------------------------------------------------------------ refinement


@dataclass(frozen=True)
class RefineSettings:
    """Multistart L-BFGS-B refinement of a role maximum and its acceptance gates.

    The defaults are the refinement settings of the RASTI paper.
    """

    original_start_count: int = 8
    start_separation_normalized_l2: float = 0.05
    start_separation_posterior_sigma: float = 1.0
    maxiter: int = 500
    ftol: float = 0.0
    gtol: float = 1.0e-10
    maxls: int = 50
    repeat_maxiter: int = 1000
    repeat_ftol: float = 0.0
    repeat_gtol: float = 1.0e-12
    support_log_likelihood_tolerance: float = 0.1
    repeat_log_likelihood_tolerance: float = 0.1
    minimum_distinct_original_start_support: int = 2
    scalar_residual_tolerance: float = 1.0e-4

    def __post_init__(self) -> None:
        _canonical(self, REFINE_TABLE)

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, Any], *, path: str = "refine") -> RefineSettings:
        return cls(**REFINE_TABLE.read(mapping, path))

    def to_mapping(self) -> dict[str, Any]:
        return _record(self)


REFINE_TABLE = _settings_table(
    RefineSettings,
    docs={"original_start_count": "posterior samples started from, besides the sampler maximum",
          "start_separation_normalized_l2": "a sample is a new start when this far (unit-box L2) from every start",
          "start_separation_posterior_sigma": "or when this many posterior sigmas from every start",
          "maxiter": "L-BFGS-B iterations per start",
          "ftol": "L-BFGS-B relative reduction tolerance per start",
          "gtol": "L-BFGS-B projected-gradient tolerance per start",
          "maxls": "line-search steps per iteration",
          "repeat_maxiter": "iterations of the tighter repeat from the best point",
          "repeat_ftol": "relative reduction tolerance of the tighter repeat",
          "repeat_gtol": "projected-gradient tolerance of the tighter repeat",
          "support_log_likelihood_tolerance": "a start supports the best when it ends within this log L",
          "repeat_log_likelihood_tolerance": "the tighter repeat may move the best by at most this log L",
          "minimum_distinct_original_start_support": "supporting original starts needed for acceptance",
          "scalar_residual_tolerance": "largest residual and direct log L inconsistency at the best"},
    checks={"original_start_count": _COUNT, "start_separation_normalized_l2": _POSITIVE,
            "start_separation_posterior_sigma": _POSITIVE, "maxiter": _COUNT, "ftol": _NON_NEGATIVE,
            "gtol": _NON_NEGATIVE, "maxls": _COUNT, "repeat_maxiter": _COUNT, "repeat_ftol": _NON_NEGATIVE,
            "repeat_gtol": _NON_NEGATIVE, "support_log_likelihood_tolerance": _NON_NEGATIVE,
            "repeat_log_likelihood_tolerance": _NON_NEGATIVE, "minimum_distinct_original_start_support": _COUNT,
            "scalar_residual_tolerance": _POSITIVE},
)
