"""Forecast inputs: position layouts, masks and the parameters profiled as nuisances.

Key tables own defaults and composition. Specs hold the read values; the effective
configuration record remains in ``EngineConfig``. All checks here are backend-free.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar, Literal, get_args

from ..config.checks import (Boolean, ConfigError, FilePath, Integer, Key, ListOf, MapOf, Nullable,
                             Pair, Real, Rule, Table, Text, Union, Variants)
from ..optics.pupils import parse_pupil
from ..optics.wavefront import (WAVEFRONT_SELECTION_TABLE, WavefrontSelection,
                               parse_wavefront_selection, select_modes)
from ..scene.image_source import frozen_value
from ..scene.parameters import match_parameters, scene_parameter_names
from ..scene.profiles import ParameterKind
from ..scene.spec import SceneSpec, parse_scene

__all__ = [
    "AllPixelsMaskSpec", "AnnulusMaskSpec", "CROSS_RULES", "ExplicitPositionsSpec", "FORECAST_TABLE",
    "ForecastSpec", "GridPositionsSpec", "MaskSpec", "NuisanceSpec", "PositionsSpec", "PsfBorderMaskSpec",
    "RingPositionsSpec", "SourceSnrMaskSpec", "WavefrontNuisanceSpec", "check_nuisance_spec", "parse_forecast",
]

Radius = Literal["einstein_radius", "critical_curve"] | float


@dataclass(frozen=True)
class GridPositionsSpec:
    spacing_arcsec: float
    half_width_arcsec: float
    annulus: tuple[float, float] | None
    kind: ClassVar[str] = "grid"


@dataclass(frozen=True)
class RingPositionsSpec:
    count: int
    radius: Radius
    offset_arcsec: float
    kind: ClassVar[str] = "ring"


@dataclass(frozen=True)
class ExplicitPositionsSpec:
    positions_yx: tuple[tuple[float, float], ...]
    kind: ClassVar[str] = "explicit"


PositionsSpec = GridPositionsSpec | RingPositionsSpec | ExplicitPositionsSpec


@dataclass(frozen=True)
class AllPixelsMaskSpec:
    kind: ClassVar[str] = "all_pixels"


@dataclass(frozen=True)
class SourceSnrMaskSpec:
    snr_min: float
    kind: ClassVar[str] = "source_snr"


@dataclass(frozen=True)
class AnnulusMaskSpec:
    inner_arcsec: float
    outer_arcsec: float
    about: Literal["lens", "grid"]
    kind: ClassVar[str] = "annulus"


@dataclass(frozen=True)
class PsfBorderMaskSpec:
    kind: ClassVar[str] = "psf_border"


MaskSpec = AllPixelsMaskSpec | SourceSnrMaskSpec | AnnulusMaskSpec | PsfBorderMaskSpec


@dataclass(frozen=True)
class WavefrontNuisanceSpec:
    modes: WavefrontSelection
    step_nm: float | Mapping[str, float]
    prior_sigma_nm: float | Mapping[str, float] | None


@dataclass(frozen=True)
class NuisanceSpec:
    fixed: tuple[str, ...]
    steps: Mapping[str, float]
    priors: Mapping[str, float]
    background_offset: bool
    wavefront: WavefrontNuisanceSpec | None


@dataclass(frozen=True)
class ForecastSpec:
    positions: PositionsSpec
    mask: MaskSpec
    nuisances: NuisanceSpec
    noise_covariance: Path | None

    @classmethod
    def from_values(cls, values: Mapping[str, Any]) -> ForecastSpec:
        """Build from the values read by ``FORECAST_TABLE``."""
        positions = values["positions"]
        if positions["kind"] == "grid":
            annulus = positions["annulus"]
            layout: PositionsSpec = GridPositionsSpec(
                positions["spacing_arcsec"], positions["half_width_arcsec"],
                None if annulus is None else (annulus["inner_arcsec"], annulus["outer_arcsec"]))
        elif positions["kind"] == "ring":
            layout = RingPositionsSpec(positions["count"], positions["radius"], positions["offset_arcsec"])
        else:
            layout = ExplicitPositionsSpec(tuple(tuple(row) for row in positions["positions_yx"]))
        mask = values["mask"]
        if mask["kind"] == "all_pixels":
            mask_spec: MaskSpec = AllPixelsMaskSpec()
        elif mask["kind"] == "source_snr":
            mask_spec = SourceSnrMaskSpec(mask["snr_min"])
        elif mask["kind"] == "annulus":
            mask_spec = AnnulusMaskSpec(mask["inner_arcsec"], mask["outer_arcsec"], mask["about"])
        else:
            mask_spec = PsfBorderMaskSpec()
        nuisance = values["nuisances"]
        wavefront = nuisance["wavefront"]
        wavefront_spec = None if wavefront is None else WavefrontNuisanceSpec(
            parse_wavefront_selection(wavefront["modes"], "forecast.nuisances.wavefront.modes"),
            frozen_value(wavefront["step_nm"]), frozen_value(wavefront["prior_sigma_nm"]))
        return cls(layout, mask_spec, NuisanceSpec(
            tuple(nuisance["fixed"]), frozen_value(nuisance["steps"]), frozen_value(nuisance["priors"]),
            nuisance["background_offset"], wavefront_spec),
            None if values["noise_covariance"] is None else Path(values["noise_covariance"]))


def _annulus(values: Mapping[str, Any], path: str) -> None:
    if values["inner_arcsec"] >= values["outer_arcsec"]:
        raise ConfigError(path, "inner_arcsec must be smaller than outer_arcsec")


def _grid(values: Mapping[str, Any], path: str) -> None:
    if values["half_width_arcsec"] < values["spacing_arcsec"]:
        raise ConfigError(f"{path}.half_width_arcsec", "must be at least spacing_arcsec")


def _ring(values: Mapping[str, Any], path: str) -> None:
    if isinstance(values["radius"], float) and values["radius"] + values["offset_arcsec"] <= 0.0:
        raise ConfigError(f"{path}.offset_arcsec", "radius + offset_arcsec must be positive")


def _explicit(values: Mapping[str, Any], path: str) -> None:
    rows = values["positions_yx"]
    for index, row in enumerate(rows):
        if row in rows[:index]:
            raise ConfigError(f"{path}.positions_yx[{index}]", "positions must not repeat a row")


def _wavefront_scales(values: Mapping[str, Any], path: str) -> None:
    families = {name for name, selection in values["modes"].items() if selection is not None}
    for name in ("step_nm", "prior_sigma_nm"):
        scale = values[name]
        if isinstance(scale, Mapping):
            missing = families - set(scale)
            if missing:
                raise ConfigError(f"{path}.{name}", "missing selected families: " + ", ".join(sorted(missing)))


_POSITIVE = Real(min=0.0, min_open=True)
_INNER = Key("inner_arcsec", Real(min=0.0), "inner radius of the closed annulus", unit="arcsec")
_OUTER = Key("outer_arcsec", _POSITIVE, "outer radius of the closed annulus", unit="arcsec")
_ANNULUS_TABLE = Table((_INNER, _OUTER), rules=(Rule("inner_arcsec < outer_arcsec", _annulus),))
_POSITIONS_TABLE = Variants("kind", {
    "grid": Table((
        Key("spacing_arcsec", _POSITIVE, "lattice spacing", unit="arcsec"),
        Key("half_width_arcsec", _POSITIVE, "half width of the square lattice", unit="arcsec"),
        Key("annulus", Nullable(_ANNULUS_TABLE), "retain nodes in this closed annulus", None),
    ), rules=(Rule("half_width_arcsec >= spacing_arcsec", _grid),)),
    "ring": Table((
        Key("count", Integer(min=1),
            "number of equally spaced positions"),
        Key("radius", Union(Text(choices=("einstein_radius", "critical_curve")), _POSITIVE),
            "radius about the lens centre", "einstein_radius", unit="arcsec"),
        Key("offset_arcsec", Real(), "offset added to the ring radius", 0.0, unit="arcsec"),
    ), rules=(Rule("a numeric radius plus offset is positive", _ring),)),
    "explicit": Table((Key("positions_yx", ListOf(Pair(Real()), min_length=1),
                           "positions in (y, x) order", unit="arcsec"),),
                      rules=(Rule("no duplicate position rows", _explicit),)),
})
_MASK_TABLE = Variants("kind", {
    "all_pixels": Table(()),
    "source_snr": Table((Key("snr_min", _POSITIVE, "minimum source-plane light signal-to-noise"),)),
    "annulus": _ANNULUS_TABLE.extend((Key("about", Text(choices=("lens", "grid")),
                                         "centre of the annulus", "lens"),)),
    "psf_border": Table(()),
})
_FAMILY_SCALE = Union(_POSITIVE, MapOf(Text(choices=("segment_hexikes", "zernikes")), _POSITIVE))
_WAVEFRONT_NUISANCE_TABLE = Table((
    Key("modes", WAVEFRONT_SELECTION_TABLE, "wavefront families and modes to profile"),
    Key("step_nm", _FAMILY_SCALE, "central-difference step, scalar or per family", 1.0, unit="nm"),
    Key("prior_sigma_nm", Nullable(_FAMILY_SCALE), "Gaussian sigma, scalar or per family", None, unit="nm"),
), rules=(Rule("family scales cover every selected family", _wavefront_scales),))
_NUISANCE_TABLE = Table((
    Key("fixed", ListOf(Text(), unique=True), "scene parameter names or fnmatch patterns held fixed", []),
    Key("steps", MapOf(Text(), _POSITIVE), "finite-difference steps per kind or scene parameter name", {}),
    Key("priors", MapOf(Text(), _POSITIVE), "Gaussian sigmas per scene parameter name", {}),
    Key("background_offset", Boolean(), "profile a constant ADU offset", True),
    Key("wavefront", Nullable(_WAVEFRONT_NUISANCE_TABLE), "wavefront-mode nuisances", None),
))
FORECAST_TABLE = Table((
    Key("positions", _POSITIONS_TABLE, "where the subhalo hypothesis is evaluated"),
    Key("mask", _MASK_TABLE, "pixels used by the statistic"),
    Key("nuisances", _NUISANCE_TABLE, "parameters profiled in the likelihood", {}),
    Key("noise_covariance", Nullable(FilePath((".npy",))), "dense covariance over the full image", None),
), doc="Fisher forecast inputs; execution options belong to Execution.")


def parse_forecast(mapping: Mapping[str, Any], path: str = "forecast") -> ForecastSpec:
    """Read the forecast section strictly, filling its table defaults."""
    return ForecastSpec.from_values(FORECAST_TABLE.read(mapping, path))


def check_nuisance_spec(scene: SceneSpec, spec: NuisanceSpec, *, model_has_basis: bool) -> None:
    """Refuse unresolved parameter names and wavefront nuisances without a model basis."""
    names = scene_parameter_names(scene)
    match_parameters(names, spec.fixed, path="forecast.nuisances.fixed")
    kinds = get_args(ParameterKind)
    for key in spec.steps:
        if key not in kinds and key not in names:
            raise ConfigError(f"forecast.nuisances.steps.{key}",
                              "unknown parameter or kind; parameters: " + ", ".join(names))
    for key in spec.priors:
        if key not in names:
            raise ConfigError(f"forecast.nuisances.priors.{key}",
                              "unknown parameter; parameters: " + ", ".join(names))
    if spec.wavefront is not None and not model_has_basis:
        raise ConfigError("forecast.nuisances.wavefront", "the model PSF has no wavefront basis")


def _check_forecast(root: Mapping[str, Any], path: str) -> None:
    values = root["forecast"]
    if values is None:
        return
    scene = parse_scene(root["scene"])
    spec = ForecastSpec.from_values(values)
    psf = root["psf"]
    model_has_basis = psf["truth"]["kind"] == "optical" and psf["model"]["kind"] != "kernel"
    check_nuisance_spec(scene, spec.nuisances, model_has_basis=model_has_basis)
    if spec.nuisances.wavefront is not None:
        section = "model" if psf["model"]["kind"] == "optical" else "truth"
        pupil = parse_pupil(psf[section]["pupil"], f"psf.{section}.pupil")
        select_modes(spec.nuisances.wavefront.modes, pupil, "forecast.nuisances.wavefront.modes")
    if isinstance(spec.positions, RingPositionsSpec) and spec.positions.radius == "einstein_radius":
        if len(scene.einstein_radii()) != 1:
            raise ConfigError("forecast.positions.radius", "einstein_radius requires exactly one lens mass "
                              "component with an Einstein radius")


CROSS_RULES = (Rule("nuisance names resolve; wavefront modes need a model basis and existing segments; "
                    "an Einstein-radius ring needs exactly one radius", _check_forecast),)
