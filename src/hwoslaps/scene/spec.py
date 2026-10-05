"""Scene specification: grid, lens and source galaxies, the subhalo hypothesis, the injection, perturbers.

``parse_scene`` reads the ``scene`` section strictly through ``SCENE_TABLE`` (component keys
from the profile registry, halo keys from ``scene.halos``, the injection from
``scene.subhalo``, perturbers from ``scene.perturbers``) and checks the scene rules:
- the source lies behind the lens;
- component names are unique within a galaxy across its mass and light components and
  avoid ``cls``, ``id``, ``redshift``, ``subhalo``, the prefix ``perturber`` and the layout
  suffixes ``_multipole_m3``, ``_multipole_m4``;
- halo redshifts lie in (0, source redshift), and ``moline2017_eq7`` is used only at the
  lens redshift and inside its calibrated mass range;
- a placement with ``radius: einstein_radius`` has exactly one lens mass component with an
  Einstein radius.

Angles and positions are arcsec in (y, x) order on the image plane with the origin at the
grid centre; placements are about the lens centre.
"""

from __future__ import annotations

import math
import types
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np

from ..config.checks import ConfigError, Integer, Key, Named, Nullable, Real, Rule, Shape, Table, Variants
from ..spectra.bandpass import BANDPASS_TABLE, BandpassSpec, parse_bandpass
from ..spectra.sed import SED, SED_TABLE, SEDSpec, build_sed, parse_sed
from .halos import HaloModel, MOLINE2017_MASS_RANGE_MSUN, halo_model_from_values, halo_model_table
from .image_source import frozen_value
from .perturbers import PERTURBERS_TABLE, PerturberSpec, perturbers_from_values
from .profiles import PROFILE_TYPES
from .subhalo import INJECTION_TABLE, InjectionSpec, injection_from_values

__all__ = [
    "ComponentSpec", "GalaxySpec", "GridSpec", "LIGHT_COMPONENT_TABLE", "LightGroup", "MASS_COMPONENT_TABLE",
    "FLUX_TABLE", "FluxSpec", "SCENE_TABLE", "SceneSpec", "component_from_values", "parse_flux", "parse_scene",
    "pixel_centres_yx",
]

Plane = Literal["lens", "source"]
Role = Literal["mass", "light"]

# The instance attributes and constructor keywords of al.Galaxy and af.Model(al.Galaxy), which a
# class-level lookup cannot see, and the hypothesis attribute; build_scene refuses class attributes.
_RESERVED_NAMES = ("cls", "id", "redshift", "subhalo")
_RESERVED_PREFIX = "perturber"
_LAYOUT_SUFFIXES = ("_multipole_m3", "_multipole_m4")


@dataclass(frozen=True)
class GridSpec:
    shape: tuple[int, int]
    pixel_scale_arcsec: float
    over_sample_size: int


@dataclass(frozen=True)
class FluxSpec:
    rate_e_per_s: float | None
    ab_mag: float | None
    reference_band: BandpassSpec | None


def _reference_flux(values: Mapping[str, Any], path: str) -> None:
    if values["reference_band"] is not None and values["ab_mag"] is None:
        raise ConfigError(f"{path}.reference_band", "a reference band requires an AB magnitude")


FLUX_TABLE = Table((
    Key("rate_e_per_s", Nullable(Real(min=0.0, min_open=True)), "intrinsic unlensed detected rate", None, unit="e-/s"),
    Key("ab_mag", Nullable(Real()), "AB magnitude in the instrument or reference band", None),
    Key("reference_band", Nullable(BANDPASS_TABLE), "band in which the AB magnitude is specified", None),
), exactly_one=(("rate_e_per_s", "ab_mag"),), rules=(Rule("reference band requires AB magnitude", _reference_flux),))


def parse_flux(mapping: Mapping[str, Any], path: str) -> FluxSpec:
    values = FLUX_TABLE.read(mapping, path)
    reference = values["reference_band"]
    return FluxSpec(values["rate_e_per_s"], values["ab_mag"],
                    None if reference is None else parse_bandpass(reference, f"{path}.reference_band"))


@dataclass(frozen=True)
class ComponentSpec:
    """One named mass or light component with its keys as read by the registry entry's table.

    ``values`` holds the profile parameters, read-only at every depth. ``sed`` and ``flux``
    hold the parsed spectral shape and intrinsic photometric normalization of a light component.
    """

    name: str
    plane: Plane
    role: Role
    type: str
    values: Mapping[str, Any]
    sed: SEDSpec | None
    flux: FluxSpec | None


@dataclass(frozen=True)
class GalaxySpec:
    plane: Plane
    redshift: float
    mass: tuple[ComponentSpec, ...]
    light: tuple[ComponentSpec, ...]


@dataclass(frozen=True)
class LightGroup:
    """Light components rendered and convolved together: one plane, one SED (None without SEDs)."""

    plane: Plane
    sed: SEDSpec | None
    components: tuple[str, ...]


def _einstein_radii(components: Any) -> tuple[float, ...]:
    """Einstein-radius parameter values of mass components given as (type, values) pairs."""
    radii = []
    for type_name, values in components:
        for definition in PROFILE_TYPES[type_name].parameters(values):
            if definition.kind == "einstein_radius":
                radii.append(values[definition.key])
    return tuple(radii)


@dataclass(frozen=True)
class SceneSpec:
    grid: GridSpec
    lens: GalaxySpec
    source: GalaxySpec
    subhalo: HaloModel
    subhalo_redshift: float | None
    injection: InjectionSpec | None
    perturbers: PerturberSpec

    @property
    def lens_centre(self) -> tuple[float, float]:
        """Centre of the first lens mass component that has one: the frame of placements and layouts."""
        for component in self.lens.mass:
            if "centre" in component.values:
                return tuple(component.values["centre"])
        raise ValueError("no lens mass component has a centre")

    def einstein_radii(self) -> tuple[float, ...]:
        """The Einstein-radius parameters of the lens mass components, in declaration order."""
        return _einstein_radii((component.type, component.values) for component in self.lens.mass)

    def einstein_radius(self) -> float:
        """The Einstein radius of the single lens mass component that has one."""
        radii = self.einstein_radii()
        if len(radii) != 1:
            raise ValueError(f"the lens has {len(radii)} mass components with an Einstein radius; exactly one is "
                             "needed to place by radius: einstein_radius")
        return radii[0]

    def light_groups(self, *, loaded_seds: Mapping[str, SED] | None = None) -> Mapping[str, LightGroup]:
        """Group by plane and spectral identity, using the preparation's loaded SEDs when supplied."""
        groups = {}
        for galaxy in (self.lens, self.source):
            keys = {}
            for component in galaxy.light:
                if component.sed is None:
                    digest = None
                else:
                    name = f"{galaxy.plane}.{component.name}"
                    if loaded_seds is None:
                        sed = build_sed(component.sed, redshift=galaxy.redshift)
                    else:
                        if name not in loaded_seds:
                            raise ValueError(f"no loaded SED for light component {name}")
                        sed = loaded_seds[name]
                    digest = sed.digest()
                if digest not in keys:
                    key = galaxy.plane if digest is None else f"{galaxy.plane}:{component.name}"
                    keys[digest] = key
                    groups[key] = LightGroup(galaxy.plane, component.sed, (component.name,))
                else:
                    key = keys[digest]
                    group = groups[key]
                    groups[key] = LightGroup(group.plane, group.sed, group.components + (component.name,))
        return types.MappingProxyType(groups)


# ------------------------------------------------------------------ tables


def _light_reference(values: Mapping[str, Any], path: str) -> None:
    flux = values.get("flux")
    if flux is not None and flux["reference_band"] is not None and values.get("sed") is None:
        raise ConfigError(f"{path}.sed", "a reference-band magnitude requires an SED")


MASS_COMPONENT_TABLE = Variants(
    "type", {name: profile.table for name, profile in PROFILE_TYPES.items() if profile.role == "mass"},
    doc="A mass component; its parameters are profiled in registry order unless fixed.")
LIGHT_COMPONENT_TABLE = Variants(
    "type", {name: profile.table.extend((Key("flux", Nullable(FLUX_TABLE), "intrinsic photometric normalization", None),
                                          Key("sed", Nullable(SED_TABLE), "fixed spectral shape", None)),
                                         exactly_one=((profile.amplitude_key, "flux"),),
                                         rules=(Rule("reference-band flux requires an SED", _light_reference),))
             for name, profile in PROFILE_TYPES.items() if profile.role == "light"},
    doc="A light component; its parameters are profiled in registry order unless fixed.")

_GRID_TABLE = Table((
    Key("shape", Shape(), "(ny, nx)", unit="pixels"),
    Key("pixel_scale_arcsec", Real(min=0.0, min_open=True), "pixel side", unit="arcsec"),
    Key("over_sample_size", Integer(min=1), "sub-pixels per axis for light rendering (the paper used 4)"),
))
_LENS_TABLE = Table((
    Key("redshift", Real(min=0.0, min_open=True), "lens redshift"),
    Key("mass", Named(MASS_COMPONENT_TABLE, min_length=1), "mass components; this order is the nuisance order"),
    Key("light", Named(LIGHT_COMPONENT_TABLE), "lens-plane light components", {}),
))
_SOURCE_TABLE = Table((
    Key("redshift", Real(min=0.0, min_open=True), "source redshift, behind the lens"),
    Key("light", Named(LIGHT_COMPONENT_TABLE, min_length=1), "source light components"),
))
_SUBHALO_TABLE = halo_model_table((
    Key("redshift", Nullable(Real(min=0.0, min_open=True)),
        "redshift of the hypothesis; null is the lens redshift. Positions of an off-plane halo are angular "
        "positions in its own plane", None),
))


def _check_redshift_order(values: Mapping[str, Any], path: str) -> None:
    if not values["source"]["redshift"] > values["lens"]["redshift"]:
        raise ConfigError(f"{path}.source.redshift",
                          f"must exceed the lens redshift {values['lens']['redshift']}, got {values['source']['redshift']}")


def _check_component_names(values: Mapping[str, Any], path: str) -> None:
    for galaxy, roles in (("lens", ("mass", "light")), ("source", ("light",))):
        seen: dict[str, str] = {}
        for role in roles:
            for name in values[galaxy][role]:
                where = f"{path}.{galaxy}.{role}.{name}"
                if name in seen:
                    raise ConfigError(where, f"name already used by {galaxy}.{seen[name]}.{name}; component names "
                                             "are unique within a galaxy")
                if name in _RESERVED_NAMES or name.startswith(_RESERVED_PREFIX) or name.endswith(
                        _LAYOUT_SUFFIXES):
                    raise ConfigError(where, f"reserved name: component names avoid {', '.join(_RESERVED_NAMES)}, "
                                             f"the prefix {_RESERVED_PREFIX!r} and the suffixes "
                                             f"{', '.join(_LAYOUT_SUFFIXES)}")
                seen[name] = role


def _uses_moline(halo: Mapping[str, Any]) -> bool:
    relation = halo.get("concentration")
    return relation is not None and relation["kind"] == "moline2017_eq7"


def _check_moline_mass(halo: Mapping[str, Any], mass: float, where: str) -> None:
    low, high = MOLINE2017_MASS_RANGE_MSUN
    if _uses_moline(halo) and not low <= mass <= high:
        raise ConfigError(where, f"moline2017_eq7 is calibrated for M200 in [{low:g}, {high:g}] Msun, got {mass:g}")


def _check_halos(values: Mapping[str, Any], path: str) -> None:
    lens_redshift, source_redshift = values["lens"]["redshift"], values["source"]["redshift"]
    halos = [(f"{path}.subhalo", values["subhalo"])]
    halos += [(f"{path}.perturbers.halos[{index}]", halo) for index, halo in enumerate(values["perturbers"]["halos"])]
    for where, halo in halos:
        redshift = halo["redshift"]
        if redshift is not None and not redshift < source_redshift:
            raise ConfigError(f"{where}.redshift", f"must lie in front of the source at {source_redshift}, got {redshift}")
        if _uses_moline(halo) and redshift is not None and redshift != lens_redshift:
            raise ConfigError(f"{where}.redshift", "moline2017_eq7 describes subhalos at the lens redshift "
                                                   f"{lens_redshift}; use power_law or fixed off the lens plane")
        if "mass_msun" in halo:
            _check_moline_mass(halo, halo["mass_msun"], f"{where}.mass_msun")
    if values["injection"] is not None:
        _check_moline_mass(values["subhalo"], values["injection"]["mass_msun"], f"{path}.injection.mass_msun")


def _check_radius_reference(values: Mapping[str, Any], path: str) -> None:
    injection = values["injection"]
    if injection is None or injection["position"].get("radius") != "einstein_radius":
        return
    count = len(_einstein_radii((component["type"], component) for component in values["lens"]["mass"].values()))
    if count != 1:
        raise ConfigError(f"{path}.injection.position.radius",
                          f"einstein_radius needs exactly one lens mass component with an Einstein radius, found {count}")


SCENE_TABLE = Table(
    (
        Key("grid", _GRID_TABLE, "image grid"),
        Key("lens", _LENS_TABLE, "lens galaxy"),
        Key("source", _SOURCE_TABLE, "source galaxy"),
        Key("subhalo", _SUBHALO_TABLE, "the detection hypothesis: forecasts and fits test this halo family"),
        Key("injection", Nullable(INJECTION_TABLE), "the subhalo hwoslaps simulate and batch simulate jobs inject",
            None),
        Key("perturbers", PERTURBERS_TABLE, "fixed perturbing halos", {}),
    ),
    rules=(
        Rule("source.redshift > lens.redshift", _check_redshift_order),
        Rule("component names unique within a galaxy and not reserved", _check_component_names),
        Rule("halo redshifts in (0, source.redshift); moline2017_eq7 only at the lens redshift and in its mass range",
             _check_halos),
        Rule("radius: einstein_radius needs exactly one lens mass component with an Einstein radius",
             _check_radius_reference),
    ),
    doc="The lensing scene.",
)


def component_from_values(name: str, plane: Plane, role: Role, values: Mapping[str, Any]) -> ComponentSpec:
    """The ``ComponentSpec`` of one component's values as read by its role's table (``type`` included)."""
    rest = {key: value for key, value in values.items() if key not in ("type", "sed", "flux")}
    sed = values.get("sed")
    flux = values.get("flux")
    path = f"scene.{plane}.{role}.{name}"
    return ComponentSpec(name, plane, role, values["type"], frozen_value(rest),
                         None if sed is None else parse_sed(sed, f"{path}.sed"),
                         None if flux is None else parse_flux(flux, f"{path}.flux"))


def _galaxy(plane: Plane, values: Mapping[str, Any]) -> GalaxySpec:
    def components(role: Role) -> tuple[ComponentSpec, ...]:
        return tuple(component_from_values(name, plane, role, item) for name, item in values.get(role, {}).items())

    return GalaxySpec(plane=plane, redshift=values["redshift"], mass=components("mass"), light=components("light"))


def parse_scene(mapping: Mapping[str, Any], path: str = "scene") -> SceneSpec:
    """Read the ``scene`` section strictly (unknown keys, domains and the scene rules raise ``ConfigError``)."""
    values = SCENE_TABLE.read(mapping, path)
    grid = values["grid"]
    return SceneSpec(
        grid=GridSpec(shape=tuple(grid["shape"]), pixel_scale_arcsec=grid["pixel_scale_arcsec"],
                      over_sample_size=grid["over_sample_size"]),
        lens=_galaxy("lens", values["lens"]),
        source=_galaxy("source", values["source"]),
        subhalo=halo_model_from_values(values["subhalo"]),
        subhalo_redshift=values["subhalo"]["redshift"],
        injection=None if values["injection"] is None else injection_from_values(values["injection"]),
        perturbers=perturbers_from_values(values["perturbers"]),
    )


def pixel_centres_yx(shape: tuple[int, int], pixel_scale_arcsec: float) -> tuple[np.ndarray, np.ndarray]:
    """(y, x) of every pixel centre, (ny, nx) each, the values of AutoLens ``Grid2D.uniform(...).native``.

    Row 0 is the top (largest y), column 0 the left (smallest x), origin at the grid centre.
    """
    ny, nx = shape
    scale = float(pixel_scale_arcsec)
    if not (math.isfinite(scale) and scale > 0):
        raise ValueError(f"pixel_scale_arcsec must be positive and finite, got {pixel_scale_arcsec!r}")
    rows, cols = np.indices((ny, nx))
    y = ((rows - float(ny - 1) / 2) * -1.0) * scale
    x = (cols - float(nx - 1) / 2) * scale
    return y, x
