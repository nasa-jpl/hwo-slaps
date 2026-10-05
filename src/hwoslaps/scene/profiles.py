"""Profile registry: the one extension point for mass and light models.

Each ``ProfileType`` defines a component type completely:
- ``table``: the configuration keys of the component (without ``type``);
- ``parameters(values)``: the ordered scalar parameters, which are the nuisance columns
  and the free parameters of a fit, with their kind, step mode and per-scalar domain;
- ``layout(values)``: the AutoLens profiles of the component and how their constructor
  arguments come from the parameters, shared by the truth scene and the fitted galaxy;
- ``amplitude_key`` and ``unit_integral(values)``: the photometric normalization, the
  integral in arcsec^2 of the profile at unit amplitude.

Component types here: ``Isothermal`` (mass), ``Exponential`` and ``Image`` (light).
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

import numpy as np

from ..config.checks import ConfigError, Ellipticity, FilePath, Key, Nullable, Pair, Real, Rule, Table

if TYPE_CHECKING:
    from .image_source import ImageAsset
    from .spec import ComponentSpec

__all__ = [
    "ArgumentLayout", "ELLIPTICITY_LIMIT", "Fixed", "Interval", "Link", "PROFILE_TYPES", "ParameterDef",
    "ParameterKind", "ParameterRef", "ProfileLayout", "ProfileType", "instantiate", "sersic_constant",
    "sersic_unit_integral",
]

ParameterKind = Literal["position", "einstein_radius", "ellipticity", "slope", "multipole", "shear", "amplitude",
                        "size", "sersic_index", "orientation"]

ELLIPTICITY_LIMIT = 0.999
"""Components need ``hypot(ell_comps) < ELLIPTICITY_LIMIT``: AutoGalaxy clamps the ellipticity there silently."""


@dataclass(frozen=True)
class Interval:
    """A real interval, open or closed at each end; infinite ends are allowed."""

    lower: float
    upper: float
    open_lower: bool = True
    open_upper: bool = True

    def contains(self, value: float) -> bool:
        if not Real().accepts(value) or not math.isfinite(value):
            return False
        above = value > self.lower if self.open_lower else value >= self.lower
        below = value < self.upper if self.open_upper else value <= self.upper
        return bool(above and below)

    def describe(self) -> str:
        return f"{'(' if self.open_lower else '['}{self.lower:g}, {self.upper:g}{')' if self.open_upper else ']'}"


@dataclass(frozen=True)
class ParameterDef:
    """One scalar parameter: value at ``values[key]`` (element ``index`` of a pair, or the scalar).

    ``domain`` is the interval of the scalar alone; constraints that join several scalars (the
    ellipticity ``hypot``) are rules of the component table.
    """

    name: str
    key: str
    index: int | None
    kind: ParameterKind
    step_mode: Literal["additive", "multiplicative"]
    domain: Interval


@dataclass(frozen=True)
class ParameterRef:
    """A constructor element taken from the named parameter."""

    parameter: str


@dataclass(frozen=True)
class Fixed:
    """A constructor element held at a constant value."""

    value: Any


@dataclass(frozen=True)
class Link:
    """A constructor element shared with an argument of the component's profile with ``suffix``."""

    suffix: str
    argument: str
    element: int | None


@dataclass(frozen=True)
class ArgumentLayout:
    """One constructor argument: a scalar for one element, a tuple in element order for several."""

    name: str
    elements: tuple[ParameterRef | Fixed | Link, ...]


@dataclass(frozen=True)
class ProfileLayout:
    """One AutoLens profile of a component, set as galaxy attribute ``<component name><suffix>``.

    The Image profile takes its asset from the fixed argument ``asset_path``; ``instantiate``
    passes the loaded asset's samples and pixel scale to ``ImageLightProfile.from_asset``.
    """

    suffix: str
    profile_class: str
    arguments: tuple[ArgumentLayout, ...]


@dataclass(frozen=True)
class ProfileType:
    name: str
    role: Literal["mass", "light"]
    table: Table
    amplitude_key: str | None
    parameters: Callable[[Mapping[str, Any]], tuple[ParameterDef, ...]]
    layout: Callable[[Mapping[str, Any]], tuple[ProfileLayout, ...]]
    unit_integral: Callable[[Mapping[str, Any]], float] | None


# ------------------------------------------------------------------ shared pieces

_REAL_LINE = Interval(-math.inf, math.inf)
_POSITIVE = Interval(0.0, math.inf)
_ELLIPTICITY = Interval(-ELLIPTICITY_LIMIT, ELLIPTICITY_LIMIT)


def _real(domain: Interval) -> Real:
    """The configuration check of a finite scalar in ``domain``, so the key and its parameter share one interval."""
    return Real(min=None if math.isinf(domain.lower) else domain.lower,
                max=None if math.isinf(domain.upper) else domain.upper,
                min_open=domain.open_lower, max_open=domain.open_upper)


_CENTRE = Key("centre", Pair(_real(_REAL_LINE)), "(y, x) centre", unit="arcsec")
_ELL_COMPS = Key("ell_comps", Ellipticity(),
                 f"elliptical components (f sin 2 phi, f cos 2 phi), f = (1 - q) / (1 + q), q the axis ratio and "
                 f"phi the major-axis angle counter-clockwise from +x; hypot below {ELLIPTICITY_LIMIT}")


def _check_ellipticity(values: Mapping[str, Any], path: str) -> None:
    if not math.hypot(*values["ell_comps"]) < ELLIPTICITY_LIMIT:
        raise ConfigError(f"{path}.ell_comps" if path else "ell_comps",
                          f"hypot(e1, e2) must be below {ELLIPTICITY_LIMIT} (AutoGalaxy clamps above it), "
                          f"got {list(values['ell_comps'])}")


_ELLIPTICITY_RULE = Rule(f"hypot(ell_comps) < {ELLIPTICITY_LIMIT}", _check_ellipticity)

_POSITION_PARAMETERS = (
    ParameterDef("centre_y", "centre", 0, "position", "additive", _REAL_LINE),
    ParameterDef("centre_x", "centre", 1, "position", "additive", _REAL_LINE),
)
_ELLIPTICITY_PARAMETERS = (
    ParameterDef("ell_comp_1", "ell_comps", 0, "ellipticity", "additive", _ELLIPTICITY),
    ParameterDef("ell_comp_2", "ell_comps", 1, "ellipticity", "additive", _ELLIPTICITY),
)


def _pair(name: str, first: str, second: str) -> ArgumentLayout:
    return ArgumentLayout(name, (ParameterRef(first), ParameterRef(second)))


def _scalar(name: str) -> ArgumentLayout:
    return ArgumentLayout(name, (ParameterRef(name),))


def sersic_constant(sersic_index: float) -> float:
    """b_n of the Sersic profile: the Ciotti and Bertin (1999) series in AutoGalaxy's operation order."""
    n = sersic_index
    return ((2 * n) - (1.0 / 3.0) + (4.0 / (405.0 * n)) + (46.0 / (25515.0 * n**2))
            + (131.0 / (1148175.0 * n**3)) - (2194697.0 / (30690717750.0 * n**4)))


def sersic_unit_integral(effective_radius: float, sersic_index: float) -> float:
    """Integral (arcsec^2) of a unit-intensity Sersic profile: 2 pi n r_e^2 e^b Gamma(2n) / b^(2n), any ellipticity."""
    from scipy.special import gamma

    n = sersic_index
    b = sersic_constant(n)
    return float(2.0 * np.pi * n * effective_radius**2 * np.exp(b) * gamma(2.0 * n) / b ** (2.0 * n))


# ------------------------------------------------------------------ Isothermal (mass)

_ISOTHERMAL_PARAMETERS = (
    *_POSITION_PARAMETERS,
    ParameterDef("einstein_radius", "einstein_radius", None, "einstein_radius", "additive", _POSITIVE),
    *_ELLIPTICITY_PARAMETERS,
)
_ISOTHERMAL_LAYOUT = (ProfileLayout("", "Isothermal", (
    _pair("centre", "centre_y", "centre_x"),
    _scalar("einstein_radius"),
    _pair("ell_comps", "ell_comp_1", "ell_comp_2"),
)),)

ISOTHERMAL = ProfileType(
    name="Isothermal",
    role="mass",
    table=Table(
        (_CENTRE,
         Key("einstein_radius", _real(_POSITIVE), "AutoLens Einstein radius of the SIE", unit="arcsec"),
         _ELL_COMPS),
        rules=(_ELLIPTICITY_RULE,),
        doc="Singular isothermal ellipsoid (al.mp.Isothermal); AutoGalaxy evaluates it at q <= 0.99999.",
    ),
    amplitude_key=None,
    parameters=lambda values: _ISOTHERMAL_PARAMETERS,
    layout=lambda values: _ISOTHERMAL_LAYOUT,
    unit_integral=None,
)

# ------------------------------------------------------------------ Exponential (light)

_EXPONENTIAL_PARAMETERS = (
    *_POSITION_PARAMETERS,
    *_ELLIPTICITY_PARAMETERS,
    ParameterDef("intensity", "intensity", None, "amplitude", "multiplicative", _POSITIVE),
    ParameterDef("effective_radius", "effective_radius", None, "size", "multiplicative", _POSITIVE),
)
_EXPONENTIAL_LAYOUT = (ProfileLayout("", "Exponential", (
    _pair("centre", "centre_y", "centre_x"),
    _pair("ell_comps", "ell_comp_1", "ell_comp_2"),
    _scalar("intensity"),
    _scalar("effective_radius"),
)),)

EXPONENTIAL = ProfileType(
    name="Exponential",
    role="light",
    table=Table(
        (_CENTRE, _ELL_COMPS,
         Key("effective_radius", _real(_POSITIVE), "circularized half-light radius", unit="arcsec"),
         Key("intensity", Nullable(_real(_POSITIVE)),
             "surface brightness at the effective radius, detected e-/s per pixel sample", None)),
        rules=(_ELLIPTICITY_RULE,),
        doc="Exponential (Sersic n = 1) light profile (al.lp.Exponential).",
    ),
    amplitude_key="intensity",
    parameters=lambda values: _EXPONENTIAL_PARAMETERS,
    layout=lambda values: _EXPONENTIAL_LAYOUT,
    unit_integral=lambda values: sersic_unit_integral(values["effective_radius"], 1.0),
)

# ------------------------------------------------------------------ Image (light)

_IMAGE_PARAMETERS = (
    *_POSITION_PARAMETERS,
    ParameterDef("flux_scale", "flux_scale", None, "amplitude", "multiplicative", _POSITIVE),
    ParameterDef("size_scale", "size_scale", None, "size", "multiplicative", _POSITIVE),
    ParameterDef("rotation_deg", "rotation_deg", None, "orientation", "additive", _REAL_LINE),
)


def _image_layout(values: Mapping[str, Any]) -> tuple[ProfileLayout, ...]:
    return (ProfileLayout("", "ImageLightProfile", (
        _pair("centre", "centre_y", "centre_x"),
        _scalar("rotation_deg"),
        ArgumentLayout("asset_path", (Fixed(values["asset_path"]),)),
        ArgumentLayout("total_flux", (Fixed(values["total_flux"]),)),
        _scalar("flux_scale"),
        _scalar("size_scale"),
    )),)


IMAGE = ProfileType(
    name="Image",
    role="light",
    table=Table(
        (Key("asset_path", FilePath((".npz",)), "prepared image asset (format version 1)"),
         _CENTRE,
         Key("rotation_deg", _real(_REAL_LINE), "counter-clockwise rotation of the image on the sky", 0.0, unit="deg"),
         Key("total_flux", Nullable(Real(min=0.0, min_open=True)),
             "integral of the image at unit flux and size scales, e-/s per pixel sample times arcsec^2", None),
         Key("flux_scale", _real(_POSITIVE), "brightness multiplier", 1.0),
         Key("size_scale", _real(_POSITIVE), "magnification of the image at fixed surface brightness",
             1.0)),
        doc="Pixelized source: a unit-integral asset evaluated by bilinear interpolation with a one-pixel zero pad.",
    ),
    amplitude_key="total_flux",
    parameters=lambda values: _IMAGE_PARAMETERS,
    layout=_image_layout,
    unit_integral=lambda values: values["flux_scale"] * values["size_scale"] ** 2,
)

PROFILE_TYPES: Mapping[str, ProfileType] = {profile.name: profile for profile in (ISOTHERMAL, EXPONENTIAL, IMAGE)}


# ------------------------------------------------------------------ instantiation


def instantiate(component: ComponentSpec, *, assets: Mapping[str, ImageAsset] | None = None) -> Mapping[str, Any]:
    """AutoLens profiles of a ``ComponentSpec``: attribute name -> profile, in layout order.

    ``assets`` maps an absolute asset path to its loaded ``ImageAsset``; when given, no file is
    read and a path missing from it raises.
    """
    profile_type = PROFILE_TYPES[component.type]
    values = component.values
    definitions = {definition.name: definition for definition in profile_type.parameters(values)}
    arguments_by_suffix: dict[str, dict[str, Any]] = {}
    profiles: dict[str, Any] = {}

    def element_value(element: ParameterRef | Fixed | Link) -> Any:
        if isinstance(element, ParameterRef):
            definition = definitions[element.parameter]
            value = values[definition.key]
            return value if definition.index is None else value[definition.index]
        if isinstance(element, Fixed):
            return element.value
        linked = arguments_by_suffix[element.suffix][element.argument]
        return linked if element.element is None else linked[element.element]

    for layout in profile_type.layout(values):
        arguments = {}
        for argument in layout.arguments:
            elements = tuple(element_value(element) for element in argument.elements)
            arguments[argument.name] = elements[0] if len(elements) == 1 else elements
        arguments_by_suffix[layout.suffix] = arguments
        profiles[component.name + layout.suffix] = _construct(profile_type.role, layout.profile_class, dict(arguments),
                                                              assets)
    return profiles


def _construct(role: str, profile_class: str, arguments: dict[str, Any],
               assets: Mapping[str, ImageAsset] | None) -> Any:
    if profile_class == "ImageLightProfile":
        from .image_profile import ImageLightProfile
        from .image_source import load_image_asset

        path = arguments.pop("asset_path")
        if assets is None:
            asset = load_image_asset(path)
        elif path in assets:
            asset = assets[path]
        else:
            raise KeyError(f"image asset {path} is not among the prepared assets {sorted(assets)}")
        return ImageLightProfile.from_asset(asset, **arguments)
    import autolens as al

    return getattr(al.mp if role == "mass" else al.lp, profile_class)(**arguments)
