"""The injected subhalo: the ``scene.injection`` block and its placement.

``scene.subhalo`` names the hypothesis family that forecasts and fits test; ``scene.injection``
is the subhalo ``hwoslaps simulate`` and batch simulate jobs put into the truth scene. The
injected halo has the hypothesis type and redshift.

Placements other than ``direct`` are about the lens centre (the centre of the first lens
mass component with one), like forecast layouts. The angle runs counter-clockwise from +x
toward +y. ``radius: einstein_radius`` is the Einstein-radius parameter of the single lens
mass component that has one; ``radius: critical_curve`` is the effective Einstein radius
of the tangential critical curve (``scene.critical_curve``); a number is in arcsec.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

from ..config.checks import Key, Pair, Real, Table, Text, Union, Variants
from .convert import polar_offset
from .cosmology import Cosmology
from .critical_curve import effective_einstein_radius
from .halos import Halo, make_halo

if TYPE_CHECKING:
    from .spec import SceneSpec

__all__ = ["AnglePlacement", "DirectPlacement", "INJECTION_TABLE", "InjectionSpec", "PLACEMENT_TABLE",
           "PlacementSpec", "configured_injection", "injection_from_values"]

Radius = Literal["einstein_radius", "critical_curve"] | float


@dataclass(frozen=True)
class DirectPlacement:
    """The subhalo at ``centre`` (y, x) in its own plane."""

    centre: tuple[float, float]


@dataclass(frozen=True)
class AnglePlacement:
    """The subhalo at ``radius + offset_arcsec`` from the lens centre, at ``angle_deg`` from +x toward +y."""

    angle_deg: float
    radius: Radius
    offset_arcsec: float


PlacementSpec = DirectPlacement | AnglePlacement


@dataclass(frozen=True)
class InjectionSpec:
    mass_msun: float
    position: PlacementSpec


_RADIUS = Key("radius", Union(Text(choices=("einstein_radius", "critical_curve")), Real(min=0.0, min_open=True)),
              "einstein_radius, critical_curve, or a radius about the lens centre", "einstein_radius", unit="arcsec")

PLACEMENT_TABLE = Variants(
    "kind",
    {
        "direct": Table((Key("centre", Pair(Real()), "(y, x) position of the subhalo in its own plane",
                             unit="arcsec"),)),
        "angle": Table((
            Key("angle_deg", Real(), "position angle about the lens centre, from +x toward +y", unit="deg"),
            _RADIUS,
            Key("offset_arcsec", Real(), "added to the radius", 0.0, unit="arcsec"),
        )),
    },
    doc="Where the injected subhalo sits.",
)

INJECTION_TABLE = Table((
    Key("mass_msun", Real(min=0.0, min_open=True), "subhalo mass in the hypothesis mass definition", unit="Msun"),
    Key("position", PLACEMENT_TABLE, "placement of the subhalo"),
))


def injection_from_values(values: Mapping[str, Any]) -> InjectionSpec:
    """The ``InjectionSpec`` of values read by ``INJECTION_TABLE``."""
    position = values["position"]
    if position["kind"] == "direct":
        placement: PlacementSpec = DirectPlacement(tuple(position["centre"]))
    else:
        placement = AnglePlacement(angle_deg=position["angle_deg"], radius=position["radius"],
                                   offset_arcsec=position["offset_arcsec"])
    return InjectionSpec(mass_msun=values["mass_msun"], position=placement)


def _radius(radius: Radius, spec: SceneSpec, cosmology: Cosmology) -> float:
    if radius == "einstein_radius":
        return spec.einstein_radius()
    if radius == "critical_curve":
        return effective_einstein_radius(spec, cosmology)
    return radius


def _position(placement: PlacementSpec, spec: SceneSpec, cosmology: Cosmology) -> tuple[float, float]:
    if isinstance(placement, DirectPlacement):
        return placement.centre
    radius = _radius(placement.radius, spec, cosmology) + placement.offset_arcsec
    if not radius > 0.0:
        raise ValueError(f"the placement radius {radius} arcsec about the lens centre is not positive")
    return polar_offset(radius, placement.angle_deg, spec.lens_centre)


def configured_injection(spec: SceneSpec, cosmology: Cosmology, *, seed: int) -> Halo | None:
    """The configured injected subhalo, or None without an injection block."""
    injection = spec.injection
    if injection is None:
        return None
    redshift = spec.lens.redshift if spec.subhalo_redshift is None else spec.subhalo_redshift
    return make_halo(spec.subhalo, injection.mass_msun, _position(injection.position, spec, cosmology),
                     redshift=redshift, source_redshift=spec.source.redshift, cosmology=cosmology)
