"""The injected subhalo: the ``scene.injection`` block and its placement.

``scene.subhalo`` names the hypothesis family that forecasts and fits test; ``scene.injection``
is the subhalo ``hwoslaps simulate`` and batch simulate jobs put into the truth scene. The
injected halo has the hypothesis type and redshift.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from ..config.checks import Key, Pair, Real, Table, Variants
from .cosmology import Cosmology
from .halos import Halo, make_halo

if TYPE_CHECKING:
    from .spec import SceneSpec

__all__ = ["DirectPlacement", "INJECTION_TABLE", "InjectionSpec", "PLACEMENT_TABLE", "PlacementSpec",
           "configured_injection", "injection_from_values"]


@dataclass(frozen=True)
class DirectPlacement:
    """The subhalo at ``centre`` (y, x) in its own plane."""

    centre: tuple[float, float]


PlacementSpec = DirectPlacement


@dataclass(frozen=True)
class InjectionSpec:
    mass_msun: float
    position: PlacementSpec


PLACEMENT_TABLE = Variants(
    "kind",
    {"direct": Table((Key("centre", Pair(Real()), "(y, x) position of the subhalo in its own plane", unit="arcsec"),))},
    doc="Where the injected subhalo sits.",
)

INJECTION_TABLE = Table((
    Key("mass_msun", Real(min=0.0, min_open=True), "subhalo mass in the hypothesis mass definition", unit="Msun"),
    Key("position", PLACEMENT_TABLE, "placement of the subhalo"),
))


def injection_from_values(values: Mapping[str, Any]) -> InjectionSpec:
    """The ``InjectionSpec`` of values read by ``INJECTION_TABLE``."""
    return InjectionSpec(mass_msun=values["mass_msun"], position=DirectPlacement(tuple(values["position"]["centre"])))


def configured_injection(spec: SceneSpec, cosmology: Cosmology, *, seed: int) -> Halo | None:
    """The configured injected subhalo, or None without an injection block."""
    injection = spec.injection
    if injection is None:
        return None
    redshift = spec.lens.redshift if spec.subhalo_redshift is None else spec.subhalo_redshift
    return make_halo(spec.subhalo, injection.mass_msun, injection.position.centre, redshift=redshift,
                     source_redshift=spec.source.redshift, cosmology=cosmology)
