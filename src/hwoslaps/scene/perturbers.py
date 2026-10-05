"""Perturbing halos: listed halos at any redshift, realized once per preparation or simulation.

The realized tuple is passed to every scene build of a preparation, so perturbers never
move between the baseline, nuisance renders and node renders. Its index is the component
name ``perturber_<index>`` of the plane assembly rule (``scene.builder``). A halo's ``centre``
is its angular position in its own plane; for a halo behind the lens that is not where it
appears on the image.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from ..config.checks import Key, ListOf, Nullable, Pair, Real, Table
from .cosmology import Cosmology
from .halos import Halo, HaloModel, halo_model_from_values, halo_model_table, make_halo

if TYPE_CHECKING:
    from .spec import SceneSpec

__all__ = ["ListedHalo", "PERTURBERS_TABLE", "PerturberSpec", "perturbers_from_values", "realize_perturbers"]


@dataclass(frozen=True)
class ListedHalo:
    """A fixed perturber; ``redshift`` None is the lens redshift."""

    model: HaloModel
    mass_msun: float
    centre_yx: tuple[float, float]
    redshift: float | None


@dataclass(frozen=True)
class PerturberSpec:
    """Listed halos; ``populations`` are the drawn halo populations of W2-PERTURB and empty here."""

    halos: tuple[ListedHalo, ...]
    populations: tuple[()]


PERTURBERS_TABLE = Table((
    Key("halos", ListOf(halo_model_table((
        Key("mass_msun", Real(min=0.0, min_open=True), "halo mass in its mass definition", unit="Msun"),
        Key("centre", Pair(Real()), "(y, x) position in the halo's own plane", unit="arcsec"),
        Key("redshift", Nullable(Real(min=0.0, min_open=True)), "halo redshift; null is the lens redshift", None),
    ))), "fixed perturbing halos, in order", []),
))


def perturbers_from_values(values: Mapping[str, Any]) -> PerturberSpec:
    """The ``PerturberSpec`` of values read by ``PERTURBERS_TABLE``."""
    halos = tuple(ListedHalo(model=halo_model_from_values(item), mass_msun=item["mass_msun"],
                             centre_yx=tuple(item["centre"]), redshift=item["redshift"]) for item in values["halos"])
    return PerturberSpec(halos=halos, populations=())


def realize_perturbers(spec: SceneSpec, cosmology: Cosmology, *, seed: int) -> tuple[Halo, ...]:
    """The scene's perturbers as halos, listed halos in list order.

    ``seed`` is the configuration seed, whose named streams draw the halo populations of
    W2-PERTURB; listed halos use no randomness.
    """
    return tuple(
        make_halo(halo.model, halo.mass_msun, halo.centre_yx,
                  redshift=spec.lens.redshift if halo.redshift is None else halo.redshift,
                  source_redshift=spec.source.redshift, cosmology=cosmology)
        for halo in spec.perturbers.halos)
