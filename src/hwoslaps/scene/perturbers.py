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
from typing import TYPE_CHECKING, Any, Literal, Sequence

import numpy as np

from ..config.checks import ConfigError, Integer, Key, ListOf, Nullable, Pair, Real, Rule, Table, Variants
from ..seeding import stream_rng
from .cosmology import Cosmology
from .halos import (MOLINE2017_MASS_RANGE_MSUN, Halo, HaloModel, halo_model_from_values, halo_model_table,
                    make_halo)

if TYPE_CHECKING:
    from .spec import SceneSpec

__all__ = ["ListedHalo", "MassFunctionSpec", "SpatialSpec", "HaloPopulationSpec", "PERTURBERS_TABLE",
           "PerturberSpec", "check_population_context", "draw_population", "perturbers_from_values",
           "realize_perturbers"]


@dataclass(frozen=True)
class ListedHalo:
    """A fixed perturber; ``redshift`` None is the lens redshift."""

    model: HaloModel
    mass_msun: float
    centre_yx: tuple[float, float]
    redshift: float | None


@dataclass(frozen=True)
class MassFunctionSpec:
    kind: Literal["power_law"]
    slope: float
    mass_min_msun: float
    mass_max_msun: float
    count: int | None
    expected_count: float | None


@dataclass(frozen=True)
class SpatialSpec:
    kind: Literal["uniform_disk", "uniform_annulus"]
    radius_arcsec: float | None = None
    inner_arcsec: float | None = None
    outer_arcsec: float | None = None


@dataclass(frozen=True)
class HaloPopulationSpec:
    model: HaloModel
    mass_function: MassFunctionSpec
    spatial: SpatialSpec
    redshift: float | None


@dataclass(frozen=True)
class PerturberSpec:
    """Listed halos followed by named-stream population realizations."""

    halos: tuple[ListedHalo, ...]
    populations: tuple[HaloPopulationSpec, ...]


def _mass_bounds(values: Mapping[str, Any], path: str) -> None:
    if not values["mass_min_msun"] < values["mass_max_msun"]:
        raise ConfigError(f"{path}.mass_max_msun", "must exceed mass_min_msun")


def _annulus_bounds(values: Mapping[str, Any], path: str) -> None:
    if not values["inner_arcsec"] < values["outer_arcsec"]:
        raise ConfigError(f"{path}.outer_arcsec", "must exceed inner_arcsec")


_MASS_FUNCTION_TABLE = Variants("kind", {"power_law": Table((
    Key("slope", Real(), "slope of dN/dM proportional to M**slope"),
    Key("mass_min_msun", Real(min=0.0, min_open=True), "minimum population mass", unit="Msun"),
    Key("mass_max_msun", Real(min=0.0, min_open=True), "maximum population mass", unit="Msun"),
    Key("count", Nullable(Integer(min=0)), "fixed number of halos", None),
    Key("expected_count", Nullable(Real(min=0.0, min_open=True)), "mean Poisson count", None),
), exactly_one=(("count", "expected_count"),), rules=(Rule("mass_max_msun exceeds mass_min_msun", _mass_bounds),))})

_SPATIAL_TABLE = Variants("kind", {
    "uniform_disk": Table((Key("radius_arcsec", Real(min=0.0, min_open=True),
                               "disc radius about the configured lens centre", unit="arcsec"),)),
    "uniform_annulus": Table((
        Key("inner_arcsec", Real(min=0.0), "inner radius about the configured lens centre", unit="arcsec"),
        Key("outer_arcsec", Real(min=0.0, min_open=True), "outer radius", unit="arcsec"),
    ), rules=(Rule("outer_arcsec exceeds inner_arcsec", _annulus_bounds),)),
})

_POPULATION_TABLE = halo_model_table((
    Key("mass_function", _MASS_FUNCTION_TABLE, "population mass distribution and count"),
    Key("spatial", _SPATIAL_TABLE, "population positions in their own plane"),
    Key("redshift", Nullable(Real(min=0.0, min_open=True)), "population redshift; null is the lens redshift", None),
))


PERTURBERS_TABLE = Table((
    Key("halos", ListOf(halo_model_table((
        Key("mass_msun", Real(min=0.0, min_open=True), "halo mass in its mass definition", unit="Msun"),
        Key("centre", Pair(Real()), "(y, x) position in the halo's own plane", unit="arcsec"),
        Key("redshift", Nullable(Real(min=0.0, min_open=True)), "halo redshift; null is the lens redshift", None),
    ))), "fixed perturbing halos, in order", []),
    Key("populations", ListOf(_POPULATION_TABLE), "drawn halo populations, in order", []),
))


def perturbers_from_values(values: Mapping[str, Any]) -> PerturberSpec:
    """The ``PerturberSpec`` of values read by ``PERTURBERS_TABLE``."""
    halos = tuple(ListedHalo(model=halo_model_from_values(item), mass_msun=item["mass_msun"],
                             centre_yx=tuple(item["centre"]), redshift=item["redshift"]) for item in values["halos"])
    populations = []
    for item in values["populations"]:
        mass, spatial = item["mass_function"], item["spatial"]
        populations.append(HaloPopulationSpec(
            model=halo_model_from_values(item), mass_function=MassFunctionSpec(**mass),
            spatial=SpatialSpec(**spatial), redshift=item["redshift"]))
    return PerturberSpec(halos=halos, populations=tuple(populations))


def check_population_context(populations: Sequence[Mapping[str, Any]], path: str, *,
                             lens_redshift: float, source_redshift: float) -> None:
    """Scene-context checks after population section domains have been read."""
    for index, population in enumerate(populations):
        where = f"{path}[{index}]"
        redshift = population["redshift"]
        if redshift is not None and not redshift < source_redshift:
            raise ConfigError(f"{where}.redshift", f"must lie in front of the source at {source_redshift}, got {redshift}")
        concentration = population.get("concentration")
        if concentration is None or concentration["kind"] != "moline2017_eq7":
            continue
        if redshift is not None and redshift != lens_redshift:
            raise ConfigError(f"{where}.redshift", "moline2017_eq7 requires the lens redshift; use fixed or power_law off plane")
        low, high = MOLINE2017_MASS_RANGE_MSUN
        for key in ("mass_min_msun", "mass_max_msun"):
            mass = population["mass_function"][key]
            if not low <= mass <= high:
                raise ConfigError(f"{where}.mass_function.{key}",
                                  f"moline2017_eq7 is calibrated for M200 in [{low:g}, {high:g}] Msun, got {mass:g}")


def draw_population(spec: HaloPopulationSpec, *, seed: int, index: int,
                    lens_centre_yx: tuple[float, float]) -> tuple[np.ndarray, np.ndarray]:
    """Draw masses and absolute own-plane positions from four independent named streams.

    Masses are drawn by inverting the mass-function CDF. Extremely conditioned finite slopes can overflow or
    lose representable output range; invalid results raise rather than being clipped.
    """
    mass = spec.mass_function
    if mass.kind != "power_law":
        raise ValueError(f"unsupported mass function {mass.kind!r}")
    count = mass.count if mass.count is not None else int(
        stream_rng(seed, "scene.perturbers.count", index).poisson(mass.expected_count))
    uniform = stream_rng(seed, "scene.perturbers.mass", index).random(count)
    low, high, exponent = mass.mass_min_msun, mass.mass_max_msun, mass.slope + 1
    try:
        with np.errstate(over="ignore", under="ignore", invalid="ignore", divide="ignore"):
            if exponent == 0:
                masses = low * (high / low) ** uniform
            else:
                masses = (low**exponent + uniform * (high**exponent - low**exponent)) ** (1 / exponent)
    except OverflowError as error:
        raise ValueError("population mass inverse CDF overflowed; use representable power-law parameters") from error
    if not np.all(np.isfinite(masses)) or np.any(masses < low) or np.any(masses > high):
        raise ValueError("population mass inverse CDF produced nonfinite or out-of-range values; "
                         "use parameters whose power-law arithmetic is representable")
    radial = stream_rng(seed, "scene.perturbers.radius", index).random(count)
    spatial = spec.spatial
    try:
        with np.errstate(over="ignore", invalid="ignore"):
            if spatial.kind == "uniform_disk":
                radius = spatial.radius_arcsec * np.sqrt(radial)
                inner, outer = 0.0, spatial.radius_arcsec
            elif spatial.kind == "uniform_annulus":
                radius = np.sqrt(spatial.inner_arcsec**2 + radial * (spatial.outer_arcsec**2 - spatial.inner_arcsec**2))
                inner, outer = spatial.inner_arcsec, spatial.outer_arcsec
            else:
                raise ValueError(f"unsupported spatial law {spatial.kind!r}")
            if not np.all(np.isfinite(radius)) or np.any(radius < inner) or np.any(radius > outer):
                raise ValueError("population radial inverse CDF produced nonfinite or out-of-domain radii; "
                                 "use radii whose squared-radius arithmetic is representable")
            phi = 2 * np.pi * stream_rng(seed, "scene.perturbers.angle", index).random(count)
            positions = np.column_stack((radius * np.sin(phi), radius * np.cos(phi))) + np.asarray(lens_centre_yx)
    except OverflowError as error:
        raise ValueError("population spatial draw overflowed; use representable radii") from error
    if not np.all(np.isfinite(positions)):
        raise ValueError("population spatial draw produced nonfinite positions")
    return masses, positions


def realize_perturbers(spec: SceneSpec, cosmology: Cosmology, *, seed: int) -> tuple[Halo, ...]:
    """The scene's perturbers as halos, listed halos in list order.

    ``seed`` is the configuration seed, whose named streams draw the halo populations;
    listed halos use no randomness.
    """
    listed = tuple(
        make_halo(halo.model, halo.mass_msun, halo.centre_yx,
                  redshift=spec.lens.redshift if halo.redshift is None else halo.redshift,
                  source_redshift=spec.source.redshift, cosmology=cosmology)
        for halo in spec.perturbers.halos)
    populations = []
    for index, population in enumerate(spec.perturbers.populations):
        masses, positions = draw_population(population, seed=seed, index=index, lens_centre_yx=spec.lens_centre)
        redshift = spec.lens.redshift if population.redshift is None else population.redshift
        populations.extend(make_halo(population.model, mass, tuple(position), redshift=redshift,
                                     source_redshift=spec.source.redshift, cosmology=cosmology)
                           for mass, position in zip(masses, positions))
    return listed + tuple(populations)
