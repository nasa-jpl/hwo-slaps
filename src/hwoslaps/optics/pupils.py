"""Telescope pupils: geometry specifications, their key table, and the sampled HCIPy pupil.

Two pupil kinds share one geometry convention. ``diameter_m`` is the side of the square
pupil grid: a circular aperture fills it, and a hexagonally segmented aperture must fit
inside it (a geometric guard on the segment vertices). A central obscuration and spiders
apply to either kind. Segment numbering is HCIPy's hexagonal-grid order (index 0 is the
central segment when present); hexagons are flat-top.

On a pupil with modifiers some segments may be dark. A segment is active when at least one
of its pixels is illuminated (``transmission > 0.5``, the aperture mask of the wavefront
RMS and of the prior basis) and lies in the segment (mask above 0.5); only active segments
carry segment modes in draws and nuisances, and segment indices never change.
"""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, ClassVar

import numpy as np

from ..config.checks import Boolean, ConfigError, Integer, Key, Nullable, Real, Rule, Table, Variants
from ..identity import json_ready

__all__ = [
    "PUPIL_TABLE", "CircularPupilSpec", "HexSegmentedPupilSpec", "Pupil", "PupilSpec", "SpiderSpec",
    "available_families", "build_pupil", "parse_pupil", "segment_count",
]


@dataclass(frozen=True)
class SpiderSpec:
    """``count`` straight spiders of full width ``width_m`` from the pupil centre outward.

    The first runs at ``angle_deg`` from +x toward +y, the others at equal angles after it.
    """

    count: int
    width_m: float
    angle_deg: float


@dataclass(frozen=True)
class HexSegmentedPupilSpec:
    """A flat-top hexagonally segmented aperture on a square pupil grid of side ``diameter_m``."""

    kind: ClassVar[str] = "hex_segmented"
    diameter_m: float
    pixels: int
    supersampling: int
    obscuration_ratio: float
    spiders: SpiderSpec | None
    rings: int
    segment_point_to_point_m: float
    gap_m: float
    central_segment: bool


@dataclass(frozen=True)
class CircularPupilSpec:
    """A circular aperture of diameter ``diameter_m`` filling the square pupil grid."""

    kind: ClassVar[str] = "circular"
    diameter_m: float
    pixels: int
    supersampling: int
    obscuration_ratio: float
    spiders: SpiderSpec | None


PupilSpec = HexSegmentedPupilSpec | CircularPupilSpec

_FAMILIES_BY_KIND = {
    "hex_segmented": ("segment_hexikes", "zernikes"),
    "circular": ("zernikes",),
}


def segment_count(spec: PupilSpec) -> int:
    """Number of segments: ``3 r (r + 1) + 1`` hexagons, one fewer without the centre; 0 for a circle."""
    if isinstance(spec, CircularPupilSpec):
        return 0
    return 3 * spec.rings * (spec.rings + 1) + (1 if spec.central_segment else 0)


def available_families(spec: PupilSpec) -> tuple[str, ...]:
    """Wavefront families defined on the pupil, in canonical order."""
    return _FAMILIES_BY_KIND[spec.kind]


def _segment_centres_xy(rings: int, pitch_m: float, central_segment: bool) -> np.ndarray:
    """Segment centres in HCIPy's order (``make_hexagonal_grid(pitch, rings, pointy_top=False)``)."""
    q, r = [0], [0]
    for n in range(1, rings + 1):
        q += list(range(n, 0, -1)) + list(range(0, -n, -1)) + [-n] * n
        r += list(range(0, n)) + [n] * n + list(range(n, 0, -1))
        q += list(range(-n, 0)) + list(range(0, n)) + [n] * n
        r += list(range(0, -n, -1)) + [-n] * n + list(range(-n, 0))
    q_axial, r_axial = np.array(q), np.array(r)
    centres = np.column_stack(((q_axial + r_axial) * pitch_m * math.sqrt(3) / 2, (r_axial - q_axial) * pitch_m / 2))
    return centres if central_segment else centres[1:]


def _segment_half_width_m(rings: int, point_to_point_m: float, gap_m: float, central_segment: bool) -> float:
    """Largest |x| or |y| of any segment vertex: the half side a square grid needs to contain the aperture."""
    flat_to_flat = point_to_point_m * math.sqrt(3) / 2
    centres = _segment_centres_xy(rings, flat_to_flat + gap_m, central_segment)
    angles = np.radians(np.arange(0.0, 360.0, 60.0))
    vertices = 0.5 * point_to_point_m * np.column_stack((np.cos(angles), np.sin(angles)))
    corners = centres[:, np.newaxis, :] + vertices[np.newaxis, :, :]
    return float(np.max(np.abs(corners)))


def _hex_geometry_problem(rings: int, point_to_point_m: float, gap_m: float, central_segment: bool,
                          diameter_m: float) -> tuple[str, str] | None:
    """The offending key and the reason when a segmented aperture cannot be built, else None."""
    if not central_segment and rings < 1:
        return "rings", "a pupil without the central segment needs rings >= 1"
    needed = _segment_half_width_m(rings, point_to_point_m, gap_m, central_segment)
    if needed > diameter_m / 2:
        return "diameter_m", (f"the segmented aperture reaches {needed:.6g} m from the centre, outside the pupil "
                              f"grid of half side {diameter_m / 2:.6g} m; set diameter_m to at least {2 * needed:.6g}")
    return None


def _check_hex_geometry(values: Mapping[str, Any], path: str) -> None:
    problem = _hex_geometry_problem(values["rings"], values["segment_point_to_point_m"], values["gap_m"],
                                    values["central_segment"], values["diameter_m"])
    if problem is not None:
        key, message = problem
        raise ConfigError(f"{path}.{key}" if path else key, message)


SPIDER_TABLE = Table((
    Key("count", Integer(min=1), "number of spiders, at equal angles"),
    Key("width_m", Real(min=0.0, min_open=True), "full width of each spider", unit="m"),
    Key("angle_deg", Real(), "direction of the first spider, from +x toward +y", default=0.0, unit="deg"),
))

_COMMON_KEYS = (
    Key("diameter_m", Real(min=0.0, min_open=True),
        "side of the square pupil grid; a circular aperture fills it, a segmented aperture must fit inside",
        unit="m"),
    Key("pixels", Integer(min=1), "pupil samples per side"),
    Key("supersampling", Integer(min=1), "sub-samples per pupil pixel side when the aperture is evaluated"),
    Key("obscuration_ratio", Real(min=0.0, max=1.0, max_open=True),
        "central obscuration diameter as a fraction of diameter_m", default=0.0),
    Key("spiders", Nullable(SPIDER_TABLE), "spiders from the centre outward", default=None),
)

PUPIL_TABLE = Variants("kind", {
    "hex_segmented": Table(_COMMON_KEYS + (
        Key("rings", Integer(min=0), "rings of hexagonal segments around the centre"),
        Key("segment_point_to_point_m", Real(min=0.0, min_open=True), "segment vertex-to-vertex size", unit="m"),
        Key("gap_m", Real(min=0.0), "gap between adjacent segments", unit="m"),
        Key("central_segment", Boolean(), "whether the central segment is present", default=True),
    ), rules=(Rule("rings >= 1 without the central segment; every segment vertex lies inside the pupil grid",
                   _check_hex_geometry),)),
    "circular": Table(_COMMON_KEYS),
})


def _spider_spec(values: Mapping[str, Any] | None) -> SpiderSpec | None:
    return None if values is None else SpiderSpec(values["count"], values["width_m"], values["angle_deg"])


def parse_pupil(mapping: Mapping[str, Any], path: str) -> PupilSpec:
    """The pupil specification of a ``pupil`` mapping, read strictly through ``PUPIL_TABLE``."""
    values = PUPIL_TABLE.read(mapping, path)
    common = dict(diameter_m=values["diameter_m"], pixels=values["pixels"], supersampling=values["supersampling"],
                  obscuration_ratio=values["obscuration_ratio"], spiders=_spider_spec(values["spiders"]))
    if values["kind"] == "circular":
        return CircularPupilSpec(**common)
    return HexSegmentedPupilSpec(**common, rings=values["rings"],
                                 segment_point_to_point_m=values["segment_point_to_point_m"],
                                 gap_m=values["gap_m"], central_segment=values["central_segment"])


@dataclass(frozen=True, eq=False)
class Pupil:
    """A pupil sampled on its HCIPy grid; built by ``build_pupil``.

    ``transmission`` is the grey amplitude transmission in [0, 1]. ``segments`` is the
    sequence of grey segment masks HCIPy evaluates for the segment generators (hex
    pupils; None for a circular pupil), in HCIPy order, and ``segment_centres`` their
    centres. ``illuminated_mask`` (``transmission > 0.5``) is the aperture over which
    wavefront RMS values and prior bases are defined. ``zernike_diameter_m`` is the disc
    the global Zernikes are normalized on: the grid extent ``x.max() - x.min()`` of a hex
    pupil (the paper convention) or the aperture diameter of a circular pupil.
    ``collecting_area_m2`` is ``sum(transmission) (diameter_m / pixels)^2`` and
    ``total_power`` is ``sum(transmission^2 w)`` over the grid weights ``w``.
    """

    spec: PupilSpec
    grid: Any
    transmission: Any
    segments: Any | None
    segment_centres: Any | None
    segment_count: int = field(init=False)
    zernike_diameter_m: float = field(init=False)
    collecting_area_m2: float = field(init=False)
    total_power: float = field(init=False)
    illuminated_mask: np.ndarray = field(init=False)
    active_segments: tuple[int, ...] = field(init=False)

    def __post_init__(self) -> None:
        values = np.asarray(self.transmission)
        illuminated = values > 0.5
        illuminated.flags.writeable = False
        count = segment_count(self.spec)
        if isinstance(self.spec, HexSegmentedPupilSpec):
            if self.segments is None or len(self.segments) != count:
                raise ValueError(f"a pupil of {self.spec.rings} rings has {count} segments, "
                                 f"got {None if self.segments is None else len(self.segments)} segment masks")
            active = tuple(s for s in range(count)
                           if np.any(illuminated & (np.asarray(self.segments[s]) > 0.5)))
            zernike_diameter = float(self.grid.x.max() - self.grid.x.min())
        else:
            if self.segments is not None:
                raise ValueError("a circular pupil has no segment masks")
            active = ()
            zernike_diameter = float(self.spec.diameter_m)
        object.__setattr__(self, "segment_count", count)
        object.__setattr__(self, "zernike_diameter_m", zernike_diameter)
        object.__setattr__(self, "collecting_area_m2",
                           float(np.sum(values)) * (self.spec.diameter_m / self.spec.pixels) ** 2)
        object.__setattr__(self, "total_power", float(np.sum(values ** 2 * self.grid.weights)))
        object.__setattr__(self, "illuminated_mask", illuminated)
        object.__setattr__(self, "active_segments", active)

    @property
    def dark_segments(self) -> tuple[int, ...]:
        return tuple(s for s in range(self.segment_count) if s not in self.active_segments)

    def to_mapping(self) -> dict[str, Any]:
        return json_ready({
            "kind": self.spec.kind,
            "spec": dataclasses.asdict(self.spec),
            "segment_count": self.segment_count,
            "active_segments": list(self.active_segments),
            "dark_segments": list(self.dark_segments),
            "zernike_diameter_m": self.zernike_diameter_m,
            "collecting_area_m2": self.collecting_area_m2,
            "total_power": self.total_power,
        })


def _with_modifiers(aperture: Any, spec: PupilSpec) -> Any:
    """The aperture generator times the obscuration and spiders; the bare aperture when there are none."""
    import hcipy

    factors = []
    if spec.obscuration_ratio > 0.0:
        inner = hcipy.make_circular_aperture(spec.obscuration_ratio * spec.diameter_m)
        factors.append(lambda grid: 1 - inner(grid))
    if spec.spiders is not None:
        for k in range(spec.spiders.count):
            angle = math.radians(spec.spiders.angle_deg) + 2 * math.pi * k / spec.spiders.count
            end = (spec.diameter_m * math.cos(angle), spec.diameter_m * math.sin(angle))
            factors.append(hcipy.make_spider((0.0, 0.0), end, spec.spiders.width_m))
    if not factors:
        return aperture

    def generator(grid: Any) -> Any:
        values = aperture(grid)
        for factor in factors:
            values = values * factor(grid)
        return values

    return generator


def build_pupil(spec: PupilSpec) -> Pupil:
    """Sample the aperture (with modifiers) and, for a hex pupil, every segment mask."""
    import hcipy

    grid = hcipy.make_pupil_grid(dims=spec.pixels, diameter=spec.diameter_m)
    segments = centres = None
    if isinstance(spec, HexSegmentedPupilSpec):
        problem = _hex_geometry_problem(spec.rings, spec.segment_point_to_point_m, spec.gap_m,
                                        spec.central_segment, spec.diameter_m)
        if problem is not None:
            raise ValueError(problem[1])
        flat_to_flat = spec.segment_point_to_point_m * np.sqrt(3) / 2
        aperture, generators = hcipy.make_hexagonal_segmented_aperture(
            spec.rings, flat_to_flat, spec.gap_m, starting_ring=0 if spec.central_segment else 1,
            return_segments=True)
        centres = hcipy.make_hexagonal_grid(flat_to_flat + spec.gap_m, spec.rings, pointy_top=False)
        if not spec.central_segment:
            keep = centres.zeros(dtype="bool")
            keep[1:] = True
            centres = centres.subset(keep)
        segments = hcipy.evaluate_supersampled(generators, grid, spec.supersampling)
    else:
        aperture = hcipy.make_circular_aperture(spec.diameter_m)
    transmission = hcipy.evaluate_supersampled(_with_modifiers(aperture, spec), grid, spec.supersampling)
    return Pupil(spec, grid, transmission, segments, centres)
