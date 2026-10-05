"""Wavefront modes, coefficients, mode selections, and the HCIPy mode basis of one pupil.

Coefficients are nanometres of optical path difference (OPD). Two families exist:
global Zernikes on any pupil (Noll order, unit RMS over the Zernike disc of the pupil)
and segment hexikes on a hex-segmented pupil (Noll order per segment; Noll 1-3 are
segment piston, tip and tilt, unit RMS over the segment). A segment wavefront tilt of
``alpha`` radians is a Noll 2 or 3 coefficient ``alpha R sqrt(5/24)`` with ``R`` half
the segment point-to-point size.

Coefficients are kept in canonical order (segment hexikes by segment then Noll, then
Zernikes by Noll), so equal configurations give equal kernels. Present entries are
significant, zeros included: the hexike surface of an evaluation is built with as many
modes as the highest hexike Noll present, and its bytes depend on that count.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from numbers import Integral, Real as RealNumber
from typing import Any, Literal

import numpy as np

from ..config.checks import ConfigError, Integer, Key, ListOf, MapOf, Nullable, Real, Rule, Table, Text, Union
from ..identity import mapping_digest
from .kernels import deterministic_blas
from .pupils import HexSegmentedPupilSpec, Pupil, PupilSpec, available_families, segment_count

__all__ = [
    "FAMILIES", "WAVEFRONT_SELECTION_TABLE", "WAVEFRONT_TABLE", "WavefrontBasis", "WavefrontCoefficients",
    "WavefrontMode", "WavefrontSelection", "parse_wavefront_selection", "select_modes", "validate_coefficients",
]

FAMILIES: tuple[str, ...] = ("segment_hexikes", "zernikes")
"""Wavefront families in canonical order."""


def _is_integer(value: Any) -> bool:
    return isinstance(value, Integral) and not isinstance(value, (bool, np.bool_))


def _canonical_nm(value: Any, what: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, RealNumber):
        raise ValueError(f"{what} must be a real number of nm, got {value!r}")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{what} must be finite, got {value!r}")
    return 0.0 if number == 0.0 else number


@dataclass(frozen=True)
class WavefrontMode:
    """One wavefront mode: a global Zernike or a hexike on one segment, by Noll index."""

    family: Literal["segment_hexikes", "zernikes"]
    noll: int
    segment: int | None = None

    def __post_init__(self) -> None:
        if self.family not in FAMILIES:
            raise ValueError(f"unknown wavefront family {self.family!r}; families are {FAMILIES}")
        if not (_is_integer(self.noll) and self.noll >= 1):
            raise ValueError(f"Noll indices are integers >= 1, got {self.noll!r}")
        if self.family == "segment_hexikes":
            if not (_is_integer(self.segment) and self.segment >= 0):
                raise ValueError(f"a segment hexike needs a segment index >= 0, got {self.segment!r}")
        elif self.segment is not None:
            raise ValueError(f"a global Zernike has no segment, got {self.segment!r}")
        object.__setattr__(self, "noll", int(self.noll))
        if self.segment is not None:
            object.__setattr__(self, "segment", int(self.segment))

    @property
    def name(self) -> str:
        """``segment_hexikes[<segment>][<noll>]`` or ``zernikes[<noll>]``."""
        if self.family == "segment_hexikes":
            return f"segment_hexikes[{self.segment}][{self.noll}]"
        return f"zernikes[{self.noll}]"

    @property
    def sort_key(self) -> tuple[int, int, int]:
        return (FAMILIES.index(self.family), -1 if self.segment is None else self.segment, self.noll)


WAVEFRONT_TABLE = Table((
    Key("segment_hexikes", MapOf(Integer(min=0), MapOf(Integer(min=1), Real(), min_length=1)),
        "segment index -> {Noll index: coefficient}; hex-segmented pupils only", default={}, unit="nm"),
    Key("zernikes", MapOf(Integer(min=1), Real()), "Noll index -> global Zernike coefficient", default={},
        unit="nm"),
), doc="Wavefront coefficients in nm of optical path difference; a listed zero is an entry.")


@dataclass(frozen=True)
class WavefrontCoefficients:
    """A wavefront as ``(mode, nm)`` entries, held in canonical order.

    Values are canonical floats (``-0.0`` becomes ``0.0``); a mode appears at most once.
    Arithmetic acts on the union of the modes, an absent mode counting as 0.0.
    """

    entries: tuple[tuple[WavefrontMode, float], ...] = ()

    def __post_init__(self) -> None:
        canonical = []
        for entry in self.entries:
            mode, value = entry
            if not isinstance(mode, WavefrontMode):
                raise ValueError(f"entries are (WavefrontMode, nm) pairs, got {entry!r}")
            canonical.append((mode, _canonical_nm(value, mode.name)))
        canonical.sort(key=lambda item: item[0].sort_key)
        modes = [mode for mode, _ in canonical]
        if len(set(modes)) != len(modes):
            raise ValueError("a wavefront mode appears more than once")
        object.__setattr__(self, "entries", tuple(canonical))

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, Any], path: str) -> WavefrontCoefficients:
        """Coefficients from ``{"segment_hexikes": {s: {n: nm}}, "zernikes": {n: nm}}``, read strictly."""
        values = WAVEFRONT_TABLE.read(mapping, path)
        entries = [(WavefrontMode("segment_hexikes", noll, segment), value)
                   for segment, modes in values["segment_hexikes"].items() for noll, value in modes.items()]
        entries += [(WavefrontMode("zernikes", noll), value) for noll, value in values["zernikes"].items()]
        return cls(tuple(entries))

    @classmethod
    def empty(cls) -> WavefrontCoefficients:
        return cls(())

    def to_mapping(self) -> dict[str, Any]:
        """The mapping form of ``from_mapping``, ascending, with integer keys."""
        hexikes: dict[int, dict[int, float]] = {}
        for segment, noll, value in self.segment_hexikes():
            hexikes.setdefault(segment, {})[noll] = value
        return {"segment_hexikes": hexikes, "zernikes": dict(self.zernikes())}

    def digest(self) -> str:
        return mapping_digest(self.to_mapping())

    def value(self, mode: WavefrontMode) -> float:
        """The coefficient of ``mode``; 0.0 when absent."""
        return dict(self.entries).get(mode, 0.0)

    def replace(self, mode: WavefrontMode, value: float) -> WavefrontCoefficients:
        """These coefficients with ``mode`` set to ``value`` (inserted when absent)."""
        return WavefrontCoefficients(tuple(item for item in self.entries if item[0] != mode) + ((mode, value),))

    def plus(self, other: WavefrontCoefficients) -> WavefrontCoefficients:
        return self._combine(other, lambda a, b: a + b)

    def minus(self, other: WavefrontCoefficients) -> WavefrontCoefficients:
        return self._combine(other, lambda a, b: a - b)

    def _combine(self, other: WavefrontCoefficients, operation: Any) -> WavefrontCoefficients:
        mine, theirs = dict(self.entries), dict(other.entries)
        modes = sorted(set(mine) | set(theirs), key=lambda mode: mode.sort_key)
        return WavefrontCoefficients(tuple(
            (mode, operation(float(mine.get(mode, 0.0)), float(theirs.get(mode, 0.0)))) for mode in modes))

    def scaled(self, factor: float) -> WavefrontCoefficients:
        return WavefrontCoefficients(tuple((mode, float(value * factor)) for mode, value in self.entries))

    @property
    def is_empty(self) -> bool:
        return not self.entries

    def segment_hexikes(self) -> tuple[tuple[int, int, float], ...]:
        """``(segment, noll, nm)`` for every hexike entry, in canonical order."""
        return tuple((mode.segment, mode.noll, value) for mode, value in self.entries
                     if mode.family == "segment_hexikes")

    def zernikes(self) -> tuple[tuple[int, float], ...]:
        """``(noll, nm)`` for every Zernike entry, ascending."""
        return tuple((mode.noll, value) for mode, value in self.entries if mode.family == "zernikes")

    def hexike_mode_count(self) -> int:
        """The highest hexike Noll present (the hexike mode count of an evaluation); 0 without hexikes."""
        return max((noll for _, noll, _ in self.segment_hexikes()), default=0)

    def max_zernike_noll(self) -> int:
        return max((noll for noll, _ in self.zernikes()), default=0)


def validate_coefficients(coefficients: WavefrontCoefficients, pupil: PupilSpec, path: str) -> None:
    """Segment hexikes only on a hex-segmented pupil, on segments it has."""
    hexikes = coefficients.segment_hexikes()
    if not hexikes:
        return
    where = f"{path}.segment_hexikes" if path else "segment_hexikes"
    if not isinstance(pupil, HexSegmentedPupilSpec):
        raise ConfigError(where, f"a {pupil.kind} pupil has no segments")
    count = segment_count(pupil)
    for segment, _, _ in hexikes:
        if segment >= count:
            raise ConfigError(f"{where}.{segment}", f"the pupil has segments 0..{count - 1}")


@dataclass(frozen=True)
class WavefrontSelection:
    """Wavefront modes requested as nuisances: hexike segments (or ``all``) and Nolls, Zernike Nolls."""

    hexike_segments: tuple[int, ...] | Literal["all"] | None
    hexike_nolls: tuple[int, ...]
    zernike_nolls: tuple[int, ...]


def _check_selection(values: Mapping[str, Any], path: str) -> None:
    if values["segment_hexikes"] is None and values["zernikes"] is None:
        raise ConfigError(path, "select at least one family: segment_hexikes or zernikes")
    if values["zernikes"] is not None and 1 in values["zernikes"]["nolls"]:
        raise ConfigError(f"{path}.zernikes.nolls" if path else "zernikes.nolls",
                          "global Noll 1 (piston) leaves the PSF unchanged and is refused")


_NOLLS = ListOf(Integer(min=1), min_length=1, unique=True)

WAVEFRONT_SELECTION_TABLE = Table((
    Key("segment_hexikes", Nullable(Table((
        Key("segments", Union(Text(choices=("all",)), ListOf(Integer(min=0), min_length=1, unique=True)),
            "segment indices, or all: every active segment of the model pupil"),
        Key("nolls", _NOLLS, "hexike Noll indices on each listed segment"),
    ))), "segment hexike modes", default=None),
    Key("zernikes", Nullable(Table((Key("nolls", _NOLLS, "global Zernike Noll indices (Noll 1 is refused)"),))),
        "global Zernike modes", default=None),
), rules=(Rule("at least one family; global Zernike Noll 1 is refused", _check_selection),))


def parse_wavefront_selection(mapping: Mapping[str, Any], path: str) -> WavefrontSelection:
    """A mode selection, read strictly through ``WAVEFRONT_SELECTION_TABLE``; lists sorted."""
    values = WAVEFRONT_SELECTION_TABLE.read(mapping, path)
    hexikes, zernikes = values["segment_hexikes"], values["zernikes"]
    segments: tuple[int, ...] | Literal["all"] | None = None
    if hexikes is not None:
        segments = "all" if hexikes["segments"] == "all" else tuple(sorted(hexikes["segments"]))
    return WavefrontSelection(segments, tuple(sorted(hexikes["nolls"])) if hexikes else (),
                              tuple(sorted(zernikes["nolls"])) if zernikes else ())


def select_modes(selection: WavefrontSelection, pupil: PupilSpec, path: str, *,
                 active_segments: Sequence[int] | None = None) -> tuple[WavefrontMode, ...]:
    """The selected modes in canonical order: hexikes segment-major (segment, then Noll), then Zernikes.

    Without ``active_segments`` (a configuration check, no pupil built) ``all`` means every
    geometric segment. With it, ``all`` means the active segments and a listed segment
    outside them raises as dark.
    """
    modes: list[WavefrontMode] = []
    if selection.hexike_segments is not None:
        where = f"{path}.segment_hexikes" if path else "segment_hexikes"
        if not isinstance(pupil, HexSegmentedPupilSpec):
            raise ConfigError(where, f"a {pupil.kind} pupil has no segments")
        count = segment_count(pupil)
        if selection.hexike_segments == "all":
            segments = tuple(range(count)) if active_segments is None else tuple(active_segments)
        else:
            segments = tuple(selection.hexike_segments)
            for segment in segments:
                if segment >= count:
                    raise ConfigError(f"{where}.segments", f"segment {segment} does not exist; the pupil has "
                                                           f"segments 0..{count - 1}")
                if active_segments is not None and segment not in active_segments:
                    raise ConfigError(f"{where}.segments", f"segment {segment} is dark on this pupil (no "
                                                           "illuminated pixel) and carries no modes")
        modes += [WavefrontMode("segment_hexikes", noll, segment)
                  for segment in segments for noll in selection.hexike_nolls]
    modes += [WavefrontMode("zernikes", noll) for noll in selection.zernike_nolls]
    return tuple(modes)


class WavefrontBasis:
    """The raw HCIPy wavefront modes of one pupil, with memoized bases.

    The global Zernike basis is built once, with the highest Noll requested so far (mode
    samples do not depend on the basis size); one hexike surface is built per hexike mode
    count. Every build runs under ``deterministic_blas()``. A basis is shared by a truth
    provider and every provider derived from it, and is not thread-safe.
    ``reference_wavelength_m`` is the wavelength of the phase route of
    ``aperture_rms_nm``, the shortest wavelength the provider propagates.
    """

    def __init__(self, pupil: Pupil, *, reference_wavelength_m: float) -> None:
        wavelength = float(reference_wavelength_m)
        if not (math.isfinite(wavelength) and wavelength > 0.0):
            raise ValueError(f"reference_wavelength_m must be finite and positive, got {reference_wavelength_m!r}")
        self._pupil = pupil
        self._reference_wavelength_m = wavelength
        self._zernike_basis: Any = None
        self._hexike_surfaces: dict[int, Any] = {}

    @property
    def pupil(self) -> Pupil:
        return self._pupil

    @property
    def reference_wavelength_m(self) -> float:
        return self._reference_wavelength_m

    @property
    def families(self) -> tuple[str, ...]:
        return available_families(self._pupil.spec)

    def select(self, selection: WavefrontSelection) -> tuple[WavefrontMode, ...]:
        """The selected modes on this pupil: ``all`` means the active segments; a dark segment raises."""
        return select_modes(selection, self._pupil.spec, "forecast.nuisances.wavefront.modes",
                            active_segments=self._pupil.active_segments)

    def _zernikes(self, count: int) -> Any:
        if self._zernike_basis is None or len(self._zernike_basis) < count:
            import hcipy

            with deterministic_blas():
                self._zernike_basis = hcipy.make_zernike_basis(count, D=self._pupil.zernike_diameter_m,
                                                               grid=self._pupil.grid)
        return self._zernike_basis

    def _hexike_surface(self, mode_count: int) -> Any:
        if mode_count not in self._hexike_surfaces:
            import hcipy

            pupil = self._pupil
            with deterministic_blas():
                self._hexike_surfaces[mode_count] = hcipy.SegmentedHexikeSurface(
                    segments=pupil.segments, segment_centers=pupil.segment_centres,
                    segment_circum_diameter=pupil.spec.segment_point_to_point_m, pupil_grid=pupil.grid,
                    num_modes=mode_count, hexagon_angle=np.pi / 2)
        return self._hexike_surfaces[mode_count]

    def _surface_for(self, coefficients: WavefrontCoefficients) -> Any:
        """The hexike surface of the evaluation, with every coefficient assigned (surface height, m)."""
        mode_count = coefficients.hexike_mode_count()
        surface = self._hexike_surface(mode_count)
        heights = np.zeros((self._pupil.segment_count, mode_count))
        for segment, noll, value in coefficients.segment_hexikes():
            heights[segment, noll - 1] = value * 1e-9 / 2
        surface.coefficients = heights
        return surface

    def _zernike_phase(self, coefficients: WavefrontCoefficients, wavelength_m: float) -> Any:
        basis = self._zernikes(coefficients.max_zernike_noll())
        phase = self._pupil.grid.zeros()
        for noll, value in coefficients.zernikes():
            phase += (2 * np.pi * (value * 1e-9) / wavelength_m) * basis[noll - 1]
        return phase

    def apply(self, wavefront: Any, coefficients: WavefrontCoefficients) -> Any:
        """``wavefront`` after the hexike surface, then the Zernike phase, as two field multiplications."""
        result = wavefront
        if coefficients.segment_hexikes():
            result = self._surface_for(coefficients)(result)
        if coefficients.zernikes():
            phase = self._zernike_phase(coefficients, result.wavelength)
            result = result.copy()
            result.electric_field *= np.exp(1j * np.array(phase))
        return result

    def _opd_m(self, coefficients: WavefrontCoefficients) -> np.ndarray:
        """The OPD on the pupil grid (flat, metres) through the phase at the reference wavelength."""
        wavelength = self._reference_wavelength_m
        to_opd = wavelength / (2.0 * np.pi)
        opd = np.zeros(self._pupil.grid.size)
        if coefficients.segment_hexikes():
            opd = opd + np.asarray(self._surface_for(coefficients).phase_for(wavelength)) * to_opd
        if coefficients.zernikes():
            opd = opd + np.asarray(self._zernike_phase(coefficients, wavelength)) * to_opd
        return opd

    def opd_nm(self, coefficients: WavefrontCoefficients) -> np.ndarray:
        """The OPD in nm on the pupil grid, shape ``(pixels, pixels)``."""
        return (self._opd_m(coefficients) * 1e9).reshape(self._pupil.grid.shape)

    def aperture_rms_nm(self, coefficients: WavefrontCoefficients) -> float:
        """Piston-removed OPD RMS over the illuminated pupil, in nm."""
        values = self._opd_m(coefficients)[self._pupil.illuminated_mask]
        values = values - np.mean(values)
        return float(np.sqrt(np.mean(values ** 2)) * 1e9)

    def zernike_samples(self, nolls: Sequence[int], mask: np.ndarray) -> np.ndarray:
        """Raw global Zernike modes on ``mask``, one column per Noll, shape ``(n_mask, len(nolls))``."""
        basis = self._zernikes(max(nolls))
        return np.column_stack([np.asarray(basis[noll - 1])[mask] for noll in nolls])

    def hexike_samples(self, segment: int, nolls: Sequence[int], mask: np.ndarray) -> np.ndarray:
        """Raw hexike modes of one segment on ``mask``, one column per Noll (unit coefficient of OPD)."""
        mode_count = max(nolls)
        surface = self._hexike_surface(mode_count)
        columns = []
        for noll in nolls:
            heights = np.zeros((self._pupil.segment_count, mode_count))
            heights[segment, noll - 1] = 0.5
            surface.coefficients = heights
            columns.append(np.asarray(surface.opd)[mask])
        return np.column_stack(columns)
