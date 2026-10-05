"""Where a subhalo hypothesis is evaluated, with the geometry the reductions need.

Coordinates are ``(y, x)`` in arcseconds. Layouts:

- ``grid``: a square lattice about the lens centre with ``n_half =
  floor(half_width / spacing + 1e-9)`` and offsets ``spacing * (-n_half, ...,
  n_half)`` per axis, nodes in row-major order, optionally only those with
  radius in the closed annulus ``[inner, outer]``. Each node carries the cell
  area ``spacing**2`` and a boundary flag: an evaluated node is on the
  boundary when one of its four lattice neighbours is outside the lattice or
  not evaluated, which on a full square is rows and columns 0 and -1. A subset
  keeps the flags only while it holds every boundary node of its layout, so a
  reduction never reads an edge it did not evaluate as unclipped.
- ``ring``: ``count`` positions at angles ``360 k / count`` degrees, measured
  from +x toward +y, at radius ``radius + offset`` about the lens centre.
- ``explicit``: given positions, finite and without duplicate rows.

The domain radius is the largest distance from the centre of any node the
layout can evaluate (every node of the square lattice for a grid). It fixes
the engine's radial domain, and subsets keep it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike

__all__ = ["GridIndex", "PositionSet", "explicit_positions", "grid_positions", "ring_positions"]

_LAYOUT_KINDS = ("grid", "ring", "explicit")


def _frozen(values: np.ndarray) -> np.ndarray:
    copy = np.array(values, order="C")
    copy.setflags(write=False)
    return copy


def _centre(centre_yx: ArrayLike) -> tuple[float, float]:
    centre = np.asarray(centre_yx, dtype=float)
    if centre.shape != (2,) or not np.all(np.isfinite(centre)):
        raise ValueError(f"a centre must be two finite coordinates (y, x), got {centre_yx!r}")
    return float(centre[0]), float(centre[1])


def _positive(value: float, what: str) -> float:
    number = float(value)
    if isinstance(value, bool) or not np.isfinite(number) or number <= 0.0:
        raise ValueError(f"{what} must be a positive finite number, got {value!r}")
    return number


def _radii(positions_yx: np.ndarray, centre_yx: tuple[float, float]) -> np.ndarray:
    offsets = positions_yx - np.asarray(centre_yx, dtype=float)[None, :]
    return np.hypot(offsets[:, 0], offsets[:, 1])


@dataclass(frozen=True, eq=False)
class GridIndex:
    """The square lattice of a grid layout and the lattice index ``(i, j)`` of each position."""

    y_coords: np.ndarray
    x_coords: np.ndarray
    indices: np.ndarray
    spacing_arcsec: float

    def __post_init__(self) -> None:
        for name in ("y_coords", "x_coords"):
            coords = np.asarray(getattr(self, name), dtype=float)
            if coords.ndim != 1 or coords.size == 0 or not np.all(np.isfinite(coords)) \
                    or np.any(np.diff(coords) <= 0.0):
                raise ValueError(f"{name} must be a non-empty, finite, increasing vector")
            object.__setattr__(self, name, _frozen(coords))
        indices = np.asarray(self.indices)
        if indices.ndim != 2 or indices.shape[1] != 2 or indices.shape[0] == 0 or indices.dtype.kind not in "iu":
            raise ValueError(f"indices must be a non-empty integer (n, 2) array, got shape {indices.shape}")
        if np.any(indices < 0) or np.any(indices >= np.asarray(self.shape)[None, :]):
            raise ValueError(f"indices must lie inside the {self.shape} lattice")
        if np.unique(indices, axis=0).shape[0] != indices.shape[0]:
            raise ValueError("indices must not repeat a lattice node")
        object.__setattr__(self, "indices", _frozen(indices.astype(np.int64)))
        object.__setattr__(self, "spacing_arcsec", _positive(self.spacing_arcsec, "spacing_arcsec"))

    @property
    def shape(self) -> tuple[int, int]:
        """Lattice shape ``(ny, nx)``."""
        return int(self.y_coords.size), int(self.x_coords.size)

    def to_image(self, values: ArrayLike) -> np.ndarray:
        """A ``(ny, nx)`` image of one value per position, NaN at lattice nodes not evaluated."""
        array = np.asarray(values, dtype=float)
        if array.shape != (self.indices.shape[0],):
            raise ValueError(f"values must have shape ({self.indices.shape[0]},), got {array.shape}")
        image = np.full(self.shape, np.nan)
        image[self.indices[:, 0], self.indices[:, 1]] = array
        return image


@dataclass(frozen=True, eq=False)
class PositionSet:
    """Evaluated positions with their layout geometry; arrays are read-only.

    Grid sets carry cell areas and the lattice index, and boundary flags while
    they hold every boundary node of their layout (``boundary`` is None for a
    subset that dropped one); ring and explicit sets carry none of them.
    """

    kind: Literal["grid", "ring", "explicit"]
    positions_yx: np.ndarray
    centre_yx: tuple[float, float]
    domain_radius_arcsec: float
    cell_areas_arcsec2: np.ndarray | None
    boundary: np.ndarray | None
    grid: GridIndex | None

    def __post_init__(self) -> None:
        if self.kind not in _LAYOUT_KINDS:
            raise ValueError(f"kind must be one of {_LAYOUT_KINDS}, got {self.kind!r}")
        positions = _position_rows(self.positions_yx)
        count = positions.shape[0]
        centre = _centre(self.centre_yx)
        domain = float(self.domain_radius_arcsec)
        if not np.isfinite(domain) or domain < float(np.max(_radii(positions, centre))):
            raise ValueError(f"domain_radius_arcsec {self.domain_radius_arcsec!r} does not contain every position")
        if self.kind != "grid":
            if any(value is not None for value in (self.cell_areas_arcsec2, self.boundary, self.grid)):
                raise ValueError(f"a {self.kind} set has no cell areas, boundary or lattice")
        else:
            lattice = self.grid
            if not isinstance(lattice, GridIndex) or lattice.indices.shape[0] != count:
                raise ValueError(f"grid must be a GridIndex with {count} lattice indices")
            nodes = np.column_stack((lattice.y_coords[lattice.indices[:, 0]],
                                     lattice.x_coords[lattice.indices[:, 1]]))
            if not np.array_equal(nodes, positions):
                raise ValueError("grid positions must be the lattice nodes their indices name")
            if self.cell_areas_arcsec2 is None:
                raise ValueError("a grid set needs cell areas")
            areas = np.asarray(self.cell_areas_arcsec2, dtype=float)
            if areas.shape != (count,) or np.any(areas != lattice.spacing_arcsec ** 2):
                raise ValueError(f"cell_areas_arcsec2 must hold {count} cell areas equal to the lattice "
                                 "spacing_arcsec**2")
            if self.boundary is not None:
                boundary = np.asarray(self.boundary)
                if boundary.dtype != bool or boundary.shape != (count,):
                    raise ValueError(f"boundary must be a boolean vector of length {count}")
                object.__setattr__(self, "boundary", _frozen(boundary))
            object.__setattr__(self, "cell_areas_arcsec2", _frozen(areas))
        object.__setattr__(self, "positions_yx", _frozen(positions))
        object.__setattr__(self, "centre_yx", centre)
        object.__setattr__(self, "domain_radius_arcsec", domain)

    def __len__(self) -> int:
        return int(self.positions_yx.shape[0])

    def within(self, centre_yx: tuple[float, float], radius_arcsec: float) -> np.ndarray:
        """Positions in the closed disc ``dy**2 + dx**2 <= r**2`` (squared distances, no tolerance)."""
        cy, cx = _centre(centre_yx)
        radius = _positive(radius_arcsec, "radius_arcsec")
        dy = self.positions_yx[:, 0] - cy
        dx = self.positions_yx[:, 1] - cx
        return dy**2 + dx**2 <= radius**2

    def select(self, keep: ArrayLike) -> PositionSet:
        """The positions where ``keep`` is true, with their areas, boundary flags and lattice rows.

        The centre and the domain radius are kept, so a subset evaluates on the
        radial domain of the full layout. The boundary flags are kept only when
        ``keep`` holds every boundary node.
        """
        mask = np.asarray(keep)
        if mask.dtype != bool or mask.shape != (len(self),):
            raise ValueError(f"keep must be a boolean vector of length {len(self)}")
        if not np.any(mask):
            raise ValueError("keep selects no position")
        lattice = None
        if self.grid is not None:
            lattice = GridIndex(self.grid.y_coords, self.grid.x_coords, self.grid.indices[mask],
                                self.grid.spacing_arcsec)
        boundary = None
        if self.boundary is not None and not np.any(self.boundary & ~mask):
            boundary = self.boundary[mask]
        return PositionSet(
            kind=self.kind,
            positions_yx=self.positions_yx[mask],
            centre_yx=self.centre_yx,
            domain_radius_arcsec=self.domain_radius_arcsec,
            cell_areas_arcsec2=None if self.cell_areas_arcsec2 is None else self.cell_areas_arcsec2[mask],
            boundary=boundary,
            grid=lattice,
        )

    def aperture(self, centre_yx: tuple[float, float], radius_arcsec: float, *,
                 include_boundary: bool) -> PositionSet:
        """The positions in the closed disc, plus the boundary nodes when asked (a sparse ladder run)."""
        inside = self.within(centre_yx, radius_arcsec)
        if not np.any(inside):
            raise ValueError(f"no position lies inside the aperture of radius {radius_arcsec} arcsec "
                             f"about {tuple(centre_yx)}")
        if include_boundary:
            if self.boundary is None:
                raise ValueError(f"this {self.kind} set has no boundary flags to include")
            inside = inside | self.boundary
        return self.select(inside)


def _position_rows(positions_yx: ArrayLike) -> np.ndarray:
    raw = np.asarray(positions_yx)
    if raw.dtype.kind not in "iuf":
        raise ValueError(f"positions must be numeric, got dtype {raw.dtype}")
    positions = raw.astype(float)
    if positions.ndim != 2 or positions.shape[1] != 2 or positions.shape[0] == 0:
        raise ValueError(f"positions must be a non-empty (n, 2) array of (y, x), got shape {positions.shape}")
    if not np.all(np.isfinite(positions)):
        raise ValueError("positions must be finite")
    if np.unique(positions, axis=0).shape[0] != positions.shape[0]:
        raise ValueError("positions must not repeat a row")
    return positions


def grid_positions(centre_yx: tuple[float, float], *, spacing_arcsec: float, half_width_arcsec: float,
                   annulus: tuple[float, float] | None) -> PositionSet:
    """The square lattice about ``centre_yx``, optionally only the nodes in a closed annulus ``(inner, outer)``."""
    cy, cx = _centre(centre_yx)
    spacing = _positive(spacing_arcsec, "spacing_arcsec")
    half_width = _positive(half_width_arcsec, "half_width_arcsec")
    if half_width < spacing:
        raise ValueError(f"half_width_arcsec {half_width_arcsec!r} must be at least "
                         f"spacing_arcsec {spacing_arcsec!r}")
    n_half = int(np.floor(half_width / spacing + 1.0e-9))
    offsets = spacing * np.arange(-n_half, n_half + 1, dtype=float)
    y_coords = cy + offsets
    x_coords = cx + offsets
    yy, xx = np.meshgrid(y_coords, x_coords, indexing="ij")
    if annulus is None:
        evaluated = np.ones(yy.shape, dtype=bool)
    else:
        inner, outer = (float(value) for value in annulus)
        if not (np.isfinite(inner) and np.isfinite(outer) and 0.0 <= inner < outer):
            raise ValueError(f"the annulus needs finite radii with 0 <= inner < outer, got {annulus!r}")
        radius = np.hypot(yy - cy, xx - cx)
        evaluated = (radius >= inner) & (radius <= outer)
        if not np.any(evaluated):
            raise ValueError(f"the annulus {annulus!r} selects no lattice node; widen it or refine the spacing")
    lattice_nodes = np.column_stack((yy.ravel(), xx.ravel()))
    padded = np.pad(evaluated, 1, constant_values=False)
    interior = padded[:-2, 1:-1] & padded[2:, 1:-1] & padded[1:-1, :-2] & padded[1:-1, 2:]
    count = int(np.count_nonzero(evaluated))
    return PositionSet(
        kind="grid",
        positions_yx=np.column_stack((yy[evaluated], xx[evaluated])),
        centre_yx=(cy, cx),
        domain_radius_arcsec=float(np.max(_radii(lattice_nodes, (cy, cx)))),
        cell_areas_arcsec2=np.full(count, spacing ** 2),
        boundary=(evaluated & ~interior)[evaluated],
        grid=GridIndex(y_coords, x_coords, np.argwhere(evaluated), spacing),
    )


def ring_positions(centre_yx: tuple[float, float], *, count: int, radius_arcsec: float,
                   offset_arcsec: float) -> PositionSet:
    """``count`` positions on the circle of radius ``radius + offset`` about ``centre_yx``."""
    cy, cx = _centre(centre_yx)
    if isinstance(count, bool) or not isinstance(count, (int, np.integer)) or count < 1:
        raise ValueError(f"count must be a positive integer, got {count!r}")
    radius = _positive(radius_arcsec, "radius_arcsec")
    offset = float(offset_arcsec)
    if not np.isfinite(offset) or radius + offset <= 0.0:
        raise ValueError(f"radius_arcsec + offset_arcsec must be positive and finite, got "
                         f"{radius_arcsec!r} + {offset_arcsec!r}")
    angles = np.deg2rad(np.linspace(0.0, 360.0, int(count), endpoint=False))
    ring = radius + offset
    positions = np.column_stack((cy + ring * np.sin(angles), cx + ring * np.cos(angles)))
    return _free_layout("ring", positions, (cy, cx))


def explicit_positions(positions_yx: ArrayLike, centre_yx: tuple[float, float]) -> PositionSet:
    """Given positions, finite and without duplicate rows, about the lens centre ``centre_yx``."""
    return _free_layout("explicit", _position_rows(positions_yx), _centre(centre_yx))


def _free_layout(kind: Literal["ring", "explicit"], positions: np.ndarray,
                 centre: tuple[float, float]) -> PositionSet:
    return PositionSet(kind=kind, positions_yx=positions, centre_yx=centre,
                       domain_radius_arcsec=float(np.max(_radii(positions, centre))),
                       cell_areas_arcsec2=None, boundary=None, grid=None)
