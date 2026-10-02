"""Pure node geometry for strong-lensing sensitivity forecasts.

Layouts describe which sky positions are evaluated; they do not depend on a
renderer, statistical model, telescope, or execution backend. Node order and
closed-boundary conventions are shared by reference and accelerated engines.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Tuple

import numpy as np


@dataclass(frozen=True)
class FisherGridLayout:
    """Node layout for a 2D sensitivity grid map.

    ``positions_yx`` lists only the evaluated nodes in row-major order and
    ``node_indices`` gives each one's ``(i, j)`` into the 2D arrays.
    """

    y_coords: np.ndarray
    x_coords: np.ndarray
    spacing_arcsec: float
    centre_yx: Tuple[float, float]
    evaluated_mask: np.ndarray
    positions_yx: Tuple[Tuple[float, float], ...]
    node_indices: Tuple[Tuple[int, int], ...]


@dataclass(frozen=True)
class SpatialSelection:
    """Reusable aperture-plus-perimeter selection on a full square lattice.

    ``selected_mask_2d`` describes exactly the nodes evaluated by the sparse
    path.  Values outside it are absent from the corresponding compact result;
    they are not represented as zero or as a partially populated grid map.
    The full coordinate arrays remain available so the selection is tied to
    the original square geometry used to construct the JAX radial engine.
    """

    y_coords: np.ndarray
    x_coords: np.ndarray
    spacing_arcsec: float
    grid_centre_yx: Tuple[float, float]
    aperture_centre_yx: Tuple[float, float]
    aperture_radius_arcsec: float
    aperture_mask_2d: np.ndarray
    perimeter_mask_2d: np.ndarray
    selected_mask_2d: np.ndarray
    positions_yx: Tuple[Tuple[float, float], ...]
    node_indices: Tuple[Tuple[int, int], ...]

    @property
    def full_grid_node_count(self) -> int:
        return int(self.y_coords.size * self.x_coords.size)

    @property
    def aperture_node_count(self) -> int:
        return int(np.count_nonzero(self.aperture_mask_2d))

    @property
    def perimeter_node_count(self) -> int:
        return int(np.count_nonzero(self.perimeter_mask_2d))

    @property
    def selected_node_count(self) -> int:
        return len(self.positions_yx)


def build_grid_layout(
    grid_config: Dict[str, Any],
    centre_yx: Tuple[float, float],
) -> FisherGridLayout:
    """Build a square lattice, optionally selecting a closed radial annulus.

    Coordinates are in arcseconds. ``centre_yx`` is supplied explicitly by
    the caller; geometry planning has no dependency on a scene configuration.
    Evaluated positions retain row-major order, including for annulus layouts.
    """
    spacing = float(grid_config["spacing_arcsec"])
    half_width = float(grid_config["half_width_arcsec"])
    if spacing <= 0.0 or not np.isfinite(spacing):
        raise ValueError(
            "modeling.fisher.map.grid.spacing_arcsec must be positive and finite."
        )
    if half_width < spacing or not np.isfinite(half_width):
        raise ValueError(
            "modeling.fisher.map.grid.half_width_arcsec must be finite and >= spacing_arcsec."
        )

    # Candidate geometry is defined by the truth/data lens centre.
    lens_centre = centre_yx
    centre_y = float(lens_centre[0])
    centre_x = float(lens_centre[1])

    n_half = int(np.floor(half_width / spacing + 1.0e-9))
    offsets = spacing * np.arange(-n_half, n_half + 1, dtype=float)
    y_coords = centre_y + offsets
    x_coords = centre_x + offsets

    yy = y_coords[:, None]
    xx = x_coords[None, :]
    radius = np.hypot(yy - centre_y, xx - centre_x)

    annulus = grid_config.get("annulus")
    if annulus is None:
        evaluated_mask = np.ones((y_coords.size, x_coords.size), dtype=bool)
    else:
        r_min = float(annulus["r_min_arcsec"])
        r_max = float(annulus["r_max_arcsec"])
        evaluated_mask = (radius >= r_min) & (radius <= r_max)
        if not np.any(evaluated_mask):
            raise ValueError(
                "modeling.fisher.map.grid.annulus selects no grid nodes; "
                "widen the annulus or refine the spacing."
            )

    node_indices = []
    positions = []
    for i in range(y_coords.size):
        for j in range(x_coords.size):
            if evaluated_mask[i, j]:
                node_indices.append((i, j))
                positions.append((float(y_coords[i]), float(x_coords[j])))

    return FisherGridLayout(
        y_coords=y_coords,
        x_coords=x_coords,
        spacing_arcsec=spacing,
        centre_yx=(centre_y, centre_x),
        evaluated_mask=evaluated_mask,
        positions_yx=tuple(positions),
        node_indices=tuple(node_indices),
    )


def select_aperture_and_perimeter(
    layout: FisherGridLayout,
    centre_arcsec: Tuple[float, float],
    radius_arcsec: float,
) -> SpatialSelection:
    """Select a closed aperture and all four edges of the original square.

    The full lattice is retained for radial interpolation bounds and clipping
    diagnostics. Restricted annulus layouts cannot supply this contract.
    Arrays in the returned selection cannot be mutated, including by toggling
    their NumPy writeable flag. A cached selection stays tied to its geometry.
    """
    if not np.all(layout.evaluated_mask):
        raise ValueError("Aperture selection requires a full square grid layout.")
    centre = np.asarray(centre_arcsec, dtype=float)
    if centre.shape != (2,) or not np.all(np.isfinite(centre)):
        raise ValueError(
            "The aperture centre must contain two finite coordinates."
        )
    radius = float(radius_arcsec)
    if not np.isfinite(radius) or radius <= 0.0:
        raise ValueError("The aperture radius must be positive and finite.")

    offsets_y = layout.y_coords[:, None] - centre[0]
    offsets_x = layout.x_coords[None, :] - centre[1]
    aperture_mask = offsets_y**2 + offsets_x**2 <= radius**2
    if not np.any(aperture_mask):
        raise ValueError(
            f"The grid map holds no node inside the requested aperture of "
            f"radius {radius} arcsec about {tuple(centre)}"
        )

    perimeter_mask = np.zeros(layout.evaluated_mask.shape, dtype=bool)
    perimeter_mask[0, :] = True
    perimeter_mask[-1, :] = True
    perimeter_mask[:, 0] = True
    perimeter_mask[:, -1] = True
    selected_mask = aperture_mask | perimeter_mask

    node_array = np.argwhere(selected_mask)
    node_indices = tuple((int(i), int(j)) for i, j in node_array)
    positions = tuple(
        (float(layout.y_coords[i]), float(layout.x_coords[j])) for i, j in node_indices
    )

    def _readonly_copy(values: np.ndarray) -> np.ndarray:
        contiguous = np.ascontiguousarray(values)
        return np.frombuffer(contiguous.tobytes(), dtype=contiguous.dtype).reshape(
            contiguous.shape
        )

    selection = SpatialSelection(
        y_coords=_readonly_copy(layout.y_coords),
        x_coords=_readonly_copy(layout.x_coords),
        spacing_arcsec=layout.spacing_arcsec,
        grid_centre_yx=layout.centre_yx,
        aperture_centre_yx=(float(centre[0]), float(centre[1])),
        aperture_radius_arcsec=radius,
        aperture_mask_2d=_readonly_copy(aperture_mask),
        perimeter_mask_2d=_readonly_copy(perimeter_mask),
        selected_mask_2d=_readonly_copy(selected_mask),
        positions_yx=positions,
        node_indices=node_indices,
    )
    return selection
