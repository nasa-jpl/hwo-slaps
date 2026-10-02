"""Geometry contracts independent of the renderer and instrument stack."""

import numpy as np
import pytest

from hwoslaps.modeling.fisher_geometry import (
    build_grid_layout,
    select_aperture_and_perimeter,
)


def test_translated_lattice_keeps_coordinates_and_row_major_positions():
    layout = build_grid_layout(
        {"spacing_arcsec": 0.5, "half_width_arcsec": 1.1}, (0.25, -0.25)
    )
    np.testing.assert_array_equal(layout.y_coords, [-0.75, -0.25, 0.25, 0.75, 1.25])
    np.testing.assert_array_equal(layout.x_coords, [-1.25, -0.75, -0.25, 0.25, 0.75])
    assert layout.node_indices[:6] == ((0, 0), (0, 1), (0, 2), (0, 3), (0, 4), (1, 0))
    assert layout.positions_yx[:2] == ((-0.75, -1.25), (-0.75, -0.75))
    assert layout.positions_yx[-1] == (1.25, 0.75)
    assert len(layout.positions_yx) == 25


def test_annulus_uses_closed_boundaries_and_retains_position_indices():
    layout = build_grid_layout(
        {
            "spacing_arcsec": 1.0,
            "half_width_arcsec": 1.0,
            "annulus": {"r_min_arcsec": 1.0, "r_max_arcsec": 1.0},
        },
        (0.0, 0.0),
    )
    assert layout.node_indices == ((0, 1), (1, 0), (1, 2), (2, 1))
    assert layout.positions_yx == ((-1.0, 0.0), (0.0, -1.0), (0.0, 1.0), (1.0, 0.0))
    with pytest.raises(ValueError, match="full square"):
        select_aperture_and_perimeter(layout, (0.0, 0.0), 1.0)


def test_compact_selection_contains_aperture_and_original_perimeter_once():
    layout = build_grid_layout(
        {"spacing_arcsec": 1.0, "half_width_arcsec": 2.0}, (0.0, 0.0)
    )
    selection = select_aperture_and_perimeter(layout, (0.0, 0.0), 1.0)
    assert selection.aperture_node_count == 5
    assert selection.perimeter_node_count == 16
    assert selection.selected_node_count == 21
    assert selection.full_grid_node_count == 25
    assert (1, 1) not in selection.node_indices
    assert (0, 0) in selection.node_indices
    assert selection.node_indices == tuple(sorted(set(selection.node_indices)))
    np.testing.assert_array_equal(selection.y_coords, layout.y_coords)
    for values in (
        selection.y_coords,
        selection.x_coords,
        selection.aperture_mask_2d,
        selection.perimeter_mask_2d,
        selection.selected_mask_2d,
    ):
        with pytest.raises(ValueError):
            values.setflags(write=True)
    layout.y_coords[0] = 999.0
    assert selection.y_coords[0] == -2.0


def test_aperture_boundary_uses_squared_distance_without_tolerance():
    layout = build_grid_layout(
        {"spacing_arcsec": 1.0, "half_width_arcsec": 1.0}, (0.0, 0.0)
    )
    below = select_aperture_and_perimeter(layout, (0.0, 0.0), np.nextafter(1.0, 0.0))
    exact = select_aperture_and_perimeter(layout, (0.0, 0.0), 1.0)
    assert below.aperture_node_count == 1
    assert exact.aperture_node_count == 5


@pytest.mark.parametrize("radius", [0.0, -1.0, np.inf, np.nan])
def test_invalid_aperture_radius_is_rejected(radius):
    layout = build_grid_layout(
        {"spacing_arcsec": 1.0, "half_width_arcsec": 1.0}, (0.0, 0.0)
    )
    with pytest.raises(ValueError, match="positive and finite"):
        select_aperture_and_perimeter(layout, (0.0, 0.0), radius)
