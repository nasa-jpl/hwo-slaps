"""Evaluation layouts of fisher.positions.

Oracles: hand coordinate and lattice-index lists, the 41621de ladder perimeter
(rows and columns 0 and -1 of the full square), squared-distance apertures
evaluated by hand, and trigonometry on the ring.
"""

import pickle

import numpy as np
import pytest

from hwoslaps.fisher.positions import GridIndex, PositionSet, explicit_positions, grid_positions, ring_positions


def full_grid(spacing=1.0, half_width=2.0, centre=(0.0, 0.0)):
    return grid_positions(centre, spacing_arcsec=spacing, half_width_arcsec=half_width, annulus=None)


@pytest.mark.parametrize("centre, spacing, half_width, coords_y, coords_x", [
    ((0.25, -0.25), 0.5, 1.1, [-0.75, -0.25, 0.25, 0.75, 1.25], [-1.25, -0.75, -0.25, 0.25, 0.75]),
    ((0.0, 0.0), 1.0, 2.9999999999, [-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0], [-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0]),
    ((1.0, 2.0), 0.5, 0.5, [0.5, 1.0, 1.5], [1.5, 2.0, 2.5]),
], ids=["translated", "floor-tolerance", "half-width-equals-spacing"])
def test_grid_lattice_is_row_major_about_the_lens_centre(centre, spacing, half_width, coords_y, coords_x):
    layout = grid_positions(centre, spacing_arcsec=spacing, half_width_arcsec=half_width, annulus=None)
    assert layout.kind == "grid"
    np.testing.assert_array_equal(layout.grid.y_coords, coords_y)
    np.testing.assert_array_equal(layout.grid.x_coords, coords_x)
    count = len(coords_y) * len(coords_x)
    expected = [(y, x) for y in coords_y for x in coords_x]
    np.testing.assert_array_equal(layout.positions_yx, expected)
    np.testing.assert_array_equal(layout.grid.indices[: len(coords_x) + 1],
                                  [(0, j) for j in range(len(coords_x))] + [(1, 0)])
    np.testing.assert_array_equal(layout.cell_areas_arcsec2, np.full(count, float(spacing) ** 2))
    assert layout.grid.spacing_arcsec == spacing
    assert layout.centre_yx == centre
    corner = max(float(np.hypot(y - centre[0], x - centre[1])) for y, x in expected)
    assert layout.domain_radius_arcsec == corner


def test_annulus_layout_keeps_closed_radii_and_lattice_indices():
    axial = grid_positions((0.0, 0.0), spacing_arcsec=1.0, half_width_arcsec=1.0, annulus=(1.0, 1.2))
    np.testing.assert_array_equal(axial.grid.indices, [(0, 1), (1, 0), (1, 2), (2, 1)])
    np.testing.assert_array_equal(axial.positions_yx, [(-1.0, 0.0), (0.0, -1.0), (0.0, 1.0), (1.0, 0.0)])
    outer_closed = grid_positions((0.0, 0.0), spacing_arcsec=1.0, half_width_arcsec=1.0, annulus=(0.5, 1.0))
    np.testing.assert_array_equal(outer_closed.grid.indices, axial.grid.indices)
    assert axial.domain_radius_arcsec == np.hypot(1.0, 1.0)
    image = axial.grid.to_image(np.array([1.0, 2.0, 3.0, 4.0]))
    np.testing.assert_array_equal(image, [[np.nan, 1.0, np.nan], [2.0, np.nan, 3.0], [np.nan, 4.0, np.nan]])
    ring = grid_positions((0.0, 0.0), spacing_arcsec=0.1, half_width_arcsec=0.2, annulus=(0.05, 0.15))
    np.testing.assert_array_equal(ring.grid.indices, [(1, 1), (1, 2), (1, 3), (2, 1), (2, 3), (3, 1), (3, 2), (3, 3)])
    with pytest.raises(ValueError, match="selects no lattice node"):
        grid_positions((0.0, 0.0), spacing_arcsec=0.1, half_width_arcsec=0.2, annulus=(0.01, 0.02))


@pytest.mark.parametrize("half_width, annulus, match", [
    (0.999, None, "must be at least spacing_arcsec"),
    (1.0, (-0.1, 1.0), "0 <= inner < outer"),
    (1.0, (1.0, 1.0), "0 <= inner < outer"),
    (1.0, (1.5, 1.0), "0 <= inner < outer"),
    (1.0, (np.nan, 1.0), "0 <= inner < outer"),
], ids=["half-width-below-spacing", "negative-inner", "equal-radii", "inverted-radii", "non-finite-inner"])
def test_grid_layout_refuses_invalid_geometry(half_width, annulus, match):
    with pytest.raises(ValueError, match=match):
        grid_positions((0.0, 0.0), spacing_arcsec=1.0, half_width_arcsec=half_width, annulus=annulus)


@pytest.mark.parametrize("annulus, expected", [
    (None, [(i, j) for i in range(5) for j in range(5) if i in (0, 4) or j in (0, 4)]),
    ((0.0, 1.5), [(1, 1), (1, 2), (1, 3), (2, 1), (2, 3), (3, 1), (3, 2), (3, 3)]),
], ids=["full-square", "annulus"])
def test_boundary_marks_nodes_next_to_the_lattice_edge(annulus, expected):
    layout = grid_positions((0.0, 0.0), spacing_arcsec=1.0, half_width_arcsec=2.0, annulus=annulus)
    flagged = [tuple(index) for index in layout.grid.indices[layout.boundary]]
    assert flagged == expected


def test_selected_and_aperture_sets_keep_geometry():
    layout = full_grid()
    exact = layout.aperture((0.0, 0.0), 1.0, include_boundary=False)
    assert len(exact) == 5
    assert exact.domain_radius_arcsec == layout.domain_radius_arcsec
    assert exact.boundary is None
    with pytest.raises(ValueError, match="no boundary flags"):
        exact.aperture((0.0, 0.0), 1.0, include_boundary=True)
    assert len(layout.aperture((0.0, 0.0), np.nextafter(1.0, 0.0), include_boundary=False)) == 1
    sparse = layout.aperture((0.0, 0.0), 1.0, include_boundary=True)
    assert len(sparse) == 21
    indices = [tuple(index) for index in sparse.grid.indices]
    assert (1, 1) not in indices and (0, 0) in indices and indices == sorted(indices)
    np.testing.assert_array_equal(sparse.positions_yx, sparse.grid.indices - 2.0)
    assert np.count_nonzero(sparse.boundary) == 16
    np.testing.assert_array_equal(sparse.cell_areas_arcsec2, 1.0)
    assert sparse.domain_radius_arcsec == layout.domain_radius_arcsec == np.hypot(2.0, 2.0)
    assert sparse.centre_yx == layout.centre_yx
    np.testing.assert_array_equal(sparse.grid.y_coords, layout.grid.y_coords)
    restored = pickle.loads(pickle.dumps(sparse))
    np.testing.assert_array_equal(restored.grid.indices, sparse.grid.indices)
    np.testing.assert_array_equal(restored.boundary, sparse.boundary)
    for copy in (sparse, restored):
        for values in (copy.positions_yx, copy.cell_areas_arcsec2, copy.boundary, copy.grid.y_coords,
                       copy.grid.x_coords, copy.grid.indices):
            with pytest.raises(ValueError, match="read-only"):
                values[0] = values[1]
    given = np.array([[0.5, 0.5], [1.0, 1.0]])
    explicit = explicit_positions(given, (0.0, 0.0))
    given[0, 0] = 9.0
    assert explicit.positions_yx[0, 0] == 0.5
    translated = grid_positions((0.25, -0.25), spacing_arcsec=0.5, half_width_arcsec=1.0, annulus=None)
    off_centre = translated.within((0.25, 0.25), 0.75).reshape(5, 5)
    expected = np.zeros((5, 5), dtype=bool)
    expected[1:4, 2:5] = True
    np.testing.assert_array_equal(off_centre, expected)
    assert len(translated.aperture((0.25, 0.25), 0.75, include_boundary=True)) == 22
    with pytest.raises(ValueError, match="no position lies inside"):
        layout.aperture((10.0, 10.0), 0.1, include_boundary=True)
    for radius in (0.0, -1.0, np.inf, np.nan):
        with pytest.raises(ValueError, match="radius_arcsec must be a positive finite number"):
            layout.within((0.0, 0.0), radius)
    with pytest.raises(ValueError, match="selects no position"):
        layout.select(np.zeros(25, dtype=bool))


def test_ring_positions_lie_on_the_offset_circle_about_the_lens_centre():
    ring = ring_positions((0.3, -0.2), count=4, radius_arcsec=1.0, offset_arcsec=0.05)
    np.testing.assert_allclose(ring.positions_yx, [(0.3, 0.85), (1.35, -0.2), (0.3, -1.25), (-0.75, -0.2)],
                               rtol=0.0, atol=1e-12)
    assert ring.kind == "ring" and ring.boundary is None and ring.cell_areas_arcsec2 is None and ring.grid is None
    assert ring.domain_radius_arcsec == pytest.approx(1.05, rel=1e-14)
    many = ring_positions((0.3, -0.2), count=36, radius_arcsec=0.6, offset_arcsec=0.0)
    radii = np.hypot(many.positions_yx[:, 0] - 0.3, many.positions_yx[:, 1] + 0.2)
    np.testing.assert_allclose(radii, 0.6, rtol=1e-14)
    with pytest.raises(ValueError, match="must be positive"):
        ring_positions((0.0, 0.0), count=4, radius_arcsec=0.5, offset_arcsec=-0.5)
    with pytest.raises(ValueError, match="count must be a positive integer"):
        ring_positions((0.0, 0.0), count=True, radius_arcsec=0.5, offset_arcsec=0.0)


@pytest.mark.parametrize("positions, match", [
    ([[0.1, 0.2], [0.3, 0.4], [0.1, 0.2]], "repeat a row"),
    ([[0.0, -0.0], [0.0, 0.0]], "repeat a row"),
    ([[0.1, np.nan]], "finite"),
    (np.empty((0, 2)), "non-empty"),
    ([[0.1, 0.2, 0.3]], r"\(n, 2\)"),
    ([[True, False]], "numeric"),
    ([["0.1", "0.2"]], "numeric"),
], ids=["duplicate", "signed-zero-duplicate", "non-finite", "empty", "three-columns", "boolean", "text"])
def test_explicit_positions_reject_duplicates_and_non_finite_rows(positions, match):
    with pytest.raises(ValueError, match=match):
        explicit_positions(positions, (0.0, 0.0))


def test_position_sets_hold_every_position_inside_their_domain():
    explicit = explicit_positions([[3.0, 4.0], [-1.5, 2.0]], (0.0, 0.0))
    assert explicit.domain_radius_arcsec == 5.0
    with pytest.raises(ValueError, match="does not contain every position"):
        PositionSet("explicit", explicit.positions_yx, (0.0, 0.0), 4.99, None, None, None)
    layout = full_grid()
    lattice = GridIndex(layout.grid.y_coords, layout.grid.x_coords, layout.grid.indices[::-1], 1.0)
    with pytest.raises(ValueError, match="lattice nodes their indices name"):
        PositionSet("grid", layout.positions_yx, layout.centre_yx, layout.domain_radius_arcsec,
                    layout.cell_areas_arcsec2, layout.boundary, lattice)
    with pytest.raises(ValueError, match=r"spacing_arcsec\*\*2"):
        PositionSet("grid", layout.positions_yx, layout.centre_yx, layout.domain_radius_arcsec,
                    np.full(len(layout), 2.0), layout.boundary, layout.grid)
    with pytest.raises(ValueError, match="no cell areas, boundary or lattice"):
        PositionSet("explicit", layout.positions_yx, layout.centre_yx, layout.domain_radius_arcsec,
                    layout.cell_areas_arcsec2, None, None)
