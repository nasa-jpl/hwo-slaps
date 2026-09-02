"""Contracts for the deterministic theta_E extraction algorithm."""

from __future__ import annotations

import math

import numpy as np
import pytest

autolens = pytest.importorskip("autolens")

from hwoslaps.lensing import critical_curve as cc  # noqa: E402

UNIT_SQUARE = np.array(
    [[0.0, 0.0], [0.0, 1.0], [1.0, 1.0], [1.0, 0.0], [0.0, 0.0]], dtype=float
)
"""Unit square as a closed ``(y, x)`` polyline, enclosing area 1."""

TALL_RECTANGLE = np.array(
    [[-2.0, -0.5], [-2.0, 0.5], [2.0, 0.5], [2.0, -0.5], [-2.0, -0.5]], dtype=float
)
"""Rectangle 4 tall by 1 wide about the origin, enclosing area 4."""

WIDE_RECTANGLE = np.array(
    [[-0.5, -2.0], [-0.5, 2.0], [0.5, 2.0], [0.5, -2.0], [-0.5, -2.0]], dtype=float
)
"""Rectangle 1 tall by 4 wide about the origin, enclosing area 4."""


def _circle(radius, centre=(0.0, 0.0), vertices=720):
    """Return a closed ``(y, x)`` polyline sampling a circle."""
    angles = np.linspace(0.0, 2.0*np.pi, vertices + 1)
    return np.stack(
        [centre[0] + radius*np.sin(angles), centre[1] + radius*np.cos(angles)], axis=1
    )


def _sie_galaxy(einstein_radius=1.0, ell_comps=(0.0, 0.0), centre=(0.0, 0.0)):
    """Return a truth SIE macro galaxy."""
    return autolens.Galaxy(
        redshift=0.2,
        mass=autolens.mp.Isothermal(
            centre=tuple(centre),
            einstein_radius=einstein_radius,
            ell_comps=tuple(ell_comps),
        ),
    )


def test_polygon_area_is_hand_calculable():
    """Shoelace areas match hand values for squares and a circle."""
    assert cc.polygon_area(UNIT_SQUARE) == pytest.approx(1.0)
    assert cc.polygon_area(2.0*UNIT_SQUARE) == pytest.approx(4.0)
    assert cc.polygon_area(TALL_RECTANGLE) == pytest.approx(4.0)
    assert cc.polygon_area(WIDE_RECTANGLE) == pytest.approx(4.0)
    assert cc.polygon_area(_circle(1.0)) == pytest.approx(math.pi, rel=1e-4)


def test_polygon_area_is_orientation_independent():
    """Reversing traversal does not change the reported area."""
    assert cc.polygon_area(UNIT_SQUARE[::-1]) == pytest.approx(1.0)


def test_polygon_contains_point_even_odd_rule():
    """The ray-crossing test separates inside from outside points."""
    assert cc.polygon_contains_point(UNIT_SQUARE, (0.5, 0.5))
    assert not cc.polygon_contains_point(UNIT_SQUARE, (0.5, 1.5))
    assert not cc.polygon_contains_point(UNIT_SQUARE, (-0.5, 0.5))
    assert cc.polygon_contains_point(_circle(1.0), (0.0, 0.0))
    assert not cc.polygon_contains_point(_circle(1.0), (0.0, 1.5))


def test_canonical_polygon_is_traversal_invariant():
    """Rotation and reversal collapse to the same canonical polyline."""
    canonical = cc.canonical_polygon(UNIT_SQUARE)
    rotated = np.vstack([UNIT_SQUARE[2:-1], UNIT_SQUARE[:2], UNIT_SQUARE[2:3]])
    assert np.array_equal(cc.canonical_polygon(rotated), canonical)
    assert np.array_equal(cc.canonical_polygon(UNIT_SQUARE[::-1]), canonical)
    assert tuple(canonical[0]) == (0.0, 0.0)
    assert np.array_equal(canonical[0], canonical[-1])


def test_polyline_digest_matches_canonical_traversals():
    """Digests are stable under traversal but not under geometry."""
    canonical = cc.canonical_polygon(UNIT_SQUARE)
    assert cc.polyline_digest(canonical) == cc.polyline_digest(
        cc.canonical_polygon(UNIT_SQUARE[::-1])
    )
    assert cc.polyline_digest(canonical) != cc.polyline_digest(
        cc.canonical_polygon(2.0*UNIT_SQUARE)
    )


def test_choice_rule_prefers_the_largest_enclosing_loop():
    """Nested loops resolve to the outer tangential critical curve."""
    curves = [_circle(0.5), _circle(1.0), _circle(2.0, centre=(0.0, 6.0))]
    polygon, area, counts = cc.select_main_tangential_curve(
        curves, lens_centre_arcsec=(0.0, 0.0), closure_tolerance_arcsec=1e-6
    )
    assert counts == {"extracted": 3, "closed": 3, "enclosing": 2}
    assert area == pytest.approx(math.pi, rel=1e-4)
    assert float(np.max(np.abs(polygon))) == pytest.approx(1.0, rel=1e-6)


def test_choice_rule_ignores_larger_non_enclosing_loops():
    """A larger loop that misses the lens centre is never chosen."""
    curves = [_circle(1.0), _circle(3.0, centre=(0.0, 10.0))]
    polygon, area, counts = cc.select_main_tangential_curve(
        curves, lens_centre_arcsec=(0.0, 0.0), closure_tolerance_arcsec=1e-6
    )
    assert counts["enclosing"] == 1
    assert area == pytest.approx(math.pi, rel=1e-4)
    assert float(np.max(np.abs(polygon))) == pytest.approx(1.0, rel=1e-6)


def test_choice_rule_breaks_equal_area_ties_deterministically():
    """Exactly equal areas resolve by the canonical start vertex."""
    curves = [TALL_RECTANGLE, WIDE_RECTANGLE]
    polygon, area, counts = cc.select_main_tangential_curve(
        curves, lens_centre_arcsec=(0.0, 0.0), closure_tolerance_arcsec=1e-6
    )
    swapped, _, _ = cc.select_main_tangential_curve(
        curves[::-1], lens_centre_arcsec=(0.0, 0.0), closure_tolerance_arcsec=1e-6
    )
    assert counts["enclosing"] == 2
    assert area == pytest.approx(4.0)
    assert np.array_equal(polygon, swapped)
    assert tuple(polygon[0]) == (-2.0, -0.5)


def test_missing_curve_fails_loudly():
    """An empty contour list raises the missing-curve error."""
    with pytest.raises(cc.NoTangentialCriticalCurveError):
        cc.select_main_tangential_curve(
            [], lens_centre_arcsec=(0.0, 0.0), closure_tolerance_arcsec=1e-6
        )


def test_open_contour_fails_loudly():
    """Contours whose endpoints do not meet raise the open-curve error."""
    with pytest.raises(cc.OpenCriticalCurveError):
        cc.select_main_tangential_curve(
            [_circle(1.0)[:-200]],
            lens_centre_arcsec=(0.0, 0.0),
            closure_tolerance_arcsec=1e-6,
        )


def test_non_enclosing_contour_fails_loudly():
    """Closed loops that miss the lens centre raise the enclosure error."""
    with pytest.raises(cc.NoEnclosingCurveError):
        cc.select_main_tangential_curve(
            [_circle(1.0, centre=(0.0, 5.0))],
            lens_centre_arcsec=(0.0, 0.0),
            closure_tolerance_arcsec=1e-6,
        )


def test_every_failure_is_a_critical_curve_error():
    """The typed failures share one catchable base class."""
    for error in (
        cc.NoTangentialCriticalCurveError,
        cc.OpenCriticalCurveError,
        cc.NoEnclosingCurveError,
        cc.GridExtentError,
        cc.GridResolutionError,
    ):
        assert issubclass(error, cc.CriticalCurveError)
        assert issubclass(error, ValueError)


def test_grid_declaration_rounds_up_to_whole_pixels():
    """Grid extents are declared, symmetric, and pixel-aligned."""
    grid = cc.CriticalCurveGrid(
        requested_half_width_arcsec=4.0, pixel_scale_arcsec=0.01
    )
    assert grid.pixels_per_side == 800
    assert grid.shape_native == (800, 800)
    assert grid.half_width_arcsec == pytest.approx(4.0)

    rounded = cc.CriticalCurveGrid(
        requested_half_width_arcsec=4.005, pixel_scale_arcsec=0.01
    )
    assert rounded.pixels_per_side == 802
    assert rounded.half_width_arcsec == pytest.approx(4.01)


def test_grid_declaration_rejects_nonpositive_inputs():
    """Non-positive grid declarations fail immediately."""
    with pytest.raises(ValueError):
        cc.CriticalCurveGrid(requested_half_width_arcsec=0.0, pixel_scale_arcsec=0.01)
    with pytest.raises(ValueError):
        cc.CriticalCurveGrid(requested_half_width_arcsec=1.0, pixel_scale_arcsec=-0.01)


def test_aperture_margin_arithmetic_is_hand_calculable():
    """Aperture radius and margin follow the declared arithmetic."""
    aperture = cc.ApertureDefinition(centre_arcsec=(0.0, 0.0), theta_e_eff_arcsec=1.0)
    assert aperture.theta_e_factor == 2.0
    assert aperture.computational_margin_fraction == 0.1
    assert aperture.radius_arcsec == pytest.approx(2.0)
    assert aperture.required_map_half_width_arcsec == pytest.approx(2.2)
    assert aperture.required_map_extent_arcsec == pytest.approx(4.4)

    wider = cc.ApertureDefinition(
        centre_arcsec=(0.1, -0.2),
        theta_e_eff_arcsec=1.5,
        computational_margin_fraction=0.25,
    )
    assert wider.radius_arcsec == pytest.approx(3.0)
    assert wider.required_map_half_width_arcsec == pytest.approx(3.75)
    assert wider.required_map_extent_arcsec == pytest.approx(7.5)


def test_zero_margin_leaves_the_aperture_radius_untouched():
    """A declared zero margin reduces to the bare 2 theta_E extent."""
    aperture = cc.ApertureDefinition(
        centre_arcsec=(0.0, 0.0),
        theta_e_eff_arcsec=1.25,
        computational_margin_fraction=0.0,
    )
    assert aperture.radius_arcsec == pytest.approx(2.5)
    assert aperture.required_map_half_width_arcsec == pytest.approx(2.5)


def test_aperture_hash_tracks_every_declared_parameter():
    """The aperture hash changes with each pinned aperture input."""
    baseline = cc.ApertureDefinition(centre_arcsec=(0.0, 0.0), theta_e_eff_arcsec=1.0)
    assert baseline.sha256 == cc.ApertureDefinition(
        centre_arcsec=(0.0, 0.0), theta_e_eff_arcsec=1.0
    ).sha256
    variants = [
        cc.ApertureDefinition(centre_arcsec=(0.1, 0.0), theta_e_eff_arcsec=1.0),
        cc.ApertureDefinition(centre_arcsec=(0.0, 0.0), theta_e_eff_arcsec=1.01),
        cc.ApertureDefinition(
            centre_arcsec=(0.0, 0.0), theta_e_eff_arcsec=1.0, theta_e_factor=1.5
        ),
        cc.ApertureDefinition(
            centre_arcsec=(0.0, 0.0),
            theta_e_eff_arcsec=1.0,
            computational_margin_fraction=0.2,
        ),
    ]
    digests = {baseline.sha256} | {variant.sha256 for variant in variants}
    assert len(digests) == len(variants) + 1


def test_aperture_rejects_invalid_declarations():
    """Invalid aperture declarations fail immediately."""
    with pytest.raises(ValueError):
        cc.ApertureDefinition(centre_arcsec=(0.0, 0.0), theta_e_eff_arcsec=0.0)
    with pytest.raises(ValueError):
        cc.ApertureDefinition(
            centre_arcsec=(0.0, 0.0),
            theta_e_eff_arcsec=1.0,
            computational_margin_fraction=-0.1,
        )
    with pytest.raises(ValueError):
        cc.ApertureDefinition(centre_arcsec=(0.0,), theta_e_eff_arcsec=1.0)


def test_sie_anchor_recovers_unit_einstein_radius():
    """A circular SIE with R_E = 1 returns theta_E_eff = 1."""
    grid = cc.CriticalCurveGrid(
        requested_half_width_arcsec=4.0, pixel_scale_arcsec=0.01
    )
    extraction = cc.extract_theta_e(
        _sie_galaxy(), lens_centre_arcsec=(0.0, 0.0), grid=grid
    )
    assert extraction.curve_counts == {"extracted": 1, "closed": 1, "enclosing": 1}
    assert extraction.theta_e_eff_arcsec == pytest.approx(1.0, abs=1e-4)
    assert extraction.area_arcsec2 == pytest.approx(math.pi, abs=1e-3)
    assert extraction.aperture.radius_arcsec == pytest.approx(
        2.0*extraction.theta_e_eff_arcsec
    )


def test_sie_anchor_converges_with_pixel_scale():
    """Refining the declared pixel scale tightens the SIE anchor."""
    coarse = cc.extract_theta_e(
        _sie_galaxy(),
        lens_centre_arcsec=(0.0, 0.0),
        grid=cc.CriticalCurveGrid(
            requested_half_width_arcsec=4.0, pixel_scale_arcsec=0.02
        ),
    )
    fine = cc.extract_theta_e(
        _sie_galaxy(),
        lens_centre_arcsec=(0.0, 0.0),
        grid=cc.CriticalCurveGrid(
            requested_half_width_arcsec=4.0, pixel_scale_arcsec=0.005
        ),
    )
    assert abs(fine.theta_e_eff_arcsec - 1.0) < abs(coarse.theta_e_eff_arcsec - 1.0)
    assert abs(fine.theta_e_eff_arcsec - 1.0) < 1.0e-5


def test_sie_anchor_scales_with_einstein_radius():
    """A doubled SIE Einstein radius doubles theta_E_eff."""
    extraction = cc.extract_theta_e(
        _sie_galaxy(einstein_radius=2.0),
        lens_centre_arcsec=(0.0, 0.0),
        grid=cc.CriticalCurveGrid(
            requested_half_width_arcsec=8.0, pixel_scale_arcsec=0.01
        ),
    )
    assert extraction.theta_e_eff_arcsec == pytest.approx(2.0, abs=1e-4)


def test_grid_below_the_curve_finds_no_contour():
    """A grid entirely inside the curve raises the missing-curve error."""
    with pytest.raises(cc.NoTangentialCriticalCurveError):
        cc.extract_theta_e(
            _sie_galaxy(),
            lens_centre_arcsec=(0.0, 0.0),
            grid=cc.CriticalCurveGrid(
                requested_half_width_arcsec=0.6, pixel_scale_arcsec=0.01
            ),
        )


def test_grid_truncating_the_curve_raises_open_curve_error():
    """A grid that clips the curve into arcs raises the open-curve error."""
    with pytest.raises(cc.OpenCriticalCurveError):
        cc.extract_theta_e(
            _sie_galaxy(),
            lens_centre_arcsec=(0.0, 0.0),
            grid=cc.CriticalCurveGrid(
                requested_half_width_arcsec=0.9, pixel_scale_arcsec=0.01
            ),
        )


def test_curve_inside_the_border_margin_raises_grid_extent_error():
    """A curve closer to the border than the margin fails loudly."""
    with pytest.raises(cc.GridExtentError):
        cc.extract_theta_e(
            _sie_galaxy(),
            lens_centre_arcsec=(0.0, 0.0),
            grid=cc.CriticalCurveGrid(
                requested_half_width_arcsec=1.005, pixel_scale_arcsec=0.01
            ),
        )


def test_lens_centre_outside_grid_raises_grid_extent_error():
    """A lens centre off the declared grid fails before extraction."""
    with pytest.raises(cc.GridExtentError):
        cc.extract_theta_e(
            _sie_galaxy(centre=(9.0, 0.0)),
            lens_centre_arcsec=(9.0, 0.0),
            grid=cc.CriticalCurveGrid(
                requested_half_width_arcsec=4.0, pixel_scale_arcsec=0.01
            ),
        )


def test_coarse_grid_raises_grid_resolution_error():
    """Too few contour vertices fail loudly rather than round off."""
    with pytest.raises(cc.GridResolutionError):
        cc.extract_theta_e(
            _sie_galaxy(),
            lens_centre_arcsec=(0.0, 0.0),
            grid=cc.CriticalCurveGrid(
                requested_half_width_arcsec=4.0, pixel_scale_arcsec=0.4
            ),
            min_contour_vertices=64,
        )


def test_massless_model_raises_missing_curve_error():
    """External shear alone has no tangential critical curve."""
    with pytest.raises(cc.NoTangentialCriticalCurveError):
        cc.extract_theta_e(
            autolens.Galaxy(
                redshift=0.2,
                shear=autolens.mp.ExternalShear(gamma_1=0.0, gamma_2=0.0),
            ),
            lens_centre_arcsec=(0.0, 0.0),
            grid=cc.CriticalCurveGrid(
                requested_half_width_arcsec=4.0, pixel_scale_arcsec=0.05
            ),
        )

