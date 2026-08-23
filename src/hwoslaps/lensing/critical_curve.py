"""Deterministic tangential critical curve and effective Einstein radius.

The D-F7 aperture ruling computes every production aperture estimand
inside ``R <= 2 theta_E``, where ``theta_E`` for a non-isothermal or
flexible truth macro is the area-equivalent Einstein radius

``theta_E_eff = sqrt(A_crit/pi)``

with ``A_crit`` the image-plane area enclosed by the main tangential
critical curve of the truth macro model.

The extraction is deliberately frozen so that ``DesignFreeze`` can pin
it:

- the tangential eigenvalue ``1 - kappa - |gamma|`` is evaluated by
  PyAutoLens ``LensCalc`` on a declared uniform grid whose half width
  and pixel scale are explicit parameters, never an adaptive or
  zoom-derived evaluation grid;
- zero contours come from the supported marching-squares path
  (``LensCalc.tangential_critical_curve_list_from``);
- the main loop is chosen by the rule recorded in `CHOICE_RULE_ID`;
- every failure mode raises a typed `CriticalCurveError` subclass, so
  no silent fallback can reach an aperture definition;
- the chosen contour polyline and the derived aperture definition are
  hashed and returned as provenance.

Angles are arcseconds throughout, and every ``(y, x)`` ordering follows
the PyAutoLens convention.

The stored contour is canonicalized so that its hash is invariant to
the traversal start vertex and direction, and reruns on the same
declared grid reproduce it byte for byte. The hash is not invariant to
the declared grid: changing the grid half width moves the marching
squares index-to-arcsecond arithmetic by an ULP or two. The grid
declaration therefore travels with the hash in the provenance record.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math

import numpy as np

ALGORITHM_ID = "tangential_critical_curve_marching_squares_v1"
"""Identity of the extraction algorithm (`str`)."""

CHOICE_RULE_ID = "largest_area_closed_curve_enclosing_lens_centre"
"""Identity of the main-loop choice rule (`str`).

Among the tangential critical curves returned on the declared grid,
keep the closed ones whose polygon encloses the declared lens centre
under the even-odd ray-crossing test, and take the one enclosing the
largest area. Nested loops therefore resolve to the outer tangential
critical curve of the main deflector, which is the standard effective
Einstein radius convention. Exactly equal areas are broken by the
lexicographically smallest canonical start vertex.
"""

DEFAULT_PIXEL_SCALE_ARCSEC = 0.01
"""Declared extraction grid pixel scale (`float`, arcsec)."""

DEFAULT_GRID_HALF_WIDTH_FACTOR = 4.0
"""Extraction grid half width in units of the macro Einstein radius
parameter (`float`)."""

DEFAULT_APERTURE_THETA_E_FACTOR = 2.0
"""Aperture radius in units of ``theta_E_eff`` (`float`), fixed at 2 by
the D-F7 ruling."""

DEFAULT_COMPUTATIONAL_MARGIN_FRACTION = 0.1
"""Fractional map extent required beyond the aperture radius
(`float`)."""

DEFAULT_CLOSURE_TOLERANCE_PIXELS = 0.5
"""Endpoint separation below which a contour counts as closed
(`float`, pixels)."""

DEFAULT_BORDER_MARGIN_PIXELS = 2.0
"""Clearance the chosen curve must keep from the grid border
(`float`, pixels)."""

DEFAULT_MIN_CONTOUR_VERTICES = 32
"""Smallest vertex count accepted for the chosen curve (`int`)."""


class CriticalCurveError(ValueError):
    """Base class for every tangential critical curve failure."""


class NoTangentialCriticalCurveError(CriticalCurveError):
    """The tangential eigenvalue has no zero contour on the grid."""


class OpenCriticalCurveError(CriticalCurveError):
    """Every extracted contour is open, so no area is enclosed."""


class NoEnclosingCurveError(CriticalCurveError):
    """No closed contour encloses the declared lens centre."""


class GridExtentError(CriticalCurveError):
    """The declared grid does not contain the curve or the centre."""


class GridResolutionError(CriticalCurveError):
    """The chosen curve is too sparsely sampled to define an area."""


def _positive_finite(value, name):
    """Return ``value`` as a positive finite float or raise."""
    number = float(value)
    if not math.isfinite(number) or number <= 0.0:
        raise ValueError(f"{name} must be positive and finite, got {value!r}")
    return number


def _nonnegative_finite(value, name):
    """Return ``value`` as a non-negative finite float or raise."""
    number = float(value)
    if not math.isfinite(number) or number < 0.0:
        raise ValueError(f"{name} must be non-negative and finite, got {value!r}")
    return number


def _centre_tuple(centre, name):
    """Return a finite ``(y, x)`` centre tuple or raise."""
    values = np.asarray(centre, dtype=float)
    if values.shape != (2,) or not np.all(np.isfinite(values)):
        raise ValueError(f"{name} must be two finite (y, x) coordinates, got {centre!r}")
    return (float(values[0]), float(values[1]))


@dataclass(frozen=True)
class CriticalCurveGrid:
    """Declared uniform grid the tangential eigenvalue is sampled on.

    The grid is centred on the coordinate origin because the supported
    marching-squares path converts contour indices to arcseconds with a
    zero origin. The realized half width is rounded up to a whole
    number of pixels so the grid is symmetric about the origin.

    Parameters
    ----------
    requested_half_width_arcsec : `float`
        Requested half width of the square grid.
    pixel_scale_arcsec : `float`
        Grid spacing.
    """

    requested_half_width_arcsec: float
    pixel_scale_arcsec: float

    def __post_init__(self):
        _positive_finite(self.requested_half_width_arcsec, "requested_half_width_arcsec")
        _positive_finite(self.pixel_scale_arcsec, "pixel_scale_arcsec")

    @property
    def pixels_per_side(self):
        """Whole even pixel count per side (`int`, read-only)."""
        return 2*int(math.ceil(self.requested_half_width_arcsec/self.pixel_scale_arcsec))

    @property
    def half_width_arcsec(self):
        """Realized half width of the grid (`float`, read-only)."""
        return 0.5*self.pixels_per_side*self.pixel_scale_arcsec

    @property
    def shape_native(self):
        """Native ``(rows, columns)`` shape (`tuple`, read-only)."""
        pixels = self.pixels_per_side
        return (pixels, pixels)

    def to_dict(self):
        """Return the grid declaration as a plain dictionary.

        Returns
        -------
        declaration : `dict`
            Requested and realized extents, pixel scale, and shape.
        """
        return {
            "requested_half_width_arcsec": float(self.requested_half_width_arcsec),
            "half_width_arcsec": float(self.half_width_arcsec),
            "pixel_scale_arcsec": float(self.pixel_scale_arcsec),
            "pixels_per_side": int(self.pixels_per_side),
        }


@dataclass(frozen=True)
class ApertureDefinition:
    """Aperture derived from an effective Einstein radius.

    Parameters
    ----------
    centre_arcsec : `tuple`
        Aperture centre as ``(y, x)``, the declared lens centre.
    theta_e_eff_arcsec : `float`
        Effective Einstein radius of the truth macro model.
    theta_e_factor : `float`
        Aperture radius in units of ``theta_e_eff_arcsec``.
    computational_margin_fraction : `float`
        Fractional map extent required beyond the aperture radius. The
        Fisher map and mask machinery cannot evaluate the aperture rim
        on a map that merely reaches it, so the manifest generator sizes
        map extents from `required_map_half_width_arcsec`.
    """

    centre_arcsec: tuple
    theta_e_eff_arcsec: float
    theta_e_factor: float = DEFAULT_APERTURE_THETA_E_FACTOR
    computational_margin_fraction: float = DEFAULT_COMPUTATIONAL_MARGIN_FRACTION

    def __post_init__(self):
        _centre_tuple(self.centre_arcsec, "centre_arcsec")
        _positive_finite(self.theta_e_eff_arcsec, "theta_e_eff_arcsec")
        _positive_finite(self.theta_e_factor, "theta_e_factor")
        _nonnegative_finite(
            self.computational_margin_fraction, "computational_margin_fraction"
        )

    @property
    def radius_arcsec(self):
        """Aperture radius ``factor*theta_E_eff`` (`float`, read-only)."""
        return self.theta_e_factor*self.theta_e_eff_arcsec

    @property
    def required_map_half_width_arcsec(self):
        """Map half width including the margin (`float`, read-only)."""
        return self.radius_arcsec*(1.0 + self.computational_margin_fraction)

    @property
    def required_map_extent_arcsec(self):
        """Full map extent including the margin (`float`, read-only)."""
        return 2.0*self.required_map_half_width_arcsec

    def to_dict(self):
        """Return the aperture declaration as a plain dictionary.

        Returns
        -------
        declaration : `dict`
            Centre, effective Einstein radius, factor, margin, and the
            derived radius and map extents.
        """
        return {
            "centre_arcsec": [float(self.centre_arcsec[0]), float(self.centre_arcsec[1])],
            "theta_e_eff_arcsec": float(self.theta_e_eff_arcsec),
            "theta_e_factor": float(self.theta_e_factor),
            "radius_arcsec": float(self.radius_arcsec),
            "computational_margin_fraction": float(self.computational_margin_fraction),
            "required_map_half_width_arcsec": float(self.required_map_half_width_arcsec),
            "required_map_extent_arcsec": float(self.required_map_extent_arcsec),
        }

    @property
    def sha256(self):
        """SHA-256 of the aperture declaration (`str`, read-only)."""
        payload = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass(frozen=True, eq=False)
class ThetaEExtraction:
    """Result of one deterministic ``theta_E_eff`` extraction.

    Parameters
    ----------
    contour_arcsec : `numpy.ndarray`
        Canonical closed polyline of the chosen tangential critical
        curve, shape ``(N, 2)`` in ``(y, x)`` arcseconds.
    area_arcsec2 : `float`
        Image-plane area enclosed by the chosen curve.
    theta_e_eff_arcsec : `float`
        ``sqrt(area/pi)``.
    aperture : `ApertureDefinition`
        Aperture derived from ``theta_e_eff_arcsec``.
    grid : `CriticalCurveGrid`
        Declared extraction grid.
    lens_centre_arcsec : `tuple`
        Declared main lens centre as ``(y, x)``.
    curve_counts : `dict`
        Contour census with keys ``extracted``, ``closed``, and
        ``enclosing``.
    """

    contour_arcsec: np.ndarray
    area_arcsec2: float
    theta_e_eff_arcsec: float
    aperture: ApertureDefinition
    grid: CriticalCurveGrid
    lens_centre_arcsec: tuple
    curve_counts: dict

    @property
    def contour_sha256(self):
        """SHA-256 of the canonical contour polyline (`str`)."""
        return polyline_digest(self.contour_arcsec)

    def to_provenance_dict(self):
        """Return the frozen record `DesignFreeze` pins.

        Returns
        -------
        provenance : `dict`
            Algorithm and choice-rule identities, grid declaration,
            contour census and hash, and the aperture declaration and
            hash.
        """
        return {
            "algorithm_id": ALGORITHM_ID,
            "choice_rule_id": CHOICE_RULE_ID,
            "grid": self.grid.to_dict(),
            "lens_centre_arcsec": [
                float(self.lens_centre_arcsec[0]),
                float(self.lens_centre_arcsec[1]),
            ],
            "curve_counts": dict(self.curve_counts),
            "contour_vertices": int(self.contour_arcsec.shape[0]),
            "contour_sha256": self.contour_sha256,
            "area_arcsec2": float(self.area_arcsec2),
            "theta_e_eff_arcsec": float(self.theta_e_eff_arcsec),
            "aperture": self.aperture.to_dict(),
            "aperture_sha256": self.aperture.sha256,
        }


def polygon_area(polygon):
    """Return the area enclosed by a closed polygon.

    Parameters
    ----------
    polygon : `numpy.ndarray`
        Closed polyline of shape ``(N, 2)`` whose last vertex repeats
        the first, in ``(y, x)`` order.

    Returns
    -------
    area : `float`
        Absolute shoelace area in squared coordinate units.
    """
    vertices = np.asarray(polygon, dtype=float)
    y = vertices[:-1, 0]
    x = vertices[:-1, 1]
    return float(abs(0.5*np.sum(x*np.roll(y, -1) - np.roll(x, -1)*y)))


def polygon_contains_point(polygon, point):
    """Return whether a closed polygon encloses a point.

    The test is the even-odd ray-crossing rule, cast along increasing
    ``x`` at fixed ``y``. Points exactly on the boundary are not
    guaranteed either answer, which is immaterial for a lens centre
    that never sits on its own critical curve.

    Parameters
    ----------
    polygon : `numpy.ndarray`
        Closed polyline of shape ``(N, 2)`` in ``(y, x)`` order.
    point : `tuple`
        Query point as ``(y, x)``.

    Returns
    -------
    contains : `bool`
        `True` when the point is enclosed.
    """
    vertices = np.asarray(polygon, dtype=float)[:-1]
    y0, x0 = float(point[0]), float(point[1])
    y_start = vertices[:, 0]
    x_start = vertices[:, 1]
    y_end = np.roll(y_start, -1)
    x_end = np.roll(x_start, -1)
    straddles = (y_start > y0) != (y_end > y0)
    with np.errstate(divide="ignore", invalid="ignore"):
        x_cross = x_start + (y0 - y_start)*(x_end - x_start)/(y_end - y_start)
    crossings = straddles & (x0 < x_cross)
    return bool(np.count_nonzero(crossings) % 2 == 1)


def canonical_polygon(polygon):
    """Return a traversal-invariant closed polygon.

    The polyline is rotated to start at its lexicographically smallest
    ``(y, x)`` vertex and oriented counter-clockwise, so that two runs
    which trace the same curve from a different vertex or in the
    opposite direction hash identically.

    Parameters
    ----------
    polygon : `numpy.ndarray`
        Closed polyline of shape ``(N, 2)`` in ``(y, x)`` order.

    Returns
    -------
    canonical : `numpy.ndarray`
        Closed polyline with the canonical start vertex and
        orientation.
    """
    vertices = np.ascontiguousarray(np.asarray(polygon, dtype=float)[:-1])
    signed = 0.5*np.sum(
        vertices[:, 1]*np.roll(vertices[:, 0], -1)
        - np.roll(vertices[:, 1], -1)*vertices[:, 0]
    )
    if signed < 0.0:
        vertices = vertices[::-1]
    order = np.lexsort((vertices[:, 1], vertices[:, 0]))
    vertices = np.roll(vertices, -int(order[0]), axis=0)
    return np.ascontiguousarray(np.vstack([vertices, vertices[:1]]))


def polyline_digest(polyline):
    """Return the SHA-256 of a polyline.

    Parameters
    ----------
    polyline : `numpy.ndarray`
        Array of shape ``(N, 2)``.

    Returns
    -------
    digest : `str`
        SHA-256 of the algorithm identity, the vertex count, and the
        contiguous float64 bytes. The byte convention is part of the
        provenance contract; changing it invalidates recorded hashes.
    """
    values = np.ascontiguousarray(polyline, dtype=np.float64)
    prefix = f"{ALGORITHM_ID}|{values.shape[0]}x{values.shape[1]}:".encode("ascii")
    return hashlib.sha256(prefix + values.tobytes()).hexdigest()


def select_main_tangential_curve(curves, lens_centre_arcsec, closure_tolerance_arcsec):
    """Apply the main-loop choice rule to extracted contours.

    Parameters
    ----------
    curves : `list`
        Contours as arrays of shape ``(N, 2)`` in ``(y, x)``
        arcseconds, in the order the extractor returned them.
    lens_centre_arcsec : `tuple`
        Declared main lens centre as ``(y, x)``.
    closure_tolerance_arcsec : `float`
        Endpoint separation below which a contour counts as closed.

    Returns
    -------
    polygon : `numpy.ndarray`
        Canonical closed polyline of the chosen curve.
    area : `float`
        Area enclosed by the chosen curve.
    counts : `dict`
        Contour census with keys ``extracted``, ``closed``, and
        ``enclosing``.

    Raises
    ------
    NoTangentialCriticalCurveError
        Raised when no contour was extracted at all.
    OpenCriticalCurveError
        Raised when every extracted contour is open.
    NoEnclosingCurveError
        Raised when no closed contour encloses the lens centre.
    GridResolutionError
        Raised when the chosen curve encloses no positive area.
    """
    centre = _centre_tuple(lens_centre_arcsec, "lens_centre_arcsec")
    tolerance = _positive_finite(closure_tolerance_arcsec, "closure_tolerance_arcsec")
    extracted = [np.asarray(curve, dtype=float) for curve in curves]
    if not extracted:
        raise NoTangentialCriticalCurveError(
            "No tangential critical curve was found on the declared grid: the "
            "tangential eigenvalue has no zero contour there."
        )

    closed = [
        curve for curve in extracted
        if curve.shape[0] >= 4
        and float(np.hypot(*(curve[0] - curve[-1]))) <= tolerance
    ]
    if not closed:
        raise OpenCriticalCurveError(
            f"All {len(extracted)} tangential critical contours are open on the "
            "declared grid, so no enclosed area is defined; enlarge the grid "
            "half width."
        )

    enclosing = [
        canonical_polygon(curve) for curve in closed
        if polygon_contains_point(curve, centre)
    ]
    if not enclosing:
        raise NoEnclosingCurveError(
            f"None of the {len(closed)} closed tangential critical curves "
            f"encloses the declared lens centre {centre}."
        )

    def _rank(polygon):
        """Rank a candidate by descending area, then start vertex."""
        return (-polygon_area(polygon), float(polygon[0, 0]), float(polygon[0, 1]))

    chosen = min(enclosing, key=_rank)
    area = polygon_area(chosen)
    if not math.isfinite(area) or area <= 0.0:
        raise GridResolutionError(
            f"The chosen tangential critical curve encloses a non-positive area "
            f"{area!r}; refine the extraction pixel scale."
        )
    counts = {
        "extracted": len(extracted),
        "closed": len(closed),
        "enclosing": len(enclosing),
    }
    return chosen, area, counts


def _evaluation_grid(grid):
    """Return the declared uniform PyAutoLens grid, marked as final."""
    import autoarray as aa

    values = aa.Grid2D.uniform(
        shape_native=grid.shape_native,
        pixel_scales=(grid.pixel_scale_arcsec, grid.pixel_scale_arcsec),
    )
    values.is_evaluation_grid = True
    return values


def extract_theta_e(
    mass_obj,
    lens_centre_arcsec,
    grid,
    theta_e_factor=DEFAULT_APERTURE_THETA_E_FACTOR,
    computational_margin_fraction=DEFAULT_COMPUTATIONAL_MARGIN_FRACTION,
    closure_tolerance_pixels=DEFAULT_CLOSURE_TOLERANCE_PIXELS,
    border_margin_pixels=DEFAULT_BORDER_MARGIN_PIXELS,
    min_contour_vertices=DEFAULT_MIN_CONTOUR_VERTICES,
):
    """Extract ``theta_E_eff`` and the aperture of a truth macro model.

    Parameters
    ----------
    mass_obj : `object`
        Any PyAutoLens object exposing ``deflections_yx_2d_from``, such
        as the truth macro `autolens.Galaxy` built from a scene
        configuration.
    lens_centre_arcsec : `tuple`
        Main lens centre as ``(y, x)``. This is the centre of the
        primary macro mass profile, which the multipole profiles share
        and which external shear does not move.
    grid : `CriticalCurveGrid`
        Declared extraction grid.
    theta_e_factor : `float`, optional
        Aperture radius in units of ``theta_E_eff``.
    computational_margin_fraction : `float`, optional
        Fractional map extent required beyond the aperture radius.
    closure_tolerance_pixels : `float`, optional
        Endpoint separation below which a contour counts as closed.
    border_margin_pixels : `float`, optional
        Clearance the chosen curve must keep from the grid border.
    min_contour_vertices : `int`, optional
        Smallest vertex count accepted for the chosen curve.

    Returns
    -------
    extraction : `ThetaEExtraction`
        Chosen contour, enclosed area, ``theta_E_eff``, aperture
        definition, and provenance hashes.

    Raises
    ------
    GridExtentError
        Raised when the lens centre lies outside the declared grid or
        the chosen curve runs closer to the border than the declared
        margin.
    GridResolutionError
        Raised when the chosen curve carries fewer than
        ``min_contour_vertices`` vertices or encloses no positive area.
    CriticalCurveError
        Raised by `select_main_tangential_curve` for the missing, open,
        and non-enclosing contour cases.
    """
    from autogalaxy.operate.lens_calc import LensCalc

    centre = _centre_tuple(lens_centre_arcsec, "lens_centre_arcsec")
    closure_pixels = _positive_finite(closure_tolerance_pixels, "closure_tolerance_pixels")
    border_pixels = _nonnegative_finite(border_margin_pixels, "border_margin_pixels")
    minimum_vertices = int(min_contour_vertices)
    if minimum_vertices < 4:
        raise ValueError(f"min_contour_vertices must be at least 4, got {min_contour_vertices!r}")

    half_width = grid.half_width_arcsec
    if max(abs(centre[0]), abs(centre[1])) >= half_width:
        raise GridExtentError(
            f"The declared lens centre {centre} lies outside the extraction grid "
            f"half width {half_width} arcsec."
        )

    lens_calc = LensCalc.from_mass_obj(mass_obj)
    evaluation_grid = _evaluation_grid(grid)
    curves = lens_calc.tangential_critical_curve_list_from(
        grid=evaluation_grid, pixel_scale=grid.pixel_scale_arcsec
    )

    polygon, area, counts = select_main_tangential_curve(
        curves,
        lens_centre_arcsec=centre,
        closure_tolerance_arcsec=closure_pixels*grid.pixel_scale_arcsec,
    )

    border_limit = half_width - border_pixels*grid.pixel_scale_arcsec
    if float(np.max(np.abs(polygon))) > border_limit:
        raise GridExtentError(
            f"The chosen tangential critical curve reaches "
            f"{float(np.max(np.abs(polygon)))} arcsec, beyond the declared border "
            f"limit {border_limit} arcsec; enlarge the grid half width."
        )
    if polygon.shape[0] < minimum_vertices:
        raise GridResolutionError(
            f"The chosen tangential critical curve has {polygon.shape[0]} vertices, "
            f"fewer than the declared minimum {minimum_vertices}; refine the "
            "extraction pixel scale."
        )

    theta_e_eff = math.sqrt(area/math.pi)
    aperture = ApertureDefinition(
        centre_arcsec=centre,
        theta_e_eff_arcsec=theta_e_eff,
        theta_e_factor=theta_e_factor,
        computational_margin_fraction=computational_margin_fraction,
    )
    return ThetaEExtraction(
        contour_arcsec=polygon,
        area_arcsec2=area,
        theta_e_eff_arcsec=theta_e_eff,
        aperture=aperture,
        grid=grid,
        lens_centre_arcsec=centre,
        curve_counts=counts,
    )


def extract_theta_e_from_lens_config(
    lens_config,
    pixel_scale_arcsec=DEFAULT_PIXEL_SCALE_ARCSEC,
    grid_half_width_arcsec=None,
    grid_half_width_factor=DEFAULT_GRID_HALF_WIDTH_FACTOR,
    **kwargs,
):
    """Extract ``theta_E_eff`` from a scene ``lens_galaxy`` block.

    The truth macro model is rebuilt with the same profile factory the
    lensing generator uses, so the extraction always sees the exact
    truth macro of the scene, multipoles and external shear included.

    Parameters
    ----------
    lens_config : `dict`
        The ``lensing.lens_galaxy`` block of a scene configuration.
    pixel_scale_arcsec : `float`, optional
        Declared extraction grid pixel scale.
    grid_half_width_arcsec : `float`, optional
        Declared extraction grid half width. Defaults to
        ``grid_half_width_factor`` times the macro ``einstein_radius``
        parameter.
    grid_half_width_factor : `float`, optional
        Half width in units of the macro ``einstein_radius`` parameter,
        used only when ``grid_half_width_arcsec`` is `None`.
    **kwargs
        Forwarded to `extract_theta_e`.

    Returns
    -------
    extraction : `ThetaEExtraction`
        Chosen contour, enclosed area, ``theta_E_eff``, aperture
        definition, and provenance hashes.
    """
    from .generator import _create_lens_galaxy

    mass_config = lens_config["mass"]
    if grid_half_width_arcsec is None:
        grid_half_width_arcsec = _positive_finite(
            grid_half_width_factor, "grid_half_width_factor"
        )*_positive_finite(mass_config["einstein_radius"], "mass.einstein_radius")
    grid = CriticalCurveGrid(
        requested_half_width_arcsec=grid_half_width_arcsec,
        pixel_scale_arcsec=pixel_scale_arcsec,
    )
    return extract_theta_e(
        _create_lens_galaxy(lens_config),
        lens_centre_arcsec=_centre_tuple(mass_config["centre"], "mass.centre"),
        grid=grid,
        **kwargs,
    )
