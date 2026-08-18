"""Physical diagnostics for an injected dark matter subhalo.

This module reports the structural envelope of the subhalo that was
actually injected into a run: the concentration and scale radius used to
build the profile, the truncation radius when the profile is truncated,
and the projected mass and deflection inside fixed apertures.

Notes
-----
The projected mass uses the exact spherical identity rather than a
quadrature. For a circularly symmetric lens the reduced deflection obeys
``alpha(R) = R * kappa_bar(R)``, so the projected mass inside ``R`` is

    M_2D(<R) = Sigma_crit * pi * R^2 * kappa_bar(R)
             = Sigma_crit * pi * R * alpha(R),

with ``R`` and ``alpha`` converted from angles to transverse physical
lengths at the lens plane.

The identity holds only for a circularly symmetric profile whose
convergence is normalised by the same critical surface density this
module computes, so ``z_lens``, ``z_source`` and ``cosmology`` must be
the values the profile was built with.

The deflection reported for each aperture is whatever the injected
profile returns there, so an aperture far inside the profile scale
radius inherits the precision that profile's own deflection
implementation has in that regime.
"""

import numpy as np
from astropy import constants as const
from astropy import units as u

from ..constants import ARCSEC_PER_RAD, KPC_TO_M, MPC_TO_M
from .mass_models import (
    angular_diameter_distance_mpc,
    angular_diameter_distance_z1z2_mpc,
)

MSUN_TO_KG: float = float((1.0 * u.Msun).to(u.kg).value)
"""Kilograms per solar mass (Msun to kg)."""

SPHERICAL_AXIS_RATIO_TOLERANCE: float = 1.0e-12
"""Largest departure from a unit axis ratio treated as circular."""


def _require_finite_number(value, name):
    """Validate a finite non-boolean scalar number."""
    if isinstance(value, bool) or not isinstance(value, (int, float, np.integer, np.floating)):
        raise ValueError(f"{name} must be a finite number")
    value_float = float(value)
    if not np.isfinite(value_float):
        raise ValueError(f"{name} must be a finite number")
    return value_float


def _require_positive_finite(value, name):
    """Validate a positive finite non-boolean scalar number."""
    value_float = _require_finite_number(value, name)
    if value_float <= 0:
        raise ValueError(f"{name} must be a finite positive number")
    return value_float


def _require_aperture_radii(values):
    """Validate the requested aperture radii.

    Parameters
    ----------
    values : `list` of `float`
        Aperture radii in arcseconds.

    Returns
    -------
    radii_arcsec : `numpy.ndarray`
        Validated radii as a one-dimensional float array.

    Raises
    ------
    ValueError
        Raised when the radii are not a non-empty one-dimensional
        sequence of finite positive numbers.
    """
    if isinstance(values, np.ndarray):
        if values.ndim != 1:
            raise ValueError("aperture_radii_arcsec must be one-dimensional")
    elif not isinstance(values, (list, tuple)):
        raise ValueError(
            "aperture_radii_arcsec must be a list, tuple or one-dimensional ndarray"
        )
    if len(values) == 0:
        raise ValueError("aperture_radii_arcsec must not be empty")
    return np.array(
        [
            _require_positive_finite(value, f"aperture_radii_arcsec[{index}]")
            for index, value in enumerate(values)
        ],
        dtype=float,
    )


def _optional_profile_length(profile, attribute):
    """Return a validated optional angular length from a mass profile.

    Parameters
    ----------
    profile : `object`
        Constructed subhalo mass profile.
    attribute : `str`
        Attribute name holding an angular length in arcseconds.

    Returns
    -------
    length_arcsec : `float` or `None`
        Attribute value in arcseconds, or None when the profile does not
        define it.

    Raises
    ------
    ValueError
        Raised when the attribute is present but is not a finite positive
        number.
    """
    value = getattr(profile, attribute, None)
    if value is None:
        return None
    return _require_positive_finite(value, f"profile.{attribute}")


def _require_circular_profile(profile):
    """Reject a mass profile that is not circularly symmetric.

    Parameters
    ----------
    profile : `object`
        Constructed subhalo mass profile.

    Raises
    ------
    ValueError
        Raised when the profile reports an axis ratio away from unity,
        for which the spherical deflection identity does not hold.

    Notes
    -----
    PyAutoGalaxy exposes ``axis_ratio`` as a method on its mass profiles
    rather than as an attribute, so a bound method is resolved by calling
    it. A profile that exposes neither form is treated as unconstrained
    and passes, matching the behaviour for a profile with no axis ratio
    at all.
    """
    axis_ratio = getattr(profile, "axis_ratio", None)
    if axis_ratio is None:
        return
    if callable(axis_ratio):
        axis_ratio = axis_ratio()
    axis_ratio_float = _require_positive_finite(axis_ratio, "profile.axis_ratio")
    if abs(axis_ratio_float - 1.0) > SPHERICAL_AXIS_RATIO_TOLERANCE:
        raise ValueError(
            "subhalo diagnostics require a circularly symmetric profile, got "
            f"axis_ratio={axis_ratio_float:g}"
        )


def _deflection_magnitude_arcsec(profile, radii_arcsec):
    """Sample the deflection magnitude of a spherical profile.

    Parameters
    ----------
    profile : `object`
        Constructed subhalo mass profile exposing ``centre`` and
        ``deflections_yx_2d_from``.
    radii_arcsec : `numpy.ndarray`
        Radii from the profile centre in arcseconds.

    Returns
    -------
    deflection_arcsec : `numpy.ndarray`
        Deflection magnitude in arcseconds at each radius.

    Raises
    ------
    ValueError
        Raised when the profile does not expose the required interface or
        returns a malformed or non-finite deflection field.
    """
    if not callable(getattr(profile, "deflections_yx_2d_from", None)):
        raise ValueError("profile must implement 'deflections_yx_2d_from'")

    centre = getattr(profile, "centre", None)
    if not isinstance(centre, (list, tuple, np.ndarray)) or len(centre) != 2:
        raise ValueError("profile must expose a (y, x) 'centre' of length two")
    centre_y = _require_finite_number(centre[0], "profile.centre[0]")
    centre_x = _require_finite_number(centre[1], "profile.centre[1]")

    # A spherical profile is sampled along +x from its own centre.
    points = np.empty((radii_arcsec.size, 2), dtype=float)
    points[:, 0] = centre_y
    points[:, 1] = centre_x + radii_arcsec

    # PyAutoGalaxy mass profiles require one of its own grid objects, not a
    # bare array: their coordinate transform reaches for ``grid.array``.
    import autolens as al

    sample_grid = al.Grid2DIrregular(values=points)
    deflections = np.asarray(
        profile.deflections_yx_2d_from(grid=sample_grid), dtype=float
    )
    if deflections.shape != (radii_arcsec.size, 2):
        raise ValueError(
            "profile deflections must have shape "
            f"{(radii_arcsec.size, 2)}, got {deflections.shape}"
        )

    magnitude = np.hypot(deflections[:, 0], deflections[:, 1])
    if not np.all(np.isfinite(magnitude)):
        raise ValueError("profile returned a non-finite deflection")
    return magnitude


def subhalo_physical_diagnostics(
    profile,
    *,
    z_lens,
    z_source,
    cosmology,
    aperture_radii_arcsec,
    concentration=None,
):
    """Derive physical diagnostics for an injected subhalo profile.

    Parameters
    ----------
    profile : `object`
        Constructed circularly symmetric subhalo mass profile, for
        example ``autolens.mp.NFWSph`` or ``autolens.mp.NFWTruncatedSph``.
        It must expose a ``centre`` of (y, x) arcseconds and
        ``deflections_yx_2d_from(grid=...)``.
    z_lens : `float`
        Lens-plane redshift the subhalo was injected at.
    z_source : `float`
        Source-plane redshift the profile normalisation refers to.
    cosmology : `object`
        Astropy-style or PyAuto cosmology object.
    aperture_radii_arcsec : `list` of `float`
        Aperture radii in arcseconds, each finite and positive.
    concentration : `float`, optional
        Concentration ``c200 = r200 / r_s`` used to build the profile.
        Accepted only for a profile exposing ``scale_radius``.

    Returns
    -------
    diagnostics : `dict`
        JSON-serialisable diagnostics with keys:

        - ``z_lens``, ``z_source``: geometry the diagnostics refer to.
        - ``c200``: concentration argument, None when not supplied.
        - ``scale_radius_arcsec``, ``scale_radius_kpc``: profile scale
          radius, None for a profile without one.
        - ``truncation_radius_arcsec``, ``truncation_radius_kpc``:
          truncation radius, None when the profile is untruncated.
        - ``r200_kpc``: ``c200`` times the scale radius, None when either
          input is unavailable.
        - ``sigma_crit_kg_m2``: critical surface density in kg m^-2.
        - ``arcsec_to_kpc``: transverse kpc per arcsecond at the lens.
        - ``apertures``: list of per-aperture dicts holding
          ``radius_arcsec``, ``radius_kpc``, ``deflection_arcsec``,
          ``mean_convergence`` and ``enclosed_mass_2d_msun``.

    Raises
    ------
    ValueError
        Raised when the redshifts are not an ordered positive pair, when
        the apertures are not finite positive radii, when
        ``concentration`` is supplied for a profile without a scale
        radius, when the profile is not circularly symmetric, or when the
        cosmology yields a non-physical distance.

    Notes
    -----
    The enclosed mass is the projected (cylindrical) mass inside the
    aperture, obtained from the exact spherical identity documented at
    module level, not from a quadrature of the convergence.

    Examples
    --------
    Diagnose an injected NFW subhalo inside two apertures:

    >>> diagnostics = subhalo_physical_diagnostics(
    ...     subhalo,
    ...     z_lens=0.5,
    ...     z_source=2.0,
    ...     cosmology=cosmology,
    ...     aperture_radii_arcsec=[0.05, 0.1],
    ...     concentration=16.8,
    ... )
    >>> diagnostics['apertures'][0]['enclosed_mass_2d_msun']
    """
    z_lens_value = _require_positive_finite(z_lens, "z_lens")
    z_source_value = _require_positive_finite(z_source, "z_source")
    if z_source_value <= z_lens_value:
        raise ValueError("z_source must be greater than z_lens")

    radii_arcsec = _require_aperture_radii(aperture_radii_arcsec)
    _require_circular_profile(profile)

    scale_radius_arcsec = _optional_profile_length(profile, "scale_radius")
    truncation_radius_arcsec = _optional_profile_length(profile, "truncation_radius")

    c200 = None
    if concentration is not None:
        c200 = _require_positive_finite(concentration, "concentration")
        if scale_radius_arcsec is None:
            raise ValueError(
                "concentration is only accepted for a profile exposing 'scale_radius'"
            )

    D_l_m = _require_positive_finite(
        angular_diameter_distance_mpc(cosmology, z_lens_value) * MPC_TO_M,
        "angular diameter distance to the lens",
    )
    D_s_m = _require_positive_finite(
        angular_diameter_distance_mpc(cosmology, z_source_value) * MPC_TO_M,
        "angular diameter distance to the source",
    )
    D_ls_m = _require_positive_finite(
        angular_diameter_distance_z1z2_mpc(cosmology, z_lens_value, z_source_value)
        * MPC_TO_M,
        "angular diameter distance between lens and source",
    )

    # Critical surface density in SI units, matching the generator path.
    c_SI = float(const.c.value)
    G_SI = float(const.G.value)
    sigma_crit = (c_SI**2 / (4 * np.pi * G_SI)) * (D_s_m / (D_l_m * D_ls_m))

    # Transverse physical length per arcsecond at the lens plane.
    arcsec_to_m = D_l_m / ARCSEC_PER_RAD
    arcsec_to_kpc = arcsec_to_m / KPC_TO_M

    deflection_arcsec = _deflection_magnitude_arcsec(profile, radii_arcsec)

    apertures = []
    for radius_arcsec, deflection in zip(radii_arcsec, deflection_arcsec):
        radius_m = float(radius_arcsec) * arcsec_to_m
        deflection_m = float(deflection) * arcsec_to_m
        enclosed_mass_kg = sigma_crit * np.pi * radius_m * deflection_m
        apertures.append(
            {
                'radius_arcsec': float(radius_arcsec),
                'radius_kpc': float(radius_arcsec) * arcsec_to_kpc,
                'deflection_arcsec': float(deflection),
                'mean_convergence': float(deflection) / float(radius_arcsec),
                'enclosed_mass_2d_msun': enclosed_mass_kg / MSUN_TO_KG,
            }
        )

    scale_radius_kpc = (
        None if scale_radius_arcsec is None else scale_radius_arcsec * arcsec_to_kpc
    )
    truncation_radius_kpc = (
        None
        if truncation_radius_arcsec is None
        else truncation_radius_arcsec * arcsec_to_kpc
    )
    # A concentration is only accepted alongside a scale radius above.
    r200_kpc = None if c200 is None else c200 * scale_radius_kpc

    return {
        'z_lens': z_lens_value,
        'z_source': z_source_value,
        'c200': c200,
        'scale_radius_arcsec': scale_radius_arcsec,
        'scale_radius_kpc': scale_radius_kpc,
        'truncation_radius_arcsec': truncation_radius_arcsec,
        'truncation_radius_kpc': truncation_radius_kpc,
        'r200_kpc': r200_kpc,
        'sigma_crit_kg_m2': sigma_crit,
        'arcsec_to_kpc': arcsec_to_kpc,
        'apertures': apertures,
    }
