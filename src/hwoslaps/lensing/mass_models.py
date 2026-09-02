"""
Mass model calculations for dark matter subhalos.

This module provides Einstein radius calculations for different mass models
used in subhalo lensing studies: Point Mass, Singular Isothermal Sphere (SIS),
Navarro-Frenk-White (NFW) profiles.
"""

from dataclasses import dataclass

import numpy as np
from astropy import constants as const
from astropy import units as u

from ..constants import ARCSEC_PER_RAD, KM_TO_M, KPC_TO_M, MPC_TO_M

MOLINE_EQ7_C0 = 19.9
MOLINE_EQ7_A1 = -0.195
MOLINE_EQ7_A2 = 0.089
MOLINE_EQ7_A3 = 0.089
MOLINE_EQ7_B = -0.54
MOLINE_EQ7_MIN_MASS_MSUN = 1.0e6
MOLINE_EQ7_MAX_MASS_MSUN = 1.0e12
MOLINE_EQ7_MAX_X_SUB = 1.5


@dataclass(frozen=True)
class LensingGeometryScalars:
    """Resolved scalar geometry and constants for mass-to-lens mappings."""

    z_lens: float
    z_source: float
    d_l_m: float
    d_s_m: float
    d_ls_m: float
    rho_crit_z_lens_kg_m3: float
    sigma_crit_kg_m2: float
    msun_kg: float
    g_si: float
    c_si: float


def _require_positive_finite(value, name):
    """Validate numeric domain for physical parameters."""
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a finite positive number")
    try:
        value_float = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"{name} must be a finite positive number")
    if not np.isfinite(value_float) or value_float <= 0:
        raise ValueError(f"{name} must be a finite positive number")
    return value_float


def _require_bounded(value, name, minimum, maximum):
    """Validate a finite physical parameter against closed bounds."""
    value_float = _require_positive_finite(value, name)
    if value_float < minimum or value_float > maximum:
        raise ValueError(f"{name} must be between {minimum:g} and {maximum:g}")
    return value_float


def _require_nonnegative_finite(value, name):
    """Validate numeric domain for non-negative physical parameters."""
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a finite non-negative number")
    try:
        value_float = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"{name} must be a finite non-negative number")
    if not np.isfinite(value_float) or value_float < 0:
        raise ValueError(f"{name} must be a finite non-negative number")
    return value_float


def _require_ordered_redshifts(z_lens, z_source):
    """Validate lens/source redshifts for a single-plane lens.

    Parameters
    ----------
    z_lens : `float`
        Lens-plane redshift.
    z_source : `float`
        Source-plane redshift.

    Returns
    -------
    z_lens : `float`
        Validated lens-plane redshift.
    z_source : `float`
        Validated source-plane redshift.

    Raises
    ------
    ValueError
        Raised when either redshift is not positive and finite, or when the
        source is not behind the lens.
    """
    z_lens_float = _require_positive_finite(z_lens, "z_lens")
    z_source_float = _require_positive_finite(z_source, "z_source")
    if z_source_float <= z_lens_float:
        raise ValueError("z_source must be greater than z_lens")
    return z_lens_float, z_source_float


def _distance_value_mpc(distance):
    """Return a scalar distance in Mpc.

    Parameters
    ----------
    distance : `object`
        Astropy quantity, PyAuto scalar, or plain numeric distance value.

    Returns
    -------
    distance_mpc : `float`
        Distance in Mpc.
    """
    if hasattr(distance, "to"):
        return float(distance.to(u.Mpc).value)
    if hasattr(distance, "value"):
        return float(distance.value)
    return float(distance)


def angular_diameter_distance_mpc(cosmology, redshift):
    """Return angular diameter distance to Earth.

    Parameters
    ----------
    cosmology : `object`
        Astropy-style or PyAuto cosmology object.
    redshift : `float`
        Redshift of the target plane.

    Returns
    -------
    distance_mpc : `float`
        Angular diameter distance to Earth in Mpc.
    """
    if hasattr(cosmology, "angular_diameter_distance"):
        return _distance_value_mpc(cosmology.angular_diameter_distance(redshift))
    return (
        _distance_value_mpc(
            cosmology.angular_diameter_distance_to_earth_in_kpc_from(redshift)
        )
        / 1000.0
    )


def angular_diameter_distance_z1z2_mpc(cosmology, z1, z2):
    """Return angular diameter distance between two redshifts.

    Parameters
    ----------
    cosmology : `object`
        Astropy-style or PyAuto cosmology object.
    z1 : `float`
        Foreground redshift.
    z2 : `float`
        Background redshift.

    Returns
    -------
    distance_mpc : `float`
        Angular diameter distance between redshifts in Mpc.
    """
    if hasattr(cosmology, "angular_diameter_distance_z1z2"):
        return _distance_value_mpc(cosmology.angular_diameter_distance_z1z2(z1, z2))
    return (
        _distance_value_mpc(
            cosmology.angular_diameter_distance_between_redshifts_in_kpc_from(z1, z2)
        )
        / 1000.0
    )


def hubble_parameter_km_s_mpc(cosmology, redshift):
    """Return the Hubble parameter at a redshift.

    Parameters
    ----------
    cosmology : `object`
        Astropy-style or PyAuto cosmology object.
    redshift : `float`
        Redshift where the Hubble parameter is evaluated.

    Returns
    -------
    hubble : `float`
        Hubble parameter in km/s/Mpc.
    """
    if hasattr(cosmology, "H"):
        hubble = cosmology.H(redshift)
        return float(hubble.value) if hasattr(hubble, "value") else float(hubble)

    if hasattr(cosmology.H0, "value"):
        h0 = float(cosmology.H0.value)
    else:
        h0 = float(cosmology.H0)
    om0 = float(cosmology.Om0)
    ode0 = float(getattr(cosmology, "Ode0", 1.0 - om0))
    return h0 * np.sqrt(om0 * (1.0 + redshift) ** 3 + ode0)


def lensing_geometry_scalars(z_lens, z_source, cosmology):
    """Resolve cosmology-dependent lensing geometry to plain floats."""
    z_lens = float(z_lens)
    z_source = float(z_source)
    d_l_m = angular_diameter_distance_mpc(cosmology, z_lens) * MPC_TO_M
    d_s_m = angular_diameter_distance_mpc(cosmology, z_source) * MPC_TO_M
    d_ls_m = (
        angular_diameter_distance_z1z2_mpc(cosmology, z_lens, z_source)
        * MPC_TO_M
    )
    g_si = float(const.G.value)
    c_si = float(const.c.value)
    h_z_si = (
        hubble_parameter_km_s_mpc(cosmology, z_lens) * KM_TO_M / MPC_TO_M
    )
    rho_crit = 3 * h_z_si**2 / (8 * np.pi * g_si)
    sigma_crit = (c_si**2 / (4 * np.pi * g_si)) * (
        d_s_m / (d_l_m * d_ls_m)
    )
    return LensingGeometryScalars(
        z_lens=z_lens,
        z_source=z_source,
        d_l_m=float(d_l_m),
        d_s_m=float(d_s_m),
        d_ls_m=float(d_ls_m),
        rho_crit_z_lens_kg_m3=float(rho_crit),
        sigma_crit_kg_m2=float(sigma_crit),
        msun_kg=float((1 * u.Msun).to(u.kg).value),
        g_si=g_si,
        c_si=c_si,
    )


def concentration_moline2017_eq7_xp(m200_msun, x_sub, h, xp):
    """Evaluate Moline et al. (2017) Eq. 7 with an array namespace."""
    log_mass_term = xp.log10((m200_msun * h) / 1.0e8)
    polynomial = (
        1.0
        + (MOLINE_EQ7_A1 * log_mass_term)
        + (MOLINE_EQ7_A2 * log_mass_term) ** 2
        + (MOLINE_EQ7_A3 * log_mass_term) ** 3
    )
    radial_factor = 1.0 + MOLINE_EQ7_B * xp.log10(x_sub)
    return MOLINE_EQ7_C0 * polynomial * radial_factor


def concentration_power_law_xp(m200_msun, z, xp):
    """Evaluate the legacy power-law concentration with an array namespace."""
    c0 = 19.9
    m0 = 1.0e8
    alpha = -0.195
    beta = -0.54
    return (
        c0
        * xp.float_power(m200_msun / m0, alpha)
        * xp.float_power(1.0 + z, beta)
    )


def nfw_lensing_parameters_xp(m200_msun, c200, geometry, xp):
    """Return NFW lensing parameters using resolved scalar geometry."""
    m200_kg = m200_msun * geometry.msun_kg
    r200_m = (
        (3 * m200_kg)
        / (4 * xp.pi * 200 * geometry.rho_crit_z_lens_kg_m3)
    ) ** (1 / 3)
    rs_m = r200_m / c200
    f_c = xp.log(1 + c200) - c200 / (1 + c200)
    rho_s = (
        geometry.rho_crit_z_lens_kg_m3
        * (200.0 / 3.0)
        * c200**3
        / f_c
    )
    kappa_s = (rho_s * rs_m) / geometry.sigma_crit_kg_m2
    scale_radius_arcsec = (rs_m / geometry.d_l_m) * ARCSEC_PER_RAD
    return kappa_s, scale_radius_arcsec


def einstein_radius_sis_m200_xp(m200_msun, geometry, xp):
    """Return the SIS Einstein radius using resolved scalar geometry."""
    m200_kg = m200_msun * geometry.msun_kg
    r200_m = (
        (3 * m200_kg)
        / (4 * xp.pi * 200 * geometry.rho_crit_z_lens_kg_m3)
    ) ** (1 / 3)
    sigma_v_squared = geometry.g_si * m200_kg / (2 * r200_m)
    sigma_v_m_s = xp.sqrt(sigma_v_squared)
    theta_e_rad = (
        4.0
        * xp.pi
        * (sigma_v_m_s / geometry.c_si) ** 2
        * (geometry.d_ls_m / geometry.d_s_m)
    )
    return theta_e_rad * ARCSEC_PER_RAD


def einstein_radius_point_mass_xp(m200_msun, geometry, xp):
    """Return the point-mass Einstein radius using resolved geometry."""
    m200_kg = m200_msun * geometry.msun_kg
    theta_e_rad_squared = (
        4 * geometry.g_si * m200_kg * geometry.d_ls_m
    ) / (
        geometry.c_si**2 * geometry.d_l_m * geometry.d_s_m
    )
    return xp.sqrt(theta_e_rad_squared) * ARCSEC_PER_RAD


def einstein_radius_point_mass(mass_msun, z_lens, z_source, cosmology):
    """Calculate Einstein radius for a point mass subhalo.

    This treats the dark matter subhalo as a point mass, which is standard
    for subhalo detection forecasts. The point mass approximation is valid
    when the subhalo is compact compared to its Einstein radius.

    Parameters
    ----------
    mass_msun : float
        Mass of the subhalo in solar masses.
    z_lens : float
        Redshift of the lens plane where the subhalo resides.
    z_source : float
        Redshift of the background source galaxy.
    cosmology : autolens cosmology object, optional
        Cosmology to use. Defaults to PyAutoLens Planck15 if None.

    Returns
    -------
    theta_E_arcsec : float
        Einstein radius in arcseconds.

    Notes
    -----
    For a point mass, the Einstein radius is:

    theta_E = sqrt[4GM/c^2 * D_ls/(D_l * D_s)]

    where M is the mass, and D_l, D_s, D_ls are angular diameter distances.
    """
    mass = _require_positive_finite(mass_msun, "mass_msun")
    z_lens, z_source = _require_ordered_redshifts(z_lens, z_source)

    D_l = angular_diameter_distance_mpc(cosmology, z_lens)
    D_s = angular_diameter_distance_mpc(cosmology, z_source)
    D_ls = angular_diameter_distance_z1z2_mpc(cosmology, z_lens, z_source)

    # Convert to meters using shared constant
    D_l_m = D_l * MPC_TO_M
    D_s_m = D_s * MPC_TO_M
    D_ls_m = D_ls * MPC_TO_M

    # Convert mass to kg
    M_kg = mass * float((1 * u.Msun).to(u.kg).value)

    # Get constants
    G_SI = float(const.G.value)
    c_SI = float(const.c.value)

    # Point mass Einstein radius: theta_E = sqrt[4GM/c^2 * D_ls/(D_l * D_s)]
    theta_E_rad_squared = (4 * G_SI * M_kg * D_ls_m) / (c_SI**2 * D_l_m * D_s_m)
    theta_E_rad = np.sqrt(theta_E_rad_squared)

    # Convert to arcseconds
    theta_E_arcsec = theta_E_rad * ARCSEC_PER_RAD

    return float(theta_E_arcsec)


def concentration_moline2017_eq7(M200_msun, x_sub, h):
    """Moline et al. (2017) Eq. 7 concentration model for subhalos.

    Parameters
    ----------
    M200_msun : `float`
        Subhalo M200 in solar masses.
    x_sub : `float`
        Dimensionless radial position, x_sub = r_sub / R_vir_host.
    h : `float`
        Reduced Hubble parameter H0 / (100 km s^-1 Mpc^-1).

    Returns
    -------
    c200 : `float`
        Concentration parameter r200 / r_s.

    Notes
    -----
    Implements Eq. (7) with Table 2 coefficients for c200:

    c200(m200, x_sub) =
        c0 * [1 + sum_{i=1}^3 (a_i * L)^i] * [1 + b * log10(x_sub)],

    where ``L = log10(m200 / (1e8 h^-1 Msun))``. HWO-SLAPS uses this
    model for subhalo masses from 1e6 to 1e12 Msun and host-centric
    positions 0 < x_sub <= 1.5.
    """
    mass = _require_bounded(
        M200_msun,
        "M200_msun",
        MOLINE_EQ7_MIN_MASS_MSUN,
        MOLINE_EQ7_MAX_MASS_MSUN,
    )
    radial_position = _require_bounded(x_sub, "x_sub", 0.0, MOLINE_EQ7_MAX_X_SUB)
    hubble_reduced = _require_positive_finite(h, "h")

    # Eq. (7) uses log10[m200 / (1e8 h^-1 Msun)] = log10[(m200*h)/1e8].
    log_mass_term = np.log10((mass * hubble_reduced) / 1.0e8)
    polynomial = (
        1.0
        + (MOLINE_EQ7_A1 * log_mass_term)
        + (MOLINE_EQ7_A2 * log_mass_term)**2
        + (MOLINE_EQ7_A3 * log_mass_term)**3
    )
    # Radial correction lowers concentration for larger x_sub because b < 0.
    radial_factor = 1.0 + MOLINE_EQ7_B * np.log10(radial_position)
    return float(MOLINE_EQ7_C0 * polynomial * radial_factor)


def concentration_power_law(M200_msun, z=0.5):
    """Power-law subhalo concentration relation.

    Notes
    -----
    Implementation:
    c = c0 * (M / M0)^alpha * (1 + z)^beta.
    """
    mass = _require_positive_finite(M200_msun, "M200_msun")
    redshift = _require_nonnegative_finite(z, "z")

    c0 = 19.9
    M0 = 1e8
    alpha = -0.195
    beta = -0.54
    c200 = c0 * (mass / M0)**alpha * (1 + redshift)**beta
    return float(c200)


def concentration_mass_relation(
    M200_msun,
    *,
    model="power_law",
    z=0.5,
    x_sub=None,
    h=None,
):
    """Calculate NFW concentration with model provenance.

    Parameters
    ----------
    M200_msun : `float`
        M200 mass in solar masses.
    model : `str`, optional
        Concentration model. Supported values:
        - 'moline2017_eq7'
        - 'power_law'
    z : `float`, optional
        Lens redshift used by the power-law model only.
    x_sub : `float`, optional
        Dimensionless radial position for Moline Eq. 7 model.
    h : `float`, optional
        Reduced Hubble parameter for Moline Eq. 7 model.

    Returns
    -------
    c200 : `float`
        Concentration parameter (r200/rs).

    """
    if model == "moline2017_eq7":
        if x_sub is None:
            raise ValueError("x_sub is required when model='moline2017_eq7'")
        if h is None:
            raise ValueError("h is required when model='moline2017_eq7'")
        # Explicit dispatch keeps concentration provenance unambiguous.
        return concentration_moline2017_eq7(M200_msun, x_sub=x_sub, h=h)
    if model == "power_law":
        return concentration_power_law(M200_msun, z=z)
    raise ValueError(
        "Unsupported concentration model. Supported: 'moline2017_eq7', "
        "'power_law'"
    )


def nfw_scale_parameters(M200_msun, c200, z_lens, cosmology):
    """Calculate NFW scale radius and density.

    Parameters
    ----------
    M200_msun : float
        M200 mass in solar masses.
    c200 : float
        Concentration parameter (r200/rs).
    z_lens : float, optional
        Redshift of the lens. Default is 0.5.
    cosmology : object, optional
        Cosmology object. If None, uses PyAutoLens Planck15.

    Returns
    -------
    rs_kpc : float
        NFW scale radius in kpc.
    rho_s : float
        NFW scale density in kg/m^3.
    """
    mass = _require_positive_finite(M200_msun, "M200_msun")
    concentration = _require_positive_finite(c200, "c200")
    z_lens = _require_positive_finite(z_lens, "z_lens")

    H_z = hubble_parameter_km_s_mpc(cosmology, z_lens)

    # Critical density at z_lens
    H_z_SI = H_z * KM_TO_M / MPC_TO_M  # 1/s
    G_SI = float(const.G.value)
    rho_crit = 3 * H_z_SI**2 / (8 * np.pi * G_SI)  # kg/m^3

    # Calculate r200 from M200
    M200_kg = float((mass * u.Msun).to(u.kg).value)
    r200_m = ((3 * M200_kg) / (4 * np.pi * 200 * rho_crit))**(1/3)

    # Scale radius
    rs_m = r200_m / concentration
    rs_kpc = float((rs_m * u.m).to(u.kpc).value)

    # NFW scale density
    # rho_s = rho_crit * (200/3) * c^3 / [ln(1+c) - c/(1+c)]
    f_c = np.log(1 + concentration) - concentration / (1 + concentration)
    rho_s = rho_crit * (200.0 / 3.0) * concentration**3 / f_c

    return rs_kpc, rho_s


def nfw_lensing_parameters(M200_msun, c200, z_lens, z_source, cosmology):
    """Return the lensing parameters of an NFW subhalo.

    Parameters
    ----------
    M200_msun : `float`
        Subhalo M200 mass in solar masses.
    c200 : `float`
        NFW concentration parameter.
    z_lens : `float`
        Lens-plane redshift.
    z_source : `float`
        Source-plane redshift.
    cosmology : `object`
        Astropy-style or PyAuto cosmology object.

    Returns
    -------
    kappa_s : `float`
        NFW scale convergence.
    scale_radius_arcsec : `float`
        NFW scale radius in arcseconds.
    """
    rs_kpc, rho_s = nfw_scale_parameters(
        M200_msun,
        c200,
        z_lens,
        cosmology,
    )

    D_l_m = angular_diameter_distance_mpc(cosmology, z_lens) * MPC_TO_M
    D_s_m = angular_diameter_distance_mpc(cosmology, z_source) * MPC_TO_M
    D_ls_m = (
        angular_diameter_distance_z1z2_mpc(cosmology, z_lens, z_source)
        * MPC_TO_M
    )

    c_SI = float(const.c.value)
    G_SI = float(const.G.value)
    Sigma_crit = (c_SI**2 / (4 * np.pi * G_SI)) * (
        D_s_m / (D_l_m * D_ls_m)
    )

    rs_m = rs_kpc * KPC_TO_M
    kappa_s = (rho_s * rs_m) / Sigma_crit
    scale_radius_arcsec = (rs_m / D_l_m) * ARCSEC_PER_RAD

    return float(kappa_s), float(scale_radius_arcsec)


def sigma_v_from_m200_sis(M200_msun, z_lens, cosmology):
    """Calculate velocity dispersion for an SIS truncated at r200.

    This assumes the SIS profile extends to r200 where the average
    density equals 200 times the critical density.

    Parameters
    ----------
    M200_msun : float
        M200 mass in solar masses (mass within r200).
    z_lens : float, optional
        Redshift of the lens/subhalo. Default is 0.5.
    cosmology : autolens cosmology object, optional
        If None, uses PyAutoLens Planck15.

    Returns
    -------
    sigma_v : float
        Velocity dispersion in km/s.

    Notes
    -----
    For an SIS truncated at r200, the velocity dispersion is derived from:

    M200 = 2*sigma_v^2*r200/G

    where r200 is calculated from the virial definition at 200 times the
    critical density at the lens redshift.
    """
    mass = _require_positive_finite(M200_msun, "M200_msun")
    z_lens = _require_positive_finite(z_lens, "z_lens")

    H_z = hubble_parameter_km_s_mpc(cosmology, z_lens)

    # Convert H(z) from km/s/Mpc to SI units (1/s)
    H_z_SI = H_z * KM_TO_M / MPC_TO_M  # Convert to 1/s

    # Critical density at z_lens
    G_SI = float(const.G.value)  # Gravitational constant in SI units
    rho_crit = 3 * H_z_SI**2 / (8 * np.pi * G_SI)  # kg/m^3

    # Calculate r200 from M200 definition
    # M200 = (4*pi/3) * r200^3 * 200 * rho_crit
    M200_kg = float((mass * u.Msun).to(u.kg).value)
    r200_m = ((3 * M200_kg) / (4 * np.pi * 200 * rho_crit))**(1/3)  # meters

    # For SIS truncated at r200: M200 = 2*sigma_v^2*r200/G
    # Therefore: sigma_v = sqrt(G * M200 / (2 * r200))
    sigma_v_squared = G_SI * M200_kg / (2 * r200_m)  # m^2/s^2
    sigma_v_m_s = np.sqrt(sigma_v_squared)  # m/s

    # Convert to km/s
    sigma_v_km_s = sigma_v_m_s / 1000.0

    return float(sigma_v_km_s)


def einstein_radius_sis_m200(M200_msun, z_lens, z_source, cosmology):
    """Calculate Einstein radius for an SIS subhalo using M200 mass.

    This method converts M200 to velocity dispersion assuming an SIS
    truncated at r200, then uses the velocity dispersion to calculate
    the SIS Einstein radius.

    Parameters
    ----------
    M200_msun : float
        M200 mass of the subhalo in solar masses.
    z_lens : float
        Redshift of the lens plane where the subhalo resides.
    z_source : float
        Redshift of the background source galaxy.
    cosmology : autolens cosmology object, optional
        Cosmology to use. Defaults to PyAutoLens Planck15 if None.

    Returns
    -------
    einstein_radius : float
        Einstein radius in arcseconds.

    Notes
    -----
    For a Singular Isothermal Sphere (SIS), the Einstein radius is:

    theta_E = 4*pi*(sigma_v/c)^2 * (D_ls/D_s)

    where sigma_v is derived from M200 using virial equilibrium at r200.
    """
    _require_ordered_redshifts(z_lens, z_source)

    # Convert M200 to velocity dispersion
    sigma_v_km_s = float(sigma_v_from_m200_sis(M200_msun, z_lens, cosmology))
    sigma_v_m_s = sigma_v_km_s * 1000.0  # Convert to m/s

    D_ls = angular_diameter_distance_z1z2_mpc(cosmology, z_lens, z_source)
    D_s = angular_diameter_distance_mpc(cosmology, z_source)

    # Calculate SIS Einstein radius
    # theta_E = 4*pi*(sigma_v/c)^2 * (D_ls/D_s)
    c_m_s = float(const.c.value)  # Speed of light in m/s
    theta_E_rad = 4.0 * np.pi * (sigma_v_m_s / c_m_s)**2 * (D_ls / D_s)

    # Convert radians to arcseconds
    theta_E_arcsec = float(theta_E_rad) * ARCSEC_PER_RAD

    return theta_E_arcsec
