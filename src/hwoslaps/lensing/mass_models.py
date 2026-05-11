"""
Mass model calculations for dark matter subhalos.

This module provides Einstein radius calculations for different mass models
used in subhalo lensing studies: Point Mass, Singular Isothermal Sphere (SIS),
and Navarro-Frenk-White (NFW) profiles.
"""

import numpy as np
from astropy import constants as const
from astropy import units as u
from ..constants import MPC_TO_M, KM_TO_M, ARCSEC_PER_RAD

MOLINE_EQ7_C0 = 19.9
MOLINE_EQ7_A1 = -0.195
MOLINE_EQ7_A2 = 0.089
MOLINE_EQ7_A3 = 0.089
MOLINE_EQ7_B = -0.54


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

    # Get angular diameter distances
    D_l_obj = cosmology.angular_diameter_distance(z_lens)
    D_s_obj = cosmology.angular_diameter_distance(z_source)
    D_ls_obj = cosmology.angular_diameter_distance_z1z2(z_lens, z_source)

    # Extract numerical values in Mpc
    D_l = float(D_l_obj.value) if hasattr(D_l_obj, 'value') else float(D_l_obj)
    D_s = float(D_s_obj.value) if hasattr(D_s_obj, 'value') else float(D_s_obj)
    D_ls = float(D_ls_obj.value) if hasattr(D_ls_obj, 'value') else float(D_ls_obj)

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
        c0 * [1 + sum_{i=1}^3 a_i * log10(m200 / (1e8 h^-1 Msun))^i]
           * [1 + b * log10(x_sub)].
    """
    mass = _require_positive_finite(M200_msun, "M200_msun")
    radial_position = _require_positive_finite(x_sub, "x_sub")
    hubble_reduced = _require_positive_finite(h, "h")

    # Eq. (7) uses log10[m200 / (1e8 h^-1 Msun)] = log10[(m200*h)/1e8].
    log_mass_term = np.log10((mass * hubble_reduced) / 1.0e8)
    polynomial = (
        1.0
        + MOLINE_EQ7_A1 * log_mass_term
        + MOLINE_EQ7_A2 * log_mass_term**2
        + MOLINE_EQ7_A3 * log_mass_term**3
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
    redshift = _require_positive_finite(1.0 + float(z), "1 + z") - 1.0

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
    """Calculate NFW concentration with explicit model provenance.

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
        "Unsupported concentration model. Supported: 'moline2017_eq7', 'power_law'"
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

    # Get Hubble parameter at z_lens
    H_z_obj = cosmology.H(z_lens)
    H_z = float(H_z_obj.value) if hasattr(H_z_obj, 'value') else float(H_z_obj)

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

    # Get Hubble parameter at z_lens - PyAutoLens returns it in km/s/Mpc
    H_z_obj = cosmology.H(z_lens)
    H_z = float(H_z_obj.value) if hasattr(H_z_obj, 'value') else float(H_z_obj)

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

    # Get angular diameter distances from PyAutoLens (in Mpc)
    D_ls_obj = cosmology.angular_diameter_distance_z1z2(z_lens, z_source)
    D_s_obj = cosmology.angular_diameter_distance(z_source)

    # Extract numerical values
    D_ls = float(D_ls_obj.value) if hasattr(D_ls_obj, 'value') else float(D_ls_obj)
    D_s = float(D_s_obj.value) if hasattr(D_s_obj, 'value') else float(D_s_obj)

    # Calculate SIS Einstein radius
    # theta_E = 4*pi*(sigma_v/c)^2 * (D_ls/D_s)
    c_m_s = float(const.c.value)  # Speed of light in m/s
    theta_E_rad = 4.0 * np.pi * (sigma_v_m_s / c_m_s)**2 * (D_ls / D_s)

    # Convert radians to arcseconds
    theta_E_arcsec = float(theta_E_rad) * ARCSEC_PER_RAD

    return theta_E_arcsec
