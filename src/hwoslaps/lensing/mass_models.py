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
import autolens as al

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
    # Cosmology must be explicitly provided by caller

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
    M_kg = mass_msun * float((1 * u.Msun).to(u.kg).value)

    # Get constants
    G_SI = float(const.G.value)
    c_SI = float(const.c.value)

    # Point mass Einstein radius: theta_E = sqrt[4GM/c^2 * D_ls/(D_l * D_s)]
    theta_E_rad_squared = (4 * G_SI * M_kg * D_ls_m) / (c_SI**2 * D_l_m * D_s_m)
    theta_E_rad = np.sqrt(theta_E_rad_squared)

    # Convert to arcseconds
    theta_E_arcsec = theta_E_rad * ARCSEC_PER_RAD

    return float(theta_E_arcsec)


def concentration_mass_relation(M200_msun, z=0.5):
    """Calculate NFW concentration parameter from mass.

    Uses the concentration-mass relation for subhalos from
    Moline et al. (2017) and similar studies.

    Parameters
    ----------
    M200_msun : float
        M200 mass in solar masses.
    z : float, optional
        Redshift. Default is 0.5.

    Returns
    -------
    c200 : float
        Concentration parameter (r200/rs).

    Notes
    -----
    For subhalos, concentrations are typically ~2x higher than
    field halos due to tidal stripping of outer regions.
    """
    # Subhalo concentration-mass relation
    # c = c0 * (M/M0)^alpha * (1+z)^beta
    c0 = 19.9  # Normalization at z=0
    M0 = 1e8   # Pivot mass in Msun
    alpha = -0.195  # Mass dependence
    beta = -0.54    # Redshift evolution

    c200 = c0 * (M200_msun / M0)**alpha * (1 + z)**beta

    return float(c200)


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
    # Cosmology must be explicitly provided by caller

    # Get Hubble parameter at z_lens
    H_z_obj = cosmology.H(z_lens)
    H_z = float(H_z_obj.value) if hasattr(H_z_obj, 'value') else float(H_z_obj)

    # Critical density at z_lens
    H_z_SI = H_z * KM_TO_M / MPC_TO_M  # 1/s
    G_SI = float(const.G.value)
    rho_crit = 3 * H_z_SI**2 / (8 * np.pi * G_SI)  # kg/m^3

    # Calculate r200 from M200
    M200_kg = float((M200_msun * u.Msun).to(u.kg).value)
    r200_m = ((3 * M200_kg) / (4 * np.pi * 200 * rho_crit))**(1/3)

    # Scale radius
    rs_m = r200_m / c200
    rs_kpc = float((rs_m * u.m).to(u.kpc).value)

    # NFW scale density
    # rho_s = rho_crit * (200/3) * c^3 / [ln(1+c) - c/(1+c)]
    f_c = np.log(1 + c200) - c200 / (1 + c200)
    rho_s = rho_crit * (200.0 / 3.0) * c200**3 / f_c

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
    # Cosmology must be explicitly provided by caller

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
    M200_kg = float((M200_msun * u.Msun).to(u.kg).value)
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
    # Cosmology must be explicitly provided by caller

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