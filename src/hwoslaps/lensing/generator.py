"""Main lensing system generation functionality.

This module provides the primary API for generating realistic galaxy-galaxy
strong lensing systems with precisely known subhalo populations.
"""

from copy import deepcopy

import numpy as np
import autolens as al
from ..constants import MPC_TO_M, KPC_TO_M, ARCSEC_PER_RAD
from .utils import LensingData, get_einstein_ring_position
from .mass_models import (
    einstein_radius_point_mass,
    einstein_radius_sis_m200,
    nfw_scale_parameters,
    concentration_mass_relation,
    angular_diameter_distance_mpc,
    angular_diameter_distance_z1z2_mpc,
)
from astropy import constants as const


def _coerce_positive_finite_redshift(value, key_path):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{key_path} must be numeric")
    redshift = float(value)
    if not np.isfinite(redshift):
        raise ValueError(f"{key_path} must be finite")
    if redshift <= 0:
        raise ValueError(f"{key_path} must be positive")
    return redshift


def generate_lensing_system(config, full_config):
    """Generate a complete lensing system from configuration.
    
    This function creates a strong lensing system including grid creation,
    galaxy generation, subhalo injection, and ray-tracing.
    
    Parameters
    ----------
    config : `dict`
        Lensing configuration dictionary containing grid, lens_galaxy,
        source_galaxy, subhalo, and cosmology parameters.
    full_config : `dict`
        Full top-level configuration dictionary. Must include
        ``global_seed`` for deterministic subhalo placement and provenance.
        
    Returns
    -------
    lensing_data : `LensingData`
        Complete lensing system data with unified structure providing
        direct access to all system parameters.
        
    Notes
    -----
    The returned LensingData object contains all information in a flat
    structure with direct property access, eliminating the need to navigate
    nested dictionaries for basic system information.
    
    Examples
    --------
    Generate a lensing system and access key properties:
    
    >>> full_config = {"global_seed": 1, "run_name": "example"}
    >>> lensing_data = generate_lensing_system(config, full_config=full_config)
    >>> print(f"Lens z={lensing_data.lens_redshift}")
    >>> print(f"Einstein radius: {lensing_data.lens_einstein_radius} arcsec")
    >>> if lensing_data.has_subhalo:
    ...     print(f"Subhalo mass: {lensing_data.subhalo_mass:.1e} M_sun")
    """
    if not isinstance(full_config, dict):
        raise ValueError("full_config must be a dict for generate_lensing_system")
    if 'global_seed' not in full_config:
        raise ValueError("Missing required key 'global_seed' in full_config")
    global_seed = full_config['global_seed']
    if isinstance(global_seed, bool) or not isinstance(global_seed, int):
        raise ValueError("full_config.global_seed must be an int")
    # Create coordinate grid
    grid = _create_grid(config['grid'])
    
    # Extract lens and source parameters for unified structure
    lens_config = config['lens_galaxy']
    source_config = config['source_galaxy']
    lens_redshift = _coerce_positive_finite_redshift(
        lens_config['redshift'], "lensing.lens_galaxy.redshift"
    )
    source_redshift = _coerce_positive_finite_redshift(
        source_config['redshift'], "lensing.source_galaxy.redshift"
    )
    if source_redshift <= lens_redshift:
        raise ValueError(
            "Physical-domain error: lensing.source_galaxy.redshift must be greater than "
            "lensing.lens_galaxy.redshift"
        )
    lens_config = {**lens_config, 'redshift': lens_redshift}
    source_config = {**source_config, 'redshift': source_redshift}

    # Create lens and source galaxies from validated redshifts.
    lens_galaxy = _create_lens_galaxy(lens_config)
    source_galaxy = _create_source_galaxy(source_config)
    
    # Create cosmology (explicit in config) before any subhalo calculations
    cosmology = _get_cosmology(config['cosmology'])

    # Initialize subhalo parameters as None
    subhalo_mass = None
    subhalo_model = None
    subhalo_position = None
    subhalo_einstein_radius = None
    subhalo_concentration = None
    subhalo_concentration_model = None
    subhalo_concentration_x_sub = None
    subhalo_concentration_h = None
    subhalo_concentration_source = None
    subhalo_kappa_s = None
    subhalo_scale_radius_arcsec = None
    subhalo_profile_parameters = None

    # Create subhalo if enabled explicitly
    if 'subhalo' in config and config['subhalo'] is not None and config['subhalo']['enabled']:
        subhalo, subhalo_info = _create_subhalo(
            config['subhalo'],
            lens_redshift,
            source_redshift,
            lens_galaxy,
            pixel_scale=config['grid']['pixel_scale'],
            cosmology=cosmology,
            global_seed=global_seed
        )
        # Add subhalo to lens galaxy
        lens_galaxy = al.Galaxy(
            redshift=lens_redshift,
            mass=lens_galaxy.mass,
            subhalo=subhalo
        )
        
        # Extract subhalo parameters from subhalo_info
        subhalo_mass = subhalo_info['mass_msun']
        subhalo_model = subhalo_info['model']
        subhalo_position = subhalo_info['position_arcsec']
        subhalo_einstein_radius = subhalo_info['einstein_radius_arcsec']
        if 'concentration' in subhalo_info:
            subhalo_concentration = subhalo_info['concentration']
            subhalo_concentration_model = subhalo_info.get('concentration_model')
            subhalo_concentration_x_sub = subhalo_info.get('concentration_x_sub')
            subhalo_concentration_h = subhalo_info.get('concentration_h')
            subhalo_concentration_source = subhalo_info.get('concentration_source')
        subhalo_kappa_s = subhalo_info.get('kappa_s')
        subhalo_scale_radius_arcsec = subhalo_info.get('scale_radius_arcsec')
        subhalo_profile_parameters = subhalo_info.get('profile_parameters')
    
    # Create tracer
    tracer = al.Tracer(
        galaxies=[lens_galaxy, source_galaxy],
        cosmology=cosmology
    )
    
    # Generate lensed image
    lensed_image = tracer.image_2d_from(grid=grid)
    
    # Extract parameters for unified structure
    config_to_store = deepcopy(full_config)
    
    return LensingData(
        # Primary data
        image=lensed_image.native,
        grid=grid,
        tracer=tracer,
        
        # System parameters
        pixel_scale=config['grid']['pixel_scale'],
        lens_redshift=lens_redshift,
        source_redshift=source_redshift,
        lens_einstein_radius=lens_config['mass']['einstein_radius'],
        cosmology_name=config['cosmology'],
        
        # Subhalo information (None if not present)
        subhalo_mass=subhalo_mass,
        subhalo_model=subhalo_model,
        subhalo_position=subhalo_position,
        subhalo_einstein_radius=subhalo_einstein_radius,
        subhalo_concentration=subhalo_concentration,
        subhalo_concentration_model=subhalo_concentration_model,
        subhalo_concentration_x_sub=subhalo_concentration_x_sub,
        subhalo_concentration_h=subhalo_concentration_h,
        subhalo_concentration_source=subhalo_concentration_source,
        subhalo_kappa_s=subhalo_kappa_s,
        subhalo_scale_radius_arcsec=subhalo_scale_radius_arcsec,
        subhalo_profile_parameters=subhalo_profile_parameters,
        
        # Galaxy parameters
        lens_centre=tuple(lens_config['mass']['centre']),
        lens_ellipticity=tuple(lens_config['mass']['ell_comps']),
        source_centre=tuple(source_config['light']['centre']),
        source_ellipticity=tuple(source_config['light']['ell_comps']),
        source_intensity=source_config['light']['intensity'],
        source_effective_radius=source_config['light']['effective_radius'],
        
        # Provenance
        config=config_to_store
    )


def _create_grid(grid_config):
    """
    Create PyAutoLens coordinate grid.
    
    Parameters
    ----------
    grid_config : dict
        Grid configuration with 'shape' and 'pixel_scale' keys.
        
    Returns
    -------
    grid : al.Grid2D
        PyAutoLens grid object.
    """
    return al.Grid2D.uniform(
        shape_native=tuple(grid_config['shape']),
        pixel_scales=grid_config['pixel_scale']
    )


def _create_lens_galaxy(lens_config):
    """
    Create lens galaxy from configuration.
    
    Parameters
    ----------
    lens_config : dict
        Lens galaxy configuration including redshift and mass profile.
        
    Returns
    -------
    lens_galaxy : al.Galaxy
        PyAutoLens galaxy object representing the lens.
    """
    mass_config = lens_config['mass']
    
    # Create mass profile
    if mass_config['type'] == 'Isothermal':
        lens_mass = al.mp.Isothermal(
            centre=tuple(mass_config['centre']),
            einstein_radius=mass_config['einstein_radius'],
            ell_comps=tuple(mass_config['ell_comps'])
        )
    else:
        raise ValueError(f"Unsupported mass profile type: {mass_config['type']}")
    
    return al.Galaxy(
        redshift=lens_config['redshift'],
        mass=lens_mass
    )


def _create_source_galaxy(source_config):
    """
    Create source galaxy from configuration.
    
    Parameters
    ----------
    source_config : dict
        Source galaxy configuration including redshift and light profile.
        
    Returns
    -------
    source_galaxy : al.Galaxy
        PyAutoLens galaxy object representing the source.
    """
    light_config = source_config['light']
    
    # Create light profile
    if light_config['type'] == 'Exponential':
        source_light = al.lp.Exponential(
            centre=tuple(light_config['centre']),
            ell_comps=tuple(light_config['ell_comps']),
            intensity=light_config['intensity'],
            effective_radius=light_config['effective_radius']
        )
    else:
        raise ValueError(f"Unsupported light profile type: {light_config['type']}")
    
    return al.Galaxy(
        redshift=source_config['redshift'],
        light=source_light
    )


def _create_subhalo(subhalo_config, lens_z, source_z, lens_galaxy, pixel_scale, cosmology, global_seed=None):
    """
    Create subhalo mass profile and truth information.
    
    Parameters
    ----------
    subhalo_config : dict
        Subhalo configuration including mass, model, and position.
    lens_z : float
        Lens redshift.
    source_z : float
        Source redshift.
    lens_galaxy : al.Galaxy
        Lens galaxy object to get Einstein radius for positioning.
    pixel_scale : float
        Pixel scale in arcseconds per pixel.
    global_seed : int, optional
        Global seed for randomization. If provided, subhalo placement uses
        ``global_seed + 1`` as a dedicated local RNG stream. If None, the
        placement RNG is initialized from entropy.
        
    Returns
    -------
    subhalo : al.mp.MassProfile
        PyAutoLens mass profile for the subhalo.
    subhalo_info : dict
        Truth information about the subhalo.
    """
    mass = float(subhalo_config['mass'])
    model = subhalo_config['model']
    # Use cosmology passed from parent context

    # Initialize subhalo_info once to prevent overwriting.
    # Einstein radius is model-dependent and not defined for NFW.
    subhalo_info = {
        'mass_msun': mass,
        'model': model,
        'einstein_radius_arcsec': None,
    }
    
    # Determine position
    position_config = subhalo_config['position']
    position_type = position_config['type']
    
    if position_type == 'random':
        # Use a local RNG stream to avoid mutating NumPy global RNG state.
        if global_seed is not None:
            # Offset for subhalo positioning.
            rng = np.random.default_rng(global_seed + 1)
        else:
            rng = np.random.default_rng()
        
        # Random angle on Einstein ring
        lens_einstein_radius = lens_galaxy.mass.einstein_radius
        angle_deg = float(rng.uniform(0.0, 360.0))
        
        # Get scatter in pixels
        scatter_pixels = position_config['scatter_pixels']
        
        # Use existing function with random offset
        offset_pixels = float(rng.uniform(-scatter_pixels, scatter_pixels))
        subhalo_position = get_einstein_ring_position(
            angle_deg=angle_deg,
            einstein_radius=lens_einstein_radius,
            offset_pixels=offset_pixels,
            pixel_scale=pixel_scale
        )
    elif position_type == 'angle':
        # Fixed-angle placement on or near the Einstein ring
        lens_einstein_radius = lens_galaxy.mass.einstein_radius
        angle_deg = float(position_config['angle'])
        # Optional radial offset in pixels (default 0)
        offset_pixels = float(position_config.get('offset_pixels', 0.0))
        subhalo_position = get_einstein_ring_position(
            angle_deg=angle_deg,
            einstein_radius=lens_einstein_radius,
            offset_pixels=offset_pixels,
            pixel_scale=pixel_scale
        )
        
    elif position_type == 'direct':
        # Direct placement for specific tests
        subhalo_position = tuple(position_config['centre'])
    else:
        raise ValueError(f"Unknown position type: {position_type}")

    subhalo_info['position_arcsec'] = subhalo_position
    
    # Create PyAutoLens mass profile
    if model == 'PointMass':
        einstein_radius = einstein_radius_point_mass(mass, lens_z, source_z, cosmology)
        subhalo_info['einstein_radius_arcsec'] = einstein_radius
        subhalo_info['profile_parameters'] = {
            'centre_0': subhalo_position[0],
            'centre_1': subhalo_position[1],
            'einstein_radius': einstein_radius,
        }
        subhalo = al.mp.PointMass(
            centre=subhalo_position,
            einstein_radius=einstein_radius
        )
    elif model == 'SIS':
        einstein_radius = einstein_radius_sis_m200(mass, lens_z, source_z, cosmology)
        subhalo_info['einstein_radius_arcsec'] = einstein_radius
        subhalo_info['profile_parameters'] = {
            'centre_0': subhalo_position[0],
            'centre_1': subhalo_position[1],
            'einstein_radius': einstein_radius,
        }
        subhalo = al.mp.IsothermalSph(
            centre=subhalo_position,
            einstein_radius=einstein_radius
        )
    elif model == 'NFW':
        concentration, concentration_meta = _resolve_nfw_concentration(
            subhalo_config=subhalo_config,
            mass_msun=mass,
            lens_z=lens_z,
            cosmology=cosmology,
        )
        
        # Get NFW parameters
        rs_kpc, rho_s = nfw_scale_parameters(mass, concentration, lens_z, cosmology)
        
        D_l_m = angular_diameter_distance_mpc(cosmology, lens_z) * MPC_TO_M
        D_s_m = angular_diameter_distance_mpc(cosmology, source_z) * MPC_TO_M
        D_ls_m = (
            angular_diameter_distance_z1z2_mpc(cosmology, lens_z, source_z)
            * MPC_TO_M
        )
        
        # Critical surface density calculated robustly in SI units
        c_SI = float(const.c.value)
        G_SI = float(const.G.value)
        Sigma_crit = (c_SI**2 / (4 * np.pi * G_SI)) * (D_s_m / (D_l_m * D_ls_m))
        
        # Calculate kappa_s
        rs_m = rs_kpc * KPC_TO_M
        kappa_s = (rho_s * rs_m) / Sigma_crit
        
        # Convert scale radius to arcsec
        scale_radius_arcsec = (rs_m / D_l_m) * ARCSEC_PER_RAD
        
        # Create ACTUAL NFW subhalo
        subhalo = al.mp.NFWSph(
            centre=subhalo_position,
            kappa_s=kappa_s,
            scale_radius=scale_radius_arcsec
        )
        
        # Add NFW-specific info to the dictionary
        subhalo_info['kappa_s'] = kappa_s
        subhalo_info['scale_radius_arcsec'] = scale_radius_arcsec
        subhalo_info['profile_parameters'] = {
            'centre_0': subhalo_position[0],
            'centre_1': subhalo_position[1],
            'kappa_s': kappa_s,
            'scale_radius': scale_radius_arcsec,
        }
        subhalo_info['concentration'] = concentration
        subhalo_info.update(concentration_meta)
    else:
        raise ValueError(f"Unsupported subhalo model: {model}")
        
    return subhalo, subhalo_info


def _resolve_nfw_concentration(subhalo_config, mass_msun, lens_z, cosmology):
    """Resolve NFW concentration and provenance metadata.

    Parameters
    ----------
    subhalo_config : `dict`
        Subhalo configuration containing a concentration block.
    mass_msun : `float`
        Subhalo mass in solar masses.
    lens_z : `float`
        Lens-plane redshift for power-law concentration mode.
    cosmology : `object`
        Cosmology object used to infer ``h`` when configured as null.

    Returns
    -------
    concentration : `float`
        Concentration value used for the NFW profile.
    metadata : `dict`
        Provenance payload with model name and model inputs.
    """
    concentration_config = subhalo_config.get('concentration')
    if not isinstance(concentration_config, dict):
        raise ValueError(
            "lensing.subhalo.concentration must be a dict when lensing.subhalo.model is 'NFW'"
        )

    model = concentration_config.get('model')
    if model == 'moline2017_eq7':
        # Eq. (7) mode requires x_sub; h may be explicit or inferred.
        x_sub = float(concentration_config['x_sub'])
        h_value = concentration_config.get('h')
        if h_value is None:
            h_value = _infer_reduced_h(cosmology)
        else:
            h_value = float(h_value)
        concentration = concentration_mass_relation(
            mass_msun,
            model='moline2017_eq7',
            x_sub=x_sub,
            h=h_value,
        )
        metadata = {
            'concentration_model': 'moline2017_eq7',
            'concentration_x_sub': x_sub,
            'concentration_h': h_value,
            'concentration_source': 'Moline2017 Eq7 Table2',
        }
        return concentration, metadata

    if model == 'power_law':
        # Power-law mode preserves the baseline c(M, z) relation.
        concentration = concentration_mass_relation(
            mass_msun,
            model='power_law',
            z=lens_z,
        )
        metadata = {
            'concentration_model': 'power_law',
            'concentration_x_sub': None,
            'concentration_h': None,
            'concentration_source': 'power_law',
        }
        return concentration, metadata

    raise ValueError(
        "lensing.subhalo.concentration.model must be 'moline2017_eq7' or 'power_law'"
    )


def _infer_reduced_h(cosmology):
    """Infer reduced Hubble parameter ``h`` from the configured cosmology.

    Parameters
    ----------
    cosmology : `object`
        Cosmology object from PyAutoLens.

    Returns
    -------
    h : `float`
        Reduced Hubble parameter ``h = H0 / 100``.
    """
    # This project currently validates to Planck15 only; use H0 directly.
    if hasattr(cosmology, 'H'):
        H0_value = cosmology.H(0.0)
        if hasattr(H0_value, 'value'):
            H0_value = H0_value.value
        H0_float = float(H0_value)
        if np.isfinite(H0_float) and H0_float > 0:
            return H0_float / 100.0

    # Fallback path for cosmology objects exposing H0 instead of H(z).
    if hasattr(cosmology, 'H0'):
        H0_value = cosmology.H0
        if hasattr(H0_value, 'value'):
            H0_value = H0_value.value
        H0_float = float(H0_value)
        if np.isfinite(H0_float) and H0_float > 0:
            return H0_float / 100.0

    # Stable default for Planck15 in case of unexpected backend behavior.
    return 0.6774


def _get_cosmology(cosmology_name):
    """
    Get PyAutoLens cosmology object.
    
    Parameters
    ----------
    cosmology_name : str
        Name of the cosmology model.
        
    Returns
    -------
    cosmology : al.cosmo object
        PyAutoLens cosmology object.
    """
    if cosmology_name == 'Planck15':
        return al.cosmo.Planck15()
    else:
        raise ValueError(f"Unsupported cosmology: {cosmology_name}")
