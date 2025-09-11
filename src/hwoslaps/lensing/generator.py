"""Main lensing system generation functionality.

This module provides the primary API for generating realistic galaxy-galaxy
strong lensing systems with precisely known subhalo populations.
"""

import numpy as np
import autolens as al
from ..constants import MPC_TO_M, KPC_TO_M, ARCSEC_PER_RAD
from .utils import LensingData, get_einstein_ring_position
from .mass_models import (
    einstein_radius_point_mass,
    einstein_radius_sis_m200,
    nfw_scale_parameters,
    concentration_mass_relation
)
from astropy import constants as const

def generate_lensing_system(config, full_config=None):
    """Generate a complete lensing system from configuration.
    
    This function creates a strong lensing system including grid creation,
    galaxy generation, subhalo injection, and ray-tracing.
    
    Parameters
    ----------
    config : `dict`
        Lensing configuration dictionary containing grid, lens_galaxy,
        source_galaxy, subhalo, and cosmology parameters.
    full_config : `dict`, optional
        Full configuration dictionary containing run_name and other top-level
        parameters. If provided, this will be stored in LensingData.
        
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
    
    >>> lensing_data = generate_lensing_system(config)
    >>> print(f"Lens z={lensing_data.lens_redshift}")
    >>> print(f"Einstein radius: {lensing_data.lens_einstein_radius} arcsec")
    >>> if lensing_data.has_subhalo:
    ...     print(f"Subhalo mass: {lensing_data.subhalo_mass:.1e} M_sun")
    """
    # Extract global seed (required globally by validation)
    global_seed = full_config['global_seed']
    # Create coordinate grid
    grid = _create_grid(config['grid'])
    
    # Create lens galaxy
    lens_galaxy = _create_lens_galaxy(config['lens_galaxy'])
    
    # Create source galaxy
    source_galaxy = _create_source_galaxy(config['source_galaxy'])
    
    # Extract lens and source parameters for unified structure
    lens_config = config['lens_galaxy']
    source_config = config['source_galaxy']
    
    # Create cosmology (explicit in config) before any subhalo calculations
    cosmology = _get_cosmology(config['cosmology'])

    # Initialize subhalo parameters as None
    subhalo_mass = None
    subhalo_model = None
    subhalo_position = None
    subhalo_einstein_radius = None
    subhalo_concentration = None

    # Create subhalo if enabled explicitly
    if 'subhalo' in config and config['subhalo'] is not None and config['subhalo']['enabled']:
        subhalo, subhalo_info = _create_subhalo(
            config['subhalo'], 
            lens_config['redshift'],
            source_config['redshift'],
            lens_galaxy,
            pixel_scale=config['grid']['pixel_scale'],
            cosmology=cosmology,
            global_seed=global_seed
        )
        # Add subhalo to lens galaxy
        lens_galaxy = al.Galaxy(
            redshift=lens_config['redshift'],
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
    
    # Create tracer
    tracer = al.Tracer(
        galaxies=[lens_galaxy, source_galaxy],
        cosmology=cosmology
    )
    
    # Generate lensed image
    lensed_image = tracer.image_2d_from(grid=grid)
    
    # Extract parameters for unified structure
    config_to_store = full_config
    
    return LensingData(
        # Primary data
        image=lensed_image.native,
        grid=grid,
        tracer=tracer,
        
        # System parameters
        pixel_scale=config['grid']['pixel_scale'],
        lens_redshift=lens_config['redshift'],
        source_redshift=source_config['redshift'],
        lens_einstein_radius=lens_config['mass']['einstein_radius'],
        cosmology_name=config['cosmology'],
        
        # Subhalo information (None if not present)
        subhalo_mass=subhalo_mass,
        subhalo_model=subhalo_model,
        subhalo_position=subhalo_position,
        subhalo_einstein_radius=subhalo_einstein_radius,
        subhalo_concentration=subhalo_concentration,
        
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
        Global seed for randomization. If None, uses current random state.
        
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

    # Initialize subhalo_info once to prevent overwriting
    subhalo_info = {
        'mass_msun': mass,
        'model': model,
        'einstein_radius_arcsec': 0.0  # Default value
    }
    
    # Determine position
    position_config = subhalo_config['position']
    position_type = position_config['type']
    
    if position_type == 'random':
        # Set random seed if provided
        if global_seed is not None:
            np.random.seed(global_seed + 1)  # offset for subhalo positioning
        
        # Random angle on Einstein ring
        lens_einstein_radius = lens_galaxy.mass.einstein_radius
        angle_deg = np.random.uniform(0, 360)
        
        # Get scatter in pixels
        scatter_pixels = position_config['scatter_pixels']
        
        # Use existing function with random offset
        offset_pixels = np.random.uniform(-scatter_pixels, scatter_pixels)
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
        subhalo = al.mp.PointMass(
            centre=subhalo_position,
            einstein_radius=einstein_radius
        )
    elif model == 'SIS':
        einstein_radius = einstein_radius_sis_m200(mass, lens_z, source_z, cosmology)
        subhalo_info['einstein_radius_arcsec'] = einstein_radius
        subhalo = al.mp.IsothermalSph(
            centre=subhalo_position,
            einstein_radius=einstein_radius
        )
    elif model == 'NFW':
        # Get concentration
        concentration = concentration_mass_relation(mass, lens_z)
        
        # Get NFW parameters
        rs_kpc, rho_s = nfw_scale_parameters(mass, concentration, lens_z, cosmology)
        
        # Get distances for critical density calculation
        D_l_obj = cosmology.angular_diameter_distance(lens_z)
        D_s_obj = cosmology.angular_diameter_distance(source_z)
        D_ls_obj = cosmology.angular_diameter_distance_z1z2(lens_z, source_z)

        D_l_m = float(D_l_obj.value) * MPC_TO_M
        D_s_m = float(D_s_obj.value) * MPC_TO_M
        D_ls_m = float(D_ls_obj.value) * MPC_TO_M
        
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
        subhalo_info['concentration'] = concentration
    else:
        raise ValueError(f"Unsupported subhalo model: {model}")
        
    return subhalo, subhalo_info

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