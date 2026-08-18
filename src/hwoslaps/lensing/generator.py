"""Main lensing system generation functionality.

This module provides the primary API for generating realistic galaxy-galaxy
strong lensing systems with precisely known subhalo populations.
"""

from collections import OrderedDict
from copy import deepcopy
import os

import autolens as al
import numpy as np

from .image_source import ImageSource, load_source_image_asset
from .mass_models import (
    apply_concentration_offset_dex,
    concentration_mass_relation,
    einstein_radius_point_mass,
    einstein_radius_sis_m200,
    nfw_lensing_parameters,
    nfw_truncation_radius_arcsec,
)
from .utils import LensingData, get_einstein_ring_position


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
    macro_profiles = _macro_profile_mapping(lens_galaxy, lens_config)
    source_galaxy = _create_source_galaxy(source_config)
    source_light_type, source_components, source_image_asset = (
        _source_truth_metadata(source_config, source_galaxy)
    )

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
    subhalo_concentration_offset_dex = None
    subhalo_concentration_pre_offset = None
    subhalo_truncation_mode = None
    subhalo_truncation_tau = None
    subhalo_truncation_radius_arcsec = None

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
            **macro_profiles,
            subhalo=subhalo,
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
            subhalo_concentration_offset_dex = subhalo_info.get(
                'concentration_offset_dex'
            )
            subhalo_concentration_pre_offset = subhalo_info.get(
                'concentration_pre_offset'
            )
        subhalo_kappa_s = subhalo_info.get('kappa_s')
        subhalo_scale_radius_arcsec = subhalo_info.get('scale_radius_arcsec')
        subhalo_profile_parameters = subhalo_info.get('profile_parameters')
        subhalo_truncation_mode = subhalo_info.get('truncation_mode')
        subhalo_truncation_tau = subhalo_info.get('truncation_tau')
        subhalo_truncation_radius_arcsec = subhalo_info.get(
            'truncation_radius_arcsec'
        )

    # Create tracer
    tracer = al.Tracer(
        galaxies=[lens_galaxy, source_galaxy],
        cosmology=cosmology
    )

    # Generate lensed image
    lensed_image = tracer.image_2d_from(grid=grid)

    # Extract parameters for unified structure
    config_to_store = deepcopy(full_config)

    source_light = source_config['light']
    if source_light_type == 'Clumpy':
        source_summary_profile = source_galaxy.host
        source_centre = tuple(source_summary_profile.centre)
        source_ellipticity = tuple(source_summary_profile.ell_comps)
        source_intensity = float(source_summary_profile.intensity)
        source_effective_radius = float(source_summary_profile.effective_radius)
    elif source_light_type == 'Image':
        source_centre = tuple(source_light['centre'])
        source_ellipticity = (0.0, 0.0)
        source_intensity = source_light['total_flux'] * source_light['flux_scale']
        source_effective_radius = source_light['size_scale']
    else:
        source_centre = tuple(source_light['centre'])
        source_ellipticity = tuple(source_light['ell_comps'])
        source_intensity = source_light['intensity']
        source_effective_radius = source_light['effective_radius']

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
        subhalo_concentration_offset_dex=subhalo_concentration_offset_dex,
        subhalo_concentration_pre_offset=subhalo_concentration_pre_offset,
        subhalo_truncation_mode=subhalo_truncation_mode,
        subhalo_truncation_tau=subhalo_truncation_tau,
        subhalo_truncation_radius_arcsec=subhalo_truncation_radius_arcsec,

        # Galaxy parameters
        lens_centre=tuple(lens_config['mass']['centre']),
        lens_ellipticity=tuple(lens_config['mass']['ell_comps']),
        lens_mass_type=lens_config['mass']['type'],
        lens_slope=(
            float(lens_config['mass']['slope'])
            if lens_config['mass']['type'] == 'PowerLaw'
            else None
        ),
        lens_multipoles=(
            {
                order: tuple(components)
                for order, components in lens_config['mass'].get(
                    'multipoles', {}
                ).items()
            }
            or None
        ),
        lens_shear=(
            tuple(lens_config['shear']) if 'shear' in lens_config else None
        ),
        source_centre=source_centre,
        source_ellipticity=source_ellipticity,
        source_intensity=source_intensity,
        source_effective_radius=source_effective_radius,
        source_light_type=source_light_type,
        source_components=source_components,
        source_image_asset=source_image_asset,

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
    return al.Galaxy(
        redshift=lens_config['redshift'],
        **_create_macro_profiles(lens_config),
    )


def _create_macro_profiles(lens_config):
    """Build the ordered macro-profile mapping for a lens configuration."""
    mass_config = lens_config['mass']
    profiles = OrderedDict()
    if mass_config['type'] == 'Isothermal':
        profiles['mass'] = al.mp.Isothermal(
            centre=tuple(mass_config['centre']),
            einstein_radius=mass_config['einstein_radius'],
            ell_comps=tuple(mass_config['ell_comps'])
        )
    elif mass_config['type'] == 'PowerLaw':
        shared = {
            'centre': tuple(mass_config['centre']),
            'einstein_radius': mass_config['einstein_radius'],
            'slope': mass_config['slope'],
        }
        profiles['mass'] = al.mp.PowerLaw(
            ell_comps=tuple(mass_config['ell_comps']),
            **shared,
        )
        for order_name in sorted(mass_config.get('multipoles', {})):
            order = int(order_name[1:])
            profiles[f'multipole_{order_name}'] = al.mp.PowerLawMultipole(
                m=order,
                multipole_comps=tuple(
                    mass_config['multipoles'][order_name]
                ),
                **shared,
            )
    else:
        raise ValueError(f"Unsupported mass profile type: {mass_config['type']}")
    if 'shear' in lens_config:
        profiles['shear'] = al.mp.ExternalShear(
            gamma_1=lens_config['shear'][0],
            gamma_2=lens_config['shear'][1],
        )
    return profiles


def _macro_profile_mapping(lens_galaxy, lens_config):
    """Return the ordered macro profiles attached to a built lens galaxy."""
    names = ['mass']
    names.extend(
        f'multipole_{order}'
        for order in sorted(lens_config['mass'].get('multipoles', {}))
    )
    if 'shear' in lens_config:
        names.append('shear')
    return OrderedDict((name, getattr(lens_galaxy, name)) for name in names)


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
        return al.Galaxy(
            redshift=source_config['redshift'],
            light=source_light
        )
    elif light_config['type'] == 'Sersic':
        source_light = _sersic_from_component(light_config)
        return al.Galaxy(
            redshift=source_config['redshift'],
            light=source_light
        )
    elif light_config['type'] == 'Clumpy':
        flux_scale = float(light_config['flux_scale'])
        size_scale = float(light_config['size_scale'])
        host_config = light_config['host']
        host_centre = np.asarray(host_config['centre'], dtype=float)
        profiles = {
            'host': _sersic_from_component(
                host_config,
                centre=host_centre,
                flux_scale=flux_scale,
                size_scale=size_scale,
            )
        }
        for index, clump_config in enumerate(light_config['clumps']):
            offset = np.asarray(clump_config['centre'], dtype=float) - host_centre
            clump_centre = host_centre + size_scale * offset
            profiles[f'clump_{index}'] = _sersic_from_component(
                clump_config,
                centre=clump_centre,
                flux_scale=flux_scale,
                size_scale=size_scale,
            )
        return al.Galaxy(redshift=source_config['redshift'], **profiles)
    elif light_config['type'] == 'Image':
        asset = load_source_image_asset(light_config['asset_path'])
        source_light = ImageSource.from_asset(
            asset,
            centre=tuple(light_config['centre']),
            rotation_deg=light_config['rotation_deg'],
            total_flux=light_config['total_flux'],
            flux_scale=light_config['flux_scale'],
            size_scale=light_config['size_scale'],
        )
        return al.Galaxy(
            redshift=source_config['redshift'],
            light=source_light
        )
    else:
        raise ValueError(f"Unsupported light profile type: {light_config['type']}")


def _sersic_from_component(
    component,
    *,
    centre=None,
    flux_scale=1.0,
    size_scale=1.0,
):
    """Build one Sersic component after optional joint transforms."""
    if centre is None:
        centre = component['centre']
    return al.lp.Sersic(
        centre=tuple(float(value) for value in centre),
        ell_comps=tuple(component['ell_comps']),
        intensity=component['intensity'] * flux_scale,
        effective_radius=component['effective_radius'] * size_scale,
        sersic_index=component['sersic_index'],
    )


def _component_truth(role, profile):
    """Return one as-built analytic source component provenance record."""
    return {
        'role': role,
        'centre': tuple(float(value) for value in profile.centre),
        'ell_comps': tuple(float(value) for value in profile.ell_comps),
        'intensity': float(profile.intensity),
        'effective_radius': float(profile.effective_radius),
        'sersic_index': float(getattr(profile, 'sersic_index', 1.0)),
    }


def _source_truth_metadata(source_config, source_galaxy):
    """Return source type, as-built components, and image-asset provenance."""
    light_config = source_config['light']
    light_type = light_config['type']
    if light_type in {'Exponential', 'Sersic'}:
        return light_type, [_component_truth('single', source_galaxy.light)], None
    if light_type == 'Clumpy':
        components = [_component_truth('host', source_galaxy.host)]
        components.extend(
            _component_truth(f'clump_{index}', getattr(source_galaxy, f'clump_{index}'))
            for index in range(len(light_config['clumps']))
        )
        return light_type, components, None
    if light_type == 'Image':
        asset = load_source_image_asset(light_config['asset_path'])
        asset_metadata = {
            'asset_path': os.path.abspath(
                os.path.expanduser(light_config['asset_path'])
            ),
            'sha256_16': asset.sha256_16,
            'pixel_scale_arcsec': asset.pixel_scale_arcsec,
            'rotation_deg': float(light_config['rotation_deg']),
            'total_flux': float(light_config['total_flux']),
            'flux_scale': float(light_config['flux_scale']),
            'size_scale': float(light_config['size_scale']),
            'metadata': deepcopy(asset.metadata),
        }
        return light_type, None, asset_metadata
    raise ValueError(f"Unsupported light profile type: {light_type}")


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
    elif model in ('NFW', 'NFWTruncated'):
        if model == 'NFW' and 'truncation' in subhalo_config:
            raise ValueError(
                "lensing.subhalo.truncation is supported only when "
                "lensing.subhalo.model is 'NFWTruncated'"
            )
        concentration, concentration_meta = _resolve_nfw_concentration(
            subhalo_config=subhalo_config,
            mass_msun=mass,
            lens_z=lens_z,
            cosmology=cosmology,
        )

        kappa_s, scale_radius_arcsec = nfw_lensing_parameters(
            mass,
            concentration,
            lens_z,
            source_z,
            cosmology,
        )

        profile_parameters = {
            'centre_0': subhalo_position[0],
            'centre_1': subhalo_position[1],
            'kappa_s': kappa_s,
            'scale_radius': scale_radius_arcsec,
        }

        if model == 'NFW':
            # Create ACTUAL NFW subhalo
            subhalo = al.mp.NFWSph(
                centre=subhalo_position,
                kappa_s=kappa_s,
                scale_radius=scale_radius_arcsec
            )
        else:
            truncation_meta = _resolve_nfw_truncation(
                subhalo_config=subhalo_config,
                scale_radius_arcsec=scale_radius_arcsec,
            )
            truncation_radius = truncation_meta['truncation_radius_arcsec']
            # Baltz, Marshall and Oguri (2009) truncated NFW. kappa_s and the
            # scale radius come from the untruncated path unchanged, so NFW
            # and NFWTruncated differ only by the truncation radius.
            subhalo = al.mp.NFWTruncatedSph(
                centre=subhalo_position,
                kappa_s=kappa_s,
                scale_radius=scale_radius_arcsec,
                truncation_radius=truncation_radius,
            )
            profile_parameters['truncation_radius'] = truncation_radius
            subhalo_info.update(truncation_meta)

        # Add NFW-specific info to the dictionary
        subhalo_info['kappa_s'] = kappa_s
        subhalo_info['scale_radius_arcsec'] = scale_radius_arcsec
        subhalo_info['profile_parameters'] = profile_parameters
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
            "lensing.subhalo.concentration must be a dict when lensing.subhalo.model is "
            "'NFW' or 'NFWTruncated'"
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
        pre_offset = concentration_mass_relation(
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
    elif model == 'power_law':
        # Power-law mode preserves the baseline c(M, z) relation.
        pre_offset = concentration_mass_relation(
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
    elif model == 'explicit':
        # Explicit mode declares c200 directly and uses no relation.
        pre_offset = concentration_mass_relation(
            mass_msun,
            model='explicit',
            c200=concentration_config['c200'],
        )
        metadata = {
            'concentration_model': 'explicit',
            'concentration_x_sub': None,
            'concentration_h': None,
            'concentration_source': 'explicit c200',
        }
    else:
        raise ValueError(
            "lensing.subhalo.concentration.model must be 'moline2017_eq7', "
            "'power_law', or 'explicit'"
        )

    # The offset is applied last, after the model has been evaluated inside
    # its own validity bounds. An absent offset leaves the value untouched.
    offset_dex = concentration_config.get('offset_dex')
    concentration = apply_concentration_offset_dex(pre_offset, offset_dex)
    metadata['concentration_offset_dex'] = (
        None if offset_dex is None else float(offset_dex)
    )
    metadata['concentration_pre_offset'] = pre_offset
    return concentration, metadata


def _resolve_nfw_truncation(subhalo_config, scale_radius_arcsec):
    """Resolve truncated-NFW truncation radius and provenance metadata.

    Parameters
    ----------
    subhalo_config : `dict`
        Subhalo configuration containing a truncation block.
    scale_radius_arcsec : `float`
        NFW scale radius in arcseconds resolved from mass and concentration.

    Returns
    -------
    metadata : `dict`
        Provenance payload with truncation mode, ratio, and radius. The
        ratio is None when the radius is declared directly in arcseconds.
    """
    truncation_config = subhalo_config.get('truncation')
    if not isinstance(truncation_config, dict):
        raise ValueError(
            "lensing.subhalo.truncation must be a dict when lensing.subhalo.model is "
            "'NFWTruncated'"
        )

    mode = truncation_config.get('mode')
    tau = truncation_config.get('tau')
    radius_arcsec = truncation_config.get('radius_arcsec')
    truncation_radius = nfw_truncation_radius_arcsec(
        mode,
        scale_radius_arcsec,
        tau=tau,
        radius_arcsec=radius_arcsec,
    )
    return {
        'truncation_mode': mode,
        'truncation_tau': None if tau is None else float(tau),
        'truncation_radius_arcsec': truncation_radius,
    }


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
