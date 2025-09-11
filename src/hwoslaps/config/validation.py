"""Strict configuration validation for HWO-SLAPS.

This module validates that all required configuration values are present and
well-typed, enforcing a fail-fast policy. It centralizes schema checks used by
the pipeline before any module code executes.

Policy enforced (per user requirements):
- Plotting: a global `plotting.enabled` boolean must be present (no defaults).
- Aberrations: `psf.aberrations` must be present; if no aberrations are desired,
  all `enable_*` flags must be set to False explicitly.
- Random seed: a global `global_seed` must be present and is used everywhere.
- Cosmology: `lensing.cosmology` must be explicitly defined.
"""

from typing import Any, Dict, List
import math


def _require(config: Dict[str, Any], key: str, ctx: str = ""):
    if key not in config:
        raise ValueError(f"Missing required key '{key}' in {ctx or 'config'}")
    return config[key]


def _require_type(value: Any, t: Any, key_path: str):
    if not isinstance(value, t):
        raise ValueError(f"Key '{key_path}' must be of type {t.__name__}, got {type(value).__name__}")
    return value


def _require_list_length(value: Any, n: int, key_path: str):
    if not isinstance(value, (list, tuple)) or len(value) != n:
        raise ValueError(f"Key '{key_path}' must be a list/tuple of length {n}")
    return value


def validate_top_level(config: Dict[str, Any]) -> None:
    # Top-level required keys
    run_name = _require(config, 'run_name', 'top-level')
    _require_type(run_name, str, 'run_name')

    global_seed = _require(config, 'global_seed', 'top-level')
    _require_type(global_seed, int, 'global_seed')

    lensing = _require(config, 'lensing', 'top-level')
    _require_type(lensing, dict, 'lensing')

    psf = _require(config, 'psf', 'top-level')
    _require_type(psf, dict, 'psf')

    observation = _require(config, 'observation', 'top-level')
    _require_type(observation, dict, 'observation')

    plotting = _require(config, 'plotting', 'top-level')
    _require_type(plotting, dict, 'plotting')
    enabled = _require(plotting, 'enabled', 'plotting')
    _require_type(enabled, bool, 'plotting.enabled')
    # Always require output_dir to be explicit even if enabled is False
    output_dir = _require(plotting, 'output_dir', 'plotting')
    _require_type(output_dir, (str,), 'plotting.output_dir')

    modeling = _require(config, 'modeling', 'top-level')
    _require_type(modeling, dict, 'modeling')
    modeling_enabled = _require(modeling, 'enabled', 'modeling')
    _require_type(modeling_enabled, bool, 'modeling.enabled')


def validate_lensing_config(lensing: Dict[str, Any]) -> None:
    grid = _require(lensing, 'grid', 'lensing')
    _require_type(grid, dict, 'lensing.grid')
    shape = _require(grid, 'shape', 'lensing.grid')
    _require_list_length(shape, 2, 'lensing.grid.shape')
    pixel_scale = _require(grid, 'pixel_scale', 'lensing.grid')
    if not isinstance(pixel_scale, (int, float)) or pixel_scale <= 0:
        raise ValueError("lensing.grid.pixel_scale must be a positive number")

    lens_galaxy = _require(lensing, 'lens_galaxy', 'lensing')
    _require_type(lens_galaxy, dict, 'lensing.lens_galaxy')
    mass = _require(lens_galaxy, 'mass', 'lensing.lens_galaxy')
    _require_type(mass, dict, 'lensing.lens_galaxy.mass')
    mass_type = _require(mass, 'type', 'lensing.lens_galaxy.mass')
    _require_type(mass_type, str, 'lensing.lens_galaxy.mass.type')
    if mass_type != 'Isothermal':
        raise ValueError("Only 'Isothermal' mass profile is supported for lens_galaxy.mass.type")
    _require_list_length(_require(mass, 'centre', 'lensing.lens_galaxy.mass'), 2, 'lensing.lens_galaxy.mass.centre')
    einstein_radius = _require(mass, 'einstein_radius', 'lensing.lens_galaxy.mass')
    if not isinstance(einstein_radius, (int, float)) or einstein_radius <= 0:
        raise ValueError("lensing.lens_galaxy.mass.einstein_radius must be positive")
    _require_list_length(_require(mass, 'ell_comps', 'lensing.lens_galaxy.mass'), 2, 'lensing.lens_galaxy.mass.ell_comps')
    _require(lens_galaxy, 'redshift', 'lensing.lens_galaxy')

    source_galaxy = _require(lensing, 'source_galaxy', 'lensing')
    _require_type(source_galaxy, dict, 'lensing.source_galaxy')
    light = _require(source_galaxy, 'light', 'lensing.source_galaxy')
    _require_type(light, dict, 'lensing.source_galaxy.light')
    light_type = _require(light, 'type', 'lensing.source_galaxy.light')
    _require_type(light_type, str, 'lensing.source_galaxy.light.type')
    if light_type != 'Exponential':
        raise ValueError("Only 'Exponential' light profile is supported for source_galaxy.light.type")
    _require_list_length(_require(light, 'centre', 'lensing.source_galaxy.light'), 2, 'lensing.source_galaxy.light.centre')
    _require_list_length(_require(light, 'ell_comps', 'lensing.source_galaxy.light'), 2, 'lensing.source_galaxy.light.ell_comps')
    intensity = _require(light, 'intensity', 'lensing.source_galaxy.light')
    if not isinstance(intensity, (int, float)) or intensity <= 0:
        raise ValueError("lensing.source_galaxy.light.intensity must be positive")
    eff_r = _require(light, 'effective_radius', 'lensing.source_galaxy.light')
    if not isinstance(eff_r, (int, float)) or eff_r <= 0:
        raise ValueError("lensing.source_galaxy.light.effective_radius must be positive")
    _require(source_galaxy, 'redshift', 'lensing.source_galaxy')

    cosmology = _require(lensing, 'cosmology', 'lensing')
    _require_type(cosmology, str, 'lensing.cosmology')
    if cosmology not in {'Planck15'}:
        raise ValueError("Unsupported cosmology. Supported: 'Planck15'")

    subhalo = _require(lensing, 'subhalo', 'lensing')
    _require_type(subhalo, dict, 'lensing.subhalo')
    enabled = _require(subhalo, 'enabled', 'lensing.subhalo')
    _require_type(enabled, bool, 'lensing.subhalo.enabled')
    if enabled:
        model = _require(subhalo, 'model', 'lensing.subhalo')
        if model not in {'PointMass', 'SIS', 'NFW'}:
            raise ValueError("lensing.subhalo.model must be one of: 'PointMass', 'SIS', 'NFW'")
        mass_val = _require(subhalo, 'mass', 'lensing.subhalo')
        try:
            mass_float = float(mass_val)
        except Exception:
            raise ValueError("lensing.subhalo.mass must be a number")
        if not math.isfinite(mass_float) or mass_float <= 0:
            raise ValueError("lensing.subhalo.mass must be positive")
        position = _require(subhalo, 'position', 'lensing.subhalo')
        _require_type(position, dict, 'lensing.subhalo.position')
        ptype = _require(position, 'type', 'lensing.subhalo.position')
        if ptype == 'random':
            scatter = _require(position, 'scatter_pixels', 'lensing.subhalo.position')
            if not isinstance(scatter, (int, float)) or scatter < 0:
                raise ValueError("lensing.subhalo.position.scatter_pixels must be non-negative")
        elif ptype == 'angle':
            angle_val = _require(position, 'angle', 'lensing.subhalo.position')
            if not isinstance(angle_val, (int, float)):
                raise ValueError("lensing.subhalo.position.angle must be numeric (degrees)")
            # Optional: offset_pixels must be non-negative if provided
            if 'offset_pixels' in position:
                off = position['offset_pixels']
                if not isinstance(off, (int, float)) or off < 0:
                    raise ValueError("lensing.subhalo.position.offset_pixels must be a non-negative number if provided")
        elif ptype == 'direct':
            _require_list_length(_require(position, 'centre', 'lensing.subhalo.position'), 2, 'lensing.subhalo.position.centre')
        else:
            raise ValueError("lensing.subhalo.position.type must be 'random', 'angle', or 'direct'")


def validate_psf_config(psf: Dict[str, Any]) -> None:
    hres = _require(psf, 'hres_psf', 'psf')
    _require_type(hres, dict, 'psf.hres_psf')
    for k in ('wavelength', 'num_pix', 'num_airy', 'sampling'):
        _require(hres, k, 'psf.hres_psf')
    if hres['wavelength'] <= 0 or hres['num_pix'] <= 0 or hres['num_airy'] <= 0 or hres['sampling'] <= 0:
        raise ValueError("psf.hres_psf numeric parameters must be positive")

    tel = _require(psf, 'telescope', 'psf')
    _require_type(tel, dict, 'psf.telescope')
    for k in ('pupil_diameter', 'focal_length', 'gap_size', 'segment_point_to_point', 'num_rings', 'supersampling_factor'):
        _require(tel, k, 'psf.telescope')

    aberr = _require(psf, 'aberrations', 'psf')
    _require_type(aberr, dict, 'psf.aberrations')
    # Require explicit flags even if all false
    flags = [
        'enable_segment_pistons',
        'enable_segment_tiptilts',
        'enable_segment_hexikes',
        'enable_global_zernikes',
    ]
    for f in flags:
        val = _require(aberr, f, 'psf.aberrations')
        _require_type(val, bool, f'psf.aberrations.{f}')
    use_api = _require(aberr, 'use_api', 'psf.aberrations')
    _require_type(use_api, bool, 'psf.aberrations.use_api')

    # If any flag enabled, require corresponding dict present
    if aberr['enable_segment_pistons']:
        _require_type(_require(aberr, 'segment_pistons', 'psf.aberrations'), dict, 'psf.aberrations.segment_pistons')
    if aberr['enable_segment_tiptilts']:
        _require_type(_require(aberr, 'segment_tiptilts', 'psf.aberrations'), dict, 'psf.aberrations.segment_tiptilts')
    if aberr['enable_segment_hexikes']:
        _require_type(_require(aberr, 'segment_hexikes', 'psf.aberrations'), dict, 'psf.aberrations.segment_hexikes')
    if aberr['enable_global_zernikes']:
        _require_type(_require(aberr, 'global_zernikes', 'psf.aberrations'), dict, 'psf.aberrations.global_zernikes')


def validate_observation_config(observation: Dict[str, Any]) -> None:
    exposure_time = _require(observation, 'exposure_time', 'observation')
    if not isinstance(exposure_time, (int, float)) or exposure_time <= 0:
        raise ValueError("observation.exposure_time must be positive")

    detector = _require(observation, 'detector', 'observation')
    _require_type(detector, dict, 'observation.detector')
    for k in ('gain', 'read_noise', 'dark_current', 'sky_background'):
        v = _require(detector, k, 'observation.detector')
        if not isinstance(v, (int, float)):
            raise ValueError(f"observation.detector.{k} must be numeric")


def validate_modeling_config(modeling: Dict[str, Any]) -> None:
    # modeling.enabled already checked at top-level; if True, require thresholds
    if modeling['enabled']:
        snr_threshold = _require(modeling, 'snr_threshold', 'modeling')
        if not isinstance(snr_threshold, (int, float)) or snr_threshold <= 0:
            raise ValueError("modeling.snr_threshold must be a positive number")
        levels = _require(modeling, 'significance_levels', 'modeling')
        if not isinstance(levels, list) or not all(isinstance(x, (int, float)) and x > 0 for x in levels):
            raise ValueError("modeling.significance_levels must be a list of positive numbers (p-values)")


def validate_or_raise(config: Dict[str, Any]) -> None:
    """Validate complete configuration, or raise ValueError with a clear message."""
    validate_top_level(config)
    validate_lensing_config(config['lensing'])
    validate_psf_config(config['psf'])
    validate_observation_config(config['observation'])
    validate_modeling_config(config['modeling'])


