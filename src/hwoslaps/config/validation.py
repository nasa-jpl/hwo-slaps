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

from typing import Any, Dict
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


def _require_positive_finite_number(value: Any, key_path: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{key_path} must be numeric")
    value_float = float(value)
    if not math.isfinite(value_float):
        raise ValueError(f"{key_path} must be finite")
    if value_float <= 0:
        raise ValueError(f"{key_path} must be positive")
    return value_float


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
    _require_positive_finite_number(einstein_radius, "lensing.lens_galaxy.mass.einstein_radius")
    _require_list_length(_require(mass, 'ell_comps', 'lensing.lens_galaxy.mass'), 2, 'lensing.lens_galaxy.mass.ell_comps')
    lens_redshift_val = _require(lens_galaxy, 'redshift', 'lensing.lens_galaxy')

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
    _require_positive_finite_number(intensity, "lensing.source_galaxy.light.intensity")
    eff_r = _require(light, 'effective_radius', 'lensing.source_galaxy.light')
    _require_positive_finite_number(eff_r, "lensing.source_galaxy.light.effective_radius")
    source_redshift_val = _require(source_galaxy, 'redshift', 'lensing.source_galaxy')
    lens_redshift = _require_positive_finite_number(
        lens_redshift_val,
        "lensing.lens_galaxy.redshift",
    )
    source_redshift = _require_positive_finite_number(
        source_redshift_val,
        "lensing.source_galaxy.redshift",
    )
    if source_redshift <= lens_redshift:
        raise ValueError(
            "Physical-domain error: lensing.source_galaxy.redshift must be greater than "
            "lensing.lens_galaxy.redshift"
        )

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
        except (TypeError, ValueError):
            raise ValueError("lensing.subhalo.mass must be a number")
        if not math.isfinite(mass_float) or mass_float <= 0:
            raise ValueError("lensing.subhalo.mass must be positive")
        if model == 'NFW':
            # NFW runs must declare concentration provenance explicitly.
            concentration = _require(subhalo, 'concentration', 'lensing.subhalo')
            _require_type(concentration, dict, 'lensing.subhalo.concentration')
            concentration_model = _require(concentration, 'model', 'lensing.subhalo.concentration')
            if concentration_model not in {'moline2017_eq7', 'power_law'}:
                raise ValueError(
                    "lensing.subhalo.concentration.model must be one of: "
                    "'moline2017_eq7', 'power_law'"
                )
            if concentration_model == 'moline2017_eq7':
                x_sub_val = _require(concentration, 'x_sub', 'lensing.subhalo.concentration')
                try:
                    x_sub_float = float(x_sub_val)
                except (TypeError, ValueError):
                    raise ValueError("lensing.subhalo.concentration.x_sub must be a number")
                if not math.isfinite(x_sub_float) or x_sub_float <= 0:
                    raise ValueError("lensing.subhalo.concentration.x_sub must be positive")

                if 'h' in concentration and concentration['h'] is not None:
                    h_val = concentration['h']
                    try:
                        h_float = float(h_val)
                    except (TypeError, ValueError):
                        raise ValueError("lensing.subhalo.concentration.h must be a number or null")
                    if not math.isfinite(h_float) or h_float <= 0:
                        raise ValueError("lensing.subhalo.concentration.h must be positive when provided")
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
            # Optional: signed radial offset in pixels (finite numeric).
            if 'offset_pixels' in position:
                off = position['offset_pixels']
                if isinstance(off, bool):
                    raise ValueError(
                        "lensing.subhalo.position.offset_pixels must be a finite number if provided"
                    )
                try:
                    off_float = float(off)
                except (TypeError, ValueError):
                    raise ValueError(
                        "lensing.subhalo.position.offset_pixels must be a finite number if provided"
                    )
                if not math.isfinite(off_float):
                    raise ValueError(
                        "lensing.subhalo.position.offset_pixels must be a finite number if provided"
                    )
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
        segment_hexikes = _require_type(
            _require(aberr, 'segment_hexikes', 'psf.aberrations'),
            dict,
            'psf.aberrations.segment_hexikes'
        )
        for seg_idx, mode_dict in segment_hexikes.items():
            if not isinstance(seg_idx, int) or seg_idx < 0:
                raise ValueError('psf.aberrations.segment_hexikes segment indices must be non-negative integers')
            if not isinstance(mode_dict, dict):
                raise ValueError(
                    f'psf.aberrations.segment_hexikes[{seg_idx}] must be a dict of mode_noll -> coeff_nm'
                )
            for mode_noll, coeff_nm in mode_dict.items():
                if not isinstance(mode_noll, int) or mode_noll < 1:
                    raise ValueError(
                        'psf.aberrations.segment_hexikes mode indices must be 1-based Noll integers (>= 1)'
                    )
                if not isinstance(coeff_nm, (int, float)):
                    raise ValueError(
                        f'psf.aberrations.segment_hexikes[{seg_idx}][{mode_noll}] must be numeric (nm RMS)'
                    )
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
    # modeling.enabled already checked at top-level
    if not modeling['enabled']:
        return

    detection = _require(modeling, 'detection', 'modeling')
    _require_type(detection, str, 'modeling.detection')
    detection = detection.lower()
    if detection not in {'gof', 'chernoff', 'mejiro', 'fisher'}:
        raise ValueError("modeling.detection must be one of: 'gof', 'chernoff', 'mejiro', 'fisher'")

    if detection == 'fisher':
        fisher = _require(modeling, 'fisher', 'modeling')
        _require_type(fisher, dict, 'modeling.fisher')
        mode = _require(fisher, 'mode', 'modeling.fisher')
        _require_type(mode, str, 'modeling.fisher.mode')
        if mode.lower() not in {'local', 'map', 'both'}:
            raise ValueError("modeling.fisher.mode must be one of: 'local', 'map', 'both'")

        snr_threshold = _require(fisher, 'snr_threshold', 'modeling.fisher')
        _require_positive_finite_number(snr_threshold, "modeling.fisher.snr_threshold")

        include_background_offset = _require(
            fisher, 'include_background_offset', 'modeling.fisher'
        )
        _require_type(include_background_offset, bool, 'modeling.fisher.include_background_offset')

        finite_diff = _require(fisher, 'finite_diff', 'modeling.fisher')
        _require_type(finite_diff, dict, 'modeling.fisher.finite_diff')
        for key in (
            'centre_arcsec',
            'einstein_radius_arcsec',
            'ell_comp',
            'source_intensity_frac',
            'source_reff_frac',
        ):
            _require_positive_finite_number(
                _require(finite_diff, key, 'modeling.fisher.finite_diff'),
                f"modeling.fisher.finite_diff.{key}",
            )

        map_cfg = _require(fisher, 'map', 'modeling.fisher')
        _require_type(map_cfg, dict, 'modeling.fisher.map')
        num_angles = _require(map_cfg, 'num_angles', 'modeling.fisher.map')
        if isinstance(num_angles, bool) or not isinstance(num_angles, int) or num_angles <= 0:
            raise ValueError("modeling.fisher.map.num_angles must be a positive integer")
        offset_pixels = _require(map_cfg, 'offset_pixels', 'modeling.fisher.map')
        if isinstance(offset_pixels, bool) or not isinstance(offset_pixels, (int, float)):
            raise ValueError("modeling.fisher.map.offset_pixels must be numeric")
        if not math.isfinite(float(offset_pixels)):
            raise ValueError("modeling.fisher.map.offset_pixels must be finite")

        explicit_positions = map_cfg.get('explicit_positions_yx')
        if explicit_positions is not None:
            if not isinstance(explicit_positions, list):
                raise ValueError("modeling.fisher.map.explicit_positions_yx must be a list when provided")
            for idx, pair in enumerate(explicit_positions):
                if not isinstance(pair, (list, tuple)) or len(pair) != 2:
                    raise ValueError(
                        "modeling.fisher.map.explicit_positions_yx entries must be [y, x] pairs"
                    )
                for jdx, value in enumerate(pair):
                    if isinstance(value, bool) or not isinstance(value, (int, float)):
                        raise ValueError(
                            "modeling.fisher.map.explicit_positions_yx entries must be numeric"
                        )
                    if not math.isfinite(float(value)):
                        raise ValueError(
                            "modeling.fisher.map.explicit_positions_yx entries must be finite"
                        )
        return

    # Legacy methods share these required thresholds.
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
