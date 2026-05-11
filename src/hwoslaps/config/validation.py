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

import math
from typing import Any, Dict

MOLINE_EQ7_MIN_MASS_MSUN = 1.0e6
MOLINE_EQ7_MAX_MASS_MSUN = 1.0e12
MOLINE_EQ7_MAX_X_SUB = 1.5


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


def _require_bounded_positive_number(
    value: Any,
    key_path: str,
    minimum: float,
    maximum: float,
) -> float:
    """Require a finite positive number within closed bounds."""
    value_float = _require_positive_finite_number(value, key_path)
    if value_float < minimum or value_float > maximum:
        raise ValueError(f"{key_path} must be between {minimum:g} and {maximum:g}")
    return value_float


def _require_finite_number(value: Any, key_path: str) -> float:
    """Require a finite scalar number.

    Parameters
    ----------
    value : `object`
        Value to validate.
    key_path : `str`
        Human-readable configuration path for error messages.

    Returns
    -------
    value_float : `float`
        Validated value.

    Raises
    ------
    ValueError
        Raised when ``value`` is not a finite non-boolean number.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{key_path} must be numeric")
    value_float = float(value)
    if not math.isfinite(value_float):
        raise ValueError(f"{key_path} must be finite")
    return value_float


def _require_nonnegative_finite_number(value: Any, key_path: str) -> float:
    """Require a finite scalar number greater than or equal to zero."""
    value_float = _require_finite_number(value, key_path)
    if value_float < 0:
        raise ValueError(f"{key_path} must be non-negative")
    return value_float


def _require_positive_int(value: Any, key_path: str) -> int:
    """Require a positive integer scalar."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{key_path} must be a positive integer")
    if value <= 0:
        raise ValueError(f"{key_path} must be a positive integer")
    return value


def _require_finite_pair(value: Any, key_path: str) -> tuple[float, float]:
    """Require a length-two finite numeric pair."""
    pair = _require_list_length(value, 2, key_path)
    return (
        _require_finite_number(pair[0], f"{key_path}[0]"),
        _require_finite_number(pair[1], f"{key_path}[1]"),
    )


def _require_ell_comps(value: Any, key_path: str) -> tuple[float, float]:
    """Require finite PyAutoLens ellipticity components."""
    e1, e2 = _require_finite_pair(value, key_path)
    if math.hypot(e1, e2) >= 1.0:
        raise ValueError(f"{key_path} magnitude must be less than 1")
    return e1, e2


def validate_top_level(config: Dict[str, Any]) -> None:
    # Top-level required keys
    run_name = _require(config, 'run_name', 'top-level')
    _require_type(run_name, str, 'run_name')

    global_seed = _require(config, 'global_seed', 'top-level')
    if isinstance(global_seed, bool) or not isinstance(global_seed, int):
        raise ValueError("global_seed must be an int")

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
    _require_positive_int(shape[0], 'lensing.grid.shape[0]')
    _require_positive_int(shape[1], 'lensing.grid.shape[1]')
    pixel_scale = _require(grid, 'pixel_scale', 'lensing.grid')
    _require_positive_finite_number(pixel_scale, 'lensing.grid.pixel_scale')

    lens_galaxy = _require(lensing, 'lens_galaxy', 'lensing')
    _require_type(lens_galaxy, dict, 'lensing.lens_galaxy')
    mass = _require(lens_galaxy, 'mass', 'lensing.lens_galaxy')
    _require_type(mass, dict, 'lensing.lens_galaxy.mass')
    mass_type = _require(mass, 'type', 'lensing.lens_galaxy.mass')
    _require_type(mass_type, str, 'lensing.lens_galaxy.mass.type')
    if mass_type != 'Isothermal':
        raise ValueError("Only 'Isothermal' mass profile is supported for lens_galaxy.mass.type")
    _require_finite_pair(
        _require(mass, 'centre', 'lensing.lens_galaxy.mass'),
        'lensing.lens_galaxy.mass.centre',
    )
    einstein_radius = _require(mass, 'einstein_radius', 'lensing.lens_galaxy.mass')
    _require_positive_finite_number(einstein_radius, "lensing.lens_galaxy.mass.einstein_radius")
    _require_ell_comps(
        _require(mass, 'ell_comps', 'lensing.lens_galaxy.mass'),
        'lensing.lens_galaxy.mass.ell_comps',
    )
    lens_redshift_val = _require(lens_galaxy, 'redshift', 'lensing.lens_galaxy')

    source_galaxy = _require(lensing, 'source_galaxy', 'lensing')
    _require_type(source_galaxy, dict, 'lensing.source_galaxy')
    light = _require(source_galaxy, 'light', 'lensing.source_galaxy')
    _require_type(light, dict, 'lensing.source_galaxy.light')
    light_type = _require(light, 'type', 'lensing.source_galaxy.light')
    _require_type(light_type, str, 'lensing.source_galaxy.light.type')
    if light_type != 'Exponential':
        raise ValueError("Only 'Exponential' light profile is supported for source_galaxy.light.type")
    _require_finite_pair(
        _require(light, 'centre', 'lensing.source_galaxy.light'),
        'lensing.source_galaxy.light.centre',
    )
    _require_ell_comps(
        _require(light, 'ell_comps', 'lensing.source_galaxy.light'),
        'lensing.source_galaxy.light.ell_comps',
    )
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
                _require_bounded_positive_number(
                    mass_float,
                    "lensing.subhalo.mass",
                    MOLINE_EQ7_MIN_MASS_MSUN,
                    MOLINE_EQ7_MAX_MASS_MSUN,
                )
                x_sub_val = _require(concentration, 'x_sub', 'lensing.subhalo.concentration')
                _require_bounded_positive_number(
                    x_sub_val,
                    "lensing.subhalo.concentration.x_sub",
                    0.0,
                    MOLINE_EQ7_MAX_X_SUB,
                )

                if 'h' in concentration and concentration['h'] is not None:
                    h_val = concentration['h']
                    _require_positive_finite_number(
                        h_val,
                        "lensing.subhalo.concentration.h",
                    )
        position = _require(subhalo, 'position', 'lensing.subhalo')
        _require_type(position, dict, 'lensing.subhalo.position')
        ptype = _require(position, 'type', 'lensing.subhalo.position')
        if ptype == 'random':
            scatter = _require(position, 'scatter_pixels', 'lensing.subhalo.position')
            _require_nonnegative_finite_number(
                scatter,
                "lensing.subhalo.position.scatter_pixels",
            )
        elif ptype == 'angle':
            angle_val = _require(position, 'angle', 'lensing.subhalo.position')
            _require_finite_number(angle_val, "lensing.subhalo.position.angle")
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
            _require_finite_pair(
                _require(position, 'centre', 'lensing.subhalo.position'),
                'lensing.subhalo.position.centre',
            )
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
    if detection != 'fisher':
        raise ValueError("modeling.detection must be 'fisher'")

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

    supported_fisher_keys = {
        'enable',
        'mode',
        'snr_threshold',
        'include_background_offset',
        'covariance_path',
        'finite_diff',
        'map',
        'mask_mode',
        'include_psf_nuisance',
        'compute_psf_mode_scan',
        'mode_scan_z_tolerance',
        'prior_sigmas',
        'psf_mode_steps',
        'psf_mode_prior_sigmas',
        'psf_mode_selection',
        'psf_basis',
        'fit_psf_mode_selection',
        'scan_psf_mode_selection',
    }
    unsupported_fisher_keys = sorted(set(fisher) - supported_fisher_keys)
    if unsupported_fisher_keys:
        raise ValueError(
            "modeling.fisher contains unsupported keys: "
            + ", ".join(unsupported_fisher_keys)
        )

    mask_mode = fisher.get('mask_mode', 'source_snr')
    _require_type(mask_mode, str, 'modeling.fisher.mask_mode')
    if mask_mode.lower() not in {'source_snr', 'all_pixels'}:
        raise ValueError(
            "modeling.fisher.mask_mode must be one of: 'source_snr', 'all_pixels'"
        )

    include_psf_nuisance = fisher.get('include_psf_nuisance', False)
    _require_type(include_psf_nuisance, bool, 'modeling.fisher.include_psf_nuisance')
    compute_psf_mode_scan = fisher.get('compute_psf_mode_scan', False)
    _require_type(compute_psf_mode_scan, bool, 'modeling.fisher.compute_psf_mode_scan')

    mode_scan_z_tolerance = fisher.get('mode_scan_z_tolerance')
    if mode_scan_z_tolerance is not None:
        _require_positive_finite_number(
            mode_scan_z_tolerance,
            'modeling.fisher.mode_scan_z_tolerance',
        )

    covariance_path = fisher.get('covariance_path')
    if covariance_path is not None:
        _require_type(covariance_path, str, 'modeling.fisher.covariance_path')

    prior_sigmas = fisher.get('prior_sigmas')
    if prior_sigmas is not None:
        _require_type(prior_sigmas, dict, 'modeling.fisher.prior_sigmas')
        for key, value in prior_sigmas.items():
            _require_positive_finite_number(value, f"modeling.fisher.prior_sigmas[{key}]")

    psf_mode_steps = fisher.get('psf_mode_steps')
    if psf_mode_steps is not None:
        _require_type(psf_mode_steps, dict, 'modeling.fisher.psf_mode_steps')
        for key, value in psf_mode_steps.items():
            _require_positive_finite_number(value, f"modeling.fisher.psf_mode_steps[{key}]")

    psf_mode_prior_sigmas = fisher.get('psf_mode_prior_sigmas')
    if psf_mode_prior_sigmas is not None:
        _require_type(psf_mode_prior_sigmas, dict, 'modeling.fisher.psf_mode_prior_sigmas')
        for key, value in psf_mode_prior_sigmas.items():
            _require_positive_finite_number(
                value,
                f"modeling.fisher.psf_mode_prior_sigmas[{key}]",
            )

    def _validate_segment_selection_block(value, path_name):
        if value is None:
            return
        if isinstance(value, str):
            if value.lower() != 'all':
                raise ValueError(f"{path_name} must be 'all', a list of segment ids, or a dict with a 'segments' field")
            return
        if isinstance(value, dict):
            segments = value.get('segments')
            if segments is None:
                raise ValueError(f"{path_name} dict form must contain a 'segments' field")
            _validate_segment_selection_block(segments, f"{path_name}.segments")
            return
        if not isinstance(value, (list, tuple)):
            raise ValueError(f"{path_name} must be 'all', a list of segment ids, or a dict with a 'segments' field")
        for idx, seg_id in enumerate(value):
            if isinstance(seg_id, bool) or not isinstance(seg_id, int) or seg_id < 0:
                raise ValueError(f"{path_name}[{idx}] must be a non-negative integer segment id")

    def _validate_global_mode_block(value, path_name):
        if value is None:
            return
        modes = value.get('mode_nolls') if isinstance(value, dict) else value
        if not isinstance(modes, (list, tuple)):
            raise ValueError(f"{path_name} must be a list of 1-based Noll mode indices or a dict with mode_nolls")
        for idx, mode_idx in enumerate(modes):
            if isinstance(mode_idx, bool) or not isinstance(mode_idx, int) or mode_idx < 1:
                raise ValueError(f"{path_name}[{idx}] must be a 1-based integer Noll index")

    def _validate_segment_hexike_block(value, path_name):
        if value is None:
            return
        if isinstance(value, dict) and ('segments' in value or 'mode_nolls' in value):
            if 'segments' not in value or 'mode_nolls' not in value:
                raise ValueError(f"{path_name} cross-product form must contain both 'segments' and 'mode_nolls'")
            _validate_segment_selection_block(value['segments'], f"{path_name}.segments")
            _validate_global_mode_block(value['mode_nolls'], f"{path_name}.mode_nolls")
            return
        if isinstance(value, dict):
            for seg_id, mode_list in value.items():
                if isinstance(seg_id, bool) or not isinstance(seg_id, int) or seg_id < 0:
                    raise ValueError(f"{path_name} segment keys must be non-negative integers")
                _validate_global_mode_block(mode_list, f"{path_name}[{seg_id}]")
            return
        if not isinstance(value, (list, tuple)):
            raise ValueError(
                f"{path_name} must be either {{segments, mode_nolls}}, a mapping seg->modes, or a list of (seg, mode) pairs"
            )
        for idx, pair in enumerate(value):
            if not isinstance(pair, (list, tuple)) or len(pair) != 2:
                raise ValueError(f"{path_name}[{idx}] must be a (segment, mode_noll) pair")
            seg_id, mode_idx = pair
            if isinstance(seg_id, bool) or not isinstance(seg_id, int) or seg_id < 0:
                raise ValueError(f"{path_name}[{idx}][0] must be a non-negative integer segment id")
            if isinstance(mode_idx, bool) or not isinstance(mode_idx, int) or mode_idx < 1:
                raise ValueError(f"{path_name}[{idx}][1] must be a 1-based integer Noll index")

    def _validate_psf_basis(value, path_name):
        if value is None:
            return
        _require_type(value, dict, path_name)
        for key, block in value.items():
            if key == 'segment_pistons':
                _validate_segment_selection_block(block, f"{path_name}.segment_pistons")
            elif key == 'segment_tiptilts':
                _validate_segment_selection_block(block, f"{path_name}.segment_tiptilts")
            elif key == 'segment_hexikes':
                _validate_segment_hexike_block(block, f"{path_name}.segment_hexikes")
            elif key == 'global_zernikes':
                _validate_global_mode_block(block, f"{path_name}.global_zernikes")
            else:
                raise ValueError(
                    f"{path_name} contains unsupported PSF family '{key}'. Supported families are: segment_pistons, segment_tiptilts, segment_hexikes, global_zernikes"
                )

    if 'psf_mode_selection' in fisher:
        raise ValueError(
            "modeling.fisher.psf_mode_selection is not supported; use modeling.fisher.psf_basis"
        )

    psf_basis = fisher.get('psf_basis')
    fit_psf_mode_selection = fisher.get('fit_psf_mode_selection')
    scan_psf_mode_selection = fisher.get('scan_psf_mode_selection')

    _validate_psf_basis(psf_basis, 'modeling.fisher.psf_basis')
    _validate_psf_basis(fit_psf_mode_selection, 'modeling.fisher.fit_psf_mode_selection')
    _validate_psf_basis(scan_psf_mode_selection, 'modeling.fisher.scan_psf_mode_selection')

    if (include_psf_nuisance or compute_psf_mode_scan) and psf_basis is None:
        raise ValueError(
            'modeling.fisher.psf_basis is required when PSF nuisance fitting or PSF mode scanning is enabled'
        )

    if compute_psf_mode_scan and scan_psf_mode_selection is None:
        raise ValueError(
            'modeling.fisher.scan_psf_mode_selection is required when compute_psf_mode_scan is true'
        )
    return


def validate_or_raise(config: Dict[str, Any]) -> None:
    """Validate complete configuration, or raise ValueError with a clear message."""
    validate_top_level(config)
    validate_lensing_config(config['lensing'])
    validate_psf_config(config['psf'])
    validate_observation_config(config['observation'])
    validate_modeling_config(config['modeling'])
