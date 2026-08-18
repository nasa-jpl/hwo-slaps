#!/usr/bin/env python
"""Derive the immutable ``hwo_eac1_hri_reference_v1`` observing reference.

Every scalar in the emitted artifact comes from a hand-checkable pure
function in this module. Two quantities need the engine and import it
lazily, so the photometric chain stays importable on a bare
interpreter: the collecting-area integral through
``hwoslaps.psf.telescope_models.create_hcipy_telescope``, and the
per-scene source renders through ``hwoslaps.lensing.generator``.

Scene light normalizations are continuous surface-brightness
amplitudes, never detected totals. The observation layer reads rendered
samples directly as per-pixel e-/s, so each scene patch is solved on
the exact production grid until the unlensed discrete pixel sum equals
the derived detected rate; the artifact records the target rate, the
realized rate, and the pixel area that maps between the two
conventions. Production configurations therefore keep
``observation.throughput: 1.0`` and the throughput chain lives in
provenance metadata only. The detector read noise is likewise the
EFFECTIVE combined-image value, because the noise model applies exactly
one squared read-noise term.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import date
from functools import lru_cache
import math
from pathlib import Path

import numpy as np
import yaml


SCRIPT_NAME = 'derive_hwo_eac1_hri_reference.py'
SCRIPT_VERSION = '2'
REFERENCE_NAME = 'hwo_eac1_hri_reference_v1'

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MASTER_CONFIG_RELPATH = 'configs/master_config.yaml'
MASTER_CONFIG_PATH = PROJECT_ROOT / MASTER_CONFIG_RELPATH
DEFAULT_OUTPUT_PATH = (
    PROJECT_ROOT / 'configs' / 'observing' / f'{REFERENCE_NAME}.yaml'
)

AB_ZERO_POINT_JY = 3631.0
"""Flux density of a zero AB magnitude source, in janskys."""

JANSKY_TO_SI = 1.0e-26
"""One jansky expressed in W m^-2 Hz^-1."""

PLANCK_H = 6.62607015e-34
"""Planck constant in J s, exact under the 2019 SI definition."""

BAND_NAME = 'HRI-V'
BAND_CENTRE_M = 5.0e-7
BAND_FRACTIONAL_WIDTH = 0.20
BAND_LAMBDA_MIN_M = 4.5e-7
BAND_LAMBDA_MAX_M = 5.5e-7

SOURCE_MAG_F814W_AB = 24.3
SOURCE_BAND = 'F814W'

SED_MODES = ('flat_fnu', 'declared_color', 'vband_photometry')
SED_MODE_DESCRIPTIONS = {
    'flat_fnu': 'flat-f_nu reference assumption',
    'declared_color': (
        'declared AB color offset applied to the F814W anchor; positive '
        'values are fainter at 500 nm'
    ),
    'vband_photometry': 'measured V-band source photometry used directly',
}

DEFAULT_EXPOSURE_S = 2000.0
DEFAULT_SYSTEM_QE = 0.21
DEFAULT_SKY_MAG = 23.0

OPTIMISTIC_OPTICAL_THROUGHPUT = 0.56
OPTIMISTIC_DETECTOR_QE = 0.9

DETECTOR_GAIN_E_PER_ADU = 1.0
DARK_CURRENT_E_PER_PIX_S = 0.002
READ_NOISE_PER_READ_E = 2.5
N_READS = 2
QUALIFICATION_SKY_BACKGROUND_E_PER_PIX_S = 1.0

SERSIC_B1 = 1.6783886549215685
"""AutoGalaxy ``sersic_constant`` polynomial evaluated at ``n = 1``.

This is the constant the installed profile code uses, so it is the one
that reproduces the frozen qualification flux. The exact root of
``(1 + b) exp(-b) = 1/2`` is 1.6783469900166605 and misses that flux by
8e-6 relative.
"""

QUALIFICATION_TOTAL_FLUX = 0.289151264
QUALIFICATION_INTENSITY = 2.0
QUALIFICATION_EFFECTIVE_RADIUS = 0.11
CLOSED_FORM_ABS_TOLERANCE = 1.0e-9
PIXEL_SCALE_ABS_TOLERANCE = 1.0e-12
DISCRETE_MAPPING_TOLERANCE = 1.0e-2
"""Accepted ``pixel_area * discrete_sum / angular_integral`` departure.

The production grid samples every profile at sub-pixel positions, so
the discrete sum reproduces the closed-form angular integral to well
inside a percent for each scene family. A wider miss means the grid,
the image asset, or the profile changed, not sampling.
"""

AREA_RATIO_BOUNDS = (0.95, 1.005)
"""Accepted mask-to-hexagon area ratio window.

The supersampled aperture carries fractional edge coverage, so at the
production grid resolution the mask integral can land a few 1e-4 above
the gapless hexagon area; the measured ratio is 1.00025 with the 512
pixel, 4x supersampled grid. A ratio outside this window means the
geometry or the integral is wrong, not edge discretization.
"""

SCENE_LIGHT_BASELINES = {
    'scene1_smooth_ring': {
        'scene_config': 'configs/scenes/scene1_smooth_ring.yaml',
        'light_type': 'Exponential',
    },
    'scene2_clumpy': {
        'scene_config': 'configs/scenes/scene2_clumpy.yaml',
        'light_type': 'Clumpy',
    },
    'scene3_bow_dot': {
        'scene_config': 'configs/scenes/scene3_bow_dot.yaml',
        'light_type': 'Exponential',
    },
    'scene4_cosmos': {
        'scene_config': 'configs/scenes/scene4_cosmos.yaml',
        'light_type': 'Image',
    },
    'scene5_flex_macro': {
        'scene_config': 'configs/scenes/scene5_flex_macro.yaml',
        'light_type': 'Exponential',
    },
    'scene5_ablation_sie_fit': {
        'scene_config': 'configs/scenes/scene5_ablation_sie_fit.yaml',
        'light_type': 'Exponential',
    },
}

SCENE_NORMALIZED_FIELD = {
    'Exponential': 'intensity',
    'Clumpy': 'flux_scale',
    'Image': 'total_flux',
}
"""Config leaf each source family scales exactly linearly with."""

CITATIONS = {
    'S1': (
        'Liu, Levine, Noecker, Feinberg, Stark et al., "Early Architecture '
        'Concepts for the Habitable Worlds Observatory", JATIS 12(4) 041017 '
        '(2026), arXiv:2602.11046'
    ),
    'S3': (
        'Stark, Steiger, Tokadjian, Savransky et al., "Cross-Model Validation '
        'of Coronagraphic Exposure Time Calculators for HWO" (2025), '
        'arXiv:2502.18556'
    ),
    'S4': 'LUVOIR Final Report (2019), arXiv:1912.06219, HDI Table 8-5',
    'S6': (
        'Stark, Latouf, Mandell, Young, "Optimized Bandpasses for the HWO '
        'ExoEarth Survey" (2024), arXiv:2404.05654'
    ),
    'L1': (
        'Newton et al. 2011, ApJ 734, 104, arXiv:1104.2608; mean '
        'magnification-corrected SLACS source magnitude, median mu 8.8'
    ),
}


def _require_positive(value, name):
    """Return one strictly positive finite float or fail loudly."""
    number = float(value)
    if not math.isfinite(number) or number <= 0.0:
        raise ValueError(f'{name} must be a positive finite number, got {value!r}.')
    return number


def ab_mag_to_fnu_jy(m_ab):
    """Convert an AB magnitude into a flux density.

    Parameters
    ----------
    m_ab : `float`
        AB magnitude.

    Returns
    -------
    fnu_jy : `float`
        Flux density in janskys.
    """
    magnitude = float(m_ab)
    if not math.isfinite(magnitude):
        raise ValueError(f'AB magnitude must be finite, got {m_ab!r}.')
    return AB_ZERO_POINT_JY * 10.0 ** (-0.4 * magnitude)


def photon_rate_per_m2(fnu_jy, lam_min_m, lam_max_m):
    """Integrate a flat-f_nu spectrum into a photon rate per unit area.

    For a source with constant ``f_nu`` the photon rate is
    ``f_nu / h * ln(lam_max / lam_min)``. The jansky conversion factor
    ``1e-26 / h = 1.50919e7`` is the shorthand used in the Q1 derivation
    notes.

    Parameters
    ----------
    fnu_jy : `float`
        Flux density in janskys.
    lam_min_m : `float`
        Lower band edge in meters.
    lam_max_m : `float`
        Upper band edge in meters.

    Returns
    -------
    photon_rate : `float`
        Photon rate in photons per second per square meter.
    """
    flux = float(fnu_jy)
    if not math.isfinite(flux) or flux < 0.0:
        raise ValueError(f'fnu_jy must be finite and non-negative, got {fnu_jy!r}.')
    lam_min = _require_positive(lam_min_m, 'lam_min_m')
    lam_max = _require_positive(lam_max_m, 'lam_max_m')
    if lam_max <= lam_min:
        raise ValueError(
            f'lam_max_m ({lam_max}) must exceed lam_min_m ({lam_min}).'
        )
    return flux * JANSKY_TO_SI / PLANCK_H * math.log(lam_max / lam_min)


def detected_source_rate_e_per_s(m_hri_v, area_m2, system_qe):
    """Convert a band magnitude into a detected electron rate.

    Parameters
    ----------
    m_hri_v : `float`
        AB magnitude inside the reference band.
    area_m2 : `float`
        Telescope collecting area in square meters.
    system_qe : `float`
        End-to-end system quantum efficiency, detector included.

    Returns
    -------
    rate : `float`
        Detected electrons per second.
    """
    area = _require_positive(area_m2, 'area_m2')
    efficiency = _require_positive(system_qe, 'system_qe')
    if efficiency > 1.0:
        raise ValueError(f'system_qe must not exceed one, got {system_qe!r}.')
    photon_rate = photon_rate_per_m2(
        ab_mag_to_fnu_jy(m_hri_v), BAND_LAMBDA_MIN_M, BAND_LAMBDA_MAX_M
    )
    return photon_rate * area * efficiency


def sky_rate_e_per_pix_s(sky_mag_per_arcsec2, area_m2, system_qe,
                         pixel_scale_arcsec):
    """Convert a sky surface brightness into a per-pixel electron rate.

    Parameters
    ----------
    sky_mag_per_arcsec2 : `float`
        Sky surface brightness in AB magnitudes per square arcsecond.
    area_m2 : `float`
        Telescope collecting area in square meters.
    system_qe : `float`
        End-to-end system quantum efficiency, detector included.
    pixel_scale_arcsec : `float`
        Detector pixel scale in arcseconds.

    Returns
    -------
    rate : `float`
        Detected electrons per pixel per second.
    """
    pixel_scale = _require_positive(pixel_scale_arcsec, 'pixel_scale_arcsec')
    rate_per_arcsec2 = detected_source_rate_e_per_s(
        sky_mag_per_arcsec2, area_m2, system_qe
    )
    return rate_per_arcsec2 * pixel_scale ** 2


def effective_read_noise(per_read_e, n_reads):
    """Combine per-read noise into one effective combined-image value.

    Parameters
    ----------
    per_read_e : `float`
        Read noise of a single detector read, in electrons.
    n_reads : `int`
        Number of reads contributing to the combined image.

    Returns
    -------
    read_noise : `float`
        Effective read noise in electrons.
    """
    per_read = _require_positive(per_read_e, 'per_read_e')
    if isinstance(n_reads, bool) or not isinstance(n_reads, (int, np.integer)):
        raise ValueError(f'n_reads must be a positive integer, got {n_reads!r}.')
    reads = int(n_reads)
    if reads < 1:
        raise ValueError(f'n_reads must be a positive integer, got {n_reads!r}.')
    return per_read * math.sqrt(reads)


def source_mag_hri_v(m_f814w, mode, color_ab=None, mag_vband=None):
    """Move the source anchor magnitude into the reference band.

    Parameters
    ----------
    m_f814w : `float`
        Input F814W AB magnitude of the source.
    mode : `str`
        One of ``flat_fnu``, ``declared_color``, or ``vband_photometry``.
    color_ab : `float`, optional
        Declared AB color offset, required by ``declared_color``.
    mag_vband : `float`, optional
        Measured V-band AB magnitude, required by ``vband_photometry``.

    Returns
    -------
    magnitude : `float`
        AB magnitude inside the reference band.
    """
    anchor = float(m_f814w)
    if not math.isfinite(anchor):
        raise ValueError(f'm_f814w must be finite, got {m_f814w!r}.')
    if mode == 'flat_fnu':
        if color_ab is not None or mag_vband is not None:
            raise ValueError(
                'sed-mode flat_fnu accepts neither --color-ab nor '
                '--source-mag-vband.'
            )
        return anchor
    if mode == 'declared_color':
        if color_ab is None:
            raise ValueError('sed-mode declared_color requires --color-ab.')
        if mag_vband is not None:
            raise ValueError(
                'sed-mode declared_color does not accept --source-mag-vband.'
            )
        color = float(color_ab)
        if not math.isfinite(color):
            raise ValueError(f'--color-ab must be finite, got {color_ab!r}.')
        return anchor + color
    if mode == 'vband_photometry':
        if mag_vband is None:
            raise ValueError(
                'sed-mode vband_photometry requires --source-mag-vband.'
            )
        if color_ab is not None:
            raise ValueError(
                'sed-mode vband_photometry does not accept --color-ab.'
            )
        measured = float(mag_vband)
        if not math.isfinite(measured):
            raise ValueError(
                f'--source-mag-vband must be finite, got {mag_vband!r}.'
            )
        return measured
    raise ValueError(f'Unknown sed-mode {mode!r}; choose one of {SED_MODES}.')


def sersic_n1_total_flux(intensity, effective_radius):
    """Return the intrinsic total flux of an n=1 Sersic profile.

    The AutoGalaxy area-preserving elliptical-radius convention makes the
    closed form independent of axis ratio:
    ``F = 2 pi exp(b1) / b1**2 * intensity * effective_radius**2``.

    Parameters
    ----------
    intensity : `float`
        Surface brightness at the effective radius.
    effective_radius : `float`
        Effective radius in arcseconds.

    Returns
    -------
    total_flux : `float`
        Intrinsic total flux in intensity units times square arcseconds.
    """
    amplitude = _require_positive(intensity, 'intensity')
    radius = _require_positive(effective_radius, 'effective_radius')
    return (
        2.0 * math.pi * math.exp(SERSIC_B1) / SERSIC_B1 ** 2
        * amplitude * radius ** 2
    )


def verify_qualification_total_flux():
    """Check the scene-1 closed form against the frozen qualification flux.

    Returns
    -------
    closed_form : `float`
        Closed-form total flux of the unscaled canonical source.
    """
    closed_form = sersic_n1_total_flux(
        QUALIFICATION_INTENSITY, QUALIFICATION_EFFECTIVE_RADIUS
    )
    if abs(closed_form - QUALIFICATION_TOTAL_FLUX) > CLOSED_FORM_ABS_TOLERANCE:
        raise ValueError(
            f'Scene-1 closed form gives {closed_form} but the frozen '
            f'qualification flux is {QUALIFICATION_TOTAL_FLUX}; the Sersic '
            'constant or the canonical source parameters changed.'
        )
    return closed_form


def source_profile_angular_integral(light_config):
    """Return the closed-form angular integral of one scene source.

    This is the continuous surface-brightness integral the configured
    normalizations control. It is not a detected rate: the observation
    layer reads the discrete pixel sum, which is this integral divided
    by the detector pixel area.

    Parameters
    ----------
    light_config : `dict`
        ``lensing.source_galaxy.light`` block of a scene configuration.

    Returns
    -------
    integral : `float`
        Angular integral in profile units times square arcseconds.
    """
    light_type = light_config['type']
    if light_type == 'Exponential':
        return sersic_n1_total_flux(
            light_config['intensity'], light_config['effective_radius']
        )
    if light_type == 'Clumpy':
        flux_scale = _require_positive(
            light_config['flux_scale'], 'clumpy flux_scale'
        )
        size_scale = _require_positive(
            light_config['size_scale'], 'clumpy size_scale'
        )
        components = [light_config['host'], *light_config['clumps']]
        total = 0.0
        for component in components:
            sersic_index = float(component['sersic_index'])
            if sersic_index != 1.0:
                raise ValueError(
                    'Clumpy components must carry sersic_index 1 for the n=1 '
                    f'closed form, got {sersic_index}.'
                )
            total += sersic_n1_total_flux(
                component['intensity'], component['effective_radius']
            )
        return flux_scale * size_scale ** 2 * total
    if light_type == 'Image':
        total_flux = _require_positive(
            light_config['total_flux'], 'image total_flux'
        )
        flux_scale = _require_positive(
            light_config['flux_scale'], 'image flux_scale'
        )
        size_scale = _require_positive(
            light_config['size_scale'], 'image size_scale'
        )
        return total_flux * flux_scale * size_scale ** 2
    raise ValueError(f'Unsupported source light type {light_type!r}.')


def load_scene_config(scene_config_relpath):
    """Read one scene configuration from the repository.

    Parameters
    ----------
    scene_config_relpath : `str`
        Scene configuration path relative to the repository root.

    Returns
    -------
    config : `dict`
        Parsed scene configuration.
    """
    config_path = PROJECT_ROOT / scene_config_relpath
    if not config_path.exists():
        raise ValueError(f'Scene configuration {config_path} does not exist.')
    with config_path.open('r', encoding='utf-8') as stream:
        return yaml.safe_load(stream)


def render_unlensed_source(source_config, grid_config):
    """Render one unlensed source through the production constructors.

    The grid and the light profile are built by
    ``hwoslaps.lensing.generator``, so the samples carry the exact
    production geometry and sub-pixel oversampling. Relative image-asset
    paths resolve against the repository root, which keeps the render
    independent of the working directory.

    Parameters
    ----------
    source_config : `dict`
        ``lensing.source_galaxy`` block of a scene configuration.
    grid_config : `dict`
        ``lensing.grid`` block of a scene configuration.

    Returns
    -------
    image : `numpy.ndarray`
        Unlensed source samples in the per-pixel convention the
        observation layer reads as detected electrons per second.
    """
    from hwoslaps.lensing.generator import _create_grid, _create_source_galaxy

    source_config = deepcopy(source_config)
    asset_path = source_config['light'].get('asset_path')
    if asset_path is not None and not Path(asset_path).is_absolute():
        source_config['light']['asset_path'] = str(PROJECT_ROOT / asset_path)
    galaxy = _create_source_galaxy(source_config)
    image = galaxy.image_2d_from(grid=_create_grid(grid_config))
    return np.asarray(image, dtype=float)


@lru_cache(maxsize=None)
def unlensed_discrete_sum(scene_config_relpath):
    """Sum one scene's unlensed source render on its production grid.

    Parameters
    ----------
    scene_config_relpath : `str`
        Scene configuration path relative to the repository root.

    Returns
    -------
    discrete_sum : `float`
        Discrete pixel sum of the configured source at its configured
        normalization.
    """
    lensing = load_scene_config(scene_config_relpath)['lensing']
    image = render_unlensed_source(lensing['source_galaxy'], lensing['grid'])
    return float(np.sum(image))


def scene_flux_patches(total_flux_e_per_s, pixel_scale_arcsec):
    """Solve every scene normalization for one detected pixel-sum rate.

    Each patch is a deep-merge configuration fragment in the exact shape
    the S1-lite observing-reference loader applies to a staged job
    config. The scaled leaf is the single field its source family is
    linear in, so the clumpy scene scales through ``flux_scale``, which
    the engine applies uniformly to the host and every clump: the frozen
    90/10 flux split is preserved structurally and the clump list is
    never replaced.

    The scale factor comes from rendering the unlensed source on the
    scene's own production grid rather than from a closed form, because
    the configured normalization controls a continuous angular integral
    while the observation layer reads rendered samples as per-pixel
    electron rates. The two conventions differ by the pixel area.

    Parameters
    ----------
    total_flux_e_per_s : `float`
        Physical unlensed total detected source rate.
    pixel_scale_arcsec : `float`
        Reference detector pixel scale; every scene grid must match it.

    Returns
    -------
    patches : `dict`
        Per-scene deep-merge config patches keyed by scene label.
    details : `dict`
        Per-scene provenance records keyed by scene label, carrying the
        target rate, the realized discrete rate, and the render that
        maps between them.
    """
    target_rate = _require_positive(total_flux_e_per_s, 'total_flux_e_per_s')
    pixel_scale = _require_positive(pixel_scale_arcsec, 'pixel_scale_arcsec')
    pixel_area = pixel_scale ** 2
    patches = {}
    details = {}
    for name, baseline in SCENE_LIGHT_BASELINES.items():
        relpath = baseline['scene_config']
        light_type = baseline['light_type']
        if light_type not in SCENE_NORMALIZED_FIELD:
            raise ValueError(
                f'Scene {name} declares unsupported light type {light_type!r}.'
            )
        lensing = load_scene_config(relpath)['lensing']
        grid_config = lensing['grid']
        light_config = lensing['source_galaxy']['light']
        if light_config['type'] != light_type:
            raise ValueError(
                f'Scene {name} is declared as {light_type!r} but {relpath} '
                f'carries light type {light_config["type"]!r}.'
            )
        scene_pixel_scale = _require_positive(
            grid_config['pixel_scale'], f'{relpath} lensing.grid.pixel_scale'
        )
        if abs(scene_pixel_scale - pixel_scale) > PIXEL_SCALE_ABS_TOLERANCE:
            raise ValueError(
                f'Scene {name} samples at {scene_pixel_scale} arcsec while '
                f'the reference pixel scale is {pixel_scale} arcsec; the '
                'per-pixel rates in this artifact would not apply to it.'
            )
        field = SCENE_NORMALIZED_FIELD[light_type]
        baseline_value = _require_positive(
            light_config[field], f'{relpath} source light {field}'
        )
        baseline_sum = unlensed_discrete_sum(relpath)
        if not math.isfinite(baseline_sum) or baseline_sum <= 0.0:
            raise ValueError(
                f'Scene {name} renders a discrete sum of {baseline_sum}; the '
                'configured source carries no light.'
            )
        baseline_integral = source_profile_angular_integral(light_config)
        mapping_ratio = pixel_area * baseline_sum / baseline_integral
        if abs(mapping_ratio - 1.0) > DISCRETE_MAPPING_TOLERANCE:
            raise ValueError(
                f'Scene {name} renders a discrete sum whose pixel-area '
                f'integral is {mapping_ratio} of the closed-form angular '
                f'integral {baseline_integral}; the grid, the image asset, '
                'or the profile changed.'
            )
        scale = target_rate / baseline_sum
        realized_rate = baseline_sum * scale
        leaf = {field: baseline_value * scale}
        patches[name] = {
            'lensing': {'source_galaxy': {'light': dict(leaf)}}
        }
        details[name] = {
            'scene_config': relpath,
            'light_type': light_type,
            **leaf,
            'normalized_field': field,
            'baseline_value': baseline_value,
            'scale_factor': scale,
            'grid_shape': [int(value) for value in grid_config['shape']],
            'pixel_scale_arcsec': scene_pixel_scale,
            'baseline_discrete_sum': baseline_sum,
            'baseline_profile_angular_integral': baseline_integral,
            'baseline_discrete_mapping_ratio': mapping_ratio,
            'target_rate_e_per_s': target_rate,
            'realized_rate_e_per_s': realized_rate,
            'profile_angular_integral': baseline_integral * scale,
        }
    return patches, details


def _require_config_value(mapping, key, section, config_path):
    """Return one required configuration value or fail with its path."""
    if not isinstance(mapping, dict) or key not in mapping:
        raise ValueError(f'{config_path} is missing {section}.{key}.')
    return mapping[key]


def load_pupil_geometry(config_path=MASTER_CONFIG_PATH):
    """Read the production pupil geometry from the master configuration.

    Parameters
    ----------
    config_path : `str` or `pathlib.Path`, optional
        Master configuration file carrying the ``psf`` section.

    Returns
    -------
    geometry : `dict`
        Dictionary with the ``telescope`` and ``hres_psf`` blocks accepted
        by ``create_hcipy_telescope``.
    """
    config_path = Path(config_path)
    if not config_path.exists():
        raise ValueError(f'Master configuration {config_path} does not exist.')
    with config_path.open('r', encoding='utf-8') as stream:
        config = yaml.safe_load(stream)
    psf = _require_config_value(config, 'psf', 'root', config_path)
    telescope = _require_config_value(psf, 'telescope', 'psf', config_path)
    hres_psf = _require_config_value(psf, 'hres_psf', 'psf', config_path)
    telescope_keys = (
        'gap_size',
        'segment_point_to_point',
        'pupil_diameter',
        'num_rings',
        'supersampling_factor',
    )
    return {
        'telescope': {
            key: _require_config_value(
                telescope, key, 'psf.telescope', config_path
            )
            for key in telescope_keys
        },
        'hres_psf': {
            key: _require_config_value(
                hres_psf, key, 'psf.hres_psf', config_path
            )
            for key in ('num_pix', 'wavelength')
        },
    }


def load_pixel_scale_arcsec(config_path=MASTER_CONFIG_PATH):
    """Read the detector pixel scale from the master configuration."""
    config_path = Path(config_path)
    if not config_path.exists():
        raise ValueError(f'Master configuration {config_path} does not exist.')
    with config_path.open('r', encoding='utf-8') as stream:
        config = yaml.safe_load(stream)
    lensing = _require_config_value(config, 'lensing', 'root', config_path)
    grid = _require_config_value(lensing, 'grid', 'lensing', config_path)
    pixel_scale = _require_config_value(
        grid, 'pixel_scale', 'lensing.grid', config_path
    )
    return _require_positive(pixel_scale, 'lensing.grid.pixel_scale')


def collecting_area_m2(pupil_geometry):
    """Integrate the supersampled pupil mask into a collecting area.

    The supersampled aperture field carries fractional edge coverage, so
    it is summed as evaluated rather than thresholded.

    Parameters
    ----------
    pupil_geometry : `dict`
        Geometry from :func:`load_pupil_geometry`.

    Returns
    -------
    area_m2 : `float`
        Light-gathering area in square meters.
    """
    from hwoslaps.psf.telescope_models import create_hcipy_telescope

    telescope_data = create_hcipy_telescope(pupil_geometry)
    aperture = np.asarray(telescope_data['aper'], dtype=float)
    diameter = _require_positive(
        pupil_geometry['telescope']['pupil_diameter'], 'pupil_diameter'
    )
    num_pix = int(pupil_geometry['hres_psf']['num_pix'])
    if num_pix < 1:
        raise ValueError(f'num_pix must be a positive integer, got {num_pix!r}.')
    if aperture.size != num_pix ** 2:
        raise ValueError(
            f'Pupil mask has {aperture.size} samples but the grid declares '
            f'{num_pix ** 2}; the pupil geometry is inconsistent.'
        )
    cell_area_m2 = (diameter / num_pix) ** 2
    return float(np.sum(aperture) * cell_area_m2)


def gapless_hexagon_area_m2(num_segments, segment_point_to_point):
    """Return the gapless area of a regular hexagonal segment mosaic.

    Parameters
    ----------
    num_segments : `int`
        Number of hexagonal segments.
    segment_point_to_point : `float`
        Segment point-to-point size in meters.

    Returns
    -------
    area_m2 : `float`
        Mosaic area in square meters, gaps ignored.
    """
    count = int(num_segments)
    if count < 1:
        raise ValueError(
            f'num_segments must be a positive integer, got {num_segments!r}.'
        )
    size = _require_positive(segment_point_to_point, 'segment_point_to_point')
    return count * (3.0 * math.sqrt(3.0) / 8.0) * size ** 2


def collecting_area_report(area_m2, pupil_geometry):
    """Cross-check a pupil-mask area against the gapless hexagon area.

    Parameters
    ----------
    area_m2 : `float`
        Mask-integrated collecting area in square meters.
    pupil_geometry : `dict`
        Geometry from :func:`load_pupil_geometry`.

    Returns
    -------
    report : `dict`
        Provenance record of the area, its cross-check, and the gap loss.
    """
    area = _require_positive(area_m2, 'area_m2')
    telescope = pupil_geometry['telescope']
    num_rings = int(telescope['num_rings'])
    num_segments = 1 + 3 * num_rings * (num_rings + 1)
    hexagon_area = gapless_hexagon_area_m2(
        num_segments, telescope['segment_point_to_point']
    )
    ratio = area / hexagon_area
    low, high = AREA_RATIO_BOUNDS
    if not low <= ratio <= high:
        raise ValueError(
            f'Pupil-mask area {area} m^2 is {ratio} of the gapless hexagon '
            f'area {hexagon_area} m^2, outside [{low}, {high}]; the pupil '
            f'geometry read from {MASTER_CONFIG_RELPATH} or the mask '
            'integral is wrong.'
        )
    return {
        'value_m2': area,
        'method': (
            'sum of the supersampled hcipy pupil mask times the pupil-grid '
            'cell area (pupil_diameter / num_pix) ** 2, no thresholding'
        ),
        'gapless_hexagon_area_m2': hexagon_area,
        'gapless_hexagon_formula': (
            'num_segments * 3 * sqrt(3) / 8 * segment_point_to_point ** 2'
        ),
        'mask_to_hexagon_ratio': ratio,
        'gap_loss_fraction': 1.0 - ratio,
        'gap_loss_note': (
            'gap_loss_fraction is 1 - mask_to_hexagon_ratio and can be '
            'slightly negative because fractional edge coverage on the '
            'finite pupil grid outweighs the 6 mm gap loss'
        ),
        'accepted_ratio_bounds': [low, high],
        'num_segments': num_segments,
        'geometry_source': (
            f'{MASTER_CONFIG_RELPATH} psf.telescope and psf.hres_psf'
        ),
        'geometry': {
            'gap_size_m': float(telescope['gap_size']),
            'segment_point_to_point_m': float(
                telescope['segment_point_to_point']
            ),
            'pupil_diameter_m': float(telescope['pupil_diameter']),
            'num_rings': num_rings,
            'supersampling_factor': int(telescope['supersampling_factor']),
            'num_pix': int(pupil_geometry['hres_psf']['num_pix']),
        },
        'aperture_citation': 'S1',
        'aperture_note': (
            'EAC1-like area-matched pupil: 19 hexagonal segments, 1.65 m '
            'point-to-point against the EAC1 1.7 m, 33.6 m^2 official area. '
            'The S3 USORT yield aperture (6.5 m inscribed, 7.87 m '
            'circumscribed) is a different aperture and is never mixed in.'
        ),
    }


def build_reference_document(area_report, pixel_scale_arcsec, sed_mode,
                             color_ab, mag_vband, exposure_s, system_qe,
                             sky_mag):
    """Assemble the full reference observing-configuration document.

    Parameters
    ----------
    area_report : `dict`
        Report from :func:`collecting_area_report`.
    pixel_scale_arcsec : `float`
        Detector pixel scale in arcseconds.
    sed_mode : `str`
        Source SED mode, one of :data:`SED_MODES`.
    color_ab : `float` or `None`
        Declared AB color offset for ``declared_color``.
    mag_vband : `float` or `None`
        Measured V-band AB magnitude for ``vband_photometry``.
    exposure_s : `float`
        Exposure time in seconds.
    system_qe : `float`
        End-to-end system quantum efficiency, detector included.
    sky_mag : `float`
        Sky surface brightness in AB magnitudes per square arcsecond.

    Returns
    -------
    document : `dict`
        Serializable document with top-level ``observation``,
        ``source_normalization``, and ``metadata`` keys, the exact shape
        the S1-lite observing-reference loader consumes.
    """
    closed_form_flux = verify_qualification_total_flux()
    area_m2 = _require_positive(area_report['value_m2'], 'area_report value_m2')
    exposure = _require_positive(exposure_s, 'exposure_s')
    efficiency = _require_positive(system_qe, 'system_qe')
    pixel_scale = _require_positive(pixel_scale_arcsec, 'pixel_scale_arcsec')

    magnitude = source_mag_hri_v(
        SOURCE_MAG_F814W_AB, sed_mode, color_ab=color_ab, mag_vband=mag_vband
    )
    fnu_jy = ab_mag_to_fnu_jy(magnitude)
    photon_rate = photon_rate_per_m2(
        fnu_jy, BAND_LAMBDA_MIN_M, BAND_LAMBDA_MAX_M
    )
    total_flux = detected_source_rate_e_per_s(magnitude, area_m2, efficiency)
    sky_rate = sky_rate_e_per_pix_s(
        sky_mag, area_m2, efficiency, pixel_scale
    )
    read_noise = effective_read_noise(READ_NOISE_PER_READ_E, N_READS)
    patches, scene_details = scene_flux_patches(total_flux, pixel_scale)

    observation = {
        'exposure_time': exposure,
        'throughput': 1.0,
        'detector': {
            'gain': DETECTOR_GAIN_E_PER_ADU,
            'read_noise': read_noise,
            'dark_current': DARK_CURRENT_E_PER_PIX_S,
            'sky_background': sky_rate,
        },
    }

    normalization_details = {
        'target_rate_e_per_s': total_flux,
        'units': (
            'detected electrons per second, unlensed intrinsic source total'
        ),
        'convention': (
            'Scene light normalizations are continuous surface-brightness '
            'amplitudes, never detected totals. The observation layer reads '
            'rendered samples directly as per-pixel e-/s, so every patch is '
            'solved on the scene production grid until the unlensed discrete '
            'pixel sum equals target_rate_e_per_s. The recorded patched '
            'profile angular integral is the baseline closed-form integral '
            'times the scale factor, approximately pixel_area_arcsec2 times '
            'that rate, and is not itself a detected rate.'
        ),
        'pixel_scale_arcsec': pixel_scale,
        'pixel_area_arcsec2': pixel_scale ** 2,
        'render_method': (
            'unlensed source galaxy from '
            'hwoslaps.lensing.generator._create_source_galaxy evaluated on '
            'hwoslaps.lensing.generator._create_grid: the exact production '
            'constructors, grid geometry, and sub-pixel oversampling'
        ),
        'scaling_note': (
            'Every source family is exactly linear in its normalized field, '
            'so the realized rate is the baseline discrete sum times the '
            'scale factor.'
        ),
        'discrete_mapping_tolerance': DISCRETE_MAPPING_TOLERANCE,
        'qualification_profile_angular_integral': QUALIFICATION_TOTAL_FLUX,
        'qualification_intensity': QUALIFICATION_INTENSITY,
        'qualification_effective_radius_arcsec': QUALIFICATION_EFFECTIVE_RADIUS,
        'qualification_closed_form': (
            'F_tot = 2 * pi * exp(b1) / b1 ** 2 * intensity * '
            'effective_radius ** 2'
        ),
        'qualification_closed_form_value': closed_form_flux,
        'qualification_note': (
            'The frozen qualification normalization is a profile angular '
            'integral in profile units, not a detected electron rate.'
        ),
        'sersic_b1': SERSIC_B1,
        'sersic_b1_note': (
            'AutoGalaxy sersic_constant polynomial at n = 1, the constant '
            'the installed profile code uses; the exact root of '
            '(1 + b) exp(-b) = 1/2 is 1.6783469900166605 and misses the '
            'frozen qualification flux by 8e-6 relative.'
        ),
        'scene_details': scene_details,
        'application': (
            'source_normalization holds one deep-merge config patch per '
            'scene label; the S1-lite observing-reference loader applies '
            'them to staged job configs. The scene files under '
            'configs/scenes are not modified by this script.'
        ),
    }

    metadata = {
        'reference_name': REFERENCE_NAME,
        'script': SCRIPT_NAME,
        'script_version': SCRIPT_VERSION,
        'generation_date': date.today().isoformat(),
        'observation_model_semantics': (
            'Source normalization is solved so the production render sums to '
            'the detected e-/s rate, so observation.throughput stays 1.0 and '
            'the throughput chain below is provenance only; carrying both '
            'would double-apply it.'
        ),
        'source_photometry': {
            'input_magnitude_ab': SOURCE_MAG_F814W_AB,
            'input_band': SOURCE_BAND,
            'input_magnitude_note': (
                'Mean magnification-corrected apparent magnitude of the '
                '46-source SLACS emission-line sample; the anchor is '
                'intrinsic, not observed through the lens.'
            ),
            'input_citation': 'L1',
            'sed_mode': sed_mode,
            'sed_mode_description': SED_MODE_DESCRIPTIONS[sed_mode],
            'color_ab': color_ab,
            'measured_vband_magnitude_ab': mag_vband,
            'derived_magnitude_ab': magnitude,
            'derived_band': BAND_NAME,
            'flux_density_jy': fnu_jy,
            'photon_rate_ph_per_s_m2': photon_rate,
            'detected_rate_e_per_s': total_flux,
            'normalization_plane': (
                'unlensed intrinsic source-plane total flux; this artifact '
                'applies no magnification and no lensing'
            ),
        },
        'band': {
            'name': BAND_NAME,
            'definition': (
                'study-defined 20 percent wide band centred at 500 nm'
            ),
            'centre_m': BAND_CENTRE_M,
            'fractional_width': BAND_FRACTIONAL_WIDTH,
            'lambda_min_m': BAND_LAMBDA_MIN_M,
            'lambda_max_m': BAND_LAMBDA_MAX_M,
            'citations': ['S6', 'S4'],
        },
        'collecting_area': area_report,
        'throughput_chain': {
            'name': 'luvoir_hdi_system_qe',
            'value': efficiency,
            'description': (
                'end-to-end LUVOIR HDI system QE at V, detector included'
            ),
            'citation': 'S4',
            'optimistic_chain_bracket': {
                'value': OPTIMISTIC_OPTICAL_THROUGHPUT * OPTIMISTIC_DETECTOR_QE,
                'optical_throughput': OPTIMISTIC_OPTICAL_THROUGHPUT,
                'detector_qe': OPTIMISTIC_DETECTOR_QE,
                'citation': 'S3',
                'used_in_baseline': False,
            },
            'applied_to': (
                'the derived source and sky rates in this artifact only'
            ),
        },
        'detector': {
            'gain_e_per_adu': DETECTOR_GAIN_E_PER_ADU,
            'pixel_scale_arcsec': pixel_scale,
            'pixel_solid_angle_arcsec2': pixel_scale ** 2,
            'pixel_scale_source': (
                f'{MASTER_CONFIG_RELPATH} lensing.grid.pixel_scale'
            ),
            'pixel_scale_note': (
                'Nyquist sampling at 500 nm for the circumscribed pupil '
                'diameter; HWO publishes no HRI pixel scale.'
            ),
            'sky_surface_brightness_ab_per_arcsec2': float(sky_mag),
            'sky_band': 'V',
            'sky_description': (
                'zodiacal light, uniform, HWO exposure-time-calculator '
                'convention'
            ),
            'sky_citation': 'S3',
            'sky_rate_e_per_pix_s': sky_rate,
            'qualification_sky_background_e_per_pix_s': (
                QUALIFICATION_SKY_BACKGROUND_E_PER_PIX_S
            ),
            'read_noise_per_read_e': READ_NOISE_PER_READ_E,
            'n_reads': N_READS,
            'effective_read_noise_e': read_noise,
            'effective_read_noise_formula': 'per_read_e * sqrt(n_reads)',
            'read_noise_note': (
                'The noise model applies exactly one squared read-noise '
                'term, so observation.detector.read_noise carries the '
                'effective combined-image value and never the per-read '
                'value.'
            ),
            'read_noise_citation': 'S4',
            'dark_current_e_per_pix_s': DARK_CURRENT_E_PER_PIX_S,
            'dark_current_description': 'LUVOIR HDI heritage',
            'dark_current_citation': 'S4',
        },
        'exposure': {
            'value_s': exposure,
            'description': 'HST full-orbit-class exposure (~2000 s)',
        },
        'source_normalization_details': normalization_details,
        'citations': dict(CITATIONS),
    }

    return {
        'observation': observation,
        'source_normalization': patches,
        'metadata': metadata,
    }


def _refuse_existing(path, force):
    """Fail unless an existing artifact may be replaced."""
    if path.exists() and not force:
        raise ValueError(
            f'Refusing to overwrite {path}; pass --force to replace it'
        )


def write_reference_document(path, document, force=False):
    """Write one reference observing configuration as portable safe YAML.

    Parameters
    ----------
    path : `str` or `pathlib.Path`
        Destination file.
    document : `dict`
        Document from :func:`build_reference_document`.
    force : `bool`, optional
        Replace an existing artifact.
    """
    path = Path(path)
    _refuse_existing(path, force)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8') as stream:
        yaml.safe_dump(document, stream, sort_keys=False)


def _print_summary(document):
    """Print the derived reference as a fixed-width summary table."""
    reference = document
    metadata = reference['metadata']
    normalization = metadata['source_normalization_details']
    detector = metadata['detector']
    photometry = metadata['source_photometry']
    area = metadata['collecting_area']
    observation_detector = reference['observation']['detector']
    sky_ratio = (
        detector['sky_rate_e_per_pix_s']
        / detector['qualification_sky_background_e_per_pix_s']
    )
    rows = [
        ('input magnitude (AB)', photometry['input_magnitude_ab'],
         f"{photometry['input_band']}, Newton+11 [L1]"),
        ('derived HRI-V magnitude (AB)', photometry['derived_magnitude_ab'],
         f"sed-mode {photometry['sed_mode']}"),
        ('flux density (Jy)', photometry['flux_density_jy'],
         f"{metadata['band']['name']} 450-550 nm"),
        ('photon rate (ph/s/m^2)', photometry['photon_rate_ph_per_s_m2'],
         'flat-f_nu band integral'),
        ('collecting area (m^2)', area['value_m2'],
         f"gapless {area['gapless_hexagon_area_m2']:.6f}, gap loss "
         f"{area['gap_loss_fraction']:.6f}"),
        ('system QE', metadata['throughput_chain']['value'],
         'luvoir_hdi_system_qe [S4]'),
        ('source rate (e-/s)', normalization['target_rate_e_per_s'],
         'unlensed intrinsic total'),
        ('pixel area (arcsec^2)', normalization['pixel_area_arcsec2'],
         'maps profile angular integral to discrete pixel sum'),
        ('sky rate (e-/pix/s)', observation_detector['sky_background'],
         f'qualification 1.0 e-/pix/s, ratio {sky_ratio:.6g}'),
        ('effective read noise (e-)', observation_detector['read_noise'],
         f"{detector['read_noise_per_read_e']} e- per read x "
         f"sqrt({detector['n_reads']})"),
        ('dark current (e-/pix/s)', observation_detector['dark_current'],
         'LUVOIR HDI heritage [S4]'),
        ('exposure time (s)', reference['observation']['exposure_time'],
         metadata['exposure']['description']),
    ]
    print(f"\n{metadata['reference_name']}")
    print(f"{'quantity':30s}{'value':>22s}  basis")
    for label, value, note in rows:
        print(f'{label:30s}{value:>22.10g}  {note}')

    print(
        f"\n{'scene':26s}{'field':14s}{'patched value':>20s}"
        f"{'realized e-/s':>18s}"
    )
    for name, detail in normalization['scene_details'].items():
        field = detail['normalized_field']
        print(
            f'{name:26s}{field:14s}{detail[field]:>20.10g}'
            f"{detail['realized_rate_e_per_s']:>18.10g}"
        )


def _build_parser():
    """Build the command-line argument parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sed-mode', choices=SED_MODES, default='flat_fnu')
    parser.add_argument('--color-ab', type=float)
    parser.add_argument('--source-mag-vband', type=float)
    parser.add_argument('--exposure-s', type=float, default=DEFAULT_EXPOSURE_S)
    parser.add_argument('--system-qe', type=float, default=DEFAULT_SYSTEM_QE)
    parser.add_argument('--sky-mag', type=float, default=DEFAULT_SKY_MAG)
    parser.add_argument('--output', default=str(DEFAULT_OUTPUT_PATH))
    parser.add_argument(
        '--force',
        action='store_true',
        help='Replace an existing reference artifact',
    )
    return parser


def main(argv=None):
    """Run the reference observing-configuration derivation CLI."""
    args = _build_parser().parse_args(argv)
    output_path = Path(args.output).expanduser().resolve()
    _refuse_existing(output_path, args.force)

    pupil_geometry = load_pupil_geometry()
    pixel_scale = load_pixel_scale_arcsec()
    area_report = collecting_area_report(
        collecting_area_m2(pupil_geometry), pupil_geometry
    )
    document = build_reference_document(
        area_report,
        pixel_scale,
        args.sed_mode,
        args.color_ab,
        args.source_mag_vband,
        args.exposure_s,
        args.system_qe,
        args.sky_mag,
    )
    write_reference_document(output_path, document, force=args.force)
    _print_summary(document)
    print(f'\nwrote {output_path}')
    return document


if __name__ == '__main__':
    main()
