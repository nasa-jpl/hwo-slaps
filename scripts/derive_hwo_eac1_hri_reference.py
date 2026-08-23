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
one squared read-noise term. Its per-read value comes from the HWO
Science Engineering Interface, the current public HWO working
engineering reference adopted for this study. The vendored SEI data
files are parsed on every run and every value this module attributes to
them is checked against them, so a missing file, or one whose cited
values moved, fails the derivation instead of passing an unread claim
into the artifact.

That discrete render-and-normalize contract is the ``intrinsic_rate``
normalization mode and stays the default. The ``arc_snr`` mode instead
solves each scene for one achieved integrated source-only
signal-to-noise on the LENSED, PSF-convolved, exposed image, so
morphologies can be compared at equal delivered arc signal rather than
at equal intrinsic rate. It runs the production forward model and
records both conventions per scene.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import date
from functools import lru_cache
import hashlib
import math
from pathlib import Path

import numpy as np
from scipy.optimize import brentq
import yaml


SCRIPT_NAME = 'derive_hwo_eac1_hri_reference.py'
SCRIPT_VERSION = '3'
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

SEI_PACKAGE = 'hwo_sci_eng'
SEI_VERSION = '0.1.9'
SEI_VENDOR_RELPATH = 'scratch/q1_observing_conditions/sei_v0.1.9'
SEI_VENDOR_PATH = PROJECT_ROOT / SEI_VENDOR_RELPATH
SEI_HRI_FILENAME = 'HRI.yaml'
SEI_EAC1_FILENAME = 'EAC1.yaml'
SEI_DESCRIPTION = (
    'current public HWO working engineering reference adopted for this '
    'study'
)

MAS_TO_ARCSEC = 1.0e-3
"""One milliarcsecond expressed in arcseconds."""

SEI_AGREEMENT_RELATIVE_TOLERANCE = 1.0e-9
"""Accepted departure between a declared constant and its SEI value.

The vendored files quote the same decimals this module declares, so the
only departure a faithful pair can carry is unit-conversion round-off at
the 1e-16 level. Any real edit to a vendored value is orders of
magnitude wider than this window and fails the run.
"""

DETECTOR_GAIN_E_PER_ADU = 1.0
DARK_CURRENT_E_PER_PIX_S = 0.002
READ_NOISE_PER_READ_E = 0.2
"""HRI UVIS per-read noise in electrons, from the SEI ``HRI.yaml``.

Checked against the vendored file on every run by
:func:`read_sei_instrument_parameters`.

Retired provenance, kept as history only: the derivation carried the
LUVOIR HDI Table 8-5 value of 2.5 e- per read until 2026-08-23, when the
SEI became the authoritative instrument source for this study. That
number is neither a configured value nor a bracket anywhere in the
design.
"""

SEI_PLATE_SCALE_ARCSEC = 0.00716
"""HRI UVIS plate scale in arcseconds, the SEI 7.16 mas converted.

Written as the exact decimal the artifact quotes, because the float
product of 7.16 and 1e-3 carries a round-off tail. The conversion is
checked against ``HRI.yaml`` on every run.
"""

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

NORMALIZATION_MODES = ('intrinsic_rate', 'arc_snr')
NORMALIZATION_MODE_DESCRIPTIONS = {
    'intrinsic_rate': (
        'every scene carries the same unlensed intrinsic detected source '
        'rate; the frozen default convention'
    ),
    'arc_snr': (
        'every scene reaches the same achieved integrated source-only '
        'signal-to-noise on its lensed, PSF-convolved, exposed image'
    ),
}

ARC_SNR_BRACKET_FACTOR = 10.0
"""Geometric step the arc-S/N bracket search takes in scale factor."""

ARC_SNR_MAX_BRACKET_STEPS = 12
"""Bracket steps allowed in one direction before the solve fails.

Twelve decades of scale factor around the intrinsic-rate starting point
cover any physically meaningful arc S/N request and keep the production
Poisson draw inside its integer domain, so a runaway target fails here
with the scanned range named rather than deep inside the noise model.
"""

ARC_SNR_LOG_SCALE_TOLERANCE = 1.0e-9
"""Bracket width in ``log(scale)`` the Brent solve converges to.

A width in ``log(scale)`` is a relative width in the scale factor, so
this is far inside the required 1e-6; the achieved arc S/N is checked
against :data:`ARC_SNR_RELATIVE_TOLERANCE` afterwards regardless.
"""

ARC_SNR_RELATIVE_TOLERANCE = 1.0e-6
"""Accepted ``|achieved / requested - 1|`` for a solved arc S/N."""

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
    'SEI': (
        f'HWO Science Engineering Interface, {SEI_PACKAGE} '
        f'v{SEI_VERSION}, B. Sitarski (NASA/GSFC) and J. Tumlinson '
        '(STScI), '
        'https://github.com/HWO-GOMAP-Working-Groups/Sci-Eng-Interface; '
        'consumed by the official HWO ETC (spacetelescope/hwo-tools via '
        'syotools, models/camera.py set_from_sei). PyPI wheel and the '
        f'HRI/EAC1 data files vendored at {SEI_VENDOR_RELPATH} with '
        'SHA256SUMS.'
    ),
}


def _require_positive(value, name):
    """Return one strictly positive finite float or fail loudly."""
    number = float(value)
    if not math.isfinite(number) or number <= 0.0:
        raise ValueError(f'{name} must be a positive finite number, got {value!r}.')
    return number


def _require_non_negative(value, name):
    """Return one non-negative finite float or fail loudly."""
    number = float(value)
    if not math.isfinite(number) or number < 0.0:
        raise ValueError(
            f'{name} must be a non-negative finite number, got {value!r}.'
        )
    return number


def _deep_merge(base, patch):
    """Deep-merge one patch mapping onto a base mapping.

    This mirrors the S1-lite campaign staging merge: nested mappings
    merge key by key while every other value replaces its counterpart,
    so a configuration assembled here is the one a staged job sees.

    Parameters
    ----------
    base : `dict`
        Mapping to merge onto; never mutated.
    patch : `dict`
        Mapping whose entries take precedence.

    Returns
    -------
    merged : `dict`
        Deep copy of ``base`` with ``patch`` applied.
    """
    merged = deepcopy(base)
    for key, value in patch.items():
        current = merged.get(key)
        if isinstance(current, dict) and isinstance(value, dict):
            merged[key] = _deep_merge(current, value)
        else:
            merged[key] = deepcopy(value)
    return merged


def _absolutize_asset_path(light_config):
    """Resolve one relative image-asset path against the repository root.

    Relative ``asset_path`` entries in the scene files are repository
    relative because the runner is invoked from the repository root.
    Rewriting them here keeps every render in this module independent of
    the working directory.

    Parameters
    ----------
    light_config : `dict`
        Source-light block, mutated in place when it carries a relative
        ``asset_path``.
    """
    asset_path = light_config.get('asset_path')
    if asset_path is not None and not Path(asset_path).is_absolute():
        light_config['asset_path'] = str(PROJECT_ROOT / asset_path)


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
    _absolutize_asset_path(source_config['light'])
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


def blank_pixel_variance_e2(observation):
    """Return the blank-pixel variance of one derived observation block.

    A pixel carrying no source light collects sky and dark electrons and
    is read once, so its variance is
    ``(sky_background + dark_current) * exposure_time + read_noise ** 2``
    in electrons squared. This is the ``B`` of the arc S/N convention and
    is exactly the source-free limit of the engine noise map.

    Parameters
    ----------
    observation : `dict`
        Observation block carrying ``exposure_time`` and a ``detector``
        mapping, in the shape this script emits.

    Returns
    -------
    variance : `float`
        Blank-pixel variance in electrons squared.
    """
    if not isinstance(observation, dict) or 'detector' not in observation:
        raise ValueError(
            'observation must be a mapping carrying a detector block, got '
            f'{observation!r}.'
        )
    exposure = _require_positive(
        observation['exposure_time'], 'observation.exposure_time'
    )
    detector = observation['detector']
    sky = _require_non_negative(
        detector['sky_background'], 'observation.detector.sky_background'
    )
    dark = _require_non_negative(
        detector['dark_current'], 'observation.detector.dark_current'
    )
    read_noise = _require_non_negative(
        detector['read_noise'], 'observation.detector.read_noise'
    )
    variance = (sky + dark) * exposure + read_noise ** 2
    if variance <= 0.0:
        raise ValueError(
            'The configured detector leaves a blank pixel with zero '
            'variance, so no arc signal-to-noise is defined.'
        )
    return variance


def integrated_source_snr(source_electrons, blank_variance_e2):
    """Integrate one source electron map into an achieved arc S/N.

    The engine reports a per-pixel source signal-to-noise of
    ``S_p / sqrt(S_p + B)``, the source electrons over the total pixel
    noise, so the integrated source-only value is
    ``sqrt(sum_p S_p ** 2 / (S_p + B))``. This is the convention the
    units audit used to recover 303.94 for the committed reference.

    Parameters
    ----------
    source_electrons : array-like
        Source-only electrons per pixel in the exposure, the lensed and
        PSF-convolved image times the exposure time.
    blank_variance_e2 : `float`
        Blank-pixel variance in electrons squared.

    Returns
    -------
    arc_snr : `float`
        Achieved integrated source-only signal-to-noise.
    """
    electrons = np.asarray(source_electrons, dtype=float)
    if electrons.size == 0:
        raise ValueError('source_electrons must not be empty.')
    if not np.all(np.isfinite(electrons)):
        raise ValueError('source_electrons must be finite.')
    variance = _require_positive(blank_variance_e2, 'blank_variance_e2')
    total_variance = electrons + variance
    minimum = float(np.min(total_variance))
    if minimum <= 0.0:
        raise ValueError(
            f'Source electrons drive a pixel variance down to {minimum} e^2 '
            f'against a blank-pixel variance of {variance} e^2; the source '
            'map is not a physical electron map.'
        )
    return float(np.sqrt(np.sum(electrons ** 2 / total_variance)))


@lru_cache(maxsize=None)
def _scene_psf_data(scene_config_relpath):
    """Build one scene's production PSF system once.

    The PSF depends only on the scene ``psf`` block and the lensing grid
    pixel scale, neither of which a source normalization touches, so one
    build per scene serves every solver iteration.
    """
    from hwoslaps.psf import generate_psf_system

    config = load_scene_config(scene_config_relpath)
    return generate_psf_system(config['psf'], full_config=config)


def scene_source_electrons(scene_config_relpath, observation, light_patch):
    """Run the production forward model for one scene normalization.

    The scene configuration is merged exactly as the S1-lite loader
    stages a job: the derived observation block first, then the
    source-normalization patch. The subhalo is disabled, mirroring
    ``Pipeline._create_baseline_config``, because subhalo mass and
    position are campaign sweep axes rather than scene properties and
    the delivered arc signal must not depend on a scene placeholder.

    Parameters
    ----------
    scene_config_relpath : `str`
        Scene configuration path relative to the repository root.
    observation : `dict`
        Derived observation block applied to the scene.
    light_patch : `dict`
        Source-normalization patch in the emitted deep-merge shape.

    Returns
    -------
    source_electrons : `numpy.ndarray`
        Source-only electrons per pixel of the lensed, PSF-convolved,
        exposed image.
    """
    from hwoslaps.lensing import generate_lensing_system
    from hwoslaps.observation import generate_observation

    config = load_scene_config(scene_config_relpath)
    config = _deep_merge(config, {'observation': observation})
    config = _deep_merge(config, light_patch)
    _absolutize_asset_path(config['lensing']['source_galaxy']['light'])
    if 'subhalo' in config['lensing']:
        config['lensing']['subhalo']['enabled'] = False
    lensing_data = generate_lensing_system(
        config['lensing'], full_config=config
    )
    observation_data = generate_observation(
        lensing_data=lensing_data,
        psf_data=_scene_psf_data(scene_config_relpath),
        observation_config=config['observation'],
        full_config=config,
    )
    return np.asarray(observation_data.source_electrons, dtype=float)


def scene_arc_snr_response(scene_config_relpath, field, baseline_value,
                           observation):
    """Return one scene's achieved arc S/N as a function of scale factor.

    Parameters
    ----------
    scene_config_relpath : `str`
        Scene configuration path relative to the repository root.
    field : `str`
        Config leaf the scene's source family is linear in.
    baseline_value : `float`
        Configured value of that leaf before scaling.
    observation : `dict`
        Derived observation block applied to the scene.

    Returns
    -------
    response : `callable`
        Function mapping one scale factor to the achieved arc S/N.
    """
    amplitude = _require_positive(baseline_value, f'scene light {field}')
    variance = blank_pixel_variance_e2(observation)

    def response(scale):
        """Return the achieved arc S/N at one source scale factor."""
        patch = {
            'lensing': {
                'source_galaxy': {'light': {field: amplitude * scale}}
            }
        }
        electrons = scene_source_electrons(
            scene_config_relpath, observation, patch
        )
        return integrated_source_snr(electrons, variance)

    return response


def solve_arc_snr_scale(response, initial_scale, target_arc_snr):
    """Solve one monotone arc-S/N response for its target scale factor.

    The achieved arc S/N grows strictly monotonically with the source
    scale factor, linearly where the blank-pixel variance dominates and
    as its square root where source shot noise does, so the root is
    unique. The solve runs Brent's method on ``log(scale)`` after a
    geometric bracket search, and every failure is loud: a bracket that
    does not close, a solve that does not converge, and an achieved arc
    S/N outside :data:`ARC_SNR_RELATIVE_TOLERANCE` all raise.

    Parameters
    ----------
    response : `callable`
        Function mapping one scale factor to an achieved arc S/N.
    initial_scale : `float`
        Scale factor the bracket search starts from.
    target_arc_snr : `float`
        Requested achieved arc S/N.

    Returns
    -------
    scale : `float`
        Scale factor whose achieved arc S/N is the requested value.
    record : `dict`
        Provenance record of the requested and achieved values, the
        bracket, and the solver effort.
    """
    target = _require_positive(target_arc_snr, 'target_arc_snr')
    start = _require_positive(initial_scale, 'initial_scale')
    evaluations = 0

    def objective(log_scale):
        """Return the log ratio of achieved to requested arc S/N."""
        nonlocal evaluations
        evaluations += 1
        scale = math.exp(log_scale)
        achieved = response(scale)
        value = float(achieved)
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(
                f'The arc S/N response returned {achieved!r} at scale factor '
                f'{scale}; it must be positive and finite.'
            )
        return math.log(value / target)

    log_start = math.log(start)
    start_value = objective(log_start)
    log_low = log_high = log_start
    low_value = high_value = start_value
    steps = 0
    if start_value != 0.0:
        log_step = math.log(ARC_SNR_BRACKET_FACTOR)
        while low_value * high_value > 0.0:
            if steps >= ARC_SNR_MAX_BRACKET_STEPS:
                raise ValueError(
                    f'Arc S/N target {target} is not bracketed by scale '
                    f'factors {math.exp(log_low)} to {math.exp(log_high)} '
                    f'after {steps} steps of factor {ARC_SNR_BRACKET_FACTOR}; '
                    f'the achieved values there are '
                    f'{target * math.exp(low_value)} and '
                    f'{target * math.exp(high_value)}.'
                )
            steps += 1
            if start_value < 0.0:
                log_high += log_step
                high_value = objective(log_high)
            else:
                log_low -= log_step
                low_value = objective(log_low)
        log_scale, result = brentq(
            objective,
            log_low,
            log_high,
            xtol=ARC_SNR_LOG_SCALE_TOLERANCE,
            full_output=True,
            disp=False,
        )
        if not result.converged:
            raise ValueError(
                f'The Brent solve for arc S/N target {target} did not '
                f'converge: {result.flag}.'
            )
        iterations = int(result.iterations)
    else:
        log_scale = log_start
        iterations = 0

    scale = math.exp(log_scale)
    achieved = float(response(scale))
    evaluations += 1
    residual = abs(achieved / target - 1.0)
    if residual > ARC_SNR_RELATIVE_TOLERANCE:
        raise ValueError(
            f'The solved scale factor {scale} achieves an arc S/N of '
            f'{achieved} against the requested {target}, a relative miss of '
            f'{residual} beyond the accepted {ARC_SNR_RELATIVE_TOLERANCE}.'
        )
    record = {
        'requested_arc_snr': target,
        'achieved_arc_snr': achieved,
        'relative_residual': residual,
        'initial_scale_factor': start,
        'bracket_low_scale_factor': math.exp(log_low),
        'bracket_high_scale_factor': math.exp(log_high),
        'bracket_steps': steps,
        'solver': 'scipy.optimize.brentq on log(scale factor)',
        'solver_iterations': iterations,
        'forward_model_evaluations': evaluations,
        'log_scale_tolerance': ARC_SNR_LOG_SCALE_TOLERANCE,
        'relative_tolerance': ARC_SNR_RELATIVE_TOLERANCE,
    }
    return scale, record


def validate_normalization_mode(normalization_mode, target_arc_snr):
    """Validate one normalization mode against its mode-only argument.

    Parameters
    ----------
    normalization_mode : `str`
        One of :data:`NORMALIZATION_MODES`.
    target_arc_snr : `float` or `None`
        Requested achieved arc S/N, accepted by ``arc_snr`` only.

    Returns
    -------
    normalization_mode : `str`
        The validated mode.
    """
    if normalization_mode not in NORMALIZATION_MODES:
        raise ValueError(
            f'Unknown normalization-mode {normalization_mode!r}; choose one '
            f'of {NORMALIZATION_MODES}.'
        )
    if normalization_mode == 'intrinsic_rate':
        if target_arc_snr is not None:
            raise ValueError(
                'normalization-mode intrinsic_rate does not accept '
                '--target-arc-snr.'
            )
        return normalization_mode
    if target_arc_snr is None:
        raise ValueError(
            'normalization-mode arc_snr requires --target-arc-snr.'
        )
    _require_positive(target_arc_snr, '--target-arc-snr')
    return normalization_mode


def scene_flux_patches(total_flux_e_per_s, pixel_scale_arcsec,
                       normalization_mode='intrinsic_rate',
                       target_arc_snr=None, observation=None):
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

    Under ``normalization_mode='arc_snr'`` that unlensed scale factor is
    only the starting point: each scene is then solved through the
    production forward model until its lensed, PSF-convolved, exposed
    image achieves ``target_arc_snr``, and the resulting intrinsic rate
    is recorded per scene alongside the requested and achieved arc S/N.
    The scenes no longer share one intrinsic rate in that mode, which is
    the point of the comparison.

    Parameters
    ----------
    total_flux_e_per_s : `float`
        Physical unlensed total detected source rate. It is the solved
        rate under ``intrinsic_rate`` and the starting scale under
        ``arc_snr``.
    pixel_scale_arcsec : `float`
        Reference detector pixel scale; every scene grid must match it.
    normalization_mode : `str`, optional
        One of :data:`NORMALIZATION_MODES`, defaulting to the frozen
        ``intrinsic_rate`` convention.
    target_arc_snr : `float`, optional
        Requested achieved arc S/N; required by ``arc_snr`` and rejected
        by ``intrinsic_rate``.
    observation : `dict`, optional
        Derived observation block the forward model observes through;
        required by ``arc_snr`` and rejected by ``intrinsic_rate``.

    Returns
    -------
    patches : `dict`
        Per-scene deep-merge config patches keyed by scene label.
    details : `dict`
        Per-scene provenance records keyed by scene label, carrying the
        target rate, the realized discrete rate, and the render that
        maps between them, plus the arc S/N solution under ``arc_snr``.
    """
    validate_normalization_mode(normalization_mode, target_arc_snr)
    if normalization_mode == 'intrinsic_rate':
        if observation is not None:
            raise ValueError(
                'normalization-mode intrinsic_rate does not accept an '
                'observation block; it never observes the scenes.'
            )
    elif observation is None:
        raise ValueError(
            'normalization-mode arc_snr requires an observation block to '
            'expose and noise the lensed scenes.'
        )
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
        arc_snr_solution = None
        if normalization_mode == 'arc_snr':
            scale, arc_snr_solution = solve_arc_snr_scale(
                scene_arc_snr_response(
                    relpath, field, baseline_value, observation
                ),
                scale,
                target_arc_snr,
            )
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
        if arc_snr_solution is not None:
            details[name]['normalization_mode'] = normalization_mode
            details[name]['arc_snr_solution'] = arc_snr_solution
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


def _sei_leaf(document, keys, expected_unit, path):
    """Return one SEI ``[value, unit]`` leaf as a float, unit checked.

    Parameters
    ----------
    document : `dict`
        Parsed SEI data file.
    keys : `tuple` of `str`
        Key path to the leaf.
    expected_unit : `str`
        Unit the derivation reads the leaf in.
    path : `str` or `pathlib.Path`
        File the document came from, for the failure messages.

    Returns
    -------
    value : `float`
        Leaf value in ``expected_unit``.
    """
    node = document
    for depth, key in enumerate(keys, start=1):
        if not isinstance(node, dict) or key not in node:
            raise ValueError(f'{path} is missing {".".join(keys[:depth])}.')
        node = node[key]
    trail = '.'.join(keys)
    if not isinstance(node, list) or len(node) != 2:
        raise ValueError(
            f'{path} {trail} is not a [value, unit] pair, got {node!r}.'
        )
    value, unit = node
    if unit != expected_unit:
        raise ValueError(
            f'{path} {trail} is quoted in {unit!r}, expected {expected_unit!r}.'
        )
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f'{path} {trail} is not a number, got {value!r}.')
    return float(value)


def read_sei_data_file(filename, vendor_path=SEI_VENDOR_PATH):
    """Parse one vendored SEI data file and digest the bytes parsed.

    Parameters
    ----------
    filename : `str`
        File name inside the vendored SEI directory.
    vendor_path : `str` or `pathlib.Path`, optional
        Directory holding the vendored SEI data files.

    Returns
    -------
    document : `dict`
        Parsed YAML mapping.
    sha256 : `str`
        Hex digest of the exact bytes that were parsed.
    """
    path = Path(vendor_path) / filename
    if not path.is_file():
        raise ValueError(
            f'Vendored SEI file {path} does not exist; the derivation reads '
            f'{SEI_VENDOR_RELPATH} for its instrument parameters and cannot '
            'claim SEI provenance without it.'
        )
    payload = path.read_bytes()
    document = yaml.safe_load(payload.decode('utf-8'))
    if not isinstance(document, dict):
        raise ValueError(f'{path} does not parse as a YAML mapping.')
    return document, hashlib.sha256(payload).hexdigest()


def read_sei_instrument_parameters(vendor_path=SEI_VENDOR_PATH):
    """Read every SEI value this derivation attributes to the SEI.

    The vendored ``HRI.yaml`` and ``EAC1.yaml`` are parsed on each run
    rather than restated here, so a removed file, or one whose cited
    values moved, fails the derivation instead of passing an unread
    claim into the artifact. The returned digests are the ones the
    provenance record carries, and they change with any edit at all.

    Parameters
    ----------
    vendor_path : `str` or `pathlib.Path`, optional
        Directory holding the vendored SEI data files.

    Returns
    -------
    parameters : `dict`
        Parsed SEI values keyed by the quantity they supply, plus a
        ``sha256`` mapping from file name to hex digest.
    """
    hri, hri_digest = read_sei_data_file(SEI_HRI_FILENAME, vendor_path)
    eac1, eac1_digest = read_sei_data_file(SEI_EAC1_FILENAME, vendor_path)
    hri_path = Path(vendor_path) / SEI_HRI_FILENAME
    eac1_path = Path(vendor_path) / SEI_EAC1_FILENAME
    segmentation = ('PM', 'segmentation_parameters')
    return {
        'read_noise_per_read_e': _sei_leaf(
            hri, ('UVIS', 'detector', 'detector_RN'), 'electrons', hri_path
        ),
        'dark_current_e_per_pix_s': _sei_leaf(
            hri,
            ('UVIS', 'detector', 'detector_DC'),
            'electrons/pixel/second',
            hri_path,
        ),
        'plate_scale_arcsec': MAS_TO_ARCSEC * _sei_leaf(
            hri, ('UVIS', 'plate_scale'), 'mas', hri_path
        ),
        'segment_point_to_point_m': _sei_leaf(
            eac1, segmentation + ('segment_size',), 'meters', eac1_path
        ),
        'gap_size_m': _sei_leaf(
            eac1, segmentation + ('optical_gap',), 'meters', eac1_path
        ),
        'num_rings': _sei_leaf(
            eac1, segmentation + ('number_rings',), 'unitless', eac1_path
        ),
        'num_segments': _sei_leaf(
            eac1, segmentation + ('number_segments',), 'unitless', eac1_path
        ),
        'pupil_diameter_m': _sei_leaf(
            eac1, ('PM', 'circumscribing_diameter'), 'meters', eac1_path
        ),
        'sha256': {
            SEI_HRI_FILENAME: hri_digest,
            SEI_EAC1_FILENAME: eac1_digest,
        },
    }


def _require_sei_agreement(name, sei_value, declared_value, path):
    """Fail unless a declared value matches the SEI value it cites."""
    declared = float(declared_value)
    if not math.isclose(
        sei_value,
        declared,
        rel_tol=SEI_AGREEMENT_RELATIVE_TOLERANCE,
        abs_tol=0.0,
    ):
        raise ValueError(
            f'{name} is declared as {declared_value} but {path} carries '
            f'{sei_value}; this derivation cites the SEI for that value, so '
            'the two must agree.'
        )


def _sei_provenance(pixel_scale_arcsec, area_report,
                    vendor_path=SEI_VENDOR_PATH):
    """Return the SEI instrument-reference provenance block.

    The SEI is the authoritative instrument source for this study as of
    2026-08-23. Every value attributed to it below is parsed out of the
    vendored files here and checked against the value this derivation
    declares, and the digests of the files parsed are recorded. Only the
    detector read noise is adopted from it; the remaining entries record
    where the independently derived engine values agree with it, so a
    future SEI release can be diffed against this artifact.

    Parameters
    ----------
    pixel_scale_arcsec : `float`
        Detector pixel scale in arcseconds, for the plate-scale
        corroboration.
    area_report : `dict`
        Report from :func:`collecting_area_report`, carrying the pupil
        geometry for the EAC1 prescription corroboration.
    vendor_path : `str` or `pathlib.Path`, optional
        Directory holding the vendored SEI data files.

    Returns
    -------
    provenance : `dict`
        Serializable instrument-reference record.
    """
    pixel_scale = _require_positive(pixel_scale_arcsec, 'pixel_scale_arcsec')
    sei = read_sei_instrument_parameters(vendor_path)
    hri_path = Path(vendor_path) / SEI_HRI_FILENAME
    eac1_path = Path(vendor_path) / SEI_EAC1_FILENAME
    if 'geometry' not in area_report or 'num_segments' not in area_report:
        raise ValueError(
            'area_report must carry the geometry and num_segments entries '
            'from collecting_area_report for the EAC1 corroboration.'
        )
    geometry = area_report['geometry']
    telescope_geometry = {
        key: geometry[key]
        for key in (
            'segment_point_to_point_m', 'gap_size_m', 'pupil_diameter_m',
            'num_rings',
        )
    }
    telescope_geometry['num_segments'] = area_report['num_segments']
    declared = {
        'read_noise_per_read_e': (READ_NOISE_PER_READ_E, hri_path),
        'dark_current_e_per_pix_s': (DARK_CURRENT_E_PER_PIX_S, hri_path),
        'plate_scale_arcsec': (SEI_PLATE_SCALE_ARCSEC, hri_path),
        **{
            key: (value, eac1_path)
            for key, value in telescope_geometry.items()
        },
    }
    for name, (value, path) in declared.items():
        _require_sei_agreement(name, sei[name], value, path)
    _require_sei_agreement(
        'derived pixel_scale_arcsec', sei['plate_scale_arcsec'], pixel_scale,
        hri_path,
    )
    return {
        'name': 'HWO Science Engineering Interface',
        'package': SEI_PACKAGE,
        'version': SEI_VERSION,
        'status': SEI_DESCRIPTION,
        'vendored_path': SEI_VENDOR_RELPATH,
        'data_files_read': list(sei['sha256']),
        'data_file_sha256': dict(sei['sha256']),
        'data_file_note': (
            'Every value attributed to the SEI below is parsed out of these '
            'files on each run and checked against the value this derivation '
            'declares; a missing file, or one whose cited values moved, '
            'fails the run. The digests are of the exact bytes parsed, so '
            'they move with any edit at all.'
        ),
        'citation': 'SEI',
        'adopted_parameters': {
            'read_noise_per_read_e': {
                'value': READ_NOISE_PER_READ_E,
                'sei_source': 'HRI.yaml, UVIS channel detector read noise',
                'replaces': (
                    'LUVOIR HDI Table 8-5 read noise of 2.5 e- per read [S4], '
                    'retired 2026-08-23'
                ),
            },
        },
        'corroborated_parameters': {
            'pixel_scale_arcsec': {
                'value': pixel_scale,
                'sei_value': SEI_PLATE_SCALE_ARCSEC,
                'sei_source': 'HRI.yaml, UVIS plate scale 7.16 mas',
                'note': (
                    'derived independently as Nyquist sampling at 500 nm for '
                    'the circumscribed pupil diameter and equal to the SEI '
                    'value'
                ),
            },
            'dark_current_e_per_pix_s': {
                'value': DARK_CURRENT_E_PER_PIX_S,
                'sei_value': sei['dark_current_e_per_pix_s'],
                'sei_source': 'HRI.yaml, UVIS channel dark current',
            },
            'telescope_geometry': {
                'value': telescope_geometry,
                'sei_source': (
                    'EAC1.yaml, PM segmentation_parameters and '
                    'circumscribing_diameter'
                ),
                'note': (
                    'the production pupil geometry read from '
                    f'{MASTER_CONFIG_RELPATH} psf.telescope equals the SEI '
                    'EAC1 prescription value for value'
                ),
            },
        },
        'caveats': (
            'v0.1.9 carries pre-formulation placeholder values and the EAC '
            'architectures are exploratory. The UVIS filter transmission '
            'curves referenced by HRI.yaml are not shipped in the wheel, so '
            'the throughput chain in this artifact is not taken from the SEI.'
        ),
    }


def _arc_snr_normalization_provenance(target_arc_snr, observation):
    """Return the arc-S/N provenance block of the normalization record.

    Parameters
    ----------
    target_arc_snr : `float`
        Requested achieved arc S/N.
    observation : `dict`
        Derived observation block the forward model observes through.

    Returns
    -------
    provenance : `dict`
        Keys added to the normalization details under ``arc_snr`` only,
        so the ``intrinsic_rate`` artifact is unchanged.
    """
    return {
        'normalization_mode': 'arc_snr',
        'normalization_mode_description': (
            NORMALIZATION_MODE_DESCRIPTIONS['arc_snr']
        ),
        'requested_arc_snr': _require_positive(
            target_arc_snr, 'target_arc_snr'
        ),
        'arc_snr_formula': (
            'SNR_arc = sqrt(sum_p S_p ** 2 / (S_p + B)) with S_p the '
            'source-only electrons per pixel of the lensed, PSF-convolved, '
            'exposed image and B the blank-pixel variance in e^2'
        ),
        'arc_snr_blank_pixel_variance_e2': blank_pixel_variance_e2(
            observation
        ),
        'arc_snr_blank_pixel_variance_formula': (
            '(sky_background + dark_current) * exposure_time + '
            'read_noise ** 2'
        ),
        'arc_snr_forward_model': (
            'hwoslaps.lensing.generate_lensing_system followed by '
            'hwoslaps.observation.generate_observation, on the scene '
            'production grid with the scene psf block and the observation '
            'block of this artifact; the subhalo is disabled because its '
            'mass and position are campaign sweep axes rather than scene '
            'properties'
        ),
        'arc_snr_solver': (
            'scipy.optimize.brentq on log(scale factor) after a geometric '
            'bracket search, started from the intrinsic-rate scale factor; '
            'the achieved arc S/N is verified against the request and every '
            'bracket, convergence, or residual failure raises'
        ),
        'target_rate_note': (
            'Under arc_snr the scenes do not share one intrinsic rate. '
            'target_rate_e_per_s stays the photometric anchor that sets the '
            'solver starting point, and each scene records the intrinsic '
            'rate it actually realizes in realized_rate_e_per_s.'
        ),
    }


def build_reference_document(area_report, pixel_scale_arcsec, sed_mode,
                             color_ab, mag_vband, exposure_s, system_qe,
                             sky_mag, normalization_mode='intrinsic_rate',
                             target_arc_snr=None):
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
    normalization_mode : `str`, optional
        One of :data:`NORMALIZATION_MODES`, defaulting to the frozen
        ``intrinsic_rate`` convention. Only ``arc_snr`` adds keys to the
        emitted document, so the default artifact is unchanged.
    target_arc_snr : `float`, optional
        Requested achieved arc S/N; required by ``arc_snr`` and rejected
        by ``intrinsic_rate``.

    Returns
    -------
    document : `dict`
        Serializable document with top-level ``observation``,
        ``source_normalization``, and ``metadata`` keys, the exact shape
        the S1-lite observing-reference loader consumes.
    """
    validate_normalization_mode(normalization_mode, target_arc_snr)
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

    patches, scene_details = scene_flux_patches(
        total_flux,
        pixel_scale,
        normalization_mode=normalization_mode,
        target_arc_snr=target_arc_snr,
        observation=observation if normalization_mode == 'arc_snr' else None,
    )

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
    if normalization_mode == 'arc_snr':
        normalization_details.update(
            _arc_snr_normalization_provenance(target_arc_snr, observation)
        )

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
        'instrument_reference': _sei_provenance(pixel_scale, area_report),
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
            'read_noise_citation': 'SEI',
            'read_noise_description': (
                f'HRI UVIS read noise from the SEI HRI.yaml, the '
                f'{SEI_DESCRIPTION}'
            ),
            'read_noise_retired_provenance': (
                'The derivation carried the LUVOIR HDI Table 8-5 value of '
                '2.5 e- per read [S4] until 2026-08-23, when the SEI became '
                'the authoritative instrument source. That value is recorded '
                'here as provenance history only: it is not a configured '
                'value and it is not a bracket.'
            ),
            'dark_current_e_per_pix_s': DARK_CURRENT_E_PER_PIX_S,
            'dark_current_description': (
                'LUVOIR HDI heritage, equal to the SEI HRI UVIS value'
            ),
            'dark_current_citation': 'S4',
            'dark_current_corroboration_citation': 'SEI',
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

    _print_arc_snr_summary(normalization)


def _print_arc_snr_summary(normalization):
    """Print the arc-S/N solutions, when the run solved for them."""
    solutions = {
        name: detail['arc_snr_solution']
        for name, detail in normalization['scene_details'].items()
        if 'arc_snr_solution' in detail
    }
    if not solutions:
        return
    print(
        f"\n{'scene':26s}{'requested S/N':>16s}{'achieved S/N':>16s}"
        f"{'intrinsic e-/s':>18s}{'evals':>8s}"
    )
    for name, solution in solutions.items():
        detail = normalization['scene_details'][name]
        print(
            f'{name:26s}{solution["requested_arc_snr"]:>16.10g}'
            f'{solution["achieved_arc_snr"]:>16.10g}'
            f'{detail["realized_rate_e_per_s"]:>18.10g}'
            f'{solution["forward_model_evaluations"]:>8d}'
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
    parser.add_argument(
        '--normalization-mode',
        choices=NORMALIZATION_MODES,
        default='intrinsic_rate',
        help=(
            'Scene normalization convention; intrinsic_rate reproduces the '
            'committed reference exactly'
        ),
    )
    parser.add_argument(
        '--target-arc-snr',
        type=float,
        help='Requested achieved arc S/N, required by --normalization-mode arc_snr',
    )
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
    validate_normalization_mode(args.normalization_mode, args.target_arc_snr)

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
        normalization_mode=args.normalization_mode,
        target_arc_snr=args.target_arc_snr,
    )
    write_reference_document(output_path, document, force=args.force)
    _print_summary(document)
    print(f'\nwrote {output_path}')
    return document


if __name__ == '__main__':
    main()
