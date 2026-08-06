"""
PSF generation module for HWO-SLAPS.

This module provides functionality to generate realistic PSFs with various
optical aberrations using HCIPy for segmented telescope modeling.
"""

from .generator import generate_psf_system
from .utils import PSFData
from ..plotting.psf_plots import plot_psf_complete_analysis
from .psf_metrics import (
    measure_fwhm,
    calculate_strehl_ratio,
    analyze_psf_quality
)
from .telescope_models import create_hcipy_telescope
from .aberration_models import (
    apply_segment_pistons,
    apply_segment_tiptilts,
    apply_segment_zernikes,
    apply_global_zernikes
)
from .opd_basis import ApertureBasisTransform
from .families import (
    ModeWeightPrior,
    make_power_law_prior,
    load_mode_weight_prior,
    noll_to_radial_order,
    draw_segment_piston_family,
    draw_segment_tiptilt_family,
    draw_segment_hexike_family,
    draw_global_zernike_family,
    draw_weighted_global_zernike_family,
    draw_weighted_segment_hexike_family,
    draw_weighted_combined_family,
    realize_weighted_draw,
    measure_aperture_rms_nm,
    renormalize_to_aperture_rms
)

__all__ = [
    'generate_psf_system',
    'PSFData',
    'measure_fwhm',
    'calculate_strehl_ratio',
    'analyze_psf_quality',
    'create_hcipy_telescope',
    'apply_segment_pistons',
    'apply_segment_tiptilts',
    'apply_segment_zernikes',
    'apply_global_zernikes',
    'ApertureBasisTransform',
    'ModeWeightPrior',
    'make_power_law_prior',
    'load_mode_weight_prior',
    'noll_to_radial_order',
    'draw_segment_piston_family',
    'draw_segment_tiptilt_family',
    'draw_segment_hexike_family',
    'draw_global_zernike_family',
    'draw_weighted_global_zernike_family',
    'draw_weighted_segment_hexike_family',
    'draw_weighted_combined_family',
    'realize_weighted_draw',
    'measure_aperture_rms_nm',
    'renormalize_to_aperture_rms',
    'plot_psf_complete_analysis'
]
