"""
PSF generation module for HWO-SLAPS.

This module provides functionality to generate realistic PSFs with various
optical aberrations using HCIPy for segmented telescope modeling.
"""

from .generator import generate_psf_system, generate_aberrated_psf, generate_calibrated_segment_aberrations
from .utils import PSFData
from ..plotting.psf_plots import plot_psf_complete_analysis
from .psf_metrics import (
    measure_fwhm,
    calculate_strehl_ratio,
    analyze_psf_quality,
    print_psf_quality_summary
)
from .telescope_models import create_hcipy_telescope
from .aberration_models import (
    apply_segment_pistons,
    apply_segment_tiptilts,
    apply_segment_zernikes,
    apply_segment_zernikes_manual,
    apply_segment_zernikes_api,
    apply_global_zernikes,
    make_hexike_basis
)

__all__ = [
    'generate_psf_system',
    'generate_aberrated_psf',
    'generate_calibrated_segment_aberrations',
    'PSFData',
    'measure_fwhm',
    'calculate_strehl_ratio',
    'analyze_psf_quality',
    'print_psf_quality_summary',
    'create_hcipy_telescope',
    'apply_segment_pistons',
    'apply_segment_tiptilts',
    'apply_segment_zernikes',
    'apply_segment_zernikes_manual',
    'apply_segment_zernikes_api',
    'apply_global_zernikes',
    'make_hexike_basis',
    'plot_psf_complete_analysis'
]