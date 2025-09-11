"""
Plotting module for HWO-SLAPS pipeline.

This module provides visualization functions for lensing, PSF, observation,
and subhalo detection analysis.
"""

from .lensing_plots import plot_lensing_comparison, plot_lensing_baseline_scene
from .psf_plots import (
    plot_psf_comparison,
    plot_psf_zoom,
    plot_psf_system_overview,
    plot_psf_complete_analysis
)
from .observation_plots import plot_observation_comparison
from .detection_plots import plot_detection_comparison, plot_chernoff_detection_comparison
from .registry import generate_all_plots, get_plot_registry

__all__ = [
    'plot_lensing_comparison',
    'plot_lensing_baseline_scene',
    'plot_psf_comparison',
    'plot_psf_zoom',
    'plot_psf_system_overview',
    'plot_psf_complete_analysis',
    'plot_observation_comparison',
    'plot_detection_comparison',
    'plot_chernoff_detection_comparison',
    'generate_all_plots',
    'get_plot_registry'
]