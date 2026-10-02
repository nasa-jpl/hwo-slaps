"""Observation simulation module for HWO-SLAPS.

This module provides functionality for simulating realistic observations
including PSF convolution, detector noise, and proper noise map generation.
"""

from importlib import import_module

__all__ = [
    'generate_observation', 'ObservationData', 'print_observation_summary',
    'DetectorMoments', 'detector_moments', 'apply_detector_noise', 'create_noise_map',
    'detector_mean_adu', 'ObservationPrediction', 'predict_observation',
    'convolve_source_rate', 'mean_adu_images_from_lensing_arrays',
]

_EXPORT_MODULES = {
    'generate_observation': '.generator',
    'ObservationData': '.utils',
    'print_observation_summary': '.utils',
    'DetectorMoments': '.noise_models',
    'detector_moments': '.noise_models',
    'apply_detector_noise': '.noise_models',
    'create_noise_map': '.noise_models',
    'detector_mean_adu': '.noise_models',
    'ObservationPrediction': '.forward',
    'predict_observation': '.forward',
    'convolve_source_rate': '.forward',
    'mean_adu_images_from_lensing_arrays': '.forward',
}


def __getattr__(name):
    """Load detector math without importing optional imaging dependencies."""
    module_name = _EXPORT_MODULES.get(name)
    if module_name is None:
        raise AttributeError(name)
    return getattr(import_module(module_name, __name__), name)


def __dir__():
    return sorted(__all__)
