"""Observation simulation module for HWO-SLAPS.

This module provides functionality for simulating realistic observations
including PSF convolution, detector noise, and proper noise map generation.
"""

from .generator import generate_observation
from .utils import ObservationData, print_observation_summary

__all__ = ['generate_observation', 'ObservationData', 'print_observation_summary']