"""
Subhalo detection and lens modeling module for HWO-SLAPS.

This module provides chi-square statistical detection of dark matter
subhalos using validated methodology from prototype studies.
"""

from .generator import perform_subhalo_detection
from .utils import DetectionData, print_detection_summary
from .chi_square_detector import ChiSquareSubhaloDetector, DetectionResult
from .generator_fisher import perform_fisher_detection
from .utils_fisher import FisherDetectionData, FisherLocalData, FisherMapData, print_fisher_summary

__all__ = [
    'perform_subhalo_detection',
    'perform_fisher_detection',
    'DetectionData', 
    'FisherDetectionData',
    'FisherLocalData',
    'FisherMapData',
    'ChiSquareSubhaloDetector',
    'DetectionResult',
    'print_detection_summary',
    'print_fisher_summary',
]
