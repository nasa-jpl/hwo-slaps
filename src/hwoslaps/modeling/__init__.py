"""
Subhalo detection and lens modeling module for HWO-SLAPS.

This module provides chi-square statistical detection of dark matter
subhalos using validated methodology from prototype studies.
"""

from .generator import perform_subhalo_detection
from .utils import DetectionData, validate_detection_results, print_detection_summary, validate_and_print_summary
from .chi_square_detector import ChiSquareSubhaloDetector, DetectionResult

__all__ = [
    'perform_subhalo_detection',
    'DetectionData', 
    'ChiSquareSubhaloDetector',
    'DetectionResult',
    'validate_detection_results',
    'print_detection_summary', 
    'validate_and_print_summary'
]