"""
Subhalo detection and lens modeling module for HWO-SLAPS.

This module provides chi-square statistical detection of dark matter
subhalos using validated methodology from prototype studies.
"""

from .generator import perform_subhalo_detection
from .utils import DetectionData, print_detection_summary
from .chi_square_detector import ChiSquareSubhaloDetector, DetectionResult

__all__ = [
    'perform_subhalo_detection',
    'DetectionData', 
    'ChiSquareSubhaloDetector',
    'DetectionResult',
    'print_detection_summary'
]
