"""
Lensing system generation module for HWO-SLAPS.

This module provides functionality to generate realistic galaxy-galaxy strong
lensing systems with precisely known subhalo populations.
"""

from .generator import generate_lensing_system
from .utils import LensingData
from .mass_models import (
    einstein_radius_point_mass,
    einstein_radius_sis_m200,
    nfw_scale_parameters,
    concentration_mass_relation
)

__all__ = [
    'generate_lensing_system',
    'LensingData',
    'einstein_radius_point_mass',
    'einstein_radius_sis_m200', 
    'nfw_scale_parameters',
    'concentration_mass_relation'
]