"""
Lensing system generation module for HWO-SLAPS.

This module provides functionality to generate realistic galaxy-galaxy strong
lensing systems with precisely known subhalo populations.
"""

from typing import Any

from .image_source import ImageSource, SourceImageAsset, load_source_image_asset
from .mass_models import (
    einstein_radius_point_mass,
    einstein_radius_sis_m200,
    nfw_scale_parameters,
    concentration_mass_relation,
)

__all__ = [
    "generate_lensing_system",
    "ImageSource",
    "SourceImageAsset",
    "load_source_image_asset",
    "LensingData",
    "einstein_radius_point_mass",
    "einstein_radius_sis_m200",
    "nfw_scale_parameters",
    "concentration_mass_relation",
]


def __getattr__(name: str) -> Any:
    """Resolve PyAutoLens-backed generation helpers only when requested."""
    if name == "generate_lensing_system":
        from .generator import generate_lensing_system

        return generate_lensing_system
    if name == "LensingData":
        from .utils import LensingData

        return LensingData
    raise AttributeError(name)
