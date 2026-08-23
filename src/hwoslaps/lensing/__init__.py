"""Lensing system generation module for HWO-SLAPS.

This module provides functionality to generate realistic galaxy-galaxy strong
lensing systems with precisely known subhalo populations.
"""

from __future__ import annotations

from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from .critical_curve import (
        ApertureDefinition,
        CriticalCurveError,
        CriticalCurveGrid,
        ThetaEExtraction,
        extract_theta_e,
        extract_theta_e_from_lens_config,
    )
    from .image_source import ImageSource, SourceImageAsset, load_source_image_asset
    from .mass_models import (
        concentration_mass_relation,
        einstein_radius_point_mass,
        einstein_radius_sis_m200,
        nfw_scale_parameters,
    )

_CRITICAL_CURVE_NAMES = {
    "ApertureDefinition",
    "CriticalCurveError",
    "CriticalCurveGrid",
    "ThetaEExtraction",
    "extract_theta_e",
    "extract_theta_e_from_lens_config",
}

__all__ = [
    "generate_lensing_system",
    "ApertureDefinition",
    "CriticalCurveError",
    "CriticalCurveGrid",
    "ThetaEExtraction",
    "extract_theta_e",
    "extract_theta_e_from_lens_config",
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
    if name in _CRITICAL_CURVE_NAMES:
        from . import critical_curve

        return getattr(critical_curve, name)
    if name in {
        "ImageSource",
        "SourceImageAsset",
        "load_source_image_asset",
    }:
        from . import image_source

        return getattr(image_source, name)
    if name in {
        "einstein_radius_point_mass",
        "einstein_radius_sis_m200",
        "nfw_scale_parameters",
        "concentration_mass_relation",
    }:
        from . import mass_models

        return getattr(mass_models, name)
    if name == "generate_lensing_system":
        from .generator import generate_lensing_system

        return generate_lensing_system
    if name == "LensingData":
        from .utils import LensingData

        return LensingData
    raise AttributeError(name)


def __dir__() -> list[str]:
    """Return package public names for IDE and star-import compatibility."""
    return sorted(__all__)
