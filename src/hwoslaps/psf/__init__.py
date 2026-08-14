"""PSF generation and analysis helpers for HWO-SLAPS."""

from __future__ import annotations

from importlib import import_module
from typing import Any


__all__ = [
    "generate_psf_system",
    "PSFData",
    "measure_fwhm",
    "calculate_strehl_ratio",
    "analyze_psf_quality",
    "create_hcipy_telescope",
    "apply_segment_pistons",
    "apply_segment_tiptilts",
    "apply_segment_zernikes",
    "apply_global_zernikes",
    "ApertureBasisTransform",
    "ModeWeightPrior",
    "make_power_law_prior",
    "load_mode_weight_prior",
    "noll_to_radial_order",
    "draw_segment_piston_family",
    "draw_segment_tiptilt_family",
    "draw_segment_hexike_family",
    "draw_global_zernike_family",
    "draw_weighted_global_zernike_family",
    "draw_weighted_segment_hexike_family",
    "draw_weighted_combined_family",
    "realize_weighted_draw",
    "measure_aperture_rms_nm",
    "renormalize_to_aperture_rms",
    "plot_psf_complete_analysis",
]


_EXPORT_MODULES = {
    "generate_psf_system": ".generator",
    "PSFData": ".utils",
    "measure_fwhm": ".psf_metrics",
    "calculate_strehl_ratio": ".psf_metrics",
    "analyze_psf_quality": ".psf_metrics",
    "create_hcipy_telescope": ".telescope_models",
    "apply_segment_pistons": ".aberration_models",
    "apply_segment_tiptilts": ".aberration_models",
    "apply_segment_zernikes": ".aberration_models",
    "apply_global_zernikes": ".aberration_models",
    "ApertureBasisTransform": ".opd_basis",
    "ModeWeightPrior": ".families",
    "make_power_law_prior": ".families",
    "load_mode_weight_prior": ".families",
    "noll_to_radial_order": ".families",
    "draw_segment_piston_family": ".families",
    "draw_segment_tiptilt_family": ".families",
    "draw_segment_hexike_family": ".families",
    "draw_global_zernike_family": ".families",
    "draw_weighted_global_zernike_family": ".families",
    "draw_weighted_segment_hexike_family": ".families",
    "draw_weighted_combined_family": ".families",
    "realize_weighted_draw": ".families",
    "measure_aperture_rms_nm": ".families",
    "renormalize_to_aperture_rms": ".families",
    "plot_psf_complete_analysis": "..plotting.psf_plots",
}


def __getattr__(name: str) -> Any:
    """Resolve one public PSF name without eager heavy imports.

    Parameters
    ----------
    name : `str`
        Requested public name.

    Returns
    -------
    value : `object`
        Public object imported from its defining submodule.
    """
    module_name = _EXPORT_MODULES.get(name)
    if module_name is None:
        raise AttributeError(name)
    module = import_module(module_name, __name__)
    return getattr(module, name)


def __dir__() -> list[str]:
    """Return package public names for IDE and star-import compatibility."""
    return sorted(__all__)
