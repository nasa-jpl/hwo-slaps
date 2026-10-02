"""Fisher-based lens modeling module for HWO-SLAPS."""

from __future__ import annotations

from typing import Any
from .forecast_results import ForecastResult, ForecastSummary, summarize_forecast
from .mass_reach import MassReach, adaptive_mass_reach, mass_reach

from .fisher_core import (
    AsimovAmplitudeResult,
    SignalBankResult,
    SpuriousAmplitudeResult,
    SystematicModeCoupling,
    SystematicModeScanResult,
    Whitener,
    ProfileLikelihoodWorkspace,
    compute_asimov_detectability,
    evaluate_signal_bank,
    compute_spurious_amplitude,
    scan_systematic_modes,
    sidak_local_p,
    sidak_local_z,
    bonferroni_local_p,
    global_p_from_local,
    detectable_area,
)
from .utils_fisher import (
    FisherDetectionData,
    FisherGridMapData,
    FisherLocalData,
    FisherMapData,
    FisherModeCouplingData,
    FisherModeScanData,
    load_fisher_grid_map_npz,
    print_fisher_summary,
    save_fisher_grid_map_npz,
)

__all__ = [
    "ForecastResult",
    "ForecastSummary",
    "summarize_forecast",
    "MassReach",
    "mass_reach",
    "adaptive_mass_reach",
    "FisherDetector",
    "FisherDetectionData",
    "FisherGridMapData",
    "FisherLocalData",
    "FisherMapData",
    "FisherModeCouplingData",
    "FisherModeScanData",
    "load_fisher_grid_map_npz",
    "print_fisher_summary",
    "save_fisher_grid_map_npz",
    "AsimovAmplitudeResult",
    "SignalBankResult",
    "SpuriousAmplitudeResult",
    "SystematicModeCoupling",
    "SystematicModeScanResult",
    "Whitener",
    "ProfileLikelihoodWorkspace",
    "compute_asimov_detectability",
    "evaluate_signal_bank",
    "compute_spurious_amplitude",
    "scan_systematic_modes",
    "sidak_local_p",
    "sidak_local_z",
    "bonferroni_local_p",
    "global_p_from_local",
    "detectable_area",
]


def __getattr__(name: str) -> Any:
    """Resolve PyAutoLens-backed Fisher entry points only when requested."""
    if name == "FisherDetector":
        from .fisher_detector import FisherDetector

        return FisherDetector
    raise AttributeError(name)


def __dir__() -> list[str]:
    """Return package public names for IDE and star-import compatibility."""
    return sorted(__all__)
