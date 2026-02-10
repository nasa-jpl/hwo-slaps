"""Utility classes and summary helpers for Fisher-based detectability.

This module defines Fisher-specific result containers used by the v1 detector.
Unlike legacy chi-square outputs, these structures represent Asimov expected
detectability metrics after profiling nuisance directions.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Optional, Tuple, List, Any
import numpy as np


@dataclass
class FisherLocalData:
    """Single-position Fisher detectability output."""

    snr_asimov: float
    delta_chi2_raw: float
    delta_chi2_profiled: float
    degradation: float
    pixels_unmasked: int
    n_nuisance: int
    gram_condition_number: float
    true_subhalo_position: Optional[Tuple[float, float]] = None
    true_subhalo_mass: Optional[float] = None
    true_subhalo_model: Optional[str] = None


@dataclass
class FisherMapData:
    """Ring-map Fisher detectability output at fixed mass."""

    positions_yx: np.ndarray
    snr_asimov_by_position: np.ndarray
    delta_chi2_profiled_by_position: np.ndarray
    delta_chi2_raw_by_position: np.ndarray
    num_positions: int
    median_snr_asimov: float
    p25_snr_asimov: float
    p75_snr_asimov: float
    min_snr_asimov: float
    max_snr_asimov: float


@dataclass
class FisherDetectionData:
    """Top-level Fisher v1 result payload for pipeline integration."""

    mode: str
    local: Optional[FisherLocalData]
    map: Optional[FisherMapData]
    snr_threshold: float
    include_background_offset: bool
    finite_diff: Dict[str, float]
    map_config: Dict[str, Any]
    pixels_unmasked: int
    n_nuisance: int
    gram_condition_number: float
    pixel_scale: float
    config: Optional[Dict] = None
    generation_timestamp: Optional[str] = None

    def __post_init__(self):
        if self.generation_timestamp is None:
            self.generation_timestamp = datetime.now().isoformat()

    @property
    def has_local(self) -> bool:
        return self.local is not None

    @property
    def has_map(self) -> bool:
        return self.map is not None


def print_fisher_summary(fisher_data: FisherDetectionData) -> None:
    """Print concise summary for Fisher v1 detectability."""
    print("Fisher Detectability Summary:")
    print("-" * 32)
    print(f"Mode: {fisher_data.mode}")
    print(f"Pixels analyzed: {fisher_data.pixels_unmasked}")
    print(f"Nuisance directions: {fisher_data.n_nuisance}")
    print(f"Gram condition number: {fisher_data.gram_condition_number:.3e}")
    print(f"SNR mask threshold: {fisher_data.snr_threshold:.3f}")

    if fisher_data.local is not None:
        local = fisher_data.local
        print("\nLocal (injected position):")
        print(f"  SNR_asimov: {local.snr_asimov:.4f}")
        print(f"  DeltaChi2 raw/profiled: {local.delta_chi2_raw:.4f} / {local.delta_chi2_profiled:.4f}")
        print(f"  Profiling degradation: {local.degradation:.4f}")
        if local.true_subhalo_mass is not None:
            print(f"  Subhalo mass: {local.true_subhalo_mass:.3e} M_sun")
        if local.true_subhalo_position is not None:
            print(f"  Subhalo position (y, x): {local.true_subhalo_position}")

    if fisher_data.map is not None:
        fmap = fisher_data.map
        print("\nMap (fixed mass):")
        print(f"  Positions evaluated: {fmap.num_positions}")
        print(
            "  SNR_asimov median [p25, p75], min, max: "
            f"{fmap.median_snr_asimov:.4f} "
            f"[{fmap.p25_snr_asimov:.4f}, {fmap.p75_snr_asimov:.4f}], "
            f"{fmap.min_snr_asimov:.4f}, {fmap.max_snr_asimov:.4f}"
        )
