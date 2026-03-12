"""Utility classes and summary helpers for Fisher-based detectability.

This module defines result containers for both the legacy Fisher v1 detector and
for the publication-grade Fisher/Asimov detector.  The publication path keeps
backward-compatible field names (for existing plots and pipeline routing) while
adding the statistically meaningful quantities needed for a paper:

- profiled Fisher information on a subhalo-template amplitude,
- Asimov / expected local significance,
- nuisance-prior bookkeeping, and
- mode-by-mode PSF/systematics coupling summaries.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Optional, Tuple, List, Any, Sequence
import numpy as np


@dataclass
class FisherModeCouplingData:
    """Coupling summary for one systematic / PSF mode.

    Parameters
    ----------
    mode_name
        Name of the systematic mode.
    amplitude_per_unit
        Spurious best-fit subhalo amplitude induced by one unit of the mode.
    z_per_unit
        Spurious local significance induced by one unit of the mode.
    one_sigma_z
        Spurious local significance for the configured 1-sigma mode amplitude,
        when such a prior / calibration uncertainty is supplied.
    tolerance_for_zmax
        Maximum allowed mode amplitude for a chosen significance budget.
    """

    mode_name: str
    amplitude_per_unit: float
    z_per_unit: float
    one_sigma_z: Optional[float] = None
    tolerance_for_zmax: Optional[float] = None


@dataclass
class FisherModeScanData:
    """Mode-by-mode PSF/systematic susceptibility summary."""

    couplings: Sequence[FisherModeCouplingData]
    sigma_amplitude_profiled: float
    fisher_profiled: float
    rms_spurious_amplitude: Optional[float] = None
    rms_spurious_z: Optional[float] = None
    z_tolerance: Optional[float] = None


@dataclass
class FisherLocalData:
    """Single-position Fisher detectability output.

    The first block of fields is kept compatible with the legacy v1 detector.
    Additional optional fields expose the publication-grade quantities.
    """

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

    # Publication-grade amplitude-test bookkeeping.
    fisher_raw: Optional[float] = None
    fisher_profiled: Optional[float] = None
    sigma_amplitude_raw: Optional[float] = None
    sigma_amplitude_profiled: Optional[float] = None
    q_asimov_local: Optional[float] = None
    z_asimov_local: Optional[float] = None
    local_p_one_sided: Optional[float] = None
    absorbed_fraction: Optional[float] = None
    residual_norm_whitened: Optional[float] = None
    nuisance_prior_penalty: Optional[float] = None
    nuisance_rank: Optional[int] = None
    whitened_size: Optional[int] = None
    psf_mode_scan: Optional[FisherModeScanData] = None


@dataclass
class FisherMapData:
    """Ring-map / signal-bank Fisher detectability output at fixed mass."""

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

    # Publication-grade vector outputs.
    fisher_raw_by_position: Optional[np.ndarray] = None
    fisher_profiled_by_position: Optional[np.ndarray] = None
    q_asimov_local_by_position: Optional[np.ndarray] = None
    z_asimov_local_by_position: Optional[np.ndarray] = None
    sigma_amplitude_profiled_by_position: Optional[np.ndarray] = None
    degradation_by_position: Optional[np.ndarray] = None
    absorbed_fraction_by_position: Optional[np.ndarray] = None


@dataclass
class FisherDetectionData:
    """Top-level Fisher result payload for pipeline integration."""

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

    # Publication / provenance extras.
    version: str = "v1"
    nuisance_names: Optional[List[str]] = None
    prior_precision_diagonal: Optional[List[float]] = None
    n_psf_modes: int = 0
    psf_mode_names: Optional[List[str]] = None
    n_psf_fit_modes: int = 0
    n_psf_scan_modes: int = 0
    psf_fit_mode_names: Optional[List[str]] = None
    psf_scan_mode_names: Optional[List[str]] = None
    publication_config: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        if self.generation_timestamp is None:
            self.generation_timestamp = datetime.now().isoformat()

    @property
    def has_local(self) -> bool:
        return self.local is not None

    @property
    def has_map(self) -> bool:
        return self.map is not None



def _format_mode_scan(mode_scan: FisherModeScanData, max_modes: int = 5) -> List[str]:
    couplings = sorted(
        list(mode_scan.couplings),
        key=lambda coupling: abs(coupling.one_sigma_z)
        if coupling.one_sigma_z is not None
        else abs(coupling.z_per_unit),
        reverse=True,
    )
    lines: List[str] = []
    for coupling in couplings[:max_modes]:
        if coupling.one_sigma_z is not None:
            lines.append(
                f"    {coupling.mode_name}: z/unit={coupling.z_per_unit:.4g}, "
                f"z(1σ)={coupling.one_sigma_z:.4g}, "
                f"tol={coupling.tolerance_for_zmax}"
            )
        else:
            lines.append(
                f"    {coupling.mode_name}: z/unit={coupling.z_per_unit:.4g}, "
                f"tol={coupling.tolerance_for_zmax}"
            )
    if mode_scan.rms_spurious_z is not None:
        lines.append(f"    RMS spurious z from supplied covariance: {mode_scan.rms_spurious_z:.4g}")
    return lines



def print_fisher_summary(fisher_data: FisherDetectionData) -> None:
    """Print concise summary for Fisher detectability.

    The function is intentionally backward-compatible with v1 output while also
    surfacing the publication-grade amplitude / mode-scan quantities when they
    are present.
    """
    version = getattr(fisher_data, "version", "v1")
    print("Fisher Detectability Summary:")
    print("-" * 32)
    print(f"Version: {version}")
    print(f"Mode: {fisher_data.mode}")
    print(f"Pixels analyzed: {fisher_data.pixels_unmasked}")
    print(f"Nuisance directions: {fisher_data.n_nuisance}")
    print(f"Gram / normal-matrix condition number: {fisher_data.gram_condition_number:.3e}")
    print(f"SNR mask threshold: {fisher_data.snr_threshold:.3f}")
    if fisher_data.n_psf_fit_modes:
        print(f"PSF fit nuisance modes tracked: {fisher_data.n_psf_fit_modes}")
    elif fisher_data.n_psf_modes:
        print(f"PSF/systematic modes tracked: {fisher_data.n_psf_modes}")
    if fisher_data.n_psf_scan_modes:
        print(f"PSF scan modes tracked: {fisher_data.n_psf_scan_modes}")

    if fisher_data.local is not None:
        local = fisher_data.local
        print("\nLocal (injected position):")
        print(f"  SNR_asimov: {local.snr_asimov:.4f}")
        print(f"  DeltaChi2 raw/profiled: {local.delta_chi2_raw:.4f} / {local.delta_chi2_profiled:.4f}")
        print(f"  Profiling degradation: {local.degradation:.4f}")
        if local.sigma_amplitude_profiled is not None:
            print(f"  Sigma_amplitude_profiled: {local.sigma_amplitude_profiled:.4g}")
        if local.local_p_one_sided is not None:
            print(f"  Local one-sided p-value: {local.local_p_one_sided:.4g}")
        if local.true_subhalo_mass is not None:
            print(f"  Subhalo mass: {local.true_subhalo_mass:.3e} M_sun")
        if local.true_subhalo_position is not None:
            print(f"  Subhalo position (y, x): {local.true_subhalo_position}")
        if local.psf_mode_scan is not None and len(local.psf_mode_scan.couplings) > 0:
            print("  Leading PSF/systematic couplings:")
            for line in _format_mode_scan(local.psf_mode_scan):
                print(line)

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
