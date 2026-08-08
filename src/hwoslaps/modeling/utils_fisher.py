"""Utility classes and summary helpers for Fisher-based detectability.

This module defines result containers for the Fisher / Asimov detector:

- profiled Fisher information on a subhalo-template amplitude,
- Asimov / expected local significance,
- nuisance-prior bookkeeping, and
- mode-by-mode PSF/systematics coupling summaries.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

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

    The existing Fisher/Asimov fields describe the fit-side analysis template
    alone.  When ``mismatch_enabled`` is true, the optional mismatch fields
    additionally describe truth-side data projected onto that template and
    the false-positive projection of the smooth truth-minus-fit residual.
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

    # Fisher / Asimov amplitude-test bookkeeping.
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
    mismatch_enabled: bool = False
    amplitude_hat_mismatch: Optional[float] = None
    q_mismatch: Optional[float] = None
    z_mismatch: Optional[float] = None
    amplitude_spurious: Optional[float] = None
    q_spurious: Optional[float] = None
    z_spurious: Optional[float] = None


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

    # Fisher / Asimov vector outputs.
    fisher_raw_by_position: Optional[np.ndarray] = None
    fisher_profiled_by_position: Optional[np.ndarray] = None
    q_asimov_local_by_position: Optional[np.ndarray] = None
    z_asimov_local_by_position: Optional[np.ndarray] = None
    sigma_amplitude_profiled_by_position: Optional[np.ndarray] = None
    degradation_by_position: Optional[np.ndarray] = None
    absorbed_fraction_by_position: Optional[np.ndarray] = None


@dataclass
class FisherGridMapData:
    """2D sensitivity-grid Fisher detectability output at fixed mass.

    All 2D arrays have shape ``(len(y_coords), len(x_coords))`` and are
    indexed so that ``array[i, j]`` corresponds to a subhalo placed at
    ``(y_coords[i], x_coords[j])`` in arcseconds.  Nodes excluded by the
    optional annulus restriction are ``NaN`` in the float arrays and
    ``False`` in all masks.

    The existing Fisher/Asimov arrays retain their matched-analysis meaning
    and are built from fit-side templates.  Optional mismatch arrays contain
    truth-data and smooth truth-minus-fit projections onto those templates.
    """

    y_coords: np.ndarray
    x_coords: np.ndarray
    spacing_arcsec: float
    centre_yx: Tuple[float, float]
    detection_q_threshold: float
    evaluated_mask_2d: np.ndarray
    detectable_mask_2d: np.ndarray
    q_asimov_2d: np.ndarray
    z_asimov_2d: np.ndarray
    fisher_raw_2d: np.ndarray
    fisher_profiled_2d: np.ndarray
    sigma_amplitude_profiled_2d: np.ndarray
    degradation_2d: np.ndarray
    absorbed_fraction_2d: np.ndarray
    num_positions_evaluated: int
    num_detectable: int
    detectable_area_arcsec2: float
    max_z_asimov: float
    median_z_asimov: float
    subhalo_mass: Optional[float] = None
    subhalo_model: Optional[str] = None
    lens_einstein_radius: Optional[float] = None
    mismatch_enabled: bool = False
    amplitude_hat_2d: Optional[np.ndarray] = None
    q_mismatch_2d: Optional[np.ndarray] = None
    z_mismatch_2d: Optional[np.ndarray] = None
    mismatch_detectable_mask_2d: Optional[np.ndarray] = None
    mismatch_detectable_area_arcsec2: Optional[float] = None
    num_mismatch_detectable: Optional[int] = None
    amplitude_spurious_2d: Optional[np.ndarray] = None
    q_spurious_2d: Optional[np.ndarray] = None
    z_spurious_2d: Optional[np.ndarray] = None
    false_positive_mask_2d: Optional[np.ndarray] = None
    false_positive_area_arcsec2: Optional[float] = None
    num_false_positive: Optional[int] = None
    max_z_spurious: Optional[float] = None
    source_image_asset_path: Optional[str] = None
    source_image_asset_sha256_16: Optional[str] = None


def save_fisher_grid_map_npz(grid_map: FisherGridMapData, path: Union[str, Path]) -> Path:
    """Persist a grid map as a compressed ``.npz`` archive.

    The archive holds every array and scalar field of
    :class:`FisherGridMapData`, so downstream analysis (detectable-area
    aggregation, mass-function folds) can run from disk without re-running
    the detector.
    """
    path = Path(path)
    payload = dict(
        y_coords=grid_map.y_coords,
        x_coords=grid_map.x_coords,
        spacing_arcsec=np.float64(grid_map.spacing_arcsec),
        centre_yx=np.asarray(grid_map.centre_yx, dtype=float),
        detection_q_threshold=np.float64(grid_map.detection_q_threshold),
        evaluated_mask_2d=grid_map.evaluated_mask_2d,
        detectable_mask_2d=grid_map.detectable_mask_2d,
        q_asimov_2d=grid_map.q_asimov_2d,
        z_asimov_2d=grid_map.z_asimov_2d,
        fisher_raw_2d=grid_map.fisher_raw_2d,
        fisher_profiled_2d=grid_map.fisher_profiled_2d,
        sigma_amplitude_profiled_2d=grid_map.sigma_amplitude_profiled_2d,
        degradation_2d=grid_map.degradation_2d,
        absorbed_fraction_2d=grid_map.absorbed_fraction_2d,
        num_positions_evaluated=np.int64(grid_map.num_positions_evaluated),
        num_detectable=np.int64(grid_map.num_detectable),
        detectable_area_arcsec2=np.float64(grid_map.detectable_area_arcsec2),
        max_z_asimov=np.float64(grid_map.max_z_asimov),
        median_z_asimov=np.float64(grid_map.median_z_asimov),
        subhalo_mass=np.float64(np.nan if grid_map.subhalo_mass is None else grid_map.subhalo_mass),
        subhalo_model=np.str_("" if grid_map.subhalo_model is None else grid_map.subhalo_model),
        lens_einstein_radius=np.float64(
            np.nan if grid_map.lens_einstein_radius is None else grid_map.lens_einstein_radius
        ),
    )
    if grid_map.q_mismatch_2d is not None:
        payload.update(
            mismatch_enabled=np.bool_(grid_map.mismatch_enabled),
            amplitude_hat_2d=grid_map.amplitude_hat_2d,
            q_mismatch_2d=grid_map.q_mismatch_2d,
            z_mismatch_2d=grid_map.z_mismatch_2d,
            mismatch_detectable_mask_2d=grid_map.mismatch_detectable_mask_2d,
            mismatch_detectable_area_arcsec2=np.float64(
                grid_map.mismatch_detectable_area_arcsec2
            ),
            num_mismatch_detectable=np.int64(grid_map.num_mismatch_detectable),
            amplitude_spurious_2d=grid_map.amplitude_spurious_2d,
            q_spurious_2d=grid_map.q_spurious_2d,
            z_spurious_2d=grid_map.z_spurious_2d,
            false_positive_mask_2d=grid_map.false_positive_mask_2d,
            false_positive_area_arcsec2=np.float64(
                grid_map.false_positive_area_arcsec2
            ),
            num_false_positive=np.int64(grid_map.num_false_positive),
            max_z_spurious=np.float64(grid_map.max_z_spurious),
        )
    if grid_map.source_image_asset_path is not None:
        payload['source_image_asset_path'] = np.str_(grid_map.source_image_asset_path)
    if grid_map.source_image_asset_sha256_16 is not None:
        payload['source_image_asset_sha256_16'] = np.str_(
            grid_map.source_image_asset_sha256_16
        )
    np.savez_compressed(path, **payload)
    return path


def load_fisher_grid_map_npz(path: Union[str, Path]) -> FisherGridMapData:
    """Load a Fisher grid map from a compressed ``.npz`` archive.

    Parameters
    ----------
    path : `str` or `pathlib.Path`
        Archive written by :func:`save_fisher_grid_map_npz`.

    Returns
    -------
    grid_map : `FisherGridMapData`
        Reconstructed grid map. Mismatch and source-image asset fields are
        optional, so archives written before those fields were introduced
        remain readable.
    """
    path = Path(path)
    with np.load(path, allow_pickle=False) as data:
        files = set(data.files)

        def optional_float(name):
            if name not in files:
                return None
            value = float(data[name])
            return None if np.isnan(value) else value

        def optional_string(name):
            if name not in files:
                return None
            return str(np.asarray(data[name]).item())

        mismatch_present = 'q_mismatch_2d' in files
        return FisherGridMapData(
            y_coords=np.asarray(data['y_coords']),
            x_coords=np.asarray(data['x_coords']),
            spacing_arcsec=float(data['spacing_arcsec']),
            centre_yx=tuple(np.asarray(data['centre_yx'], dtype=float)),
            detection_q_threshold=float(data['detection_q_threshold']),
            evaluated_mask_2d=np.asarray(data['evaluated_mask_2d']),
            detectable_mask_2d=np.asarray(data['detectable_mask_2d']),
            q_asimov_2d=np.asarray(data['q_asimov_2d']),
            z_asimov_2d=np.asarray(data['z_asimov_2d']),
            fisher_raw_2d=np.asarray(data['fisher_raw_2d']),
            fisher_profiled_2d=np.asarray(data['fisher_profiled_2d']),
            sigma_amplitude_profiled_2d=np.asarray(
                data['sigma_amplitude_profiled_2d']
            ),
            degradation_2d=np.asarray(data['degradation_2d']),
            absorbed_fraction_2d=np.asarray(data['absorbed_fraction_2d']),
            num_positions_evaluated=int(data['num_positions_evaluated']),
            num_detectable=int(data['num_detectable']),
            detectable_area_arcsec2=float(data['detectable_area_arcsec2']),
            max_z_asimov=float(data['max_z_asimov']),
            median_z_asimov=float(data['median_z_asimov']),
            subhalo_mass=optional_float('subhalo_mass'),
            subhalo_model=optional_string('subhalo_model') or None,
            lens_einstein_radius=optional_float('lens_einstein_radius'),
            mismatch_enabled=(
                bool(data['mismatch_enabled']) if 'mismatch_enabled' in files else False
            ),
            amplitude_hat_2d=(
                np.asarray(data['amplitude_hat_2d']) if mismatch_present else None
            ),
            q_mismatch_2d=(
                np.asarray(data['q_mismatch_2d']) if mismatch_present else None
            ),
            z_mismatch_2d=(
                np.asarray(data['z_mismatch_2d']) if mismatch_present else None
            ),
            mismatch_detectable_mask_2d=(
                np.asarray(data['mismatch_detectable_mask_2d'])
                if mismatch_present
                else None
            ),
            mismatch_detectable_area_arcsec2=optional_float(
                'mismatch_detectable_area_arcsec2'
            ),
            num_mismatch_detectable=(
                int(data['num_mismatch_detectable'])
                if 'num_mismatch_detectable' in files
                else None
            ),
            amplitude_spurious_2d=(
                np.asarray(data['amplitude_spurious_2d']) if mismatch_present else None
            ),
            q_spurious_2d=(
                np.asarray(data['q_spurious_2d']) if mismatch_present else None
            ),
            z_spurious_2d=(
                np.asarray(data['z_spurious_2d']) if mismatch_present else None
            ),
            false_positive_mask_2d=(
                np.asarray(data['false_positive_mask_2d'])
                if mismatch_present
                else None
            ),
            false_positive_area_arcsec2=optional_float('false_positive_area_arcsec2'),
            num_false_positive=(
                int(data['num_false_positive'])
                if 'num_false_positive' in files
                else None
            ),
            max_z_spurious=optional_float('max_z_spurious'),
            source_image_asset_path=optional_string('source_image_asset_path'),
            source_image_asset_sha256_16=optional_string(
                'source_image_asset_sha256_16'
            ),
        )


@dataclass
class FisherDetectionData:
    """Top-level Fisher result payload for pipeline integration.

    Local and grid payloads optionally include truth-vs-fit model mismatch
    statistics while their existing fields retain fit-template semantics.
    """

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

    # Provenance extras.
    nuisance_names: Optional[List[str]] = None
    prior_precision_diagonal: Optional[List[float]] = None
    n_psf_modes: int = 0
    psf_mode_names: Optional[List[str]] = None
    n_psf_fit_modes: int = 0
    n_psf_scan_modes: int = 0
    psf_fit_mode_names: Optional[List[str]] = None
    psf_scan_mode_names: Optional[List[str]] = None
    grid_map: Optional[FisherGridMapData] = None
    psf_mismatch_enabled: bool = False
    lens_mismatch_enabled: bool = False

    def __post_init__(self):
        """Stamp the generation timestamp when one was not supplied."""
        if self.generation_timestamp is None:
            self.generation_timestamp = datetime.now().isoformat()

    @property
    def has_local(self) -> bool:
        """Whether a local result is present (`bool`, read-only)."""
        return self.local is not None

    @property
    def has_map(self) -> bool:
        """Whether a map result is present (`bool`, read-only)."""
        return self.map is not None

    @property
    def has_grid_map(self) -> bool:
        """Whether a grid-map result is present (`bool`, read-only)."""
        return self.grid_map is not None


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
    """Print concise summary for Fisher detectability."""
    print("Fisher Detectability Summary:")
    print("-" * 32)
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

    if fisher_data.grid_map is not None:
        gmap = fisher_data.grid_map
        print("\nGrid map (fixed mass):")
        print(
            f"  Grid: {gmap.y_coords.size}x{gmap.x_coords.size} nodes, "
            f"spacing {gmap.spacing_arcsec:.4f} arcsec, centre {gmap.centre_yx}"
        )
        print(f"  Nodes evaluated: {gmap.num_positions_evaluated}")
        print(
            f"  Detectable nodes (q_F >= {gmap.detection_q_threshold:.1f}): "
            f"{gmap.num_detectable}"
        )
        print(f"  Detectable area: {gmap.detectable_area_arcsec2:.4f} arcsec^2")
        print(
            "  Z_asimov max / median over evaluated nodes: "
            f"{gmap.max_z_asimov:.4f} / {gmap.median_z_asimov:.4f}"
        )

    local_mismatch = (
        fisher_data.local is not None and fisher_data.local.mismatch_enabled
    )
    grid_mismatch = (
        fisher_data.grid_map is not None
        and fisher_data.grid_map.mismatch_enabled
    )
    if local_mismatch or grid_mismatch:
        print("\nModel mismatch:")
        if fisher_data.psf_mismatch_enabled:
            print("  fit_psf mode: explicit")
        if fisher_data.lens_mismatch_enabled:
            print("  fit_lens mode: explicit")
            print(
                "  Reference/pool lens mismatch uses ~2x per-node "
                "ray-tracing work."
            )
        if local_mismatch:
            local = fisher_data.local
            assert local is not None
            print(f"  Local q_mismatch: {local.q_mismatch:.4f}")
            print(f"  Local q_spurious: {local.q_spurious:.4f}")
        if grid_mismatch:
            gmap = fisher_data.grid_map
            assert gmap is not None
            print(
                "  Mismatch-detectable area: "
                f"{gmap.mismatch_detectable_area_arcsec2:.4f} arcsec^2"
            )
            print(
                "  False-positive area: "
                f"{gmap.false_positive_area_arcsec2:.4f} arcsec^2"
            )
