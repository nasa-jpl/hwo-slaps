"""Orchestration wrapper for Fisher detectability evaluation."""

from __future__ import annotations

import os
from copy import deepcopy
from time import perf_counter
from typing import Dict, Optional

from ..lensing.utils import LensingData
from ..observation.utils import ObservationData
from ..psf.utils import PSFData
from .fisher_detector import FisherDetector
from .utils_fisher import FisherDetectionData


def _fisher_timing_enabled() -> bool:
    disable_env = os.environ.get("HWOSLAPS_DISABLE_FISHER_TIMING", "").strip().lower()
    return disable_env not in {"1", "true", "yes", "on"}


def _log_fisher_timing(label: str, elapsed_s: float) -> None:
    if _fisher_timing_enabled():
        print(f"[Fisher] timing: {label} finished in {elapsed_s:.2f} s")


def perform_fisher_detection(
    observation_baseline: ObservationData,
    observation_test: ObservationData,
    lensing_baseline: LensingData,
    lensing_test: LensingData,
    psf_data: PSFData,
    detection_config: Optional[Dict] = None,
    full_config: Optional[Dict] = None,
) -> FisherDetectionData:
    """Run Fisher detectability with local / map modes.

    Parameters
    ----------
    observation_baseline
        Baseline observation (no subhalo).
    observation_test
        Test observation (with injected subhalo).
    lensing_baseline
        Baseline lensing data used for nuisance linearization.
    lensing_test
        Test lensing data providing subhalo truth metadata.
    psf_data
        PSF system object shared by baseline and test observations.
    detection_config
        Full ``modeling`` config section containing the nested ``fisher``
        block.
    full_config
        Full pipeline config for provenance.
    """
    if detection_config is None:
        raise ValueError("detection_config must be provided for Fisher detection.")
    if full_config is None:
        raise ValueError("full_config must be provided for Fisher detection.")
    if "fisher" not in detection_config:
        raise ValueError("modeling.fisher block is required for Fisher detection.")

    fisher_cfg = deepcopy(detection_config["fisher"])
    mode = fisher_cfg["mode"].lower()
    if mode not in {"local", "map", "both"}:
        raise ValueError("modeling.fisher.mode must be one of: local, map, both")

    start = perf_counter()
    detector = FisherDetector(
        observation_baseline=observation_baseline,
        lensing_baseline=lensing_baseline,
        psf_data=psf_data,
        full_config=full_config,
        fisher_config=fisher_cfg,
    )
    _log_fisher_timing("detector initialization", perf_counter() - start)

    local_data = None
    map_data = None
    grid_map_data = None
    if mode in {"local", "both"}:
        start = perf_counter()
        local_data = detector.compute_local(
            observation_test=observation_test,
            lensing_test=lensing_test,
        )
        _log_fisher_timing("top-level local computation", perf_counter() - start)
    if mode in {"map", "both"}:
        start = perf_counter()
        if detector.map_type == "grid":
            grid_map_data = detector.compute_grid_map()
            _log_fisher_timing("top-level grid map computation", perf_counter() - start)
        else:
            map_data = detector.compute_map()
            _log_fisher_timing("top-level map computation", perf_counter() - start)

    return FisherDetectionData(
        mode=mode,
        local=local_data,
        map=map_data,
        grid_map=grid_map_data,
        snr_threshold=float(fisher_cfg["snr_threshold"]),
        include_background_offset=bool(fisher_cfg["include_background_offset"]),
        finite_diff=deepcopy(fisher_cfg["finite_diff"]),
        map_config=deepcopy(fisher_cfg["map"]),
        pixels_unmasked=detector.pixels_unmasked,
        n_nuisance=detector.n_nuisance,
        gram_condition_number=float(detector.gram_condition_number),
        pixel_scale=observation_baseline.pixel_scale,
        config=full_config,
        nuisance_names=list(getattr(detector, "nuisance_names", []) or []),
        prior_precision_diagonal=list(
            getattr(detector, "prior_precision_diagonal", []) or []
        ),
        n_psf_modes=int(getattr(detector, "n_psf_modes", 0)),
        psf_mode_names=list(getattr(detector, "psf_mode_names", []) or []),
        n_psf_fit_modes=int(getattr(detector, "n_psf_fit_modes", 0)),
        n_psf_scan_modes=int(getattr(detector, "n_psf_scan_modes", 0)),
        psf_fit_mode_names=list(getattr(detector, "psf_fit_mode_names", []) or []),
        psf_scan_mode_names=list(getattr(detector, "psf_scan_mode_names", []) or []),
        psf_mismatch_enabled=detector.psf_mismatch_enabled,
        lens_mismatch_enabled=detector.lens_mismatch_enabled,
    )
