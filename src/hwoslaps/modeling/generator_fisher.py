"""Orchestration wrapper for Fisher detectability evaluation.

This module now supports two detector backends:

- ``version='v1'``: the historical hard-coded prototype.
- ``version='publication'``: the publication-grade Fisher / Asimov detector.

The public pipeline interface is unchanged.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Optional, Dict

from ..lensing.utils import LensingData
from ..observation.utils import ObservationData
from ..psf.utils import PSFData
from .fisher_detector import FisherDetector
from .fisher_publication_detector import PublicationFisherDetector
from .utils_fisher import FisherDetectionData


_DEF_VERSION = "publication"


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
        Full ``modeling`` config section containing the nested ``fisher`` block.
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

    version = str(fisher_cfg.get("version", _DEF_VERSION)).lower()
    if version not in {"v1", "publication"}:
        raise ValueError("modeling.fisher.version must be one of: v1, publication")

    detector_cls = FisherDetector if version == "v1" else PublicationFisherDetector
    detector = detector_cls(
        observation_baseline=observation_baseline,
        lensing_baseline=lensing_baseline,
        psf_data=psf_data,
        full_config=full_config,
        fisher_config=fisher_cfg,
    )

    local_data = None
    map_data = None
    if mode in {"local", "both"}:
        local_data = detector.compute_local(
            observation_test=observation_test,
            lensing_test=lensing_test,
        )
    if mode in {"map", "both"}:
        map_data = detector.compute_map()

    publication_cfg = deepcopy(fisher_cfg.get("publication")) if version == "publication" else None

    return FisherDetectionData(
        mode=mode,
        local=local_data,
        map=map_data,
        snr_threshold=float(fisher_cfg["snr_threshold"]),
        include_background_offset=bool(fisher_cfg["include_background_offset"]),
        finite_diff=deepcopy(fisher_cfg["finite_diff"]),
        map_config=deepcopy(fisher_cfg["map"]),
        pixels_unmasked=detector.pixels_unmasked,
        n_nuisance=detector.n_nuisance,
        gram_condition_number=float(detector.gram_condition_number),
        pixel_scale=observation_baseline.pixel_scale,
        config=full_config,
        version=version,
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
        publication_config=publication_cfg,
    )
