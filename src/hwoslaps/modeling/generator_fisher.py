"""Orchestration wrapper for Fisher v1 detectability evaluation."""

from copy import deepcopy
from typing import Optional, Dict

from ..lensing.utils import LensingData
from ..observation.utils import ObservationData
from ..psf.utils import PSFData
from .fisher_detector import FisherDetector
from .utils_fisher import FisherDetectionData


def perform_fisher_detection(
    observation_baseline: ObservationData,
    observation_test: ObservationData,
    lensing_baseline: LensingData,
    lensing_test: LensingData,
    psf_data: PSFData,
    detection_config: Optional[Dict] = None,
    full_config: Optional[Dict] = None,
) -> FisherDetectionData:
    """Run Fisher v1 detectability with local/map modes.

    Parameters
    ----------
    observation_baseline : ObservationData
        Baseline observation (no subhalo).
    observation_test : ObservationData
        Test observation (with injected subhalo).
    lensing_baseline : LensingData
        Baseline lensing data used for nuisance linearization.
    lensing_test : LensingData
        Test lensing data providing subhalo truth metadata.
    psf_data : PSFData
        PSF system object shared by baseline and test observations.
    detection_config : dict, optional
        Full `modeling` config section containing the nested `fisher` block.
    full_config : dict, optional
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

    detector = FisherDetector(
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
        gram_condition_number=detector.gram_condition_number,
        pixel_scale=observation_baseline.pixel_scale,
        config=full_config,
    )
