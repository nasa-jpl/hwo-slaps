"""
Orchestration for Chernoff minimal-fit detection.

Provides a high-level function to execute the Chernoff detector using existing
ObservationData and LensingData products, mirroring the style of the classic
chi-square generator while keeping the original implementation untouched.
"""

from typing import Optional, Dict

from ..observation.utils import ObservationData
from ..lensing.utils import LensingData
from .chernoff_detector import ChernoffSubhaloDetector, ChernoffDetectionData


def perform_chernoff_detection(
    observation_baseline: ObservationData,     # No subhalo
    observation_ref_with_subhalo: ObservationData,  # Reference H1 (noiseless_source used)
    observation_test: ObservationData,         # With or without subhalo (noisy)
    lensing_test: LensingData,                 # Truth metadata (position, mass)
    detection_config: Optional[Dict] = None,
) -> ChernoffDetectionData:
    """Run Chernoff minimal-fit detection at a fixed known position.

    Parameters
    ----------
    observation_baseline : ObservationData
        Baseline observation (H0) with no subhalo.
    observation_ref_with_subhalo : ObservationData
        Reference H1 observation whose noiseless source defines the linear template.
    observation_test : ObservationData
        The noisy observation to test (may contain a subhalo).
    lensing_test : LensingData
        Lensing metadata for the reference H1 (records position, mass).
    detection_config : dict, optional
        Configuration with keys like 'snr_threshold'.

    Returns
    -------
    ChernoffDetectionData
        Complete results and diagnostics for the Chernoff test.
    """
    snr_threshold = 1.0
    if detection_config is not None and 'snr_threshold' in detection_config:
        snr_threshold = float(detection_config['snr_threshold'])

    detector = ChernoffSubhaloDetector(
        observation_data_no_subhalo=observation_baseline,
        observation_data_with_subhalo_ref=observation_ref_with_subhalo,
        lensing_test=lensing_test,
        snr_threshold=snr_threshold,
        use_template=True,
    )

    subhalo_position = (
        lensing_test.subhalo_position if lensing_test.subhalo_position is not None else (0.0, 0.0)
    )

    return detector.detect_at_position(
        observation_with_subhalo=observation_test,
        subhalo_position=subhalo_position,
        compute_asimov=True,
    )


