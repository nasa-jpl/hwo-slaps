"""
Orchestration for Mejiro detectability analysis (paper-exact).

Provides a high-level function to execute the Mejiro detector using existing
ObservationData products, mirroring the style of the GOF and Chernoff wrappers.
Returns the common DetectionData container so downstream plotting/summary hooks
remain compatible.
"""

from typing import Optional, Dict

import numpy as np
from scipy.stats import chi2 as chi2_dist

from ..observation.utils import ObservationData
from ..lensing.utils import LensingData
from .utils import DetectionData
from .mejiro_detector import MejiroDetector, MejiroConfig


def perform_mejiro_detection(
    observation_baseline: ObservationData,   # B (no subhalo)
    observation_test: ObservationData,       # A (with subhalo)
    lensing_test: Optional[LensingData] = None,  # optional for truth metadata
    detection_config: Optional[Dict] = None,
    full_config: Optional[Dict] = None,
) -> DetectionData:
    """Run Mejiro paper-exact detectability and return unified DetectionData.

    Parameters
    ----------
    observation_baseline : ObservationData
        Baseline observation (no subhalo), provides SNR mask and B exposure.
    observation_test : ObservationData
        Test observation (with single subhalo) providing A exposure.
    lensing_test : LensingData, optional
        Lensing metadata for truth recording (position, mass).
    detection_config : dict, optional
        Configuration fields: 'snr_threshold', 'alpha', 'dof_offset'.
    full_config : dict, optional
        Full pipeline configuration for provenance.
    """
    # Defaults and overrides from config
    snr_threshold = 1.0
    alpha = 1.349898e-3
    dof_offset = -3
    if detection_config is not None:
        if 'snr_threshold' in detection_config:
            snr_threshold = float(detection_config['snr_threshold'])
        if 'alpha' in detection_config:
            alpha = float(detection_config['alpha'])
        if 'dof_offset' in detection_config:
            dof_offset = int(detection_config['dof_offset'])

    config = MejiroConfig(alpha=alpha, dof_offset=dof_offset, snr_threshold=snr_threshold)
    detector = MejiroDetector(baseline=observation_baseline, config=config)

    # Compute statistic
    chi2_value, dof, residual_full = detector.compute_chi2(observation_with_subhalo=observation_test)

    # Assemble DetectionData using standard GOF-style fields
    # Our Mejiro is a single chi2 with user-chosen alpha; we populate the
    # significance table at standard 3σ/4σ/5σ for consistency.
    significance_levels = [1.349898e-3, 3.167124e-5, 2.866516e-7]

    # Build per-threshold results with the same DetectionResult class as GOF
    detection_results = {}
    from .chi_square_detector import DetectionResult  # reuse container

    # Scalars used across thresholds
    dof_for_thresholds = dof
    rv = chi2_dist(df=dof_for_thresholds)
    for p in significance_levels:
        threshold = float(rv.isf(p))
        detected = bool(chi2_value > threshold)
        detection_results[p] = DetectionResult(
            chi2_value=chi2_value,
            threshold=threshold,
            detected=detected,
            significance_level=p,
            dof=dof_for_thresholds,
            position=(0.0, 0.0) if lensing_test is None or lensing_test.subhalo_position is None else lensing_test.subhalo_position,
            snr_mask=detector.snr_mask,
            residual=residual_full,
            num_regions=detector.num_regions,
            max_region_snr=detector.max_region_snr,
        )

    # Unified DetectionData
    return DetectionData(
        detection_results=detection_results,
        chi2_value=chi2_value,
        degrees_of_freedom=dof,
        snr_threshold=snr_threshold,
        significance_levels=significance_levels,
        pixels_unmasked=detector.pixels_unmasked,
        num_regions=detector.num_regions,
        max_region_snr=detector.max_region_snr,
        snr_mask=detector.snr_mask,
        snr_array=detector.snr_array,
        labeled_regions=detector.labeled_regions,
        residual_map=residual_full,
        variance_2d=None,
        true_subhalo_position=None if lensing_test is None else lensing_test.subhalo_position,
        true_subhalo_mass=None if lensing_test is None else lensing_test.subhalo_mass,
        true_subhalo_model=None if lensing_test is None else lensing_test.subhalo_model,
        baseline_exposure_time=observation_baseline.exposure_time,
        pixel_scale=observation_baseline.pixel_scale,
        detector_config=observation_baseline.detector_config,
        config=full_config,
    )


