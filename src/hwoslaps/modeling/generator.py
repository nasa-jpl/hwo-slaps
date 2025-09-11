"""Generator functions for subhalo detection analysis.

This module implements the main orchestration function for chi-square
subhalo detection, following the established HWO-SLAPS pattern.
"""

from typing import Dict, Optional
from ..lensing.utils import LensingData
from ..observation.utils import ObservationData
from .utils import DetectionData
from .chi_square_detector import ChiSquareSubhaloDetector


def perform_subhalo_detection(
    observation_baseline: ObservationData,     # No subhalo
    observation_test: ObservationData,         # With subhalo
    lensing_baseline: LensingData,            # For ground truth source
    lensing_test: LensingData,                # For subhalo truth info
    detection_config: Optional[Dict] = None,
    full_config: Optional[Dict] = None
) -> DetectionData:
    """Perform chi-square subhalo detection analysis.
    
    This function implements the validated chi-square methodology from
    the prototype notebook, comparing baseline and test observations
    to detect subhalo signatures with statistical significance testing.
    
    Parameters
    ----------
    observation_baseline : ObservationData
        Observation data without subhalo (expected/null hypothesis).
    observation_test : ObservationData  
        Observation data with subhalo (test case).
    lensing_baseline : LensingData
        Lensing system without subhalo (for ground truth source).
    lensing_test : LensingData
        Lensing system with subhalo (for subhalo truth parameters).
    detection_config : dict, optional
        Detection-specific parameters. If None, uses defaults.
    full_config : dict, optional
        Full configuration dictionary for provenance.
        
    Returns
    -------
    detection_data : DetectionData
        Complete detection results with unified structure providing
        direct access to all results, diagnostics, and metadata.
        
    Notes
    -----
    This function preserves the exact methodology from the validated
    prototype implementation in notebooks/mod4-chisquare.py.
    
    The chi-square detection uses:
    - SNR-based pixel masking (absolute SNR threshold)
    - Regional connectivity analysis with cross-shaped structure  
    - Pearson's chi-square test with full variance model
    - Multi-significance testing at 3σ, 4σ, 5σ levels (defaults shown with
      both sigma and one-sided p-values): 3σ (p≈1.35e-3), 4σ (p≈3.17e-5),
      5σ (p≈2.87e-7)
    
    Examples
    --------
    Perform detection and access results:
    
    >>> detection_data = perform_subhalo_detection(obs_baseline, obs_test, 
    ...                                           lens_baseline, lens_test)
    >>> print(f"5σ detection: {detection_data.is_detected_5sigma}")
    >>> print(f"Chi² value: {detection_data.chi2_value:.2f}")
    >>> print(f"Max significance: {detection_data.max_significance_detected}")
    """
    # Strict: detection_config must be provided by pipeline validation if modeling is enabled
    if detection_config is None:
        raise ValueError("detection_config must be provided explicitly when modeling is enabled")
    
    # Extract ground truth source counts (EXACTLY as in prototype lines 313-315)
    source_counts_ground_truth = (
        observation_baseline.noiseless_source_eps * observation_baseline.exposure_time
    )
    
    # Initialize detector (EXACTLY as in prototype lines 318-322)
    detector = ChiSquareSubhaloDetector(
        observation_data_no_subhalo=observation_baseline,
        source_counts_ground_truth=source_counts_ground_truth,
        snr_threshold=detection_config['snr_threshold'],
        significance_levels=detection_config['significance_levels']
    )
    
    # Get subhalo position from test case (EXACTLY as in prototype lines 344-345)
    if lensing_test.has_subhalo:
        subhalo_x = lensing_test.subhalo_position[1]  # x is second coordinate
        subhalo_y = lensing_test.subhalo_position[0]  # y is first coordinate
        subhalo_position = (subhalo_x, subhalo_y)
    else:
        subhalo_position = (0.0, 0.0)  # Default for null tests
    
    # Perform detection (EXACTLY as in prototype line 349)
    results = detector.detect_at_position(observation_test, subhalo_position)
    
    # Package results in unified structure
    return DetectionData(
        # Primary results
        detection_results=results,
        # Use the most stringent (smallest p-value) result for main value
        chi2_value=results[min(detector.significance_levels)].chi2_value,
        degrees_of_freedom=results[min(detector.significance_levels)].dof,
        
        # Detection parameters
        snr_threshold=detector.snr_threshold,
        significance_levels=detector.significance_levels,
        pixels_unmasked=detector.pixels_unmasked,
        num_regions=detector.num_regions,
        max_region_snr=detector.max_region_snr,
        
        # Arrays and masks
        snr_mask=detector.snr_mask,
        snr_array=detector.snr_array,
        labeled_regions=detector.labeled_regions,
        residual_map=results[min(detector.significance_levels)].residual,
        variance_2d=getattr(detector, 'variance_2d', None),
        
        # Truth information
        true_subhalo_position=subhalo_position if lensing_test.has_subhalo else None,
        true_subhalo_mass=lensing_test.subhalo_mass,
        true_subhalo_model=lensing_test.subhalo_model,
        
        # Metadata
        baseline_exposure_time=observation_baseline.exposure_time,
        pixel_scale=observation_baseline.pixel_scale,
        detector_config=observation_baseline.detector_config,
        config=full_config
    )