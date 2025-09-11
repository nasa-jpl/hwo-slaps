"""Utility classes and functions for subhalo detection.

This module provides the DetectionData class and related utilities
for managing detection results and metadata, following the established
HWO-SLAPS pattern.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional, Tuple, List
from datetime import datetime
import numpy as np
from scipy.stats import chi2 as chi2_dist, norm
from .chi_square_detector import DetectionResult


@dataclass
class DetectionData:
    """Complete subhalo detection results with unified access.
    
    This class contains all products from chi-square subhalo detection
    in a unified structure with direct access to all key parameters,
    results, and diagnostic information.
    """
    # === PRIMARY RESULTS ===
    detection_results: Dict[float, DetectionResult]  # By significance level
    chi2_value: float
    degrees_of_freedom: int
    
    # === DETECTION PARAMETERS ===
    snr_threshold: float
    significance_levels: List[float]
    pixels_unmasked: int
    num_regions: int
    max_region_snr: float
    
    # === MASKS AND ARRAYS ===
    snr_mask: np.ndarray
    snr_array: np.ndarray
    labeled_regions: np.ndarray
    residual_map: np.ndarray
    variance_2d: Optional[np.ndarray] = None
    
    # === SUBHALO TRUTH ===
    true_subhalo_position: Optional[Tuple[float, float]] = None
    true_subhalo_mass: Optional[float] = None
    true_subhalo_model: Optional[str] = None
    
    # === OBSERVATION METADATA ===
    baseline_exposure_time: float = 1000.0
    pixel_scale: float = 0.05
    detector_config: Dict[str, float] = field(default_factory=dict)
    
    # === PROVENANCE ===
    config: Optional[Dict] = None
    generation_timestamp: Optional[str] = None
    
    def __post_init__(self):
        """Set generation timestamp if not provided."""
        if self.generation_timestamp is None:
            self.generation_timestamp = datetime.now().isoformat()
    
    # === RICH COMPUTED PROPERTIES ===
    @property
    def max_significance_detected(self) -> Optional[str]:
        """Highest significance achieved (formatted as σ)."""
        if not self.detection_results:
            return None
        detected_ps = [p for p, r in self.detection_results.items() if r.detected]
        if not detected_ps:
            return None
        # Highest significance = smallest p (one-sided)
        p_min = min(detected_ps)
        sigma = norm.isf(p_min)
        return f"{sigma:.2f}σ"
        
    @property
    def detection_summary(self) -> Dict:
        """Summary of all detection results keyed by p-value with sigma labels."""
        summary = {}
        for p in sorted(self.significance_levels):
            if p in self.detection_results:
                result = self.detection_results[p]
                sigma = float(norm.isf(p))
                summary[p] = {
                    'sigma': sigma,
                    'detected': result.detected,
                    'chi2_value': result.chi2_value,
                    'chi2_threshold': result.threshold,
                    'global_p_value': self.chi2_p_value,
                }
        return summary
        
    @property
    def is_detected_3sigma(self) -> bool:
        """Whether detected at 3σ significance."""
        return 0.001 in self.detection_results and self.detection_results[0.001].detected
        
    @property
    def is_detected_4sigma(self) -> bool:
        """Whether detected at 4σ significance."""
        return 0.0001 in self.detection_results and self.detection_results[0.0001].detected
        
    @property
    def is_detected_5sigma(self) -> bool:
        """Whether detected at 5σ significance."""
        return 0.00001 in self.detection_results and self.detection_results[0.00001].detected
    
    @property
    def chi2_p_value(self) -> float:
        """P-value for the chi-square statistic."""
        return chi2_dist(self.degrees_of_freedom).sf(self.chi2_value)
    
    @property
    def detection_mask_fraction(self) -> float:
        """Fraction of pixels used in detection analysis."""
        return self.pixels_unmasked / self.snr_array.size
    
    @property
    def has_subhalo_truth(self) -> bool:
        """Whether ground truth subhalo information is available."""
        return self.true_subhalo_position is not None
    
    @property
    def snr_array_2d(self) -> np.ndarray:
        """SNR array reshaped to 2D image format."""
        # Infer 2D shape from the square root of array size (assuming square images)
        side_length = int(np.sqrt(self.snr_array.size))
        return self.snr_array.reshape(side_length, side_length)
    
    @property
    def snr_mask_2d(self) -> np.ndarray:
        """SNR mask reshaped to 2D image format."""
        side_length = int(np.sqrt(self.snr_mask.size))
        return self.snr_mask.reshape(side_length, side_length)
    
    @property
    def residual_map_2d(self) -> np.ndarray:
        """Residual map reshaped to 2D image format."""
        side_length = int(np.sqrt(self.residual_map.size))
        return self.residual_map.reshape(side_length, side_length)
    
    @property
    def image_shape(self) -> Tuple[int, int]:
        """Shape of the detection arrays as (height, width)."""
        side_length = int(np.sqrt(self.snr_array.size))
        return (side_length, side_length)
    
    @property
    def field_of_view_arcsec(self) -> Tuple[float, float]:
        """Field of view in arcseconds as (height, width)."""
        height, width = self.image_shape
        return (height * self.pixel_scale, width * self.pixel_scale)


def validate_detection_results(detection_data: DetectionData) -> Dict[str, bool]:
    """Comprehensive validation of detection results.
    
    Implements key validation checks from the prototype notebook
    to ensure mathematical accuracy and physical plausibility.
    
    Returns
    -------
    validation_results : dict
        Dictionary of validation check results (True = passed).
    """
    validation = {}
    
    # 1. Basic data consistency
    validation['pixels_consistent'] = (
        detection_data.pixels_unmasked == np.sum(detection_data.snr_mask)
    )
    validation['arrays_same_size'] = (
        len(detection_data.snr_array) == len(detection_data.snr_mask) == 
        len(detection_data.residual_map)
    )
    
    # 2. SNR calculation verification
    snr_nonzero = detection_data.snr_array[detection_data.snr_array > 0]
    validation['snr_reasonable'] = len(snr_nonzero) > 0 and np.max(snr_nonzero) > detection_data.snr_threshold
    
    # 3. Detection mask consistency
    validation['mask_threshold_consistent'] = np.all(
        detection_data.snr_array[detection_data.snr_mask] > detection_data.snr_threshold
    )
    
    # 4. Chi-square value plausibility
    validation['chi2_positive'] = detection_data.chi2_value > 0
    validation['chi2_finite'] = np.isfinite(detection_data.chi2_value)
    validation['dof_positive'] = detection_data.degrees_of_freedom > 0
    
    # 5. Detection results consistency
    all_results_valid = True
    for sig_level, result in detection_data.detection_results.items():
        if not (result.chi2_value == detection_data.chi2_value and 
                result.dof == detection_data.degrees_of_freedom):
            all_results_valid = False
            break
    validation['detection_results_consistent'] = all_results_valid
    
    # 6. P-value reasonableness
    validation['p_value_valid'] = 0 <= detection_data.chi2_p_value <= 1
    
    return validation


def print_detection_summary(detection_data: DetectionData) -> None:
    """Print concise detection results summary."""
    print("Detection Summary:")
    print("-" * 30)
    
    # Overall result
    print(f"Max significance detected: {detection_data.max_significance_detected or 'None'}")
    
    # Per-threshold results (sorted by increasing significance p -> decreasing sigma)
    for p in sorted(detection_data.significance_levels):
        if p in detection_data.detection_results:
            result = detection_data.detection_results[p]
            sigma = norm.isf(p)
            status = 'YES' if result.detected else 'NO'
            print(
                f"{sigma:.2f}σ (p={p:.2e}): {status} "
                f"(χ²={result.chi2_value:.2f}, χ²_threshold={result.threshold:.2f})"
            )
    
    # Key statistics
    print(f"\nAnalysis parameters:")
    print(f"  SNR threshold: {detection_data.snr_threshold}")
    print(f"  Pixels analyzed: {detection_data.pixels_unmasked}")
    print(f"  Analysis fraction: {detection_data.detection_mask_fraction:.3f}")
    print(f"  Degrees of freedom: {detection_data.degrees_of_freedom}")
    print(f"  Global p-value (from χ²): {detection_data.chi2_p_value:.2e}")

    # Standard sigma checks (always shown)
    print("\nStandard significance checks:")
    standard_ps = [
        (1.349898e-3, "3σ"),
        (3.167124e-5, "4σ"),
        (2.866516e-7, "5σ"),
    ]
    dof = detection_data.degrees_of_freedom
    chi2_val = detection_data.chi2_value
    for p, label in standard_ps:
        sigma = norm.isf(p)
        chi2_thresh = chi2_dist.ppf(1.0 - p, dof)
        detected = chi2_val > chi2_thresh
        status = "YES" if detected else "NO"
        print(
            f"  {label} (p={p:.2e}, σ={sigma:.2f}): {status} "
            f"(χ²={chi2_val:.2f} vs χ²_threshold={chi2_thresh:.2f})"
        )
    
    # Subhalo info if available
    if detection_data.has_subhalo_truth:
        print(f"\nSubhalo truth:")
        print(f"  Mass: {detection_data.true_subhalo_mass:.1e} M_sun")
        print(f"  Model: {detection_data.true_subhalo_model}")
        print(f"  Position: {detection_data.true_subhalo_position}")


def validate_and_print_summary(detection_data: DetectionData) -> Dict[str, bool]:
    """Validate detection results and print summary if validation passes.
    
    Parameters
    ----------
    detection_data : DetectionData
        Detection results to validate and summarize.
        
    Returns
    -------
    validation_results : dict
        Dictionary of validation check results.
    """
    validation = validate_detection_results(detection_data)
    
    # Print validation results
    failed_checks = [check for check, passed in validation.items() if not passed]
    if failed_checks:
        print(f"⚠️  Validation failed for: {', '.join(failed_checks)}")
    else:
        print("✓ All validation checks passed")
    
    # Print summary if basic validation passes
    basic_checks = ['arrays_same_size', 'chi2_positive', 'chi2_finite', 'dof_positive']
    if all(validation.get(check, False) for check in basic_checks):
        print()
        print_detection_summary(detection_data)
    
    return validation