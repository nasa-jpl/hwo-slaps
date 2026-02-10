"""Utility classes and functions for subhalo detection.

This module provides the DetectionData class and related utilities
for managing detection results and metadata, following the established
HWO-SLAPS pattern.
"""

from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple, List
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

    Notes
    -----
    Subhalo truth positions use the canonical lensing convention `(y, x)`
    in arcseconds.
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
    image_shape: Tuple[int, int]
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
        return self.snr_array.reshape(self.image_shape)
    
    @property
    def snr_mask_2d(self) -> np.ndarray:
        """SNR mask reshaped to 2D image format."""
        return self.snr_mask.reshape(self.image_shape)
    
    @property
    def residual_map_2d(self) -> np.ndarray:
        """Residual map reshaped to 2D image format."""
        return self.residual_map.reshape(self.image_shape)
    
    @property
    def field_of_view_arcsec(self) -> Tuple[float, float]:
        """Field of view in arcseconds as (height, width)."""
        height, width = self.image_shape
        return (height * self.pixel_scale, width * self.pixel_scale)


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
    print("\nAnalysis parameters:")
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
        print("\nSubhalo truth:")
        print(f"  Mass: {detection_data.true_subhalo_mass:.1e} M_sun")
        print(f"  Model: {detection_data.true_subhalo_model}")
        print(f"  Position: {detection_data.true_subhalo_position}")
