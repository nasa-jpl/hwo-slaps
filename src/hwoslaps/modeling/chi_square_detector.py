"""Chi-square subhalo detection implementation.

This module contains the core detection classes for statistically rigorous
subhalo detection using chi-square statistics with proper variance modeling.
"""

from dataclasses import dataclass
from typing import Dict, Tuple, List
import numpy as np
from scipy import ndimage
from scipy.stats import chi2
from ..observation.utils import ObservationData


@dataclass
class DetectionResult:
    """Store results from subhalo detection."""
    chi2_value: float
    threshold: float
    detected: bool
    significance_level: float
    dof: int
    position: Tuple[float, float]
    snr_mask: np.ndarray
    residual: np.ndarray
    num_regions: int
    max_region_snr: float


class ChiSquareSubhaloDetector:
    """Subhalo detection using chi-square statistics.
    
    Uses physically motivated variance model including both Poisson shot noise
    and detector read noise for statistically rigorous detection.
    
    Best-practice implementation: reuse the baseline observation's noise map
    (which encodes per-pixel standard deviation in ADU including source, sky,
    dark, and read noise) for both SNR masking and chi-square weighting. This
    keeps modeling exactly consistent with the observation physics and avoids
    recomputing variance with potential omissions.
    """
    
    def __init__(self, 
                 observation_data_no_subhalo: ObservationData,
                 source_counts_ground_truth: np.ndarray,
                  snr_threshold: float,
                  significance_levels: List[float]):
        """
        Initialize the detector.
        
        Parameters
        ----------
        observation_data_no_subhalo : ObservationData
            The final simulated observation data (without subhalo).
        source_counts_ground_truth : np.ndarray
            The noise-free, PSF-convolved source image in total electron counts.
        snr_threshold : float
            Absolute SNR threshold for pixel selection (default: 1.0).
        significance_levels : List[float]
            Significance levels for detection thresholds (e.g., 3σ, 4σ, 5σ).
        """
        self.observation_data_no_subhalo = observation_data_no_subhalo
        self.source_counts_ground_truth = source_counts_ground_truth
        self.snr_threshold = snr_threshold
        self.significance_levels = significance_levels
        
        # Calculate SNR regions using the baseline observation's noise map
        self.snr_array, self.labeled_regions, self.snr_mask, self.num_regions, self.max_region_snr = self._calculate_snr_regions()
        self.pixels_unmasked = np.sum(self.snr_mask)
    
    def _calculate_snr_regions(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int, float]:
        """
        Create mask based on SNR regions using the baseline observation's
        per-pixel noise map. SNR is computed as source (ADU) divided by the
        noise (ADU). The stored variance is the squared noise map (ADU^2).
        
        Returns
        -------
        snr_array : np.ndarray
            1D array of SNR values for all pixels.
        labeled_regions : np.ndarray
            1D array of region labels for connected high-SNR areas.
        snr_mask : np.ndarray
            1D boolean mask for pixels above SNR threshold.
        num_regions : int
            Number of connected regions above threshold.
        max_region_snr : float
            Maximum regional SNR value.
        """
        # Baseline noise map (ADU) and source in ADU
        gain = self.observation_data_no_subhalo.gain
        noise_map_adu_2d = self.observation_data_no_subhalo.noise_map.native
        source_counts_2d_e = self.source_counts_ground_truth  # electrons
        source_adu_2d = source_counts_2d_e / gain

        # Compute SNR in ADU space
        eps = 1e-12
        snr_array_2d = source_adu_2d / np.maximum(noise_map_adu_2d, eps)
        snr_array_2d = np.nan_to_num(snr_array_2d, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Create mask using absolute SNR threshold
        snr_mask_2d = snr_array_2d > self.snr_threshold
        
        # Label connected regions using cross-shaped connectivity
        structure = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]])
        labeled_regions_2d, num_regions = ndimage.label(snr_mask_2d, structure=structure)
        
        # Calculate regional SNR for each region using ADU domain
        regional_snrs = []
        for i in range(1, num_regions + 1):
            region_mask = labeled_regions_2d == i
            region_source_adu = float(np.sum(source_adu_2d[region_mask]))
            region_variance_adu = float(np.sum((noise_map_adu_2d[region_mask]) ** 2))
            regional_snr = region_source_adu / np.sqrt(region_variance_adu) if region_variance_adu > 0 else 0.0
            regional_snrs.append(regional_snr)
        
        max_region_snr = max(regional_snrs) if regional_snrs else 0
        
        # Store variance for diagnostics (ADU^2)
        self.variance_2d = noise_map_adu_2d**2
        
        # Flatten arrays to 1D for consistency with rest of pipeline
        snr_array = snr_array_2d.flatten()
        labeled_regions = labeled_regions_2d.flatten()
        snr_mask = snr_mask_2d.flatten()
        
        return snr_array, labeled_regions, snr_mask, num_regions, max_region_snr
    
    def detect_at_position(
        self,
        observation_with_subhalo: ObservationData,
        subhalo_position: Tuple[float, float]
    ) -> Dict[float, DetectionResult]:
        """
        Perform Pearson's chi-square test to detect subhalo.
        
        Compares observed (noisy) data to expected (noiseless) model
        using proper variance that accounts for both Poisson and read noise.
        
        Parameters
        ----------
        observation_with_subhalo : ObservationData
            Observation that may contain a subhalo.
        subhalo_position : Tuple[float, float]
            Position where subhalo was injected (for recording).
            
        Returns
        -------
        results : Dict[float, DetectionResult]
            Detection results for each significance level.
        """
        
        # Get expected values from noiseless source and include background means
        # to match observed data statistics (work in ADU).
        baseline = self.observation_data_no_subhalo
        gain = baseline.gain
        exposure_time = baseline.exposure_time
        expected_source_e = baseline.noiseless_source_eps * exposure_time  # e-
        sky_e = baseline.sky_background * exposure_time  # e-/pix
        dark_e = baseline.dark_current * exposure_time  # e-/pix
        expected_adu = (expected_source_e + sky_e + dark_e) / gain  # ADU
        
        # Get observed data (includes noise and possibly subhalo)
        observed_adu = observation_with_subhalo.data.native  # ADU
        
        # Flatten arrays
        expected_full = expected_adu.flatten()
        observed_full = observed_adu.flatten()
        
        # Use baseline noise map for chi-square variance (ADU^2)
        variance_adu_full = (baseline.noise_map.native ** 2).flatten()
        
        # Apply mask to select high-SNR pixels
        expected = expected_full[self.snr_mask]
        observed = observed_full[self.snr_mask]
        variance_adu = variance_adu_full[self.snr_mask]
        
        # Calculate Pearson's chi-square statistic
        epsilon = 1e-10  # Avoid division by zero
        chi2_value = np.sum((observed - expected)**2 / (variance_adu + epsilon))
        
        # Degrees of freedom (number of pixels)
        dof = self.pixels_unmasked
        
        # Test against significance thresholds
        results = {}
        for sig_level in self.significance_levels:
            # Get threshold from chi-square distribution
            rv = chi2(dof)
            threshold = rv.isf(sig_level)
            detected = chi2_value > threshold
            
            # Create residual map
            full_residual = np.zeros_like(expected_full)
            residual = observed - expected
            full_residual[self.snr_mask] = residual
            
            # Store result
            results[sig_level] = DetectionResult(
                chi2_value=chi2_value,
                threshold=threshold,
                detected=detected,
                significance_level=sig_level,
                dof=dof,
                position=subhalo_position,
                snr_mask=self.snr_mask,
                residual=full_residual,
                num_regions=self.num_regions,
                max_region_snr=self.max_region_snr
            )
        
        return results