"""
Chernoff minimal-fit chi-square subhalo detection.

This module implements a fixed-position likelihood-ratio test (LRT) for
subhalo detectability using the Chernoff 1/2·chi^2_1 reference. It profiles
one nonnegative amplitude (mass or linearized amplitude) to form the
improvement statistic Δχ², and maps it to a p-value and sigma without
requiring null simulations. The implementation mirrors the existing
chi-square pipeline conventions: arrays are handled in ADU, the SNR mask is
defined from the H0 (no-subhalo) baseline, and per-pixel variances are taken
from the baseline noise map.
"""

from dataclasses import dataclass
from typing import Dict, Tuple, List, Optional

import numpy as np
from scipy import ndimage
from scipy.stats import chi2 as chi2_dist, norm

from ..observation.utils import ObservationData
from ..lensing.utils import LensingData


@dataclass
class ChernoffDetectionResult:
    """Primary results from Chernoff minimal-fit detection at a fixed position.

    Attributes
    ----------
    delta_chi2 : float
        Likelihood-ratio improvement Δχ² = χ²(H0) − min_{α>=0} χ²(H1(α)).
    p_value : float
        One-sided p-value from Chernoff mixture reference: p = 0.5 * sf(χ¹², Δχ²).
    sigma : float
        Equivalent Gaussian sigma: σ = Φ^{-1}(1 − p_value).
    position : Tuple[float, float]
        Subhalo position (y, x) in arcseconds, recorded for provenance.
    alpha_hat : float
        Best-fit nonnegative amplitude (in the same units used to form the
        template; for a per-mass template this approximates best mass scaling).
    used_template : bool
        True if the linear template route was used; False if a 1-D profile over
        explicit masses was performed.
    snr_mask : np.ndarray
        Boolean 1D mask of analyzed pixels.
    pixels_unmasked : int
        Number of unmasked pixels (mask cardinality) used in the statistic.
    asimov_delta_chi2 : Optional[float]
        Δχ² computed on the noiseless H1 reference (Asimov) for expected
        significance forecasting. None if not computed.
    """

    delta_chi2: float
    p_value: float
    sigma: float
    position: Tuple[float, float]
    alpha_hat: float
    used_template: bool
    snr_mask: np.ndarray
    pixels_unmasked: int
    asimov_delta_chi2: Optional[float] = None


@dataclass
class ChernoffDetectionData:
    """Complete Chernoff-based detection outputs and diagnostics.

    This mirrors the structure provided for the classic chi-square detector,
    while focusing on the Chernoff minimal-fit metrics.
    """

    result: ChernoffDetectionResult

    # Detection parameters
    snr_threshold: float
    max_region_snr: float
    num_regions: int

    # Arrays and diagnostics
    snr_array: np.ndarray
    labeled_regions: np.ndarray
    variance_2d: Optional[np.ndarray] = None

    # Provenance
    config: Optional[Dict] = None


class ChernoffSubhaloDetector:
    """Fixed-position Chernoff minimal-fit detector.

    This detector constructs a mask from the H0 (no-subhalo) baseline
    observation, then evaluates the Chernoff likelihood-ratio test at a fixed
    position using either a fast linear template or a 1-D amplitude profile.

    Parameters
    ----------
    observation_data_no_subhalo : ObservationData
        Baseline observation data (no subhalo). Provides the noise map, gain,
        exposure, and H0 noiseless source needed for mask/variance.
    observation_data_with_subhalo_ref : ObservationData
        Reference observation for H1 (with subhalo) used to derive a linear
        template T from E1_ref − E0. This should be noiseless (its
        ``noiseless_source_eps`` is used) or at least provide the noiseless
        source image via that property.
    lensing_test : LensingData
        Lensing system metadata for the reference H1, used to record position
        and to infer the reference mass for the template scaling.
    snr_threshold : float
        Absolute SNR threshold for pixel selection when forming the mask.
    use_template : bool
        If True, use the fast linear matched-filter route. If False, expect the
        caller to provide an explicit amplitude generator externally (not used
        in this module) and call the profiling path; by default this detector
        uses the template path.
    """

    def __init__(
        self,
        observation_data_no_subhalo: ObservationData,
        observation_data_with_subhalo_ref: ObservationData,
        lensing_test: LensingData,
        snr_threshold: float = 1.0,
        use_template: bool = True,
    ) -> None:
        self.baseline = observation_data_no_subhalo
        self.ref_h1 = observation_data_with_subhalo_ref
        self.lensing_test = lensing_test
        self.snr_threshold = snr_threshold
        self.use_template = use_template

        # Build SNR-based mask and region diagnostics using H0 baseline
        (
            self.snr_array,
            self.labeled_regions,
            self.snr_mask,
            self.num_regions,
            self.max_region_snr,
        ) = self._calculate_snr_regions()
        self.pixels_unmasked = int(np.sum(self.snr_mask))

        # Cache variance (ADU^2) for diagnostics
        self.variance_2d = (self.baseline.noise_map.native) ** 2

        # Precompute E0 (ADU) and the reference template T (ADU per unit mass)
        self._E0_adu, self._T_adu, self._m_ref = self._build_E0_and_template()

    def _calculate_snr_regions(
        self,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int, float]:
        """Create SNR mask from H0 baseline and compute connected regions.

        SNR is computed as source (ADU) divided by the noise map (ADU).
        """
        gain = self.baseline.gain
        noise_map_adu_2d = self.baseline.noise_map.native

        # Source ADU for H0 (no subhalo)
        exposure_time = self.baseline.exposure_time
        source_counts_2d_e = self.baseline.noiseless_source_eps * exposure_time
        source_adu_2d = source_counts_2d_e / gain

        eps = 1e-12
        snr_array_2d = source_adu_2d / np.maximum(noise_map_adu_2d, eps)
        snr_array_2d = np.nan_to_num(snr_array_2d, nan=0.0, posinf=0.0, neginf=0.0)

        snr_mask_2d = snr_array_2d > self.snr_threshold

        # Cross connectivity (no diagonals)
        structure = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]])
        labeled_regions_2d, num_regions = ndimage.label(snr_mask_2d, structure=structure)

        # Regional SNRs (sum S / sqrt(sum Var)) for info only
        regional_snrs: List[float] = []
        for i in range(1, num_regions + 1):
            reg = labeled_regions_2d == i
            s = float(np.sum(source_adu_2d[reg]))
            v = float(np.sum((noise_map_adu_2d[reg]) ** 2))
            regional_snrs.append(s / np.sqrt(v) if v > 0 else 0.0)

        max_region_snr = max(regional_snrs) if regional_snrs else 0.0

        return (
            snr_array_2d.flatten(),
            labeled_regions_2d.flatten(),
            snr_mask_2d.flatten(),
            num_regions,
            max_region_snr,
        )

    def _build_E0_and_template(self) -> Tuple[np.ndarray, np.ndarray, float]:
        """Construct H0 expectation E0 (ADU) and linear template T (ADU per mass).

        The template uses a small reference mass m_ref drawn from the provided
        ``lensing_test.subhalo_mass`` and the reference H1 noiseless source.
        """
        gain = self.baseline.gain
        exposure_time = self.baseline.exposure_time

        # Background means in electrons per pixel
        sky_e = self.baseline.sky_background * exposure_time
        dark_e = self.baseline.dark_current * exposure_time

        # E0 in ADU (H0 expectation)
        E0_e = self.baseline.noiseless_source_eps * exposure_time
        E0_adu_2d = (E0_e + sky_e + dark_e) / gain

        # Reference H1 noiseless in ADU
        E1_ref_e = self.ref_h1.noiseless_source_eps * exposure_time
        E1_ref_adu_2d = (E1_ref_e + sky_e + dark_e) / gain

        # Reference mass (solar masses) from LensingData
        m_ref = float(self.lensing_test.subhalo_mass) if self.lensing_test.subhalo_mass is not None else 1.0
        if m_ref == 0.0:
            m_ref = 1.0

        # Linear template in ADU per unit mass
        T_adu_2d = (E1_ref_adu_2d - E0_adu_2d) / m_ref

        return E0_adu_2d.flatten(), T_adu_2d.flatten(), m_ref

    def detect_at_position(
        self,
        observation_with_subhalo: ObservationData,
        subhalo_position: Tuple[float, float],
        compute_asimov: bool = True,
    ) -> ChernoffDetectionData:
        """Run the Chernoff minimal-fit test at the given fixed position.

        Parameters
        ----------
        observation_with_subhalo : ObservationData
            Observation that may contain a subhalo (noisy data array in ADU is used).
        subhalo_position : tuple of float
            Injection/assumed position (y, x) in arcseconds.
        compute_asimov : bool
            If True, compute the Asimov Δχ² using the reference H1 expectation.
        """
        # Flatten observed ADU and variance
        O_full = observation_with_subhalo.data.native.flatten()
        Var_full = (self.baseline.noise_map.native ** 2).flatten()

        # Apply mask
        mask = self.snr_mask
        O = O_full[mask]
        E0 = self._E0_adu[mask]
        T = self._T_adu[mask]
        V = Var_full[mask] + 1e-10

        # Matched-filter projection with nonnegative amplitude constraint
        w = 1.0 / V
        r = O - E0
        N = float(np.sum(T * r * w))
        D = float(np.sum(T * T * w))

        if D <= 0.0:
            # Degenerate template under mask; report null-like result
            delta_chi2 = 0.0
            alpha_hat = 0.0
        else:
            alpha_unc = N / D
            alpha_hat = max(0.0, alpha_unc)
            delta_chi2 = 0.0 if alpha_hat <= 0.0 else (N * N) / D

        # Chernoff mixture: for 1 dof, χ1² = Z² ⇒ sf_χ1²(Δ) = 2 * Φ(-√Δ)
        # Hence p = 0.5 * sf_χ1²(Δ) = Φ(-√Δ) = norm.sf(√Δ) and σ = √Δ (exact).
        sigma = float(np.sqrt(delta_chi2))
        p_delta = float(norm.sf(sigma))

        asimov_delta = None
        if compute_asimov:
            # Asimov Δχ² on E1_ref (noiseless): (E1_ref − E0)^T W (E1_ref − E0)
            E1_ref_full = (
                (self.ref_h1.noiseless_source_eps * self.baseline.exposure_time +
                 self.baseline.sky_background * self.baseline.exposure_time +
                 self.baseline.dark_current * self.baseline.exposure_time) / self.baseline.gain
            ).flatten()
            E1_ref = E1_ref_full[mask]
            d = E1_ref - E0
            asimov_delta = float(np.sum((d * d) * w))

        # === TRIAGE DEBUG SECTION ===
        print("\n=== TRIAGE ===")
        N_mask = int(np.sum(mask))
        image_shape = observation_with_subhalo.data.native.shape
        print(f"N_mask: {N_mask}")
        print(f"image shape: {image_shape}")
        print(f"sigma = np.sqrt(delta_chi2): {sigma}")
        print(f"delta_chi2: {delta_chi2}")
        
        var_masked = Var_full[mask]
        print(f"np.median(var_masked): {np.median(var_masked)}")
        print(f"np.percentile(var_masked,[5,50,95]): {np.percentile(var_masked, [5,50,95])}")
        
        T_squared_over_Var = (T**2) / V
        print(f"np.median(((T**2)/Var)[mask]): {np.median(T_squared_over_Var)}")
        print(f"95th percentile of ((T**2)/Var)[mask]: {np.percentile(T_squared_over_Var, 95)}")
        print(f"sum to get D: {np.sum(T_squared_over_Var)}")
        
        r_T_over_Var = (r * T) / V
        print(f"np.median(((r*T)/Var)[mask]): {np.median(r_T_over_Var)}")
        print(f"sum to get N: {np.sum(r_T_over_Var)}")
        
        if compute_asimov:
            asimov_per_pixel = ((E1_ref - E0)**2) / V
            print(f"np.median(((E1_ref-E0)**2)/Var)[mask]) - rough per-pixel Asimov: {np.median(asimov_per_pixel)}")
            
            # Histogram of c_i = (Δμ_i)^2/Var_i
            c_i = ((E1_ref - E0)**2) / V
            fraction_above_1 = np.mean(c_i > 1)
            print(f"Histogram of c_i = (Δμ_i)^2/Var_i:")
            print(f"  min: {np.min(c_i):.6f}, max: {np.max(c_i):.6f}")
            print(f"  median: {np.median(c_i):.6f}, mean: {np.mean(c_i):.6f}")
            print(f"  fraction of mask pixels with c_i > 1: {fraction_above_1:.6f}")
        print("=== END TRIAGE ===\n")

        result = ChernoffDetectionResult(
            delta_chi2=delta_chi2,
            p_value=float(p_delta),
            sigma=sigma,
            position=subhalo_position,
            alpha_hat=alpha_hat,
            used_template=self.use_template,
            snr_mask=mask,
            pixels_unmasked=int(np.sum(mask)),
            asimov_delta_chi2=asimov_delta,
        )

        return ChernoffDetectionData(
            result=result,
            snr_threshold=self.snr_threshold,
            max_region_snr=self.max_region_snr,
            num_regions=self.num_regions,
            snr_array=self.snr_array,
            labeled_regions=self.labeled_regions,
            variance_2d=self.variance_2d,
            config=None,
        )


