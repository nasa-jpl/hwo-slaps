"""
Mejiro SNR-gated chi-square detectability (paper-exact).

Implements the statistic described in the provided Mejiro/Roman paper excerpt:

- Mask: all pixels with per-pixel SNR > snr_threshold computed from the
  no-subhalo exposure using SNR_i = S_i / sqrt(T_i), where S are source counts
  and T are total counts, in the same units (we use ADU consistently).
- Test statistic (Pearson on counts in ADU):
    chi2 = sum_{i in mask} (A_i - B_i)^2 / max(B_i, eps)
  where A is the exposure with the single subhalo, B is the exposure without.
- Decision rule: compare chi2 to chi2 tail threshold at a target one-sided
  alpha, with dof = N_mask + dof_offset (default dof_offset = -3).

Notes
-----
- Paired-noise assumption (paper): upstream generation should reuse detector
  effects for A and B so that differences isolate the subhalo. This module
  computes the statistic on provided arrays and does not enforce pairing.
"""

from dataclasses import dataclass
from typing import Tuple, Optional

import numpy as np
from scipy import ndimage

from ..observation.utils import ObservationData


@dataclass
class MejiroConfig:
    """Configuration for Mejiro detectability.

    Parameters
    ----------
    alpha : float
        One-sided significance level for detectability checks (~3σ by default).
    dof_offset : int
        Offset applied to degrees of freedom: dof = N_mask + dof_offset.
    snr_threshold : float
        Absolute SNR threshold for pixel masking.
    eps_counts_floor : float
        Floor for denominators in counts (ADU) to maintain numerical stability.
    paired_noise : bool
        Flag documenting whether upstream used paired noise (not enforced here).
    """

    alpha: float = 1.349898e-3
    dof_offset: int = -3
    snr_threshold: float = 1.0
    eps_counts_floor: float = 1e-8
    paired_noise: bool = True


class MejiroDetector:
    """Paper-exact Mejiro detectability implementation.

    Builds an SNR>threshold mask using the H0 (no-subhalo) baseline, then
    evaluates the Pearson chi-square statistic using A (with subhalo) and
    B (without subhalo) exposures in ADU.
    """

    def __init__(self, baseline: ObservationData, config: Optional[MejiroConfig] = None) -> None:
        self.baseline = baseline
        self.config = config or MejiroConfig()

        (
            self.snr_array,
            self.snr_mask,
            self.labeled_regions,
            self.num_regions,
            self.max_region_snr,
        ) = self._build_mask_and_regions()
        self.pixels_unmasked = int(np.sum(self.snr_mask))

        # Cache B (H0 exposure in ADU) for reuse
        self._B_adu_2d = self.baseline.data.native

    def _build_mask_and_regions(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int, float]:
        """Construct SNR>threshold mask from baseline using S/sqrt(T) (ADU domain).

        S = noiseless source counts (ADU) = (noiseless_source_eps * exposure_time) / gain
        T = total counts (ADU) from baseline exposure B = baseline.data.native
        """
        gain = self.baseline.gain
        exposure_time = self.baseline.exposure_time

        # Source counts in ADU
        source_e = self.baseline.noiseless_source_eps * exposure_time
        S_adu_2d = source_e / gain

        # Total baseline exposure in ADU (includes source + sky + dark + noise realization)
        B_adu_2d = self.baseline.data.native

        eps = float(self.config.eps_counts_floor)
        denom = np.sqrt(np.maximum(B_adu_2d, eps))
        snr_2d = S_adu_2d / denom
        snr_2d = np.nan_to_num(snr_2d, nan=0.0, posinf=0.0, neginf=0.0)

        mask_2d = snr_2d > float(self.config.snr_threshold)

        # Label connected regions (cross connectivity) for diagnostics only
        structure = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]])
        labeled_regions_2d, num_regions = ndimage.label(mask_2d, structure=structure)

        # Regional SNRs for summary information
        regional_snrs = []
        for i in range(1, num_regions + 1):
            reg = labeled_regions_2d == i
            s = float(np.sum(S_adu_2d[reg]))
            t = float(np.sum(np.maximum(B_adu_2d[reg], eps)))
            regional_snrs.append(s / np.sqrt(t) if t > 0 else 0.0)
        max_region_snr = max(regional_snrs) if regional_snrs else 0.0

        return (
            snr_2d.flatten(),
            mask_2d.flatten(),
            labeled_regions_2d.flatten(),
            int(num_regions),
            float(max_region_snr),
        )

    def compute_chi2(self, observation_with_subhalo: ObservationData) -> Tuple[float, int, np.ndarray]:
        """Compute Mejiro chi-square over the SNR mask.

        Returns
        -------
        chi2_value : float
            Sum over masked pixels of (A-B)^2 / max(B, eps) in ADU units.
        dof : int
            Degrees of freedom as N_mask + dof_offset (bounded below by 1).
        residual_full : np.ndarray
            Full-size residual map (A-B) in ADU with zeros outside the mask.
        """
        A_adu_2d = observation_with_subhalo.data.native
        B_adu_2d = self._B_adu_2d

        A_full = A_adu_2d.flatten()
        B_full = B_adu_2d.flatten()
        mask = self.snr_mask

        eps = float(self.config.eps_counts_floor)
        denom = np.maximum(B_full[mask], eps)
        diff = A_full[mask] - B_full[mask]
        chi2_value = float(np.sum((diff * diff) / denom))

        dof_raw = int(np.sum(mask)) + int(self.config.dof_offset)
        dof = max(1, dof_raw)

        # Build full residual map (zeros outside mask) for diagnostics/plots
        residual_full = np.zeros_like(B_full)
        residual_full[mask] = diff

        return chi2_value, dof, residual_full


