"""
Utilities for presenting and validating Chernoff-based detection outputs.
"""

from typing import Dict

import numpy as np
from scipy.stats import chi2 as chi2_dist, norm

from .chernoff_detector import ChernoffDetectionData


def validate_chernoff_results(detection_data: ChernoffDetectionData) -> Dict[str, bool]:
    """Basic validations for Chernoff detection outputs.

    Returns a dictionary of boolean flags for quick health checks.
    """
    result = detection_data.result
    validation: Dict[str, bool] = {}

    validation["pixels_unmasked_positive"] = detection_data.result.pixels_unmasked > 0
    validation["delta_chi2_nonnegative"] = result.delta_chi2 >= 0
    validation["p_in_unit_interval"] = (0.0 <= result.p_value <= 1.0)
    validation["sigma_finite"] = np.isfinite(result.sigma)

    # Mask consistency
    validation["mask_matches_count"] = (
        int(np.sum(result.snr_mask)) == detection_data.result.pixels_unmasked
    )

    return validation


def print_chernoff_summary(detection_data: ChernoffDetectionData) -> None:
    """Chernoff detection summary aligned with the GOF printout style."""
    r = detection_data.result

    # Header
    print("\n=== Subhalo Detection Summary (Chernoff) ===")

    # Primary results
    print("\nPrimary results:")
    print(
        f"Chernoff LRT: Δχ²={r.delta_chi2:.3f}, p={r.p_value:.3e}, σ={r.sigma:.2f}"
    )

    # Key statistics (mirror structure of GOF summary where applicable)
    frac = r.pixels_unmasked / detection_data.snr_array.size if detection_data.snr_array.size > 0 else 0.0
    print(f"\nAnalysis parameters:")
    print(f"  SNR threshold: {detection_data.snr_threshold}")
    print(f"  Pixels analyzed: {r.pixels_unmasked}")
    print(f"  Analysis fraction: {frac:.3f}")
    print(f"  Degrees of freedom (Chernoff test): 1 (½·χ¹²)")
    print(f"  Global p-value (Chernoff): {r.p_value:.2e}")

    # Standard sigma checks (use Chernoff thresholds): Δχ² >= χ¹²_isf(2p)
    standard_ps = [
        (1.349898e-3, "3σ"),
        (3.167124e-5, "4σ"),
        (2.866516e-7, "5σ"),
    ]
    delta = r.delta_chi2
    for p, label in standard_ps:
        sigma = norm.isf(p)
        delta_thresh = chi2_dist(df=1).isf(2.0 * p)
        detected = delta >= delta_thresh
        status = "YES" if detected else "NO"
        print(
            f"  {label} (p={p:.2e}, σ={sigma:.2f}): {status} "
            f"(Δχ²={delta:.2f} vs Δχ²_threshold={delta_thresh:.2f})"
        )

    # Extras
    print(
        f"\nPosition (y, x arcsec): ({r.position[0]:.6f}, {r.position[1]:.6f})"
    )
    if r.asimov_delta_chi2 is not None:
        print(f"Asimov Δχ² (expected): {r.asimov_delta_chi2:.2f}")



