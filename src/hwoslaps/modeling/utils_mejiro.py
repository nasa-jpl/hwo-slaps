"""
Utilities for presenting and validating Mejiro detectability outputs.
"""

from typing import Dict

import numpy as np
from scipy.stats import norm, chi2 as chi2_dist

from .utils import DetectionData


def validate_mejiro_results(detection_data: DetectionData) -> Dict[str, bool]:
    """Basic validations for Mejiro detection outputs.

    Returns a dictionary of boolean flags for quick health checks.
    """
    validation: Dict[str, bool] = {}

    validation["pixels_unmasked_positive"] = detection_data.pixels_unmasked > 0
    validation["chi2_positive_or_zero"] = detection_data.chi2_value >= 0
    validation["p_in_unit_interval"] = (0.0 <= detection_data.chi2_p_value <= 1.0)
    validation["dof_positive"] = detection_data.degrees_of_freedom > 0
    validation["arrays_same_size"] = (
        len(detection_data.snr_array) == len(detection_data.snr_mask) == len(detection_data.residual_map)
    )

    return validation


def print_mejiro_summary(detection_data: DetectionData) -> None:
    """Mejiro detectability summary aligned with GOF/Chernoff style."""
    # Header
    print("\n=== Subhalo Detectability Summary (Mejiro) ===")

    # Primary results
    dof = detection_data.degrees_of_freedom
    chi2_val = detection_data.chi2_value
    p_global = detection_data.chi2_p_value
    print("\nPrimary results:")
    print(f"Mejiro χ²={chi2_val:.3f}, dof={dof}, global p={p_global:.3e}")

    # Analysis parameters
    frac = detection_data.detection_mask_fraction if detection_data.snr_array.size > 0 else 0.0
    print("\nAnalysis parameters:")
    print(f"  SNR threshold: {detection_data.snr_threshold}")
    print(f"  Pixels analyzed: {detection_data.pixels_unmasked}")
    print(f"  Analysis fraction: {frac:.3f}")

    # Standard sigma checks
    print("\nStandard significance checks:")
    standard_ps = [
        (1.349898e-3, "3σ"),
        (3.167124e-5, "4σ"),
        (2.866516e-7, "5σ"),
    ]
    for p, label in standard_ps:
        sigma = norm.isf(p)
        chi2_thresh = chi2_dist.ppf(1.0 - p, dof)
        detected = chi2_val > chi2_thresh
        status = "YES" if detected else "NO"
        print(
            f"  {label} (p={p:.2e}, σ={sigma:.2f}): {status} "
            f"(χ²={chi2_val:.2f} vs χ²_threshold={chi2_thresh:.2f})"
        )


