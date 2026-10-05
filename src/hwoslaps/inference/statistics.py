"""The nonlinear detection statistic from the two role maxima.

``q_signed = 2 (max log L(H1) - max log L(H0))``. H0 and H1 are not nested in general (a fixed
template, or a freed subhalo with a positive minimum mass), so ``q_signed`` can be negative
and carries no nested-model significance claim; ``q_clipped`` is its positive part.
"""

from __future__ import annotations

import math

__all__ = ["likelihood_ratio", "z_from_q"]


def likelihood_ratio(log_l_smooth: float, log_l_subhalo: float) -> tuple[float, float]:
    """``(q_signed, q_clipped)`` from the H0 and H1 maximum log-likelihoods."""
    smooth, subhalo = float(log_l_smooth), float(log_l_subhalo)
    if not (math.isfinite(smooth) and math.isfinite(subhalo)):
        raise ValueError(f"role log-likelihoods must be finite, got H0 {smooth!r} and H1 {subhalo!r}")
    q_signed = 2.0 * (subhalo - smooth)
    return q_signed, max(0.0, q_signed)


def z_from_q(q: float) -> float:
    """``sqrt(q)`` for ``q >= 0``; NaN for a negative ``q``, which has no Gaussian equivalent."""
    value = float(q)
    if not math.isfinite(value):
        raise ValueError(f"q must be finite, got {value!r}")
    return math.sqrt(value) if value >= 0.0 else math.nan
