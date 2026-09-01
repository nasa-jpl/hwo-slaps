"""Likelihood-ratio metric conventions for nonlinear validation.

This module centralizes the smooth-versus-subhalo likelihood-ratio
convention used to validate Fisher forecasts against nonlinear fits.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from math import isfinite, sqrt
from typing import Dict

import numpy as np

SCDD_DELTA_LOG_L_THRESHOLD = 5.0
"""SCDD local-detection threshold in maximum log likelihood."""

SCDD_Q_THRESHOLD = 2.0*SCDD_DELTA_LOG_L_THRESHOLD
"""Equivalent likelihood-ratio threshold, ``q = 2 Delta log L``."""

STRONG_EVIDENCE_DELTA_LOG_Z_THRESHOLD = 5.0
"""Strong-evidence threshold on marginalized ``Delta log Z`` (`float`)."""


@dataclass(frozen=True)
class LikelihoodRatioMetric:
    """Profile-likelihood summary for one nonlinear validation case.

    Parameters
    ----------
    log_l_smooth : `float`
        Maximum log likelihood under the smooth lens model.
    log_l_subhalo : `float`
        Maximum log likelihood under the subhalo lens model.
    signed_delta_log_l : `float`
        Signed value ``log_l_subhalo - log_l_smooth``.
    delta_log_l : `float`
        Non-negative log-likelihood difference used for the clipped
        likelihood-ratio statistic.
    q : `float`
        Likelihood-ratio statistic, ``q = 2*delta_log_l``.
    z_local : `float`
        Local Gaussian-equivalent significance, ``sqrt(q)``.
    detected_scdd_local : `bool`
        Whether the signed log-likelihood difference exceeds the SCDD
        threshold.
    threshold_delta_log_l : `float`, optional
        Detection threshold in signed maximum log-likelihood difference.
    threshold_q : `float`, optional
        Equivalent detection threshold in ``q``.
    clip_negative_q : `bool`, optional
        Whether negative signed differences are clipped to zero for ``q``.
    convention : `str`, optional
        Human-readable metric convention.
    """

    log_l_smooth: float
    log_l_subhalo: float
    signed_delta_log_l: float
    delta_log_l: float
    q: float
    z_local: float
    detected_scdd_local: bool
    threshold_delta_log_l: float = SCDD_DELTA_LOG_L_THRESHOLD
    threshold_q: float = SCDD_Q_THRESHOLD
    clip_negative_q: bool = True
    convention: str = "q = 2 * Delta log L"

    def to_dict(self) -> Dict[str, object]:
        """Convert the metric to a JSON-compatible dictionary.

        Returns
        -------
        data : `dict`
            Dictionary representation of the metric.
        """
        return asdict(self)


def _require_finite(value: float, name: str) -> float:
    """Validate that a scalar is finite.

    Parameters
    ----------
    value : `float`
        Input scalar.
    name : `str`
        Name used in error messages.

    Returns
    -------
    value : `float`
        Validated value.

    Raises
    ------
    ValueError
        Raised when the input is not finite.
    """
    value_float = float(value)
    if not isfinite(value_float):
        raise ValueError(f"{name} must be finite")
    return value_float


def q_from_delta_log_l(delta_log_l: float, clip_negative: bool = True) -> float:
    """Convert a log-likelihood difference to a likelihood-ratio statistic.

    Parameters
    ----------
    delta_log_l : `float`
        Signed maximum log-likelihood difference.
    clip_negative : `bool`, optional
        If True, negative differences are clipped to zero before converting.

    Returns
    -------
    q : `float`
        Likelihood-ratio statistic.
    """
    delta = _require_finite(delta_log_l, "delta_log_l")
    if clip_negative:
        delta = max(0.0, delta)
    return 2.0*delta


def delta_log_l_from_q(q_value: float) -> float:
    """Convert a likelihood-ratio statistic to ``Delta log L``.

    Parameters
    ----------
    q_value : `float`
        Likelihood-ratio statistic.

    Returns
    -------
    delta_log_l : `float`
        Equivalent log-likelihood difference.
    """
    q_float = _require_finite(q_value, "q_value")
    return 0.5*q_float


def z_from_q(q_value: float) -> float:
    """Convert a non-negative likelihood-ratio statistic to local ``Z``.

    Parameters
    ----------
    q_value : `float`
        Likelihood-ratio statistic.

    Returns
    -------
    z_local : `float`
        Local significance. Negative ``q`` values return NaN.
    """
    q_float = _require_finite(q_value, "q_value")
    if q_float < 0.0:
        return float("nan")
    return sqrt(q_float)


def profile_likelihood_ratio(
    log_l_smooth: float,
    log_l_subhalo: float,
    threshold_delta_log_l: float = SCDD_DELTA_LOG_L_THRESHOLD,
    clip_negative_q: bool = True,
) -> LikelihoodRatioMetric:
    """Compute the nonlinear smooth-versus-subhalo validation statistic.

    Parameters
    ----------
    log_l_smooth : `float`
        Maximum log likelihood from the smooth-model fit.
    log_l_subhalo : `float`
        Maximum log likelihood from the subhalo-model fit.
    threshold_delta_log_l : `float`, optional
        SCDD local detection threshold in signed ``Delta log L``.
    clip_negative_q : `bool`, optional
        Whether to clip negative signed differences when computing ``q``.

    Returns
    -------
    metric : `LikelihoodRatioMetric`
        Complete likelihood-ratio summary.
    """
    log_l_smooth = _require_finite(log_l_smooth, "log_l_smooth")
    log_l_subhalo = _require_finite(log_l_subhalo, "log_l_subhalo")
    threshold_delta_log_l = _require_finite(
        threshold_delta_log_l,
        "threshold_delta_log_l",
    )
    if threshold_delta_log_l < 0.0:
        raise ValueError("threshold_delta_log_l must be non-negative")

    signed_delta_log_l = log_l_subhalo - log_l_smooth
    delta_log_l = max(0.0, signed_delta_log_l) if clip_negative_q else signed_delta_log_l
    q_value = q_from_delta_log_l(delta_log_l, clip_negative=False)
    z_local = z_from_q(q_value) if q_value >= 0.0 else np.nan

    return LikelihoodRatioMetric(
        log_l_smooth=log_l_smooth,
        log_l_subhalo=log_l_subhalo,
        signed_delta_log_l=signed_delta_log_l,
        delta_log_l=delta_log_l,
        q=q_value,
        z_local=z_local,
        detected_scdd_local=signed_delta_log_l >= threshold_delta_log_l,
        threshold_delta_log_l=threshold_delta_log_l,
        threshold_q=2.0*threshold_delta_log_l,
        clip_negative_q=clip_negative_q,
    )
