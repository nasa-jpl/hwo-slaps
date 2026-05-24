"""Calibrate Fisher forecasts against nonlinear validation cases."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np


@dataclass(frozen=True)
class CalibrationPair:
    """Matched Fisher/nonlinear metric pair for one validation case."""

    case_id: str
    mass_msun: float
    position_yx_arcsec: Tuple[float, float]
    psf_case: str
    fit_mode: str
    q_fisher: float
    q_fit: float
    delta_log_l_fisher_equiv: float
    delta_log_l_fit: float
    detected_fisher_scdd: bool
    detected_fit_scdd: bool

    def to_dict(self) -> Dict[str, Any]:
        """Convert the pair to a dictionary."""
        return asdict(self)


@dataclass
class FisherNonlinearCalibration:
    """Summary calibration from sparse nonlinear validation cases."""

    schema_version: str
    n_pairs: int
    fit_mode: str
    psf_case: str
    alpha_linear_q: Optional[float]
    beta_linear_q: Optional[float]
    median_q_ratio: Optional[float]
    median_log10_q_ratio: Optional[float]
    rms_log10_q_ratio: Optional[float]
    spearman_rank: Optional[float]
    threshold_confusion: Dict[str, int]
    q_fit_threshold: float
    inferred_q_fisher_threshold: Optional[float]
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert the calibration summary to a dictionary."""
        return asdict(self)


def _threshold_confusion(pairs: Sequence[CalibrationPair]) -> Dict[str, int]:
    """Count Fisher/nonlinear threshold agreement categories."""
    confusion = {
        "fisher_detected_fit_detected": 0,
        "fisher_detected_fit_not_detected": 0,
        "fisher_not_detected_fit_detected": 0,
        "fisher_not_detected_fit_not_detected": 0,
    }
    for pair in pairs:
        if pair.detected_fisher_scdd and pair.detected_fit_scdd:
            confusion["fisher_detected_fit_detected"] += 1
        elif pair.detected_fisher_scdd and not pair.detected_fit_scdd:
            confusion["fisher_detected_fit_not_detected"] += 1
        elif not pair.detected_fisher_scdd and pair.detected_fit_scdd:
            confusion["fisher_not_detected_fit_detected"] += 1
        else:
            confusion["fisher_not_detected_fit_not_detected"] += 1
    return confusion


def _rank_correlation(x_values: np.ndarray, y_values: np.ndarray) -> Optional[float]:
    """Compute a simple Spearman rank correlation without SciPy."""
    if x_values.size < 2:
        return None
    x_rank = np.argsort(np.argsort(x_values)).astype(float)
    y_rank = np.argsort(np.argsort(y_values)).astype(float)
    x_std = float(np.std(x_rank))
    y_std = float(np.std(y_rank))
    if x_std == 0.0 or y_std == 0.0:
        return None
    return float(np.corrcoef(x_rank, y_rank)[0, 1])


def pair_fisher_and_nonlinear(
    fisher_data: Any,
    nonlinear_data: Any = None,
) -> List[CalibrationPair]:
    """Create calibration pairs from nonlinear validation output.

    Parameters
    ----------
    fisher_data : `object`
        Fisher payload, retained for call-site clarity. If
        ``nonlinear_data`` is omitted, this argument is treated as the
        nonlinear payload for lightweight tests and scripts.
    nonlinear_data : `object`, optional
        Object with a ``cases`` sequence of nonlinear case results.

    Returns
    -------
    pairs : `list` [`CalibrationPair`]
        Matched Fisher/nonlinear metric pairs for successful cases.
    """
    if nonlinear_data is None:
        nonlinear_data = fisher_data

    pairs = []
    for case in nonlinear_data.cases:
        if case.metric is None or case.fisher_q is None:
            continue
        pairs.append(
            CalibrationPair(
                case_id=case.case_id,
                mass_msun=float(case.trial.mass_msun),
                position_yx_arcsec=tuple(case.trial.position_yx_arcsec),
                psf_case=case.psf_case,
                fit_mode=case.fit_mode,
                q_fisher=float(case.fisher_q),
                q_fit=float(case.metric.q),
                delta_log_l_fisher_equiv=0.5*float(case.fisher_q),
                delta_log_l_fit=float(case.metric.signed_delta_log_l),
                detected_fisher_scdd=float(case.fisher_q) >= case.metric.threshold_q,
                detected_fit_scdd=bool(case.metric.detected_scdd_local),
            )
        )
    return pairs


def fit_q_calibration(
    pairs: Sequence[CalibrationPair],
    method: str = "linear",
    q_fit_threshold: float = 10.0,
) -> FisherNonlinearCalibration:
    """Fit a lightweight relation between Fisher and nonlinear ``q``.

    Parameters
    ----------
    pairs : sequence [`CalibrationPair`]
        Calibration pairs.
    method : `str`, optional
        Calibration method. The initial supported value is ``"linear"``.
    q_fit_threshold : `float`, optional
        Detection threshold in nonlinear ``q``.

    Returns
    -------
    calibration : `FisherNonlinearCalibration`
        Calibration summary.
    """
    if method != "linear":
        raise ValueError("Only method='linear' is currently supported")

    warnings = []
    if not pairs:
        return FisherNonlinearCalibration(
            schema_version="fisher_nonlinear_calibration.v1",
            n_pairs=0,
            fit_mode="mixed",
            psf_case="mixed",
            alpha_linear_q=None,
            beta_linear_q=None,
            median_q_ratio=None,
            median_log10_q_ratio=None,
            rms_log10_q_ratio=None,
            spearman_rank=None,
            threshold_confusion=_threshold_confusion(pairs),
            q_fit_threshold=q_fit_threshold,
            inferred_q_fisher_threshold=None,
            warnings=["no valid calibration pairs"],
        )

    q_fisher = np.asarray([pair.q_fisher for pair in pairs], dtype=float)
    q_fit = np.asarray([pair.q_fit for pair in pairs], dtype=float)
    positive = (q_fisher > 0.0) & (q_fit > 0.0)
    ratios = q_fit[positive] / q_fisher[positive]
    if ratios.size == 0:
        warnings.append("no positive q pairs available for ratio statistics")
        median_ratio = None
        median_log_ratio = None
        rms_log_ratio = None
    else:
        log_ratios = np.log10(ratios)
        median_ratio = float(np.median(ratios))
        median_log_ratio = float(np.median(log_ratios))
        rms_log_ratio = float(np.sqrt(np.mean((log_ratios - median_log_ratio)**2)))

    alpha = None
    beta = None
    inferred_threshold = None
    if len(pairs) >= 2 and np.std(q_fisher) > 0.0:
        alpha, beta = np.polyfit(q_fisher, q_fit, deg=1)
        alpha = float(alpha)
        beta = float(beta)
        if alpha > 0.0:
            inferred_threshold = float((q_fit_threshold - beta) / alpha)
    else:
        warnings.append("fewer than two distinct Fisher q values; linear fit skipped")

    fit_modes = {pair.fit_mode for pair in pairs}
    psf_cases = {pair.psf_case for pair in pairs}
    return FisherNonlinearCalibration(
        schema_version="fisher_nonlinear_calibration.v1",
        n_pairs=len(pairs),
        fit_mode=next(iter(fit_modes)) if len(fit_modes) == 1 else "mixed",
        psf_case=next(iter(psf_cases)) if len(psf_cases) == 1 else "mixed",
        alpha_linear_q=alpha,
        beta_linear_q=beta,
        median_q_ratio=median_ratio,
        median_log10_q_ratio=median_log_ratio,
        rms_log10_q_ratio=rms_log_ratio,
        spearman_rank=_rank_correlation(q_fisher, q_fit),
        threshold_confusion=_threshold_confusion(pairs),
        q_fit_threshold=q_fit_threshold,
        inferred_q_fisher_threshold=inferred_threshold,
        warnings=warnings,
    )
