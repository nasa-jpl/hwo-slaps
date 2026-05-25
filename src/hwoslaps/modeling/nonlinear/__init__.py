"""Nonlinear validation helpers for Fisher detectability metrics."""

from .autolens_runner import AutoLensFitRunner, NonlinearSearchSettings
from .calibration import (
    CalibrationPair,
    FisherNonlinearCalibration,
    fit_q_calibration,
    pair_fisher_and_nonlinear,
)
from .likelihood_metrics import (
    SCDD_DELTA_LOG_L_THRESHOLD,
    SCDD_Q_THRESHOLD,
    LikelihoodRatioMetric,
    delta_log_l_from_q,
    profile_likelihood_ratio,
    q_from_delta_log_l,
    z_from_q,
)
from .local_profile import (
    LocalFitAttempt,
    LocalProfileFitResult,
    fit_local_least_squares_profile,
    profile_likelihood_q,
)
from .output_schema import (
    NonlinearCaseResult,
    NonlinearDetectionData,
    NonlinearFitSummary,
)
from .trial import SubhaloTrial, trial_from_lensing_truth
from .validator import NonlinearMetricValidator

__all__ = [
    "AutoLensFitRunner",
    "CalibrationPair",
    "FisherNonlinearCalibration",
    "LikelihoodRatioMetric",
    "LocalFitAttempt",
    "LocalProfileFitResult",
    "NonlinearCaseResult",
    "NonlinearDetectionData",
    "NonlinearFitSummary",
    "NonlinearMetricValidator",
    "NonlinearSearchSettings",
    "SCDD_DELTA_LOG_L_THRESHOLD",
    "SCDD_Q_THRESHOLD",
    "SubhaloTrial",
    "delta_log_l_from_q",
    "fit_q_calibration",
    "fit_local_least_squares_profile",
    "pair_fisher_and_nonlinear",
    "profile_likelihood_ratio",
    "profile_likelihood_q",
    "q_from_delta_log_l",
    "trial_from_lensing_truth",
    "z_from_q",
]
