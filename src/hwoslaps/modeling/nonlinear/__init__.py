"""Nonlinear validation helpers for Fisher detectability metrics."""

from __future__ import annotations

from typing import Any

from .autolens_model_builder import (
    autofit_model_from_spec,
    fixed_point_model_spec_from_trial,
    smooth_model_spec_from_config,
    subhalo_model_spec_from_trial,
)
from .autolens_runner import (
    AutoLensFitRunner,
    NonlinearSearchSettings,
    analysis_key_from,
)
from .calibration import (
    CalibrationPair,
    FisherNonlinearCalibration,
    fit_q_calibration,
    pair_fisher_and_nonlinear,
)
from .clumpy_profiles import ClumpyTemplateContext
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
    SubhaloRecovery,
    extract_subhalo_recovery,
)
from .mass_mapping import (
    MassMappingContext,
    build_mass_mapping_context,
    build_mass_mapping_context_explicit,
    evaluate_mass_mapping,
)
from .model_specs import linked
from .trial import (
    SubhaloTrial,
    trial_from_fisher_map_position,
    trial_from_lensing_truth,
)
from .validator import NonlinearMetricValidator

__all__ = [
    "AutoLensFitRunner",
    "CalibrationPair",
    "ClumpyTemplateContext",
    "ClumpyTransformedSource",
    "FisherNonlinearCalibration",
    "LikelihoodRatioMetric",
    "LocalFitAttempt",
    "LocalProfileFitResult",
    "MassMappingContext",
    "NFWMCRSubhaloSph",
    "NonlinearCaseResult",
    "NonlinearDetectionData",
    "NonlinearFitSummary",
    "NonlinearMetricValidator",
    "NonlinearSearchSettings",
    "PointMassMCRSubhalo",
    "PsfBank",
    "PsfBankCandidate",
    "PsfBankCandidateFit",
    "PsfBankCaseResult",
    "PsfBankSummary",
    "SCDD_DELTA_LOG_L_THRESHOLD",
    "SCDD_Q_THRESHOLD",
    "SISMCRSubhalo",
    "SubhaloRecovery",
    "SubhaloTrial",
    "analysis_key_from",
    "autofit_model_from_spec",
    "build_mass_mapping_context",
    "build_mass_mapping_context_explicit",
    "delta_log_l_from_q",
    "evaluate_mass_mapping",
    "extract_subhalo_recovery",
    "fixed_point_model_spec_from_trial",
    "fit_q_calibration",
    "fit_local_least_squares_profile",
    "linked",
    "pair_fisher_and_nonlinear",
    "profile_likelihood_ratio",
    "profile_likelihood_q",
    "q_from_delta_log_l",
    "build_psf_bank",
    "combine_psf_bank_fits",
    "load_psf_bank_npz",
    "run_psf_bank_case",
    "save_psf_bank_npz",
    "smooth_model_spec_from_config",
    "subhalo_model_spec_from_trial",
    "trial_from_fisher_map_position",
    "trial_from_lensing_truth",
    "z_from_q",
]


def __getattr__(name: str) -> Any:
    """Resolve lazy custom profile and PSF-bank exports.

    Parameters
    ----------
    name : `str`
        Requested export name.

    Returns
    -------
    value : `object`
        Requested custom profile or context class.
    """
    if name == "ClumpyTransformedSource":
        from .clumpy_profiles import ClumpyTransformedSource

        return ClumpyTransformedSource
    if name in {
        "NFWMCRSubhaloSph",
        "PointMassMCRSubhalo",
        "SISMCRSubhalo",
    }:
        from . import mass_mapping

        return getattr(mass_mapping, name)
    if name in {
        "PsfBank",
        "PsfBankCandidate",
        "PsfBankCandidateFit",
        "PsfBankCaseResult",
        "PsfBankSummary",
        "build_psf_bank",
        "combine_psf_bank_fits",
        "load_psf_bank_npz",
        "run_psf_bank_case",
        "save_psf_bank_npz",
    }:
        from . import psf_bank

        return getattr(psf_bank, name)
    raise AttributeError(name)


def __dir__() -> list[str]:
    """Return package public names for IDE and star-import compatibility."""
    return sorted(__all__)
