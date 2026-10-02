"""Public nonlinear forecasting interfaces, loaded only when requested.

Importing this package or its settings does not initialize AutoLens, AutoFit,
JAX, plotting, or a worker pool. Heavy backends remain execution-time choices.
"""
from __future__ import annotations

from importlib import import_module
from typing import Any

_EXPORT_MODULES = {
    'validate_nonlinear': 'api',
    'AutoLensFitRunner': 'autolens_runner',
    'CalibrationPair': 'calibration',
    'FisherNonlinearCalibration': 'calibration',
    'FreshProfileSettings': 'profile_settings',
    'LikelihoodRatioMetric': 'likelihood_metrics',
    'MassMappingContext': 'mass_mapping',
    'NFWMCRSubhaloSph': 'mass_mapping',
    'NonlinearCaseResult': 'output_schema',
    'NonlinearDetectionData': 'output_schema',
    'NonlinearFitSummary': 'output_schema',
    'NonlinearMetricValidator': 'validator',
    'NonlinearSearchSettings': 'autolens_runner',
    'PointMassMCRSubhalo': 'mass_mapping',
    'PsfMismatchCaseResult': 'psf_mismatch',
    'PsfMismatchSpec': 'psf_mismatch',
    'SCDD_DELTA_LOG_L_THRESHOLD': 'likelihood_metrics',
    'SCDD_Q_THRESHOLD': 'likelihood_metrics',
    'SISMCRSubhalo': 'mass_mapping',
    'SubhaloRecovery': 'output_schema',
    'SubhaloTrial': 'trial',
    'analysis_key_from': 'autolens_runner',
    'autofit_model_from_spec': 'autolens_model_builder',
    'build_mass_mapping_context': 'mass_mapping',
    'build_mass_mapping_context_explicit': 'mass_mapping',
    'build_psf_mismatch_spec': 'psf_mismatch',
    'delta_log_l_from_q': 'likelihood_metrics',
    'evaluate_mass_mapping': 'mass_mapping',
    'extract_subhalo_recovery': 'output_schema',
    'fit_q_calibration': 'calibration',
    'fixed_point_model_spec_from_trial': 'autolens_model_builder',
    'linked': 'model_specs',
    'pair_fisher_and_nonlinear': 'calibration',
    'profile_likelihood_ratio': 'likelihood_metrics',
    'q_from_delta_log_l': 'likelihood_metrics',
    'run_psf_mismatch_case': 'psf_mismatch',
    'smooth_model_spec_from_config': 'autolens_model_builder',
    'subhalo_model_spec_from_trial': 'autolens_model_builder',
    'trial_from_fisher_map_position': 'trial',
    'trial_from_lensing_truth': 'trial',
    'z_from_q': 'likelihood_metrics',
}

__all__ = list(_EXPORT_MODULES)


def __getattr__(name: str) -> Any:
    """Load the module implementing one explicitly declared public export."""
    module_name = _EXPORT_MODULES.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    return getattr(import_module(f".{module_name}", __name__), name)


def __dir__() -> list[str]:
    """Expose declared exports to IDEs and interactive completion."""
    return sorted(__all__)
