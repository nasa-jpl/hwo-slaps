"""Serializable result containers for nonlinear validation."""

from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass, field, is_dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from .dataset_builder import NonlinearDatasetMetadata
from .likelihood_metrics import LikelihoodRatioMetric
from .trial import SubhaloTrial

NONLINEAR_CASE_CSV_COLUMNS = (
    "run_name",
    "case_id",
    "fit_mode",
    "psf_case",
    "dataset_kind",
    "mass_msun",
    "y_arcsec",
    "x_arcsec",
    "subhalo_model",
    "profile_class",
    "kappa_s",
    "scale_radius_arcsec",
    "fisher_q",
    "fisher_z",
    "fisher_delta_log_l_equiv",
    "log_l_smooth",
    "log_l_subhalo",
    "signed_delta_log_l_fit",
    "q_fit",
    "z_fit_local",
    "detected_fisher_scdd",
    "detected_fit_scdd",
    "q_fit_over_q_fisher",
    "fit_status_smooth",
    "fit_status_subhalo",
    "n_unmasked_pixels",
    "background_treatment",
    "use_jax_requested",
    "search_engine",
    "n_live_smooth",
    "n_live_subhalo",
    "runtime_s_smooth",
    "runtime_s_subhalo",
    "result_path_smooth",
    "result_path_subhalo",
    "error",
)
"""CSV columns emitted for nonlinear validation cases."""


def _json_safe(value: Any) -> Any:
    """Convert arrays, dataclasses, and non-finite floats to JSON values."""
    if is_dataclass(value):
        return _json_safe(asdict(value))
    if isinstance(value, dict):
        return {str(key): _json_safe(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(val) for val in value]
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


@dataclass
class NonlinearFitSummary:
    """Summary of one PyAutoLens/PyAutoFit nonlinear fit.

    Parameters
    ----------
    model_role : `str`
        Role of the fitted model, for example ``"smooth"`` or
        ``"subhalo"``.
    fit_mode : `str`
        Validation mode.
    status : `str`
        Fit status, such as ``"success"`` or ``"failed"``.
    log_likelihood_max : `float`, optional
        Maximum log likelihood extracted from the result.
    figure_of_merit_max : `float`, optional
        Maximum figure of merit, retained for diagnostics.
    log_evidence : `float`, optional
        Nested-sampling log evidence. This is secondary metadata, not the
        primary detection statistic.
    n_free_parameters : `int`, optional
        Number of free parameters in the PyAutoFit model.
    result_path : `str`, optional
        Output path for the PyAutoFit result.
    runtime_s : `float`, optional
        Wall-clock runtime in seconds.
    error : `str`, optional
        Error message for failed fits.
    warnings : `list` [`str`], optional
        Non-fatal warnings.
    log_likelihood_extraction_method : `str`, optional
        Accessor used to extract ``log_likelihood_max``.
    use_jax_requested : `bool`, optional
        Whether JAX execution was requested.
    search_engine : `str`, optional
        Search backend name.
    n_live : `int`, optional
        Number of live points requested for nested sampling.
    """

    model_role: str
    fit_mode: str
    status: str
    log_likelihood_max: Optional[float] = None
    figure_of_merit_max: Optional[float] = None
    log_evidence: Optional[float] = None
    n_free_parameters: Optional[int] = None
    result_path: Optional[str] = None
    runtime_s: Optional[float] = None
    error: Optional[str] = None
    warnings: List[str] = field(default_factory=list)
    log_likelihood_extraction_method: Optional[str] = None
    use_jax_requested: Optional[bool] = None
    search_engine: Optional[str] = None
    n_live: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert the fit summary to a JSON-compatible dictionary."""
        return _json_safe(asdict(self))


@dataclass
class NonlinearCaseResult:
    """Result for one Fisher-versus-nonlinear validation case.

    Parameters
    ----------
    case_id : `str`
        Stable case identifier.
    trial : `SubhaloTrial`
        Physical subhalo trial.
    dataset_metadata : `NonlinearDatasetMetadata`
        Dataset provenance.
    fit_mode : `str`
        Validation mode.
    psf_case : `str`
        PSF treatment label.
    smooth_fit : `NonlinearFitSummary`
        Smooth-model fit summary.
    subhalo_fit : `NonlinearFitSummary`
        Subhalo-model fit summary.
    metric : `LikelihoodRatioMetric`, optional
        Likelihood-ratio metric. None when either fit failed.
    fisher_q : `float`, optional
        Fisher statistic for the same case.
    fisher_z : `float`, optional
        Fisher local significance.
    fisher_delta_log_l_equiv : `float`, optional
        Fisher-equivalent ``Delta log L``.
    quality_flags : `list` [`str`], optional
        Diagnostic flags.
    """

    case_id: str
    trial: SubhaloTrial
    dataset_metadata: NonlinearDatasetMetadata
    fit_mode: str
    psf_case: str
    smooth_fit: NonlinearFitSummary
    subhalo_fit: NonlinearFitSummary
    metric: Optional[LikelihoodRatioMetric]
    fisher_q: Optional[float] = None
    fisher_z: Optional[float] = None
    fisher_delta_log_l_equiv: Optional[float] = None
    quality_flags: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert the case result to a JSON-compatible dictionary."""
        return _json_safe(asdict(self))

    def to_csv_row(self, run_name: str = "") -> Dict[str, Any]:
        """Convert the case result to one flat CSV row.

        Parameters
        ----------
        run_name : `str`, optional
            Run name to store in the row.

        Returns
        -------
        row : `dict`
            Flat dictionary keyed by ``NONLINEAR_CASE_CSV_COLUMNS``.
        """
        metric = self.metric
        q_fit = metric.q if metric is not None else None
        q_ratio = None
        if q_fit is not None and self.fisher_q not in (None, 0.0):
            q_ratio = q_fit / self.fisher_q
        errors = [
            fit.error
            for fit in (self.smooth_fit, self.subhalo_fit)
            if fit.error
        ]

        row = {
            "run_name": run_name,
            "case_id": self.case_id,
            "fit_mode": self.fit_mode,
            "psf_case": self.psf_case,
            "dataset_kind": self.dataset_metadata.dataset_kind,
            "mass_msun": self.trial.mass_msun,
            "y_arcsec": self.trial.position_yx_arcsec[0],
            "x_arcsec": self.trial.position_yx_arcsec[1],
            "subhalo_model": self.trial.model,
            "profile_class": self.trial.profile_class,
            "kappa_s": self.trial.kappa_s,
            "scale_radius_arcsec": self.trial.scale_radius_arcsec,
            "fisher_q": self.fisher_q,
            "fisher_z": self.fisher_z,
            "fisher_delta_log_l_equiv": self.fisher_delta_log_l_equiv,
            "log_l_smooth": self.smooth_fit.log_likelihood_max,
            "log_l_subhalo": self.subhalo_fit.log_likelihood_max,
            "signed_delta_log_l_fit": (
                metric.signed_delta_log_l if metric is not None else None
            ),
            "q_fit": q_fit,
            "z_fit_local": metric.z_local if metric is not None else None,
            "detected_fisher_scdd": (
                self.fisher_q >= 10.0 if self.fisher_q is not None else None
            ),
            "detected_fit_scdd": (
                metric.detected_scdd_local if metric is not None else None
            ),
            "q_fit_over_q_fisher": q_ratio,
            "fit_status_smooth": self.smooth_fit.status,
            "fit_status_subhalo": self.subhalo_fit.status,
            "n_unmasked_pixels": self.dataset_metadata.n_unmasked_pixels,
            "background_treatment": self.dataset_metadata.background_treatment,
            "use_jax_requested": self.subhalo_fit.use_jax_requested,
            "search_engine": self.subhalo_fit.search_engine,
            "n_live_smooth": self.smooth_fit.n_live,
            "n_live_subhalo": self.subhalo_fit.n_live,
            "runtime_s_smooth": self.smooth_fit.runtime_s,
            "runtime_s_subhalo": self.subhalo_fit.runtime_s,
            "result_path_smooth": self.smooth_fit.result_path,
            "result_path_subhalo": self.subhalo_fit.result_path,
            "error": " | ".join(errors) if errors else None,
        }
        return _json_safe(row)


@dataclass
class NonlinearDetectionData:
    """Top-level nonlinear validation payload.

    Parameters
    ----------
    run_name : `str`
        Run name.
    backend : `str`
        Backend label, normally ``"pyautolens"``.
    cases : `list` [`NonlinearCaseResult`]
        Validation cases.
    thresholds : `dict`
        Detection thresholds and conventions.
    config : `dict`
        Configuration used to build the validation run.
    schema_version : `str`, optional
        Output schema version.
    generation_timestamp : `str`, optional
        ISO timestamp.
    summary : `dict`, optional
        Aggregate validation summary.
    """

    run_name: str
    backend: str
    cases: List[NonlinearCaseResult]
    thresholds: Dict[str, Any]
    config: Dict[str, Any]
    schema_version: str = "nonlinear_detection.v1"
    generation_timestamp: Optional[str] = None
    summary: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Set timestamp and default summary fields."""
        if self.generation_timestamp is None:
            self.generation_timestamp = datetime.now().isoformat()
        if not self.summary:
            n_success = sum(
                case.smooth_fit.status == "success"
                and case.subhalo_fit.status == "success"
                for case in self.cases
            )
            self.summary = {
                "n_cases": len(self.cases),
                "n_success": n_success,
                "n_failed": len(self.cases) - n_success,
            }

    def to_dict(self) -> Dict[str, Any]:
        """Convert the payload to a JSON-compatible dictionary."""
        return _json_safe(asdict(self))

    def write_json(self, path: str) -> None:
        """Write the payload as formatted JSON.

        Parameters
        ----------
        path : `str`
            Output path.
        """
        path_obj = Path(path)
        path_obj.parent.mkdir(parents=True, exist_ok=True)
        with path_obj.open("w", encoding="utf-8") as handle:
            json.dump(self.to_dict(), handle, indent=2, sort_keys=True)

    def write_cases_csv(self, path: str) -> None:
        """Write the case table as CSV.

        Parameters
        ----------
        path : `str`
            Output path.
        """
        path_obj = Path(path)
        path_obj.parent.mkdir(parents=True, exist_ok=True)
        with path_obj.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=NONLINEAR_CASE_CSV_COLUMNS)
            writer.writeheader()
            for case in self.cases:
                writer.writerow(case.to_csv_row(run_name=self.run_name))
