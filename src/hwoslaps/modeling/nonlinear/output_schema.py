"""Serializable result containers for nonlinear validation."""

from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass, field, is_dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, TYPE_CHECKING

import numpy as np

from .likelihood_metrics import LikelihoodRatioMetric
from .trial import SubhaloTrial

if TYPE_CHECKING:
    from .dataset_builder import NonlinearDatasetMetadata

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
    "analysis_key",
    "recovered_log10_m200_ml",
    "recovered_log10_m200_p16",
    "recovered_log10_m200_p50",
    "recovered_log10_m200_p84",
    "recovered_y_ml",
    "recovered_x_ml",
    "recovered_concentration_ml",
    "mass_at_lower_bound",
    "mass_at_upper_bound",
    "pdf_converged",
    "smooth_reused",
    "freed_below_fixed_template",
    "n_like_max_reached",
    "use_jax_effective",
    "jax_n_batch_effective",
    "smooth_engine_mismatch",
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
    use_jax_effective : `bool`, optional
        Whether the constructed analysis and search used JAX effectively.
    jax_n_batch_effective : `int`, optional
        Effective vectorized likelihood batch size from the search object.
    search_engine : `str`, optional
        Search backend name.
    n_live : `int`, optional
        Number of live points requested for nested sampling.
    analysis_key : `str`, optional
        Dataset-and-model identity embedded in the search name.
    n_like_max_reached : `bool`, optional
        Whether the configured likelihood-call limit was reached.
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
    analysis_key: Optional[str] = None
    n_like_max_reached: Optional[bool] = None
    use_jax_effective: Optional[bool] = None
    jax_n_batch_effective: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert the fit summary to a JSON-compatible dictionary."""
        return _json_safe(asdict(self))


@dataclass
class SubhaloRecovery:
    """Recovered freed-subhalo parameters and posterior diagnostics.

    Parameters
    ----------
    log10_m200_ml : `float`
        Maximum-likelihood log10 M200.
    centre_ml_y : `float`
        Maximum-likelihood centre y coordinate.
    centre_ml_x : `float`
        Maximum-likelihood centre x coordinate.
    concentration_ml : `float`, optional
        Derived maximum-likelihood NFW concentration.
    kappa_s_ml : `float`, optional
        Derived maximum-likelihood NFW scale convergence.
    scale_radius_arcsec_ml : `float`, optional
        Derived maximum-likelihood NFW scale radius.
    log10_m200_p16, log10_m200_p50, log10_m200_p84 : `float`, optional
        Posterior log-mass quantiles.
    centre_y_p16, centre_y_p50, centre_y_p84 : `float`, optional
        Posterior centre-y quantiles.
    centre_x_p16, centre_x_p50, centre_x_p84 : `float`, optional
        Posterior centre-x quantiles.
    mass_at_lower_bound : `bool`, optional
        Whether the ML mass is within 0.01 dex of the lower bound.
    mass_at_upper_bound : `bool`, optional
        Whether the ML mass is within 0.01 dex of the upper bound.
    posterior_mass_frac_lower : `float`, optional
        Posterior fraction within 0.05 dex of the lower bound.
    posterior_mass_frac_upper : `float`, optional
        Posterior fraction within 0.05 dex of the upper bound.
    pdf_converged : `bool`, optional
        Backend PDF convergence flag.
    extraction_method : `str`, optional
        Accessor path used for extraction.
    n_samples : `int`, optional
        Number of posterior samples used.
    """

    log10_m200_ml: float
    centre_ml_y: float
    centre_ml_x: float
    concentration_ml: Optional[float] = None
    kappa_s_ml: Optional[float] = None
    scale_radius_arcsec_ml: Optional[float] = None
    log10_m200_p16: Optional[float] = None
    log10_m200_p50: Optional[float] = None
    log10_m200_p84: Optional[float] = None
    centre_y_p16: Optional[float] = None
    centre_y_p50: Optional[float] = None
    centre_y_p84: Optional[float] = None
    centre_x_p16: Optional[float] = None
    centre_x_p50: Optional[float] = None
    centre_x_p84: Optional[float] = None
    mass_at_lower_bound: bool = False
    mass_at_upper_bound: bool = False
    posterior_mass_frac_lower: Optional[float] = None
    posterior_mass_frac_upper: Optional[float] = None
    pdf_converged: Optional[bool] = None
    extraction_method: str = "unknown"
    n_samples: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert the recovery payload to a JSON-compatible dictionary.

        Returns
        -------
        data : `dict`
            JSON-safe recovery values.
        """
        return _json_safe(asdict(self))


def _nested_attr(value: Any, path: str) -> Any:
    """Return a dotted attribute path from an object."""
    for name in path.split("."):
        value = getattr(value, name)
    return value


def _maximum_likelihood_subhalo(result: Any) -> tuple[Any, str]:
    """Return the maximum-likelihood subhalo and accessor path."""
    for accessor in (
        "max_log_likelihood_instance",
        "instance",
        "samples.instance",
    ):
        try:
            instance = _nested_attr(result, accessor)
            if callable(instance):
                instance = instance()
            if instance is None:
                continue
            return instance.galaxies.lens.subhalo, accessor
        except Exception:
            continue
    raise AttributeError("Could not extract the maximum-likelihood instance")


def _sample_values(samples: Any, paths: tuple[tuple[str, ...], ...]) -> Any:
    """Return sample values for the first available parameter path."""
    for path in paths:
        try:
            values = samples.values_for_path(path)
            return np.asarray(values, dtype=float)
        except Exception:
            continue
    return None


def _weighted_quantiles(
    values: np.ndarray,
    weights: Optional[np.ndarray],
) -> tuple[float, float, float]:
    """Return posterior 16th, 50th, and 84th percentiles."""
    if weights is None or weights.shape != values.shape:
        return tuple(
            float(value) for value in np.quantile(values, [0.16, 0.5, 0.84])
        )
    order = np.argsort(values)
    sorted_values = values[order]
    sorted_weights = np.asarray(weights[order], dtype=float)
    if (
        sorted_weights.size < 2
        or np.all(sorted_weights == sorted_weights[0])
    ):
        return tuple(
            float(value) for value in np.quantile(values, [0.16, 0.5, 0.84])
        )
    cumulative = np.cumsum(sorted_weights)[:-1]
    normalization = float(cumulative[-1])
    if not np.isfinite(normalization) or normalization <= 0.0:
        return tuple(
            float(value) for value in np.quantile(values, [0.16, 0.5, 0.84])
        )
    cumulative = cumulative / normalization
    cumulative = np.append(0.0, cumulative)
    return tuple(
        float(np.interp(quantile, cumulative, sorted_values))
        for quantile in (0.16, 0.5, 0.84)
    )


def _posterior_fraction(
    selected: np.ndarray,
    weights: Optional[np.ndarray],
) -> float:
    """Return an unweighted or posterior-weighted selected fraction."""
    if weights is None or weights.shape != selected.shape:
        return float(np.mean(selected))
    total = float(np.sum(weights))
    if not np.isfinite(total) or total <= 0.0:
        return float(np.mean(selected))
    return float(np.sum(weights[selected]) / total)


def extract_subhalo_recovery(
    result: Any,
    mapping_context: Any,
) -> SubhaloRecovery:
    """Extract freed-subhalo ML values and converged PDF summaries.

    Parameters
    ----------
    result : `object`
        PyAutoFit result carrying an instance and samples.
    mapping_context : `MassMappingContext`
        Mass conversion context used by the fitted profile.

    Returns
    -------
    recovery : `SubhaloRecovery`
        Recovered parameters, boundary flags, and convergence metadata.
    """
    from .mass_mapping import evaluate_mass_mapping

    subhalo, instance_accessor = _maximum_likelihood_subhalo(result)
    log_mass = float(subhalo.log10_m200)
    centre_y, centre_x = (float(value) for value in subhalo.centre)
    derived = evaluate_mass_mapping(mapping_context, log_mass)
    samples = getattr(result, "samples", None)
    pdf_converged = None
    mass_values = None
    centre_y_values = None
    centre_x_values = None
    weights = None
    n_samples = None
    extraction_method = instance_accessor
    if samples is not None:
        if hasattr(samples, "pdf_converged"):
            try:
                pdf_converged = bool(samples.pdf_converged)
            except Exception:
                pdf_converged = None
        mass_values = _sample_values(
            samples,
            (("galaxies", "lens", "subhalo", "log10_m200"),),
        )
        centre_y_values = _sample_values(
            samples,
            (
                (
                    "galaxies",
                    "lens",
                    "subhalo",
                    "centre",
                    "centre_0",
                ),
                ("galaxies", "lens", "subhalo", "centre_0"),
            ),
        )
        centre_x_values = _sample_values(
            samples,
            (
                (
                    "galaxies",
                    "lens",
                    "subhalo",
                    "centre",
                    "centre_1",
                ),
                ("galaxies", "lens", "subhalo", "centre_1"),
            ),
        )
        if mass_values is not None:
            n_samples = int(mass_values.size)
            extraction_method += "+samples.values_for_path"
        try:
            weights = np.asarray(samples.weight_list, dtype=float)
        except Exception:
            weights = None

    mass_quantiles = (None, None, None)
    centre_y_quantiles = (None, None, None)
    centre_x_quantiles = (None, None, None)
    if pdf_converged is not False:
        if mass_values is not None and mass_values.size:
            mass_quantiles = _weighted_quantiles(mass_values, weights)
        if centre_y_values is not None and centre_y_values.size:
            centre_y_quantiles = _weighted_quantiles(centre_y_values, weights)
        if centre_x_values is not None and centre_x_values.size:
            centre_x_quantiles = _weighted_quantiles(centre_x_values, weights)

    lower = mapping_context.log10_m200_lower
    upper = mapping_context.log10_m200_upper
    lower_fraction = None
    upper_fraction = None
    if mass_values is not None and mass_values.size:
        lower_fraction = _posterior_fraction(
            mass_values <= lower + 0.05,
            weights,
        )
        upper_fraction = _posterior_fraction(
            mass_values >= upper - 0.05,
            weights,
        )

    return SubhaloRecovery(
        log10_m200_ml=log_mass,
        centre_ml_y=centre_y,
        centre_ml_x=centre_x,
        concentration_ml=derived.get("c200"),
        kappa_s_ml=derived.get("kappa_s"),
        scale_radius_arcsec_ml=derived.get("scale_radius_arcsec"),
        log10_m200_p16=mass_quantiles[0],
        log10_m200_p50=mass_quantiles[1],
        log10_m200_p84=mass_quantiles[2],
        centre_y_p16=centre_y_quantiles[0],
        centre_y_p50=centre_y_quantiles[1],
        centre_y_p84=centre_y_quantiles[2],
        centre_x_p16=centre_x_quantiles[0],
        centre_x_p50=centre_x_quantiles[1],
        centre_x_p84=centre_x_quantiles[2],
        mass_at_lower_bound=abs(log_mass - lower) <= 0.01,
        mass_at_upper_bound=abs(log_mass - upper) <= 0.01,
        posterior_mass_frac_lower=lower_fraction,
        posterior_mass_frac_upper=upper_fraction,
        pdf_converged=pdf_converged,
        extraction_method=extraction_method,
        n_samples=n_samples,
    )


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
    subhalo_recovery : `SubhaloRecovery`, optional
        Freed-subhalo recovery values.
    diagnostics : `dict`, optional
        Additional invariant and likelihood diagnostics.
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
    subhalo_recovery: Optional[SubhaloRecovery] = None
    diagnostics: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert the case result to a JSON-compatible dictionary."""
        return _json_safe(asdict(self))

    def to_csv_row(self, run_name: str = "") -> Dict[str, Any]:
        """Convert the case result to one flat CSV row.

        Freed significance requires the empirical null from brief section 3.

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
        recovery = self.subhalo_recovery

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
            "z_fit_local": (
                metric.z_local
                if metric is not None and self.fit_mode != "freed"
                else None
            ),
            "detected_fisher_scdd": (
                self.fisher_q >= 10.0 if self.fisher_q is not None else None
            ),
            "detected_fit_scdd": (
                metric.detected_scdd_local
                if metric is not None and self.fit_mode != "freed"
                else None
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
            "analysis_key": self.subhalo_fit.analysis_key,
            "recovered_log10_m200_ml": (
                recovery.log10_m200_ml if recovery is not None else None
            ),
            "recovered_log10_m200_p16": (
                recovery.log10_m200_p16 if recovery is not None else None
            ),
            "recovered_log10_m200_p50": (
                recovery.log10_m200_p50 if recovery is not None else None
            ),
            "recovered_log10_m200_p84": (
                recovery.log10_m200_p84 if recovery is not None else None
            ),
            "recovered_y_ml": (
                recovery.centre_ml_y if recovery is not None else None
            ),
            "recovered_x_ml": (
                recovery.centre_ml_x if recovery is not None else None
            ),
            "recovered_concentration_ml": (
                recovery.concentration_ml if recovery is not None else None
            ),
            "mass_at_lower_bound": (
                recovery.mass_at_lower_bound if recovery is not None else None
            ),
            "mass_at_upper_bound": (
                recovery.mass_at_upper_bound if recovery is not None else None
            ),
            "pdf_converged": (
                recovery.pdf_converged if recovery is not None else None
            ),
            "smooth_reused": "smooth_reused" in self.quality_flags,
            "freed_below_fixed_template": (
                "freed_below_fixed_template" in self.quality_flags
            ),
            "n_like_max_reached": self.subhalo_fit.n_like_max_reached,
            "use_jax_effective": self.subhalo_fit.use_jax_effective,
            "jax_n_batch_effective": (
                self.subhalo_fit.jax_n_batch_effective
            ),
            "smooth_engine_mismatch": (
                "smooth_engine_mismatch" in self.quality_flags
            ),
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
    schema_version: str = "nonlinear_detection.v2"
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
