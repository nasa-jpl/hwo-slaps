"""High-level nonlinear validator for Fisher metric calibration."""

from __future__ import annotations

from typing import Any, Optional, TYPE_CHECKING

from .autolens_model_builder import (
    DEFAULT_PRIOR_WIDTHS,
    autofit_model_from_spec,
    fixed_point_model_spec_from_trial,
    smooth_model_spec_from_config,
    subhalo_model_spec_from_trial,
)
from .autolens_runner import AutoLensFitRunner, analysis_key_from
from .likelihood_metrics import profile_likelihood_ratio
from .mass_mapping import MassMappingContext
from .output_schema import (
    NonlinearCaseResult,
    NonlinearFitSummary,
    extract_subhalo_recovery,
)
from .trial import SubhaloTrial

if TYPE_CHECKING:
    from .dataset_builder import NonlinearDatasetMetadata


class NonlinearMetricValidator:
    """Validate Fisher cases with smooth-versus-subhalo nonlinear fits.

    Parameters
    ----------
    runner : `AutoLensFitRunner`
        Runtime fit wrapper and search settings.
    """

    def __init__(self, runner: AutoLensFitRunner):
        self.runner = runner

    def validate_case(
        self,
        dataset: Any,
        dataset_metadata: NonlinearDatasetMetadata,
        full_config: dict,
        trial: SubhaloTrial,
        fit_mode: str = "fixed_template",
        psf_case: str = "nominal",
        priors_config: Optional[dict] = None,
        mass_context: Optional[MassMappingContext] = None,
        clumpy_fit_parameterization: str = "host_free",
        smooth_result: Optional[NonlinearFitSummary] = None,
        analysis_key: Optional[str] = None,
    ) -> NonlinearCaseResult:
        """Run one generalized nonlinear validation case.

        Parameters
        ----------
        dataset : `object`
            PyAutoLens imaging dataset.
        dataset_metadata : `NonlinearDatasetMetadata`
            Dataset provenance.
        full_config : `dict`
            HWO-SLAPS configuration used to build the model.
        trial : `SubhaloTrial`
            Trial subhalo and Fisher values.
        fit_mode : `str`, optional
            ``"fixed_template"``, ``"local_search"``, or ``"freed"``.
        psf_case : `str`, optional
            PSF-treatment label.
        priors_config : `dict`, optional
            Prior-width overrides.
        mass_context : `MassMappingContext`, optional
            Required for freed fits.
        clumpy_fit_parameterization : `str`, optional
            Clumpy-source fit parameterization.
        smooth_result : `NonlinearFitSummary`, optional
            Reusable smooth denominator. If supplied, no smooth search runs.
        analysis_key : `str`, optional
            Precomputed analysis identity.

        Returns
        -------
        result : `NonlinearCaseResult`
            Validation metric, fit summaries, and diagnostics.

        Notes
        -----
        ``local_search`` and ``freed`` use ``n_live_subhalo_search``;
        ``fixed_template`` retains ``n_live_subhalo_fixed``.
        """
        smooth_spec = smooth_model_spec_from_config(
            full_config,
            priors_config=priors_config,
            clumpy_fit_parameterization=clumpy_fit_parameterization,
        )
        subhalo_spec = subhalo_model_spec_from_trial(
            full_config,
            trial=trial,
            priors_config=priors_config,
            fit_mode=fit_mode,
            mass_context=mass_context,
            clumpy_fit_parameterization=clumpy_fit_parameterization,
        )
        resolved_widths = dict(DEFAULT_PRIOR_WIDTHS)
        if priors_config:
            resolved_widths.update(priors_config)
        clumpy_parameterization = (
            clumpy_fit_parameterization
            if full_config["lensing"]["source_galaxy"]["light"]["type"]
            == "Clumpy"
            else None
        )
        smooth_metadata = dict(smooth_spec.metadata)
        smooth_metadata.update(
            {
                "fit_mode": "smooth",
                "clumpy_fit_parameterization": clumpy_parameterization,
                "resolved_prior_widths": resolved_widths,
            }
        )
        smooth_key = analysis_key_from(
            dataset,
            dataset_metadata,
            smooth_metadata,
        )
        model_metadata = dict(smooth_spec.metadata)
        model_metadata.update(subhalo_spec.metadata)
        model_metadata.update(
            {
                "fit_mode": fit_mode,
                "clumpy_fit_parameterization": clumpy_parameterization,
                "resolved_prior_widths": resolved_widths,
            }
        )
        if analysis_key is None:
            analysis_key = analysis_key_from(
                dataset,
                dataset_metadata,
                model_metadata,
            )
        analysis = self.runner.make_analysis(
            dataset,
            model_metadata=model_metadata,
        )

        quality_flags = []
        diagnostics = {}
        if smooth_result is None:
            smooth_fit = self.runner.run_model(
                model=autofit_model_from_spec(smooth_spec),
                analysis=analysis,
                role="smooth",
                fit_mode=fit_mode,
                case_id=trial.case_id,
                n_live=self.runner.settings.n_live_smooth,
                analysis_key=smooth_key,
            )
        else:
            if smooth_result.analysis_key != smooth_key:
                raise ValueError(
                    "smooth_result analysis_key "
                    f"{smooth_result.analysis_key!r} does not match "
                    f"expected smooth analysis key {smooth_key!r}"
                )
            smooth_fit = smooth_result
            quality_flags.append("smooth_reused")
        if (
            smooth_fit.use_jax_requested is not None
            and smooth_fit.use_jax_requested != self.runner.settings.use_jax
        ):
            quality_flags.append("smooth_engine_mismatch")

        recovery_holder = {}

        def recovery_callback(result: Any, model: Any) -> None:
            """Extract freed-subhalo recovery before result disposal."""
            recovery_holder["value"] = extract_subhalo_recovery(
                result,
                mass_context,
            )

        n_live_subhalo = (
            self.runner.settings.n_live_subhalo_fixed
            if fit_mode == "fixed_template"
            else self.runner.settings.n_live_subhalo_search
        )
        subhalo_run_kwargs = {}
        if fit_mode == "freed":
            subhalo_run_kwargs["result_callback"] = recovery_callback
        subhalo_fit = self.runner.run_model(
            model=autofit_model_from_spec(subhalo_spec),
            analysis=analysis,
            role="subhalo",
            fit_mode=fit_mode,
            case_id=trial.case_id,
            n_live=n_live_subhalo,
            analysis_key=analysis_key,
            **subhalo_run_kwargs,
        )
        if (
            fit_mode == "freed"
            and subhalo_fit.status == "success"
            and recovery_holder.get("value") is None
        ):
            quality_flags.append("recovery_extraction_failed")

        if fit_mode == "freed":
            try:
                fixed_spec = fixed_point_model_spec_from_trial(
                    full_config,
                    trial=trial,
                    priors_config=priors_config,
                    clumpy_fit_parameterization=(
                        clumpy_fit_parameterization
                    ),
                )
                fixed_model = autofit_model_from_spec(fixed_spec)
                fixed_instance = fixed_model.instance_from_prior_medians()
                diagnostics["log_l_fixed_template_point"] = float(
                    analysis.log_likelihood_function(fixed_instance)
                )
            except Exception as exc:
                diagnostics["fixed_template_point_error"] = str(exc)
                quality_flags.append("fixed_template_point_failed")

        metric = None
        if (
            smooth_fit.status == "success"
            and subhalo_fit.status == "success"
            and smooth_fit.log_likelihood_max is not None
            and subhalo_fit.log_likelihood_max is not None
        ):
            metric = profile_likelihood_ratio(
                log_l_smooth=smooth_fit.log_likelihood_max,
                log_l_subhalo=subhalo_fit.log_likelihood_max,
            )
        else:
            quality_flags.append("fit_failed")

        fixed_point = diagnostics.get("log_l_fixed_template_point")
        if (
            fit_mode == "freed"
            and fixed_point is not None
            and subhalo_fit.log_likelihood_max is not None
            and subhalo_fit.log_likelihood_max < fixed_point - 0.5
        ):
            quality_flags.append("freed_below_fixed_template")

        return NonlinearCaseResult(
            case_id=trial.case_id,
            trial=trial,
            dataset_metadata=dataset_metadata,
            fit_mode=fit_mode,
            psf_case=psf_case,
            smooth_fit=smooth_fit,
            subhalo_fit=subhalo_fit,
            metric=metric,
            fisher_q=trial.fisher_q,
            fisher_z=trial.fisher_z,
            fisher_delta_log_l_equiv=trial.fisher_delta_log_l_equiv,
            quality_flags=quality_flags,
            subhalo_recovery=recovery_holder.get("value"),
            diagnostics=diagnostics,
        )

    def validate_fixed_template(
        self,
        dataset: Any,
        dataset_metadata: NonlinearDatasetMetadata,
        full_config: dict,
        trial: SubhaloTrial,
        psf_case: str = "nominal",
        priors_config: dict | None = None,
    ) -> NonlinearCaseResult:
        """Run a fixed-template nonlinear validation case.

        Parameters
        ----------
        dataset : `object`
            PyAutoLens imaging dataset.
        dataset_metadata : `NonlinearDatasetMetadata`
            Dataset provenance.
        full_config : `dict`
            HWO-SLAPS configuration used to build the model.
        trial : `SubhaloTrial`
            Fixed subhalo trial.
        psf_case : `str`, optional
            PSF-treatment label.
        priors_config : `dict`, optional
            Prior-width overrides.

        Returns
        -------
        result : `NonlinearCaseResult`
            Validation result.
        """
        return self.validate_case(
            dataset=dataset,
            dataset_metadata=dataset_metadata,
            full_config=full_config,
            trial=trial,
            fit_mode="fixed_template",
            psf_case=psf_case,
            priors_config=priors_config,
        )
