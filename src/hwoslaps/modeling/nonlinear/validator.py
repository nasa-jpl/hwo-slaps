"""High-level nonlinear validator for Fisher metric calibration."""

from __future__ import annotations

from typing import Any

from .autolens_model_builder import (
    autofit_model_from_spec,
    smooth_model_spec_from_config,
    subhalo_model_spec_from_trial,
)
from .autolens_runner import AutoLensFitRunner
from .dataset_builder import NonlinearDatasetMetadata
from .likelihood_metrics import profile_likelihood_ratio
from .output_schema import NonlinearCaseResult
from .trial import SubhaloTrial


class NonlinearMetricValidator:
    """Validate Fisher cases with smooth-versus-subhalo nonlinear fits."""

    def __init__(self, runner: AutoLensFitRunner):
        self.runner = runner

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
        fit_mode = "fixed_template"
        analysis = self.runner.make_analysis(dataset)
        smooth_spec = smooth_model_spec_from_config(
            full_config,
            priors_config=priors_config,
        )
        subhalo_spec = subhalo_model_spec_from_trial(
            full_config,
            trial=trial,
            priors_config=priors_config,
            fit_mode=fit_mode,
        )

        smooth_fit = self.runner.run_model(
            model=autofit_model_from_spec(smooth_spec),
            analysis=analysis,
            role="smooth",
            fit_mode=fit_mode,
            case_id=trial.case_id,
            n_live=self.runner.settings.n_live_smooth,
        )
        subhalo_fit = self.runner.run_model(
            model=autofit_model_from_spec(subhalo_spec),
            analysis=analysis,
            role="subhalo",
            fit_mode=fit_mode,
            case_id=trial.case_id,
            n_live=self.runner.settings.n_live_subhalo_fixed,
        )

        metric = None
        quality_flags = []
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
        )
