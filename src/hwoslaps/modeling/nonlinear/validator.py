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


_PSF_MISMATCH_PREFIXES = ("bank:", "delta:", "explicit:")


def _metadata_field(metadata: Any, name: str, default: Any) -> Any:
    """Read one metadata field from dictionary or attribute storage."""
    if isinstance(metadata, dict):
        return metadata.get(name, default)
    return getattr(metadata, name, default)


def _validate_fit_psf_dataset(
    full_config: dict,
    dataset: Any,
    dataset_metadata: Any,
    psf_case: str,
    matched_control: bool,
    expected_psf_fit_sha256: Optional[str],
) -> None:
    """Require coherent mode, labels, supplied state, and kernel identity.

    For mismatch modes the fitted dataset's own PSF kernel is rehashed and
    must equal both the dataset metadata digest and the executor-computed
    digest, the ``psf_case`` argument must equal the metadata label, and
    for delta and explicit modes the label must equal the exact identity
    recomputed from ``full_config``. This is mistake-proofing against
    swapped, stale, or mutated dataset/metadata pairs, not authentication
    against a hostile in-process caller.
    """
    fit_psf = (full_config.get("modeling") or {}).get("fit_psf") or {}
    mode = str(fit_psf.get("mode", "matched")).lower()
    label = str(_metadata_field(dataset_metadata, "psf_fit_label", "fit"))
    supplied = bool(_metadata_field(
        dataset_metadata,
        "psf_fit_supplied",
        False,
    ))
    dataset_psf_fit_sha256 = str(_metadata_field(
        dataset_metadata,
        "psf_fit_sha256",
        "",
    ))
    mismatch_label = label.startswith(_PSF_MISMATCH_PREFIXES)
    mismatch_mode = mode in {"bank", "delta", "explicit"}

    if matched_control or mode == "matched":
        if expected_psf_fit_sha256 is not None:
            raise ValueError(
                "expected_psf_fit_sha256 must be None for matched mode or "
                "matched_control=True"
            )
    elif mismatch_mode:
        if expected_psf_fit_sha256 is None:
            raise ValueError(
                "mismatch-mode datasets must be executed through "
                "run_psf_mismatch_case"
            )
        if expected_psf_fit_sha256 != dataset_psf_fit_sha256:
            raise ValueError(
                "expected_psf_fit_sha256 "
                f"{expected_psf_fit_sha256!r} does not match dataset "
                f"psf_fit_sha256 {dataset_psf_fit_sha256!r}"
            )

    if matched_control:
        if supplied or mismatch_label:
            raise ValueError(
                "matched_control=True requires a truth-PSF dataset without "
                f"a mismatch label; configured mode is '{mode}', dataset "
                f"label is {label!r}, psf_fit_supplied is {supplied}"
            )
        return

    expected_prefix = {
        "bank": "bank:",
        "delta": "delta:",
        "explicit": "explicit:",
    }.get(mode)
    valid = (
        (mode == "matched" and not supplied and not mismatch_label)
        or (
            expected_prefix is not None
            and supplied
            and label.startswith(expected_prefix)
        )
    )
    if valid:
        if mismatch_mode:
            _validate_fit_psf_dataset_identity(
                full_config,
                dataset,
                mode,
                label,
                psf_case,
                expected_psf_fit_sha256,
            )
        return
    if not supplied and mode != "matched":
        detail = "dataset was built with the truth PSF"
    elif supplied and mode == "matched":
        detail = "dataset was built with a supplied fit PSF"
    else:
        detail = "dataset PSF provenance is incoherent"
    raise ValueError(
        f"{detail} but modeling.fit_psf.mode is '{mode}'; dataset label is "
        f"{label!r}, psf_fit_supplied is {supplied}; use "
        "run_psf_mismatch_case or pass matched_control=True"
    )


def _validate_fit_psf_dataset_identity(
    full_config: dict,
    dataset: Any,
    mode: str,
    label: str,
    psf_case: str,
    expected_psf_fit_sha256: str,
) -> None:
    """Bind the fitted dataset kernel and labels to the executor identity."""
    from ...psf.mismatch import _kernel_sha256
    from .autolens_runner import _native_array

    dataset_psf = getattr(dataset, "psf", None)
    if dataset_psf is None:
        raise ValueError(
            "mismatch-mode datasets must expose the fitted PSF kernel as "
            "dataset.psf so the guard can rehash it"
        )
    actual_psf_sha256 = _kernel_sha256(_native_array(dataset_psf))
    if actual_psf_sha256 != expected_psf_fit_sha256:
        raise ValueError(
            f"dataset PSF sha256 {actual_psf_sha256!r} does not match the "
            f"executor kernel digest {expected_psf_fit_sha256!r}; the "
            "fitted dataset does not contain the kernel the executor "
            "supplied"
        )
    if str(psf_case) != label:
        raise ValueError(
            f"psf_case {psf_case!r} does not match the dataset label "
            f"{label!r}; the recorded PSF case must be the label that "
            "passed the guard"
        )
    if mode in {"delta", "explicit"}:
        from ...psf.mismatch import build_psf_mismatch_spec

        spec = build_psf_mismatch_spec(full_config)
        expected_label = f"{spec.mode}:{spec.delta_id}"
        if label != expected_label:
            raise ValueError(
                f"dataset label {label!r} does not match the identity "
                f"{expected_label!r} recomputed from modeling.fit_psf"
            )


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
        *,
        matched_control: bool = False,
        expected_psf_fit_sha256: Optional[str] = None,
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
        matched_control : `bool`, optional
            Whether this is a truth-kernel reference under a mismatch config.
        expected_psf_fit_sha256 : `str`, optional
            Executor-computed digest of the fit kernel supplied to the
            dataset.

        Returns
        -------
        result : `NonlinearCaseResult`
            Validation metric, fit summaries, and diagnostics.

        Notes
        -----
        ``local_search`` and ``freed`` use ``n_live_subhalo_search``;
        ``fixed_template`` retains ``n_live_subhalo_fixed``.
        """
        _validate_fit_psf_dataset(
            full_config,
            dataset,
            dataset_metadata,
            psf_case,
            matched_control,
            expected_psf_fit_sha256,
        )
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
