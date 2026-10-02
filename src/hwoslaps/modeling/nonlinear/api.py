"""Optional nonlinear validation over a prepared strong-lensing forecast.

Importing this module does not initialize a fitting backend. The caller owns
preparation and explicitly selects an output directory for fit artifacts.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .autolens_runner import NonlinearSearchSettings
    from .output_schema import NonlinearCaseResult
    from .profile_settings import FreshProfileSettings
    from .trial import SubhaloTrial


def validate_nonlinear(
    prepared: Any,
    trial: SubhaloTrial,
    settings: NonlinearSearchSettings | None = None,
    *,
    output_dir: str | Path,
    profile_settings: FreshProfileSettings | None = None,
    observation: Any = None,
    fit_mode: str = "freed",
    dataset_kind: str = "asimov",
    background_treatment: str = "subtract_known",
    priors: dict | None = None,
    mass_context: Any = None,
    mask_bool_use: Any = None,
) -> NonlinearCaseResult:
    """Fit an injected trial, or an explicitly supplied control observation.

    ``output_dir`` is required because AutoFit writes search artifacts. Supplying
    ``profile_settings`` enables local refinement of the current sampler result;
    independent-start support and scalar/residual agreement gates still apply.
    Refinement requires explicit ``settings.use_jax=True`` for a differentiable
    analysis; this is checked before rendering or starting a sampler.
    With no ``observation``, the public simulator injects the trial using the
    prepared truth PSF. Pass ``prepared.observation`` explicitly for a null
    control. Prepared truth and fit PSFs are reused; no telescope optics rebuild.
    Freed fits require ``mass_context`` with explicitly chosen physical mass
    support; the new entry point never chooses a mass-prior range.
    Pixel support defaults to the prepared Fisher include-mask, intersected
    with the fit-kernel safe border; ``mask_bool_use`` explicitly overrides it.
    The tested AutoArray backend cannot fit a 1x1 kernel (empty blurring grid);
    this input is rejected before simulation or sampler execution.
    """
    if output_dir is None:
        raise ValueError("output_dir is required for nonlinear fit artifacts")
    if dataset_kind not in {"asimov", "noisy"}:
        raise ValueError("dataset_kind must be asimov or noisy")
    if fit_mode not in {"fixed_template", "local_search", "freed"}:
        raise ValueError("Unsupported nonlinear fit_mode")
    if fit_mode == "freed" and mass_context is None:
        raise ValueError("freed fits require an explicit mass_context and mass support")

    from .autolens_runner import AutoLensFitRunner, NonlinearSearchSettings
    from .dataset_builder import imaging_from_observation, fitted_kernel_sha256
    from .validator import NonlinearMetricValidator
    from ...lensing.sampling import configured_sub_size
    from ...psf.mismatch import _kernel_sha256, build_psf_mismatch_spec
    from ...psf.utils import pyauto_kernel_native

    settings = settings or NonlinearSearchSettings()
    if profile_settings is not None and not settings.use_jax:
        raise ValueError("profile refinement requires settings.use_jax=True")
    fit_kernel = prepared.fit_psf.kernel
    if pyauto_kernel_native(fit_kernel).shape == (1, 1):
        raise ValueError(
            "1x1 fit kernels are unsupported by the tested nonlinear backend: "
            "AutoArray cannot render its empty blurring grid; use a wider kernel"
        )
    validate_identity = getattr(prepared, "validate_identity", None)
    if callable(validate_identity):
        validate_identity()
    custom_mask = mask_bool_use is not None
    if mask_bool_use is None:
        mask_bool_use = getattr(getattr(prepared, "detector", None), "mask_2d", None)
    if observation is None:
        from ... import simulate

        observation = simulate(prepared, trial=trial, sample_noise=(dataset_kind == "noisy"))
    config = deepcopy(prepared.config)
    mode = str((config.get("modeling", {}).get("fit_psf") or {}).get("mode", "matched")).lower()
    if mode not in {"matched", "bank", "delta", "explicit", "kernel"}:
        raise ValueError("Unsupported nonlinear fit-PSF mode")
    fit_label = "matched"
    supplied = None
    if mode == "matched":
        if pyauto_kernel_native(fit_kernel).shape != pyauto_kernel_native(observation.psf).shape:
            raise ValueError("matched nonlinear validation requires the prepared truth PSF")
        try:
            fitted_kernel_sha256(observation, fit_kernel, observation.pixel_scale)
        except ValueError as exc:
            raise ValueError("matched nonlinear validation requires the prepared truth PSF") from exc
    else:
        supplied = fit_kernel
        if mode == "kernel":
            descriptor = config["modeling"]["fit_psf"]
            native = pyauto_kernel_native(fit_kernel)
            if _kernel_sha256(native) != descriptor["kernel_sha256"]:
                raise ValueError("prepared fit kernel differs from its declared identity")
            if list(native.shape) != list(descriptor["shape_native"]):
                raise ValueError("prepared fit kernel differs from its declared shape")
            from math import isclose

            if not isclose(float(descriptor["pixel_scale_arcsec"]), float(observation.pixel_scale), rel_tol=0.0, abs_tol=1e-12):
                raise ValueError("prepared fit kernel differs from observation sampling")
            fit_label = f"kernel:{descriptor['kernel_sha256']}"
        elif mode in {"delta", "explicit"}:
            spec = build_psf_mismatch_spec(config)
            fit_label = f"{mode}:{spec.delta_id}"
        else:
            fit_label = f"bank:{_kernel_sha256(pyauto_kernel_native(fit_kernel))}"
    dataset, metadata = imaging_from_observation(
        observation,
        psf_for_fit=supplied,
        dataset_kind=dataset_kind,
        background_treatment=background_treatment,
        mask_bool_use=mask_bool_use,
        psf_truth_label="prepared_truth",
        psf_fit_label=fit_label,
        objective_version="consistent_sampling_v2",
        generation_sub_size=configured_sub_size(config["lensing"]["grid"]),
    )
    if custom_mask:
        metadata = replace(metadata, mask_name="custom_minus_psf_border")
    expected_psf = None if supplied is None else fitted_kernel_sha256(
        dataset, supplied, observation.pixel_scale,
    )
    if mode == "kernel":
        config["modeling"]["fit_psf"]["kernel_sha256"] = expected_psf
        fit_label = f"kernel:{expected_psf}"
        metadata = replace(metadata, psf_fit_label=fit_label)
    if profile_settings is None:
        runner = AutoLensFitRunner(settings, output_dir=Path(output_dir))
        validator = NonlinearMetricValidator(runner)
    else:
        from .fresh_profile import FreshProfileRunner, FreshProfileValidator

        runner = FreshProfileRunner(settings, output_dir=Path(output_dir), profile_settings=profile_settings)
        validator = FreshProfileValidator(runner)
    return validator.validate_case(
        dataset, metadata, config, trial,
        fit_mode=fit_mode,
        psf_case=fit_label,
        priors_config=priors,
        mass_context=mass_context,
        expected_psf_fit_sha256=expected_psf,
    )
