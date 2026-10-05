"""Prepare and fit one nonlinear subhalo case, with the observation kind supplied by its data."""

from __future__ import annotations

import math
import os
import types
from collections.abc import Mapping, Sequence
from contextlib import nullcontext
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from ..identity import array_digest, mapping_digest
from ..observation.observation import Observation
from ..scene.halos import Halo
from ..scene.image_source import frozen_value
from .data import FitData, build_fit_data, support_half_widths
from .fit_model import FitModel, autofit_model
from .hypotheses import RoleModels, build_role_models
from .objective import BoxObjective, jax_objective
from .result import CaseResult, ForecastReference, ObservationRecord
from .settings import FitSpec, PixelMask, RefineSettings, SamplerSettings
from .statistics import likelihood_ratio

if TYPE_CHECKING:
    from ..fisher.api import PreparedForecast
    from .backend import BackendSession

__all__ = ["PreparedCase", "prepare_case", "validate_nonlinear"]


@dataclass(frozen=True, eq=False)
class PreparedCase:
    hypothesis: Halo
    observation: Observation
    fit: FitSpec
    use_jax: bool
    data: FitData
    models: RoleModels
    autofit_models: Mapping[str, Any]
    analysis: Any
    record: Mapping[str, Any]
    _objectives: dict[str, BoxObjective] = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "record", frozen_value(self.record))

    def model(self, role: str) -> FitModel:
        if role not in ("smooth", "subhalo"):
            raise ValueError(f"role must be smooth or subhalo, got {role!r}")
        return getattr(self.models, role)

    def truth_vector(self, role: str) -> np.ndarray:
        return self.model(role).truth

    def _instance(self, role: str, vector: Sequence[float]) -> Any:
        model = self.model(role)
        values = np.asarray(vector, dtype=float)
        if values.shape != model.truth.shape or not np.all(np.isfinite(values)):
            raise ValueError(f"{role} vector must contain {len(model.parameter_names)} finite parameters")
        return self.autofit_models[role].instance_from_vector(vector=values.tolist())

    def log_likelihood(self, role: str, vector: Sequence[float]) -> float:
        return float(self.analysis.log_likelihood_function(self._instance(role, vector)))

    def fit_at(self, role: str, vector: Sequence[float]) -> Any:
        return self.analysis.fit_from(instance=self._instance(role, vector))

    def chi_squared(self, role: str, vector: Sequence[float]) -> float:
        return float(self.fit_at(role, vector).chi_squared)

    def objective(self, role: str) -> BoxObjective:
        model = self.model(role)
        if not self.use_jax:
            raise ValueError("refinement requires a prepared JAX analysis")
        if role not in self._objectives:
            checked = any(component.profile_class == "hwoslaps.inference.light_profiles:Exponential"
                          for galaxy in model.galaxies for _, component in galaxy.components)
            self._objectives[role] = jax_objective(self.analysis, self.autofit_models[role], model.lower, model.upper,
                                                  check_gradient_domain=checked)
        return self._objectives[role]


def _check_inputs(prepared: PreparedForecast, trial: Halo, observation: Observation, fit: FitSpec) -> Any:
    prepared.validate_identity()
    if observation.config_digest != prepared.record["config_digest"]:
        raise ValueError("simulate the observation from this PreparedForecast: configuration digests differ")
    if prepared.data_space.whitener.mode == "dense":
        raise ValueError("dense forecast noise covariance is unsupported by the diagonal nonlinear likelihood")
    truth = prepared.psfs.truth_kernels
    if set(observation.psfs.group_index) != set(truth.group_index) or any(
            observation.psfs.for_group(group).kernel_identity() != truth.for_group(group).kernel_identity()
            for group in truth.group_index):
        raise ValueError("the observation truth kernel binding differs from this PreparedForecast")
    if len(prepared.psfs.model_kernels.kernels) != 1:
        raise ValueError("nonlinear fitting needs one distinct model kernel; use a kernel or monochromatic model "
                         "PSF with a chromatic truth")
    kernel = prepared.psfs.model_kernels.single
    if kernel.shape == (1, 1):
        raise ValueError("a 1x1 model kernel leaves AutoArray an empty blurring grid; use a larger kernel")
    if observation.subhalo is not None and observation.subhalo != trial:
        raise ValueError("the observation injects another hypothesis")
    spec = prepared.scene.spec
    redshift = spec.lens.redshift if spec.subhalo_redshift is None else spec.subhalo_redshift
    if trial.redshift != redshift:
        raise ValueError(f"trial redshift {trial.redshift} differs from configured hypothesis redshift {redshift}")
    if trial.source_redshift != spec.source.redshift or trial.cosmology != prepared.scene.cosmology:
        raise ValueError("trial source redshift or cosmology differs from the prepared scene")
    support = support_half_widths(tuple(observation.grid.shape), observation.pixel_scale_arcsec, kernel.shape)
    if any(abs(value) > half for value, half in zip(trial.position_yx_arcsec, support, strict=True)):
        raise ValueError(f"trial position {trial.position_yx_arcsec} lies outside fit support {support}")
    if fit.h1 == "truth_anchor" and observation.kind != "expected":
        raise ValueError("a truth anchor requires an expected observation")
    return kernel


def prepare_case(prepared: PreparedForecast, trial: Halo, observation: Observation, *, fit: FitSpec,
                 use_jax: bool) -> PreparedCase:
    kernel = _check_inputs(prepared, trial, observation, fit)
    from .backend import ensure_jax_x64, make_analysis, require_jax_fitness_api

    if use_jax:
        ensure_jax_x64()
        require_jax_fitness_api()
    if isinstance(fit.mask, PixelMask):
        mask_name, base_mask = "custom_minus_psf_border", fit.mask.values
    else:
        mask_name = fit.mask
        base_mask = (np.ones(observation.grid.shape, dtype=bool) if fit.mask == "all_pixels_minus_psf_border"
                     else prepared.mask)
    data = build_fit_data(observation, kernel, mask_name=mask_name, base_mask=base_mask,
                          over_sample_size=prepared.scene.spec.grid.over_sample_size)
    free = tuple(parameter.name for parameter in prepared.nuisances.parameters if parameter.kind == "scene")
    models = build_role_models(prepared.scene, trial, free, fit, use_jax=use_jax)
    converted = {role: autofit_model(getattr(models, role)) for role in ("smooth", "subhalo")}
    analysis = make_analysis(data.imaging, cosmology=prepared.scene.cosmology.autogalaxy(), use_jax=use_jax)
    record = {"prepared": dict(prepared.record), "hypothesis": trial.to_mapping(),
              "observation": {"kind": observation.kind, "noise_seed": observation.noise_seed,
                              "config_digest": observation.config_digest, "data_digest": array_digest(observation.data_adu),
                              "subhalo": None if observation.subhalo is None else observation.subhalo.to_mapping()},
              "data": data.record, "models": {role: getattr(models, role).digest() for role in converted},
              "fit": fit.to_mapping(), "fitted_parameters": free,
              "comparison_digest": prepared.record["comparison_digest"]}
    return PreparedCase(trial, observation, fit, use_jax, data, models, types.MappingProxyType(converted),
                         analysis, types.MappingProxyType(record))


def validate_nonlinear(prepared: PreparedForecast, trial: Halo, observation: Observation, *, fit: FitSpec,
                       sampler: SamplerSettings, sampler_seed: int, output_dir: str | os.PathLike[str],
                       refine: RefineSettings | None = None, session: BackendSession | None = None,
                       forecast_reference: ForecastReference | None = None, case_id: str | None = None) -> CaseResult:
    if refine is not None and not sampler.use_jax:
        raise ValueError("refinement requires sampler.use_jax")
    if isinstance(sampler_seed, bool) or not isinstance(sampler_seed, (int, np.integer)) or sampler_seed < 0:
        raise ValueError("sampler_seed must be a nonnegative integer")
    if session is not None and not session.active:
        raise RuntimeError("a supplied BackendSession must be active before fitting a case")
    if forecast_reference is not None and (forecast_reference.mass_msun != trial.mass_msun
                                         or forecast_reference.position_yx_arcsec != trial.position_yx_arcsec):
        raise ValueError(f"forecast reference node {(forecast_reference.mass_msun, forecast_reference.position_yx_arcsec)} "
                         f"differs from trial node {(trial.mass_msun, trial.position_yx_arcsec)}")
    case = prepare_case(prepared, trial, observation, fit=fit, use_jax=sampler.use_jax)
    for role in ("smooth", "subhalo"):
        if not math.isfinite(case.log_likelihood(role, case.truth_vector(role))):
            raise ValueError(f"{role} truth likelihood is non-finite before any search")
    if fit.h1 == "truth_anchor":
        chi2 = case.chi_squared("subhalo", case.truth_vector("subhalo"))
        if not chi2 <= fit.anchor_chi2_tolerance:
            raise ValueError(f"truth-anchor chi-square {chi2} exceeds tolerance {fit.anchor_chi2_tolerance}")
    if case_id is None:
        case_id = mapping_digest(dict(case.record) | {"sampler": sampler.to_mapping(), "sampler_seed": int(sampler_seed),
                                                     "refine": None if refine is None else refine.to_mapping()})[:16]
    if not isinstance(case_id, str) or not case_id or Path(case_id).name != case_id or case_id in (".", ".."):
        raise ValueError("case_id must be a nonempty directory name")
    case_dir = Path(output_dir).resolve() / case_id
    if case_dir.exists() and (not case_dir.is_dir() or any(case_dir.iterdir())):
        raise ValueError(f"case directory {case_dir} must be absent or empty")
    case_dir.mkdir(parents=True, exist_ok=True)
    from .backend import BackendSession
    from .recovery import extract_recovery
    from .roles import anchor_role, run_search_role

    scope = BackendSession() if session is None else nullcontext(session)
    with scope as active:
        smooth, smooth_outcome = run_search_role(case, "smooth", sampler=sampler, sampler_seed=int(sampler_seed),
                                                  refine=refine, session=active, case_dir=case_dir, case_id=case_id)
        if fit.h1 == "truth_anchor":
            subhalo, subhalo_outcome = anchor_role(case), None
        else:
            subhalo, subhalo_outcome = run_search_role(case, "subhalo", sampler=sampler, sampler_seed=int(sampler_seed),
                                                       refine=refine, session=active, case_dir=case_dir, case_id=case_id)
    recovery = None
    if fit.mode == "freed" and subhalo_outcome is not None and subhalo_outcome.result is not None:
        recovery = extract_recovery(subhalo_outcome.result, case.models.mass_mapping, case.model("subhalo"),
                                    subhalo.refinement)
    q_signed = q_clipped = evidence_difference = None
    if smooth.log_likelihood is not None and subhalo.log_likelihood is not None:
        q_signed, q_clipped = likelihood_ratio(smooth.log_likelihood, subhalo.log_likelihood)
    if subhalo_outcome is not None and smooth.status == subhalo.status == "success":
        evidence0, evidence1 = smooth_outcome.record.log_evidence, subhalo_outcome.record.log_evidence
        if evidence0 is not None and evidence1 is not None and math.isfinite(evidence0) and math.isfinite(evidence1):
            evidence_difference = evidence1 - evidence0
    from ..provenance import capture_provenance

    return CaseResult(case_id, trial, ObservationRecord(observation.kind, observation.noise_seed,
                                                        observation.config_digest, array_digest(observation.data_adu),
                                                        observation.subhalo),
                       case.data.record, fit, sampler, int(sampler_seed), refine,
                       case.record["models"], smooth, subhalo, q_signed, q_clipped, evidence_difference,
                       recovery, forecast_reference, case.record["fitted_parameters"],
                       case.record["comparison_digest"], capture_provenance())
