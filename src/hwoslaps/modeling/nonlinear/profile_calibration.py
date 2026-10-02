"""Truth-anchor construction and likelihood-matched calibration diagnostics.

These optional diagnostics never modify the sampler or local-profile objective.
"""
from __future__ import annotations

from typing import Any, Mapping, Sequence, TYPE_CHECKING

import numpy as np
from scipy.optimize import lsq_linear

from .output_schema import NonlinearFitSummary
from .profile_settings import OBJECTIVE_VERSION, PROCEDURE_VERSION

if TYPE_CHECKING:
    from .fresh_profile import FreshProfileRunner


def linearized_comparator(
    residual, smooth_truth, injected_residual, lower, upper, step_fraction=1.0e-5, background_column=None
):
    """Profile a finite injected signal over the common smooth-model tangent.

    Derivatives use central differences at a declared truth expansion point.
    Return unrestricted and finite-box linearized predictions separately.
    """
    smooth_truth = np.asarray(smooth_truth)
    r0 = np.asarray(residual(smooth_truth))
    signal = r0 - np.asarray(injected_residual)
    columns = []
    steps = []
    for j, width in enumerate(upper - lower):
        h = min(
            step_fraction * width, 0.25 * (smooth_truth[j] - lower[j]), 0.25 * (upper[j] - smooth_truth[j])
        )
        if h <= 0:
            raise ValueError("Comparator expansion point must be inside its finite box")
        plus, minus = smooth_truth.copy(), smooth_truth.copy()
        plus[j] += h
        minus[j] -= h
        columns.append(-(np.asarray(residual(plus)) - np.asarray(residual(minus))) / (2 * h))
        steps.append(float(h))
    jac = np.column_stack(columns)
    scale = np.linalg.norm(jac, axis=0)
    if np.any(scale == 0):
        raise ValueError("Zero comparator nuisance derivative")
    normalized = jac / scale
    coeff, _, rank, singular = np.linalg.lstsq(normalized, signal, rcond=1.0e-12)
    projected = signal - normalized @ coeff
    bounded = lsq_linear(
        normalized,
        signal,
        bounds=((lower - smooth_truth) * scale, (upper - smooth_truth) * scale),
        tol=1.0e-10,
        max_iter=100,
    )
    answer = {
        "q": float(projected @ projected),
        "q_with_finite_prior_box": float(np.linalg.norm(signal - normalized @ bounded.x) ** 2),
        "bounded_solver_success": bool(bounded.success),
        "rank": int(rank),
        "singular_values": singular.tolist(),
        "derivative_steps": steps,
        "n_pixels": len(signal),
        "n_shared_nuisance": len(smooth_truth),
        "unbounded_nuisance_delta": (coeff / scale).tolist(),
        "bounds_active": np.asarray(bounded.active_mask).tolist(),
        "position_policy": "fixed injected position; H1 profile searches its declared box",
        "background": "known subtracted, no free offset",
        "regularization": None,
    }
    if background_column is not None:
        bg = np.asarray(background_column)
        augmented = np.column_stack([normalized, bg / np.linalg.norm(bg)])
        delta = np.linalg.lstsq(augmented, signal, rcond=1.0e-12)[0]
        answer["q_with_free_background_only"] = float(np.linalg.norm(signal - augmented @ delta) ** 2)
    return answer

def _instance_value(instance: Any, path: Sequence[str]) -> float:
    """Read one dotted/tuple AutoFit path from an instance."""
    value = instance
    for part in path:
        if isinstance(value, tuple) and str(part).rsplit("_", 1)[-1].isdigit():
            value = value[int(str(part).rsplit("_", 1)[-1])]
        else:
            value = getattr(value, part)
    return float(value)


def _native_residual(value: Any) -> np.ndarray:
    value = getattr(value, "array", getattr(value, "native", value))
    return np.asarray(value, dtype=float).reshape(-1)


def likelihood_matched_tangent(
    runner: Any,
    dataset: Any,
    full_config: Mapping[str, Any],
    trial: Any,
    *,
    fit_mode: str,
    mass_context: Any = None,
    comparator_tolerance: float = 1.0e-3,
) -> dict[str, Any]:
    """Reuse the validated likelihood-matched tangent comparator.

    The comparator is evaluated at the physical truth point with the same
    dataset and model construction used by the case.  It is a diagnostic
    linkage and does not alter the fresh sampler or local-profile maxima.
    """
    from .autolens_model_builder import (
        autofit_model_from_spec,
        fixed_point_model_spec_from_trial,
        smooth_model_spec_from_config,
        subhalo_model_spec_from_trial,
    )

    fixed_model = autofit_model_from_spec(
        fixed_point_model_spec_from_trial(full_config, trial)
    )
    truth_instance = fixed_model.instance_from_prior_medians()
    smooth_model = autofit_model_from_spec(
        smooth_model_spec_from_config(full_config)
    )
    subhalo_model = autofit_model_from_spec(
        subhalo_model_spec_from_trial(
            full_config,
            trial,
            fit_mode=fit_mode,
            mass_context=mass_context,
        )
    )
    smooth_names = list(smooth_model.unique_prior_paths)
    subhalo_names = list(subhalo_model.unique_prior_paths)
    smooth_x = np.asarray(
        [_instance_value(truth_instance, path) for path in smooth_names],
        dtype=float,
    )
    subhalo_x = np.asarray(
        [
            np.log10(trial.mass_msun)
            if path[-1] == "log10_m200"
            else _instance_value(truth_instance, path)
            for path in subhalo_names
        ],
        dtype=float,
    )
    smooth_lower = np.asarray(
        [prior.lower_limit for prior in smooth_model.priors_ordered_by_id],
        dtype=float,
    )
    smooth_upper = np.asarray(
        [prior.upper_limit for prior in smooth_model.priors_ordered_by_id],
        dtype=float,
    )
    subhalo_lower = np.asarray(
        [prior.lower_limit for prior in subhalo_model.priors_ordered_by_id],
        dtype=float,
    )
    subhalo_upper = np.asarray(
        [prior.upper_limit for prior in subhalo_model.priors_ordered_by_id],
        dtype=float,
    )
    if np.any(subhalo_x < subhalo_lower) or np.any(subhalo_x > subhalo_upper):
        raise ValueError("truth H1 vector is outside the declared profile support")
    analysis = runner.make_analysis(dataset)

    def residual(model: Any, vector: np.ndarray) -> np.ndarray:
        instance = model.instance_from_vector(vector=vector.tolist())
        return _native_residual(analysis.fit_from(instance=instance).normalized_residual_map)

    fixed_residual = residual(subhalo_model, subhalo_x)
    noise = _native_residual(dataset.noise_map)
    base = linearized_comparator(
        lambda vector: residual(smooth_model, np.asarray(vector, dtype=float)),
        smooth_x,
        fixed_residual,
        smooth_lower,
        smooth_upper,
        step_fraction=1.0e-5,
        background_column=1.0 / noise,
    )
    tighter = linearized_comparator(
        lambda vector: residual(smooth_model, np.asarray(vector, dtype=float)),
        smooth_x,
        fixed_residual,
        smooth_lower,
        smooth_upper,
        step_fraction=5.0e-6,
        background_column=1.0 / noise,
    )
    base.update(
        {
            "computed": True,
            "smooth_truth_vector": smooth_x.tolist(),
            "fixed_h1_reference_vector": subhalo_x.tolist(),
            "fixed_h1_reference_chi2": float(fixed_residual @ fixed_residual),
            "half_step_q_difference": abs(float(base["q"]) - float(tighter["q"])),
            "derivatives_stable": abs(float(base["q"]) - float(tighter["q"]))
            <= float(comparator_tolerance),
            "comparator_tolerance": float(comparator_tolerance),
            "objective_version": OBJECTIVE_VERSION,
            "no_sampler_or_fit_state_import": True,
        }
    )
    return base


class ZeroResidualAnchorRunner:
    """Evaluate a declared H1 anchor without sampling."""

    def __init__(
        self,
        fresh_runner: FreshProfileRunner,
        anchor: Mapping[str, Any],
        tolerance: float = 1.0e-8,
    ):
        self._fresh_runner = fresh_runner
        self.anchor = dict(anchor)
        self.tolerance = float(tolerance)
        self.profile_records = fresh_runner.profile_records
        self.profile_settings = fresh_runner.profile_settings
        self.settings = fresh_runner.settings
        self.output_dir = fresh_runner.output_dir
        self._preflight_anchors: dict[tuple[str, ...], dict[str, Any]] = {}

    def make_analysis(self, *args: Any, **kwargs: Any) -> Any:
        return self._fresh_runner.make_analysis(*args, **kwargs)

    def preflight_anchor(
        self,
        dataset: Any,
        full_config: Mapping[str, Any],
        trial: Any,
        *,
        fit_mode: str,
        mass_context: Any = None,
    ) -> None:
        """Verify the H1 anchor before the delegate starts the H0 search."""
        from .autolens_model_builder import (
            autofit_model_from_spec,
            subhalo_model_spec_from_trial,
        )

        spec = subhalo_model_spec_from_trial(
            dict(full_config),
            trial,
            fit_mode=fit_mode,
            mass_context=mass_context,
        )
        model = autofit_model_from_spec(spec)
        analysis = self.make_analysis(dataset, model_metadata=spec.metadata)
        result = self._evaluate_anchor(model, analysis, fit_mode)
        self._preflight_anchors[tuple(result["parameter_names"])] = result

    def _evaluate_anchor(
        self,
        model: Any,
        analysis: Any,
        fit_mode: str,
    ) -> dict[str, Any]:
        names = [".".join(path) for path in model.unique_prior_paths]
        expected_names = self.anchor.get("parameter_names")
        if list(expected_names or []) != names:
            raise ValueError(
                "zero-residual H1 anchor parameter names differ from the runtime model"
            )
        lower = np.asarray(
            [prior.lower_limit for prior in model.priors_ordered_by_id],
            dtype=float,
        )
        upper = np.asarray(
            [prior.upper_limit for prior in model.priors_ordered_by_id],
            dtype=float,
        )
        from .fresh_profile import _finite_vector

        vector = _finite_vector(self.anchor.get("vector", []), lower, upper)
        instance = model.instance_from_vector(vector=vector.tolist())
        fit = analysis.fit_from(instance=instance)
        residual = np.asarray(
            getattr(fit.normalized_residual_map, "native", fit.normalized_residual_map),
            dtype=float,
        )
        chi2 = float(residual.reshape(-1) @ residual.reshape(-1))
        if not np.isfinite(chi2) or chi2 > self.tolerance:
            raise ValueError(
                f"declared H1 anchor is not numerical-zero residual: chi2={chi2}"
            )
        return {
            "parameter_names": names,
            "vector": vector.tolist(),
            "chi2": chi2,
            "log_likelihood": float(analysis.log_likelihood_function(instance)),
            "fit_mode": fit_mode,
        }

    def run_model(
        self,
        *,
        model: Any,
        analysis: Any,
        role: str,
        analysis_key: str,
        fit_mode: str,
        **kwargs: Any,
    ) -> NonlinearFitSummary:
        if role != "subhalo":
            return self._fresh_runner.run_model(
                model=model,
                analysis=analysis,
                role=role,
                analysis_key=analysis_key,
                fit_mode=fit_mode,
                **kwargs,
            )
        names = tuple(".".join(path) for path in model.unique_prior_paths)
        cached = self._preflight_anchors.pop(names, None)
        if cached is None:
            cached = self._evaluate_anchor(model, analysis, fit_mode)
        vector = np.asarray(cached["vector"], dtype=float)
        chi2 = float(cached["chi2"])
        log_likelihood = float(cached["log_likelihood"])
        summary = NonlinearFitSummary(
            model_role="subhalo",
            fit_mode=fit_mode,
            status="success",
            log_likelihood_max=log_likelihood,
            figure_of_merit_max=log_likelihood,
            log_evidence=None,
            n_free_parameters=len(names),
            result_path=None,
            runtime_s=0.0,
            warnings=["verified_zero_residual_h1_anchor_no_evidence"],
            log_likelihood_extraction_method="direct_zero_residual_anchor",
            use_jax_requested=self.settings.use_jax,
            use_jax_effective=getattr(analysis, "_use_jax", None),
            search_engine="VerifiedZeroResidualAnchor",
            analysis_key=analysis_key,
            search_internal_retention_requested=False,
            search_internal_retained=False,
        )
        self.profile_records["subhalo"] = {
            "procedure_version": PROCEDURE_VERSION,
            "candidate_acceptance_status": "verified_zero_residual_anchor",
            "sampler_executed": False,
            "evidence_claim": False,
            "anchor_vector": vector.tolist(),
            "anchor_chi2": chi2,
            "candidate_best_log_likelihood": log_likelihood,
        }
        return summary
