"""Versioned replay of archived vectors through the original imaging objective.

The sampler is never called. Results distinguish identity, local maximization,
and a fixed-position linearized comparator; no posterior/evidence is updated.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np
from scipy.optimize import lsq_linear

from .autolens_runner import AutoLensFitRunner, _native_array
from .local_profile import fit_local_least_squares_profile
from .output_schema import NonlinearFitSummary, _json_safe
from .validator import NonlinearMetricValidator


def atomic_json(path, payload):
    """Replace a JSON checkpoint atomically; refuse non-standard NaN tokens."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    encoded = json.dumps(_json_safe(payload), indent=2, allow_nan=False) + "\n"
    with temporary.open("w") as stream:
        stream.write(encoded)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def residual_array(value, xp):
    """Unwrap AutoArray without invoking its host-only NumPy protocol."""
    return xp.asarray(getattr(value, "array", value)).reshape(-1)


def array_identity(value):
    array = _native_array(value)
    return {
        "shape": list(array.shape),
        "dtype": array.dtype.str,
        "sha256": hashlib.sha256(array.tobytes()).hexdigest(),
    }


def archive_vectors(summary_path, samples_path, names, lower, upper, max_starts=4, separation=0.05):
    """Select the archived ML point and separated high-likelihood samples."""
    summary = json.loads(Path(summary_path).read_text())["arguments"]["max_log_likelihood_sample"][
        "arguments"
    ]
    values = summary["kwargs"]["arguments"]
    if set(values) != set(names):
        raise ValueError("Archived ML parameter names do not match runtime model")
    first = np.array([values[name] for name in names], dtype=float)
    if not np.all(np.isfinite(first)) or np.any(first < lower) or np.any(first > upper):
        raise ValueError("Archived ML vector is not admissible")
    selected = [first]
    provenance = [{"origin": "archived_ML", "saved_logL": summary["log_likelihood"]}]
    if max_starts < 1:
        raise ValueError("max_starts must be positive")
    if samples_path is not None and max_starts > 1:
        with Path(samples_path).open() as stream:
            reader = csv.DictReader(stream, skipinitialspace=True)
            rows = [{key.strip(): value for key, value in row.items()} for row in reader]
        ordered = sorted(enumerate(rows), key=lambda pair: (-float(pair[1]["log_likelihood"]), pair[0]))
        for index, row in ordered:
            x = np.array([float(row[name]) for name in names])
            if not np.all(np.isfinite(x)) or np.any(x < lower) or np.any(x > upper):
                continue
            if min(np.linalg.norm((x - old) / (upper - lower)) for old in selected) < separation:
                continue
            selected.append(x)
            provenance.append({"origin": "sample", "row": index, "saved_logL": float(row["log_likelihood"])})
            if len(selected) >= max_starts:
                break
    return selected, provenance


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


def profile_stability(attempts, initial_best, repeat_best, tolerance, minimum_support=2):
    """Assess repeatability of the best value, retaining poorer local optima.

    Agreement is evidence for a tested local maximum, not global optimality.
    The distinct starting vectors are selected separately from archived data.
    """
    best_chi2 = min(initial_best, repeat_best)
    spread = (max(attempt.chi2 for attempt in attempts) - best_chi2) / 2
    support = [
        attempt.label
        for attempt in attempts
        if np.isfinite(attempt.chi2) and (attempt.chi2 - best_chi2) / 2 <= tolerance
    ]
    repeat_change = abs(repeat_best - initial_best) / 2
    return {
        "stable": repeat_change <= tolerance and len(support) >= minimum_support,
        "supporting_starts": support,
        "minimum_supporting_starts": minimum_support,
        "start_best_logL_spread": spread,
        "repeat_logL_change": repeat_change,
        "stability_scope": "best observed local maximum; full endpoint spread retained; no global proof",
    }


def instance_value(instance, path):
    for part in path:
        if isinstance(instance, tuple) and part.rsplit("_", 1)[-1].isdigit():
            instance = instance[int(part.rsplit("_", 1)[-1])]
        else:
            instance = getattr(instance, part)
    return float(instance)


class ProfileReplayRunner(AutoLensFitRunner):
    """Runner used only by the explicitly versioned profiling entry point."""

    def __init__(self, settings, output_dir, replay, procedure):
        super().__init__(settings, output_dir)
        self.replay = replay
        self.procedure = procedure
        self.records = {}
        self.functions = {}
        self.context = None

    def run_model(self, *, model, analysis, role, fit_mode, case_id, analysis_key, **kwargs):
        import jax
        import jax.numpy as jnp

        started = time.perf_counter()
        old = self.replay["case"]["case"][role + "_fit"]
        if analysis_key != old["analysis_key"]:
            raise ValueError(f"{role}: reconstructed analysis key differs from archived identity")
        names = [".".join(path) for path in model.unique_prior_paths]
        lower = np.array([p.lower_limit for p in model.priors_ordered_by_id], dtype=float)
        upper = np.array([p.upper_limit for p in model.priors_ordered_by_id], dtype=float)
        starts, origins = archive_vectors(
            self.replay[role]["summary"],
            self.replay[role].get("samples"),
            names,
            lower,
            upper,
            self.procedure["max_starts"],
            self.procedure["start_separation"],
        )

        def residual_call(x):
            instance = model.instance_from_vector(vector=x, xp=jnp)
            fit = analysis.fit_from(instance=instance)
            return residual_array(fit.normalized_residual_map, jnp)

        def likelihood_call(x):
            instance = model.instance_from_vector(vector=x, xp=jnp)
            return analysis.log_likelihood_function(instance=instance)

        compiled_residual = jax.jit(residual_call)
        compiled_likelihood = jax.jit(likelihood_call)
        compile_start = time.perf_counter()

        def residual(x):
            return np.asarray(compiled_residual(x), dtype=np.float64)

        r_ml = residual(starts[0])
        l_ml = float(compiled_likelihood(starts[0]))
        compile_seconds = time.perf_counter() - compile_start
        direct_l = float(likelihood_call(starts[0]))
        direct_r = np.asarray(residual_call(starts[0]), dtype=float)
        saved_l = float(old["log_likelihood_max"])
        error = abs(l_ml - saved_l)
        same_path_error = float(np.linalg.norm(direct_r - r_ml) ** 2)
        identity = {
            "saved_logL": saved_l,
            "replayed_logL": l_ml,
            "absolute_error": error,
            "direct_compiled_logL_error": abs(direct_l - l_ml),
            "direct_compiled_squared_residual_error": same_path_error,
            "analysis_key": analysis_key,
            "parameter_names": names,
            "lower_bounds": lower.tolist(),
            "upper_bounds": upper.tolist(),
            "archived_ML": starts[0].tolist(),
            "compile_and_initial_evaluation_s": compile_seconds,
            "data": array_identity(analysis.dataset.data.native),
            "noise": array_identity(analysis.dataset.noise_map.native),
            "mask": array_identity(analysis.dataset.mask),
            "psf": array_identity(analysis.dataset.psf),
        }
        identity["passed"] = bool(
            np.all(np.isfinite([saved_l, l_ml, direct_l, same_path_error]))
            and max(error, abs(direct_l - l_ml), same_path_error) <= self.procedure["identity_tolerance"]
        )
        record = {"identity": identity, "starts": origins, "mode": self.procedure["mode"]}
        self.records[role] = record
        atomic_json(Path(self.output_dir) / "profile_progress.json", self.records)
        if not identity["passed"]:
            raise ValueError(f"{role}: fixed-vector identity failed: {identity}")
        self.functions[role] = (residual, starts, lower, upper, model, compiled_likelihood)
        best_x = starts[0]
        best_chi2 = float(r_ml @ r_ml)
        jacobian = None
        if self.procedure.get("jacobian", "finite_difference") == "jax_forward":
            before_jacobian = time.perf_counter()
            compiled_jacobian = jax.jit(jax.jacfwd(residual_call))

            def analytic_jacobian(x):
                return np.asarray(compiled_jacobian(x), dtype=np.float64)

            jacobian = analytic_jacobian
            matrix = jacobian(starts[0])
            if not np.all(np.isfinite(matrix)):
                raise ValueError("JAX residual Jacobian has nonfinite elements")
            direction = (upper - lower) * np.where(np.arange(len(lower)) % 2, -1.0, 1.0)
            step = min(
                1.0e-6, float(np.min(np.minimum(starts[0] - lower, upper - starts[0]) / (upper - lower))) / 4
            )
            if step <= 0:
                raise ValueError(
                    "Archived ML point is on a bound; directional Jacobian gate needs an interior point"
                )
            central = (residual(starts[0] + step * direction) - residual(starts[0] - step * direction)) / (
                2 * step
            )
            predicted = matrix @ direction
            relative_error = float(np.linalg.norm(central - predicted) / max(np.linalg.norm(predicted), 1.0))
            record["jacobian_gate"] = {
                "relative_directional_error": relative_error,
                "step": step,
                "compile_and_check_s": time.perf_counter() - before_jacobian,
                "passed": relative_error <= self.procedure["jacobian_tolerance"],
            }
            atomic_json(Path(self.output_dir) / "profile_progress.json", self.records)
            if not record["jacobian_gate"]["passed"]:
                raise ValueError("Analytic Jacobian directional check failed")
        if self.procedure["mode"] == "profile":

            def checkpoint(point):
                record["partial_best"] = point
                atomic_json(Path(self.output_dir) / "profile_progress.json", self.records)

            profile = fit_local_least_squares_profile(
                model_name=role,
                residual_fn=residual,
                initial_points=starts,
                lower_bounds=lower,
                upper_bounds=upper,
                max_nfev=self.procedure["max_nfev"],
                ftol=None if self.procedure.get("disable_ftol") else self.procedure["solver_tolerance"],
                xtol=self.procedure["solver_tolerance"],
                gtol=self.procedure["solver_tolerance"],
                progress_callback=checkpoint,
                jacobian_fn=jacobian,
            )
            repeat = fit_local_least_squares_profile(
                model_name=role,
                residual_fn=residual,
                initial_points=[profile.best.x],
                lower_bounds=lower,
                upper_bounds=upper,
                max_nfev=self.procedure["max_nfev"],
                ftol=None if self.procedure.get("disable_ftol") else self.procedure["solver_tolerance"] / 10,
                xtol=self.procedure["solver_tolerance"] / 10,
                gtol=self.procedure["solver_tolerance"] / 10,
                progress_callback=checkpoint,
                jacobian_fn=jacobian,
            )
            best = min([profile.best, repeat.best], key=lambda item: item.chi2)
            best_x, best_chi2 = np.asarray(best.x), best.chi2
            record.update(profile=profile.to_dict(), repeat=repeat.to_dict())
            record.update(
                profile_stability(
                    profile.attempts,
                    profile.chi2_min,
                    repeat.chi2_min,
                    self.procedure["stability_logL"],
                    self.procedure.get("minimum_supporting_starts", 2),
                )
            )
        new_l = float(compiled_likelihood(best_x))
        scalar_residual_error = abs((new_l - l_ml) - (float(r_ml @ r_ml) - best_chi2) / 2)
        record.update(
            best_vector=np.asarray(best_x).tolist(),
            best_chi2=best_chi2,
            logL=new_l,
            improvement=new_l - saved_l,
            scalar_residual_logL_error=scalar_residual_error,
            wall_s=time.perf_counter() - started,
        )
        if scalar_residual_error > self.procedure["identity_tolerance"]:
            raise ValueError("Scalar likelihood and residual objective diverged")
        atomic_json(Path(self.output_dir) / "profile_progress.json", self.records)
        return NonlinearFitSummary(
            model_role=role,
            fit_mode=fit_mode,
            status="success",
            log_likelihood_max=new_l,
            log_evidence=None,
            n_free_parameters=len(names),
            analysis_key=analysis_key,
            search_engine="ArchivedVectorProfile-v1",
            use_jax_requested=True,
            use_jax_effective=True,
            runtime_s=record["wall_s"],
            result_path=self.output_dir,
            log_likelihood_extraction_method="same_analysis_compiled_log_likelihood",
            runtime_provenance={"procedure": self.procedure},
        )


class ProfileReplayValidator(NonlinearMetricValidator):
    """Keep dataset/model guards and add explicit reference diagnostics."""

    def validate_case(self, dataset, dataset_metadata, full_config, trial, *args, **kwargs):
        from .autolens_model_builder import autofit_model_from_spec, fixed_point_model_spec_from_trial

        result = super().validate_case(dataset, dataset_metadata, full_config, trial, *args, **kwargs)
        # A local profile has no posterior recovery distribution to extract.
        result.quality_flags = [flag for flag in result.quality_flags if flag != "recovery_extraction_failed"]
        result.quality_flags.append("posterior_and_evidence_not_recomputed")
        runner = self.runner
        fixed = autofit_model_from_spec(fixed_point_model_spec_from_trial(full_config, trial))
        instance = fixed.instance_from_prior_medians()
        r0, _, lo0, hi0, model0, _ = runner.functions["smooth"]
        r1, _, lo1, hi1, model1, _ = runner.functions["subhalo"]
        x0 = np.array([instance_value(instance, path) for path in model0.unique_prior_paths])
        x1 = np.array(
            [
                np.log10(trial.mass_msun) if path[-1] == "log10_m200" else instance_value(instance, path)
                for path in model1.unique_prior_paths
            ]
        )
        admissible = bool(np.all(x1 >= lo1) and np.all(x1 <= hi1))
        if not admissible:
            raise ValueError("Reference vector is outside H1 support")
        ref_residual = r1(x1)
        analysis = runner.make_analysis(dataset)
        fixed_residual = np.asarray(
            analysis.fit_from(instance=instance).normalized_residual_map, dtype=float
        ).reshape(-1)
        path_error = float(np.linalg.norm(ref_residual - fixed_residual) ** 2)
        reference = {
            "admissible": admissible,
            "fixed_freed_residual_difference_squared": path_error,
            "chi2": float(ref_residual @ ref_residual),
            "vector": x1.tolist(),
            "operational_start": False,
        }
        if path_error > runner.procedure["identity_tolerance"]:
            raise ValueError("Fixed/freed reference rendering identity failed")
        runner.records["reference"] = reference
        noise = np.asarray(dataset.noise_map).reshape(-1)
        comparison = linearized_comparator(r0, x0, ref_residual, lo0, hi0, background_column=1 / noise)
        tighter = linearized_comparator(r0, x0, ref_residual, lo0, hi0, step_fraction=5.0e-6)
        comparison["half_step_q_difference"] = abs(comparison["q"] - tighter["q"])
        comparison["derivatives_stable"] = (
            comparison["half_step_q_difference"] <= runner.procedure["comparator_tolerance"]
        )
        comparison["q_F_production"] = runner.replay["case"]["rung"]["q_f_production_at_position"]
        comparison["q_F_support_matched"] = runner.replay["case"]["rung"]["q_f_matched"]
        runner.records["comparator"] = comparison
        q = 2 * (runner.records["subhalo"]["logL"] - runner.records["smooth"]["logL"])
        q_from_chi2 = runner.records["smooth"]["best_chi2"] - runner.records["subhalo"]["best_chi2"]
        runner.records["q_identity_error"] = abs(q - q_from_chi2)
        if runner.records["q_identity_error"] > runner.procedure["identity_tolerance"]:
            raise ValueError("Signed q disagrees between scalar likelihood and chi-squared")
        stable = all(runner.records[role].get("stable", False) for role in ("smooth", "subhalo"))
        in_band = abs(q - 10) <= runner.procedure["threshold_band"]
        runner.records["profile_decision"] = bool(q > 10) if stable and not in_band else None
        runner.records["numerical_status"] = (
            "stable_local_maxima" if stable else "unresolved_or_identity_only"
        )
        if not stable:
            result.quality_flags.append("profile_maximum_unresolved")
        if in_band:
            result.quality_flags.append("profile_threshold_unresolved")
        runner.records["q_signed"] = q
        runner.records["threshold_unresolved"] = abs(q - 10) <= runner.procedure["threshold_band"]
        atomic_json(Path(runner.output_dir) / "profile_result.json", runner.records)
        return result
