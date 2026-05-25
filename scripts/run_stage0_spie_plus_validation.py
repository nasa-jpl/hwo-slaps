#!/usr/bin/env python
"""Run SPIE-plus local nonlinear profile verification for Stage 0."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

SCRIPT_ROOT = REPO_ROOT / "scripts"
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))

import calibrate_stage0_profile_likelihood as profile_cal  # noqa: E402

from hwoslaps.modeling.nonlinear.local_profile import (  # noqa: E402
    LocalFitAttempt,
    LocalProfileFitResult,
    fit_local_least_squares_profile,
    profile_likelihood_q,
)


DEFAULT_RESULTS = REPO_ROOT / "outputs/stage0_internal_review/results.csv"
DEFAULT_CONFIG_DIR = REPO_ROOT / "outputs/stage0_internal_review/generated_configs"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "outputs/stage0_spie_plus_validation"
DEFAULT_INJECTED_CASES = (
    "stage0_internal_review_mass_m1e7_perfect",
    "stage0_internal_review_mass_m10p7p25_perfect",
    "stage0_internal_review_mass_m10p7p75_perfect",
    "stage0_internal_review_hexike_s0_n2_a100p0nm_m1e7",
)
FALSE_POSITIVE_CASE = "stage0_spie_plus_false_positive_hexike100_truth_fit_perfect"
FALSE_POSITIVE_TEMPLATE = "stage0_internal_review_hexike_s0_n2_a100p0nm_m1e7"
GLOBAL_FALSE_POSITIVE_CASE = "stage0_spie_plus_false_positive_global_zernike100_truth_fit_perfect"
GLOBAL_FALSE_POSITIVE_TEMPLATE = "stage0_internal_review_global_zernike_n4_a100p0nm_m1e7"
DEFAULT_FALSE_POSITIVE_TEMPLATES = (
    FALSE_POSITIVE_TEMPLATE,
    GLOBAL_FALSE_POSITIVE_TEMPLATE,
)
FALSE_POSITIVE_CASE_IDS = {
    FALSE_POSITIVE_TEMPLATE: FALSE_POSITIVE_CASE,
    GLOBAL_FALSE_POSITIVE_TEMPLATE: GLOBAL_FALSE_POSITIVE_CASE,
}

CSV_COLUMNS = (
    "case_id",
    "run_name",
    "case_kind",
    "status",
    "error",
    "runtime_s",
    "mass_msun",
    "truth_psf_case",
    "fit_psf_case",
    "psf_amplitude",
    "has_injected_subhalo",
    "pixels_unmasked",
    "n_nuisance",
    "q_fisher",
    "q_fit",
    "q_fit_over_q_fisher",
    "smooth_chi2_min",
    "subhalo_chi2_min",
    "smooth_best_start",
    "subhalo_best_start",
    "smooth_n_starts",
    "subhalo_n_starts",
    "smooth_convergence_rel_spread",
    "subhalo_convergence_rel_spread",
    "threshold_agreement",
    "ratio_agreement",
    "false_positive_pass",
    "spie_plus_pass",
    "smooth_reliability_note",
    "subhalo_reliability_note",
)


def _set_perfect_psf(config: Dict[str, Any]) -> None:
    aberr = config["psf"]["aberrations"]
    aberr["enable_segment_pistons"] = False
    aberr["enable_segment_tiptilts"] = False
    aberr["enable_segment_hexikes"] = False
    aberr["enable_global_zernikes"] = False
    aberr["segment_pistons"] = {}
    aberr["segment_tiptilts"] = {}
    aberr["segment_hexikes"] = {}
    aberr["global_zernikes"] = {}


def _write_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(profile_cal._json_safe(payload), handle, indent=2, sort_keys=True)


def _generate_psf(config: Dict[str, Any]) -> Any:
    from hwoslaps.psf import generate_psf_system

    return generate_psf_system(config["psf"], full_config=config)


def _generate_smooth_observation(config: Dict[str, Any], psf_data: Any) -> tuple[Any, Any]:
    from hwoslaps.lensing import generate_lensing_system
    from hwoslaps.observation import generate_observation

    smooth_config = deepcopy(config)
    smooth_config["lensing"]["subhalo"]["enabled"] = False
    lensing = generate_lensing_system(smooth_config["lensing"], full_config=smooth_config)
    observation = generate_observation(
        lensing_data=lensing,
        psf_data=psf_data,
        observation_config=smooth_config["observation"],
        full_config=smooth_config,
    )
    return lensing, observation


def _build_false_positive_inputs(config: Dict[str, Any]) -> tuple[Any, Any, Any, Any, np.ndarray]:
    """Build no-subhalo data with truth PSF and models with perfect fit PSF."""
    from hwoslaps.modeling.fisher_detector import FisherDetector

    truth_config = deepcopy(config)
    truth_config["lensing"]["subhalo"]["enabled"] = False
    fit_config = deepcopy(config)
    _set_perfect_psf(fit_config)

    psf_truth = _generate_psf(truth_config)
    psf_fit = _generate_psf(fit_config)
    lensing_truth, observation_truth = _generate_smooth_observation(truth_config, psf_truth)

    # The detector provides the Fisher nuisance definitions, mask, noise, and
    # fit-PSF forward-model helper. Its observation object intentionally carries
    # the truth-data mask/noise for the false-positive test.
    detector = FisherDetector(
        observation_baseline=observation_truth,
        lensing_baseline=lensing_truth,
        psf_data=psf_fit,
        full_config=fit_config,
        fisher_config=fit_config["modeling"]["fisher"],
    )
    data_mu = np.asarray(detector._mean_adu_from_observation(observation_truth), dtype=float)  # noqa: SLF001
    return detector, fit_config, observation_truth, psf_fit, data_mu


def _fisher_profile_coefficients(detector: Any, obs_test: Any) -> tuple[float, float, np.ndarray]:
    from hwoslaps.modeling.fisher_adapter import flatten_masked_image

    mu0 = np.asarray(detector.mu0_adu_2d, dtype=float)
    mu1 = np.asarray(detector._mean_adu_from_observation(obs_test), dtype=float)  # noqa: SLF001
    sigma = np.asarray(detector.sigma_adu_2d, dtype=float)
    mask = np.asarray(detector.mask_2d, dtype=bool)
    signal_flat = flatten_masked_image(mu1 - mu0, mask=mask)
    sigma_flat = flatten_masked_image(sigma, mask=mask)
    signal_whitened = signal_flat / np.maximum(sigma_flat, 1.0e-12)
    raw, profiled, coeffs, _residual_norm, _prior_penalty = detector.workspace._profiled_information(  # noqa: SLF001
        signal_whitened
    )
    return float(raw), float(profiled), np.asarray(coeffs, dtype=float)


def _coefficient_scales(detector: Any, base_config: Dict[str, Any], seeds: Sequence[np.ndarray]) -> np.ndarray:
    max_seed = np.zeros(len(detector.scalar_nuisance_specs), dtype=float)
    for seed in seeds:
        if seed.size:
            max_seed = np.maximum(max_seed, np.abs(seed))

    scales = []
    for idx, spec in enumerate(detector.scalar_nuisance_specs):
        seed_scale = float(max_seed[idx]) if idx < max_seed.size else 0.0
        if spec.path is None:
            scales.append(max(seed_scale, 0.05))
            continue
        base_value = abs(float(profile_cal._get_path_value(base_config, spec.path)))
        if spec.step_mode == "multiplicative":
            fd_scale = max(base_value * float(detector.finite_diff[spec.step_key]), 1.0e-8)
        else:
            fd_scale = max(float(detector.finite_diff[spec.step_key]), 1.0e-8)
        scales.append(max(seed_scale, fd_scale, 1.0e-8))
    return np.asarray(scales, dtype=float)


def _coefficient_bounds(detector: Any, base_config: Dict[str, Any], scales: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    lower = []
    upper = []
    for idx, spec in enumerate(detector.scalar_nuisance_specs):
        scale = float(scales[idx])
        if spec.path is None:
            lo, hi = -5.0, 5.0
        elif "centre" in spec.name:
            lo, hi = -0.05, 0.05
        elif spec.name == "lens.einstein_radius":
            lo, hi = -0.05, 0.05
        elif "ell_comp" in spec.name:
            lo, hi = -0.2, 0.2
        elif spec.name in {"source.intensity", "source.effective_radius"}:
            base = float(profile_cal._get_path_value(base_config, spec.path))
            lo, hi = -0.8 * base, 2.0 * base
        else:
            lo, hi = -np.inf, np.inf
        lower.append(lo / scale)
        upper.append(hi / scale)
    return np.asarray(lower, dtype=float), np.asarray(upper, dtype=float)


def _apply_scaled_coefficients(
    *,
    config: Dict[str, Any],
    detector: Any,
    x_scaled: np.ndarray,
    scales: np.ndarray,
) -> tuple[Dict[str, Any], float]:
    coeffs = np.asarray(x_scaled, dtype=float) * np.asarray(scales, dtype=float)
    profiled, background_offset, _coeff_by_name = profile_cal._apply_scalar_profile_coefficients(
        config=config,
        detector=detector,
        coeffs=coeffs,
    )
    return profiled, float(background_offset)


def _residual_function(
    *,
    detector: Any,
    fit_config: Dict[str, Any],
    data_mu: np.ndarray,
    has_subhalo: bool,
    scales: np.ndarray,
) -> Any:
    sigma = np.asarray(detector.sigma_adu_2d, dtype=float)
    mask = np.asarray(detector.mask_2d, dtype=bool)
    data = np.asarray(data_mu, dtype=float)
    n_masked = int(np.count_nonzero(mask))

    def residual(x_scaled: np.ndarray) -> np.ndarray:
        model_config = deepcopy(fit_config)
        model_config["lensing"]["subhalo"]["enabled"] = bool(has_subhalo)
        try:
            model_config, background_offset = _apply_scaled_coefficients(
                config=model_config,
                detector=detector,
                x_scaled=np.asarray(x_scaled, dtype=float),
                scales=scales,
            )
            model = profile_cal._mean_adu_for_config(
                config=model_config,
                detector=detector,
            )
            model = model + background_offset
            return (data - model)[mask] / np.maximum(sigma[mask], 1.0e-12)
        except Exception:
            return np.full(n_masked, 1.0e6, dtype=float)

    return residual


def _fit_model(
    *,
    model_name: str,
    detector: Any,
    fit_config: Dict[str, Any],
    data_mu: np.ndarray,
    has_subhalo: bool,
    initial_coefficients: Sequence[np.ndarray],
    initial_labels: Sequence[str],
    max_nfev: int,
    reliability_note: str,
) -> tuple[LocalProfileFitResult, np.ndarray]:
    scales = _coefficient_scales(detector, fit_config, initial_coefficients)
    lower, upper = _coefficient_bounds(detector, fit_config, scales)
    initial_scaled = [np.asarray(coeffs, dtype=float) / scales for coeffs in initial_coefficients]
    residual_fn = _residual_function(
        detector=detector,
        fit_config=fit_config,
        data_mu=data_mu,
        has_subhalo=has_subhalo,
        scales=scales,
    )
    result = fit_local_least_squares_profile(
        model_name=model_name,
        residual_fn=residual_fn,
        initial_points=initial_scaled,
        labels=initial_labels,
        lower_bounds=lower,
        upper_bounds=upper,
        max_nfev=max_nfev,
        reliability_note=reliability_note,
    )
    return result, scales


def _zero_coefficients(detector: Any) -> np.ndarray:
    return np.zeros(len(detector.scalar_nuisance_specs), dtype=float)


def _jitter_coefficients(
    detector: Any,
    base_config: Dict[str, Any],
    fraction: float = 0.25,
) -> np.ndarray:
    values = []
    for spec in detector.scalar_nuisance_specs:
        if spec.path is None:
            values.append(0.01 * float(fraction))
            continue
        if spec.step_mode == "multiplicative":
            base = abs(float(profile_cal._get_path_value(base_config, spec.path)))
            values.append(base * float(detector.finite_diff[spec.step_key]) * float(fraction))
        else:
            values.append(float(detector.finite_diff[spec.step_key]) * float(fraction))
    return np.asarray(values, dtype=float)


def run_injected_case(
    *,
    run_name: str,
    stage0_row: Dict[str, str],
    config_dir: Path,
    output_dir: Path,
    max_nfev: int,
    ratio_tolerance: float,
) -> Dict[str, Any]:
    from hwoslaps.modeling.fisher_detector import FisherDetector

    start = time.perf_counter()
    try:
        config = profile_cal._load_config(config_dir / f"{run_name}.yaml")
        psf_data, lensing_baseline, obs_baseline, _lensing_test, obs_test = profile_cal._build_case_inputs(config)
        detector = FisherDetector(
            observation_baseline=obs_baseline,
            lensing_baseline=lensing_baseline,
            psf_data=psf_data,
            full_config=config,
            fisher_config=config["modeling"]["fisher"],
        )
        if detector.masked_covariance is not None:
            raise NotImplementedError("SPIE-plus validation currently supports diagonal noise only.")

        _raw, q_fisher_recomputed, fisher_coeffs = _fisher_profile_coefficients(detector, obs_test)
        data_mu = np.asarray(detector._mean_adu_from_observation(obs_test), dtype=float)  # noqa: SLF001
        zero = _zero_coefficients(detector)

        smooth_fit, _smooth_scales = _fit_model(
            model_name="smooth",
            detector=detector,
            fit_config=config,
            data_mu=data_mu,
            has_subhalo=False,
            initial_coefficients=[fisher_coeffs, zero],
            initial_labels=["fisher_profile", "zero"],
            max_nfev=max_nfev,
            reliability_note="Two starts: Fisher-profile solution and unprofiled truth nuisance point.",
        )
        subhalo_fit, _subhalo_scales = _fit_model(
            model_name="subhalo",
            detector=detector,
            fit_config=config,
            data_mu=data_mu,
            has_subhalo=True,
            initial_coefficients=[zero],
            initial_labels=["truth"],
            max_nfev=max_nfev,
            reliability_note="One start is sufficient because the injected Asimov truth is inside the fixed-subhalo model and has zero residual.",
        )
        q_fit = profile_likelihood_q(
            smooth_chi2_min=smooth_fit.chi2_min,
            subhalo_chi2_min=subhalo_fit.chi2_min,
        )
        q_fisher = float(stage0_row["q_f"])
        ratio = q_fit / q_fisher if q_fisher > 0.0 else np.nan
        threshold_agreement = bool((q_fisher > 10.0) == (q_fit > 10.0))
        ratio_agreement = bool(abs(ratio - 1.0) <= ratio_tolerance)
        spie_plus_pass = bool(threshold_agreement and ratio_agreement)

        detail = {
            "case_id": run_name,
            "smooth_fit": smooth_fit.to_dict(),
            "subhalo_fit": subhalo_fit.to_dict(),
            "q_fisher_csv": q_fisher,
            "q_fisher_recomputed": q_fisher_recomputed,
            "q_fit": q_fit,
            "q_fit_over_q_fisher": ratio,
            "spie_plus_pass": spie_plus_pass,
        }
        _write_json(output_dir / f"{run_name}.json", detail)

        return _row(
            case_id=run_name,
            run_name=run_name,
            case_kind="injected_subhalo",
            status="success",
            error=None,
            runtime_s=time.perf_counter() - start,
            mass_msun=float(stage0_row["mass_msun"]),
            truth_psf_case=stage0_row["psf_case"],
            fit_psf_case=stage0_row["psf_case"],
            psf_amplitude=float(stage0_row["psf_amplitude"]),
            has_injected_subhalo=True,
            pixels_unmasked=int(detector.pixels_unmasked),
            n_nuisance=int(detector.n_nuisance),
            q_fisher=q_fisher,
            q_fit=q_fit,
            q_fit_over_q_fisher=ratio,
            smooth_fit=smooth_fit,
            subhalo_fit=subhalo_fit,
            threshold_agreement=threshold_agreement,
            ratio_agreement=ratio_agreement,
            false_positive_pass=None,
            spie_plus_pass=spie_plus_pass,
        )
    except Exception as exc:  # pragma: no cover - runtime diagnostics.
        return _error_row(
            case_id=run_name,
            run_name=run_name,
            case_kind="injected_subhalo",
            error=repr(exc),
            runtime_s=time.perf_counter() - start,
            stage0_row=stage0_row,
        )


def run_false_positive_case(
    *,
    stage0_row: Dict[str, str],
    template_run_name: str,
    case_id: str,
    config_dir: Path,
    output_dir: Path,
    max_nfev: int,
) -> Dict[str, Any]:
    start = time.perf_counter()
    try:
        config = profile_cal._load_config(config_dir / f"{template_run_name}.yaml")
        detector, fit_config, _observation_truth, _psf_fit, data_mu = _build_false_positive_inputs(config)
        zero = _zero_coefficients(detector)
        jitter = _jitter_coefficients(detector, fit_config, fraction=0.25)
        smooth_fit, _smooth_scales = _fit_model(
            model_name="smooth",
            detector=detector,
            fit_config=fit_config,
            data_mu=data_mu,
            has_subhalo=False,
            initial_coefficients=[zero, jitter],
            initial_labels=["zero", "positive_jitter"],
            max_nfev=max_nfev,
            reliability_note="Two starts test convergence under deliberate truth-fit PSF mismatch.",
        )
        subhalo_fit, _subhalo_scales = _fit_model(
            model_name="subhalo",
            detector=detector,
            fit_config=fit_config,
            data_mu=data_mu,
            has_subhalo=True,
            initial_coefficients=[zero, jitter],
            initial_labels=["zero", "positive_jitter"],
            max_nfev=max_nfev,
            reliability_note="Two starts test whether a fixed 1e7 Msun subhalo spuriously absorbs PSF mismatch.",
        )
        q_fit = profile_likelihood_q(
            smooth_chi2_min=smooth_fit.chi2_min,
            subhalo_chi2_min=subhalo_fit.chi2_min,
        )
        false_positive_pass = bool(q_fit < 10.0)
        truth_psf_case = str(stage0_row["psf_case"])
        detail = {
            "case_id": case_id,
            "template_run": template_run_name,
            "smooth_fit": smooth_fit.to_dict(),
            "subhalo_fit": subhalo_fit.to_dict(),
            "q_fit": q_fit,
            "false_positive_pass": false_positive_pass,
            "truth_psf_case": truth_psf_case,
            "fit_psf_case": "perfect",
        }
        _write_json(output_dir / f"{case_id}.json", detail)

        return _row(
            case_id=case_id,
            run_name=template_run_name,
            case_kind="false_positive_psf_mismatch",
            status="success",
            error=None,
            runtime_s=time.perf_counter() - start,
            mass_msun=float(stage0_row["mass_msun"]),
            truth_psf_case=truth_psf_case,
            fit_psf_case="perfect",
            psf_amplitude=float(stage0_row["psf_amplitude"]),
            has_injected_subhalo=False,
            pixels_unmasked=int(detector.pixels_unmasked),
            n_nuisance=int(detector.n_nuisance),
            q_fisher=None,
            q_fit=q_fit,
            q_fit_over_q_fisher=None,
            smooth_fit=smooth_fit,
            subhalo_fit=subhalo_fit,
            threshold_agreement=None,
            ratio_agreement=None,
            false_positive_pass=false_positive_pass,
            spie_plus_pass=false_positive_pass,
        )
    except Exception as exc:  # pragma: no cover - runtime diagnostics.
        return _error_row(
            case_id=case_id,
            run_name=template_run_name,
            case_kind="false_positive_psf_mismatch",
            error=repr(exc),
            runtime_s=time.perf_counter() - start,
            stage0_row=stage0_row,
        )


def _row(
    *,
    case_id: str,
    run_name: str,
    case_kind: str,
    status: str,
    error: Optional[str],
    runtime_s: float,
    mass_msun: float,
    truth_psf_case: str,
    fit_psf_case: str,
    psf_amplitude: float,
    has_injected_subhalo: bool,
    pixels_unmasked: Optional[int],
    n_nuisance: Optional[int],
    q_fisher: Optional[float],
    q_fit: Optional[float],
    q_fit_over_q_fisher: Optional[float],
    smooth_fit: Optional[LocalProfileFitResult],
    subhalo_fit: Optional[LocalProfileFitResult],
    threshold_agreement: Optional[bool],
    ratio_agreement: Optional[bool],
    false_positive_pass: Optional[bool],
    spie_plus_pass: bool,
) -> Dict[str, Any]:
    return {
        "case_id": case_id,
        "run_name": run_name,
        "case_kind": case_kind,
        "status": status,
        "error": error,
        "runtime_s": runtime_s,
        "mass_msun": mass_msun,
        "truth_psf_case": truth_psf_case,
        "fit_psf_case": fit_psf_case,
        "psf_amplitude": psf_amplitude,
        "has_injected_subhalo": has_injected_subhalo,
        "pixels_unmasked": pixels_unmasked,
        "n_nuisance": n_nuisance,
        "q_fisher": q_fisher,
        "q_fit": q_fit,
        "q_fit_over_q_fisher": q_fit_over_q_fisher,
        "smooth_chi2_min": None if smooth_fit is None else smooth_fit.chi2_min,
        "subhalo_chi2_min": None if subhalo_fit is None else subhalo_fit.chi2_min,
        "smooth_best_start": None if smooth_fit is None else smooth_fit.best.label,
        "subhalo_best_start": None if subhalo_fit is None else subhalo_fit.best.label,
        "smooth_n_starts": None if smooth_fit is None else len(smooth_fit.attempts),
        "subhalo_n_starts": None if subhalo_fit is None else len(subhalo_fit.attempts),
        "smooth_convergence_rel_spread": None if smooth_fit is None else smooth_fit.convergence_rel_spread,
        "subhalo_convergence_rel_spread": None if subhalo_fit is None else subhalo_fit.convergence_rel_spread,
        "threshold_agreement": threshold_agreement,
        "ratio_agreement": ratio_agreement,
        "false_positive_pass": false_positive_pass,
        "spie_plus_pass": spie_plus_pass,
        "smooth_reliability_note": None if smooth_fit is None else smooth_fit.reliability_note,
        "subhalo_reliability_note": None if subhalo_fit is None else subhalo_fit.reliability_note,
    }


def _error_row(
    *,
    case_id: str,
    run_name: str,
    case_kind: str,
    error: str,
    runtime_s: float,
    stage0_row: Dict[str, str],
) -> Dict[str, Any]:
    return _row(
        case_id=case_id,
        run_name=run_name,
        case_kind=case_kind,
        status="error",
        error=error,
        runtime_s=runtime_s,
        mass_msun=float(stage0_row.get("mass_msun", "nan")),
        truth_psf_case=stage0_row.get("psf_case", ""),
        fit_psf_case=stage0_row.get("psf_case", ""),
        psf_amplitude=float(stage0_row.get("psf_amplitude", "nan")),
        has_injected_subhalo=case_kind != "false_positive_psf_mismatch",
        pixels_unmasked=None,
        n_nuisance=None,
        q_fisher=float(stage0_row.get("q_f", "nan")) if case_kind != "false_positive_psf_mismatch" else None,
        q_fit=None,
        q_fit_over_q_fisher=None,
        smooth_fit=None,
        subhalo_fit=None,
        threshold_agreement=False,
        ratio_agreement=False,
        false_positive_pass=False if case_kind == "false_positive_psf_mismatch" else None,
        spie_plus_pass=False,
    )


def _make_plot(output_dir: Path, rows: Sequence[Dict[str, Any]]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    injected = [
        row for row in rows
        if row["status"] == "success" and row["case_kind"] == "injected_subhalo"
    ]
    if not injected:
        return

    q_f = np.asarray([float(row["q_fisher"]) for row in injected], dtype=float)
    q_fit = np.asarray([float(row["q_fit"]) for row in injected], dtype=float)
    max_q = float(max(np.max(q_f), np.max(q_fit), 10.0))
    fig, ax = plt.subplots(figsize=(5.0, 4.5), constrained_layout=True)
    ax.scatter(q_f, q_fit, color="#26547c", s=55, zorder=3)
    ax.plot([0.0, max_q * 1.05], [0.0, max_q * 1.05], color="#333333", linewidth=1.5)
    ax.axvline(10.0, color="#b23a48", linestyle="--", linewidth=1.0)
    ax.axhline(10.0, color="#b23a48", linestyle="--", linewidth=1.0)
    ax.set_xlabel(r"Fisher profile statistic $q_F$")
    ax.set_ylabel(r"Local nonlinear profile statistic $q_\mathrm{fit}$")
    ax.set_xlim(0.0, max_q * 1.05)
    ax.set_ylim(0.0, max_q * 1.05)
    ax.grid(alpha=0.25)
    fig.savefig(output_dir / "q_f_vs_q_fit.png", dpi=180)
    plt.close(fig)


def _run_cases(
    *,
    stage0_rows: Dict[str, Dict[str, str]],
    config_dir: Path,
    output_dir: Path,
    cases: Sequence[str],
    include_false_positive: bool,
    false_positive_templates: Sequence[str],
    max_nfev: int,
    ratio_tolerance: float,
    workers: int,
) -> List[Dict[str, Any]]:
    tasks = [("injected", run_name) for run_name in cases]
    if include_false_positive:
        tasks.extend(("false_positive", template) for template in false_positive_templates)

    if workers <= 1:
        rows = []
        for kind, run_name in tasks:
            if kind == "injected":
                rows.append(
                    run_injected_case(
                        run_name=run_name,
                        stage0_row=stage0_rows[run_name],
                        config_dir=config_dir,
                        output_dir=output_dir,
                        max_nfev=max_nfev,
                        ratio_tolerance=ratio_tolerance,
                    )
                )
            else:
                case_id = FALSE_POSITIVE_CASE_IDS.get(
                    run_name,
                    f"stage0_spie_plus_false_positive_{run_name}_truth_fit_perfect",
                )
                rows.append(
                    run_false_positive_case(
                        stage0_row=stage0_rows[run_name],
                        template_run_name=run_name,
                        case_id=case_id,
                        config_dir=config_dir,
                        output_dir=output_dir,
                        max_nfev=max_nfev,
                    )
                )
        return rows

    rows: List[Dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {}
        for kind, run_name in tasks:
            if kind == "injected":
                future = executor.submit(
                    run_injected_case,
                    run_name=run_name,
                    stage0_row=stage0_rows[run_name],
                    config_dir=config_dir,
                    output_dir=output_dir,
                    max_nfev=max_nfev,
                    ratio_tolerance=ratio_tolerance,
                )
            else:
                case_id = FALSE_POSITIVE_CASE_IDS.get(
                    run_name,
                    f"stage0_spie_plus_false_positive_{run_name}_truth_fit_perfect",
                )
                future = executor.submit(
                    run_false_positive_case,
                    stage0_row=stage0_rows[run_name],
                    template_run_name=run_name,
                    case_id=case_id,
                    config_dir=config_dir,
                    output_dir=output_dir,
                    max_nfev=max_nfev,
                )
            futures[future] = (kind, run_name)
        for future in as_completed(futures):
            rows.append(future.result())
    order = {run_name: idx for idx, (_kind, run_name) in enumerate(tasks)}
    rows.sort(key=lambda row: order.get(row["case_id"], 999))
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage0-results", default=str(DEFAULT_RESULTS))
    parser.add_argument("--config-dir", default=str(DEFAULT_CONFIG_DIR))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--cases", nargs="*", default=list(DEFAULT_INJECTED_CASES))
    parser.add_argument("--include-false-positive", action="store_true", default=True)
    parser.add_argument("--no-false-positive", dest="include_false_positive", action="store_false")
    parser.add_argument("--false-positive-cases", nargs="*", default=list(DEFAULT_FALSE_POSITIVE_TEMPLATES))
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--max-nfev", type=int, default=12)
    parser.add_argument("--ratio-tolerance", type=float, default=0.2)
    return parser.parse_args()


def main() -> int:
    os.environ.setdefault("HWOSLAPS_DISABLE_TQDM", "1")
    args = parse_args()
    stage0_rows = profile_cal._read_stage0_rows(Path(args.stage0_results).resolve())
    missing = [case for case in args.cases if case not in stage0_rows]
    if args.include_false_positive:
        missing.extend(case for case in args.false_positive_cases if case not in stage0_rows)
    if missing:
        raise KeyError(f"Cases missing from Stage 0 results: {missing}")

    output_dir = Path(args.output_dir).resolve()
    rows = _run_cases(
        stage0_rows=stage0_rows,
        config_dir=Path(args.config_dir).resolve(),
        output_dir=output_dir,
        cases=args.cases,
        include_false_positive=bool(args.include_false_positive),
        false_positive_templates=list(args.false_positive_cases),
        max_nfev=max(1, int(args.max_nfev)),
        ratio_tolerance=float(args.ratio_tolerance),
        workers=max(1, int(args.workers)),
    )
    _write_csv(output_dir / "results.csv", rows)
    _make_plot(output_dir, rows)
    failures = [row for row in rows if row["status"] != "success" or not row["spie_plus_pass"]]
    print(f"Wrote SPIE-plus validation to {output_dir / 'results.csv'}")
    print(f"SPIE-plus failures: {len(failures)} / {len(rows)}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
