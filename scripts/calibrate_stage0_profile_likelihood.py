#!/usr/bin/env python
"""Calibrate Stage 0 Fisher metrics with local profiled forward-model checks.

This script does not run a global nonlinear sampler.  It uses the Fisher
profile solution to move the smooth lens/source model in the nuisance
directions, then evaluates the full HWO-SLAPS nonlinear forward model at that
profiled point.  The resulting chi-squared should match the profiled Fisher
``q_F`` when the local linearization is valid.
"""

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
from typing import Any, Dict, Iterable, List, Optional, Sequence

import numpy as np
import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

DEFAULT_RESULTS = REPO_ROOT / "outputs/stage0_internal_review/results.csv"
DEFAULT_CONFIG_DIR = REPO_ROOT / "outputs/stage0_internal_review/generated_configs"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "outputs/stage0_profile_calibration"
DEFAULT_CASES = (
    "stage0_internal_review_mass_m1e7_perfect",
    "stage0_internal_review_mass_m10p7p25_perfect",
    "stage0_internal_review_mass_m10p7p75_perfect",
    "stage0_internal_review_hexike_s0_n2_a100p0nm_m1e7",
)

CSV_COLUMNS = (
    "run_name",
    "status",
    "error",
    "runtime_s",
    "mass_msun",
    "psf_case",
    "psf_amplitude",
    "pixels_unmasked",
    "n_nuisance",
    "q_fisher_csv",
    "q_fisher_recomputed",
    "q_raw",
    "q_linear_profiled_pixel_residual",
    "q_nonlinear_at_fisher_profile",
    "q_subhalo_truth",
    "q_nonlinear_over_q_fisher",
    "abs_diff_q",
    "rel_diff_q",
    "max_abs_nuisance_coeff",
    "background_offset_adu",
    "alignment_pass",
)


def _read_stage0_rows(path: Path) -> Dict[str, Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return {row["run_name"]: row for row in csv.DictReader(handle)}


def _load_config(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    config.setdefault("plotting", {})["enabled"] = False
    return config


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def _write_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def _json_safe(value: Any) -> Any:
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


def _get_path_value(config: Dict[str, Any], path: Sequence[Any]) -> Any:
    node: Any = config
    for key in path:
        node = node[key]
    return node


def _set_path_value(config: Dict[str, Any], path: Sequence[Any], value: Any) -> None:
    node: Any = config
    for key in path[:-1]:
        node = node[key]
    node[path[-1]] = value


def _chi2(
    data: np.ndarray,
    model: np.ndarray,
    sigma: np.ndarray,
    mask: np.ndarray,
) -> float:
    residual = (np.asarray(data, dtype=float) - np.asarray(model, dtype=float))[mask]
    sigma_masked = np.asarray(sigma, dtype=float)[mask]
    return float(np.sum((residual / np.maximum(sigma_masked, 1.0e-12)) ** 2))


def _masked_chi2_from_flat_residual(residual: np.ndarray, sigma: np.ndarray) -> float:
    return float(np.sum((np.asarray(residual, dtype=float) / np.maximum(sigma, 1.0e-12)) ** 2))


def _apply_scalar_profile_coefficients(
    *,
    config: Dict[str, Any],
    detector: Any,
    coeffs: np.ndarray,
) -> tuple[Dict[str, Any], float, Dict[str, float]]:
    """Apply Fisher best-fit scalar nuisance coefficients to a config."""
    if detector.n_psf_fit_modes:
        raise ValueError(
            "This Stage 0 calibration script currently handles scalar nuisance "
            "profiling only. Disable PSF nuisance fitting or extend PSF coefficient "
            "application before using include_psf_nuisance=True."
        )

    profiled = deepcopy(config)
    background_offset_adu = 0.0
    coeff_by_name: Dict[str, float] = {}

    for spec, coeff in zip(detector.scalar_nuisance_specs, coeffs):
        coeff_float = float(coeff)
        coeff_by_name[spec.name] = coeff_float
        if spec.path is None:
            background_offset_adu += coeff_float
            continue
        base_value = float(_get_path_value(profiled, spec.path))
        _set_path_value(profiled, spec.path, base_value + coeff_float)

    return profiled, float(background_offset_adu), coeff_by_name


def _mean_adu_for_config(
    *,
    config: Dict[str, Any],
    detector: Any,
) -> np.ndarray:
    from hwoslaps.lensing import generate_lensing_system

    lensing_data = generate_lensing_system(config["lensing"], full_config=config)
    return detector._mean_adu_from_lensing(  # noqa: SLF001 - calibration must match Fisher implementation.
        lensing_data=lensing_data,
        observation_config=config["observation"],
    )


def _build_case_inputs(config: Dict[str, Any]) -> tuple[Any, Any, Any, Any, Any]:
    from hwoslaps.config.validation import validate_or_raise
    from hwoslaps.lensing import generate_lensing_system
    from hwoslaps.observation import generate_observation
    from hwoslaps.psf import generate_psf_system

    validate_or_raise(config)
    psf_data = generate_psf_system(config["psf"], full_config=config)

    baseline_config = deepcopy(config)
    baseline_config["lensing"]["subhalo"]["enabled"] = False
    test_config = deepcopy(config)
    test_config["lensing"]["subhalo"]["enabled"] = True

    lensing_baseline = generate_lensing_system(
        baseline_config["lensing"],
        full_config=baseline_config,
    )
    observation_baseline = generate_observation(
        lensing_data=lensing_baseline,
        psf_data=psf_data,
        observation_config=baseline_config["observation"],
        full_config=baseline_config,
    )
    lensing_test = generate_lensing_system(test_config["lensing"], full_config=test_config)
    observation_test = generate_observation(
        lensing_data=lensing_test,
        psf_data=psf_data,
        observation_config=test_config["observation"],
        full_config=test_config,
    )
    return psf_data, lensing_baseline, observation_baseline, lensing_test, observation_test


def calibrate_case(
    *,
    run_name: str,
    stage0_row: Dict[str, str],
    config_dir: Path,
    output_dir: Path,
    rel_tolerance: float,
) -> Dict[str, Any]:
    start = time.perf_counter()
    os.environ.setdefault("HWOSLAPS_DISABLE_TQDM", "1")

    from hwoslaps.modeling.fisher_adapter import flatten_masked_image, stack_masked_images
    from hwoslaps.modeling.fisher_detector import FisherDetector

    try:
        config = _load_config(config_dir / f"{run_name}.yaml")
        psf_data, lensing_baseline, obs_baseline, _lensing_test, obs_test = _build_case_inputs(config)

        detector = FisherDetector(
            observation_baseline=obs_baseline,
            lensing_baseline=lensing_baseline,
            psf_data=psf_data,
            full_config=config,
            fisher_config=config["modeling"]["fisher"],
        )
        if detector.masked_covariance is not None:
            raise NotImplementedError(
                "Stage 0 profile calibration currently supports diagonal noise only."
            )

        mu0 = np.asarray(detector.mu0_adu_2d, dtype=float)
        mu1 = np.asarray(detector._mean_adu_from_observation(obs_test), dtype=float)  # noqa: SLF001
        sigma = np.asarray(detector.sigma_adu_2d, dtype=float)
        mask = np.asarray(detector.mask_2d, dtype=bool)

        signal_flat = flatten_masked_image(mu1 - mu0, mask=mask)
        sigma_flat = flatten_masked_image(sigma, mask=mask)
        signal_whitened = signal_flat / np.maximum(sigma_flat, 1.0e-12)

        raw, profiled, coeffs, residual_norm, prior_penalty = detector.workspace._profiled_information(  # noqa: SLF001
            signal_whitened
        )

        if detector.nuisance_images:
            nuisance_design = stack_masked_images(detector.nuisance_images, mask=mask)
            linear_profile_flat = (
                flatten_masked_image(mu0, mask=mask) + nuisance_design @ coeffs
            )
            linear_profile_residual = flatten_masked_image(mu1, mask=mask) - linear_profile_flat
            q_linear_pixel = _masked_chi2_from_flat_residual(linear_profile_residual, sigma_flat)
        else:
            q_linear_pixel = raw

        smooth_profile_config = deepcopy(config)
        smooth_profile_config["lensing"]["subhalo"]["enabled"] = False
        smooth_profile_config, background_offset_adu, coeff_by_name = (
            _apply_scalar_profile_coefficients(
                config=smooth_profile_config,
                detector=detector,
                coeffs=coeffs,
            )
        )
        smooth_profile_mu = _mean_adu_for_config(
            config=smooth_profile_config,
            detector=detector,
        )
        smooth_profile_mu = smooth_profile_mu + background_offset_adu

        subhalo_truth_config = deepcopy(config)
        subhalo_truth_config["lensing"]["subhalo"]["enabled"] = True
        subhalo_truth_mu = _mean_adu_for_config(
            config=subhalo_truth_config,
            detector=detector,
        )

        q_nonlinear = _chi2(mu1, smooth_profile_mu, sigma, mask)
        q_subhalo_truth = _chi2(mu1, subhalo_truth_mu, sigma, mask)
        q_profile_delta = max(0.0, q_nonlinear - q_subhalo_truth)
        q_fisher_csv = float(stage0_row["q_f"])
        ratio = q_profile_delta / profiled if profiled > 0.0 else np.nan
        abs_diff = abs(q_profile_delta - profiled)
        rel_diff = abs_diff / max(abs(profiled), 1.0e-12)
        alignment_pass = bool(rel_diff <= rel_tolerance)

        detail = {
            "run_name": run_name,
            "coefficients": coeff_by_name,
            "q_raw_recomputed": float(raw),
            "q_fisher_recomputed": float(profiled),
            "q_linear_profiled_pixel_residual": float(q_linear_pixel),
            "q_linear_profiled_prior_penalty": float(prior_penalty),
            "q_linear_profiled_residual_norm": float(residual_norm),
            "q_nonlinear_at_fisher_profile": float(q_profile_delta),
            "q_subhalo_truth": float(q_subhalo_truth),
            "background_offset_adu": float(background_offset_adu),
            "alignment_relative_tolerance": float(rel_tolerance),
            "alignment_pass": alignment_pass,
        }
        _write_json(output_dir / f"{run_name}.json", _json_safe(detail))

        return {
            "run_name": run_name,
            "status": "success",
            "error": None,
            "runtime_s": time.perf_counter() - start,
            "mass_msun": float(stage0_row["mass_msun"]),
            "psf_case": stage0_row["psf_case"],
            "psf_amplitude": float(stage0_row["psf_amplitude"]),
            "pixels_unmasked": int(detector.pixels_unmasked),
            "n_nuisance": int(detector.n_nuisance),
            "q_fisher_csv": q_fisher_csv,
            "q_fisher_recomputed": float(profiled),
            "q_raw": float(raw),
            "q_linear_profiled_pixel_residual": float(q_linear_pixel),
            "q_nonlinear_at_fisher_profile": float(q_profile_delta),
            "q_subhalo_truth": float(q_subhalo_truth),
            "q_nonlinear_over_q_fisher": float(ratio),
            "abs_diff_q": float(abs_diff),
            "rel_diff_q": float(rel_diff),
            "max_abs_nuisance_coeff": float(np.max(np.abs(coeffs))) if coeffs.size else 0.0,
            "background_offset_adu": float(background_offset_adu),
            "alignment_pass": alignment_pass,
        }
    except Exception as exc:  # pragma: no cover - exercised by runtime failures.
        return {
            "run_name": run_name,
            "status": "error",
            "error": repr(exc),
            "runtime_s": time.perf_counter() - start,
            "mass_msun": float(stage0_row.get("mass_msun", "nan")),
            "psf_case": stage0_row.get("psf_case"),
            "psf_amplitude": float(stage0_row.get("psf_amplitude", "nan")),
            "pixels_unmasked": None,
            "n_nuisance": None,
            "q_fisher_csv": float(stage0_row.get("q_f", "nan")),
            "q_fisher_recomputed": None,
            "q_raw": None,
            "q_linear_profiled_pixel_residual": None,
            "q_nonlinear_at_fisher_profile": None,
            "q_subhalo_truth": None,
            "q_nonlinear_over_q_fisher": None,
            "abs_diff_q": None,
            "rel_diff_q": None,
            "max_abs_nuisance_coeff": None,
            "background_offset_adu": None,
            "alignment_pass": False,
        }


def _run_cases(
    *,
    cases: Sequence[str],
    stage0_rows: Dict[str, Dict[str, str]],
    config_dir: Path,
    output_dir: Path,
    rel_tolerance: float,
    workers: int,
) -> List[Dict[str, Any]]:
    if workers <= 1:
        return [
            calibrate_case(
                run_name=run_name,
                stage0_row=stage0_rows[run_name],
                config_dir=config_dir,
                output_dir=output_dir,
                rel_tolerance=rel_tolerance,
            )
            for run_name in cases
        ]

    rows: List[Dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(
                calibrate_case,
                run_name=run_name,
                stage0_row=stage0_rows[run_name],
                config_dir=config_dir,
                output_dir=output_dir,
                rel_tolerance=rel_tolerance,
            ): run_name
            for run_name in cases
        }
        for future in as_completed(futures):
            rows.append(future.result())
    rows.sort(key=lambda row: cases.index(row["run_name"]))
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage0-results", default=str(DEFAULT_RESULTS))
    parser.add_argument("--config-dir", default=str(DEFAULT_CONFIG_DIR))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--cases", nargs="*", default=list(DEFAULT_CASES))
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--rel-tolerance", type=float, default=0.05)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    stage0_results = Path(args.stage0_results).resolve()
    config_dir = Path(args.config_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    stage0_rows = _read_stage0_rows(stage0_results)

    missing = [case for case in args.cases if case not in stage0_rows]
    if missing:
        raise KeyError(f"Cases missing from Stage 0 results: {missing}")

    rows = _run_cases(
        cases=args.cases,
        stage0_rows=stage0_rows,
        config_dir=config_dir,
        output_dir=output_dir,
        rel_tolerance=float(args.rel_tolerance),
        workers=max(1, int(args.workers)),
    )
    _write_csv(output_dir / "results.csv", rows)
    failures = [row for row in rows if row["status"] != "success" or not row["alignment_pass"]]
    print(f"Wrote profile calibration to {output_dir / 'results.csv'}")
    print(f"Alignment failures: {len(failures)} / {len(rows)}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
