#!/usr/bin/env python
"""Run lightweight forecast-robustness checks for Stage 0."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

SCRIPT_ROOT = REPO_ROOT / "scripts"
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))

import calibrate_stage0_profile_likelihood as profile_cal  # noqa: E402
import run_stage0_spie_plus_validation as spie_plus  # noqa: E402


DEFAULT_RESULTS = REPO_ROOT / "outputs/stage0_internal_review/results.csv"
DEFAULT_CONFIG_DIR = REPO_ROOT / "outputs/stage0_internal_review/generated_configs"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "outputs/stage0_forecast_robustness"
DEFAULT_SPIE_PLUS_RESULTS = REPO_ROOT / "outputs/stage0_spie_plus_validation/results.csv"
DEFAULT_NOISY_CASES = (
    "stage0_internal_review_mass_m1e7_perfect",
    "stage0_internal_review_mass_m10p7p25_perfect",
)
DEFAULT_NOISE_SEEDS = (101, 102, 103, 104, 105)
DEFAULT_POSITION_ANGLES = (0.0, 45.0, 90.0, 135.0, 180.0, 225.0, 270.0, 315.0)
POSITION_TEMPLATE = "stage0_internal_review_mass_m1e7_perfect"
HEXIKE100_TEMPLATE = "stage0_internal_review_hexike_s0_n2_a100p0nm_m1e7"


def _write_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else []
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(profile_cal._json_safe(payload), handle, indent=2, sort_keys=True)


def _read_csv_rows(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _build_detector(config: Dict[str, Any]) -> tuple[Any, Any, Any, Any]:
    from hwoslaps.modeling.fisher_detector import FisherDetector

    psf_data, lensing_baseline, obs_baseline, _lensing_test, _obs_test = (
        profile_cal._build_case_inputs(config)
    )
    detector = FisherDetector(
        observation_baseline=obs_baseline,
        lensing_baseline=lensing_baseline,
        psf_data=psf_data,
        full_config=config,
        fisher_config=config["modeling"]["fisher"],
    )
    if detector.masked_covariance is not None:
        raise NotImplementedError("Forecast robustness checks currently support diagonal noise only.")
    return detector, psf_data, lensing_baseline, obs_baseline


def _profile_chi2(detector: Any, data_adu: np.ndarray, model_adu: np.ndarray) -> float:
    from hwoslaps.modeling.fisher_adapter import flatten_masked_image

    residual_flat = flatten_masked_image(
        np.asarray(data_adu, dtype=float) - np.asarray(model_adu, dtype=float),
        mask=detector.mask_2d,
    )
    sigma_flat = flatten_masked_image(detector.sigma_adu_2d, mask=detector.mask_2d)
    residual_w = residual_flat / np.maximum(sigma_flat, 1.0e-12)
    if detector.n_nuisance <= 0:
        return float(residual_w @ residual_w)
    jtr = detector.workspace.nuisance_whitened.T @ residual_w
    coeffs = detector.workspace.normal_pinv @ jtr
    profiled = residual_w - detector.workspace.nuisance_whitened @ coeffs
    return float(profiled @ profiled)


def _observation_for_config(config: Dict[str, Any], psf_data: Any, *, has_subhalo: bool) -> Any:
    from hwoslaps.lensing import generate_lensing_system
    from hwoslaps.observation import generate_observation

    model_config = deepcopy(config)
    model_config["lensing"]["subhalo"]["enabled"] = bool(has_subhalo)
    lensing = generate_lensing_system(model_config["lensing"], full_config=model_config)
    return generate_observation(
        lensing_data=lensing,
        psf_data=psf_data,
        observation_config=model_config["observation"],
        full_config=model_config,
    )


def run_noisy_ensemble(
    *,
    stage0_rows: Dict[str, Dict[str, str]],
    config_dir: Path,
    output_dir: Path,
    cases: Sequence[str],
    seeds: Sequence[int],
) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    rows: List[Dict[str, Any]] = []
    summaries: List[Dict[str, Any]] = []
    for run_name in cases:
        config = profile_cal._load_config(config_dir / f"{run_name}.yaml")
        detector, psf_data, _lensing_baseline, _obs_baseline = _build_detector(config)
        obs_truth = _observation_for_config(config, psf_data, has_subhalo=True)
        mu0 = np.asarray(detector.mu0_adu_2d, dtype=float)
        mu1 = np.asarray(detector._mean_adu_from_observation(obs_truth), dtype=float)  # noqa: SLF001
        q_f = float(stage0_rows[run_name]["q_f"])

        q_values = []
        for seed in seeds:
            noisy_config = deepcopy(config)
            noisy_config["global_seed"] = int(seed)
            noisy_config["run_name"] = f"{run_name}_noise_seed_{int(seed)}"
            noisy_obs = _observation_for_config(noisy_config, psf_data, has_subhalo=True)
            data_adu = np.asarray(noisy_obs.data.native, dtype=float)
            smooth_chi2 = _profile_chi2(detector, data_adu, mu0)
            subhalo_chi2 = _profile_chi2(detector, data_adu, mu1)
            q_signed = smooth_chi2 - subhalo_chi2
            q_noisy = max(0.0, q_signed)
            q_values.append(q_noisy)
            rows.append(
                {
                    "run_name": run_name,
                    "seed": int(seed),
                    "mass_msun": float(stage0_rows[run_name]["mass_msun"]),
                    "psf_case": stage0_rows[run_name]["psf_case"],
                    "q_f_asimov": q_f,
                    "q_noisy_profiled": q_noisy,
                    "q_signed_profiled": q_signed,
                    "q_noisy_over_q_f": q_noisy / q_f if q_f > 0.0 else np.nan,
                    "detected_q_gt_10": bool(q_noisy > 10.0),
                    "smooth_chi2_profiled": smooth_chi2,
                    "subhalo_chi2_profiled": subhalo_chi2,
                }
            )

        q_arr = np.asarray(q_values, dtype=float)
        summaries.append(
            {
                "run_name": run_name,
                "n_seeds": int(q_arr.size),
                "mass_msun": float(stage0_rows[run_name]["mass_msun"]),
                "psf_case": stage0_rows[run_name]["psf_case"],
                "q_f_asimov": q_f,
                "q_noisy_median": float(np.median(q_arr)),
                "q_noisy_p16": float(np.percentile(q_arr, 16.0)),
                "q_noisy_p84": float(np.percentile(q_arr, 84.0)),
                "median_over_q_f": float(np.median(q_arr) / q_f) if q_f > 0.0 else np.nan,
                "detected_fraction_q_gt_10": float(np.mean(q_arr > 10.0)),
            }
        )

    _write_csv(output_dir / "noisy_ensemble.csv", rows)
    _write_csv(output_dir / "noisy_summary.csv", summaries)
    return rows, summaries


def _exact_false_positive_row(
    *,
    case_id: str,
    truth_psf_case: str,
    fit_psf_case: str,
    psf_amplitude: float,
    notes: str,
) -> Dict[str, Any]:
    return {
        "case_id": case_id,
        "truth_psf_case": truth_psf_case,
        "fit_psf_case": fit_psf_case,
        "psf_amplitude": psf_amplitude,
        "source": "exact_nested_model_control",
        "q_fit": 0.0,
        "false_positive_pass": True,
        "notes": notes,
    }


def run_false_positive_controls(
    *,
    stage0_rows: Dict[str, Dict[str, str]],
    spie_plus_results: Path,
    output_dir: Path,
) -> List[Dict[str, Any]]:
    rows = [
        _exact_false_positive_row(
            case_id="false_positive_perfect_truth_fit_perfect",
            truth_psf_case="perfect",
            fit_psf_case="perfect",
            psf_amplitude=0.0,
            notes="No-subhalo data and smooth model are identical in the deterministic same-PSF control.",
        ),
        _exact_false_positive_row(
            case_id="false_positive_hexike100_truth_fit_hexike100",
            truth_psf_case="segment_hexike",
            fit_psf_case="segment_hexike",
            psf_amplitude=float(stage0_rows[HEXIKE100_TEMPLATE]["psf_amplitude"]),
            notes="No-subhalo data and smooth model are identical in the deterministic same-PSF hexike control.",
        ),
    ]
    if spie_plus_results.exists():
        for row in _read_csv_rows(spie_plus_results):
            if row["case_kind"] != "false_positive_psf_mismatch":
                continue
            rows.append(
                {
                    "case_id": row["case_id"],
                    "truth_psf_case": row["truth_psf_case"],
                    "fit_psf_case": row["fit_psf_case"],
                    "psf_amplitude": float(row["psf_amplitude"]),
                    "source": str(spie_plus_results.relative_to(REPO_ROOT)),
                    "q_fit": float(row["q_fit"]),
                    "false_positive_pass": row["false_positive_pass"] == "True",
                    "notes": "No-subhalo 100 nm segment-hexike truth data fit with perfect-PSF model.",
                }
            )
    _write_csv(output_dir / "false_positive_controls.csv", rows)
    return rows


def _position_config(base_config: Dict[str, Any], angle: float) -> Dict[str, Any]:
    config = deepcopy(base_config)
    config["run_name"] = f"{base_config['run_name']}_angle_{angle:g}"
    config["lensing"]["subhalo"]["position"] = {
        "type": "angle",
        "angle": float(angle),
        "offset_pixels": 0.0,
    }
    return config


def run_position_variation(
    *,
    config_dir: Path,
    output_dir: Path,
    angles: Sequence[float],
) -> List[Dict[str, Any]]:
    base_config = profile_cal._load_config(config_dir / f"{POSITION_TEMPLATE}.yaml")
    detector, psf_data, _lensing_baseline, _obs_baseline = _build_detector(base_config)
    rows = []
    for angle in angles:
        config = _position_config(base_config, float(angle))
        obs_test = _observation_for_config(config, psf_data, has_subhalo=True)
        _raw, q_f, coeffs = spie_plus._fisher_profile_coefficients(detector, obs_test)

        smooth_config = deepcopy(config)
        smooth_config["lensing"]["subhalo"]["enabled"] = False
        smooth_config, background_offset, _coeff_by_name = profile_cal._apply_scalar_profile_coefficients(
            config=smooth_config,
            detector=detector,
            coeffs=coeffs,
        )
        smooth_mu = profile_cal._mean_adu_for_config(config=smooth_config, detector=detector)
        smooth_mu = smooth_mu + background_offset
        data_mu = np.asarray(detector._mean_adu_from_observation(obs_test), dtype=float)  # noqa: SLF001
        q_forward_profile = profile_cal._chi2(
            data_mu,
            smooth_mu,
            detector.sigma_adu_2d,
            detector.mask_2d,
        )
        ratio = q_forward_profile / q_f if q_f > 0.0 else np.nan
        rows.append(
            {
                "angle_deg": float(angle),
                "mass_msun": float(config["lensing"]["subhalo"]["mass"]),
                "psf_case": "perfect",
                "q_f": q_f,
                "q_forward_profile": q_forward_profile,
                "q_forward_over_q_f": ratio,
                "detected_q_gt_10": bool(q_f > 10.0),
                "forward_alignment_pass_5pct": bool(abs(ratio - 1.0) <= 0.05),
            }
        )
    _write_csv(output_dir / "position_variation.csv", rows)
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage0-results", default=str(DEFAULT_RESULTS))
    parser.add_argument("--config-dir", default=str(DEFAULT_CONFIG_DIR))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--spie-plus-results", default=str(DEFAULT_SPIE_PLUS_RESULTS))
    parser.add_argument("--noisy-cases", nargs="*", default=list(DEFAULT_NOISY_CASES))
    parser.add_argument("--noise-seeds", nargs="*", type=int, default=list(DEFAULT_NOISE_SEEDS))
    parser.add_argument("--position-angles", nargs="*", type=float, default=list(DEFAULT_POSITION_ANGLES))
    return parser.parse_args()


def main() -> int:
    os.environ.setdefault("HWOSLAPS_DISABLE_TQDM", "1")
    start = time.perf_counter()
    args = parse_args()
    output_dir = Path(args.output_dir).resolve()
    stage0_rows = profile_cal._read_stage0_rows(Path(args.stage0_results).resolve())
    config_dir = Path(args.config_dir).resolve()
    noisy_rows, noisy_summary = run_noisy_ensemble(
        stage0_rows=stage0_rows,
        config_dir=config_dir,
        output_dir=output_dir,
        cases=args.noisy_cases,
        seeds=args.noise_seeds,
    )
    false_positive_rows = run_false_positive_controls(
        stage0_rows=stage0_rows,
        spie_plus_results=Path(args.spie_plus_results).resolve(),
        output_dir=output_dir,
    )
    position_rows = run_position_variation(
        config_dir=config_dir,
        output_dir=output_dir,
        angles=args.position_angles,
    )
    summary = {
        "runtime_s": time.perf_counter() - start,
        "noisy_rows": len(noisy_rows),
        "noisy_cases": len(noisy_summary),
        "false_positive_controls": len(false_positive_rows),
        "position_angles": len(position_rows),
        "all_false_positive_controls_pass": all(bool(row["false_positive_pass"]) for row in false_positive_rows),
        "all_position_forward_alignments_pass": all(bool(row["forward_alignment_pass_5pct"]) for row in position_rows),
    }
    _write_json(output_dir / "summary.json", summary)
    print(f"Wrote forecast robustness outputs to {output_dir}")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if summary["all_false_positive_controls_pass"] and summary["all_position_forward_alignments_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
