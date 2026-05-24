#!/usr/bin/env python
"""Run sparse Stage 0 PyAutoLens nonlinear validation on selected GPUs."""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
STAGE0_RESULTS = REPO_ROOT / "outputs/stage0_internal_review/results.csv"
STAGE0_CONFIG_DIR = REPO_ROOT / "outputs/stage0_internal_review/generated_configs"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "outputs/stage0_nonlinear_validation"

DEFAULT_CASES = (
    "stage0_internal_review_mass_m1e7_perfect",
    "stage0_internal_review_mass_m10p7p25_perfect",
    "stage0_internal_review_mass_m10p7p75_perfect",
    "stage0_internal_review_hexike_s0_n2_a100p0nm_m1e7",
)

CASE_LABELS = {
    "stage0_internal_review_mass_m1e7_perfect": "perfect_m1e7_near_threshold",
    "stage0_internal_review_mass_m10p7p25_perfect": "perfect_m10p7p25_moderate",
    "stage0_internal_review_mass_m10p7p75_perfect": "perfect_m10p7p75_high",
    "stage0_internal_review_hexike_s0_n2_a100p0nm_m1e7": "hexike100_m1e7_endpoint",
}

NONLINEAR_COLUMNS = (
    "stage0_run_name",
    "validation_label",
    "gpu",
    "status",
    "runtime_s_total",
    "worker_log",
)


def _read_stage0_rows(path: Path) -> Dict[str, Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return {row["run_name"]: row for row in csv.DictReader(handle)}


def _json_safe(value: Any) -> Any:
    try:
        import numpy as np
    except Exception:
        np = None

    if np is not None:
        if isinstance(value, np.ndarray):
            return [_json_safe(item) for item in value.tolist()]
        if isinstance(value, np.generic):
            return _json_safe(value.item())
    if isinstance(value, dict):
        return {str(key): _json_safe(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, float):
        if value != value or value in (float("inf"), -float("inf")):
            return None
    return value


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(_json_safe(payload), handle, indent=2, sort_keys=True)


def _load_config(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    config["plotting"]["enabled"] = False
    return config


def _fisher_mask_from_observation(observation: Any, config: Dict[str, Any]) -> Any:
    import numpy as np

    fisher_config = config["modeling"]["fisher"]
    mask_mode = str(fisher_config.get("mask_mode", "source_snr")).lower()
    if mask_mode == "all_pixels":
        return np.ones_like(observation.source_electrons, dtype=bool)
    if mask_mode != "source_snr":
        raise ValueError("Only source_snr and all_pixels Fisher masks are supported")

    source_adu = np.asarray(observation.source_electrons, dtype=float) / float(observation.gain)
    noise_adu = np.asarray(observation.noise_map.native, dtype=float)
    threshold = float(fisher_config["snr_threshold"])
    return source_adu / np.maximum(noise_adu, 1.0e-12) > threshold


def _build_validation_payload(
    *,
    run_name: str,
    case_label: str,
    gpu: str,
    stage0_row: Dict[str, str],
    config_path: Path,
    output_dir: Path,
    n_live_smooth: int,
    n_live_subhalo: int,
    maxcall: Optional[int],
    use_jax: bool,
    resume: bool,
    dataset_kind: str,
) -> Dict[str, Any]:
    from hwoslaps.config.validation import validate_or_raise
    from hwoslaps.lensing import generate_lensing_system
    from hwoslaps.modeling.nonlinear.autolens_runner import (
        AutoLensFitRunner,
        NonlinearSearchSettings,
    )
    from hwoslaps.modeling.nonlinear.dataset_builder import imaging_from_observation
    from hwoslaps.modeling.nonlinear.output_schema import NonlinearDetectionData
    from hwoslaps.modeling.nonlinear.trial import trial_from_lensing_truth
    from hwoslaps.modeling.nonlinear.validator import NonlinearMetricValidator
    from hwoslaps.observation import generate_observation
    from hwoslaps.psf import generate_psf_system

    start = time.time()
    config = _load_config(config_path)
    config["run_name"] = f"stage0_nonlinear_{case_label}"
    validate_or_raise(config)

    psf_data = generate_psf_system(config["psf"], full_config=config)

    baseline_config = _load_config(config_path)
    baseline_config["run_name"] = config["run_name"]
    baseline_config["lensing"]["subhalo"]["enabled"] = False
    validate_or_raise(baseline_config)
    lensing_baseline = generate_lensing_system(
        baseline_config["lensing"],
        full_config=baseline_config,
    )
    obs_baseline = generate_observation(
        lensing_data=lensing_baseline,
        psf_data=psf_data,
        observation_config=baseline_config["observation"],
        full_config=baseline_config,
    )

    test_config = _load_config(config_path)
    test_config["run_name"] = config["run_name"]
    test_config["lensing"]["subhalo"]["enabled"] = True
    validate_or_raise(test_config)
    lensing_test = generate_lensing_system(
        test_config["lensing"],
        full_config=test_config,
    )
    obs_test = generate_observation(
        lensing_data=lensing_test,
        psf_data=psf_data,
        observation_config=test_config["observation"],
        full_config=test_config,
    )

    mask_bool_use = _fisher_mask_from_observation(obs_baseline, config)
    dataset, dataset_metadata = imaging_from_observation(
        obs_test,
        psf_for_fit=None,
        dataset_kind=dataset_kind,
        background_treatment="subtract_known",
        mask_bool_use=mask_bool_use,
        psf_truth_label=str(stage0_row["psf_case"]),
        psf_fit_label=str(stage0_row["psf_case"]),
    )

    trial = trial_from_lensing_truth(
        lensing_test,
        case_id=f"{case_label}_fixed_template",
    )
    fisher_q = float(stage0_row["q_f"])
    trial = replace(
        trial,
        fisher_q=fisher_q,
        fisher_z=float(stage0_row["z_f"]),
        fisher_delta_log_l_equiv=float(stage0_row["delta_log_l_f_equiv"]),
        metadata={
            **trial.metadata,
            "stage0_run_name": run_name,
            "validation_label": case_label,
            "gpu": gpu,
        },
    )

    settings = NonlinearSearchSettings(
        n_live_smooth=n_live_smooth,
        n_live_subhalo_fixed=n_live_subhalo,
        number_of_cores=1,
        maxcall=maxcall,
        path_prefix="searches",
        unique_tag=case_label,
        resume=resume,
        use_jax=use_jax,
    )
    runner = AutoLensFitRunner(settings=settings, output_dir=str(output_dir))
    validator = NonlinearMetricValidator(runner=runner)
    case_result = validator.validate_fixed_template(
        dataset=dataset,
        dataset_metadata=dataset_metadata,
        full_config=config,
        trial=trial,
        psf_case=str(stage0_row["psf_case"]),
    )

    payload = NonlinearDetectionData(
        run_name=case_label,
        backend="pyautolens",
        cases=[case_result],
        thresholds={
            "scdd_q_threshold": 10.0,
            "q_fit_definition": "2 * (log_l_subhalo - log_l_smooth)",
        },
        config={
            "stage0_run_name": run_name,
            "config_path": str(config_path.relative_to(REPO_ROOT)),
            "use_jax": use_jax,
            "gpu": gpu,
            "dataset_kind": dataset_kind,
            "n_live_smooth": n_live_smooth,
            "n_live_subhalo": n_live_subhalo,
            "maxcall": maxcall,
        },
    )

    case_dir = output_dir / "cases" / case_label
    payload.write_json(str(case_dir / "nonlinear_result.json"))
    payload.write_cases_csv(str(case_dir / "nonlinear_result.csv"))

    row = case_result.to_csv_row(run_name=case_label)
    row.update(
        {
            "stage0_run_name": run_name,
            "validation_label": case_label,
            "gpu": gpu,
            "status": "success"
            if case_result.smooth_fit.status == "success"
            and case_result.subhalo_fit.status == "success"
            else "failed",
            "runtime_s_total": time.time() - start,
        }
    )
    return row


def _run_worker(args: argparse.Namespace) -> int:
    output_dir = Path(args.output_dir).resolve()
    rows = _read_stage0_rows(Path(args.stage0_results).resolve())
    stage0_row = rows[args.worker_case]
    case_label = args.case_label or CASE_LABELS.get(args.worker_case, args.worker_case)
    config_path = Path(args.config_dir).resolve() / f"{args.worker_case}.yaml"
    case_dir = output_dir / "cases" / case_label

    start = time.time()
    try:
        row = _build_validation_payload(
            run_name=args.worker_case,
            case_label=case_label,
            gpu=str(args.gpu),
            stage0_row=stage0_row,
            config_path=config_path,
            output_dir=output_dir,
            n_live_smooth=args.n_live_smooth,
            n_live_subhalo=args.n_live_subhalo,
            maxcall=args.maxcall,
            use_jax=args.use_jax,
            resume=args.resume,
            dataset_kind=args.dataset_kind,
        )
        row["worker_log"] = str((case_dir / "worker.log").relative_to(REPO_ROOT))
        _write_json(case_dir / "case_row.json", row)
        return 0 if row["status"] == "success" else 2
    except Exception as exc:
        row = {
            "stage0_run_name": args.worker_case,
            "validation_label": case_label,
            "gpu": str(args.gpu),
            "status": "failed",
            "runtime_s_total": time.time() - start,
            "worker_log": str((case_dir / "worker.log").relative_to(REPO_ROOT)),
            "error": str(exc),
        }
        _write_json(case_dir / "case_row.json", row)
        raise


def _write_results_csv(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    from hwoslaps.modeling.nonlinear.output_schema import NONLINEAR_CASE_CSV_COLUMNS

    rows = list(rows)
    fieldnames = list(NONLINEAR_COLUMNS) + [
        name for name in NONLINEAR_CASE_CSV_COLUMNS if name not in NONLINEAR_COLUMNS
    ]
    extras = sorted({key for row in rows for key in row} - set(fieldnames))
    fieldnames.extend(extras)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(_json_safe(row))


def _launch_cases(args: argparse.Namespace) -> List[Dict[str, Any]]:
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    gpus = [item.strip() for item in args.gpus.split(",") if item.strip()]
    if not gpus:
        raise ValueError("--gpus must contain at least one GPU index")

    cases = list(args.cases or DEFAULT_CASES)
    if args.limit is not None:
        cases = cases[: args.limit]

    processes = []
    for index, run_name in enumerate(cases):
        gpu = gpus[index % len(gpus)]
        case_label = CASE_LABELS.get(run_name, run_name)
        case_dir = output_dir / "cases" / case_label
        case_dir.mkdir(parents=True, exist_ok=True)
        log_path = case_dir / "worker.log"
        cmd = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--worker-case",
            run_name,
            "--case-label",
            case_label,
            "--gpu",
            gpu,
            "--output-dir",
            str(output_dir),
            "--stage0-results",
            str(Path(args.stage0_results).resolve()),
            "--config-dir",
            str(Path(args.config_dir).resolve()),
            "--n-live-smooth",
            str(args.n_live_smooth),
            "--n-live-subhalo",
            str(args.n_live_subhalo),
            "--dataset-kind",
            args.dataset_kind,
        ]
        if args.maxcall is not None:
            cmd.extend(["--maxcall", str(args.maxcall)])
        if args.use_jax:
            cmd.append("--use-jax")
        if args.resume:
            cmd.append("--resume")

        env = dict(os.environ)
        env["CUDA_VISIBLE_DEVICES"] = gpu
        env.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
        env.setdefault("PYAUTO_SKIP_WORKSPACE_VERSION_CHECK", "1")
        env.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
        env.setdefault("NUMBA_CACHE_DIR", "/tmp/numba_cache")
        for env_name in (
            "OMP_NUM_THREADS",
            "OPENBLAS_NUM_THREADS",
            "MKL_NUM_THREADS",
            "BLIS_NUM_THREADS",
            "VECLIB_MAXIMUM_THREADS",
            "NUMEXPR_MAX_THREADS",
        ):
            env.setdefault(env_name, "1")

        log_file = log_path.open("w", encoding="utf-8", buffering=1)
        process = subprocess.Popen(
            cmd,
            cwd=str(REPO_ROOT),
            stdout=log_file,
            stderr=subprocess.STDOUT,
            env=env,
        )
        processes.append((run_name, case_label, gpu, process, log_file, log_path))
        print(f"Started {case_label} on GPU {gpu} (pid {process.pid})")

    failures = []
    for run_name, case_label, gpu, process, log_file, log_path in processes:
        return_code = process.wait()
        log_file.close()
        if return_code != 0:
            failures.append((case_label, gpu, return_code, log_path))
        print(f"Finished {case_label} on GPU {gpu} with exit code {return_code}")

    rows = []
    for _, case_label, _, _, _, _ in processes:
        row_path = output_dir / "cases" / case_label / "case_row.json"
        if row_path.exists():
            with row_path.open("r", encoding="utf-8") as handle:
                rows.append(json.load(handle))

    _write_results_csv(output_dir / "results.csv", rows)
    _write_json(
        output_dir / "run_summary.json",
        {
            "cases": cases,
            "gpus": gpus,
            "n_live_smooth": args.n_live_smooth,
            "n_live_subhalo": args.n_live_subhalo,
            "maxcall": args.maxcall,
            "use_jax": args.use_jax,
            "dataset_kind": args.dataset_kind,
            "n_cases": len(rows),
            "n_success": sum(row.get("status") == "success" for row in rows),
            "n_failed": sum(row.get("status") != "success" for row in rows),
            "failures": [
                {
                    "case_label": case_label,
                    "gpu": gpu,
                    "return_code": return_code,
                    "log_path": str(log_path.relative_to(REPO_ROOT)),
                }
                for case_label, gpu, return_code, log_path in failures
            ],
        },
    )
    if failures:
        details = "; ".join(
            f"{case_label} gpu={gpu} rc={return_code} log={log_path}"
            for case_label, gpu, return_code, log_path in failures
        )
        raise RuntimeError(f"Nonlinear validation failures: {details}")
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--stage0-results", default=str(STAGE0_RESULTS))
    parser.add_argument("--config-dir", default=str(STAGE0_CONFIG_DIR))
    parser.add_argument("--gpus", default="0,1,2,3")
    parser.add_argument("--cases", nargs="*", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--n-live-smooth", type=int, default=100)
    parser.add_argument("--n-live-subhalo", type=int, default=100)
    parser.add_argument("--maxcall", type=int, default=None)
    parser.add_argument("--dataset-kind", choices=("asimov", "noisy"), default="asimov")
    parser.add_argument("--use-jax", action="store_true", default=False)
    parser.add_argument("--resume", action="store_true", default=False)
    parser.add_argument("--worker-case", default=None)
    parser.add_argument("--case-label", default=None)
    parser.add_argument("--gpu", default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.worker_case:
        return _run_worker(args)

    rows = _launch_cases(args)
    print(f"Wrote nonlinear validation results for {len(rows)} cases to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
