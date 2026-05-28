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
from typing import Any, Dict, Iterable, List, Optional, Tuple

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


def _disable_expensive_autofit_output() -> None:
    from autoconf import conf

    output = conf.instance["output"]
    output["latent_during_fit"] = False
    output["latent_after_fit"] = False
    output["latent_csv"] = False
    output["latent_results"] = False
    try:
        conf.instance["visualize"]["plots_search"]["nest"]["corner_anesthetic"] = False
    except KeyError:
        pass


def _disable_analysis_visualization(runner: Any) -> None:
    original_make_analysis = runner.make_analysis

    def make_analysis_no_visuals(dataset: Any) -> Any:
        analysis = original_make_analysis(dataset)

        def never_visualize(*_args: Any, **_kwargs: Any) -> bool:
            return False

        analysis.should_visualize = never_visualize
        return analysis

    runner.make_analysis = make_analysis_no_visuals


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
    fast_output: bool,
    fit_mode: str,
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

    if fast_output:
        _disable_expensive_autofit_output()

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
        case_id=f"{case_label}_{fit_mode}",
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
    if fast_output:
        _disable_analysis_visualization(runner)
    validator = NonlinearMetricValidator(runner=runner)
    case_result = validator.validate_fixed_template(
        dataset=dataset,
        dataset_metadata=dataset_metadata,
        full_config=config,
        trial=trial,
        psf_case=str(stage0_row["psf_case"]),
        fit_mode=fit_mode,
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
            "fast_output": fast_output,
            "fit_mode": fit_mode,
            "case_kind": "injected_subhalo",
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
            "case_kind": "injected_subhalo",
            "gpu": gpu,
            "status": "success"
            if case_result.smooth_fit.status == "success"
            and case_result.subhalo_fit.status == "success"
            else "failed",
            "runtime_s_total": time.time() - start,
        }
    )
    return row


def _build_false_positive_payload(
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
    fast_output: bool,
    false_positive_fit_psf: str,
    fit_mode: str,
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

    if fast_output:
        _disable_expensive_autofit_output()

    start = time.time()
    truth_config = _load_config(config_path)
    truth_config["run_name"] = f"stage0_nonlinear_{case_label}_truth"
    validate_or_raise(truth_config)
    truth_psf = generate_psf_system(truth_config["psf"], full_config=truth_config)

    truth_smooth_config = _load_config(config_path)
    truth_smooth_config["run_name"] = truth_config["run_name"]
    truth_smooth_config["lensing"]["subhalo"]["enabled"] = False
    validate_or_raise(truth_smooth_config)
    lensing_truth_smooth = generate_lensing_system(
        truth_smooth_config["lensing"],
        full_config=truth_smooth_config,
    )
    obs_truth_smooth = generate_observation(
        lensing_data=lensing_truth_smooth,
        psf_data=truth_psf,
        observation_config=truth_smooth_config["observation"],
        full_config=truth_smooth_config,
    )

    fit_config = _load_config(config_path)
    fit_config["run_name"] = f"stage0_nonlinear_{case_label}_fit"
    if false_positive_fit_psf == "perfect":
        _set_perfect_psf(fit_config)
    fit_config["lensing"]["subhalo"]["enabled"] = False
    validate_or_raise(fit_config)
    fit_psf = generate_psf_system(fit_config["psf"], full_config=fit_config)
    fit_psf_label = (
        "perfect"
        if false_positive_fit_psf == "perfect"
        else str(stage0_row["psf_case"])
    )

    trial_config = _load_config(config_path)
    trial_config["run_name"] = fit_config["run_name"]
    trial_config["lensing"]["subhalo"]["enabled"] = True
    validate_or_raise(trial_config)
    lensing_trial = generate_lensing_system(
        trial_config["lensing"],
        full_config=trial_config,
    )

    mask_bool_use = _fisher_mask_from_observation(obs_truth_smooth, truth_smooth_config)
    dataset, dataset_metadata = imaging_from_observation(
        obs_truth_smooth,
        psf_for_fit=fit_psf.kernel,
        dataset_kind=dataset_kind,
        background_treatment="subtract_known",
        mask_bool_use=mask_bool_use,
        psf_truth_label=str(stage0_row["psf_case"]),
        psf_fit_label=fit_psf_label,
    )

    trial = trial_from_lensing_truth(
        lensing_trial,
        case_id=f"{case_label}_false_positive_{fit_mode}",
    )
    trial = replace(
        trial,
        fisher_q=None,
        fisher_z=None,
        fisher_delta_log_l_equiv=None,
        metadata={
            **trial.metadata,
            "stage0_run_name": run_name,
            "validation_label": case_label,
            "gpu": gpu,
            "false_positive_template_q_f": float(stage0_row["q_f"]),
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
    if fast_output:
        _disable_analysis_visualization(runner)
    validator = NonlinearMetricValidator(runner=runner)
    case_result = validator.validate_fixed_template(
        dataset=dataset,
        dataset_metadata=dataset_metadata,
        full_config=fit_config,
        trial=trial,
        psf_case=f"{stage0_row['psf_case']} truth / {fit_psf_label} fit",
        fit_mode=fit_mode,
    )

    payload = NonlinearDetectionData(
        run_name=case_label,
        backend="pyautolens",
        cases=[case_result],
        thresholds={
            "false_positive_q_threshold": 10.0,
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
            "fast_output": fast_output,
            "fit_mode": fit_mode,
            "case_kind": "false_positive_psf_mismatch",
            "false_positive_fit_psf": false_positive_fit_psf,
        },
    )

    case_dir = output_dir / "cases" / case_label
    payload.write_json(str(case_dir / "nonlinear_result.json"))
    payload.write_cases_csv(str(case_dir / "nonlinear_result.csv"))

    row = case_result.to_csv_row(run_name=case_label)
    q_fit = row.get("q_fit")
    false_positive_pass = q_fit is not None and float(q_fit) < 10.0
    row.update(
        {
            "stage0_run_name": run_name,
            "validation_label": case_label,
            "case_kind": "false_positive_psf_mismatch",
            "gpu": gpu,
            "status": "success"
            if case_result.smooth_fit.status == "success"
            and case_result.subhalo_fit.status == "success"
            else "failed",
            "runtime_s_total": time.time() - start,
            "template_fisher_q": float(stage0_row["q_f"]),
            "truth_psf_case": str(stage0_row["psf_case"]),
            "fit_psf_case": fit_psf_label,
            "false_positive_fit_psf": false_positive_fit_psf,
            "false_positive_pass": bool(false_positive_pass),
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
        builder = (
            _build_false_positive_payload
            if args.worker_case_kind == "false_positive_psf_mismatch"
            else _build_validation_payload
        )
        builder_kwargs = {
            "run_name": args.worker_case,
            "case_label": case_label,
            "gpu": str(args.gpu),
            "stage0_row": stage0_row,
            "config_path": config_path,
            "output_dir": output_dir,
            "n_live_smooth": args.n_live_smooth,
            "n_live_subhalo": args.n_live_subhalo,
            "maxcall": args.maxcall,
            "use_jax": args.use_jax,
            "resume": args.resume,
            "dataset_kind": args.dataset_kind,
            "fast_output": args.fast_output,
            "fit_mode": args.fit_mode,
        }
        if args.worker_case_kind == "false_positive_psf_mismatch":
            builder_kwargs["false_positive_fit_psf"] = args.false_positive_fit_psf
        row = builder(**builder_kwargs)
        row["worker_log"] = str((case_dir / "worker.log").relative_to(REPO_ROOT))
        _write_json(case_dir / "case_row.json", row)
        return 0 if row["status"] == "success" else 2
    except Exception as exc:
        row = {
            "stage0_run_name": args.worker_case,
            "validation_label": case_label,
            "gpu": str(args.gpu),
            "case_kind": args.worker_case_kind,
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


def _case_tasks(args: argparse.Namespace) -> List[Tuple[str, str]]:
    injected_cases = DEFAULT_CASES if args.cases is None else args.cases
    injected = [("injected_subhalo", case) for case in list(injected_cases)]
    false_positive = [
        ("false_positive_psf_mismatch", case)
        for case in list(args.false_positive_cases or [])
    ]
    tasks = injected + false_positive
    if args.limit is not None:
        tasks = tasks[: args.limit]
    return tasks


def _launch_cases(args: argparse.Namespace) -> List[Dict[str, Any]]:
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    gpus = [item.strip() for item in args.gpus.split(",") if item.strip()]
    if not gpus:
        raise ValueError("--gpus must contain at least one GPU index")

    tasks = _case_tasks(args)
    rows = []
    all_failures = []
    max_concurrent = max(1, int(args.max_concurrent))

    for batch_start in range(0, len(tasks), max_concurrent):
        batch = tasks[batch_start : batch_start + max_concurrent]
        processes = []
        for batch_index, (case_kind, run_name) in enumerate(batch):
            index = batch_start + batch_index
            gpu = gpus[index % len(gpus)]
            if case_kind == "false_positive_psf_mismatch":
                case_label = f"false_positive_{CASE_LABELS.get(run_name, run_name)}"
            else:
                case_label = CASE_LABELS.get(run_name, run_name)
            case_dir = output_dir / "cases" / case_label
            case_dir.mkdir(parents=True, exist_ok=True)
            log_path = case_dir / "worker.log"
            cmd = [
                sys.executable,
                str(Path(__file__).resolve()),
                "--worker-case",
                run_name,
                "--worker-case-kind",
                case_kind,
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
                "--fit-mode",
                args.fit_mode,
            ]
            if args.maxcall is not None:
                cmd.extend(["--maxcall", str(args.maxcall)])
            if args.use_jax:
                cmd.append("--use-jax")
            if args.resume:
                cmd.append("--resume")
            if args.fast_output:
                cmd.append("--fast-output")
            if case_kind == "false_positive_psf_mismatch":
                cmd.extend(["--false-positive-fit-psf", args.false_positive_fit_psf])

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
            print(
                f"Started {index + 1}/{len(tasks)} {case_kind} {case_label} "
                f"on GPU {gpu} (pid {process.pid})",
                flush=True,
            )

        failures = []
        for run_name, case_label, gpu, process, log_file, log_path in processes:
            return_code = process.wait()
            log_file.close()
            if return_code != 0:
                failures.append((case_label, gpu, return_code, log_path))
            print(f"Finished {case_label} on GPU {gpu} with exit code {return_code}", flush=True)

        for _, case_label, _, _, _, _ in processes:
            row_path = output_dir / "cases" / case_label / "case_row.json"
            if row_path.exists():
                with row_path.open("r", encoding="utf-8") as handle:
                    rows.append(json.load(handle))
        all_failures.extend(failures)
        _write_results_csv(output_dir / "results.csv", rows)
        _write_json(
            output_dir / "run_summary.json",
            {
                "cases": [case for _, case in tasks],
                "tasks": [
                    {"case_kind": case_kind, "run_name": run_name}
                    for case_kind, run_name in tasks
                ],
                "gpus": gpus,
                "max_concurrent": max_concurrent,
                "n_live_smooth": args.n_live_smooth,
                "n_live_subhalo": args.n_live_subhalo,
                "maxcall": args.maxcall,
                "use_jax": args.use_jax,
                "fast_output": args.fast_output,
                "dataset_kind": args.dataset_kind,
                "fit_mode": args.fit_mode,
                "false_positive_fit_psf": args.false_positive_fit_psf,
                "n_cases": len(rows),
                "n_expected": len(tasks),
                "n_success": sum(row.get("status") == "success" for row in rows),
                "n_failed": sum(row.get("status") != "success" for row in rows),
                "failures": [
                    {
                        "case_label": case_label,
                        "gpu": gpu,
                        "return_code": return_code,
                        "log_path": str(log_path.relative_to(REPO_ROOT)),
                    }
                    for case_label, gpu, return_code, log_path in all_failures
                ],
            },
        )

    if all_failures:
        details = "; ".join(
            f"{case_label} gpu={gpu} rc={return_code} log={log_path}"
            for case_label, gpu, return_code, log_path in all_failures
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
    parser.add_argument("--false-positive-cases", nargs="*", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max-concurrent", type=int, default=1)
    parser.add_argument("--n-live-smooth", type=int, default=100)
    parser.add_argument("--n-live-subhalo", type=int, default=100)
    parser.add_argument("--maxcall", type=int, default=None)
    parser.add_argument("--dataset-kind", choices=("asimov", "noisy"), default="asimov")
    parser.add_argument(
        "--fit-mode",
        choices=("fixed_template", "local_search"),
        default="fixed_template",
        help=(
            "Subhalo fit mode. fixed_template keeps the trial center fixed; "
            "local_search gives the subhalo center a local prior window."
        ),
    )
    parser.add_argument("--use-jax", action="store_true", default=False)
    parser.add_argument("--resume", action="store_true", default=False)
    parser.add_argument("--fast-output", action="store_true", default=False)
    parser.add_argument(
        "--false-positive-fit-psf",
        choices=("perfect", "truth"),
        default="perfect",
        help=(
            "PSF kernel used when fitting false-positive no-subhalo controls. "
            "'perfect' reproduces the PSF-mismatch stress test; 'truth' is the "
            "matched-PSF control."
        ),
    )
    parser.add_argument("--worker-case", default=None)
    parser.add_argument(
        "--worker-case-kind",
        choices=("injected_subhalo", "false_positive_psf_mismatch"),
        default="injected_subhalo",
    )
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
