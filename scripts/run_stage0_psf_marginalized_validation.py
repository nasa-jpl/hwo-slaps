#!/usr/bin/env python
"""Run Stage 0 PyAutoLens validation with a discrete PSF nuisance bank.

This script approximates PSF-nuisance marginalization by fitting each dataset
with a small bank of PSF kernels generated from scaled versions of the truth
aberration coefficients. For each case it runs smooth and subhalo models for
every PSF candidate, then compares log-sum-exp marginalized evidences.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import subprocess
import sys
import time
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS_DIR))

from run_stage0_nonlinear_validation import (  # noqa: E402
    CASE_LABELS,
    _disable_analysis_visualization,
    _disable_expensive_autofit_output,
    _fisher_mask_from_observation,
    _json_safe,
    _load_config,
    _read_stage0_rows,
    _set_perfect_psf,
    _write_json,
)


def _logsumexp(values: Iterable[float]) -> Optional[float]:
    vals = [float(val) for val in values if val is not None and np.isfinite(float(val))]
    if not vals:
        return None
    max_val = max(vals)
    return float(max_val + math.log(sum(math.exp(val - max_val) for val in vals)))


def _scale_numeric_tree(value: Any, scale: float) -> Any:
    if isinstance(value, dict):
        return {key: _scale_numeric_tree(val, scale) for key, val in value.items()}
    if isinstance(value, list):
        return [_scale_numeric_tree(val, scale) for val in value]
    if isinstance(value, tuple):
        return tuple(_scale_numeric_tree(val, scale) for val in value)
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return float(value) * float(scale)
    return value


def _has_nonzero_numeric(value: Any) -> bool:
    if isinstance(value, dict):
        return any(_has_nonzero_numeric(val) for val in value.values())
    if isinstance(value, (list, tuple)):
        return any(_has_nonzero_numeric(val) for val in value)
    if isinstance(value, bool):
        return False
    if isinstance(value, (int, float)):
        return abs(float(value)) > 0.0
    return False


def _scale_psf_aberrations(config: Dict[str, Any], scale: float) -> None:
    """Scale all configured aberration coefficients in-place."""
    if float(scale) == 0.0:
        _set_perfect_psf(config)
        return
    aberr = config["psf"]["aberrations"]
    for key in ("segment_pistons", "segment_tiptilts", "segment_hexikes", "global_zernikes"):
        aberr[key] = _scale_numeric_tree(aberr.get(key, {}), float(scale))
    aberr["enable_segment_pistons"] = _has_nonzero_numeric(aberr.get("segment_pistons", {}))
    aberr["enable_segment_tiptilts"] = _has_nonzero_numeric(aberr.get("segment_tiptilts", {}))
    aberr["enable_segment_hexikes"] = _has_nonzero_numeric(aberr.get("segment_hexikes", {}))
    aberr["enable_global_zernikes"] = _has_nonzero_numeric(aberr.get("global_zernikes", {}))


def _scale_label(scale: float) -> str:
    if float(scale) == 0.0:
        return "perfect"
    if float(scale) == 1.0:
        return "truth"
    return f"scale_{str(float(scale)).replace('.', 'p').replace('-', 'm')}"


def _case_label(run_name: str, case_kind: str) -> str:
    base = CASE_LABELS.get(run_name, run_name)
    if case_kind == "false_positive_psf_mismatch":
        return f"psf_marginalized_false_positive_{base}"
    return f"psf_marginalized_{base}"


def _trial_from_truth(
    *,
    config: Dict[str, Any],
    case_label: str,
    fit_mode: str,
    stage0_row: Dict[str, str],
    gpu: str,
) -> Any:
    from hwoslaps.lensing import generate_lensing_system
    from hwoslaps.modeling.nonlinear.trial import trial_from_lensing_truth

    trial_config = deepcopy(config)
    trial_config["run_name"] = f"stage0_psf_marginalized_{case_label}_trial"
    trial_config["lensing"]["subhalo"]["enabled"] = True
    lensing_trial = generate_lensing_system(
        trial_config["lensing"],
        full_config=trial_config,
    )
    trial = trial_from_lensing_truth(
        lensing_trial,
        case_id=f"{case_label}_{fit_mode}",
    )
    return replace(
        trial,
        fisher_q=float(stage0_row["q_f"]) if stage0_row.get("q_f") not in (None, "") else None,
        fisher_z=float(stage0_row["z_f"]) if stage0_row.get("z_f") not in (None, "") else None,
        fisher_delta_log_l_equiv=(
            float(stage0_row["delta_log_l_f_equiv"])
            if stage0_row.get("delta_log_l_f_equiv") not in (None, "")
            else None
        ),
        metadata={
            **trial.metadata,
            "stage0_run_name": stage0_row["run_name"],
            "validation_label": case_label,
            "gpu": gpu,
        },
    )


def _build_truth_observation(
    *,
    config: Dict[str, Any],
    case_label: str,
    case_kind: str,
) -> Tuple[Any, Any, Any]:
    from hwoslaps.config.validation import validate_or_raise
    from hwoslaps.lensing import generate_lensing_system
    from hwoslaps.observation import generate_observation
    from hwoslaps.psf import generate_psf_system

    truth_config = deepcopy(config)
    truth_config["run_name"] = f"stage0_psf_marginalized_{case_label}_truth"
    validate_or_raise(truth_config)
    truth_psf = generate_psf_system(truth_config["psf"], full_config=truth_config)

    baseline_config = deepcopy(truth_config)
    baseline_config["lensing"]["subhalo"]["enabled"] = False
    validate_or_raise(baseline_config)
    lensing_baseline = generate_lensing_system(
        baseline_config["lensing"],
        full_config=baseline_config,
    )
    obs_baseline = generate_observation(
        lensing_data=lensing_baseline,
        psf_data=truth_psf,
        observation_config=baseline_config["observation"],
        full_config=baseline_config,
    )

    data_config = deepcopy(truth_config)
    data_config["lensing"]["subhalo"]["enabled"] = case_kind != "false_positive_psf_mismatch"
    validate_or_raise(data_config)
    lensing_data = generate_lensing_system(data_config["lensing"], full_config=data_config)
    observation = generate_observation(
        lensing_data=lensing_data,
        psf_data=truth_psf,
        observation_config=data_config["observation"],
        full_config=data_config,
    )
    return truth_psf, obs_baseline, observation


def _candidate_result_rows(
    *,
    run_name: str,
    case_label: str,
    case_kind: str,
    gpu: str,
    stage0_row: Dict[str, str],
    config_path: Path,
    output_dir: Path,
    psf_scales: List[float],
    n_live_smooth: int,
    n_live_subhalo: int,
    maxcall: Optional[int],
    use_jax: bool,
    resume: bool,
    dataset_kind: str,
    fast_output: bool,
    fit_mode: str,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    from hwoslaps.config.validation import validate_or_raise
    from hwoslaps.modeling.nonlinear.autolens_runner import (
        AutoLensFitRunner,
        NonlinearSearchSettings,
    )
    from hwoslaps.modeling.nonlinear.dataset_builder import imaging_from_observation
    from hwoslaps.modeling.nonlinear.validator import NonlinearMetricValidator
    from hwoslaps.psf import generate_psf_system

    if fast_output:
        _disable_expensive_autofit_output()

    start = time.time()
    base_config = _load_config(config_path)
    base_config["run_name"] = f"stage0_psf_marginalized_{case_label}"
    validate_or_raise(base_config)

    truth_psf, obs_baseline, observation = _build_truth_observation(
        config=base_config,
        case_label=case_label,
        case_kind=case_kind,
    )
    mask_bool_use = _fisher_mask_from_observation(obs_baseline, base_config)
    trial = _trial_from_truth(
        config=base_config,
        case_label=case_label,
        fit_mode=fit_mode,
        stage0_row=stage0_row,
        gpu=gpu,
    )

    rows: List[Dict[str, Any]] = []
    log_prior = -math.log(len(psf_scales))
    for scale in psf_scales:
        candidate_label = _scale_label(scale)
        candidate_config = deepcopy(base_config)
        candidate_config["run_name"] = f"stage0_psf_marginalized_{case_label}_{candidate_label}_fit"
        _scale_psf_aberrations(candidate_config, scale)
        validate_or_raise(candidate_config)
        candidate_psf = generate_psf_system(candidate_config["psf"], full_config=candidate_config)

        dataset, dataset_metadata = imaging_from_observation(
            observation,
            psf_for_fit=candidate_psf.kernel,
            dataset_kind=dataset_kind,
            background_treatment="subtract_known",
            mask_bool_use=mask_bool_use,
            psf_truth_label=str(stage0_row["psf_case"]),
            psf_fit_label=candidate_label,
        )
        candidate_trial = replace(
            trial,
            case_id=f"{case_label}_{fit_mode}_{candidate_label}",
        )
        settings = NonlinearSearchSettings(
            n_live_smooth=n_live_smooth,
            n_live_subhalo_fixed=n_live_subhalo,
            number_of_cores=1,
            maxcall=maxcall,
            path_prefix="searches",
            unique_tag=candidate_label,
            resume=resume,
            use_jax=use_jax,
        )
        runner = AutoLensFitRunner(settings=settings, output_dir=str(output_dir))
        if fast_output:
            _disable_analysis_visualization(runner)
        validator = NonlinearMetricValidator(runner=runner)
        result = validator.validate_fixed_template(
            dataset=dataset,
            dataset_metadata=dataset_metadata,
            full_config=candidate_config,
            trial=candidate_trial,
            psf_case=f"{stage0_row['psf_case']} truth / {candidate_label} fit",
            fit_mode=fit_mode,
        )
        row = result.to_csv_row(run_name=case_label)
        row.update(
            {
                "stage0_run_name": run_name,
                "validation_label": case_label,
                "case_kind": case_kind,
                "gpu": gpu,
                "status": (
                    "success"
                    if result.smooth_fit.status == "success"
                    and result.subhalo_fit.status == "success"
                    else "failed"
                ),
                "truth_psf_case": str(stage0_row["psf_case"]),
                "fit_psf_case": candidate_label,
                "psf_candidate_scale": float(scale),
                "psf_candidate_log_prior": float(log_prior),
                "psf_candidate_total_rms_nm": getattr(candidate_psf, "total_rms_nm", None),
                "psf_candidate_strehl": getattr(candidate_psf, "strehl_ratio", None),
                "truth_psf_total_rms_nm": getattr(truth_psf, "total_rms_nm", None),
                "truth_psf_strehl": getattr(truth_psf, "strehl_ratio", None),
            }
        )
        rows.append(row)

    success_rows = [row for row in rows if row.get("status") == "success"]
    smooth_logz = _logsumexp(
        float(row["log_evidence_smooth"]) + float(row["psf_candidate_log_prior"])
        for row in success_rows
        if row.get("log_evidence_smooth") not in (None, "")
    )
    subhalo_logz = _logsumexp(
        float(row["log_evidence_subhalo"]) + float(row["psf_candidate_log_prior"])
        for row in success_rows
        if row.get("log_evidence_subhalo") not in (None, "")
    )
    smooth_logl_values = [
        float(row["log_l_smooth"])
        for row in success_rows
        if row.get("log_l_smooth") not in (None, "")
    ]
    subhalo_logl_values = [
        float(row["log_l_subhalo"])
        for row in success_rows
        if row.get("log_l_subhalo") not in (None, "")
    ]
    smooth_best = max(success_rows, key=lambda row: float(row["log_l_smooth"])) if smooth_logl_values else None
    subhalo_best = max(success_rows, key=lambda row: float(row["log_l_subhalo"])) if subhalo_logl_values else None
    log_l_smooth_profile = max(smooth_logl_values) if smooth_logl_values else None
    log_l_subhalo_profile = max(subhalo_logl_values) if subhalo_logl_values else None
    signed_delta_log_l = (
        None
        if log_l_smooth_profile is None or log_l_subhalo_profile is None
        else log_l_subhalo_profile - log_l_smooth_profile
    )
    q_fit = None if signed_delta_log_l is None else max(0.0, 2.0 * signed_delta_log_l)
    delta_logz = None if smooth_logz is None or subhalo_logz is None else subhalo_logz - smooth_logz
    summary_row = {
        "stage0_run_name": run_name,
        "validation_label": case_label,
        "case_kind": case_kind,
        "status": "success" if len(success_rows) == len(rows) else "partial_failure",
        "runtime_s_total": time.time() - start,
        "n_psf_candidates": len(rows),
        "n_success_candidates": len(success_rows),
        "dataset_kind": dataset_kind,
        "fit_mode": fit_mode,
        "truth_psf_case": str(stage0_row["psf_case"]),
        "fisher_q": trial.fisher_q,
        "fisher_z": trial.fisher_z,
        "log_l_smooth_psf_profile": log_l_smooth_profile,
        "log_l_subhalo_psf_profile": log_l_subhalo_profile,
        "signed_delta_log_l_psf_profile": signed_delta_log_l,
        "q_fit_psf_profile": q_fit,
        "log_evidence_smooth_psf_marg": smooth_logz,
        "log_evidence_subhalo_psf_marg": subhalo_logz,
        "delta_log_evidence_psf_marg": delta_logz,
        "log10_bayes_factor_psf_marg": None if delta_logz is None else delta_logz / math.log(10.0),
        "detected_fisher_scdd": None if trial.fisher_q is None else trial.fisher_q >= 10.0,
        "detected_fit_scdd_psf_profile": None if q_fit is None else q_fit >= 10.0,
        "detected_evidence_psf_marg": None if delta_logz is None else delta_logz > 5.0,
        "best_smooth_fit_psf_case": None if smooth_best is None else smooth_best.get("fit_psf_case"),
        "best_subhalo_fit_psf_case": None if subhalo_best is None else subhalo_best.get("fit_psf_case"),
        "best_smooth_fit_scale": None if smooth_best is None else smooth_best.get("psf_candidate_scale"),
        "best_subhalo_fit_scale": None if subhalo_best is None else subhalo_best.get("psf_candidate_scale"),
        "worker_log": str((output_dir / "cases" / case_label / "worker.log").relative_to(REPO_ROOT)),
    }
    return rows, summary_row


def _write_csv(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    rows = list(rows)
    keys = sorted({key for row in rows for key in row})
    preferred = [
        "stage0_run_name",
        "validation_label",
        "case_kind",
        "status",
        "dataset_kind",
        "fit_mode",
        "truth_psf_case",
        "fit_psf_case",
        "psf_candidate_scale",
        "fisher_q",
        "q_fit",
        "q_fit_psf_profile",
        "delta_log_evidence",
        "delta_log_evidence_psf_marg",
        "log_l_smooth",
        "log_l_subhalo",
        "log_evidence_smooth",
        "log_evidence_subhalo",
    ]
    fieldnames = [key for key in preferred if key in keys] + [key for key in keys if key not in preferred]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(_json_safe(row))


def _run_worker(args: argparse.Namespace) -> int:
    output_dir = Path(args.output_dir).resolve()
    rows = _read_stage0_rows(Path(args.stage0_results).resolve())
    stage0_row = rows[args.worker_case]
    case_label = args.case_label or _case_label(args.worker_case, args.worker_case_kind)
    config_path = Path(args.config_dir).resolve() / f"{args.worker_case}.yaml"
    case_dir = output_dir / "cases" / case_label
    start = time.time()
    try:
        candidate_rows, summary_row = _candidate_result_rows(
            run_name=args.worker_case,
            case_label=case_label,
            case_kind=args.worker_case_kind,
            gpu=str(args.gpu),
            stage0_row=stage0_row,
            config_path=config_path,
            output_dir=output_dir,
            psf_scales=[float(val) for val in args.psf_scales],
            n_live_smooth=args.n_live_smooth,
            n_live_subhalo=args.n_live_subhalo,
            maxcall=args.maxcall,
            use_jax=args.use_jax,
            resume=args.resume,
            dataset_kind=args.dataset_kind,
            fast_output=args.fast_output,
            fit_mode=args.fit_mode,
        )
        _write_json(case_dir / "candidate_rows.json", candidate_rows)
        _write_json(case_dir / "marginalized_row.json", summary_row)
        _write_csv(case_dir / "candidate_rows.csv", candidate_rows)
        _write_csv(case_dir / "marginalized_row.csv", [summary_row])
        return 0 if summary_row["status"] == "success" else 2
    except Exception as exc:
        row = {
            "stage0_run_name": args.worker_case,
            "validation_label": case_label,
            "case_kind": args.worker_case_kind,
            "status": "failed",
            "runtime_s_total": time.time() - start,
            "worker_log": str((case_dir / "worker.log").relative_to(REPO_ROOT)),
            "error": str(exc),
        }
        _write_json(case_dir / "marginalized_row.json", row)
        raise


def _case_tasks(args: argparse.Namespace) -> List[Tuple[str, str]]:
    tasks = [("injected_subhalo", case) for case in list(args.cases or [])]
    tasks += [
        ("false_positive_psf_mismatch", case)
        for case in list(args.false_positive_cases or [])
    ]
    if args.limit is not None:
        tasks = tasks[: args.limit]
    return tasks


def _launch(args: argparse.Namespace) -> List[Dict[str, Any]]:
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    gpus = [item.strip() for item in args.gpus.split(",") if item.strip()]
    tasks = _case_tasks(args)
    if not tasks:
        raise ValueError("No cases requested")
    if not gpus:
        raise ValueError("--gpus must contain at least one GPU index")
    all_candidate_rows: List[Dict[str, Any]] = []
    all_summary_rows: List[Dict[str, Any]] = []
    failures = []
    max_concurrent = max(1, int(args.max_concurrent))

    for batch_start in range(0, len(tasks), max_concurrent):
        batch = tasks[batch_start : batch_start + max_concurrent]
        processes = []
        for batch_index, (case_kind, run_name) in enumerate(batch):
            index = batch_start + batch_index
            gpu = gpus[index % len(gpus)]
            case_label = _case_label(run_name, case_kind)
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
                "--dataset-kind",
                args.dataset_kind,
                "--fit-mode",
                args.fit_mode,
                "--n-live-smooth",
                str(args.n_live_smooth),
                "--n-live-subhalo",
                str(args.n_live_subhalo),
                "--psf-scales",
                *[str(val) for val in args.psf_scales],
            ]
            if args.maxcall is not None:
                cmd.extend(["--maxcall", str(args.maxcall)])
            if args.use_jax:
                cmd.append("--use-jax")
            if args.resume:
                cmd.append("--resume")
            if args.fast_output:
                cmd.append("--fast-output")

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
            processes.append((case_label, gpu, process, log_file, log_path))
            print(
                f"Started {index + 1}/{len(tasks)} {case_kind} {case_label} "
                f"on GPU {gpu} (pid {process.pid})",
                flush=True,
            )

        for case_label, gpu, process, log_file, log_path in processes:
            rc = process.wait()
            log_file.close()
            print(f"Finished {case_label} on GPU {gpu} with exit code {rc}", flush=True)
            if rc != 0:
                failures.append((case_label, gpu, rc, log_path))

        for case_label, *_ in processes:
            case_dir = output_dir / "cases" / case_label
            candidate_path = case_dir / "candidate_rows.json"
            summary_path = case_dir / "marginalized_row.json"
            if candidate_path.exists():
                with candidate_path.open("r", encoding="utf-8") as handle:
                    all_candidate_rows.extend(json.load(handle))
            if summary_path.exists():
                with summary_path.open("r", encoding="utf-8") as handle:
                    all_summary_rows.append(json.load(handle))

        _write_csv(output_dir / "candidate_results.csv", all_candidate_rows)
        _write_csv(output_dir / "marginalized_results.csv", all_summary_rows)
        _write_json(
            output_dir / "run_summary.json",
            {
                "tasks": [{"case_kind": kind, "run_name": run} for kind, run in tasks],
                "n_expected": len(tasks),
                "n_cases_written": len(all_summary_rows),
                "n_candidate_rows_written": len(all_candidate_rows),
                "n_success": sum(row.get("status") == "success" for row in all_summary_rows),
                "n_failed": sum(row.get("status") != "success" for row in all_summary_rows),
                "gpus": gpus,
                "max_concurrent": max_concurrent,
                "psf_scales": [float(val) for val in args.psf_scales],
                "dataset_kind": args.dataset_kind,
                "fit_mode": args.fit_mode,
                "n_live_smooth": args.n_live_smooth,
                "n_live_subhalo": args.n_live_subhalo,
                "maxcall": args.maxcall,
                "use_jax": args.use_jax,
                "fast_output": args.fast_output,
                "failures": [
                    {
                        "case_label": label,
                        "gpu": gpu,
                        "return_code": rc,
                        "log_path": str(path.relative_to(REPO_ROOT)),
                    }
                    for label, gpu, rc, path in failures
                ],
            },
        )

    if failures:
        detail = "; ".join(f"{label} gpu={gpu} rc={rc} log={path}" for label, gpu, rc, path in failures)
        raise RuntimeError(f"PSF-marginalized validation failures: {detail}")
    return all_summary_rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--stage0-results", required=True)
    parser.add_argument("--config-dir", required=True)
    parser.add_argument("--gpus", default="0")
    parser.add_argument("--cases", nargs="*", default=None)
    parser.add_argument("--false-positive-cases", nargs="*", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max-concurrent", type=int, default=1)
    parser.add_argument("--psf-scales", nargs="+", type=float, default=[0.0, 0.5, 1.0, 1.5])
    parser.add_argument("--n-live-smooth", type=int, default=200)
    parser.add_argument("--n-live-subhalo", type=int, default=200)
    parser.add_argument("--maxcall", type=int, default=None)
    parser.add_argument("--dataset-kind", choices=("asimov", "noisy"), default="noisy")
    parser.add_argument("--fit-mode", choices=("fixed_template", "local_search"), default="local_search")
    parser.add_argument("--use-jax", action="store_true", default=False)
    parser.add_argument("--resume", action="store_true", default=False)
    parser.add_argument("--fast-output", action="store_true", default=False)
    parser.add_argument("--worker-case", default=None)
    parser.add_argument("--worker-case-kind", choices=("injected_subhalo", "false_positive_psf_mismatch"), default=None)
    parser.add_argument("--case-label", default=None)
    parser.add_argument("--gpu", default="0")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.worker_case:
        return _run_worker(args)
    rows = _launch(args)
    print(f"Wrote PSF-marginalized validation results for {len(rows)} cases to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
