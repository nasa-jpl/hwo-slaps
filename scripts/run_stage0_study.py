#!/usr/bin/env python
"""Run the Stage 0 SCDD/SPIE Fisher study grid."""

from __future__ import annotations

import argparse
import os
import contextlib
import csv
import hashlib
import importlib.metadata
import json
import subprocess
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import deepcopy
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

SCDD_Q_THRESHOLD = 10.0

CSV_COLUMNS = (
    "study_name",
    "sweep",
    "run_name",
    "status",
    "error",
    "runtime_s",
    "run_dir",
    "config_path",
    "config_hash",
    "git_hash",
    "python",
    "mass_msun",
    "subhalo_model",
    "subhalo_position_y",
    "subhalo_position_x",
    "psf_case",
    "psf_family",
    "psf_mode",
    "psf_amplitude",
    "psf_units",
    "global_seed",
    "fisher_mode",
    "q_f",
    "z_f",
    "delta_log_l_f_equiv",
    "detected_scdd",
    "local_p_one_sided",
    "local_degradation",
    "local_absorbed_fraction",
    "sigma_amplitude_profiled",
    "pixels_unmasked",
    "n_nuisance",
    "gram_condition_number",
    "map_num_positions",
    "map_median_z_f",
    "map_max_z_f",
    "map_median_q_f",
    "map_max_q_f",
    "map_detectable_ring_fraction",
    "psf_strehl",
    "psf_total_rms_nm",
    "psf_segment_hexike_present",
    "psf_kernel_shape",
    "psf_kernel_sum",
    "psf_kernel_peak",
    "psf_fwhm_mas",
)


class _Tee:
    def __init__(self, *streams):
        self._streams = streams

    def write(self, data: str) -> int:
        for stream in self._streams:
            stream.write(data)
        return len(data)

    def flush(self) -> None:
        for stream in self._streams:
            stream.flush()


def _repo_path(path_like: str | Path) -> Path:
    path = Path(path_like).expanduser()
    if path.is_absolute():
        return path
    return REPO_ROOT / path


def _load_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _write_yaml(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False)


def _json_safe(value: Any) -> Any:
    if is_dataclass(value):
        return _json_safe(asdict(value))
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


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(_json_safe(payload), handle, indent=2, sort_keys=True)


def _config_hash(config: Dict[str, Any]) -> str:
    rendered = yaml.safe_dump(config, sort_keys=True).encode("utf-8")
    return hashlib.sha256(rendered).hexdigest()[:16]


def _git_hash() -> Optional[str]:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=REPO_ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return None


def _package_versions() -> Dict[str, Optional[str]]:
    names = ("numpy", "scipy", "matplotlib", "pyyaml", "autolens", "autofit", "hcipy", "hwoslaps")
    versions: Dict[str, Optional[str]] = {}
    for name in names:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def _set_fisher_common(config: Dict[str, Any], *, mode: str, run_map: bool, mode_scan: bool, manifest: Dict[str, Any]) -> None:
    fisher = config["modeling"]["fisher"]
    fisher["mode"] = "both" if run_map else mode
    fisher["compute_psf_mode_scan"] = bool(mode_scan)
    fisher["map"]["num_angles"] = int(manifest.get("map", {}).get("num_angles", fisher["map"]["num_angles"]))
    fisher["map"]["offset_pixels"] = float(
        manifest.get("map", {}).get("offset_pixels", fisher["map"]["offset_pixels"])
    )
    fisher["map"]["explicit_positions_yx"] = None


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


def _set_segment_hexike(config: Dict[str, Any], *, segment: int, mode_noll: int, amplitude_nm: float) -> None:
    _set_perfect_psf(config)
    if float(amplitude_nm) == 0.0:
        return
    aberr = config["psf"]["aberrations"]
    aberr["enable_segment_hexikes"] = True
    aberr["segment_hexikes"] = {int(segment): {int(mode_noll): float(amplitude_nm)}}


def _expanded_runs(manifest: Dict[str, Any], baseline: Dict[str, Any]) -> List[Dict[str, Any]]:
    study_name = str(manifest["study_name"])
    runs: List[Dict[str, Any]] = []

    mass_sweep = manifest.get("mass_sweep", {})
    if mass_sweep.get("enabled", False):
        for mass in mass_sweep["masses"]:
            run_name = f"{study_name}_mass_{mass['label']}_perfect"
            config = deepcopy(baseline)
            config["run_name"] = run_name
            config["plotting"]["output_dir"] = str(manifest["output_root"])
            config["lensing"]["subhalo"]["mass"] = float(mass["value"])
            _set_perfect_psf(config)
            _set_fisher_common(
                config,
                mode="local",
                run_map=bool(mass.get("run_map", False)),
                mode_scan=False,
                manifest=manifest,
            )
            runs.append(
                {
                    "sweep": "perfect_mass",
                    "run_name": run_name,
                    "config": config,
                    "mass_msun": float(mass["value"]),
                    "psf_case": "perfect",
                    "psf_family": "none",
                    "psf_mode": "none",
                    "psf_amplitude": 0.0,
                    "psf_units": "",
                }
            )

    psf_sweep = manifest.get("psf_sweep", {})
    if psf_sweep.get("enabled", False):
        pivot_mass = psf_sweep["pivot_mass"]
        segment = int(psf_sweep["segment"])
        mode_noll = int(psf_sweep["mode_noll"])
        map_amplitudes = {float(val) for val in psf_sweep.get("map_amplitudes", [])}
        scan_amplitudes = {float(val) for val in psf_sweep.get("mode_scan_amplitudes", [])}
        for amplitude in psf_sweep["amplitudes"]:
            amp = float(amplitude)
            amp_label = str(amp).replace(".", "p").replace("-", "m")
            run_name = f"{study_name}_hexike_s{segment}_n{mode_noll}_a{amp_label}nm_{pivot_mass['label']}"
            config = deepcopy(baseline)
            config["run_name"] = run_name
            config["plotting"]["output_dir"] = str(manifest["output_root"])
            config["lensing"]["subhalo"]["mass"] = float(pivot_mass["value"])
            _set_segment_hexike(config, segment=segment, mode_noll=mode_noll, amplitude_nm=amp)
            _set_fisher_common(
                config,
                mode="local",
                run_map=amp in map_amplitudes,
                mode_scan=amp in scan_amplitudes,
                manifest=manifest,
            )
            runs.append(
                {
                    "sweep": "segment_hexike_amplitude",
                    "run_name": run_name,
                    "config": config,
                    "mass_msun": float(pivot_mass["value"]),
                    "psf_case": "perfect" if amp == 0.0 else "segment_hexike",
                    "psf_family": str(psf_sweep["family"]),
                    "psf_mode": f"segment_{segment}_noll_{mode_noll}",
                    "psf_amplitude": amp,
                    "psf_units": str(psf_sweep["units"]),
                }
            )

    return runs


def _run_fisher_config(config: Dict[str, Any], *, verbose: bool):
    from hwoslaps.config.validation import validate_or_raise
    from hwoslaps.lensing import generate_lensing_system
    from hwoslaps.observation import generate_observation
    from hwoslaps.psf import generate_psf_system
    from hwoslaps.modeling.generator_fisher import perform_fisher_detection
    from hwoslaps.plotting import generate_all_plots

    validate_or_raise(config)

    psf_data = generate_psf_system(config["psf"], full_config=config)

    baseline_config = deepcopy(config)
    baseline_config["lensing"]["subhalo"]["enabled"] = False
    test_config = deepcopy(config)
    test_config["lensing"]["subhalo"]["enabled"] = True

    lensing_baseline = generate_lensing_system(baseline_config["lensing"], full_config=baseline_config)
    obs_baseline = generate_observation(
        lensing_data=lensing_baseline,
        psf_data=psf_data,
        observation_config=baseline_config["observation"],
        full_config=baseline_config,
    )
    lensing_test = generate_lensing_system(test_config["lensing"], full_config=test_config)
    obs_test = generate_observation(
        lensing_data=lensing_test,
        psf_data=psf_data,
        observation_config=test_config["observation"],
        full_config=test_config,
    )
    fisher_data = perform_fisher_detection(
        observation_baseline=obs_baseline,
        observation_test=obs_test,
        lensing_baseline=lensing_baseline,
        lensing_test=lensing_test,
        psf_data=psf_data,
        detection_config=config["modeling"],
        full_config=config,
    )

    if config["plotting"]["enabled"]:
        context = {
            "mode": "detection",
            "has_subhalo": lensing_test.has_subhalo,
            "lensing_data": lensing_test,
            "psf_data": psf_data,
            "obs_data": obs_baseline,
            "detection_data": fisher_data,
            "obs_baseline": obs_baseline,
            "obs_test": obs_test,
            "run_name": config["run_name"],
        }
        generate_all_plots(context, config["plotting"], verbose=verbose)

    return psf_data, lensing_test, fisher_data


def _extract_row(
    *,
    manifest: Dict[str, Any],
    run: Dict[str, Any],
    run_dir: Path,
    config_path: Path,
    config_hash: str,
    git_hash: Optional[str],
    runtime_s: float,
    status: str,
    error: Optional[str],
    psf_data: Any = None,
    lensing_test: Any = None,
    fisher_data: Any = None,
) -> Dict[str, Any]:
    row = {key: None for key in CSV_COLUMNS}
    row.update(
        {
            "study_name": manifest["study_name"],
            "sweep": run["sweep"],
            "run_name": run["run_name"],
            "status": status,
            "error": error,
            "runtime_s": runtime_s,
            "run_dir": str(run_dir.relative_to(REPO_ROOT)),
            "config_path": str(config_path.relative_to(REPO_ROOT)),
            "config_hash": config_hash,
            "git_hash": git_hash,
            "python": sys.version.split()[0],
            "mass_msun": run["mass_msun"],
            "subhalo_model": run["config"]["lensing"]["subhalo"]["model"],
            "psf_case": run["psf_case"],
            "psf_family": run["psf_family"],
            "psf_mode": run["psf_mode"],
            "psf_amplitude": run["psf_amplitude"],
            "psf_units": run["psf_units"],
            "global_seed": run["config"]["global_seed"],
            "fisher_mode": run["config"]["modeling"]["fisher"]["mode"],
        }
    )

    if lensing_test is not None and getattr(lensing_test, "subhalo_position", None) is not None:
        row["subhalo_position_y"] = float(lensing_test.subhalo_position[0])
        row["subhalo_position_x"] = float(lensing_test.subhalo_position[1])

    if fisher_data is not None:
        row["pixels_unmasked"] = int(fisher_data.pixels_unmasked)
        row["n_nuisance"] = int(fisher_data.n_nuisance)
        row["gram_condition_number"] = float(fisher_data.gram_condition_number)
        local = fisher_data.local
        if local is not None:
            q_f = local.q_asimov_local if local.q_asimov_local is not None else local.delta_chi2_profiled
            z_f = local.z_asimov_local if local.z_asimov_local is not None else local.snr_asimov
            row["q_f"] = float(q_f)
            row["z_f"] = float(z_f)
            row["delta_log_l_f_equiv"] = 0.5 * float(q_f)
            row["detected_scdd"] = bool(float(q_f) > SCDD_Q_THRESHOLD)
            row["local_p_one_sided"] = local.local_p_one_sided
            row["local_degradation"] = local.degradation
            row["local_absorbed_fraction"] = local.absorbed_fraction
            row["sigma_amplitude_profiled"] = local.sigma_amplitude_profiled
        fmap = fisher_data.map
        if fmap is not None:
            q_values = (
                np.asarray(fmap.q_asimov_local_by_position, dtype=float)
                if fmap.q_asimov_local_by_position is not None
                else np.asarray(fmap.delta_chi2_profiled_by_position, dtype=float)
            )
            row["map_num_positions"] = int(fmap.num_positions)
            row["map_median_z_f"] = float(fmap.median_snr_asimov)
            row["map_max_z_f"] = float(fmap.max_snr_asimov)
            row["map_median_q_f"] = float(np.median(q_values))
            row["map_max_q_f"] = float(np.max(q_values))
            row["map_detectable_ring_fraction"] = float(np.count_nonzero(q_values > SCDD_Q_THRESHOLD) / q_values.size)

    if psf_data is not None:
        from hwoslaps.psf.utils import pyauto_kernel_native

        kernel = pyauto_kernel_native(psf_data.kernel)
        row["psf_strehl"] = psf_data.strehl_ratio
        row["psf_total_rms_nm"] = float(psf_data.total_rms_nm)
        row["psf_segment_hexike_present"] = bool(psf_data.has_segment_hexikes)
        row["psf_kernel_shape"] = "x".join(str(dim) for dim in kernel.shape)
        row["psf_kernel_sum"] = float(np.sum(kernel))
        row["psf_kernel_peak"] = float(np.max(kernel))
        row["psf_fwhm_mas"] = psf_data.fwhm_mas

    return row


def _write_results_csv(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(_json_safe(row))


def _apply_worker_thread_caps(worker_threads: int) -> None:
    value = str(int(worker_threads))
    for env_name in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "BLIS_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "NUMEXPR_MAX_THREADS",
    ):
        os.environ[env_name] = value


def _execute_run_worker(payload: Dict[str, Any]) -> Dict[str, Any]:
    _apply_worker_thread_caps(int(payload["worker_threads"]))

    manifest = payload["manifest"]
    run = payload["run"]
    run_dir = Path(payload["run_dir"])
    config_path = Path(payload["config_path"])
    config_hash = payload["config_hash"]
    git_hash = payload["git_hash"]
    verbose = bool(payload["verbose"])
    worker_index = payload["worker_index"]

    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / "run.log"
    start = time.time()
    psf_data = None
    lensing_test = None
    fisher_data = None
    status = "success"
    error = None
    with log_path.open("w", encoding="utf-8", buffering=1) as log_file:
        with contextlib.redirect_stdout(log_file), contextlib.redirect_stderr(log_file):
            print(f"Run log: {log_path}")
            print(f"Config snapshot: {config_path}")
            print(f"Worker index: {worker_index}")
            print(f"Worker thread cap: {payload['worker_threads']}")
            try:
                psf_data, lensing_test, fisher_data = _run_fisher_config(
                    run["config"],
                    verbose=verbose,
                )
            except Exception as exc:
                status = "failed"
                error = str(exc)
                traceback.print_exc()
    runtime_s = time.time() - start

    _write_json(
        run_dir / "result_summary.json",
        {
            "status": status,
            "error": error,
            "runtime_s": runtime_s,
            "config_hash": config_hash,
            "git_hash": git_hash,
            "fisher": fisher_data,
        },
    )

    row = _extract_row(
        manifest=manifest,
        run=run,
        run_dir=run_dir,
        config_path=config_path,
        config_hash=config_hash,
        git_hash=git_hash,
        runtime_s=runtime_s,
        status=status,
        error=error,
        psf_data=psf_data,
        lensing_test=lensing_test,
        fisher_data=fisher_data,
    )
    row["_order"] = payload["order"]
    return row


def _auto_worker_threads(workers: int) -> int:
    cpu_count = os.cpu_count() or 1
    return max(1, cpu_count // max(1, int(workers)))


def run_manifest(
    manifest_path: Path,
    *,
    dry_run: bool,
    limit: Optional[int],
    no_plots: bool,
    verbose: bool,
    workers: int,
    worker_threads: Optional[int],
) -> List[Dict[str, Any]]:
    manifest = _load_yaml(manifest_path)
    baseline_path = _repo_path(manifest["baseline_config"])
    baseline = _load_yaml(baseline_path)
    output_root = _repo_path(manifest["output_root"])
    git_hash = _git_hash()
    versions = _package_versions()
    runs = _expanded_runs(manifest, baseline)
    if limit is not None:
        runs = runs[: int(limit)]

    workers = max(1, int(workers))
    if dry_run:
        workers = 1
    if workers > len(runs) and runs:
        workers = len(runs)
    worker_threads_final = (
        int(worker_threads)
        if worker_threads is not None
        else _auto_worker_threads(workers)
    )
    if worker_threads_final <= 0:
        raise ValueError("--worker-threads must be positive")

    rows: List[Dict[str, Any]] = []
    worker_payloads: List[Dict[str, Any]] = []
    generated_config_dir = output_root / "generated_configs"
    results_path = output_root / "results.csv"
    manifest_snapshot_path = output_root / "manifest_used.yaml"
    _write_yaml(manifest_snapshot_path, manifest)
    _write_json(
        output_root / "study_provenance.json",
        {
            "manifest": str(manifest_path.relative_to(REPO_ROOT)),
            "baseline_config": str(baseline_path.relative_to(REPO_ROOT)),
            "git_hash": git_hash,
            "python_executable": sys.executable,
            "python_version": sys.version,
            "package_versions": versions,
            "command_line": sys.argv,
            "dry_run": dry_run,
            "workers": workers,
            "worker_threads": worker_threads_final,
        },
    )

    for index, run in enumerate(runs, start=1):
        config = deepcopy(run["config"])
        if no_plots:
            config["plotting"]["enabled"] = False
        config_hash = _config_hash(config)
        run["config"] = config
        run_name = run["run_name"]
        run_dir = output_root / run_name
        config_path = generated_config_dir / f"{run_name}.yaml"
        _write_yaml(config_path, config)

        print(f"[queued {index}/{len(runs)}] {run_name}")
        if dry_run:
            row = _extract_row(
                manifest=manifest,
                run=run,
                run_dir=run_dir,
                config_path=config_path,
                config_hash=config_hash,
                git_hash=git_hash,
                runtime_s=0.0,
                status="dry_run",
                error=None,
            )
            row["_order"] = index
            rows.append(row)
            continue

        worker_payloads.append(
            {
                "manifest": manifest,
                "run": run,
                "run_dir": str(run_dir),
                "config_path": str(config_path),
                "config_hash": config_hash,
                "git_hash": git_hash,
                "verbose": verbose,
                "worker_threads": worker_threads_final,
                "worker_index": ((index - 1) % workers) + 1,
                "order": index,
            }
        )

    if dry_run:
        rows = sorted(rows, key=lambda row: row["_order"])
        _write_results_csv(results_path, rows)
        print(f"Wrote aggregate results: {results_path}")
        return rows

    print(
        f"Executing {len(worker_payloads)} runs with {workers} workers "
        f"and {worker_threads_final} threads/worker."
    )
    if workers == 1:
        for payload in worker_payloads:
            row = _execute_run_worker(payload)
            rows.append(row)
            rows = sorted(rows, key=lambda item: item["_order"])
            _write_results_csv(results_path, rows)
            print(
                f"[done {len(rows)}/{len(worker_payloads)}] "
                f"{row['run_name']} status={row['status']} runtime={float(row['runtime_s']):.1f}s"
            )
            if row["status"] != "success":
                break
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            future_to_name = {
                executor.submit(_execute_run_worker, payload): payload["run"]["run_name"]
                for payload in worker_payloads
            }
            for future in as_completed(future_to_name):
                run_name = future_to_name[future]
                try:
                    row = future.result()
                except Exception as exc:
                    row = {key: None for key in CSV_COLUMNS}
                    row.update(
                        {
                            "study_name": manifest["study_name"],
                            "run_name": run_name,
                            "status": "failed",
                            "error": repr(exc),
                            "runtime_s": 0.0,
                            "_order": len(rows) + 10_000,
                        }
                    )
                rows.append(row)
                rows = sorted(rows, key=lambda item: item["_order"])
                _write_results_csv(results_path, rows)
                print(
                    f"[done {len(rows)}/{len(worker_payloads)}] "
                    f"{row['run_name']} status={row['status']} runtime={float(row['runtime_s'] or 0.0):.1f}s"
                )
                if row["status"] != "success":
                    print(f"Failure in {row['run_name']}: {row['error']}")

    rows = sorted(rows, key=lambda row: row["_order"])
    _write_results_csv(results_path, rows)
    print(f"Wrote aggregate results: {results_path}")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        default="scratch/study/stage0_manifest.yaml",
        help="Stage 0 manifest path.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Expand configs and write empty aggregate rows only.")
    parser.add_argument("--limit", type=int, default=None, help="Run only the first N expanded cases.")
    parser.add_argument("--no-plots", action="store_true", help="Disable per-run plot generation.")
    parser.add_argument("--quiet", action="store_true", help="Reduce plot registry verbosity.")
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of independent study cases to run in parallel.",
    )
    parser.add_argument(
        "--worker-threads",
        type=int,
        default=None,
        help="Thread cap for BLAS/OpenMP/NumExpr inside each worker. Defaults to CPU count divided by workers.",
    )
    args = parser.parse_args()

    run_manifest(
        _repo_path(args.manifest),
        dry_run=args.dry_run,
        limit=args.limit,
        no_plots=args.no_plots,
        verbose=not args.quiet,
        workers=args.workers,
        worker_threads=args.worker_threads,
    )


if __name__ == "__main__":
    main()
