"""Cold-process forecast workloads and arrays for the performance comparison gates.

Run with one BLAS thread and no existing JAX compilation cache. Every repeat runs
in a new process with a fresh cache. B5 is B3 with Exponential lens light and a
second Exponential source; its exact component parameters are recorded below.
NPZ files hold the numeric comparison arrays; JSON records their configuration,
provenance and measured stages. Product artifact persistence has its own format.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from time import perf_counter

WORKLOADS = {
    "B1": ("p1", "reference", (1e7, 1e8, 1e9), 1),
    "B1w": ("p1", "reference", (1e7, 1e8, 1e9), 8),
    "B2": ("p1", "jax", (1e7, 1e8, 1e9), 1),
    "B3": ("paper_scale", "jax", (1e8,), 1),
    "B3m": ("paper_scale", "jax", (1e7, 1e8, 1e9), 1),
    "B5": ("paper_scale", "jax", (1e8,), 1),
}
STATISTICS = ("fisher_raw", "fisher_profiled", "sigma_amplitude", "q_asimov", "degradation",
              "amplitude_hat", "q_mismatch", "z_mismatch", "amplitude_spurious", "q_spurious", "z_spurious")
THREADS = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS")
COMPILE_EVENT = "/jax/core/compile/backend_compile_duration"


def workload_mapping(name):
    import yaml
    tree = Path(__file__).resolve().parents[1]
    path = (tree / "tests/fixtures/paper_parity/engine/p1_optical_matched.yaml" if WORKLOADS[name][0] == "p1"
            else tree / "tools/benchmarks/paper_scale.yaml")
    mapping = yaml.safe_load(path.read_text())
    if name == "B5":
        mapping["scene"]["lens"]["light"] = {
            "light": {"type": "Exponential", "centre": [0.0, 0.0], "ell_comps": [0.1, 0.0],
                      "intensity": 0.2, "effective_radius": 0.3}}
        mapping["scene"]["source"]["light"]["secondary"] = {
            "type": "Exponential", "centre": [0.02, 0.04], "ell_comps": [0.0, 0.1],
            "intensity": 0.5, "effective_radius": 0.07}
    return mapping, path.parent


def measure(args):
    import numpy as np
    from hwoslaps.config.schema import resolve_config
    from hwoslaps.identity import array_digest
    from hwoslaps.scene.builder import native_sampling_variation
    from hwoslaps.fisher.api import Execution, forecast, prepare_forecast
    from hwoslaps.fisher.engines.base import BankAccumulator
    import hwoslaps.fisher.api as forecast_api

    start = perf_counter()
    mapping, directory = workload_mapping(args.workload)
    config = resolve_config(mapping, base_dir=directory)
    seconds = {"config_load": perf_counter() - start}
    _, engine, masses, workers = WORKLOADS[args.workload]
    execution = Execution(engine=engine, reference_workers=workers, batch_size=args.batch_size)
    builds = []
    original_factory = forecast_api.make_engine
    def timed_factory(*positional, **keywords):
        started = perf_counter()
        built = original_factory(*positional, **keywords)
        builds.append(perf_counter() - started)
        return built
    forecast_api.make_engine = timed_factory
    start = perf_counter()
    try:
        prepared = prepare_forecast(config, execution=execution)
    finally:
        forecast_api.make_engine = original_factory
    seconds["prepare"] = perf_counter() - start
    seconds["engine_construction"] = sum(builds)
    start = perf_counter()
    native_sampling_variation(prepared.scene)
    seconds["sampling_diagnostic"] = perf_counter() - start
    seconds["prepare_without_sampling"] = seconds["prepare"] - seconds["sampling_diagnostic"]
    compile_events = []
    if engine == "jax":
        import jax
        import jax.monitoring
        def listener(event, duration, *positional, **keywords):
            if event == COMPILE_EVENT:
                compile_events.append(float(duration))
        jax.monitoring.register_event_duration_secs_listener(listener)
    positions = prepared.positions.positions_yx
    if args.estimate_nodes is not None:
        positions = positions[:args.estimate_nodes]
    if args.stage == "first-batch":
        positions, masses = positions[:args.batch_size], masses[:1]
    per_mass = []
    original_finish = BankAccumulator.finish
    mass_start = perf_counter()
    mass_compiles = len(compile_events)
    def timed_finish(bank):
        nonlocal mass_start, mass_compiles
        value = original_finish(bank)
        finished = perf_counter()
        per_mass.append({"mass_msun": masses[len(per_mass)],
                         "seconds": finished - mass_start if engine == "jax" else None,
                         "backend_compiles": len(compile_events) - mass_compiles})
        mass_start, mass_compiles = finished, len(compile_events)
        return value
    start = perf_counter()
    BankAccumulator.finish = timed_finish
    try:
        first = forecast(prepared, masses_msun=masses, positions=positions)
    finally:
        BankAccumulator.finish = original_finish
    seconds["forecast_first"] = perf_counter() - start
    seconds["first_call_backend_compile"] = sum(compile_events)
    start = perf_counter()
    warm = forecast(prepared, masses_msun=masses, positions=positions)
    seconds["forecast_warm"] = perf_counter() - start
    arrays = {"masses_msun": first.masses_msun, "positions_yx": first.positions_yx}
    for field in STATISTICS:
        left, right = getattr(first, field), getattr(warm, field)
        if left is not None:
            if not np.array_equal(left, right, equal_nan=True):
                raise RuntimeError(f"warm forecast changed {field}")
            arrays[field] = left
    args.out.mkdir(parents=True, exist_ok=True)
    np.savez(args.out / f"{args.workload}.npz", **arrays)
    memory = jax.devices()[0].memory_stats() if engine == "jax" and args.gpu else None
    record = {"schema": 1, "tool": "benchmark_forecast", "workload": args.workload,
              "definition": mapping, "engine": engine, "seconds": seconds, "per_mass": per_mass,
              "backend_compiles": len(compile_events), "nodes": len(positions),
              "steady_nodes_per_second": len(positions) * len(masses) / seconds["forecast_warm"],
              "outputs": {name: array_digest(value) for name, value in arrays.items()},
              "device_memory": memory, "provenance": dict(first.provenance),
              "environment": {"python": sys.version, "threads": {key: os.environ[key] for key in THREADS},
                              "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
                              "jax_version": jax.__version__ if engine == "jax" else None},
              "estimate_nodes": args.estimate_nodes}
    prepared.close()
    (args.out / f"{args.workload}.json").write_text(json.dumps(record, indent=2, sort_keys=True))
    return record


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workload", choices=WORKLOADS, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--gpu", action="store_true")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--estimate-nodes", type=int)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--stage", choices=("full", "first-batch"), default="full", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.child:
        measure(args)
        return 0
    if os.environ.get("JAX_COMPILATION_CACHE_DIR"):
        parser.error("unset JAX_COMPILATION_CACHE_DIR; every repeat requires a fresh cache")
    if any(os.environ.get(key) != "1" for key in THREADS):
        parser.error("set every BLAS thread variable to 1")
    if args.repeats < 1 or args.batch_size < 1 or (args.estimate_nodes is not None and args.estimate_nodes < 1):
        parser.error("repeats, batch-size and estimate-nodes must be positive")
    args.out.mkdir(parents=True, exist_ok=True)
    records = []
    for repeat in range(args.repeats):
        stages = {}
        for stage in ("first-batch", "full"):
            with tempfile.TemporaryDirectory(prefix="forecast-benchmark-", dir=os.environ.get("TMPDIR")) as cache:
                environment = dict(os.environ, JAX_COMPILATION_CACHE_DIR=cache, PYTHONDONTWRITEBYTECODE="1")
                if not args.gpu:
                    environment.update(CUDA_VISIBLE_DEVICES="", JAX_PLATFORMS="cpu")
                else:
                    environment.update(JAX_ENABLE_X64="1", XLA_PYTHON_CLIENT_PREALLOCATE="false")
                child_out = args.out / f"repeat-{repeat}" / stage
                command = [sys.executable, str(Path(__file__).resolve()), "--child", "--workload", args.workload,
                           "--out", str(child_out), "--batch-size", str(args.batch_size), "--stage", stage]
                if args.gpu:
                    command.append("--gpu")
                if args.estimate_nodes is not None:
                    command.extend(("--estimate-nodes", str(args.estimate_nodes)))
                subprocess.run(command, env=environment, check=True, timeout=540 if args.gpu else 3600)
                stages[stage] = json.loads((child_out / f"{args.workload}.json").read_text())
        record = stages["full"]
        record["first_batch"] = stages["first-batch"]
        if records and records[0]["outputs"] != record["outputs"]:
            raise RuntimeError("repeated cold forecasts changed result bytes")
        records.append(record)
    output = {"schema": 1, "workload": args.workload, "repeats": records}
    (args.out / f"{args.workload}.json").write_text(json.dumps(output, indent=2, sort_keys=True))
    (args.out / f"{args.workload}.npz").write_bytes((args.out / f"repeat-0/full/{args.workload}.npz").read_bytes())
    print(json.dumps({"workload": args.workload, "steady_nodes_per_second":
                      [record["steady_nodes_per_second"] for record in records]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
