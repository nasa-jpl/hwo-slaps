"""Small client utilities shared by the runnable examples."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import time


def positive_number(text):
    import argparse

    value = float(text)
    if not math.isfinite(value) or value <= 0:
        raise argparse.ArgumentTypeError("a positive finite number is required")
    return value


def verify_sei(directory):
    directory = Path(directory)
    verified, absent = {}, []
    for line in (directory / "SHA256SUMS").read_text().splitlines():
        expected, filename = line.split(maxsplit=1)
        target = directory / filename
        if not target.is_file():
            absent.append(filename)
            continue
        actual = hashlib.sha256(target.read_bytes()).hexdigest()
        if actual != expected:
            raise ValueError(f"SEI file hash differs: {filename}")
        verified[filename] = actual
    print("SEI verified:", ", ".join(verified))
    print("SEI manifest entries not shipped:", ", ".join(absent))
    return {"sha256": verified, "not_shipped": absent}


def require_lane(engine):
    if engine == "reference":
        return {"engine": engine, "lane": "cpu"}
    import jax

    devices = jax.devices()
    if not devices or any(device.platform != "gpu" for device in devices):
        raise RuntimeError("this JAX example requires an actual GPU; select its device before running")
    return {"engine": engine, "lane": "gpu", "devices": [str(device) for device in devices]}


def new_output(path):
    path = Path(path).expanduser().resolve()
    path.mkdir(parents=True, exist_ok=False)
    return path


def run_product(config, output, *, masses, engine, q_threshold, budget_s, command, noise_seed=None,
                check=None, extra=None, plot=False):
    from hwoslaps.analysis.reductions import summarize
    from hwoslaps.artifacts import save_forecast, save_observation, write_json
    from hwoslaps.fisher.api import Execution, forecast, prepare_forecast
    from hwoslaps.provenance import capture_provenance
    from hwoslaps.simulation import simulate

    started = time.perf_counter()
    lane = require_lane(engine)
    environment = capture_provenance(command=command)
    with prepare_forecast(config, execution=Execution(engine=engine)) as prepared:
        if check is not None:
            check(prepared)
        output = new_output(output)
        result = forecast(prepared, masses_msun=masses)
        summary = summarize(result, q_threshold=q_threshold)
        save_forecast(result, output / "forecast.npz")
        save_observation(prepared.observation, output / "expected.npz")
        if noise_seed is not None:
            noisy = simulate(prepared, subhalo=None, noise_seed=noise_seed)
            save_observation(noisy, output / "noisy.npz")
        if plot:
            from hwoslaps.plotting import plot_observation, plot_statistic_map
            import matplotlib.pyplot as plt

            for filename, axes in (
                ("observation.png", plot_observation(prepared.observation, "expected")),
                ("forecast.png", plot_statistic_map(result, result.detection_metric, mass_index=0)),
            ):
                try:
                    axes.figure.savefig(output / filename)
                finally:
                    plt.close(axes.figure)
        elapsed = time.perf_counter() - started
        record = {
            "command": list(command), "execution": lane, "environment": environment,
            "elapsed_s": elapsed, "budget_s": budget_s, "within_budget": elapsed <= budget_s,
            "config_digest": result.provenance["config_digest"],
            "comparison_digest": result.provenance["comparison_digest"],
            "input_files": result.provenance["file_digests"], "q_threshold": q_threshold,
            "masses_msun": result.masses_msun.tolist(), "positions_count": len(result.positions),
            "metric": result.detection_metric, "q_max": summary.q_max.tolist(),
            "sampling": dict(prepared.observation.sampling),
            "photometry": result.provenance["photometry"], "spectral": result.provenance["spectral"],
            "noise_seed": noise_seed, "extra": extra,
        }
        write_json(output / "run.json", record)
        print(json.dumps(record, indent=2, allow_nan=False))
        if elapsed > budget_s:
            raise RuntimeError(f"example exceeds its {budget_s} s budget: {elapsed:.3f} s; see {output / 'run.json'}")
        return result, record
