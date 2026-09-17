#!/usr/bin/env python3
"""Dry-run validator for an opt-in Stage 3 v7 worker manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hwoslaps.modeling.nonlinear.profile_execution import (  # noqa: E402
    STAGE3_POLICY_VERSION,
    stage3_policy,
    validate_deadline,
    validate_stage3_job,
)


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(2**20), b""):
            h.update(block)
    return h.hexdigest()


def read(path: Path):
    return json.loads(path.read_text())


def max_admission_wave(jobs, maximum, per_gpu):
    """Bound one simultaneous admission wave, rather than the whole catalog."""
    selected = []
    counts = {}
    exclusive = set()
    for job in sorted(jobs, key=lambda item: item["timeout_seconds"], reverse=True):
        if len(selected) >= maximum:
            break
        gpu = job["gpu"]
        if counts.get(gpu, 0) >= per_gpu or gpu in exclusive:
            continue
        if job.get("exclusive_gpu") and counts.get(gpu, 0):
            continue
        selected.append(job)
        counts[gpu] = counts.get(gpu, 0) + 1
        if job.get("exclusive_gpu"):
            exclusive.add(gpu)
    return sum(job["timeout_seconds"] + 60 for job in selected), selected


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--deadline", type=Path, required=True)
    args = parser.parse_args(argv)
    manifest_path = args.manifest.expanduser().resolve()
    deadline_path = args.deadline.expanduser().resolve()
    manifest = read(manifest_path)
    deadline = read(deadline_path)
    if manifest.get("execution_policy_version") != STAGE3_POLICY_VERSION:
        raise ValueError("manifest must opt into stage3_v7")
    policy = stage3_policy(manifest)
    if manifest.get("max_workers") > policy["worker_limit"]:
        raise ValueError("manifest max_workers exceeds authorized worker limit")
    validate_deadline(deadline)
    jobs = manifest.get("jobs")
    if not isinstance(jobs, list) or not jobs:
        raise ValueError("manifest jobs must be a non-empty list")
    root = Path(manifest["task_root"]).expanduser().resolve()
    if not root.is_dir():
        raise ValueError(f"task root is missing: {root}")
    keys = set()
    specs = set()
    outputs = set()
    reservation_seconds = 0
    for job in jobs:
        key = job.get("key")
        if not isinstance(key, str) or key in keys:
            raise ValueError(f"duplicate or invalid job key: {key}")
        keys.add(key)
        spec_path = Path(job["spec"]).expanduser().resolve()
        if not spec_path.is_relative_to(root) or spec_path in specs or not spec_path.is_file():
            raise ValueError(f"invalid or duplicate job spec: {key}")
        specs.add(spec_path)
        spec = read(spec_path)
        output = Path(spec["output"]).expanduser().resolve()
        if not output.is_relative_to(root / "attempts") or output in outputs or output.exists():
            raise ValueError(f"invalid, duplicate, or existing job output: {key}")
        outputs.add(output)
        if job.get("gpu") not in manifest["gpus"]:
            raise ValueError(f"job GPU is outside manifest allocation: {key}")
        validate_stage3_job(job, policy)
        timeout = job.get("timeout_seconds")
        if isinstance(timeout, bool) or not isinstance(timeout, int) or timeout <= 0:
            raise ValueError(f"invalid timeout: {key}")
        reservation_seconds += timeout + 60
        if spec.get("prepared_only") is not False:
            raise ValueError(f"job spec is still preparation-only: {key}")
        if spec.get("require_cuda_execution") is not True:
            raise ValueError(f"job spec lacks mandatory CUDA execution flag: {key}")
        hashes = spec.get("hashes")
        if not isinstance(hashes, dict) or not hashes:
            raise ValueError(f"job spec lacks source hashes: {key}")
        for filename, expected in hashes.items():
            path = Path(filename)
            if not path.is_file() or digest(path).lower() != str(expected).lower():
                raise ValueError(f"source hash mismatch: {filename}")
    cap = manifest.get("cap_seconds")
    maximum_wave_seconds, wave_jobs = max_admission_wave(
        jobs,
        manifest["max_workers"],
        manifest["max_workers_per_gpu"],
    )
    if (
        isinstance(cap, bool)
        or not isinstance(cap, (int, float))
        or not math.isfinite(cap)
        or maximum_wave_seconds > cap
    ):
        raise ValueError("one simultaneous worker-admission wave exceeds finite manifest cap")
    print(
        json.dumps(
            {
                "status": "DRY_RUN_VALID",
                "policy": policy,
                "manifest": str(manifest_path),
                "deadline": str(deadline_path),
                "jobs": len(jobs),
                "max_admission_wave_seconds": maximum_wave_seconds,
                "max_admission_wave_jobs": [job["key"] for job in wave_jobs],
                "full_manifest_worst_case_seconds": reservation_seconds,
                "full_manifest_exceeds_cap": reservation_seconds > cap,
                "cap_seconds": cap,
                "physical_gpus": manifest["gpus"],
                "existing_ledger": (root / "state/budget.json").is_file(),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
