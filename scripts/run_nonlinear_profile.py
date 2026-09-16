#!/usr/bin/env python
"""Run a bound persistent replay worker or supervise its manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("spec")
    parser.add_argument("--supervise", action="store_true")
    args = parser.parse_args()
    if args.supervise:
        from hwoslaps.modeling.nonlinear.profile_execution import supervise

        supervise(args.spec)
        return
    import psutil
    from hwoslaps.modeling.nonlinear.profile_replay import atomic_json

    spec = json.loads(Path(args.spec).read_text())
    output = Path(spec["output"])
    output.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    atomic_json(
        output / "worker_started.json",
        {
            "pid": os.getpid(),
            "process_start": psutil.Process().create_time(),
            "spec_path": str(Path(args.spec).resolve()),
        },
    )
    status = "FAILED"
    error = None
    try:
        for filename, expected in spec["hashes"].items():
            if hashlib.sha256(Path(filename).read_bytes()).hexdigest() != expected:
                raise ValueError("Input/code hash mismatch: " + filename)
        import jax

        jax.config.update("jax_enable_x64", True)
        from hwoslaps.modeling.nonlinear.experimental_device import require_cuda_execution
        from hwoslaps.modeling.nonlinear.experimental_persistent import persistent_preparation
        from hwoslaps.modeling.nonlinear.profile_replay import ProfileReplayRunner, ProfileReplayValidator
        from hwoslaps.provenance import revision_provenance, revision_digest
        import yaml
        import run_nonlinear_validation

        atomic_json(output / "device.json", require_cuda_execution())
        revision = revision_provenance()
        atomic_json(output / "execution.json", {"revision": revision, "spec": spec})
        with persistent_preparation() as cache_stats:
            for job in spec["jobs"]:
                case_output = output / (job["system_id"] + "_" + job["arm"])
                if case_output.exists():
                    raise ValueError("Refusing to repeat an existing case output")
                case_output.mkdir()
                cfg = yaml.safe_load(Path(job["config"]).read_text())
                original_revision = cfg["stage0"]["code_revision"]
                cfg["stage0"]["code_revision"] = {
                    "git_hash": revision["git_hash"],
                    "git_dirty": revision["git_dirty"],
                    "sha256": revision_digest(revision),
                }
                staged = case_output / "execution_config.yaml"
                staged.write_text(yaml.safe_dump(cfg, sort_keys=False))
                replay = dict(job["replay"], case=json.loads(Path(job["case"]).read_text()))
                atomic_json(
                    case_output / "replay_provenance.json",
                    {
                        "original_revision": original_revision,
                        "execution_revision": revision,
                        "procedure": spec["procedure"],
                        "inputs": job,
                    },
                )

                def factory(settings, output_dir):
                    return ProfileReplayRunner(settings, output_dir, replay, spec["procedure"])

                run_nonlinear_validation.main(
                    [str(staged), job["positions"], job["arm"], str(case_output)],
                    runner_factory=factory,
                    validator_factory=ProfileReplayValidator,
                    artifact_prefix="profile_protocol",
                )
                case_result = json.loads((case_output / "profile_result.json").read_text())
                atomic_json(
                    case_output / "case_complete.json",
                    {
                        "execution_status": "COMPLETE",
                        "system_id": job["system_id"],
                        "arm": job["arm"],
                        "numerical_status": case_result["numerical_status"],
                        "profile_decision": case_result["profile_decision"],
                        "procedure": spec["procedure"],
                        "revision": revision,
                        "artifacts": {
                            str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                            for p in case_output.iterdir()
                            if p.is_file() and p.name != "case_complete.json"
                        },
                    },
                )
                atomic_json(output / "cache_stats.json", cache_stats)
        status = "COMPLETE"
    except BaseException as exc:
        error = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        atomic_json(
            output / "worker_exit.json",
            {
                "status": status,
                "error": error,
                "elapsed_s": time.monotonic() - started,
                "ended_unix": time.time(),
                "artifacts": {
                    str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                    for p in output.rglob("*.json")
                    if p.name != "worker_exit.json"
                },
            },
        )


if __name__ == "__main__":
    main()
