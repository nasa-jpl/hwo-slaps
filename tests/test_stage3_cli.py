"""CPU-only tests for the Stage 3 v7 dry-run manifest validator."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import time

import pytest


def load_cli():
    import importlib.util

    path = Path(__file__).parents[1] / "studies/rasti/scripts/validate_stage3_manifest.py"
    spec = importlib.util.spec_from_file_location("stage3_manifest_cli", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module, path


def manifest_fixture(tmp_path):
    cli, source = load_cli()
    root = tmp_path / "stage"
    (root / "attempts").mkdir(parents=True)
    spec_path = root / "job.json"
    output = root / "attempts/job"
    spec_path.write_text(
        json.dumps(
            {
                "output": str(output),
                "prepared_only": False,
                "require_cuda_execution": True,
                "entrypoint": str(source),
                "hashes": {str(source): hashlib.sha256(source.read_bytes()).hexdigest()},
            }
        )
    )
    deadline = root / "deadline.json"
    from studies.rasti.campaign.profile_execution import clock_epoch

    epoch = clock_epoch()
    now = time.monotonic()
    deadline.write_text(
        json.dumps(
            {
                "clock_epoch": epoch,
                "captured_monotonic": now,
                "admission_stop_monotonic": now + 100,
                "hard_stop_monotonic": now + 200,
            }
        )
    )
    manifest = root / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "execution_policy_version": "stage3_v7",
                "task_root": str(root),
                "gpus": [0, 1, 2, 3],
                "authorized_gpu_limit": 4,
                "authorized_worker_limit": 12,
                "max_workers": 1,
                "max_workers_per_gpu": 3,
                "admission_memory_fraction": 0.85,
                "runtime_gpu_memory_fraction": 0.90,
                "cap_seconds": 1000,
                "jobs": [
                    {
                        "key": "job",
                        "spec": str(spec_path),
                        "gpu": 0,
                        "peak_mib": 51200,
                        "memory_class": "790",
                        "memory_profile_id": "stage3_b200_790_v1",
                        "image_shape": [790, 790],
                        "kernel_shape": [51, 51],
                        "batch_size": 32,
                        "precision": "float64",
                        "timeout_seconds": 10,
                    }
                ],
            }
        )
    )
    return cli, manifest, deadline


def test_stage3_cli_dry_run_validates_source_and_memory_contract(tmp_path, capsys):
    cli, manifest, deadline = manifest_fixture(tmp_path)
    assert cli.main([str(manifest), "--deadline", str(deadline)]) == 0
    output = json.loads(capsys.readouterr().out)
    assert output["status"] == "DRY_RUN_VALID"
    assert output["policy"]["per_gpu_limit"] == 3
    assert output["max_admission_wave_seconds"] == 70
    assert output["full_manifest_worst_case_seconds"] == 70


def test_stage3_cli_rejects_preexisting_output(tmp_path):
    cli, manifest, deadline = manifest_fixture(tmp_path)
    (tmp_path / "stage/attempts/job").mkdir()
    with pytest.raises(ValueError, match="existing job output"):
        cli.main([str(manifest), "--deadline", str(deadline)])


def test_stage3_cli_wave_cap_does_not_treat_catalog_as_one_reservation():
    cli, _ = load_cli()
    jobs = [
        {"key": str(index), "gpu": index % 4, "timeout_seconds": 100}
        for index in range(20)
    ]
    wave, selected = cli.max_admission_wave(jobs, maximum=8, per_gpu=2)
    assert wave == 8 * 160
    assert len(selected) == 8
    assert sum(job["timeout_seconds"] + 60 for job in jobs) > wave
