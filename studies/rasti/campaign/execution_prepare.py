"""
Prepare and explicitly activate a hash-bound fresh nonlinear execution batch.

Preparation is CPU-only. It never writes a deadline, approval, or launch state.
Activation is a separate operation against an exact, externally supplied
approval.
"""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import math
import re
import time
import uuid
from pathlib import Path

from studies.rasti.campaign.profile_execution import (
    STAGE3_MEMORY_PROFILE_REGISTRY,
    clock_epoch,
    stage3_policy,
    supervise,
    validate_deadline,
    validate_stage3_job,
)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(2**20), b""):
            h.update(chunk)
    return h.hexdigest()


def read(path):
    value = json.loads(Path(path).read_text())
    if not isinstance(value, dict):
        raise TypeError(f"expected JSON object: {path}")
    return value


def write_new(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def positive(value, name):
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value <= 0
    ):
        raise ValueError(f"{name} must be finite and positive")
    return value


def source_bindings(source_worktree, runtime_worktree):
    """
    Bind all production Python/config code, including the activation gateway.
    """
    source = Path(source_worktree).resolve()
    runtime = Path(runtime_worktree)
    paths = set()
    for directory, suffixes in (
        ("src", {".py"}),
        ("scripts", {".py"}),
        ("studies", {".py"}),
        ("configs", {".yaml", ".yml"}),
    ):
        paths.update(
            path for path in (source / directory).rglob("*") if path.is_file() and path.suffix in suffixes
        )
    if not paths:
        raise ValueError("source worktree has no bindable sources")
    return {str(runtime / path.relative_to(source)): digest(path) for path in sorted(paths)}


def scientific_config_sha256(path):
    import yaml

    config = yaml.safe_load(Path(path).read_text())
    config = copy.deepcopy(config)
    config.get("stage0", {}).pop("code_revision", None)
    return hashlib.sha256(
        json.dumps(config, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def resource_job(spec):
    """A lower reservation requires a measured, exact-case resource receipt."""
    conservative = {
        "memory_class": "unmeasured_conservative",
        "peak_mib": 140000,
        "exclusive_gpu": True,
    }
    receipt_path = spec.get("memory_profile_receipt")
    if receipt_path is None:
        return conservative
    hashes = spec.get("hashes", {})
    if receipt_path not in hashes or digest(receipt_path) != hashes[receipt_path]:
        raise ValueError("memory profile receipt is not hash-bound")
    receipt = read(receipt_path)
    if receipt.get("status") != "MEASURED" or receipt.get("case_id") != spec.get("case_id"):
        raise ValueError("memory receipt must be a measured exact-case binding")
    if receipt.get("config_sha256") != hashes.get(spec["config"]):
        measured_config = receipt.get("measured_config_path")
        if (
            not isinstance(measured_config, str)
            or measured_config not in hashes
            or digest(measured_config) != hashes[measured_config]
            or receipt.get("config_scientific_sha256") != scientific_config_sha256(spec["config"])
            or receipt.get("config_scientific_sha256") != scientific_config_sha256(measured_config)
        ):
            raise ValueError("memory receipt config binding differs beyond code_revision")
    matches = [
        (key, value)
        for key, value in STAGE3_MEMORY_PROFILE_REGISTRY.items()
        if value["registry_id"] == receipt.get("registry_id")
    ]
    if len(matches) != 1:
        raise ValueError("memory receipt registry differs")
    memory_class, profile = matches[0]
    for key in ("image_shape", "kernel_shape", "batch_size", "precision"):
        if receipt.get(key) != profile[key]:
            raise ValueError(f"measured memory profile differs for {key}")
    if receipt.get("registry_id") != profile["registry_id"]:
        raise ValueError("memory receipt registry differs")
    evidence = receipt.get("measurement_artifacts")
    if not isinstance(evidence, dict) or not evidence:
        raise ValueError("measured memory profile lacks hash-bound measurement artifacts")
    for path, expected in evidence.items():
        if hashes.get(path) != expected or digest(path) != expected:
            raise ValueError("memory measurement artifact mismatch")
    import yaml

    config = yaml.safe_load(Path(spec["config"]).read_text())
    if config.get("lensing", {}).get("grid", {}).get("shape") != profile["image_shape"]:
        raise ValueError("memory profile image shape disagrees with actual config")
    return {
        "memory_class": memory_class,
        "peak_mib": profile["peak_mib"],
        "memory_profile_id": profile["registry_id"],
        "exclusive_gpu": False,
        **{key: profile[key] for key in ("image_shape", "kernel_shape", "batch_size", "precision")},
    }


def prepare(
    spec_paths,
    output_dir,
    *,
    task_root,
    source_worktree,
    worktree,
    python,
    gpus,
    wallclock_seconds,
    worker_wall_seconds,
    timeout_seconds,
    admission_buffer_seconds=300,
    synthetic_example=False,
):
    """
    Write a movable review package; all future runtime paths are explicit.
    """
    output_dir = Path(output_dir).resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError("preparation output must be empty")
    task_root = Path(task_root)
    if not task_root.is_absolute() or not Path(worktree).is_absolute() or not Path(python).is_absolute():
        raise ValueError("runtime task root, worktree and Python must be absolute")
    for value, name in (
        (wallclock_seconds, "wallclock_seconds"),
        (worker_wall_seconds, "worker_wall_seconds"),
        (timeout_seconds, "timeout_seconds"),
        (admission_buffer_seconds, "admission_buffer_seconds"),
    ):
        positive(value, name)
    if not isinstance(timeout_seconds, int):
        raise TypeError("timeout_seconds must be integer")
    if wallclock_seconds <= max(timeout_seconds + 60, admission_buffer_seconds):
        raise ValueError("wallclock budget cannot admit one job and its closing margin")
    bindings = source_bindings(source_worktree, worktree)
    manifest = {
        "schema_version": 1,
        "status": "PREPARED_PENDING_APPROVAL",
        "synthetic_example": synthetic_example,
        "prepared_only": True,
        "execution_policy_version": "stage3_v7",
        "task_root": str(task_root),
        "worktree": str(worktree),
        "python": str(python),
        "worker_entrypoint": str(Path(worktree) / "studies/rasti/scripts/run_nonlinear_production.py"),
        "campaign_uuid": str(uuid.uuid4()),
        "gpus": list(gpus),
        "authorized_gpu_limit": len(gpus),
        "authorized_worker_limit": len(gpus) * 3,
        "max_workers": len(gpus) * 3,
        "max_workers_per_gpu": 3,
        "admission_memory_fraction": 0.85,
        "runtime_gpu_memory_fraction": 0.90,
        "max_owned_rss_gib": 384,
        "task_disk_limit_gib": 512,
        "min_disk_free_gib": 100,
        "disk_check_interval_seconds": 30,
        "cap_seconds": worker_wall_seconds,
        "proposed_wallclock_seconds": wallclock_seconds,
        "admission_buffer_seconds": admission_buffer_seconds,
        "source_hashes": bindings,
        "jobs": [],
        "approval": None,
        "deadline": None,
        "notes": "Worker limits are ceilings; memory admission can require one exclusive worker per card.",
    }
    policy = stage3_policy(manifest)
    seen = set()
    for index, path in enumerate(spec_paths):
        spec = read(path)
        case_id = spec.get("case_id")
        if not isinstance(case_id, str) or not case_id or case_id in seen:
            raise ValueError("spec case IDs must be nonempty and unique")
        seen.add(case_id)
        if spec.get("pending") or spec.get("status", "").startswith("PENDING"):
            raise ValueError("dependent case has not been materialized")
        for key in (
            "config",
            "positions",
            "release_freeze_path",
            "release_freeze_sha256",
            "catalog_sha256",
        ):
            if not isinstance(spec.get(key), str) or not spec[key]:
                raise ValueError(f"materialized spec missing {key}")
        if not isinstance(spec.get("hashes"), dict) or not spec["hashes"]:
            raise ValueError("materialized spec needs input hashes")
        for key in ("config", "positions", "release_freeze_path"):
            if spec[key] not in spec["hashes"]:
                raise ValueError(f"materialized {key} is not hash-bound")
        key = f"{index:04d}_" + re.sub(r"[^a-zA-Z0-9_.-]", "_", case_id)
        relative = f"specs/{key}.json"
        spec = copy.deepcopy(spec)
        old_approval = spec.get("approval_receipt")
        if old_approval:
            spec["hashes"].pop(old_approval, None)
        spec.update(
            prepared_only=True,
            require_cuda_execution=True,
            approval_receipt=None,
            approval_receipt_sha256=None,
            output=str(task_root / "attempts" / key),
            case_output=str(task_root / "attempts" / key / "case"),
        )
        job = {
            "key": key,
            "case_id": case_id,
            "prepared_spec": relative,
            "spec": str(task_root / "activated/specs" / f"{key}.json"),
            "gpu": gpus[index % len(gpus)],
            "timeout_seconds": timeout_seconds,
            "expected_worker_receipt": str(task_root / "attempts" / key / "worker_exit.json"),
            **resource_job(spec),
        }
        validate_stage3_job(job, policy)
        write_new(output_dir / relative, spec)
        job["prepared_spec_sha256"] = digest(output_dir / relative)
        manifest["jobs"].append(job)
    if not manifest["jobs"]:
        raise ValueError("at least one explicitly materialized case is required")
    if (
        len({read(output_dir / job["prepared_spec"])["release_freeze_sha256"] for job in manifest["jobs"]})
        != 1
    ):
        raise ValueError("mixed release freezes are forbidden")
    if len({read(output_dir / job["prepared_spec"])["catalog_sha256"] for job in manifest["jobs"]}) != 1:
        raise ValueError("mixed release catalogs are forbidden")
    # Bound the largest simultaneous reservation across admitted workers.
    per_card = []
    for gpu in gpus:
        jobs = [job for job in manifest["jobs"] if job["gpu"] == gpu]
        measured = sorted(
            (job["timeout_seconds"] + 60 for job in jobs if not job["exclusive_gpu"]),
            reverse=True,
        )[:3]
        exclusive = max(
            (job["timeout_seconds"] + 60 for job in jobs if job["exclusive_gpu"]),
            default=0,
        )
        per_card.append(max(sum(measured), exclusive))
    manifest["maximum_simultaneous_reservation_seconds"] = sum(per_card)
    if worker_wall_seconds < sum(per_card):
        raise ValueError("worker-wall budget cannot reserve one maximum simultaneous admission wave")
    write_new(output_dir / "manifest.json", manifest)
    return validate_prepared(output_dir / "manifest.json")


def validate_prepared(manifest_path):
    path = Path(manifest_path).resolve()
    manifest = read(path)
    if manifest.get("prepared_only") is not True or manifest.get("status") != "PREPARED_PENDING_APPROVAL":
        raise ValueError("expected a preparation-only manifest")
    if manifest.get("approval") is not None or manifest.get("deadline") is not None:
        raise ValueError("prepared manifest must not carry an approval or a deadline")
    policy = stage3_policy(manifest)
    if manifest.get("worker_entrypoint") != str(
        Path(manifest["worktree"]) / "studies/rasti/scripts/run_nonlinear_production.py"
    ):
        raise ValueError("unexpected worker entrypoint")
    if manifest["worker_entrypoint"] not in manifest.get("source_hashes", {}):
        raise ValueError("worker entrypoint is not source-bound")
    positive(manifest["cap_seconds"], "cap_seconds")
    positive(manifest["proposed_wallclock_seconds"], "proposed_wallclock_seconds")
    positive(manifest["admission_buffer_seconds"], "admission_buffer_seconds")
    if manifest["admission_buffer_seconds"] >= manifest["proposed_wallclock_seconds"]:
        raise ValueError("admission buffer exhausts wallclock window")
    if not manifest.get("jobs"):
        raise ValueError("prepared batch is empty")
    ids, outputs = set(), set()
    for job in manifest["jobs"]:
        validate_stage3_job(job, policy)
        spec_path = (path.parent / job["prepared_spec"]).resolve()
        if not spec_path.is_relative_to(path.parent) or digest(spec_path) != job["prepared_spec_sha256"]:
            raise ValueError("prepared spec binding differs")
        spec = read(spec_path)
        if spec.get("prepared_only") is not True or spec.get("approval_receipt") is not None:
            raise ValueError("prepared spec carries execution approval")
        if spec["case_id"] != job["case_id"] or spec["case_id"] in ids or spec["output"] in outputs:
            raise ValueError("duplicate or inconsistent prepared case")
        ids.add(spec["case_id"])
        outputs.add(spec["output"])
        positive(job["timeout_seconds"], "timeout_seconds")
        if Path(job["spec"]) != Path(manifest["task_root"]) / "activated/specs" / f"{job['key']}.json":
            raise ValueError("active spec escapes exact activation namespace")
        expected = Path(manifest["task_root"]) / "attempts" / job["key"]
        if Path(spec["output"]) != expected or Path(spec["case_output"]) != expected / "case":
            raise ValueError("prepared output escapes its exact attempt namespace")
        if job["gpu"] not in manifest["gpus"]:
            raise ValueError("job GPU outside allocation")
    return {
        "status": "PREPARED_DRY_RUN_VALID",
        "prepared_only": True,
        "jobs": len(ids),
        "manifest_sha256": digest(path),
        "gpus": manifest["gpus"],
        "worker_ceiling": manifest["max_workers"],
        "deadline_created": False,
        "approval_granted": False,
        "launch_performed": False,
    }


def check_approval(manifest_path, approval_path):
    validate_prepared(manifest_path)
    manifest_path, approval_path = (
        Path(manifest_path).resolve(),
        Path(approval_path).resolve(),
    )
    manifest, approval = read(manifest_path), read(approval_path)
    if manifest.get("synthetic_example"):
        raise ValueError("synthetic examples can never be activated")
    expected = {job["case_id"]: job["prepared_spec_sha256"] for job in manifest["jobs"]}
    if approval.get("status") != "APPROVED" or approval.get("prepared_manifest_sha256") != digest(
        manifest_path
    ):
        raise ValueError("approval must name the exact prepared manifest")
    if approval.get("prepared_spec_sha256") != expected or sorted(approval.get("case_ids", [])) != sorted(
        expected
    ):
        raise ValueError("approval must name exactly the admitted case specs")
    for key, expected_value in (
        ("authorized_gpu_limit", len(manifest["gpus"])),
        ("gpus", manifest["gpus"]),
        ("task_root", manifest["task_root"]),
        ("wallclock_seconds", manifest["proposed_wallclock_seconds"]),
        ("worker_wall_seconds", manifest["cap_seconds"]),
    ):
        if approval.get(key) != expected_value:
            raise ValueError(f"approval disagrees with prepared {key}")
    for job in manifest["jobs"]:
        spec = read(manifest_path.parent / job["prepared_spec"])
        from studies.rasti.campaign.release_catalog import validate_case_approval

        validate_case_approval(spec, approval, spec["catalog_sha256"])
        for key in ("release_freeze_sha256", "catalog_sha256"):
            if approval.get(key) != spec[key]:
                raise ValueError(f"approval differs for {key}")
    return manifest, approval


def _verify_hashes(hashes):
    for filename, expected in hashes.items():
        if not Path(filename).is_file() or digest(filename) != expected:
            raise ValueError(f"execution input/source hash mismatch: {filename}")


def activate(manifest_path, approval_path, *, supervisor=None):
    """
    Validate external approval, create a boot-bound deadline once, then
    supervise.

    This is a launch function. Call only after explicit approval. A restart
    uses the original deadline and exact active manifest; it never grants more
    time.
    """
    manifest_path, approval_path = (
        Path(manifest_path).resolve(),
        Path(approval_path).resolve(),
    )
    manifest, _ = check_approval(manifest_path, approval_path)
    _verify_hashes(manifest["source_hashes"])
    root = Path(manifest["task_root"])
    active_path = root / "activated/manifest.json"
    receipt_path = root / "activated/activation.json"
    deadline_path = root / "state/deadline.json"
    if receipt_path.exists():
        receipt = read(receipt_path)
        if receipt.get("prepared_manifest_sha256") != digest(manifest_path) or receipt.get(
            "approval_sha256"
        ) != digest(approval_path):
            raise ValueError("existing activation binds a different preparation or approval")
        if digest(active_path) != receipt["active_manifest_sha256"]:
            raise ValueError("active manifest changed after activation")
        active = read(active_path)
        for job in active["jobs"]:
            if digest(job["spec"]) != job["spec_sha256"]:
                raise ValueError("active spec changed after activation")
            _verify_hashes(read(job["spec"])["hashes"])
        validate_deadline(read(deadline_path))
    else:
        if (root / "activated").exists() or deadline_path.exists() or (root / "state/budget.json").exists():
            raise ValueError(
                "partial or preexisting execution state requires inspection; refusing a new deadline"
            )
        active = copy.deepcopy(manifest)
        active.update(
            status="APPROVED_FOR_EXECUTION",
            prepared_only=False,
            approval=str(approval_path),
            approval_sha256=digest(approval_path),
        )
        # Validation does not import scientific code or discover CUDA.
        module_spec = importlib.util.spec_from_file_location(
            "production_worker_validation", manifest["worker_entrypoint"]
        )
        module = importlib.util.module_from_spec(module_spec)
        module_spec.loader.exec_module(module)
        specs = []
        for job in active["jobs"]:
            spec = read(manifest_path.parent / job["prepared_spec"])
            spec.update(
                prepared_only=False,
                approval_receipt=str(approval_path),
                approval_receipt_sha256=digest(approval_path),
            )
            spec["hashes"][str(approval_path)] = digest(approval_path)
            _verify_hashes(spec["hashes"])
            memory = resource_job(spec)
            if any(job.get(key) != value for key, value in memory.items()):
                raise ValueError("runtime memory reservation differs from bound case metadata")
            module.validate_spec(spec, Path(job["spec"]))
            specs.append((job, spec))
        for job, spec in specs:
            write_new(job["spec"], spec)
            job["spec_sha256"] = digest(job["spec"])
        now = time.monotonic()
        deadline = {
            "clock_epoch": clock_epoch(),
            "captured_monotonic": now,
            "admission_stop_monotonic": now
            + manifest["proposed_wallclock_seconds"]
            - manifest["admission_buffer_seconds"],
            "hard_stop_monotonic": now + manifest["proposed_wallclock_seconds"],
            "approval_sha256": digest(approval_path),
            "prepared_manifest_sha256": digest(manifest_path),
        }
        validate_deadline(deadline)
        write_new(deadline_path, deadline)
        write_new(active_path, active)
        write_new(
            receipt_path,
            {
                "status": "ACTIVATED",
                "prepared_manifest_sha256": digest(manifest_path),
                "approval_sha256": digest(approval_path),
                "active_manifest_sha256": digest(active_path),
                "deadline_sha256": digest(deadline_path),
            },
        )
    if digest(deadline_path) != read(receipt_path)["deadline_sha256"]:
        raise ValueError("activation deadline changed; refusing extension")
    return (supervisor or supervise)(active_path)
