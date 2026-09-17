"""Durable bounded execution for explicitly declared nonlinear replay jobs."""

from __future__ import annotations

import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import signal
import subprocess
import time

from .profile_replay import atomic_json


def clock_epoch():
    """Identify the boot that defines the monotonic clock domain."""
    path = Path("/proc/sys/kernel/random/boot_id")
    if path.exists():
        return path.read_text().strip()
    import psutil

    return str(psutil.boot_time())


def attempt_elapsed(item):
    """Use monotonic duration and a conservative charge after a boot change."""
    if item.get("clock_epoch") == clock_epoch() and "start_monotonic" in item:
        return max(0.0, time.monotonic() - item["start_monotonic"])
    return float(item["reservation_seconds"])


def validate_deadline(deadline, epoch=None):
    """Validate a persisted monotonic deadline before it can gate dispatch."""
    if not isinstance(deadline, dict):
        raise ValueError("Deadline must be a JSON object")
    if epoch is None:
        epoch = clock_epoch()
    if deadline.get("clock_epoch") != epoch:
        raise ValueError("Deadline clock domain changed; refuse dispatch")
    values = {}
    for name in ("captured_monotonic", "admission_stop_monotonic", "hard_stop_monotonic"):
        value = deadline.get(name)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError(f"Deadline field {name} must be finite")
        if value < 0:
            raise ValueError(f"Deadline field {name} must be non-negative")
        values[name] = float(value)
    if values["captured_monotonic"] > values["hard_stop_monotonic"]:
        raise ValueError("Deadline was captured after its hard stop")
    if values["admission_stop_monotonic"] > values["hard_stop_monotonic"]:
        raise ValueError("Admission stop is later than the hard stop")
    return deadline


def deadline_reached(deadline, state):
    """Return whether no new work may be admitted or kept running."""
    return (state / "STOP").exists() or (
        deadline is not None and time.monotonic() >= deadline["hard_stop_monotonic"]
    )


class BudgetLedger:
    """Reserve complete attempt limits and retain charges across restarts."""

    def __init__(self, path, cap_seconds):
        self.path = Path(path)
        self.data = (
            json.loads(self.path.read_text())
            if self.path.exists()
            else {
                "cap_seconds": float(cap_seconds),
                "attempts": {},
                "accounting": (
                    "sum worker wall time including preparation; "
                    "active attempts reserve full timeout plus shutdown margin"
                ),
            }
        )
        if self.data["cap_seconds"] != cap_seconds:
            raise ValueError("Cannot change an existing budget cap implicitly")

    @property
    def committed(self):
        return sum(
            item["charged_seconds"] if item["status"] != "RUNNING" else item["reservation_seconds"]
            for item in self.data["attempts"].values()
        )

    def reserve(self, key, seconds, details):
        if key in self.data["attempts"]:
            raise ValueError("Attempt already exists; reconcile or use an explicit retry ID")
        if seconds <= 0 or self.committed + seconds > self.data["cap_seconds"]:
            return False
        self.data["attempts"][key] = dict(
            details,
            status="RUNNING",
            reservation_seconds=seconds,
            charged_seconds=0.0,
            start_unix=time.time(),
            start_monotonic=time.monotonic(),
            clock_epoch=clock_epoch(),
        )
        self.save()
        return True

    def finish(self, key, status, elapsed):
        item = self.data["attempts"][key]
        if item["status"] != "RUNNING":
            return
        item.update(status=status, charged_seconds=max(0.0, float(elapsed)), ended_unix=time.time())
        self.save()

    def save(self):
        self.data["committed_seconds"] = self.committed
        self.data["remaining_unreserved_seconds"] = self.data["cap_seconds"] - self.committed
        atomic_json(self.path, self.data)


def memory_admissible(used_mib, reservations, new_peak_mib, total_mib, fraction=0.8):
    return max(used_mib, sum(reservations)) + new_peak_mib <= fraction * total_mib


STAGE3_POLICY_VERSION = "stage3_v7"
STAGE3_MEMORY_PROFILE_REGISTRY = {
    "790": {
        "peak_mib": 51200,
        "registry_id": "stage3_b200_790_v1",
        "image_shape": [790, 790],
        "kernel_shape": [51, 51],
        "batch_size": 32,
        "precision": "float64",
    },
    "900": {
        "peak_mib": 70000,
        "registry_id": "stage3_b200_900_v1",
        "image_shape": [900, 900],
        "kernel_shape": [51, 51],
        "batch_size": 32,
        "precision": "float64",
    },
    "1284": {
        "peak_mib": 80000,
        "registry_id": "stage3_b200_1284_v1",
        "image_shape": [1284, 1284],
        "kernel_shape": [51, 51],
        "batch_size": 32,
        "precision": "float64",
    },
}


def stage3_policy(manifest):
    """Return opt-in Stage 3 limits while preserving legacy defaults."""
    if manifest.get("execution_policy_version") != STAGE3_POLICY_VERSION:
        return {
            "version": "legacy",
            "worker_limit": manifest.get("authorized_worker_limit", manifest.get("authorized_gpu_limit", 4)),
            "per_gpu_limit": manifest.get("max_workers_per_gpu", manifest.get("max_workers", 4)),
            "admission_fraction": 0.8,
            "runtime_fraction": 0.85,
            "rss_limit_gib": manifest.get("max_owned_rss_gib", 192),
            "task_disk_limit_gib": 15,
            "min_disk_free_gib": 70,
            "disk_check_interval_seconds": 2,
        }
    allocation_limit = manifest.get("authorized_gpu_limit", 4)
    gpus = manifest.get("gpus", [])
    worker_limit = manifest.get("authorized_worker_limit")
    per_gpu_limit = manifest.get("max_workers_per_gpu")
    admission_fraction = manifest.get("admission_memory_fraction")
    runtime_fraction = manifest.get("runtime_gpu_memory_fraction")
    rss_limit_gib = manifest.get("max_owned_rss_gib", 384)
    memory_profiles = manifest.get("memory_profile_registry", STAGE3_MEMORY_PROFILE_REGISTRY)
    task_disk_limit_gib = manifest.get("task_disk_limit_gib", 512)
    min_disk_free_gib = manifest.get("min_disk_free_gib", 100)
    disk_check_interval_seconds = manifest.get("disk_check_interval_seconds", 30)
    if (
        allocation_limit not in (4, 8)
        or not isinstance(gpus, list)
        or len(gpus) not in (4, 8)
        or any(isinstance(gpu, bool) or not isinstance(gpu, int) or not 0 <= gpu < 8 for gpu in gpus)
        or len(set(gpus)) != len(gpus)
        or allocation_limit != len(gpus)
        or isinstance(worker_limit, bool)
        or not isinstance(worker_limit, int)
        or not 1 <= worker_limit <= 3 * len(gpus)
        or isinstance(manifest.get("max_workers"), bool)
        or not isinstance(manifest.get("max_workers"), int)
        or not 1 <= manifest.get("max_workers", 0) <= worker_limit
        or isinstance(per_gpu_limit, bool)
        or not isinstance(per_gpu_limit, int)
        or not 1 <= per_gpu_limit <= 3
        or worker_limit > len(gpus) * per_gpu_limit
        or admission_fraction != 0.85
        or runtime_fraction != 0.90
        or isinstance(task_disk_limit_gib, bool)
        or not isinstance(task_disk_limit_gib, int)
        or task_disk_limit_gib <= 0
        or isinstance(min_disk_free_gib, bool)
        or not isinstance(min_disk_free_gib, int)
        or min_disk_free_gib <= 0
        or isinstance(disk_check_interval_seconds, bool)
        or not isinstance(disk_check_interval_seconds, int)
        or disk_check_interval_seconds <= 0
        or not isinstance(memory_profiles, dict)
        or any(
            not isinstance(profile, dict)
            or not isinstance(profile.get("registry_id"), str)
            or isinstance(profile.get("peak_mib"), bool)
            or not isinstance(profile.get("peak_mib"), int)
            or profile["peak_mib"] <= 0
            or not isinstance(profile.get("image_shape"), list)
            or len(profile["image_shape"]) != 2
            or any(isinstance(value, bool) or not isinstance(value, int) or value <= 0 for value in profile["image_shape"])
            or not isinstance(profile.get("kernel_shape"), list)
            or len(profile["kernel_shape"]) != 2
            or any(isinstance(value, bool) or not isinstance(value, int) or value <= 0 for value in profile["kernel_shape"])
            or isinstance(profile.get("batch_size"), bool)
            or not isinstance(profile.get("batch_size"), int)
            or profile["batch_size"] <= 0
            or not isinstance(profile.get("precision"), str)
            for profile in memory_profiles.values()
        )
        or isinstance(rss_limit_gib, bool)
        or not isinstance(rss_limit_gib, int)
        or not 1 <= rss_limit_gib <= 512
    ):
        raise ValueError("Invalid Stage 3 v7 worker/card/memory policy")
    return {
        "version": STAGE3_POLICY_VERSION,
        "worker_limit": worker_limit,
        "per_gpu_limit": per_gpu_limit,
        "admission_fraction": admission_fraction,
        "runtime_fraction": runtime_fraction,
        "rss_limit_gib": rss_limit_gib,
        "memory_profiles": memory_profiles,
        "task_disk_limit_gib": task_disk_limit_gib,
        "min_disk_free_gib": min_disk_free_gib,
        "disk_check_interval_seconds": disk_check_interval_seconds,
    }


def validate_stage3_job(job, policy):
    """Require measured or explicitly known per-worker memory reservations."""
    if policy["version"] != STAGE3_POLICY_VERSION:
        return
    peak = job.get("peak_mib")
    if isinstance(peak, bool) or not isinstance(peak, int) or peak <= 0:
        raise ValueError("Stage 3 jobs require an integer positive peak_mib")
    memory_class = job.get("memory_class")
    profile = policy["memory_profiles"].get(memory_class)
    if profile is not None:
        if (
            peak != profile["peak_mib"]
            or job.get("memory_profile_id") != profile["registry_id"]
            or job.get("image_shape") != profile["image_shape"]
            or job.get("kernel_shape") != profile["kernel_shape"]
            or job.get("batch_size") != profile["batch_size"]
            or job.get("precision") != profile["precision"]
        ):
            raise ValueError("Stage 3 job does not match its hash-bound memory profile registry")
    elif memory_class == "unmeasured_conservative":
        if peak != 140000 or job.get("exclusive_gpu") is not True:
            raise ValueError("Unmeasured Stage 3 jobs require a 140000 MiB exclusive-card reservation")
    else:
        raise ValueError("Stage 3 worker requires a registry profile or conservative exclusive reservation")


def stop_overfull_cards(active, gpus, ledger, blocked_cards, state, threshold=0.9):
    """Stop only this manifest's workers on overfull cards and persist evidence."""
    overfull = {
        item["gpu"]
        for item in active.values()
        if gpus[item["gpu"]]["used"] > threshold * gpus[item["gpu"]]["total"]
    }
    if not overfull:
        return False
    stopped = []
    for key, item in active.items():
        if item["gpu"] in overfull:
            stop_owned(item)
            ledger.finish(key, "STOPPED_CARD_MEMORY_LIMIT", attempt_elapsed(item))
            stopped.append(key)
    blocked_cards.update(overfull)
    receipt = {
        "utc_unix": time.time(),
        "blocked_cards": sorted(blocked_cards),
        "stopped_attempts": stopped,
        "gpu_snapshot": gpus,
        "threshold_fraction": threshold,
        "other_cards_preserved": True,
        "next_action": "Parent must inspect memory and declare a new bounded retry; no automatic same-card retry.",
    }
    with (Path(state) / "card_memory_events.jsonl").open("a") as stream:
        stream.write(json.dumps(receipt) + "\n")
    return True


def cached_task_bytes(root, cache, interval_seconds):
    """Return task size, rescanning only at the configured interval."""
    now = time.monotonic()
    if cache.get("sample_monotonic") is None or now - cache["sample_monotonic"] >= interval_seconds:
        cache["bytes"] = sum(
            path.stat().st_size
            for path in Path(root).rglob("*")
            if path.is_file() and not path.is_symlink()
        )
        cache["sample_monotonic"] = now
    return cache["bytes"], cache["sample_monotonic"]


def process_matches(record):
    import psutil

    try:
        proc = psutil.Process(record["pid"])
        return (
            abs(proc.create_time() - record["process_start"]) < 0.02
            and record["spec_path"] in proc.cmdline()
            and proc.status() != psutil.STATUS_ZOMBIE
        )
    except (psutil.Error, KeyError):
        return False


def worker_receipt(item):
    """Return a validated worker receipt, or ``None`` when it is unusable."""
    try:
        output = Path(item["output"]).resolve()
        receipt_path = output / "worker_exit.json"
        receipt = json.loads(receipt_path.read_text())
        status = receipt["status"]
        elapsed = receipt["elapsed_s"]
        artifacts = receipt.get("artifacts", {})
        if (
            not isinstance(status, str)
            or isinstance(elapsed, bool)
            or not isinstance(elapsed, (int, float))
            or not math.isfinite(elapsed)
            or elapsed < 0
            or not isinstance(artifacts, dict)
        ):
            if (
                isinstance(status, str)
                and not isinstance(elapsed, bool)
                and isinstance(elapsed, (int, float))
                and math.isfinite(elapsed)
                and elapsed >= 0
            ):
                receipt["status"] = "FAILED_ARTIFACT_INTEGRITY"
                return receipt
            return None
        for filename, expected in artifacts.items():
            if not isinstance(filename, str) or not isinstance(expected, str):
                receipt["status"] = "FAILED_ARTIFACT_INTEGRITY"
                return receipt
            artifact = Path(filename)
            if (
                not artifact.resolve().is_relative_to(output)
                or not artifact.is_file()
                or hashlib.sha256(artifact.read_bytes()).hexdigest() != expected
            ):
                receipt["status"] = "FAILED_ARTIFACT_INTEGRITY"
                return receipt
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
        return None
    return receipt


def classify_compute_apps(compute_apps, owned_snapshot, process_start_fn=None):
    """Classify NVML rows against an ownership snapshot taken before querying.

    A row for a process that exited after the snapshot is tolerated only
    when its UUID is the worker's recorded UUID. PID reuse or an unknown
    PID remains foreign and is returned for fail-closed handling.
    """
    import psutil

    if process_start_fn is None:
        def process_start_fn(pid):
            return psutil.Process(pid).create_time()

    by_pid = {item["pid"]: item for item in owned_snapshot.values()}
    foreign = []
    for pid, uuid in compute_apps:
        owner = by_pid.get(pid)
        if owner is None:
            foreign.append({"pid": pid, "gpu_uuid": uuid, "reason": "unrecognized_pid"})
            continue
        try:
            current_start = process_start_fn(pid)
        except (psutil.NoSuchProcess, psutil.ZombieProcess):
            # The owned process exited after the pre-query snapshot. NVML can
            # retain its row briefly; tolerate only the recorded GPU UUID.
            if uuid == owner["gpu_uuid"]:
                continue
            foreign.append({"pid": pid, "gpu_uuid": uuid, "reason": "exited_pid_uuid_mismatch"})
            continue
        except Exception:
            foreign.append({"pid": pid, "gpu_uuid": uuid, "reason": "unverifiable_pid"})
            continue
        if abs(current_start - owner["process_start"]) >= 0.02:
            foreign.append({"pid": pid, "gpu_uuid": uuid, "reason": "pid_reused"})
        elif uuid != owner["gpu_uuid"]:
            foreign.append({"pid": pid, "gpu_uuid": uuid, "reason": "owned_pid_uuid_mismatch"})
        elif not owner.get("verified", True):
            try:
                status = psutil.Process(pid).status()
            except (psutil.NoSuchProcess, psutil.ZombieProcess):
                continue
            if status != psutil.STATUS_ZOMBIE:
                foreign.append({"pid": pid, "gpu_uuid": uuid, "reason": "unverified_owner"})
    return foreign


def ownership_snapshot(active):
    """Capture PID/start/UUID ownership before querying NVML.

    Records whose process has already exited are retained as unverified so a
    stale NVML row can be tolerated, while PID reuse remains detectable.
    """
    import psutil

    snapshot = {}
    for key, item in active.items():
        try:
            pid = item["pid"]
            process_start = item["process_start"]
            gpu_uuid = item["gpu_uuid"]
        except (KeyError, TypeError):
            continue
        verified = False
        try:
            process = psutil.Process(pid)
            verified = (
                abs(process.create_time() - process_start) < 0.02
                and item["spec_path"] in process.cmdline()
                and process.status() != psutil.STATUS_ZOMBIE
            )
        except psutil.Error:
            pass
        record = {
            "pid": pid,
            "process_start": process_start,
            "gpu": item["gpu"],
            "gpu_uuid": gpu_uuid,
            "verified": verified,
        }
        snapshot[key] = record
        if verified:
            try:
                children = process.children(recursive=True)
            except psutil.Error:
                children = []
            for child in children:
                try:
                    child_start = child.create_time()
                except psutil.Error:
                    continue
                snapshot[f"{key}:child:{child.pid}"] = {
                    "pid": child.pid,
                    "process_start": child_start,
                    "gpu": item["gpu"],
                    "gpu_uuid": gpu_uuid,
                    "verified": True,
                }
    return snapshot


def stop_owned(record):
    """Stop only a group whose process start and spec match the ledger."""
    if not process_matches(record):
        return
    try:
        os.killpg(record["pid"], signal.SIGTERM)
    except ProcessLookupError:
        return
    deadline = time.monotonic() + 10
    while process_matches(record) and time.monotonic() < deadline:
        time.sleep(0.2)
    if process_matches(record):
        try:
            os.killpg(record["pid"], signal.SIGKILL)
        except ProcessLookupError:
            pass


def concurrency_limits(manifest, state):
    """Read bounded admission limits without changing the physical GPU cap."""
    maximum = manifest["max_workers"]
    per_gpu = manifest.get("max_workers_per_gpu", maximum)
    control = Path(state) / "concurrency.json"
    if control.exists():
        value = json.loads(control.read_text())
        current = value.get("max_workers", maximum)
        current_per_gpu = value.get("max_workers_per_gpu", per_gpu)
        if (
            isinstance(current, bool) or not isinstance(current, int)
            or isinstance(current_per_gpu, bool) or not isinstance(current_per_gpu, int)
            or not 1 <= current <= maximum or not 1 <= current_per_gpu <= per_gpu
        ):
            raise ValueError("Admission override exceeds the frozen worker limits")
        return current, current_per_gpu
    return maximum, per_gpu


def supervise(manifest_path):
    """Run a fixed manifest with an exclusive lock and durable budget."""
    import psutil

    manifest_path = Path(manifest_path).resolve()
    manifest = json.loads(manifest_path.read_text())
    root = Path(manifest["task_root"])
    state = root / "state"
    state.mkdir(parents=True, exist_ok=True)
    lock = (state / "controller.lock").open("a+")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    policy = stage3_policy(manifest)
    allocation_limit = manifest.get("authorized_gpu_limit", 4)
    worker_limit = policy["worker_limit"]
    per_gpu_limit = policy["per_gpu_limit"]
    if policy["version"] == STAGE3_POLICY_VERSION:
        for job in manifest.get("jobs", []):
            validate_stage3_job(job, policy)
    if worker_limit > allocation_limit and (
        isinstance(per_gpu_limit, bool) or not isinstance(per_gpu_limit, int)
        or not 1 <= per_gpu_limit <= 8
        or worker_limit > len(manifest["gpus"]) * per_gpu_limit
    ):
        raise ValueError("Packed allocation requires an explicit limit of at most eight workers per GPU")
    if (
        allocation_limit not in (4, 8)
        or isinstance(worker_limit, bool) or not isinstance(worker_limit, int)
        or not 1 <= worker_limit <= 8 * allocation_limit
        or not 1 <= manifest["max_workers"] <= worker_limit
        or not 1 <= len(manifest["gpus"]) <= allocation_limit
        or len(set(manifest["gpus"])) != len(manifest["gpus"])
    ):
        raise ValueError("Invalid GPU allocation or worker limit; default authorization is four")
    rss_limit_gib = policy["rss_limit_gib"]
    if isinstance(rss_limit_gib, bool) or not isinstance(rss_limit_gib, int) or not 1 <= rss_limit_gib <= 512:
        raise ValueError("Invalid declared host-RAM cap")
    ledger = BudgetLedger(state / "budget.json", manifest["cap_seconds"])
    deadline_path = root / "state" / "deadline.json"
    handles = {}
    block_state_path = state / ("blocked_cards_" + manifest_path.stem + ".json")
    blocked_cards = set(json.loads(block_state_path.read_text())) if block_state_path.exists() else set()
    if policy["version"] == STAGE3_POLICY_VERSION and not blocked_cards.issubset(set(manifest["gpus"])):
        raise ValueError("Persisted blocked card is outside this manifest allocation")
    disk_cache = {"sample_monotonic": None, "bytes": 0}
    deadline = None
    try:
        # Adopt recoverable starts, including the crash window after launch.
        for key, item in ledger.data["attempts"].items():
            if item["status"] != "RUNNING":
                continue
            started = Path(item["output"]) / "worker_started.json"
            if started.exists():
                item.update(json.loads(started.read_text()))
            elif "pid" not in item:
                for proc in psutil.process_iter(["pid", "cmdline", "create_time"]):
                    if item["spec_path"] in (proc.info["cmdline"] or []):
                        item.update(pid=proc.pid, process_start=proc.info["create_time"])
                        break
        ledger.save()
        if policy["version"] == STAGE3_POLICY_VERSION and not deadline_path.exists():
            raise ValueError("Stage 3 v7 requires state/deadline.json before dispatch")
        if deadline_path.exists():
            deadline = validate_deadline(json.loads(deadline_path.read_text()))
        while True:
            stopping = deadline_reached(deadline, state)
            active = {k: v for k, v in ledger.data["attempts"].items() if v["status"] == "RUNNING"}
            for key, item in list(active.items()):
                if key in handles:
                    handles[key].poll()
                if not process_matches(item):
                    receipt = worker_receipt(item)
                    if receipt is not None:
                        status = receipt["status"]
                        ledger.finish(key, status, max(receipt["elapsed_s"], attempt_elapsed(item)))
                    else:
                        ledger.finish(key, "INTERRUPTED_UNCERTAIN", item["reservation_seconds"])
                    continue
                elapsed = attempt_elapsed(item)
                if not stopping and elapsed >= item["timeout_seconds"]:
                    stop_owned(item)
                    ledger.finish(key, "TIMED_OUT", attempt_elapsed(item))
            active = {k: v for k, v in ledger.data["attempts"].items() if v["status"] == "RUNNING"}
            pending = [j for j in manifest["jobs"] if j["key"] not in ledger.data["attempts"]]
            failed = [
                k
                for k, v in ledger.data["attempts"].items()
                if k in {j["key"] for j in manifest["jobs"]} and v["status"] not in ("RUNNING", "COMPLETE")
            ]
            if failed and policy["version"] != STAGE3_POLICY_VERSION:
                raise RuntimeError("Manifest failure; dependent admissions stopped: " + ",".join(failed))
            if not active and not pending:
                # Reconcile completed workers before a deadline that may
                # have elapsed during the receipt/ledger update.
                break
            if deadline_reached(deadline, state):
                raise RuntimeError("Investigation deadline or explicit stop reached")
            owned_snapshot = ownership_snapshot(active)
            raw = subprocess.check_output(
                [
                    "nvidia-smi",
                    "--query-gpu=index,uuid,memory.total,memory.used",
                    "--format=csv,noheader,nounits",
                ],
                text=True,
            )
            gpus = {}
            for row in raw.splitlines():
                index, uuid, total, used = [v.strip() for v in row.split(",")]
                gpus[int(index)] = {"uuid": uuid, "total": int(total), "used": int(used)}
            app_rows = subprocess.check_output(
                ["nvidia-smi", "--query-compute-apps=pid,gpu_uuid", "--format=csv,noheader,nounits"],
                text=True,
            )
            compute_apps = [
                (int(parts[0].strip()), parts[1].strip())
                for row in app_rows.splitlines()
                if len(parts := row.split(",")) == 2
            ]
            foreign_rows = classify_compute_apps(compute_apps, owned_snapshot)
            foreign_uuids = {row["gpu_uuid"] for row in foreign_rows}
            assigned_uuids = {
                gpus[item["gpu"]]["uuid"] for item in active.values() if item["gpu"] in gpus
            }
            pid_reused = any(row["reason"] == "pid_reused" for row in foreign_rows)
            if pid_reused or foreign_uuids.intersection(assigned_uuids):
                failure = {
                    "active": {
                        key: {
                            "pid": item.get("pid"),
                            "process_start": item.get("process_start"),
                            "gpu": item.get("gpu"),
                            "gpu_uuid": item.get("gpu_uuid"),
                        }
                        for key, item in active.items()
                    },
                    "owned_snapshot": owned_snapshot,
                    "gpu_rows": gpus,
                    "compute_apps": [
                        {"pid": pid, "gpu_uuid": uuid} for pid, uuid in compute_apps
                    ],
                    "foreign_rows": foreign_rows,
                }
                atomic_json(state / "foreign_gpu_failure.json", failure)
                raise RuntimeError(
                    "Foreign GPU process entered an owned assignment: "
                    + json.dumps(foreign_rows, sort_keys=True)
                )
            rss = 0
            for item in active.values():
                if process_matches(item):
                    proc = psutil.Process(item["pid"])
                    for child in [proc] + proc.children(recursive=True):
                        try:
                            rss += child.memory_info().rss
                        except psutil.Error:
                            pass
            disk, disk_sample_monotonic = cached_task_bytes(
                root,
                disk_cache,
                policy["disk_check_interval_seconds"],
            )
            current_limit, per_gpu_limit = concurrency_limits(manifest, state)
            telemetry = {
                "utc_unix": time.time(),
                "gpus": gpus,
                "owned_rss": rss,
                "host_available": psutil.virtual_memory().available,
                "disk_free": shutil.disk_usage(root).free,
                "task_bytes": disk,
                "active": list(active),
                "pending": [j["key"] for j in pending],
                "admission_max_workers": current_limit,
                "admission_max_workers_per_gpu": per_gpu_limit,
                "physical_gpu_indices": manifest["gpus"],
                "max_owned_rss_gib": rss_limit_gib,
                "admission_memory_fraction": policy["admission_fraction"],
                "runtime_gpu_memory_fraction": policy["runtime_fraction"],
                "failed_attempts": failed,
                "blocked_cards": sorted(blocked_cards),
                "task_bytes_sample_monotonic": disk_sample_monotonic,
            }
            with (state / "resources.jsonl").open("a") as stream:
                stream.write(json.dumps(telemetry) + "\n")
            if (
                rss > rss_limit_gib * 2**30
                or telemetry["host_available"] < 512 * 2**30
                or telemetry["disk_free"] < policy["min_disk_free_gib"] * 2**30
                or disk > policy["task_disk_limit_gib"] * 2**30
            ):
                raise RuntimeError("RAM/disk guard failed")
            if policy["version"] == STAGE3_POLICY_VERSION:
                if stop_overfull_cards(
                    active,
                    gpus,
                    ledger,
                    blocked_cards,
                    state,
                    policy["runtime_fraction"],
                ):
                    atomic_json(block_state_path, sorted(blocked_cards))
                    continue
            else:
                for item in active.values():
                    if gpus[item["gpu"]]["used"] > policy["runtime_fraction"] * gpus[item["gpu"]]["total"]:
                        raise RuntimeError("GPU memory ceiling exceeded")
            admitted = False
            for job in pending:
                if deadline is not None and (
                    time.monotonic() >= deadline["admission_stop_monotonic"]
                    or time.monotonic() + job["timeout_seconds"] + 60 > deadline["hard_stop_monotonic"]
                ):
                    continue
                if len(active) >= current_limit:
                    break
                gpu = job["gpu"]
                if policy["version"] == STAGE3_POLICY_VERSION and gpu in blocked_cards:
                    continue
                if gpu not in manifest["gpus"]:
                    raise ValueError("GPU outside manifest allocation")
                same = [v for v in active.values() if v["gpu"] == gpu]
                if len(same) >= per_gpu_limit:
                    continue
                if job.get("exclusive_gpu") and same:
                    continue
                if any(item.get("exclusive_gpu") for item in same):
                    continue
                if gpus[gpu]["uuid"] in foreign_uuids:
                    continue
                if not same and gpus[gpu]["used"] > 1024:
                    continue
                if not memory_admissible(
                    gpus[gpu]["used"],
                    [v["peak_mib"] for v in same],
                    job["peak_mib"],
                    gpus[gpu]["total"],
                    policy["admission_fraction"],
                ):
                    continue
                # Resource inspection can cross the hard stop. Recheck
                # immediately before creating a durable reservation.
                if deadline_reached(deadline, state) or (
                    deadline is not None and time.monotonic() >= deadline["admission_stop_monotonic"]
                ):
                    continue
                spec = Path(job["spec"]).resolve()
                payload = json.loads(spec.read_text())
                output = Path(payload["output"])
                if output.exists():
                    raise ValueError("Unaccounted output directory exists: " + str(output))
                output.mkdir(parents=True)
                item = dict(
                    output=str(output),
                    spec_path=str(spec),
                    gpu=gpu,
                    gpu_uuid=gpus[gpu]["uuid"],
                    peak_mib=job["peak_mib"],
                    timeout_seconds=job["timeout_seconds"],
                    exclusive_gpu=bool(job.get("exclusive_gpu", False)),
                    memory_class=job.get("memory_class"),
                    memory_profile_id=job.get("memory_profile_id"),
                )
                if not ledger.reserve(job["key"], job["timeout_seconds"] + 60, item):
                    output.rmdir()
                    continue
                # Close the reservation if persistence crossed the
                # deadline. No worker has been launched.
                if deadline_reached(deadline, state) or (
                    deadline is not None and time.monotonic() >= deadline["admission_stop_monotonic"]
                ):
                    ledger.finish(job["key"], "NOT_LAUNCHED_AFTER_DEADLINE", 0.0)
                    output.rmdir()
                    continue
                env = dict(
                    os.environ,
                    CUDA_VISIBLE_DEVICES=str(gpu),
                    JAX_PLATFORMS="cuda",
                    JAX_ENABLE_X64="True",
                    XLA_PYTHON_CLIENT_PREALLOCATE="false",
                    OMP_NUM_THREADS="1",
                    OPENBLAS_NUM_THREADS="1",
                    MKL_NUM_THREADS="1",
                    NUMEXPR_NUM_THREADS="1",
                    HWOSLAPS_NAUTILUS_TRAINING_WORKERS="4",
                    PYAUTO_SKIP_WORKSPACE_VERSION_CHECK="1",
                    PYTHONPATH=manifest["worktree"] + "/src",
                    HWOSLAPS_CAMPAIGN_UUID=manifest["campaign_uuid"],
                )
                env.pop("JAX_COMPILATION_CACHE_DIR", None)
                cmd = [
                    manifest["python"],
                    manifest.get(
                        "worker_entrypoint", manifest["worktree"] + "/scripts/run_nonlinear_profile.py"
                    ),
                    str(spec),
                ]
                if job.get("cores"):
                    cmd = ["taskset", "-c", job["cores"]] + cmd
                with (output / "worker.log").open("w") as stream:
                    proc = subprocess.Popen(
                        cmd,
                        cwd=manifest["worktree"],
                        env=env,
                        stdout=stream,
                        stderr=subprocess.STDOUT,
                        start_new_session=True,
                    )
                record = ledger.data["attempts"][job["key"]]
                record.update(pid=proc.pid, process_start=psutil.Process(proc.pid).create_time(), command=cmd)
                ledger.save()
                handles[job["key"]] = proc
                active[job["key"]] = record
                admitted = True
                if deadline_reached(deadline, state):
                    # Popen is not atomic with the clock. If launch crosses
                    # the cutoff, stop this owned process at once.
                    stop_owned(record)
                    ledger.finish(job["key"], "STOPPED_AFTER_DEADLINE", attempt_elapsed(record))
                    raise RuntimeError("Investigation deadline or explicit stop reached")
            if not active and pending and not admitted:
                atomic_json(
                    state / "dispatch_blocked.json",
                    {
                        "pending": pending,
                        "budget": ledger.data,
                        "reason": "No admission within current resources/reserved budget",
                    },
                )
                break
            ledger.save()
            time.sleep(2)
    except BaseException:
        for key, item in ledger.data["attempts"].items():
            if item["status"] == "RUNNING":
                stop_owned(item)
                receipt = worker_receipt(item)
                if receipt is not None:
                    ledger.finish(
                        key,
                        receipt["status"],
                        max(receipt["elapsed_s"], attempt_elapsed(item)),
                    )
                else:
                    ledger.finish(key, "STOPPED_AFTER_FAILURE", attempt_elapsed(item))
        raise
    finally:
        ledger.save()
        lock.close()
