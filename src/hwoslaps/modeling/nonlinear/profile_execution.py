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


def memory_admissible(used_mib, reservations, new_peak_mib, total_mib):
    return max(used_mib, sum(reservations)) + new_peak_mib <= 0.8 * total_mib


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
    allocation_limit = manifest.get("authorized_gpu_limit", 4)
    if (
        allocation_limit not in (4, 8)
        or not 1 <= manifest["max_workers"] <= allocation_limit
        or not 1 <= len(manifest["gpus"]) <= allocation_limit
        or len(set(manifest["gpus"])) != len(manifest["gpus"])
    ):
        raise ValueError("Invalid GPU allocation or worker limit; default authorization is four")
    ledger = BudgetLedger(state / "budget.json", manifest["cap_seconds"])
    deadline_path = root / "state" / "deadline.json"
    handles = {}
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
            if failed:
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
            disk = sum(p.stat().st_size for p in root.rglob("*") if p.is_file() and not p.is_symlink())
            telemetry = {
                "utc_unix": time.time(),
                "gpus": gpus,
                "owned_rss": rss,
                "host_available": psutil.virtual_memory().available,
                "disk_free": shutil.disk_usage(root).free,
                "task_bytes": disk,
                "active": list(active),
                "pending": [j["key"] for j in pending],
            }
            with (state / "resources.jsonl").open("a") as stream:
                stream.write(json.dumps(telemetry) + "\n")
            if (
                rss > 192 * 2**30
                or telemetry["host_available"] < 512 * 2**30
                or telemetry["disk_free"] < 70 * 2**30
                or disk > 15 * 2**30
            ):
                raise RuntimeError("RAM/disk guard failed")
            for item in active.values():
                if gpus[item["gpu"]]["used"] > 0.85 * gpus[item["gpu"]]["total"]:
                    raise RuntimeError("GPU memory ceiling exceeded")
            admitted = False
            for job in pending:
                if deadline is not None and (
                    time.monotonic() >= deadline["admission_stop_monotonic"]
                    or time.monotonic() + job["timeout_seconds"] + 60 > deadline["hard_stop_monotonic"]
                ):
                    continue
                if len(active) >= manifest["max_workers"]:
                    break
                gpu = job["gpu"]
                if gpu not in manifest["gpus"]:
                    raise ValueError("GPU outside manifest allocation")
                same = [v for v in active.values() if v["gpu"] == gpu]
                if gpus[gpu]["uuid"] in foreign_uuids:
                    continue
                if not same and gpus[gpu]["used"] > 1024:
                    continue
                if not memory_admissible(
                    gpus[gpu]["used"], [v["peak_mib"] for v in same], job["peak_mib"], gpus[gpu]["total"]
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
