"""Durable bounded execution for explicitly declared nonlinear replay jobs."""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import time

from .profile_replay import atomic_json


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


def stop_owned(record):
    """Stop only a group whose process start and spec match the ledger."""
    if not process_matches(record):
        return
    os.killpg(record["pid"], signal.SIGTERM)
    deadline = time.monotonic() + 10
    while process_matches(record) and time.monotonic() < deadline:
        time.sleep(0.2)
    if process_matches(record):
        os.killpg(record["pid"], signal.SIGKILL)


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
    if manifest["max_workers"] > 4 or len(manifest["gpus"]) > 4:
        raise ValueError("A0002 permits at most four GPU workers/cards")
    ledger = BudgetLedger(state / "budget.json", manifest["cap_seconds"])
    handles = {}
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
        while True:
            active = {k: v for k, v in ledger.data["attempts"].items() if v["status"] == "RUNNING"}
            for key, item in list(active.items()):
                ended = Path(item["output"]) / "worker_exit.json"
                if key in handles:
                    handles[key].poll()
                if not process_matches(item):
                    if ended.exists():
                        receipt = json.loads(ended.read_text())
                        status = receipt["status"]
                        for filename, expected in receipt.get("artifacts", {}).items():
                            artifact = Path(filename)
                            if (
                                not artifact.resolve().is_relative_to(Path(item["output"]).resolve())
                                or not artifact.is_file()
                                or hashlib.sha256(artifact.read_bytes()).hexdigest() != expected
                            ):
                                status = "FAILED_ARTIFACT_INTEGRITY"
                                break
                        ledger.finish(
                            key, status, max(receipt["elapsed_s"], time.time() - item["start_unix"])
                        )
                    else:
                        ledger.finish(key, "INTERRUPTED_UNCERTAIN", item["reservation_seconds"])
                    continue
                elapsed = time.time() - item["start_unix"]
                if elapsed >= item["timeout_seconds"]:
                    stop_owned(item)
                    ledger.finish(key, "TIMED_OUT", time.time() - item["start_unix"])
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
                break
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
            owned_pids = set()
            for item in active.values():
                if process_matches(item):
                    owner = psutil.Process(item["pid"])
                    owned_pids.update(p.pid for p in [owner] + owner.children(recursive=True))
            foreign_uuids = {uuid for pid, uuid in compute_apps if pid not in owned_pids}
            if any(gpus[item["gpu"]]["uuid"] in foreign_uuids for item in active.values()):
                raise RuntimeError("Foreign GPU process entered an owned assignment")
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
                if not ledger.reserve(job["key"], job["timeout_seconds"] + 30, item):
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
                    manifest["worktree"] + "/scripts/run_nonlinear_profile.py",
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
                ledger.finish(key, "STOPPED_AFTER_FAILURE", time.time() - item["start_unix"])
        raise
    finally:
        ledger.save()
        lock.close()
