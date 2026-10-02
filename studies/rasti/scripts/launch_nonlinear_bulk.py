#!/usr/bin/env python3
"""Preflight and supervise one approved bulk controller and archive sweeper.

Without --launch-approved this performs read-only validation. No approval is
created here. The operator supplies the exact allocation-bound approval after
review; a fresh runtime directory is created only by the explicit launch path.
"""

import argparse
import fcntl
import json
import os
from pathlib import Path

import sys
_STUDY_REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(_STUDY_REPO_ROOT), str(_STUDY_REPO_ROOT / "src")]
import shutil
import signal
import subprocess
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))
from studies.rasti.campaign.execution_prepare import (  # noqa: E402
    _verify_hashes,
    check_approval,
    digest,
    read,
    validate_prepared,
)


def fresh_root(root):
    if root.is_symlink() or (root.exists() and (not root.is_dir() or any(root.iterdir()))):
        raise ValueError("runtime root must be absent or empty; reconcile stale state separately")


def existing_parent(path):
    while not path.exists():
        path = path.parent
    return path


def preflight(manifest_path, archive, approval_path=None):
    result = validate_prepared(manifest_path)
    manifest = read(manifest_path)
    worktree = Path(manifest["worktree"])
    if subprocess.check_output(["git", "-C", str(worktree), "status", "--porcelain"], text=True).strip():
        raise ValueError("source worktree is dirty")
    revision = subprocess.check_output(["git", "-C", str(worktree), "rev-parse", "HEAD"], text=True).strip()
    _verify_hashes(manifest["source_hashes"])
    inputs = {}
    for job in manifest["jobs"]:
        for path, expected in read(manifest_path.parent / job["prepared_spec"])["hashes"].items():
            if path in inputs and inputs[path] != expected:
                raise ValueError(f"conflicting input binding: {path}")
            inputs[path] = expected
    _verify_hashes(inputs)
    sweeper = worktree / "studies/rasti/scripts/sweep_completed_attempts.py"
    if manifest["source_hashes"].get(str(sweeper)) != digest(sweeper):
        raise ValueError("running sweeper is not source-bound")
    root = Path(manifest["task_root"])
    fresh_root(root)
    if archive == root or archive.is_relative_to(root) or root.is_relative_to(archive):
        raise ValueError("task and archive roots must be disjoint")
    if archive.exists() and any(archive.iterdir()):
        raise ValueError("archive root must be absent or empty for a fresh launch")
    if shutil.disk_usage(existing_parent(root)).free < 130 * 2**30:
        raise ValueError("less than 130 GiB free on runtime volume")
    # The review estimates 1.25 TB retained output. Reserve 1.5 TiB of visible
    # filesystem capacity; the operator must separately confirm user quota.
    if shutil.disk_usage(existing_parent(archive)).free < 1536 * 2**30:
        raise ValueError("less than 1.5 TiB free on archive volume")
    if not os.access(existing_parent(archive), os.W_OK):
        raise ValueError("archive parent is not writable")
    if approval_path is not None:
        _, approval = check_approval(manifest_path, approval_path)
        for key in ("issued_by", "authority", "archive_capacity_confirmation"):
            if not isinstance(approval.get(key), str) or not approval[key].strip():
                raise ValueError(f"approval needs nonempty {key}")
        if approval.get("source_revision") != revision:
            raise ValueError("approval source revision differs")
        if approval.get("sweeper_sha256") != digest(sweeper):
            raise ValueError("approval sweeper hash differs")
        if approval.get("archive_root") != str(archive):
            raise ValueError("approval archive root differs")
    return manifest, dict(
        result,
        source_revision=revision,
        sweeper_sha256=digest(sweeper),
        allocation_approved=approval_path is not None,
    )


def healthy_sweeper(process, root, expected_hash, archive):
    if process.poll() is not None:
        raise RuntimeError("sweeper exited before controller completion")
    health = read(root / "sweeper_health.json")
    if (
        health.get("pid") != process.pid
        or health.get("script_sha256") != expected_hash
        or health.get("task_root") != str(root)
        or health.get("archive_root") != str(archive)
        or health.get("status") not in {"READY", "RUNNING"}
        or not 0 <= time.time() - health["heartbeat_unix"] < 30
        or not 0 <= time.time() - health["progress_unix"] < 3600
    ):
        raise RuntimeError("sweeper heartbeat/progress identity is stale or invalid")


def stop_controller(process):
    if process is not None and process.poll() is None:
        # SIGINT enters the controller's BaseException handler, which stops and
        # accounts for its own workers (each worker has its own process group).
        process.send_signal(signal.SIGINT)
        process.wait()


def launch(manifest_path, archive, approval_path):
    manifest, result = preflight(manifest_path, archive, approval_path)
    root = Path(manifest["task_root"])
    root.parent.mkdir(parents=True, exist_ok=True)
    with root.with_name(root.name + ".launch.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        fresh_root(root)
        root.mkdir(exist_ok=True)  # No activation, deadline or budget state.
        archive.mkdir(parents=True, exist_ok=True)
        probe = archive / ".write_probe"
        with probe.open("xb") as stream:
            stream.write(b"archive write preflight\n")
            stream.flush()
            os.fsync(stream.fileno())
        probe.unlink()
        worktree = Path(manifest["worktree"])
        sweeper_command = [
            manifest["python"],
            str(worktree / "studies/rasti/scripts/sweep_completed_attempts.py"),
            str(root),
            str(archive),
            "--interval",
            "120",
            "--settle",
            "120",
        ]
        controller = None
        with (
            (root / "sweeper.log").open("x") as sweep_log,
            (root / "controller.log").open("x") as controller_log,
        ):
            sweeper = subprocess.Popen(
                sweeper_command,
                stdout=sweep_log,
                stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL,
                start_new_session=True,
            )
            try:
                deadline = time.monotonic() + 30
                while not (root / "sweeper_health.json").is_file():
                    if sweeper.poll() is not None or time.monotonic() >= deadline:
                        raise RuntimeError("sweeper failed readiness before dispatch")
                    time.sleep(0.1)
                healthy_sweeper(sweeper, root, result["sweeper_sha256"], archive)
                controller = subprocess.Popen(
                    [
                        manifest["python"],
                        str(worktree / "studies/rasti/scripts/run_nonlinear_execution.py"),
                        str(manifest_path),
                        "--approval",
                        str(approval_path),
                        "--launch-approved",
                    ],
                    cwd=worktree,
                    stdout=controller_log,
                    stderr=subprocess.STDOUT,
                    stdin=subprocess.DEVNULL,
                    start_new_session=True,
                )
                while controller.poll() is None:
                    try:
                        healthy_sweeper(sweeper, root, result["sweeper_sha256"], archive)
                    except Exception:
                        # The sweeper may finish normally just after the
                        # controller releases its lock and before poll reaps it.
                        try:
                            controller.wait(timeout=2)
                        except subprocess.TimeoutExpired:
                            raise RuntimeError("sweeper unhealthy while controller is running")
                    time.sleep(2)
                controller_code = controller.wait()
                if controller_code:
                    raise RuntimeError(f"controller failed with exit {controller_code}")
                # The sweeper drains COMPLETE attempts before exiting. Never
                # finalize concurrently with this last copy/deletion pass.
                sweeper_code = sweeper.wait(timeout=3720)
                if sweeper_code:
                    raise RuntimeError(f"sweeper failed with exit {sweeper_code}")
            except BaseException:
                stop_controller(controller)
                if sweeper.poll() is None:
                    sweeper.terminate()
                    sweeper.wait()
                raise
        subprocess.run(sweeper_command + ["--finalize"], check=True)
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("archive", type=Path)
    parser.add_argument("--approval", type=Path)
    parser.add_argument("--launch-approved", action="store_true")
    args = parser.parse_args(argv)

    def interrupted(signum, frame):
        raise KeyboardInterrupt(f"launch wrapper received signal {signum}")

    signal.signal(signal.SIGTERM, interrupted)
    manifest, archive = args.manifest.resolve(), args.archive.resolve()
    if args.launch_approved:
        if args.approval is None:
            parser.error("--launch-approved requires an external allocation approval")
        return launch(manifest, archive, args.approval.resolve())
    _, result = preflight(manifest, archive, args.approval)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
