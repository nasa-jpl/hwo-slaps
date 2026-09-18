"""CPU-only launch orchestration checks; all subprocesses use tiny fixtures."""

import json
from pathlib import Path
import shutil
import sys
from types import SimpleNamespace
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import launch_nonlinear_bulk as bulk


def test_fresh_root_does_not_reuse_activation_or_any_old_state(tmp_path):
    root = tmp_path / "runtime"
    bulk.fresh_root(root)
    assert not root.exists()
    root.mkdir()
    bulk.fresh_root(root)
    (root / "old.log").write_text("old")
    with pytest.raises(ValueError, match="absent or empty"):
        bulk.fresh_root(root)


def test_fresh_start_waits_for_ready_sweeper_and_finalizes_only_after_exit(tmp_path, monkeypatch):
    worktree = tmp_path / "worktree"
    (worktree / "scripts").mkdir(parents=True)
    sweeper = worktree / "scripts/sweep_completed_attempts.py"
    shutil.copy2(Path(bulk.__file__).with_name("sweep_completed_attempts.py"), sweeper)
    controller = worktree / "scripts/run_nonlinear_execution.py"
    controller.write_text("""import fcntl, hashlib, json, pathlib, sys, time
m=json.loads(pathlib.Path(sys.argv[1]).read_text()); root=pathlib.Path(m['task_root'])
health=json.loads((root/'sweeper_health.json').read_text())
assert health['status'] in ('READY','RUNNING')
assert not (root/'state').exists() and not (root/'activated').exists()
(root/'state').mkdir()
lock=(root/'state/controller.lock').open('a+'); fcntl.flock(lock,fcntl.LOCK_EX)
attempt=root/'attempts/0000_fixture'; attempt.mkdir(parents=True)
p=attempt/'payload.bin'; p.write_bytes(b'fixture')
(attempt/'worker_exit.json').write_text(json.dumps({'status':'COMPLETE','artifacts':{str(p):hashlib.sha256(p.read_bytes()).hexdigest()}}))
(root/'state/budget.json').write_text(json.dumps({'attempts':{'0000_fixture':{'status':'COMPLETE','output':str(attempt),'ended_unix':time.time()-600}}}))
time.sleep(.2)
lock.close()
""")
    # Exercise the real sweeper loop with short CPU-fixture polling intervals.
    original = bulk.subprocess.Popen
    children = []

    def popen(command, **kwargs):
        if str(sweeper) in command:
            command = list(command)
            command[command.index("--interval") + 1] = "0.05"
        process = original(command, **kwargs)
        children.append(process)
        return process

    monkeypatch.setattr(bulk.subprocess, "Popen", popen)
    root, archive = tmp_path / "runtime", tmp_path / "archive"
    manifest = {"task_root": str(root), "worktree": str(worktree), "python": sys.executable}
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest))
    monkeypatch.setattr(bulk, "preflight", lambda *a: (manifest, {"sweeper_sha256": bulk.digest(sweeper)}))
    assert bulk.launch(path, archive, tmp_path / "fixture_approval.json") == 0
    assert all(p.poll() == 0 for p in children)
    assert (root / "attempts/0000_fixture").is_symlink()
    assert json.loads((archive / "FINALIZE_RECEIPT.json").read_text())["status"] == "FINALIZED"


@pytest.mark.parametrize("failure", ["exited", "hash", "stale", "stalled", "pid"])
def test_health_failures_block_dispatch_or_stop_monitoring(tmp_path, failure):
    import time

    health = dict(
        pid=42,
        script_sha256="expected",
        task_root=str(tmp_path),
        archive_root="/archive",
        status="RUNNING",
        heartbeat_unix=time.time(),
        progress_unix=time.time(),
    )
    if failure == "hash":
        health["script_sha256"] = "other"
    if failure == "stale":
        health["heartbeat_unix"] -= 60
    if failure == "stalled":
        health["progress_unix"] -= 4000
    if failure == "pid":
        health["pid"] = 43
    (tmp_path / "sweeper_health.json").write_text(json.dumps(health))
    process = SimpleNamespace(pid=42, poll=lambda: 1 if failure == "exited" else None)
    with pytest.raises(RuntimeError):
        bulk.healthy_sweeper(process, tmp_path, "expected", Path("/archive"))


def test_controller_shutdown_uses_cleanup_signal():
    import signal

    calls = []
    process = SimpleNamespace(
        poll=lambda: None, send_signal=lambda s: calls.append(s), wait=lambda: calls.append("wait")
    )
    bulk.stop_controller(process)
    assert calls == [signal.SIGINT, "wait"]


@pytest.mark.parametrize(
    "missing",
    [
        "issued_by",
        "authority",
        "archive_capacity_confirmation",
        "source_revision",
        "sweeper_sha256",
        "archive_root",
    ],
)
def test_preflight_requires_operator_fields_and_exact_bindings(tmp_path, monkeypatch, missing):
    worktree = tmp_path / "worktree"
    worktree.mkdir()
    root, archive = tmp_path / "runtime", tmp_path / "archive"
    sweeper = worktree / "scripts/sweep_completed_attempts.py"
    sweeper.parent.mkdir()
    sweeper.write_text("fixture")
    manifest = dict(
        worktree=str(worktree),
        task_root=str(root),
        jobs=[],
        source_hashes={str(sweeper): bulk.digest(sweeper)},
    )
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest))
    approval = dict(
        issued_by="user",
        authority="explicit user allocation",
        archive_capacity_confirmation="quota checked",
        source_revision="revision",
        sweeper_sha256=bulk.digest(sweeper),
        archive_root=str(archive),
    )
    del approval[missing]
    monkeypatch.setattr(bulk, "validate_prepared", lambda p: {})
    monkeypatch.setattr(bulk, "check_approval", lambda *a: (manifest, approval))
    monkeypatch.setattr(
        bulk.subprocess, "check_output", lambda args, **k: "" if "status" in args else "revision\n"
    )
    monkeypatch.setattr(bulk.shutil, "disk_usage", lambda p: SimpleNamespace(free=2000 * 2**30))
    with pytest.raises(ValueError, match="approval"):
        bulk.preflight(path, archive, tmp_path / "approval.json")
    assert not root.exists() and not archive.exists()


def test_sweeper_failure_before_readiness_never_starts_controller(tmp_path, monkeypatch):
    worktree = tmp_path / "worktree"
    (worktree / "scripts").mkdir(parents=True)
    sweeper = worktree / "scripts/sweep_completed_attempts.py"
    sweeper.write_text("raise SystemExit(2)\n")
    controller = worktree / "scripts/run_nonlinear_execution.py"
    marker = tmp_path / "DISPATCHED"
    controller.write_text(f"from pathlib import Path\nPath({str(marker)!r}).touch()\n")
    root, archive = tmp_path / "runtime", tmp_path / "archive"
    manifest = dict(task_root=str(root), worktree=str(worktree), python=sys.executable)
    monkeypatch.setattr(bulk, "preflight", lambda *a: (manifest, {"sweeper_sha256": bulk.digest(sweeper)}))
    with pytest.raises(RuntimeError, match="failed readiness"):
        bulk.launch(tmp_path / "manifest.json", archive, tmp_path / "approval.json")
    assert not marker.exists() and not (root / "state").exists()
