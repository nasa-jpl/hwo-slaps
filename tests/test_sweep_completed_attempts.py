"""Tests for the completed-attempt sweeper against a fake Stage 3 task root."""

import fcntl
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "studies/rasti/scripts"))
import sweep_completed_attempts as sweep  # noqa: E402

WORKTREE_SRC = Path(os.environ.get("HWO_WORKTREE", Path(__file__).resolve().parents[1])) / "src"
sys.path.insert(0, str(WORKTREE_SRC))
from studies.rasti.campaign.profile_execution import cached_task_bytes  # noqa: E402


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def make_attempt(task_root: Path, key: str, status: str, payload_bytes: int, old: bool = True) -> Path:
    attempt = task_root / "attempts" / key
    case = attempt / "case" / "ladder" / "search_abc"
    case.mkdir(parents=True)
    (case / "search_internal.dill").write_bytes(os.urandom(payload_bytes))
    (case / "result.zip").write_bytes(os.urandom(2048))
    (attempt / "case" / "payload.json").write_text(json.dumps({"case": key}))
    (attempt / "production_run.json").write_text(json.dumps({"status": status}))
    (attempt / "worker.log").write_text("log\n")
    (attempt / "worker_started.json").write_text("{}")
    artifacts = {
        str(p): sha(p)
        for p in attempt.rglob("*")
        if p.is_file() and p.name not in {"worker_exit.json", "worker.log"}
    }
    receipt = attempt / "worker_exit.json"
    receipt.write_text(
        json.dumps({"status": status, "case_id": f"case:{key}", "artifacts": artifacts, "elapsed_s": 5.0})
    )
    if old:
        stale = time.time() - 600
        os.utime(receipt, (stale, stale))
    return attempt


def make_root(tmp_path: Path):
    task_root = tmp_path / "task_root"
    (task_root / "state").mkdir(parents=True)
    (task_root / "activated" / "specs").mkdir(parents=True)
    (task_root / "activated" / "specs" / "0000.json").write_text("{}")
    done = make_attempt(task_root, "0000_done", "COMPLETE", 200_000)
    running = make_attempt(task_root, "0001_running", "RUNNING", 100_000)
    ledger = {
        "cap_seconds": 100.0,
        "attempts": {
            "0000_done": {"output": str(done), "status": "COMPLETE", "ended_unix": time.time() - 600},
            "0001_running": {"output": str(running), "status": "RUNNING"},
        },
    }
    (task_root / "state" / "budget.json").write_text(json.dumps(ledger))
    archive = tmp_path / "archive"
    return task_root, archive, done, running


def test_sweep_moves_only_complete_attempts_and_keeps_paths_resolving(tmp_path):
    task_root, archive, done, running = make_root(tmp_path)
    before_hashes = {p.relative_to(done): sha(p) for p in done.rglob("*") if p.is_file()}
    before_bytes, _ = cached_task_bytes(task_root, {}, 0)
    log = archive / "SWEEP_LOG.jsonl"
    archive.mkdir()
    swept = sweep.sweep_once(task_root, archive, sweep.PathMap({}), 0.0, log)
    assert [r["key"] for r in swept] == ["0000_done"]
    assert done.is_symlink() and done.resolve() == (archive / "attempts" / "0000_done").resolve()
    assert not running.is_symlink() and running.is_dir()
    after_hashes = {p.relative_to(done): sha(p) for p in done.rglob("*") if p.is_file()}
    assert after_hashes == before_hashes
    receipt = json.loads((done / "worker_exit.json").read_text())
    for recorded, digest in receipt["artifacts"].items():
        path = Path(recorded)
        assert path.is_file() and sha(path) == digest
        assert path.resolve() == (archive / "attempts" / "0000_done" / path.relative_to(done)).resolve()
    after_bytes, _ = cached_task_bytes(task_root, {}, 0)
    assert after_bytes < before_bytes - 200_000
    assert not (task_root / "attempts" / "0000_done.swept").exists()
    assert not (archive / "attempts" / "0000_done.partial").exists()
    assert len(log.read_text().splitlines()) == 1
    assert sweep.sweep_once(task_root, archive, sweep.PathMap({}), 0.0, log) == []


def test_sweep_respects_settle_time(tmp_path):
    task_root, archive, done, _ = make_root(tmp_path)
    ledger_path = task_root / "state" / "budget.json"
    ledger = json.loads(ledger_path.read_text())
    ledger["attempts"]["0000_done"]["ended_unix"] = time.time()
    ledger_path.write_text(json.dumps(ledger))
    archive.mkdir()
    assert sweep.sweep_once(task_root, archive, sweep.PathMap({}), 120.0, archive / "log") == []
    assert done.is_dir() and not done.is_symlink()


def test_sweep_refuses_receipt_mismatch_and_leaves_original(tmp_path):
    task_root, archive, done, _ = make_root(tmp_path)
    (done / "case" / "ladder" / "search_abc" / "search_internal.dill").write_bytes(b"tampered")
    archive.mkdir()
    with pytest.raises(sweep.SweepError, match="hash mismatch"):
        sweep.sweep_once(task_root, archive, sweep.PathMap({}), 0.0, archive / "log")
    assert done.is_dir() and not done.is_symlink()
    assert not (archive / "attempts").exists()


def test_sweep_refuses_ledger_receipt_disagreement(tmp_path):
    task_root, archive, done, _ = make_root(tmp_path)
    receipt = json.loads((done / "worker_exit.json").read_text())
    receipt["status"] = "FAILED"
    (done / "worker_exit.json").write_text(json.dumps(receipt))
    stale = time.time() - 600
    os.utime(done / "worker_exit.json", (stale, stale))
    archive.mkdir()
    with pytest.raises(sweep.SweepError, match="ledger says COMPLETE"):
        sweep.sweep_once(task_root, archive, sweep.PathMap({}), 0.0, archive / "log")


def test_sweep_requires_ledger_end_time(tmp_path):
    task_root, archive, done, _ = make_root(tmp_path)
    ledger_path = task_root / "state" / "budget.json"
    ledger = json.loads(ledger_path.read_text())
    del ledger["attempts"]["0000_done"]["ended_unix"]
    ledger_path.write_text(json.dumps(ledger))
    archive.mkdir()
    with pytest.raises(sweep.SweepError, match="ended_unix"):
        sweep.sweep_once(task_root, archive, sweep.PathMap({}), 0.0, archive / "log")
    assert done.is_dir() and not done.is_symlink()


def test_rerun_after_copy_completed_verifies_existing_archive_and_swaps(tmp_path):
    task_root, archive, done, _ = make_root(tmp_path)
    archive.mkdir()
    import shutil

    shutil.copytree(done, archive / "attempts" / "0000_done")
    swept = sweep.sweep_once(task_root, archive, sweep.PathMap({}), 0.0, archive / "log")
    assert [r["key"] for r in swept] == ["0000_done"]
    assert done.is_symlink()


def test_rerun_refuses_existing_archive_that_differs(tmp_path):
    task_root, archive, done, _ = make_root(tmp_path)
    archive.mkdir()
    import shutil

    shutil.copytree(done, archive / "attempts" / "0000_done")
    (archive / "attempts" / "0000_done" / "case" / "ladder" / "search_abc" / "result.zip").write_bytes(
        b"different"
    )
    with pytest.raises(sweep.SweepError, match="copy (hash|size) differs"):
        sweep.sweep_once(task_root, archive, sweep.PathMap({}), 0.0, archive / "log")
    assert done.is_dir() and not done.is_symlink()


def test_rerun_after_rename_aside_finishes_the_link(tmp_path):
    task_root, archive, done, _ = make_root(tmp_path)
    archive.mkdir()
    import shutil

    shutil.copytree(done, archive / "attempts" / "0000_done")
    done.rename(done.with_name("0000_done.swept"))
    swept = sweep.sweep_once(task_root, archive, sweep.PathMap({}), 0.0, archive / "log")
    assert [r["key"] for r in swept] == ["0000_done"] and swept[0]["resumed_after_interruption"] is True
    assert done.is_symlink() and not done.with_name("0000_done.swept").exists()


def test_path_map_relocates_recorded_outputs(tmp_path):
    task_root, archive, done, _ = make_root(tmp_path)
    ledger_path = task_root / "state" / "budget.json"
    ledger = json.loads(ledger_path.read_text())
    recorded_root = Path("/remote/original/task_root")
    for key, item in ledger["attempts"].items():
        item["output"] = str(recorded_root / "attempts" / key)
    ledger_path.write_text(json.dumps(ledger))
    receipt_path = done / "worker_exit.json"
    receipt = json.loads(receipt_path.read_text())
    receipt["artifacts"] = {
        str(recorded_root / Path(p).relative_to(task_root)): d for p, d in receipt["artifacts"].items()
    }
    receipt_path.write_text(json.dumps(receipt))
    stale = time.time() - 600
    os.utime(receipt_path, (stale, stale))
    archive.mkdir()
    paths = sweep.PathMap({str(recorded_root): str(task_root)})
    swept = sweep.sweep_once(task_root, archive, paths, 0.0, archive / "log")
    assert [r["key"] for r in swept] == ["0000_done"]
    assert done.is_symlink()


def test_controller_alive_follows_the_flock(tmp_path):
    task_root, _, _, _ = make_root(tmp_path)
    lock = task_root / "state" / "controller.lock"
    assert sweep.controller_alive(task_root) is False
    handle = lock.open("a+")
    fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    try:
        assert sweep.controller_alive(task_root) is True
    finally:
        fcntl.flock(handle, fcntl.LOCK_UN)
        handle.close()
    assert sweep.controller_alive(task_root) is False


def test_finalize_copies_remaining_files_without_duplicating_swept_attempts(tmp_path):
    task_root, archive, done, running = make_root(tmp_path)
    archive.mkdir()
    sweep.sweep_once(task_root, archive, sweep.PathMap({}), 0.0, archive / "log")
    receipt = sweep.finalize(task_root, archive)
    assert receipt["status"] == "FINALIZED_WITH_UNSWEPT_ATTEMPTS"
    assert receipt["unswept_attempt_directories"] == ["0001_running"]
    assert (archive / "state" / "budget.json").is_file()
    assert (archive / "activated" / "specs" / "0000.json").is_file()
    assert (archive / "attempts" / "0001_running" / "worker_exit.json").is_file()
    assert not any("0000_done" in k for k in receipt["files"])
    assert json.loads((archive / "FINALIZE_RECEIPT.json").read_text())["status"] == receipt["status"]


def test_main_waits_for_a_controller_that_has_not_started(tmp_path, monkeypatch):
    task_root = tmp_path / "task_root"
    task_root.mkdir()
    archive = tmp_path / "archive"
    ticks = []

    def fake_sleep(seconds):
        ticks.append(seconds)
        if len(ticks) == 3:
            (task_root / "state").mkdir()
            (task_root / "state" / "budget.json").write_text(json.dumps({"cap_seconds": 1.0, "attempts": {}}))

    monkeypatch.setattr(sweep.time, "sleep", fake_sleep)
    assert sweep.main([str(task_root), str(archive), "--interval", "1"]) == 0
    assert len(ticks) == 3


def test_main_rejects_archive_inside_task_root(tmp_path):
    task_root, _, _, _ = make_root(tmp_path)
    with pytest.raises(SystemExit):
        sweep.main([str(task_root), str(task_root / "archive"), "--once"])


@pytest.mark.parametrize("partially_deleted", [False, True])
def test_restart_after_link_and_during_delete_reclaims_remaining_bytes(
    tmp_path, monkeypatch, partially_deleted
):
    task, archive, done, running = make_root(tmp_path)
    archive.mkdir()
    swept = done.with_name(done.name + ".swept")
    real_rmtree = sweep.shutil.rmtree

    def interrupt(path, *args, **kwargs):
        if Path(path) == swept:
            if partially_deleted:
                (swept / "worker_exit.json").unlink()
                (swept / "case/ladder/search_abc/search_internal.dill").unlink()
            raise RuntimeError("interrupted cleanup")
        return real_rmtree(path, *args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(sweep.shutil, "rmtree", interrupt)
        with pytest.raises(RuntimeError, match="interrupted cleanup"):
            sweep.sweep_once(task, archive, sweep.PathMap({}), 0, archive / "log")
    active_hashes = {p.relative_to(running): sha(p) for p in running.rglob("*") if p.is_file()}
    assert done.is_symlink() and swept.is_dir()
    records = sweep.sweep_once(task, archive, sweep.PathMap({}), 0, archive / "log")
    assert len(records) == 1 and records[0]["resumed_after_interruption"]
    assert not swept.exists() and (done / "worker_exit.json").is_file()
    assert active_hashes == {p.relative_to(running): sha(p) for p in running.rglob("*") if p.is_file()}


@pytest.mark.parametrize(
    "damage",
    ["archive_payload", "archive_receipt", "local_payload", "wrong_link", "extra_local", "archive_symlink"],
)
def test_cleanup_refuses_unverified_archive_or_remainder(tmp_path, damage):
    task, archive, done, _ = make_root(tmp_path)
    destination = archive / "attempts" / done.name
    sweep.shutil.copytree(done, destination)
    swept = done.with_name(done.name + ".swept")
    done.rename(swept)
    done.symlink_to(destination)
    payload = Path("case/ladder/search_abc/search_internal.dill")
    if damage == "archive_payload":
        (destination / payload).write_bytes(b"bad archive")
    elif damage == "archive_receipt":
        receipt = json.loads((destination / "worker_exit.json").read_text())
        receipt["status"] = "FAILED"
        (destination / "worker_exit.json").write_text(json.dumps(receipt))
    elif damage == "local_payload":
        (swept / payload).write_bytes(b"bad local")
    elif damage == "wrong_link":
        done.unlink()
        done.symlink_to(swept)
    elif damage == "extra_local":
        (swept / "extra").write_bytes(b"not archived")
    else:
        (destination / "worker.log").unlink()
        (destination / "worker.log").symlink_to(swept / "worker.log")
    with pytest.raises(sweep.SweepError):
        sweep.sweep_once(task, archive, sweep.PathMap({}), 0, archive / "log")
    assert swept.is_dir()


@pytest.mark.parametrize("point", ["before_directory_open", "after_directory_open", "during_deletion"])
def test_live_disk_scan_survives_archive_swap_without_entering_archive(tmp_path, monkeypatch, point):
    import threading

    task, archive, done, running = make_root(tmp_path)
    archive.mkdir()
    entered, resume = threading.Event(), threading.Event()
    result = {}
    real_open, real_scandir = os.open, os.scandir
    target_fd = []

    def opened(path, flags, *args, **kwargs):
        if threading.current_thread().name == "disk-scan" and path == done.name:
            if point == "before_directory_open":
                entered.set()
                assert resume.wait(5)
            fd = real_open(path, flags, *args, **kwargs)
            target_fd.append(fd)
            if point == "after_directory_open":
                entered.set()
                assert resume.wait(5)
            return fd
        return real_open(path, flags, *args, **kwargs)

    def scandir(fd):
        iterator = real_scandir(fd)
        if (
            threading.current_thread().name == "disk-scan"
            and target_fd
            and fd == target_fd[0]
            and point == "during_deletion"
        ):
            # Cache directory entries, then let the sweeper unlink the files.
            entries = list(iterator)
            iterator.close()
            entered.set()
            assert resume.wait(5)
            from contextlib import nullcontext

            return nullcontext(iter(entries))
        return iterator

    def scan():
        try:
            result["bytes"] = cached_task_bytes(task, {}, 0)[0]
        except BaseException as exc:
            result["error"] = exc

    monkeypatch.setattr(os, "open", opened)
    monkeypatch.setattr(os, "scandir", scandir)
    thread = threading.Thread(target=scan, name="disk-scan")
    thread.start()
    try:
        assert entered.wait(5)
        sweep.sweep_once(task, archive, sweep.PathMap({}), 0, archive / "log")
        # Inflate the archive with an unreceipted sentinel: a scan accidentally
        # redirected through the new symlink would count these remote bytes.
        (done / "remote_only").write_bytes(b"x" * 2_000_000)
    finally:
        resume.set()
        thread.join(5)
    assert not thread.is_alive() and "error" not in result, result
    assert 100_000 <= result["bytes"] < 500_000
    assert running.is_dir() and not running.is_symlink()


@pytest.mark.parametrize("error", [PermissionError(13, "denied"), OSError(5, "I/O error")])
def test_disk_scan_does_not_swallow_genuine_io_errors(tmp_path, monkeypatch, error):
    (tmp_path / "child").mkdir()
    real_open = os.open

    def fail(path, *args, **kwargs):
        if path == "child":
            raise error
        return real_open(path, *args, **kwargs)

    monkeypatch.setattr(os, "open", fail)
    with pytest.raises(type(error)):
        cached_task_bytes(tmp_path, {}, 0)


def test_single_sweeper_and_finalize_exclusion(tmp_path):
    task, archive, _, _ = make_root(tmp_path)
    archive.mkdir()
    with sweep.sweeper_lock(task):
        with pytest.raises(sweep.SweepError, match="sweeper still running"):
            sweep.main([str(task), str(archive), "--once"])
        with pytest.raises(sweep.SweepError, match="sweeper still running"):
            sweep.finalize(task, archive)
    with (task / "state/controller.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(sweep.SweepError, match="controller still running"):
            sweep.finalize(task, archive)


def test_actual_controller_scans_during_sweep_with_an_owned_worker(tmp_path, monkeypatch):
    """Real supervise/ledger/disk guard and real sweeper; only worker/GPU are fixtures."""
    import threading
    from test_profile_replay import controller_fixture
    from studies.rasti.campaign import profile_execution as execution

    root, manifest = controller_fixture(
        tmp_path,
        monkeypatch,
        'time.sleep(2.5)\n(out/"worker_exit.json").write_text(json.dumps('
        '{"status":"COMPLETE","elapsed_s":2.5,"artifacts":{}}))\n',
    )
    done = make_attempt(root, "done", "COMPLETE", 1000)
    (root / "state").mkdir()
    ledger = execution.BudgetLedger(root / "state/budget.json", 100)
    ledger.data["attempts"]["done"] = dict(
        output=str(done), status="COMPLETE", charged_seconds=0.1, ended_unix=time.time() - 600
    )
    ledger.save()
    archive = tmp_path / "archive"
    archive.mkdir()
    original_open, original_scan = os.open, execution.cached_task_bytes
    observations, failures = [], []

    def scan(root, cache, interval):
        return original_scan(root, cache, 0)

    def opened(path, flags, *args, **kwargs):
        if path == "done" and not done.is_symlink():
            current = json.loads((root / "state/budget.json").read_text())
            if current["attempts"].get("a", {}).get("status") == "RUNNING":

                def concurrent_sweep():
                    try:
                        observations.extend(
                            sweep.sweep_once(root, archive, sweep.PathMap({}), 0, archive / "log")
                        )
                    except BaseException as exc:
                        failures.append(exc)

                thread = threading.Thread(target=concurrent_sweep)
                thread.start()
                thread.join(5)
                assert not thread.is_alive()
        return original_open(path, flags, *args, **kwargs)

    monkeypatch.setattr(execution, "cached_task_bytes", scan)
    monkeypatch.setattr(os, "open", opened)
    execution.supervise(manifest)
    assert not failures and len(observations) == 1
    assert done.is_symlink() and not done.with_name("done.swept").exists()
    final = json.loads((root / "state/budget.json").read_text())
    assert final["attempts"]["a"]["status"] == "COMPLETE"
    assert not (root / "attempt").is_symlink()
