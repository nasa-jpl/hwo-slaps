"""Move COMPLETE production attempts from the /data task root to an NFS archive.

The Stage 3 worker aborts the whole controller once the bytes under the task
root exceed ``task_disk_limit_gib`` (512), and /data cannot hold the ~1.25 TB
that 1,597 retained fits write. This sweeper runs beside the controller. For
every attempt the budget ledger marks COMPLETE it verifies the worker receipt
against the bytes on disk, copies the attempt directory to the archive root,
verifies the copy byte for byte, then replaces the original directory with a
symbolic link to the copy. The disk guard does not descend symbolic links, so
the swept bytes leave the task-root total, while every absolute path recorded
in receipts, ledgers and payloads keeps resolving for the controller and the
canonical harvester.

Nothing that is not COMPLETE is touched. Any verification failure raises and
leaves both copies on disk.
"""

from __future__ import annotations

import argparse
import datetime as dt
import fcntl
import hashlib
import json
import os
import shutil
import sys
import time
import threading
from contextlib import contextmanager
from pathlib import Path

_STUDY_REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(_STUDY_REPO_ROOT), str(_STUDY_REPO_ROOT / "src")]


class SweepError(RuntimeError):
    pass


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 22), b""):
            digest.update(chunk)
    return digest.hexdigest()


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


class PathMap:
    """Prefix relocation with the same semantics as the harvester's Paths."""

    def __init__(self, mappings: dict[str, str]):
        self.mappings = sorted(
            ((Path(a), Path(b)) for a, b in mappings.items()),
            key=lambda pair: len(pair[0].parts),
            reverse=True,
        )

    def __call__(self, value: str | Path) -> Path:
        path = Path(value)
        for source, destination in self.mappings:
            if path.is_relative_to(source):
                return destination / path.relative_to(source)
        return path


def relative_files(root: Path) -> dict[Path, int]:
    """Every regular file under root, relative, with its size; no symlinks."""
    result = {}
    for path in root.rglob("*"):
        if path.is_symlink():
            raise SweepError(f"unexpected symbolic link inside attempt: {path}")
        if path.is_file():
            result[path.relative_to(root)] = path.stat().st_size
    return result


def verify_receipt_against_disk(
    attempt_dir: Path, receipt: dict, recorded_output: Path, paths: PathMap
) -> int:
    """Every receipt artifact must exist under the attempt and match its digest."""
    artifacts = receipt.get("artifacts")
    if not isinstance(artifacts, dict) or not artifacts:
        raise SweepError(f"receipt without artifacts: {attempt_dir}")
    total = 0
    for recorded, digest in artifacts.items():
        original = Path(recorded)
        if not original.is_absolute() or not original.is_relative_to(recorded_output):
            raise SweepError(f"receipt artifact escapes attempt: {recorded}")
        actual = attempt_dir / original.relative_to(recorded_output)
        if not actual.is_file():
            raise SweepError(f"receipt artifact missing on disk: {actual}")
        if sha256_file(actual) != digest:
            raise SweepError(f"receipt artifact hash mismatch: {actual}")
        total += actual.stat().st_size
    return total


def verify_copy(source: Path, copy: Path) -> tuple[int, int]:
    source_files = relative_files(source)
    copy_files = relative_files(copy)
    if set(source_files) != set(copy_files):
        raise SweepError(f"copy file set differs from source: {copy}")
    total = 0
    for rel, size in source_files.items():
        if copy_files[rel] != size:
            raise SweepError(f"copy size differs: {copy / rel}")
        if sha256_file(source / rel) != sha256_file(copy / rel):
            raise SweepError(f"copy hash differs: {copy / rel}")
        total += size
    return len(source_files), total


def sweep_attempt(
    key: str, item: dict, task_root: Path, archive_root: Path, paths: PathMap, settle_seconds: float
) -> dict | None:
    """Archive one COMPLETE attempt; safe to rerun after a crash at any step.

    Steps: verify receipt against disk, copy to <archive>/attempts/<key>.partial,
    verify the copy, rename to the final name, rename the original to
    <key>.swept, create the symbolic link, verify through the link, delete the
    swept directory. A rerun finds the state left by an interrupted step and
    continues from there, always verifying bytes before trusting a copy.
    """
    recorded_output = Path(item["output"])
    attempt_dir = paths(recorded_output)
    attempts_dir = task_root / "attempts"
    if attempt_dir.parent != attempts_dir or attempt_dir.name != key:
        raise SweepError(f"ledger output is not attempts/<key> under the task root: {attempt_dir}")
    destination = archive_root / "attempts" / key
    swept = attempt_dir.with_name(key + ".swept")
    if destination.is_symlink() or swept.is_symlink():
        raise SweepError("archive or local remainder must not be a symbolic link")
    if attempt_dir.is_symlink():
        if attempt_dir.resolve(strict=True) != destination.resolve(strict=True):
            raise SweepError(f"symbolic link does not resolve into the archive: {attempt_dir}")
        if swept.exists():
            return _resume_swap(key, swept, attempt_dir, destination, recorded_output, paths)
        return None
    if not attempt_dir.exists() and swept.is_dir() and destination.is_dir():
        return _resume_swap(key, swept, attempt_dir, destination, recorded_output, paths)
    if not attempt_dir.is_dir():
        raise SweepError(f"COMPLETE attempt directory is missing: {attempt_dir}")
    receipt_path = attempt_dir / "worker_exit.json"
    if not receipt_path.is_file():
        raise SweepError(f"COMPLETE attempt has no worker receipt: {receipt_path}")
    ended = item.get("ended_unix")
    if not isinstance(ended, (int, float)) or isinstance(ended, bool):
        raise SweepError(f"COMPLETE ledger entry without ended_unix: {key}")
    if time.time() - float(ended) < settle_seconds:
        return None
    receipt = json.loads(receipt_path.read_text())
    if receipt.get("status") != "COMPLETE":
        raise SweepError(f"ledger says COMPLETE but receipt says {receipt.get('status')!r}: {key}")
    started = time.monotonic()
    artifact_bytes = verify_receipt_against_disk(attempt_dir, receipt, recorded_output, paths)

    if destination.is_symlink():
        raise SweepError(f"archive destination is a symbolic link: {destination}")
    partial = destination.with_name(key + ".partial")
    if destination.is_dir():
        file_count, total_bytes = verify_copy(attempt_dir, destination)
    else:
        if partial.exists():
            shutil.rmtree(partial)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(attempt_dir, partial, symlinks=False)
        file_count, total_bytes = verify_copy(attempt_dir, partial)
        partial.rename(destination)
    copied = time.monotonic()

    if swept.exists():
        raise SweepError(f"stale swept directory present beside a live attempt: {swept}")
    attempt_dir.rename(swept)
    os.symlink(destination, attempt_dir)
    _verify_link(attempt_dir, destination, receipt, recorded_output, paths)
    shutil.rmtree(swept)
    return {
        "key": key,
        "case_id": receipt.get("case_id"),
        "archived_to": str(destination),
        "files": file_count,
        "bytes": total_bytes,
        "receipt_artifact_bytes": artifact_bytes,
        "worker_exit_sha256": sha256_file(destination / "worker_exit.json"),
        "copy_seconds": round(copied - started, 3),
        "swap_seconds": round(time.monotonic() - copied, 3),
        "utc": utc_now(),
    }


def _verify_link(
    attempt_dir: Path, destination: Path, receipt: dict, recorded_output: Path, paths: PathMap
) -> None:
    if (
        not attempt_dir.is_symlink()
        or (attempt_dir / "worker_exit.json").resolve() != (destination / "worker_exit.json").resolve()
    ):
        raise SweepError(f"symbolic link does not resolve into the archive: {attempt_dir}")
    verify_receipt_against_disk(attempt_dir, receipt, recorded_output, paths)


def _resume_swap(
    key: str, swept: Path, attempt_dir: Path, destination: Path, recorded_output: Path, paths: PathMap
) -> dict:
    """Recover before linking, after linking, or during partial local deletion."""
    # The local receipt may already have been deleted. Verify the complete
    # archive first, then every remaining local byte before reclaiming it.
    if destination.is_symlink() or swept.is_symlink():
        raise SweepError("archive or local remainder must not be a symbolic link")
    archive_files = relative_files(destination)
    receipt = json.loads((destination / "worker_exit.json").read_text())
    if receipt.get("status") != "COMPLETE":
        raise SweepError(f"archive is not a COMPLETE attempt: {destination}")
    artifact_bytes = verify_receipt_against_disk(destination, receipt, recorded_output, paths)
    remaining = relative_files(swept)
    for rel, size in remaining.items():
        if archive_files.get(rel) != size or sha256_file(swept / rel) != sha256_file(destination / rel):
            raise SweepError(f"local remainder differs from verified archive: {swept / rel}")
    if not attempt_dir.is_symlink():
        # Before link creation, deletion has never started: require a full copy.
        verify_copy(swept, destination)
        os.symlink(destination, attempt_dir)
    _verify_link(attempt_dir, destination, receipt, recorded_output, paths)
    shutil.rmtree(swept)
    return {
        "key": key,
        "case_id": receipt.get("case_id"),
        "archived_to": str(destination),
        "files": len(remaining),
        "bytes": sum(remaining.values()),
        "receipt_artifact_bytes": artifact_bytes,
        "worker_exit_sha256": sha256_file(destination / "worker_exit.json"),
        "copy_seconds": None,
        "swap_seconds": None,
        "resumed_after_interruption": True,
        "utc": utc_now(),
    }


def sweep_once(
    task_root: Path, archive_root: Path, paths: PathMap, settle_seconds: float, log_path: Path
) -> list[dict]:
    ledger_path = task_root / "state" / "budget.json"
    if not ledger_path.is_file():
        return []
    ledger = json.loads(ledger_path.read_text())
    swept = []
    for key, item in sorted(ledger.get("attempts", {}).items()):
        if item.get("status") != "COMPLETE":
            continue
        record = sweep_attempt(key, item, task_root, archive_root, paths, settle_seconds)
        if record is not None:
            with log_path.open("a") as stream:
                stream.write(json.dumps(record, sort_keys=True) + "\n")
            print("swept", key, record["bytes"], "bytes", flush=True)
            swept.append(record)
    return swept


def controller_alive(task_root: Path) -> bool:
    """The controller holds an exclusive flock on state/controller.lock while it runs."""
    lock = task_root / "state" / "controller.lock"
    if not lock.is_file():
        return False
    with lock.open("a+") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            return True
        fcntl.flock(handle, fcntl.LOCK_UN)
        return False


@contextmanager
def sweeper_lock(task_root: Path):
    """One sweeper/finalizer per task, with a stable lock outside execution state."""
    with (task_root / "sweeper.lock").open("a+") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise SweepError("sweeper still running; refusing concurrent sweep/finalize") from exc
        yield


def finalize(task_root: Path, archive_root: Path) -> dict:
    with sweeper_lock(task_root):
        return _finalize(task_root, archive_root)


def _finalize(task_root: Path, archive_root: Path) -> dict:
    """Copy everything except the swept attempt links into the archive root."""
    if controller_alive(task_root):
        raise SweepError("controller still running; finalize only after it exits")
    copied = {}
    for path in task_root.rglob("*"):
        if path.is_symlink() or not path.is_file():
            continue
        if path.is_relative_to(task_root / "attempts") and any(p.is_symlink() for p in path.parents):
            continue
        rel = path.relative_to(task_root)
        target = archive_root / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            if sha256_file(target) != sha256_file(path):
                raise SweepError(f"archive already holds a different {rel}")
        else:
            shutil.copy2(path, target)
            if sha256_file(target) != sha256_file(path):
                raise SweepError(f"finalize copy hash differs: {rel}")
        copied[str(rel)] = sha256_file(target)
    unswept = (
        [p.name for p in (task_root / "attempts").iterdir() if p.is_dir() and not p.is_symlink()]
        if (task_root / "attempts").is_dir()
        else []
    )
    receipt = {
        "status": "FINALIZED" if not unswept else "FINALIZED_WITH_UNSWEPT_ATTEMPTS",
        "task_root": str(task_root),
        "archive_root": str(archive_root),
        "files_copied_or_verified": len(copied),
        "unswept_attempt_directories": unswept,
        "utc": utc_now(),
        "files": copied,
    }
    (archive_root / "FINALIZE_RECEIPT.json").write_text(json.dumps(receipt, indent=2) + "\n")
    return receipt


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("task_root", type=Path)
    parser.add_argument("archive_root", type=Path)
    parser.add_argument("--interval", type=float, default=120.0, help="seconds between sweeps")
    parser.add_argument(
        "--settle",
        type=float,
        default=120.0,
        help="minimum seconds since the ledger recorded the attempt end",
    )
    parser.add_argument("--once", action="store_true", help="sweep once and exit")
    parser.add_argument(
        "--finalize",
        action="store_true",
        help="copy the remaining task-root files after the controller exits",
    )
    parser.add_argument("--path-map", action="append", default=[], metavar="RECORDED=ACTUAL")
    args = parser.parse_args(argv)
    task_root = args.task_root.resolve()
    archive_root = args.archive_root.resolve()
    if not task_root.is_dir():
        parser.error(f"task root missing: {task_root}")
    if archive_root == task_root or archive_root.is_relative_to(task_root):
        parser.error("archive root must lie outside the task root")
    mappings = {}
    for value in args.path_map:
        if "=" not in value:
            parser.error("path-map requires RECORDED=ACTUAL")
        source, destination = value.split("=", 1)
        mappings[source] = destination
    paths = PathMap(mappings)
    archive_root.mkdir(parents=True, exist_ok=True)
    if args.finalize:
        receipt = finalize(task_root, archive_root)
        print(json.dumps({k: v for k, v in receipt.items() if k != "files"}, indent=2))
        return 0 if receipt["status"] == "FINALIZED" else 3
    with sweeper_lock(task_root):
        return run_loop(task_root, archive_root, paths, args)


def run_loop(task_root, archive_root, paths, args):
    """Publish readiness and a heartbeat even while copying a large attempt."""
    health_path = task_root / "sweeper_health.json"
    health = {
        "pid": os.getpid(),
        "script_sha256": sha256_file(Path(__file__)),
        "task_root": str(task_root),
        "archive_root": str(archive_root),
        "status": "READY",
        "progress_unix": time.time(),
    }
    stop = threading.Event()

    def heartbeat():
        while not stop.is_set():
            value = dict(health, heartbeat_unix=time.time())
            temporary = health_path.with_suffix(".tmp")
            temporary.write_text(json.dumps(value) + "\n")
            temporary.replace(health_path)
            stop.wait(5)

    thread = threading.Thread(target=heartbeat, daemon=True)
    thread.start()
    try:
        return _run_loop(task_root, archive_root, paths, args, health)
    finally:
        stop.set()
        thread.join()


def _run_loop(task_root, archive_root, paths, args, health):
    log_path = archive_root / "SWEEP_LOG.jsonl"
    ledger_seen = False
    while True:
        health.update(status="RUNNING", progress_unix=time.time())
        sweep_once(task_root, archive_root, paths, args.settle, log_path)
        if args.once:
            return 0
        ledger_seen = ledger_seen or (task_root / "state" / "budget.json").is_file()
        if (
            ledger_seen
            and not controller_alive(task_root)
            and not sweep_once(task_root, archive_root, paths, 0.0, log_path)
        ):
            print("controller gone and nothing left to sweep", utc_now(), flush=True)
            return 0
        time.sleep(args.interval)


if __name__ == "__main__":
    sys.exit(main())
