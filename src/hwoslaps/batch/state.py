"""Completion authority, atomic claims and controller-owned batch storage."""
from __future__ import annotations

from contextlib import contextmanager
import fcntl
import json
import os
from pathlib import Path
import re
import time
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from typing import Any, get_args

from ..analysis.nonlinear import CaseStatus, StatusResult
from ..artifacts import write_json
from ..config.checks import ConfigError, Key, ListOf, Sha256, Table, Text
from ..identity import canonical_json, file_digest
from ..inference.result import RoleStatus


_VERDICT_TABLE = Table((
    Key('status', Text(choices=get_args(CaseStatus)), 'actual worker classification'),
    Key('role_statuses', Table(tuple(Key(role, Text(choices=tuple(value.value for value in RoleStatus)),
                                        'actual role classification') for role in ('smooth', 'subhalo'))), 'role classifications'),
    Key('reasons', ListOf(Text()), 'actual classifier reasons'),
    Key('policy_digest', Sha256(), 'retry policy used by the classifier'),
    Key('case_sha256', Sha256(), 'fully validated case artifact bytes'),
))


@dataclass(frozen=True)
class RetryVerdict:
    """Typed transport of CLASS's outcome bound to the validated scientific artifact."""

    status: StatusResult
    policy_digest: str
    case_sha256: str

    def to_mapping(self):
        return {'status': self.status.status,
                'role_statuses': {role: value.value for role, value in self.status.role_statuses.items()},
                'reasons': list(self.status.reasons), 'policy_digest': self.policy_digest, 'case_sha256': self.case_sha256}

    @classmethod
    def from_mapping(cls, mapping):
        try:
            values = _VERDICT_TABLE.read(mapping, 'retry_verdict')
        except ConfigError as error:
            raise BatchConflict(str(error)) from error
        status = StatusResult(values['status'], {role: RoleStatus(value) for role, value in values['role_statuses'].items()},
                              tuple(values['reasons']))
        return cls(status, values['policy_digest'], values['case_sha256'])


class BatchError(RuntimeError):
    """A batch could not complete its requested work."""


class BatchConflict(BatchError):
    """Stored batch state disagrees with the requested inputs or artifacts."""


class BatchLocked(BatchError):
    """Another controller holds this batch's lock."""


class BatchIncomplete(BatchError):
    """A completed session contains failed jobs; the report remains available."""

    def __init__(self, report):
        self.report = report
        super().__init__(f"{report.counts['failed']} batch jobs failed; see {report.output_dir}")


@contextmanager
def batch_lock(output_dir: Path) -> Iterator[None]:
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / 'batch.lock').open('a+') as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise BatchLocked(f"another controller holds {output_dir / 'batch.lock'}") from error
        try:
            yield
        finally:
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


def next_session(output_dir: Path) -> int:
    sessions = output_dir / 'sessions'
    sessions.mkdir(exist_ok=True)
    number = 1 + max((int(path.name) for path in sessions.iterdir()
                      if path.is_dir() and path.name.isdecimal()), default=0)
    while True:
        try:
            (sessions / str(number)).mkdir()
        except FileExistsError:
            number += 1
        else:
            return number


def claim_run_dir(job_dir: Path) -> Path:
    job_dir.mkdir(parents=True, exist_ok=True)
    number = 1
    while True:
        path = job_dir / f'run_{number:03d}'
        try:
            path.mkdir()
        except FileExistsError:
            number += 1
        else:
            return path


def _invalid_constant(value):
    raise ValueError(f"nonfinite JSON constant {value}")


def read_json(path: Path) -> Mapping[str, Any]:
    try:
        value = json.loads(path.read_text(encoding='utf-8'), parse_constant=_invalid_constant)
    except (json.JSONDecodeError, ValueError) as error:
        raise BatchConflict(f"invalid JSON in {path}: {error}") from error
    if not isinstance(value, dict):
        raise BatchConflict(f"{path} must contain a JSON mapping")
    return value


def read_marker(job_dir: Path) -> Mapping[str, Any] | None:
    try:
        return read_json(job_dir / 'complete.json')
    except FileNotFoundError:
        return None


def verify_marker(job_dir: Path, marker: Mapping[str, Any], *, digests: bool) -> None:
    if marker.get('schema') != 1 or marker.get('kind') not in ('simulate', 'forecast', 'nonlinear'):
        raise BatchConflict(f"invalid completion schema/kind in {job_dir / 'complete.json'}")
    if not isinstance(marker.get('job_id'), str) or not re.fullmatch(r'[0-9a-f]{64}', str(marker.get('job_digest'))):
        raise BatchConflict(f"invalid completion identity in {job_dir / 'complete.json'}")
    run = marker.get('run')
    if not isinstance(run, str) or not re.fullmatch(r'run_[0-9]{3,}', run):
        raise BatchConflict(f"invalid run directory in {job_dir / 'complete.json'}")
    artifacts = marker.get('artifacts')
    if not isinstance(artifacts, Mapping) or not artifacts:
        raise BatchConflict(f"missing artifacts in {job_dir / 'complete.json'}")
    root = job_dir.resolve()
    for name, record in artifacts.items():
        if not isinstance(record, Mapping) or set(record) != {'path', 'bytes', 'sha256'}:
            raise BatchConflict(f"invalid artifact record {name!r} in {job_dir}")
        relative = record['path']
        if not isinstance(relative, str):
            raise BatchConflict(f"invalid artifact path {name!r} in {job_dir}")
        path = Path(relative)
        if path.is_absolute() or '..' in path.parts or not path.parts or path.parts[0] != run:
            raise BatchConflict(f"artifact path escapes its claimed run: {relative}")
        artifact = (root / path).resolve()
        if not artifact.is_relative_to(root / run) or not artifact.is_file():
            raise BatchConflict(f"missing or escaped completed artifact {artifact}")
        size, sha = record['bytes'], record['sha256']
        if isinstance(size, bool) or not isinstance(size, int) or size < 0:
            raise BatchConflict(f"invalid recorded byte size for {artifact}")
        if not isinstance(sha, str) or not re.fullmatch(r'[0-9a-f]{64}', sha):
            raise BatchConflict(f"invalid recorded SHA-256 for {artifact}")
        if artifact.stat().st_size != size:
            raise BatchConflict(f"completed artifact byte size changed: {artifact}")
        if digests and file_digest(artifact) != sha:
            raise BatchConflict(f"completed artifact SHA-256 changed: {artifact}")


def publish_marker(job_dir: Path, marker: Mapping[str, Any]) -> Path:
    verify_marker(job_dir, marker, digests=True)
    return write_json(job_dir / 'complete.json', marker)


def write_failure(run_dir: Path, failure: Mapping[str, Any]) -> Path:
    return write_json(run_dir / 'failure.json', failure)


def job_state(job_dir: Path) -> str:
    if read_marker(job_dir) is not None:
        return 'complete'
    if not job_dir.is_dir():
        return 'absent'
    runs = sorted((path for path in job_dir.iterdir() if path.is_dir()
                   and re.fullmatch(r'run_[0-9]{3,}', path.name)), key=lambda path: int(path.name[4:]))
    if not runs:
        return 'absent'
    return 'failed' if (runs[-1] / 'failure.json').is_file() else 'incomplete'


class EventLog:
    """One fsynced canonical line per event; only the controller writes this log."""

    def __init__(self, path: Path, *, session: int) -> None:
        self.session = session
        self._stream = path.open('a', encoding='utf-8')

    def append(self, event: Mapping[str, Any]) -> None:
        self._stream.write(canonical_json({**event, 'session': self.session, 'time': time.time()}) + '\n')
        self._stream.flush()
        os.fsync(self._stream.fileno())

    def close(self) -> None:
        self._stream.close()


def source_revision(provenance: Mapping[str, Any]) -> str:
    """Readable source key used consistently by session, worker and completed jobs."""
    source = provenance['source']
    if source is None:
        return 'version:' + str(provenance['hwoslaps_version'])
    commit = source['commit']
    return commit if not source['dirty'] else commit + '+worktree:' + source['worktree_sha256']
