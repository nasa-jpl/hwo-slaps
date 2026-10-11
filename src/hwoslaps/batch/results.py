"""Batch metadata inspection and strict loading of completed scientific artifacts.

Import and open_batch are backend-free. Loading a case or an injected observation
retains the full physical archive validation of the existing scientific loaders.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from collections.abc import Mapping
from typing import Any

from ..config.loading import read_yaml
from .state import BatchConflict, BatchError, job_state, read_json, read_marker, verify_marker


_JOB_PATH = re.compile(r'members/(?P<run>[^/]+)/(?P<arm>[^/]+(?:/d[1-9][0-9]*)?)/'
    r'(?:(?P<forecast>forecast)|(?P<simulate>simulate)/r(?P<simulation>[0-9]{3,})|'
    r'(?P<nonlinear>nonlinear)/(?P<family>[^/]+)/(?P<trial>[^/]+)/r(?P<replicate>[0-9]{3,})/a(?P<attempt>[01])|'
    r'(?P<followup>followup)/(?P<followup_family>[^/]+)/m[0-9a-f]{16})')


@dataclass(frozen=True)
class JobRecord:
    job_id: str
    kind: str
    run_name: str
    arm: str
    status: str
    marker: Mapping[str, Any] | None
    path: Path
    family: str | None = None
    attempt: int | None = None
    trial: str | None = None
    replicate: int | None = None
    source_revision: str | None = None
    source_job_id: str | None = None


@dataclass(frozen=True)
class BatchResults:
    output_dir: Path
    spec: Mapping[str, Any]
    members: tuple[Mapping[str, Any], ...]
    jobs: tuple[JobRecord, ...]

    def forecast(self, run_name, arm):
        from ..artifacts import load_forecast
        jobs = [job for job in self.jobs if job.kind == 'forecast' and job.run_name == run_name
                and job.arm == arm and job.status == 'complete']
        if len(jobs) != 1:
            raise BatchError(f'expected one completed forecast for {run_name}/{arm}, found {len(jobs)}')
        job = jobs[0]
        verify_marker(job.path, job.marker, digests=False)
        return load_forecast(job.path / job.marker['artifacts']['forecast']['path'])

    def observations(self, run_name, arm):
        from ..artifacts import load_observation
        jobs = sorted((job for job in self.jobs if job.kind == 'simulate' and job.run_name == run_name
                       and job.arm == arm and job.status == 'complete'), key=lambda job: int(job.path.name[1:]))
        result = []
        for job in jobs:
            verify_marker(job.path, job.marker, digests=False)
            result.append(load_observation(job.path / job.marker['artifacts']['observation']['path']))
        return tuple(result)

    def cases(self, *, family=None, run_name=None, arm=None, attempt=0):
        """Load first attempts by default; ``attempt=None`` exposes both raw attempts."""
        from ..artifacts import load_case
        for job in self.jobs:
            if (job.kind != 'nonlinear' or job.status != 'complete'
                    or run_name is not None and job.run_name != run_name
                    or arm is not None and job.arm != arm
                    or attempt is not None and job.attempt != attempt):
                continue
            if family is not None and job.family != family:
                continue
            verify_marker(job.path, job.marker, digests=False)
            yield job, load_case(job.path / job.marker['artifacts']['case']['path'])

    def case_pairs(self, *, family=None, run_name=None, arm=None):
        """Yield ``(first_job, first_case, retry_case_or_None)`` for attempt selection."""
        paired = {}
        for job, case in self.cases(family=family, run_name=run_name, arm=arm, attempt=None):
            key = (job.run_name, job.arm, job.family, job.trial, job.replicate)
            entries = paired.setdefault(key, {})
            if job.attempt in entries:
                raise BatchConflict(f'duplicate nonlinear attempt for {key}')
            entries[job.attempt] = (job, case)
        for key, entries in sorted(paired.items()):
            if 0 not in entries:
                raise BatchConflict(f'retry has no completed first attempt: {key}')
            job, case = entries[0]
            yield job, case, None if 1 not in entries else entries[1][1]


def open_batch(output_dir) -> BatchResults:
    root = Path(output_dir).expanduser().resolve()
    sessions = root / 'sessions'
    numbers = sorted(int(path.name) for path in sessions.iterdir()
                     if path.is_dir() and path.name.isdecimal() and (path / 'batch_spec.yaml').is_file())
    if not numbers:
        raise BatchError(f'no recorded batch session in {root}')
    spec = read_yaml(sessions / str(numbers[-1]) / 'batch_spec.yaml')
    member_files = list((root / 'members').glob('*/member.json'))
    members = tuple(sorted((read_json(path) for path in member_files), key=lambda record: record['index']))
    paths = {path.parent for path in (root / 'members').rglob('complete.json')}
    paths.update(path.parent for path in (root / 'members').rglob('run_*')
                 if path.is_dir() and re.fullmatch(r'run_[0-9]{3,}', path.name))
    paths.update(path.parent for path in (root / 'members').rglob('failure.json')
                 if not re.fullmatch(r'run_[0-9]{3,}', path.parent.name))
    jobs = []
    for path in sorted(paths):
        identifier = path.relative_to(root).as_posix()
        match = _JOB_PATH.fullmatch(identifier)
        if match is None:
            continue
        marker = read_marker(path)
        if marker is not None:
            if marker.get('job_id') != identifier:
                raise BatchConflict(f'completion marker names another job: {path}')
            verify_marker(path, marker, digests=False)
        kind = next(name for name in ('forecast', 'simulate', 'nonlinear', 'followup') if match[name] is not None)
        failure = read_json(path / 'failure.json') if kind == 'followup' else None
        jobs.append(JobRecord(identifier, kind, match['run'], match['arm'], job_state(path), marker, path,
                              match['family'] or match['followup_family'],
                              None if match['attempt'] is None else int(match['attempt']), match['trial'],
                              None if match['replicate'] is None and match['simulation'] is None
                                   else int(match['replicate'] or match['simulation']),
                              None if marker is None else marker.get('source_revision'),
                              None if failure is None else failure['source_job_id']))
    return BatchResults(root, spec, members, tuple(jobs))
