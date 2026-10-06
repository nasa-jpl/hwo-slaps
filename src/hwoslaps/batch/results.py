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
                       r'(?P<kind>forecast|simulate/r[0-9]{3,}|nonlinear/[^/]+/[^/]+/r[0-9]{3,}/a[01])')


@dataclass(frozen=True)
class JobRecord:
    job_id: str
    kind: str
    run_name: str
    arm: str
    status: str
    marker: Mapping[str, Any] | None
    path: Path


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

    def cases(self, *, family=None, run_name=None, arm=None):
        from ..artifacts import load_case
        for job in self.jobs:
            if (job.kind != 'nonlinear' or job.status != 'complete'
                    or run_name is not None and job.run_name != run_name
                    or arm is not None and job.arm != arm):
                continue
            tail = job.job_id.split('/nonlinear/', 1)[1]
            if family is not None and tail.split('/', 1)[0] != family:
                continue
            verify_marker(job.path, job.marker, digests=False)
            yield job, load_case(job.path / job.marker['artifacts']['case']['path'])


def open_batch(output_dir) -> BatchResults:
    root = Path(output_dir).expanduser().resolve()
    sessions = root / 'sessions'
    numbers = sorted(int(path.name) for path in sessions.iterdir() if path.is_dir() and path.name.isdecimal())
    if not numbers:
        raise BatchError(f'no recorded batch session in {root}')
    spec = read_yaml(sessions / str(numbers[-1]) / 'batch_spec.yaml')
    member_files = list((root / 'members').glob('*/member.json'))
    members = tuple(sorted((read_json(path) for path in member_files), key=lambda record: record['index']))
    paths = {path.parent for path in (root / 'members').rglob('complete.json')}
    paths.update(path.parent for path in (root / 'members').rglob('run_*')
                 if path.is_dir() and re.fullmatch(r'run_[0-9]{3,}', path.name))
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
        jobs.append(JobRecord(identifier, match['kind'].split('/')[0], match['run'], match['arm'],
                              job_state(path), marker, path))
    return BatchResults(root, spec, members, tuple(jobs))
