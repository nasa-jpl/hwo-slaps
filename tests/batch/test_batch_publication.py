"""Publication, rather than notification delivery, determines a batch outcome."""
from dataclasses import replace
import json
from pathlib import Path
import subprocess
import sys

import pytest

from hwoslaps.artifacts import write_json
from hwoslaps.batch import BatchConflict, BatchIncomplete, plan_batch, run_batch
from hwoslaps.batch.processes import group_members
from hwoslaps.batch.runner import _terminal_outcome
from hwoslaps.batch.state import claim_run_dir, publish_marker
from hwoslaps.identity import file_digest


def _records(tmp_path, tiny_batch_spec):
    job = plan_batch(tiny_batch_spec()).jobs[0]
    run = claim_run_dir(tmp_path / job.job_id)
    worker = {'slot': 0, 'pid': 1234, 'device': 'cpu', 'device_kind': 'cpu'}
    failure = {'schema': 1, 'job_id': job.job_id, 'job_digest': job.digest(), 'session': 2,
               'worker': worker, 'error_type': 'ValueError', 'message': 'original cause', 'traceback': 'original traceback'}
    return job, run, worker, failure


@pytest.mark.parametrize('defect', ['job_id', 'job_digest', 'session', 'worker', 'schema', 'traceback'])
def test_active_failure_identity_is_checked(tiny_batch_spec, tmp_path, defect):
    job, run, worker, failure = _records(tmp_path, tiny_batch_spec)
    failure[defect] = None
    write_json(run / 'failure.json', failure)
    with pytest.raises(BatchConflict, match='active-attempt failure'):
        _terminal_outcome(job, tmp_path, run=run.name, session=2, worker=worker)


def test_old_attempt_is_never_consumed_when_current_publication_is_absent(tiny_batch_spec, tmp_path):
    job, old, worker, failure = _records(tmp_path, tiny_batch_spec)
    write_json(old / 'failure.json', failure)
    active = claim_run_dir(tmp_path / job.job_id)
    outcome, recorded = _terminal_outcome(job, tmp_path, run=active.name, session=2, worker=worker,
        synthetic={'error_type': 'WorkerExited', 'message': 'transport', 'exit_code': 17})
    assert outcome == 'failed' and recorded['error_type'] == 'WorkerExited'
    assert json.loads((old / 'failure.json').read_text()) == failure


@pytest.mark.parametrize('winner', ['failure', 'complete', 'invalid'])
def test_synthetic_publication_collision_validates_the_winner(tiny_batch_spec, tmp_path, monkeypatch, winner):
    import hwoslaps.batch.runner as runner
    job, run, worker, failure = _records(tmp_path, tiny_batch_spec)
    def race(directory, synthetic):
        if winner == 'complete':
            artifact = run / 'science.bin'
            artifact.write_bytes(b'scientific result')
            publish_marker(tmp_path / job.job_id, {'schema': 1, 'kind': job.kind, 'job_id': job.job_id,
                'job_digest': job.digest(), 'run': run.name, 'source_revision': 'test', 'artifacts': {'forecast': {
                    'path': str(artifact.relative_to(tmp_path / job.job_id)), 'bytes': artifact.stat().st_size,
                    'sha256': file_digest(artifact)}}, 'session': 2, 'worker': worker})
        else:
            write_json(directory / 'failure.json', failure if winner == 'failure' else {**failure, 'session': 1})
        raise FileExistsError('concurrent publication won')
    monkeypatch.setattr(runner, 'write_failure', race)
    if winner == 'invalid':
        with pytest.raises(BatchConflict, match='active-attempt failure'):
            _terminal_outcome(job, tmp_path, run=run.name, session=2, worker=worker,
                synthetic={'error_type': 'WorkerExited', 'message': 'transport'})
    else:
        outcome, record = _terminal_outcome(job, tmp_path, run=run.name, session=2, worker=worker,
            synthetic={'error_type': 'WorkerExited', 'message': 'transport'})
        assert outcome == ('complete' if winner == 'complete' else 'failed')
        if winner == 'failure':
            assert record == failure


@pytest.mark.parametrize('defect', ['digest', 'artifact', 'identity', 'session', 'worker'])
def test_corrupt_completion_is_an_integrity_conflict(tiny_batch_spec, tmp_path, defect):
    job, run, worker, failure = _records(tmp_path, tiny_batch_spec)
    artifact = run / 'result.bin'
    artifact.write_bytes(b'abcd')
    marker = {'schema': 1, 'kind': job.kind, 'job_id': job.job_id, 'job_digest': job.digest(),
              'run': run.name, 'source_revision': 'test', 'session': 2, 'worker': worker, 'artifacts': {'forecast': {
                  'path': str(artifact.relative_to(tmp_path / job.job_id)), 'bytes': 4, 'sha256': file_digest(artifact)}}}
    publish_marker(tmp_path / job.job_id, marker)
    if defect == 'artifact':
        artifact.write_bytes(b'WXYZ')
    else:
        path = tmp_path / job.job_id / 'complete.json'
        marker[{'digest': 'job_digest', 'identity': 'job_id'}.get(defect, defect)] = '0' * 64
        path.write_text(json.dumps(marker))
    with pytest.raises(BatchConflict):
        _terminal_outcome(job, tmp_path, run=run.name, session=2, worker=worker,
            synthetic={'error_type': 'WorkerExited', 'message': 'transport'})
    assert not (run / 'failure.json').exists()


def _fault_worker(tmp_path, monkeypatch, state, delivery, lost_start=False, duplicate=False):
    entry = tmp_path / 'publication-worker.py'
    entry.write_text(f'''
import os, time
import hwoslaps.batch.worker as worker
import hwoslaps.simulation as simulation
actual = worker.run_job
actual_simulate = simulation.simulate
class Connection:
    def __init__(self, connection): self.connection = connection
    def send(self, message):
        if not ({lost_start!r} and message['type'] == 'started'):
            self.connection.send(message)
def fault(*args, **kwargs):
    if {state!r} == 'conflict': raise worker.BatchConflict('original published conflict')
    raise ValueError('original published worker failure')
def wrapped(payload, *args, **kwargs):
    first = 'system_000000/' in payload.job_id and not payload.job_id.endswith('/followup')
    if first and {state!r} == 'none':
        if {delivery!r} == 'event':
            return {{'type': 'failed', 'job_id': payload.job_id, 'run': payload.run, 'failure': {{}}}}
        if {delivery!r} == 'exit': os._exit(17)
        time.sleep(60.)
    if first and {state!r} in ('failure', 'conflict'): simulation.simulate = fault
    kwargs['connection'] = Connection(kwargs['connection'])
    try: result = actual(payload, *args, **kwargs)
    finally: simulation.simulate = actual_simulate
    if first:
        if {delivery!r} == 'exit': os._exit(17)
        if {delivery!r} == 'timeout': time.sleep(60.)
        if {duplicate!r}: kwargs['connection'].send(result)
    return result
worker.run_job = wrapped
if __name__ == '__main__': raise SystemExit(worker.main())
''')
    actual_popen = subprocess.Popen
    def wrapped(command, *args, **kwargs):
        if command == [sys.executable, '-m', 'hwoslaps.batch.worker']:
            command = [sys.executable, str(entry)]
        return actual_popen(command, *args, **kwargs)
    monkeypatch.setattr(subprocess, 'Popen', wrapped)


@pytest.mark.backend
@pytest.mark.parametrize('state,delivery,lost_start', [
    *((state, delivery, False) for state in ('none', 'failure', 'complete') for delivery in ('event', 'exit', 'timeout')),
    *((state, delivery, True) for state in ('failure', 'complete') for delivery in ('exit', 'timeout')),
])
def test_real_publication_delivery_matrix(tiny_batch_spec, tmp_path, monkeypatch, state, delivery, lost_start):
    _fault_worker(tmp_path, monkeypatch, state, delivery, lost_start)
    spec = tiny_batch_spec(forecast=None, simulate={'inject': False, 'noise': False},
                          execution={'workers_per_device': 1, 'job_timeout_seconds': 5. if delivery == 'timeout' else None})
    output = tmp_path / 'batch'
    if state == 'none' and delivery == 'event':
        with pytest.raises(BatchConflict, match='no published outcome'):
            run_batch(spec, output)
    else:
        if state == 'complete':
            report = run_batch(spec, output)
        else:
            with pytest.raises(BatchIncomplete) as error:
                run_batch(spec, output)
            report = error.value.report
        assert report.counts['completed'] == (2 if state == 'complete' else 1)
        assert report.counts['failed'] == (0 if state == 'complete' else 1)
        assert sum(report.revisions.values()) == report.counts['completed']
        if state == 'failure':
            failure = next((output / 'members/system_000000').rglob('failure.json'))
            recorded = json.loads(failure.read_text())
            assert recorded['error_type'] == 'ValueError'
            assert recorded['message'] == 'original published worker failure'
            assert 'ValueError' in recorded['traceback']
        if state == 'complete':
            assert not list(output.rglob('failure.json'))
    workers = [json.loads(line) for line in (output / 'sessions/1/workers.jsonl').read_text().splitlines()]
    assert workers and all(not group_members(worker) for worker in workers)


@pytest.mark.backend
@pytest.mark.parametrize('delivery', ['event', 'exit', 'timeout'])
def test_published_conflict_stops_new_work_in_every_delivery_route(tiny_batch_spec, tmp_path, monkeypatch, delivery):
    _fault_worker(tmp_path, monkeypatch, 'conflict', delivery, lost_start=True)
    spec = tiny_batch_spec(forecast=None, simulate={'inject': False, 'noise': False},
                          execution={'workers_per_device': 1, 'job_timeout_seconds': 5. if delivery == 'timeout' else None})
    output = tmp_path / 'conflict'
    with pytest.raises(BatchConflict, match='original published conflict'):
        run_batch(spec, output)
    report = json.loads((output / 'sessions/1/report.json').read_text())
    assert report['counts']['failed'] == 1 and report['counts']['completed'] == 0
    assert report['failures'][0]['error_type'] == 'BatchConflict'
    assert not list((output / 'members/system_000001').rglob('run_*'))
    workers = [json.loads(line) for line in (output / 'sessions/1/workers.jsonl').read_text().splitlines()]
    assert workers and all(not group_members(worker) for worker in workers)


@pytest.mark.backend
def test_duplicate_terminal_events_count_revisions_and_enqueue_followups_once(tiny_batch_spec, tmp_path, monkeypatch):
    import hwoslaps.batch.runner as runner
    _fault_worker(tmp_path, monkeypatch, 'complete', 'event', duplicate=True)
    calls = []
    def following(plan, job, output):
        calls.append(job.job_id)
        return () if job.job_id.endswith('/followup') else (replace(job, job_id=job.job_id + '/followup'),)
    monkeypatch.setattr(runner, 'follow_ups', following)
    spec = tiny_batch_spec(population={'count': 1}, forecast=None, simulate={'inject': False, 'noise': False},
                          execution={'workers_per_device': 1})
    output = tmp_path / 'duplicates'
    report = run_batch(spec, output)
    assert report.counts['completed'] == 2 and report.counts['failed'] == 0
    assert sum(report.revisions.values()) == 2 and len(calls) == len(set(calls)) == 2
    events = [json.loads(line) for line in (output / 'events.jsonl').read_text().splitlines()]
    assert sum(event['type'] == 'reconciled' for event in events) == 2
    assert any(event['type'] == 'duplicate_event' for event in events)
