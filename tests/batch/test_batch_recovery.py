"""Real controller interruption, earlier-worker barriers and worker death recovery."""
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import pytest

from hwoslaps.artifacts import write_yaml
from hwoslaps.batch import BatchIncomplete, run_batch
from hwoslaps.batch.processes import group_members, signal_owned, worker_record
from hwoslaps.batch.runner import OWNER_ENV

pytestmark = pytest.mark.backend


def _wait(predicate, timeout=180.):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        value = predicate()
        if value:
            return value
        time.sleep(.02)
    raise AssertionError('real controller condition did not occur within its bounded owner control')


def _events(root):
    path = root / 'events.jsonl'
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines(keepends=True) if line.endswith('\n')]


def _worker_records(root):
    records = []
    for path in (root / 'sessions').glob('*/workers.jsonl'):
        records.extend(json.loads(line) for line in path.read_text().splitlines(keepends=True) if line.endswith('\n'))
    return records


def _controller(spec, root, tmp_path, label, extra=(), program=None):
    source = write_yaml(tmp_path / (label + '.yaml'), spec.to_mapping())
    environment = dict(os.environ)
    identity = os.urandom(32).hex()
    environment[OWNER_ENV] = identity
    actual_source = Path(__file__).resolve().parents[2] / 'src'
    environment['PYTHONPATH'] = str(actual_source) + (os.pathsep + environment['PYTHONPATH'] if environment.get('PYTHONPATH') else '')
    script = 'import sys\nif sys.stdin.readline() != "go\\n": raise SystemExit(2)\n'
    if program is None:
        script += 'from hwoslaps.cli import main\nraise SystemExit(main(sys.argv[1:]))\n'
    else:
        script += program
    path = tmp_path / (label + '.py')
    path.write_text(script)
    log = (tmp_path / (label + '.log')).open('wb')
    command = [sys.executable, str(path), 'batch', 'run', str(source), '-o', str(root), *extra]
    process = subprocess.Popen(command, env=environment, stdin=subprocess.PIPE, stdout=log,
                               stderr=subprocess.STDOUT, start_new_session=True)
    record = worker_record(process.pid, slot=0, device='cpu', identity=identity, identity_env=OWNER_ENV)
    process.stdin.write(b'go\n')
    process.stdin.close()
    return process, record, log


def _cleanup(controller, identity, log, output):
    try:
        for record in [identity, *_worker_records(output)]:
            signal_owned(record, signal.SIGKILL)
        controller.wait(timeout=10.)
        _wait(lambda: not any(group_members(record) for record in _worker_records(output)), timeout=10.)
    finally:
        log.close()


def test_controller_imports_no_backend(tiny_batch_spec, tmp_path):
    spec = tiny_batch_spec(population={'count': 1})
    root = tmp_path / 'pure'
    result = tmp_path / 'modules.json'
    program = f'''
import json
from pathlib import Path
from hwoslaps.batch import load_batch_spec, run_batch, open_batch
run_batch(load_batch_spec(sys.argv[3]), sys.argv[5])
open_batch(sys.argv[5])
blocked = ('autolens','autogalaxy','autoarray','autofit','hcipy','jax','nautilus','matplotlib')
Path({str(result)!r}).write_text(json.dumps([name for name in sys.modules if name.split('.')[0] in blocked]))
'''
    process, identity, log = _controller(spec, root, tmp_path, 'pure-controller', program=program)
    try:
        assert process.wait(timeout=240.) == 0
        assert json.loads(result.read_text()) == []
    finally:
        _cleanup(process, identity, log, root)


def test_controller_crash_then_resume_and_immediate_barrier(tiny_batch_spec, tmp_path):
    spec = tiny_batch_spec(execution={'workers_per_device': 1},
        forecast_positions={'kind': 'grid', 'spacing_arcsec': .01, 'half_width_arcsec': .6})
    root = tmp_path / 'crash'
    process, identity, log = _controller(spec, root, tmp_path, 'crashed')
    try:
        _wait(lambda: any(event['type'] == 'started' for event in _events(root)))
        worker = _worker_records(root)[0]
        assert group_members(worker), 'actual earlier worker must be live before the kill'
        descriptor = os.pidfd_open(process.pid)
        try:
            signal.pidfd_send_signal(descriptor, signal.SIGKILL)
        finally:
            os.close(descriptor)
        process.wait(timeout=10.)
        assert group_members(worker), 'enlarge real forecast workload if the immediate recovery control is vacuous'
        resumed_at = time.time()
        report = run_batch(spec, root)
        second = [event for event in _events(root) if event['session'] == 2]
        waiting = next(event for event in second if event['type'] == 'waiting_for_earlier_workers')
        exited = next(event for event in second if event['type'] == 'earlier_workers_exited')
        ready = next(event for event in second if event['type'] == 'worker_ready')
        assert worker['pid'] in waiting['pids']
        assert ready['time'] > exited['time'] >= resumed_at
        assert exited['time'] - resumed_at > ready['time'] - exited['time'], 'real orphan work must outlast a new worker startup'
        assert report.counts['skipped'] == 1 and report.counts['completed'] == 1
        first_job = root / 'members/system_000000/base/forecast'
        assert len(list(first_job.glob('run_*'))) == 1
        assert not any(group_members(record) for record in _worker_records(root))
    finally:
        _cleanup(process, identity, log, root)


def test_worker_death_is_a_job_failure(tiny_batch_spec, tmp_path):
    spec = tiny_batch_spec(execution={'workers_per_device': 1},
        forecast_positions={'kind': 'grid', 'spacing_arcsec': .01, 'half_width_arcsec': .6})
    root = tmp_path / 'death'
    process, identity, log = _controller(spec, root, tmp_path, 'worker-death')
    try:
        _wait(lambda: any(event['type'] == 'started' for event in _events(root)))
        worker = _worker_records(root)[0]
        descriptor = os.pidfd_open(worker['pid'])
        try:
            signal.pidfd_send_signal(descriptor, signal.SIGKILL)
        finally:
            os.close(descriptor)
        assert process.wait(timeout=300.) == 3
        failure = json.loads((root / 'members/system_000000/base/forecast/run_001/failure.json').read_text())
        assert failure['error_type'] == 'WorkerExited' and failure['exit_code'] == -signal.SIGKILL
        assert (root / 'members/system_000001/base/forecast/complete.json').exists()
        assert len(_worker_records(root)) >= 2
    finally:
        _cleanup(process, identity, log, root)


@pytest.mark.parametrize('erase_records,change_job', [(False, True), (True, True), (True, False)])
def test_changed_job_during_recovery_is_a_conflict(tiny_batch_spec, tmp_path, erase_records, change_job):
    from hwoslaps.batch import BatchConflict, parse_batch
    from hwoslaps.batch.jobs import plan_batch
    family = {'trials': {'kind': 'forecast_argmax', 'masses_msun': [1e8]}, 'inject': True,
              'noise': False, 'fit': {'mode': 'fixed_template'}}
    spec = tiny_batch_spec(population={'count': 1}, execution={'workers_per_device': 1}, nonlinear={'n': family},
                          forecast_positions={'kind': 'grid', 'spacing_arcsec': .01, 'half_width_arcsec': .6})
    forecast_job = next(job for job in plan_batch(spec).jobs if job.kind == 'forecast')
    root = tmp_path / 'duplicate'
    process, identity, log = _controller(spec, root, tmp_path, 'duplicate-controller',
                                        extra=('--select', forecast_job.job_id))
    original_records = []
    try:
        _wait(lambda: any(event['type'] == 'started' for event in _events(root)))
        original_records = _worker_records(root)
        descriptor = os.pidfd_open(process.pid)
        try:
            signal.pidfd_send_signal(descriptor, signal.SIGKILL)
        finally:
            os.close(descriptor)
        process.wait(timeout=10.)
        assert any(group_members(record) for record in original_records), 'duplicate publication control must be live'
        if erase_records:
            (root / 'sessions/1/workers.jsonl').unlink()
        mapping = spec.to_mapping()
        if change_job:
            mapping['forecast']['masses_msun'] = [1e7, 1e8]
        resumed = parse_batch(mapping, base_dir=spec.base_dir)
        if change_job:
            with pytest.raises(BatchConflict, match=forecast_job.job_id):
                run_batch(resumed, root, select=forecast_job.job_id)
            events = [event for event in _events(root) if event['session'] == 2]
            if not erase_records:
                assert not any(event['type'] == 'started' for event in events)
            else:
                failed = next(event for event in events if event['type'] == 'failed')
                assert failed['failure']['error_type'] == 'BatchConflict'
        else:
            report = run_batch(resumed, root, select=forecast_job.job_id)
            assert report.counts['duplicate'] == 1 and report.counts['failed'] == 0
            assert report.counts['not_selected'] == 1
        marker = json.loads((root / forecast_job.job_id / 'complete.json').read_text())
        assert marker['job_digest'] == forecast_job.digest()
        assert not (root / 'members/system_000000/base/nonlinear').exists()
    finally:
        for record in original_records:
            signal_owned(record, signal.SIGKILL)
        _cleanup(process, identity, log, root)
