"""Actual CLI plan/run/status wiring with inherited source binding."""
import json
import os
from pathlib import Path
import subprocess
import sys
import signal
import time

import numpy as np
import pytest

from hwoslaps.artifacts import write_yaml
from hwoslaps.batch.processes import boot_id, group_members, signal_owned, worker_record
from hwoslaps.batch.runner import OWNER_ENV

pytestmark = pytest.mark.backend


def _cli(arguments):
    environment = dict(os.environ)
    source = Path(__file__).resolve().parents[2] / 'src'
    environment['PYTHONPATH'] = str(source) + (os.pathsep + environment['PYTHONPATH'] if environment.get('PYTHONPATH') else '')
    identity = os.urandom(32).hex()
    environment[OWNER_ENV] = identity
    generation_boot = boot_id()
    output = Path(arguments[arguments.index('-o') + 1]) if '-o' in arguments else None
    entry = ('if __name__ == "__main__":\n'
             '    import sys, runpy\n'
             '    if sys.stdin.readline() != "go\\n": raise SystemExit(2)\n'
             '    sys.argv = ["hwoslaps", *sys.argv[1:]]\n'
             '    runpy.run_module("hwoslaps", run_name="__main__")\n')
    command = [sys.executable, '-c', entry, *arguments]
    process = subprocess.Popen(command, env=environment, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE, text=True, start_new_session=True)
    own = {'slot': 0, 'device': 'cpu', 'pid': process.pid, 'process_group': process.pid,
           'session_id': process.pid, 'identity': identity, 'identity_env': OWNER_ENV,
           'uid': os.getuid(), 'start_time': None, 'boot_id': generation_boot}
    known, errors, blocked = {}, [], set()
    def records():
        required = {'slot', 'device', 'pid', 'process_group', 'session_id', 'identity',
                    'identity_env', 'uid', 'start_time', 'boot_id'}
        if output is not None:
            for path in (output / 'sessions').glob('*/workers.jsonl'):
                try:
                    lines = path.read_text().splitlines(keepends=True)
                except OSError as error:
                    message = f'{path}: {error}'
                    if message not in errors:
                        errors.append(message)
                    continue
                for line in lines:
                    if not line.endswith('\n'):
                        continue
                    try:
                        record = json.loads(line)
                        if not isinstance(record, dict) or set(record) != required or record['identity_env'] != OWNER_ENV:
                            raise ValueError('invalid ownership shape or environment name')
                        for field in ('pid', 'process_group', 'session_id', 'start_time', 'uid'):
                            if isinstance(record[field], bool) or not isinstance(record[field], int) or record[field] < 0:
                                raise ValueError(f'invalid ownership field {field}')
                        if not isinstance(record['identity'], str) or not isinstance(record['boot_id'], str):
                            raise ValueError('invalid ownership identity or boot')
                        known[(record['pid'], record['identity'])] = record
                    except (ValueError, KeyError, TypeError) as error:
                        message = f'{path}: {error}'
                        if message not in errors:
                            errors.append(message)
        return [own, *known.values()]
    def guarded(record, number=None):
        key = (record['pid'], record['identity'])
        if key in blocked:
            return False
        try:
            if number is not None:
                signal_owned(record, number)
            return bool(group_members(record))
        except Exception as error:
            blocked.add(key)
            errors.append(str(error))
            return False
    try:
        own = worker_record(process.pid, slot=0, device='cpu', identity=identity, identity_env=OWNER_ENV)
        stdout, stderr = process.communicate('go\n', timeout=240.)
        if any(guarded(record) for record in records()) or errors:
            raise AssertionError('actual CLI returned with survivors or invalid ownership metadata')
        return subprocess.CompletedProcess(command, process.returncode, stdout, stderr)
    except BaseException as original:
        for record in records():
            guarded(record, signal.SIGTERM)
        deadline = time.monotonic() + 5.
        while time.monotonic() < deadline and any(guarded(record) for record in records()):
            process.poll()
            time.sleep(.02)
        deadline = time.monotonic() + 5.
        while True:
            for record in records():
                guarded(record, signal.SIGKILL)
            if not any(guarded(record) for record in records()):
                break
            if time.monotonic() >= deadline:
                errors.append('verified CLI descendants did not drain')
                break
            time.sleep(.02)
        try:
            process.wait(timeout=5.)
        except subprocess.TimeoutExpired:
            errors.append('CLI leader remains after pinned cleanup')
        if errors:
            raise AssertionError('CLI ownership cleanup failed: ' + '; '.join(errors)) from original
        raise
    finally:
        for stream in (process.stdin, process.stdout, process.stderr):
            stream.close()


def test_cli_batch_exit_codes_and_overrides(minimal_mapping, tmp_path):
    mapping = {'name': 'cli', 'seed': 7, 'config': minimal_mapping,
               'arms': [{'name': 'a'}, {'name': 'b'}], 'forecast': {'masses_msun': [1e8]},
               'execution': {'devices': 'cpu', 'workers_per_device': 2}}
    source = write_yaml(tmp_path / 'batch.yaml', mapping)
    before = sorted(path.relative_to(tmp_path).as_posix() for path in tmp_path.rglob('*'))
    planned = _cli(['batch', 'plan', str(source)])
    assert planned.returncode == 0, planned.stderr
    assert json.loads(planned.stdout)['static_jobs']['forecast'] == 2
    assert sorted(path.relative_to(tmp_path).as_posix() for path in tmp_path.rglob('*')) == before
    output = tmp_path / 'run'
    ran = _cli(['batch', 'run', str(source), '-o', str(output), '--devices', 'cpu',
                '--workers-per-device', '1', '--select', 'members/run/a/*'])
    assert ran.returncode == 0, ran.stderr
    report = json.loads(ran.stdout)
    assert report['counts']['completed'] == 1 and report['counts']['not_selected'] == 1
    session = json.loads((output / 'sessions/1/session.json').read_text())
    assert session['execution']['devices'] == 'cpu' and session['execution']['workers_per_device'] == 1
    status = _cli(['batch', 'status', str(output)])
    assert status.returncode == 0, status.stderr
    assert json.loads(status.stdout)['counts'] == {'forecast:complete': 1}
    fresh = _cli(['batch', 'run', str(source), '-o', str(output), '--fresh'])
    assert fresh.returncode == 2
    covariance = tmp_path / 'bad.npy'
    np.save(covariance, np.eye(2))
    mapping['arms'][1]['overrides'] = {'forecast': {'noise_covariance': str(covariance)}}
    failing = write_yaml(tmp_path / 'failing.yaml', mapping)
    failed = _cli(['batch', 'run', str(failing), '-o', str(tmp_path / 'failed')])
    assert failed.returncode == 3, failed.stderr
    assert json.loads(failed.stdout)['counts']['failed'] == 1
