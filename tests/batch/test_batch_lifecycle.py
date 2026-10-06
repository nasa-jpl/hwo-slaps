"""Actual owned subprocess survivors and foreign-identity refusal."""
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import pytest

from hwoslaps.batch.processes import boot_id, group_members, signal_owned, worker_record
from hwoslaps.batch.runner import OWNER_ENV

pytestmark = pytest.mark.backend


def _wait(predicate, timeout=90.):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        value = predicate()
        if value:
            return value
        time.sleep(.02)
    raise AssertionError('actual process lifecycle non-vacuity/condition did not complete before bounded deadline')


def _start(script, tmp_path, name):
    path = tmp_path / (name + '.py')
    path.write_text('import sys\nif __name__ == "__main__":\n' +
                    '    if sys.stdin.readline() != "go\\n": raise SystemExit(2)\n' +
                    ''.join('    ' + line for line in script.splitlines(keepends=True)))
    environment = dict(os.environ)
    identity = os.urandom(32).hex()
    environment[OWNER_ENV] = identity
    source = Path(__file__).resolve().parents[2] / 'src'
    environment['PYTHONPATH'] = str(source) + (os.pathsep + environment['PYTHONPATH'] if environment.get('PYTHONPATH') else '')
    generation_boot = boot_id()
    log = (tmp_path / (name + '.log')).open('wb')
    process = None
    record = None
    try:
        process = subprocess.Popen([sys.executable, str(path)], env=environment, stdin=subprocess.PIPE,
            stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        record = {'slot': 0, 'device': 'cpu', 'pid': process.pid, 'process_group': process.pid,
                  'session_id': process.pid, 'identity': identity, 'identity_env': OWNER_ENV,
                  'uid': os.getuid(), 'start_time': None, 'boot_id': generation_boot}
        record = worker_record(process.pid, slot=0, device='cpu', identity=identity, identity_env=OWNER_ENV)
        process.stdin.write(b'go\n')
        process.stdin.close()
        return process, record, log
    except BaseException:
        if process is None:
            log.close()
        else:
            process.stdin.close()
            _close(process, record, log)
        raise


def _close(process, record, log):
    try:
        signal_owned(record, signal.SIGKILL)
        process.wait(timeout=10.)
        _wait(lambda: not group_members(record), timeout=10.)
    finally:
        log.close()


def test_reused_or_foreign_identity_is_never_signaled(tmp_path):
    process, record, log = _start('import time\ntime.sleep(60)\n', tmp_path, 'foreign')
    try:
        mismatches = ({**record, 'identity': '0' * 64}, {**record, 'start_time': record['start_time'] + 1},
                      {**record, 'boot_id': 'another-boot'})
        for wrong in mismatches:
            assert group_members(wrong) == ()
            signal_owned(wrong, signal.SIGKILL)
            assert process.poll() is None, 'a real foreign identity must remain alive'
        assert [member['pid'] for member in group_members(record)] == [process.pid]
    finally:
        _close(process, record, log)


def test_pidfd_ownership_works_above_the_select_descriptor_limit(tmp_path):
    import resource
    limits = resource.getrlimit(resource.RLIMIT_NOFILE)
    if limits[0] < 1200:
        if limits[1] != resource.RLIM_INFINITY and limits[1] < 1200:
            pytest.fail('high-descriptor owner control needs a1200-descriptor hard limit')
        resource.setrlimit(resource.RLIMIT_NOFILE, (1200, limits[1]))
    descriptors = []
    process = record = log = None
    try:
        for _ in range(1100):
            descriptors.append(os.open(os.devnull, os.O_RDONLY))
        process, record, log = _start('import time\ntime.sleep(60)\n', tmp_path, 'high-fd')
        assert max(descriptors) >= 1024
        assert group_members(record)
        signal_owned(record, signal.SIGTERM)
        process.wait(timeout=10.)
        assert not group_members(record)
    finally:
        if process is not None:
            _close(process, record, log)
        for descriptor in descriptors:
            os.close(descriptor)
        resource.setrlimit(resource.RLIMIT_NOFILE, limits)


def test_real_training_descendants_remain_owned_after_leader_sigkill(tmp_path):
    ready = tmp_path / 'training-ready'
    script = f'''
import json
import multiprocessing
import numpy as np
from pathlib import Path
from hwoslaps.inference.backend import BackendSession
from nautilus.neural import NeuralNetworkEmulator
rng = np.random.default_rng(21)
with BackendSession(training_workers=2) as session:
    with session.search_scope(retain_search_internal=False, number_of_cores=1):
        small = rng.normal(size=(50, 3))
        NeuralNetworkEmulator.train(small, np.sin(small[:, 0]) + small[:, 1]**2, n_networks=2)
        Path({str(ready)!r}).write_text(json.dumps([child.pid for child in multiprocessing.active_children()]))
        points = rng.normal(size=(20000, 8))
        NeuralNetworkEmulator.train(points, np.sin(points[:, 0]) + points[:, 1]**2, n_networks=2)
'''
    process, record, log = _start(script, tmp_path, 'training')
    try:
        _wait(ready.exists)
        training_pids = set(json.loads(ready.read_text()))
        assert len(training_pids) == 2, 'the actual BackendSession must expose its two training workers'
        def training_members():
            members = tuple(member for member in group_members(record) if member['pid'] in training_pids)
            return members if {member['pid'] for member in members} == training_pids else None
        children = _wait(training_members)
        before = {member['pid']: member for member in children}
        def cpu_ticks(pid):
            fields = Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()
            return int(fields[11]) + int(fields[12])
        initial = {pid: cpu_ticks(pid) for pid in before}
        _wait(lambda: any(cpu_ticks(pid) > ticks for pid, ticks in initial.items()))
        descriptor = os.pidfd_open(process.pid)
        try:
            signal.pidfd_send_signal(descriptor, signal.SIGKILL)
        finally:
            os.close(descriptor)
        process.wait(timeout=10.)
        survivors = tuple(member for member in group_members(record) if member['pid'] in training_pids)
        assert survivors, 'real training must outlast leader death; enlarge actual work if this control is vacuous'
        assert all(member['pid'] != process.pid for member in survivors)
        assert any(member['pid'] in before and member['start_time'] == before[member['pid']]['start_time']
                   for member in survivors)
        signal_owned(record, signal.SIGKILL)
        _wait(lambda: not group_members(record), timeout=10.)
    finally:
        _close(process, record, log)
