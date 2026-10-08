"""Actual owned subprocess survivors and foreign-identity refusal."""
import json
import errno
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import pytest

from hwoslaps.batch.processes import boot_id, group_members, open_pidfd, pidfd_exited, signal_owned, worker_record
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
        with (tmp_path / (name + '.ownership.json')).open('w', encoding='utf-8') as metadata:
            metadata.write(json.dumps(record) + '\n')
            metadata.flush()
            os.fsync(metadata.fileno())
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


def test_real_pidfd_descriptor_is_pinned_and_noninheritable():
    from hwoslaps.batch.state import BatchError
    descriptor = open_pidfd(os.getpid())
    try:
        assert not os.get_inheritable(descriptor)
        signal.pidfd_send_signal(descriptor, 0)
        import select
        poller = select.poll()
        poller.register(descriptor, select.POLLIN)
        assert poller.poll(0) == [], 'the current live process must not appear exited'
        assert pidfd_exited(descriptor) is False
    finally:
        os.close(descriptor)
    with pytest.raises(BatchError, match='error or invalid descriptor'):
        pidfd_exited(descriptor)
    with pytest.raises(OSError) as invalid:
        open_pidfd(-1)
    assert invalid.value.errno == errno.EINVAL
    import ctypes
    integer_bits = ctypes.sizeof(ctypes.c_int) * 8
    # This would silently wrap to our real live PID at the libc boundary.
    with pytest.raises(OverflowError):
        open_pidfd(os.getpid() + (1 << integer_bits))
    with pytest.raises(OverflowError):
        open_pidfd(-(1 << integer_bits))
    with pytest.raises(TypeError):
        open_pidfd(1.5)


def test_reused_or_foreign_identity_is_never_signaled(tmp_path):
    from hwoslaps.batch.state import BatchError
    process, record, log = _start('import time\ntime.sleep(60)\n', tmp_path, 'foreign')
    descriptor = None
    try:
        descriptor = open_pidfd(process.pid)
        with pytest.raises(BatchError, match='cookie_matches=False'):
            worker_record(process.pid, slot=0, device='cpu', identity='0' * 64, identity_env=OWNER_ENV)
        mismatches = ({**record, 'identity': '0' * 64}, {**record, 'start_time': record['start_time'] + 1},
                      {**record, 'boot_id': 'another-boot'})
        for wrong in mismatches:
            assert group_members(wrong) == ()
            signal_owned(wrong, signal.SIGKILL)
            assert pidfd_exited(descriptor, 50) is False, 'a real foreign identity must remain alive after an unverified signal request'
            assert process.poll() is None, 'a real foreign identity must remain alive'
        assert [member['pid'] for member in group_members(record)] == [process.pid]
    finally:
        try:
            _close(process, record, log)
        finally:
            if descriptor is not None:
                os.close(descriptor)


def test_missing_bootstrap_cookie_refuses_and_actual_worker_eof_skips_science(tmp_path):
    import hashlib
    import hwoslaps.batch.worker as worker
    from hwoslaps.batch.state import BatchError
    source = Path(worker.__file__).resolve()
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    binding, finished = tmp_path / 'binding.json', tmp_path / 'eof.json'
    script = tmp_path / 'actual-worker.py'
    script.write_text(f'''
if __name__ == '__main__':
    import hashlib, json, sys
    from pathlib import Path
    import hwoslaps.batch.worker as worker
    actual = Path(worker.__file__).resolve()
    assert str(actual) == {str(source)!r}
    assert hashlib.sha256(actual.read_bytes()).hexdigest() == {digest!r}
    Path({str(binding)!r}).write_text(json.dumps({{'path': str(actual), 'sha256': {digest!r}}}))
    code = worker.main()
    Path({str(finished)!r}).write_text(json.dumps({{'code': code,
        'backend_imported': 'hwoslaps.inference.backend' in sys.modules, 'jax_imported': 'jax' in sys.modules}}))
    raise SystemExit(code)
''')
    environment = dict(os.environ)
    environment.pop(OWNER_ENV, None)
    environment['PYTHONPATH'] = str(source.parents[2])
    log = (tmp_path / 'cookie-free.log').open('wb')
    process = descriptor = None
    try:
        process = subprocess.Popen([sys.executable, str(script)], env=environment, stdin=subprocess.PIPE,
                                   stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        descriptor = open_pidfd(process.pid)
        _wait(binding.exists, timeout=10.)
        with pytest.raises(BatchError, match='cookie_present=False'):
            worker_record(process.pid, slot=0, device='cpu', identity='a' * 64, identity_env=OWNER_ENV)
        assert process.poll() is None and pidfd_exited(descriptor) is False
        process.stdin.close()
        assert process.wait(timeout=10.) == 0
        assert json.loads(finished.read_text()) == {'code': 0, 'backend_imported': False, 'jax_imported': False}
        assert json.loads(binding.read_text()) == {'path': str(source), 'sha256': digest}
    finally:
        try:
            if process is not None:
                process.stdin.close()
                if descriptor is not None:
                    try:
                        signal.pidfd_send_signal(descriptor, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                process.wait(timeout=10.)
        finally:
            if descriptor is not None:
                os.close(descriptor)
            log.close()


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
        pinned = open_pidfd(process.pid)
        descriptors.append(pinned)
        assert max(descriptors) >= 1024
        assert group_members(record)
        signal_owned(record, signal.SIGTERM)
        process.wait(timeout=10.)
        assert pidfd_exited(pinned) is True, 'actual reaped child must report a kernel exit event'
        assert not group_members(record)
    finally:
        if process is not None:
            _close(process, record, log)
        for descriptor in descriptors:
            os.close(descriptor)
        resource.setrlimit(resource.RLIMIT_NOFILE, limits)


def test_cleanup_drains_clear_groups_and_preserves_mixed_cookie_group(tiny_batch_spec, tmp_path, monkeypatch):
    import hashlib
    from multiprocessing.connection import Listener
    import threading
    from hwoslaps.batch import run_batch
    import hwoslaps.batch.worker as worker
    import hwoslaps.fisher.api as api
    from hwoslaps.batch.state import BatchError
    root = tmp_path / 'public-mixed'
    foreign_cookie = os.urandom(32).hex()
    source = {name: {'path': str(Path(module.__file__).resolve()),
                     'sha256': hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest()}
              for name, module in [('worker', worker), ('forecast', api)]}
    entry = tmp_path / 'actual-mixed-worker.py'
    entry.write_text(f'''
if __name__ == '__main__':
    import hashlib, json, os, subprocess, sys
    from pathlib import Path
    import hwoslaps.batch.worker as worker
    actual_worker = worker.run_job
    assert str(Path(worker.__file__).resolve()) == {source['worker']['path']!r}
    assert hashlib.sha256(Path(worker.__file__).read_bytes()).hexdigest() == {source['worker']['sha256']!r}
    def actual_job(payload, cache, session, **keywords):
        import hwoslaps.fisher.api as api
        assert str(Path(api.__file__).resolve()) == {source['forecast']['path']!r}
        assert hashlib.sha256(Path(api.__file__).read_bytes()).hexdigest() == {source['forecast']['sha256']!r}
        actual_forecast = api.forecast
        def observed_forecast(prepared, **arguments):
            slot = keywords['worker']['slot']
            record = {{'slot': slot, 'pid': os.getpid(), 'config_digest': payload.config_digest,
                       'source': {source!r}}}
            if slot == 0:
                environment = {{**os.environ, {OWNER_ENV!r}: {foreign_cookie!r}}}
                child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(120)'], env=environment)
                fields = Path(f'/proc/{{child.pid}}/stat').read_text().rsplit(')', 1)[1].split()
                record['foreign'] = {{'pid': child.pid, 'start_time': int(fields[19]),
                                     'process_group': int(fields[2]), 'session_id': int(fields[3])}}
            destination = Path({str(tmp_path)!r}) / f'forecast-entered-{{slot}}.json'
            partial = destination.with_suffix('.partial')
            with partial.open('x') as stream:
                stream.write(json.dumps(record) + '\\n')
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(partial, destination)
            return actual_forecast(prepared, **arguments)
        api.forecast = observed_forecast
        try:
            return actual_worker(payload, cache, session, **keywords)
        finally:
            api.forecast = actual_forecast
    worker.run_job = actual_job
    raise SystemExit(worker.main())
''')
    actual_popen, actual_accept, actual_listener_close = subprocess.Popen, Listener.accept, Listener.close
    processes, logs, connections, descriptors = [], [], [], []
    listener_addresses, closed_listeners = {}, []
    def observed_popen(command, *arguments, **keywords):
        is_worker = isinstance(command, (list, tuple)) and list(command) == [sys.executable, '-m', 'hwoslaps.batch.worker']
        process = actual_popen([sys.executable, str(entry)] if is_worker else command, *arguments, **keywords)
        if is_worker:
            processes.append(process)
            logs.append(keywords['stdout'])
            descriptors.append(open_pidfd(process.pid))
        return process
    def observed_accept(listener):
        connection = actual_accept(listener)
        connections.append(connection)
        return connection
    def observed_listener_close(listener):
        if listener not in listener_addresses:
            listener_addresses[listener] = listener.address
        result = actual_listener_close(listener)
        closed_listeners.append(listener)
        return result
    monkeypatch.setattr(subprocess, 'Popen', observed_popen)
    monkeypatch.setattr(Listener, 'accept', observed_accept)
    monkeypatch.setattr(Listener, 'close', observed_listener_close)
    spec = tiny_batch_spec(population={'count': 3}, execution={'workers_per_device': 3},
        forecast_positions={'kind': 'grid', 'spacing_arcsec': .01, 'half_width_arcsec': .6})
    cancel, synchronized, synchronization_errors = threading.Event(), [], []
    controller = open_pidfd(os.getpid())
    previous_term = signal.getsignal(signal.SIGTERM)
    foreign_descriptor = None
    def pin_foreign(observation):
        nonlocal foreign_descriptor
        if foreign_descriptor is not None:
            return
        foreign = observation['foreign']
        descriptor = open_pidfd(foreign['pid'])
        try:
            fields = Path(f"/proc/{foreign['pid']}/stat").read_text().rsplit(')', 1)[1].split()
            assert int(fields[19]) == foreign['start_time']
            assert int(fields[2]) == int(fields[3]) == observation['pid']
            assert (OWNER_ENV + '=' + foreign_cookie).encode() in Path(f"/proc/{foreign['pid']}/environ").read_bytes().split(b'\0')
            assert not pidfd_exited(descriptor)
        except BaseException:
            os.close(descriptor)
            raise
        foreign_descriptor = descriptor
        descriptors.append(descriptor)
    def interrupt_actual_forecasts():
        try:
            deadline = time.monotonic() + 90.
            paths = [tmp_path / f'forecast-entered-{slot}.json' for slot in range(3)]
            while not all(path.exists() for path in paths):
                if paths[0].exists():
                    pin_foreign(json.loads(paths[0].read_text()))
                if cancel.wait(.01):
                    return
                if time.monotonic() >= deadline:
                    raise AssertionError('three actual forecast boundaries did not publish within90s')
            observations = [json.loads(path.read_text()) for path in paths]
            assert all(record['source'] == source for record in observations)
            assert len({record['config_digest'] for record in observations}) == 3
            pin_foreign(observations[0])
            assert [record['pid'] for record in observations] == [process.pid for process in processes]
            assert len(processes) == 3 and all(process.poll() is None for process in processes)
            synchronized.extend(observations)
            signal.pidfd_send_signal(controller, signal.SIGTERM)
        except BaseException as error:
            synchronization_errors.append(error)
    interrupter = threading.Thread(target=interrupt_actual_forecasts)
    interrupter.start()
    try:
        with pytest.raises(BatchError, match='worker ownership cleanup failed') as refused:
            run_batch(spec, root, resume=False)
        cancel.set()
        interrupter.join(timeout=5.)
        assert not interrupter.is_alive() and not synchronization_errors
        assert len(synchronized) == 3
        assert isinstance(refused.value.__cause__, KeyboardInterrupt)
        assert 'ambiguous/foreign pids=' in str(refused.value)
        assert str(synchronized[0]['foreign']['pid']) in str(refused.value)
        assert not pidfd_exited(foreign_descriptor, 50), 'the actual foreign descendant must remain unsignaled'
        assert not pidfd_exited(descriptors[0], 50), 'the actual mixed worker must remain unsignaled'
        assert all(process.poll() == -signal.SIGTERM for process in processes[1:]), 'clear actual workers must drain despite a mixed group'
        assert len(connections) >= len(processes) and all(connection.closed for connection in connections)
        assert all(log.closed and process.stdin.closed for process, log in zip(processes, logs, strict=True))
        assert listener_addresses and all(listener in closed_listeners for listener in listener_addresses)
        assert all(not Path(address).exists() for address in listener_addresses.values())
        assert signal.getsignal(signal.SIGTERM) == previous_term
        report = json.loads((root / 'sessions/1/report.json').read_text())
        assert report['interrupted'] and 'ambiguous/foreign pids=' in report['cleanup_error']
    finally:
        cancel.set()
        interrupter.join(timeout=5.)
        os.close(controller)
        foreign_cleanup_error = None
        if foreign_descriptor is None and (tmp_path / 'forecast-entered-0.json').exists():
            try:
                pin_foreign(json.loads((tmp_path / 'forecast-entered-0.json').read_text()))
            except BaseException as error:
                foreign_cleanup_error = error
        # Intentionally mixed fixtures require independently retained kernel handles;
        # the scanner under test never certifies their final cleanup.
        for descriptor in reversed(descriptors):
            try:
                signal.pidfd_send_signal(descriptor, signal.SIGKILL)
            except ProcessLookupError:
                pass
        for process, log in zip(processes, logs, strict=True):
            process.wait(timeout=10.)
            process.stdin.close()
            log.close()
        for connection in connections:
            connection.close()
        for descriptor in descriptors:
            _wait(lambda: pidfd_exited(descriptor), timeout=10.)
            os.close(descriptor)
        if foreign_cleanup_error is not None:
            raise foreign_cleanup_error


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
        descriptor = open_pidfd(process.pid)
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
