"""A backend-free controller for deterministic jobs and owned worker processes."""
from __future__ import annotations

from collections import OrderedDict
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
import fnmatch
import json
import logging
from multiprocessing.connection import Client, Listener, wait
import os
from pathlib import Path
import queue
import signal
import subprocess
import sys
import tempfile
import threading
import time

from ..artifacts import write_json, write_yaml
from ..config.checks import ConfigError
from ..config.loading import read_yaml
from ..config.schema import resolve_config
from ..identity import canonical_json, mapping_digest
from ..provenance import capture_provenance
from .jobs import follow_ups, plan_batch
from .processes import (boot_id, group_members, require_process_support, signal_owned,
                        worker_record)
from .spec import BatchExecution
from .results import open_batch
from .state import (BatchConflict, BatchError, BatchIncomplete, EventLog, batch_lock, claim_run_dir,
                    next_session, read_json, read_marker, source_revision, verify_marker, write_failure)

_LOG = logging.getLogger(__name__)
OWNER_ENV = 'HWOSLAPS_BATCH_PROCESS_ID'


@dataclass(frozen=True)
class BatchReport:
    output_dir: Path
    session: int
    counts: Mapping[str, int]
    revisions: Mapping[str, int]
    failures: tuple[Mapping, ...]

    def to_mapping(self):
        return {'output_dir': str(self.output_dir), 'session': self.session, 'counts': dict(self.counts),
                'revisions': dict(self.revisions), 'failures': list(self.failures)}


@dataclass
class _Slot:
    number: int
    device: str
    process: subprocess.Popen
    log: object
    connection: object = None
    worker: dict = field(default_factory=dict)
    keys: tuple[str, ...] = ()
    busy: object = None
    run: str | None = None
    ownership: dict = field(default_factory=dict)


class _WorkerListener:
    """Authenticate and receive startup messages without blocking controller death checks."""

    def __init__(self, address, authkey):
        self.address, self.authkey = address, authkey
        self.listener = Listener(address, family='AF_UNIX', authkey=authkey)
        self.ready = queue.Queue()
        self.thread = threading.Thread(target=self._receive, daemon=True)
        self.thread.start()

    def _receive(self):
        while True:
            connection = None
            try:
                connection = self.listener.accept()
                first = connection.recv()
                if isinstance(first, Mapping) and first.get('type') == 'stop_listener':
                    connection.close()
                    return
                self.ready.put((connection, first))
            except (EOFError, ConnectionResetError) as error:
                if connection is not None:
                    connection.close()
                self.ready.put(error)
            except Exception as error:
                if connection is not None:
                    connection.close()
                self.ready.put(error)
                return

    def close(self):
        if self.thread.is_alive():
            with Client(self.address, family='AF_UNIX', authkey=self.authkey) as wakeup:
                wakeup.send({'type': 'stop_listener'})
            self.thread.join(timeout=5.)
        self.listener.close()
        while True:
            try:
                item = self.ready.get_nowait()
            except queue.Empty:
                break
            if isinstance(item, tuple):
                item[0].close()
        if self.thread.is_alive():
            raise BatchError('worker listener did not finish its owned startup receiver')


def _process_record(slot):
    return dict(slot.ownership)


def _live(record):
    return bool(group_members(record))


def _wait_for_earlier_workers(output, session, events):
    records = []
    for previous in sorted(path for path in (output / 'sessions').iterdir()
                           if path.name.isdecimal() and int(path.name) < session):
        path = previous / 'workers.jsonl'
        if not path.exists():
            continue
        for line in path.read_text().splitlines(keepends=True):
            if not line.endswith('\n'):
                _LOG.warning('ignoring a truncated worker record in %s; completion publication remains authoritative', path)
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise BatchConflict(f'invalid worker record in {path}: {error}') from error
            if not isinstance(record, dict) or set(record) != {'slot', 'pid', 'process_group', 'device', 'start_time', 'boot_id',
                               'session_id', 'identity', 'identity_env', 'uid'}:
                raise BatchConflict(f'invalid worker record in {path}')
            if record['identity_env'] != OWNER_ENV:
                raise BatchConflict(f'invalid ownership metadata name in {path}')
            if isinstance(record['pid'], bool) or not isinstance(record['pid'], int) or record['pid'] < 1:
                raise BatchConflict(f'invalid worker pid in {path}')
            records.append((int(previous.name), record))
    live = [(number, record) for number, record in records if _live(record)]
    if not live:
        return
    _LOG.warning('waiting for workers of sessions %s to finish their current jobs; kill them to resume now; pids=%s groups=%s',
                 sorted({number for number, _ in live}), [record['pid'] for _, record in live],
                 [record['process_group'] for _, record in live])
    events.append({'type': 'waiting_for_earlier_workers', 'sessions': sorted({number for number, _ in live}),
                   'pids': [record['pid'] for _, record in live], 'process_groups': [record['process_group'] for _, record in live]})
    while any(_live(record) for _, record in live):
        time.sleep(5.)
    events.append({'type': 'earlier_workers_exited'})


def _devices(execution):
    if execution.devices == 'cpu':
        return ['cpu'] * execution.workers_per_device
    visible = os.environ.get('CUDA_VISIBLE_DEVICES')
    if visible is not None:
        tokens = [token.strip() for token in visible.split(',')]
        if not all(tokens) or any(index >= len(tokens) for index in execution.devices):
            raise BatchError('execution.devices lies outside the parent CUDA_VISIBLE_DEVICES allocation')
        if len({tokens[index] for index in execution.devices}) != len(execution.devices):
            raise BatchError('execution.devices repeats a physical token in the parent allocation')
    else:
        tokens = None
    return [str(index) if tokens is None else tokens[index]
            for index in execution.devices for _ in range(execution.workers_per_device)]


def _start_slot(number, device, session_dir, session, execution, listener, records):
    generation_boot = boot_id()
    directory = session_dir / 'workers' / f'w{number}'
    directory.mkdir(parents=True, exist_ok=True)
    log = (session_dir / 'workers' / f'w{number}.log').open('ab', buffering=0)
    environment = dict(os.environ)
    identity = os.urandom(32).hex()
    environment[OWNER_ENV] = identity
    for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        environment[name] = str(execution.threads_per_worker)
    environment['CUDA_VISIBLE_DEVICES'] = '' if device == 'cpu' else device
    environment['JAX_PLATFORMS'] = 'cpu' if device == 'cpu' else 'cuda'
    if device != 'cpu':
        environment['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'true'
        environment['XLA_PYTHON_CLIENT_MEM_FRACTION'] = f'{execution.memory_fraction / execution.workers_per_device:.4f}'
    source = str(Path(__file__).resolve().parents[2])
    environment['PYTHONPATH'] = source + (os.pathsep + environment['PYTHONPATH'] if environment.get('PYTHONPATH') else '')
    slot = _Slot(number, device, None, log)
    # Before releasing bootstrap, capture this actual spawned generation. The provisional
    # cookie/session record is used only to clean up a startup failure, never persisted.
    slot.ownership = {'slot': number, 'device': device, 'pid': None, 'process_group': None,
                      'session_id': None, 'identity': identity, 'identity_env': OWNER_ENV, 'uid': os.getuid(),
                      'start_time': None, 'boot_id': generation_boot}
    startup = {'address': listener.address, 'authkey': listener.authkey.hex(), 'slot': number,
               'session': session, 'device': device, 'execution': execution.to_mapping()}
    try:
        slot.process = subprocess.Popen([sys.executable, '-m', 'hwoslaps.batch.worker'], env=environment,
            cwd=directory, stdin=subprocess.PIPE, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        pid = slot.process.pid
        slot.ownership.update(pid=pid, process_group=pid, session_id=pid)
        slot.ownership = worker_record(pid, slot=number, device=device, identity=identity, identity_env=OWNER_ENV)
        records.write(canonical_json(_process_record(slot)) + '\n')
        records.flush()
        os.fsync(records.fileno())
        slot.process.stdin.write((canonical_json(startup) + '\n').encode())
        slot.process.stdin.close()
    except BaseException:
        if slot.process is None:
            log.close()
        else:
            pid = slot.process.pid
            if slot.ownership['pid'] is None:
                slot.ownership.update(pid=pid, process_group=pid, session_id=pid)
            slot.process.stdin.close()
            _close_slots([slot], interrupt=True)
        raise
    return slot


def _signal_group(slot, number):
    signal_owned(slot.ownership, number)


def _close_slots(slots, *, interrupt):
    errors, blocked = [], set()
    def guarded(slot, operation):
        if slot.process.pid in blocked:
            return False
        try:
            return operation()
        except BatchError as error:
            errors.append(error)
            blocked.add(slot.process.pid)
            return False
    def live(slot):
        return guarded(slot, lambda: _live(slot.ownership))
    try:
        for slot in slots:
            if slot.connection is not None and not slot.connection.closed and slot.process.poll() is None:
                try:
                    slot.connection.send({'type': 'stop'})
                except (EOFError, BrokenPipeError, ConnectionResetError):
                    pass
            if interrupt:
                guarded(slot, lambda: _signal_group(slot, signal.SIGTERM))
        deadline = time.monotonic() + (30. if interrupt else 60.)
        while any(live(slot) for slot in slots) and time.monotonic() < deadline:
            for slot in slots:
                slot.process.poll()
            time.sleep(.05)
        remaining = [slot for slot in slots if live(slot)]
        kill_deadline = time.monotonic() + 5.
        while remaining:
            for slot in remaining:
                guarded(slot, lambda: _signal_group(slot, signal.SIGKILL))
                slot.process.poll()
            remaining = [slot for slot in remaining if live(slot)]
            if remaining and time.monotonic() >= kill_deadline:
                errors.append(BatchError('verified owned worker descendants did not drain after pinned SIGKILL; inspect worker records'))
                break
            if remaining:
                time.sleep(.05)
        for slot in slots:
            try:
                slot.process.wait(timeout=1.)
            except subprocess.TimeoutExpired:
                errors.append(BatchError(f'worker {slot.process.pid} remains after safe cleanup; no unverified signal was sent'))
    finally:
        for slot in slots:
            if slot.connection is not None:
                slot.connection.close()
            slot.log.close()
    if errors:
        raise BatchError('worker ownership cleanup failed: ' + '; '.join(str(error) for error in errors)) from errors[0]


def _ready_slot(listener, slots, revision, events):
    while True:
        try:
            received = listener.ready.get(timeout=.1)
        except queue.Empty:
            dead = [slot for slot in slots if slot.connection is None and slot.process.poll() is not None]
            if dead:
                raise BatchError(f'worker {dead[0].number} exited before ready; see its worker log')
            continue
        if isinstance(received, Exception):
            raise BatchError(f'worker startup connection failed: {received}') from received
        connection, message = received
        if not isinstance(message, Mapping) or message.get('type') not in ('ready', 'fatal'):
            connection.close()
            raise BatchError('worker sent an invalid startup message')
        matches = [slot for slot in slots if slot.number == message['slot'] and slot.process.pid == message['pid']]
        if len(matches) != 1 or matches[0].connection is not None:
            connection.close()
            raise BatchError('worker ready does not identify an owned new process')
        slot = matches[0]
        slot.connection = connection
        slot.worker = {key: message[key] for key in ('slot', 'pid', 'device', 'device_kind')}
        if message['type'] == 'fatal':
            raise BatchError(message['message'])
        if message['device'] != slot.device or source_revision(message['provenance']) != revision:
            raise BatchError(f'worker {slot.number} source revision/device differs from this session')
        events.append({'type': 'worker_ready', **slot.worker, 'provenance': message['provenance']})
        return slot


def _persist_plan(output, plan):
    for member in plan.members:
        path = output / 'members' / member.run_name / 'member.json'
        record = member.to_mapping()
        if path.exists():
            if mapping_digest(read_json(path)) != mapping_digest(record):
                raise BatchConflict(f'persisted population member changed: {path}')
        else:
            write_json(path, record)
    for arm in plan.arm_configs:
        path = output / 'members' / arm.member.run_name / arm.name / 'effective_config.yaml'
        if path.exists():
            try:
                stored = resolve_config(read_yaml(path))
            except ConfigError as error:
                raise BatchConflict(f'invalid persisted effective configuration: {path}: {error}') from error
            if stored.digest() != arm.config_digest:
                raise BatchConflict(f'persisted effective configuration changed: {path}')
        else:
            write_yaml(path, arm.config.to_mapping())


def _choose_job(slot, pending, slots):
    for key in reversed(slot.keys):
        for job in pending.values():
            if job.preparation_key == key:
                return job
    groups = OrderedDict()
    for job in pending.values():
        key = job.preparation_key or ('simulate', job.config_digest)
        groups.setdefault(key, []).append(job)
    if not groups:
        return None
    chosen = min(groups, key=lambda key: sum(key in worker.keys for worker in slots))
    return groups[chosen][0]


def run_batch(spec, output_dir, *, resume=True, execution=None, select=None, verify=False, require_single_revision=False):
    require_process_support()
    output = Path(output_dir).expanduser().resolve()
    if not resume and output.exists() and any(output.iterdir()):
        raise BatchConflict(f'fresh batch requires a missing or empty directory: {output}')
    execution = spec.execution if execution is None else execution
    if not isinstance(execution, BatchExecution):
        raise TypeError('execution must be a BatchExecution')
    effective = replace(spec, execution=execution)
    assigned = _devices(execution)
    counts = dict.fromkeys(('completed', 'skipped', 'failed', 'duplicate', 'not_selected', 'orphaned', 'prepared'), 0)
    failures, revisions = [], {}
    with batch_lock(output):
        if not resume and any(path.name != 'batch.lock' for path in output.iterdir()):
            raise BatchConflict(f'fresh batch found existing output after acquiring its lock: {output}')
        session = next_session(output)
        session_dir = output / 'sessions' / str(session)
        events = EventLog(output / 'events.jsonl', session=session)
        slots, listener, temporary = [], None, None
        report_written = False
        main_thread = threading.current_thread() is threading.main_thread()
        previous_term = signal.getsignal(signal.SIGTERM) if main_thread else None
        if main_thread:
            def interrupted(signum, frame):
                raise KeyboardInterrupt
            signal.signal(signal.SIGTERM, interrupted)
        def report():
            return BatchReport(output, session, dict(counts), dict(revisions), tuple(failures))
        def count_revision(marker):
            key = marker['source_revision']
            if require_single_revision and key != revision:
                raise BatchConflict(f'mixed source revisions are forbidden: {marker["job_id"]}: {key} != {revision}')
            revisions[key] = revisions.get(key, 0) + 1
        try:
            provenance = capture_provenance(command=sys.argv)
            revision = source_revision(provenance)
            start = time.time()
            write_yaml(session_dir / 'batch_spec.yaml', effective.to_mapping())
            if resume:
                _wait_for_earlier_workers(output, session, events)
            plan = plan_batch(effective)
            write_json(session_dir / 'session.json', {'session': session, 'started': start, 'provenance': provenance,
                'source': provenance['source'], 'source_revision': revision, 'execution': execution.to_mapping(),
                'spec_digest': plan.spec_digest, 'population_digest': plan.population_digest,
                'file_digests': plan.file_digests})
            _persist_plan(output, plan)
            known, pending, reached = {}, OrderedDict(), set()
            def enqueue(job):
                if job.job_id in known:
                    if known[job.job_id].digest() != job.digest():
                        raise BatchConflict(f'two planned jobs disagree: {job.job_id}')
                    return
                known[job.job_id] = job
                reached.add(job.job_id)
                marker = read_marker(output / job.job_id)
                if marker is not None:
                    if marker['job_digest'] != job.digest():
                        raise BatchConflict(f'completed job digest differs: {job.job_id}')
                    verify_marker(output / job.job_id, marker, digests=verify)
                    count_revision(marker)
                    counts['skipped'] += 1
                    for following in follow_ups(plan, job, output):
                        enqueue(following)
                elif select is not None and not fnmatch.fnmatchcase(job.job_id, select):
                    counts['not_selected'] += 1
                else:
                    pending[job.job_id] = job
            for job in plan.jobs:
                enqueue(job)
            counts['orphaned'] = sum(record.status == 'complete' and record.job_id not in reached
                                     for record in open_batch(output).jobs)
            if pending:
                temporary = tempfile.TemporaryDirectory(prefix='hwb')
                listener = _WorkerListener(str(Path(temporary.name) / 's'), os.urandom(32))
                with (session_dir / 'workers.jsonl').open('a', encoding='utf-8') as records:
                    for number, device in enumerate(assigned):
                        slots.append(_start_slot(number, device, session_dir, session, execution, listener, records))
                    for _ in slots:
                        _ready_slot(listener, slots, revision, events)
                    stop_error = None
                    while pending or any(slot.busy is not None for slot in slots):
                        if stop_error is None:
                            for slot in slots:
                                if slot.busy is None and slot.process.poll() is None:
                                    job = _choose_job(slot, pending, slots)
                                    if job is not None:
                                        del pending[job.job_id]
                                        slot.busy, slot.run = job, None
                                        slot.connection.send({'type': 'job', 'payload': job.payload(output, session, execution).to_mapping()})
                        if not any(slot.busy is not None for slot in slots) and stop_error is not None:
                            break
                        active = [slot.connection for slot in slots if slot.connection is not None]
                        readable = wait(active, timeout=5.)
                        for slot in list(slots):
                            if slot.connection not in readable and slot.process.poll() is None:
                                continue
                            message = None
                            if slot.connection in readable:
                                try:
                                    message = slot.connection.recv()
                                except (EOFError, ConnectionResetError):
                                    pass
                            if message is not None:
                                kind = message['type']
                                job = slot.busy
                                if job is None or message['job_id'] != job.job_id:
                                    raise BatchError(f'worker {slot.number} reported a job it does not own')
                                events.append({**message, 'slot': slot.number})
                                if kind == 'started':
                                    slot.run = message['run']
                                    continue
                                counts['prepared'] += message.get('prepared', 0)
                                if kind in ('complete', 'duplicate'):
                                    marker = read_marker(output / job.job_id)
                                    if marker is None or marker['job_digest'] != job.digest():
                                        raise BatchConflict(f'winning completion differs: {job.job_id}')
                                    verify_marker(output / job.job_id, marker, digests=verify)
                                    count_revision(marker)
                                    counts['completed' if kind == 'complete' else 'duplicate'] += 1
                                    slot.keys = tuple(message['cache_keys'])
                                    if stop_error is None:
                                        for following in follow_ups(plan, job, output):
                                            enqueue(following)
                                elif kind == 'failed':
                                    failure = message['failure']
                                    counts['failed'] += 1
                                    failures.append({'job_id': job.job_id, 'kind': job.kind, 'error_type': failure['error_type'],
                                        'message': failure['message'].splitlines()[0] if failure['message'] else '',
                                        'log': str(output / job.job_id / message['run'] / 'job.log')})
                                    if failure['error_type'] == 'BatchConflict':
                                        stop_error = BatchConflict(f'job conflict: {job.job_id}: {failure["message"]}')
                                else:
                                    raise BatchError(f'unknown worker event {kind!r}')
                                slot.busy, slot.run = None, None
                                if kind != 'failed':
                                    continue
                            if slot.busy is not None:
                                job = slot.busy
                                run_dir = claim_run_dir(output / job.job_id) if slot.run is None else output / job.job_id / slot.run
                                code = slot.process.wait()
                                failure = {'schema': 1, 'job_id': job.job_id, 'job_digest': job.digest(), 'session': session,
                                    'error_type': 'WorkerExited', 'message': f'worker exited with code {code}',
                                    'traceback': '', 'worker': slot.worker, 'exit_code': code}
                                write_failure(run_dir, failure)
                                counts['failed'] += 1
                                failures.append({'job_id': job.job_id, 'kind': job.kind, 'error_type': 'WorkerExited',
                                    'message': failure['message'], 'log': str(run_dir / 'job.log')})
                                events.append({'type': 'failed', 'job_id': job.job_id, 'failure': failure})
                                slot.busy = None
                            events.append({'type': 'worker_exit', 'slot': slot.number, 'pid': slot.process.pid})
                            slots.remove(slot)
                            _close_slots([slot], interrupt=False)
                            if stop_error is None and (pending or any(worker.busy is not None for worker in slots)):
                                replacement = _start_slot(slot.number, slot.device, session_dir, session, execution, listener, records)
                                slots.append(replacement)
                                try:
                                    _ready_slot(listener, slots, revision, events)
                                except BatchError as error:
                                    stop_error = error
                    closing = list(slots)
                    slots.clear()
                    _close_slots(closing, interrupt=False)
                    if stop_error is not None:
                        current = report()
                        write_json(session_dir / 'report.json', current.to_mapping())
                        report_written = True
                        raise stop_error
            current = report()
            write_json(session_dir / 'report.json', current.to_mapping())
            report_written = True
        except BaseException as original:
            closing = list(slots)
            slots.clear()
            cleanup_error = None
            try:
                _close_slots(closing, interrupt=True)
            except BatchError as error:
                cleanup_error = error
            if not report_written:
                write_json(session_dir / 'report.json', {**report().to_mapping(), 'interrupted': True,
                    'cleanup_error': None if cleanup_error is None else str(cleanup_error)})
            if cleanup_error is not None:
                raise cleanup_error from original
            raise
        finally:
            if listener is not None:
                listener.close()
            if temporary is not None:
                temporary.cleanup()
            events.close()
            if main_thread:
                signal.signal(signal.SIGTERM, previous_term)
    if current.counts['failed']:
        raise BatchIncomplete(current)
    return current
