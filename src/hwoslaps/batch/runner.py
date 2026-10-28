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
from ..identity import canonical_json
from ..provenance import capture_provenance
from .jobs import FollowupFailure, follow_ups
from .frozen import persist_plan, restored_plan
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
    dispatched_at: float | None = None


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
    persist_plan(output, plan)


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


def _terminal_outcome(job, output, *, run, session, worker, synthetic=None):
    """Reconcile published authority identically for IPC, EOF, death and timeout.

    Transport callers drain the owned generation before using ``synthetic``.
    The assigned run identifies this attempt even if no started event arrived.
    """
    job_dir = output / job.job_id
    def completion():
        marker = read_marker(job_dir)
        if marker is None:
            return None
        if (marker.get('job_id') != job.job_id or marker.get('job_digest') != job.digest()
                or marker.get('kind') != job.kind or not isinstance(marker.get('source_revision'), str)):
            raise BatchConflict(f'winning completion differs: {job.job_id}')
        verify_marker(job_dir, marker, digests=True)
        if marker['run'] == run and (marker.get('session') != session or marker.get('worker') != worker):
            raise BatchConflict(f'completion differs from its active attempt: {job.job_id}')
        return marker
    marker = completion()
    if marker is not None:
        return 'complete', marker
    if run is None:
        raise BatchConflict(f'active attempt has no assigned run: {job.job_id}')
    run_dir = job_dir / run
    if not run_dir.is_dir() or run_dir.is_symlink() or not run_dir.resolve().is_relative_to(job_dir.resolve()):
        raise BatchConflict(f'active attempt directory changed: {run_dir}')
    path = run_dir / 'failure.json'
    def failure_record():
        try:
            failure = read_json(path)
        except FileNotFoundError:
            return None
        if (run_dir.is_symlink() or path.is_symlink() or failure.get('schema') != 1
                or failure.get('job_id') != job.job_id or failure.get('job_digest') != job.digest()
                or type(failure.get('session')) is not int or failure['session'] != session
                or failure.get('worker') != worker
                or any(not isinstance(failure.get(key), str) for key in ('error_type', 'message', 'traceback'))
                or not failure['error_type']
                or failure.get('exit_code') is not None and type(failure['exit_code']) is not int):
            raise BatchConflict(f'invalid active-attempt failure: {path}')
        return failure
    failure = failure_record()
    if failure is not None:
        return 'failed', failure
    if synthetic is None:
        raise BatchConflict(f'worker terminal event has no published outcome: {job.job_id}')
    failure = {'schema': 1, 'job_id': job.job_id, 'job_digest': job.digest(), 'session': session,
               'traceback': '', 'worker': worker, **synthetic}
    try:
        write_failure(run_dir, failure)
    except FileExistsError:
        # Another atomic publisher won: validate its actual record, never merely
        # suppress the collision or consume an older attempt's failure.
        return _terminal_outcome(job, output, run=run, session=session, worker=worker)
    marker = completion()
    if marker is not None:
        if failure_record() != failure:
            raise BatchConflict(f'synthetic failure changed during completion race: {path}')
        path.unlink()
        return 'complete', marker
    return 'failed', failure


def run_batch(spec, output_dir, *, resume=True, execution=None, select=None, verify=False, require_single_revision=False):
    """Execute selected missing jobs with owned workers and completion markers.

    Parameters
    ----------
    spec : BatchSpec
        Parsed inputs to plan and execute.
    output_dir : path-like
        Batch storage directory; one controller holds its exclusive lock.
    resume : bool
        Reuse matching completed jobs and wait for recorded earlier workers.
        False requires missing or empty storage.
    execution : BatchExecution or None
        Override runtime placement and controls, or None for the spec's settings.
    select : str or None
        Case-sensitive glob over job IDs for missing jobs; None selects all.
    verify : bool
        Recheck completed artifact SHA-256 values, in addition to structural and
        size checks performed when inspecting completion markers.
    require_single_revision : bool
        Raise an error if completed jobs come from another source revision.

    Returns
    -------
    BatchReport
        Session counts, participating source revisions and job failures.

    Raises
    ------
    BatchIncomplete
        Jobs failed; the exception carries the completed session report.
    BatchError
        Runtime, ownership or storage prevents safe completion. Subclasses
        distinguish conflicting storage and an already-held controller lock.

    Notes
    -----
    Execution requires Linux /proc and pidfd support. Only verified owned worker
    identities are signaled during cleanup; scientific completion is authorized by
    validated completion markers, not directory presence.
    """
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
            if resume:
                _wait_for_earlier_workers(output, session, events)
            plan = restored_plan(output, effective)
            write_yaml(session_dir / 'batch_spec.yaml', effective.to_mapping())
            write_json(session_dir / 'session.json', {'session': session, 'started': start, 'provenance': provenance,
                'source': provenance['source'], 'source_revision': revision, 'execution': execution.to_mapping(),
                'spec_digest': plan.spec_digest, 'population_digest': plan.population_digest,
                'file_digests': plan.file_digests})
            _persist_plan(output, plan)
            known, pending, reached = {}, OrderedDict(), set()
            def enqueue(job):
                if isinstance(job, FollowupFailure):
                    if job.job_id in known:
                        return
                    known[job.job_id] = job
                    reached.add(job.job_id)
                    failure_path = output / job.job_id / 'failure.json'
                    if failure_path.exists():
                        if read_json(failure_path) != job.to_mapping():
                            raise BatchConflict(f'terminal follow-up record changed: {job.job_id}')
                    else:
                        write_json(failure_path, job.to_mapping())
                    counts['failed'] += 1
                    failures.append({'job_id': job.job_id, 'kind': job.kind, 'error_type': 'NoUsableTrial',
                                     'message': job.reason, 'log': str(failure_path),
                                     'source_job_id': job.source_job_id, 'family': job.family,
                                     'member_index': job.member_index, 'arm': job.arm, 'mass_msun': job.mass_msun})
                    events.append({'type': 'followup_failed', **job.to_mapping()})
                    return
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
                    finalized = {}
                    def finalize(slot, message=None, synthetic=None):
                        nonlocal stop_error
                        job, run = slot.busy, slot.run
                        outcome, record = _terminal_outcome(job, output, run=run, session=session,
                                                            worker=slot.worker, synthetic=synthetic)
                        if message is not None and message['type'] == 'failed' and outcome == 'failed' and record != message.get('failure'):
                            raise BatchConflict(f'failed event differs from its published attempt: {job.job_id}')
                        prepared = message.get('prepared', 0) if message is not None else (
                            record.get('prepared', 0) if record.get('worker') == slot.worker and record.get('session') == session else 0)
                        if type(prepared) is not int or prepared < 0:
                            raise BatchConflict(f'invalid preparation count: {job.job_id}')
                        counts['prepared'] += prepared
                        if outcome == 'complete':
                            count_revision(record)
                            counts['duplicate' if record['run'] != run else 'completed'] += 1
                            if message is not None:
                                slot.keys = tuple(message.get('cache_keys', ()))
                            if stop_error is None:
                                for following in follow_ups(plan, job, output):
                                    enqueue(following)
                        else:
                            counts['failed'] += 1
                            failures.append({'job_id': job.job_id, 'kind': job.kind, 'error_type': record['error_type'],
                                'message': record['message'].splitlines()[0] if record['message'] else '',
                                'log': str(output / job.job_id / run / 'job.log')})
                            if record['error_type'] == 'BatchConflict':
                                stop_error = BatchConflict(f'job conflict: {job.job_id}: {record["message"]}')
                        events.append({'type': 'reconciled', 'job_id': job.job_id, 'outcome': outcome,
                                       'run': run, 'record': record})
                        finalized[job.job_id] = (slot.process.pid, run)
                        slot.busy, slot.run, slot.dispatched_at = None, None, None
                    while pending or any(slot.busy is not None for slot in slots):
                        if stop_error is None:
                            for slot in slots:
                                if slot.busy is None and slot.process.poll() is None:
                                    job = _choose_job(slot, pending, slots)
                                    if job is not None:
                                        del pending[job.job_id]
                                        slot.busy = job
                                        slot.run = claim_run_dir(output / job.job_id).name
                                        slot.dispatched_at = time.monotonic()
                                        payload = replace(job.payload(output, session, execution), run=slot.run)
                                        try:
                                            slot.connection.send({'type': 'job', 'payload': payload.to_mapping()})
                                        except (BrokenPipeError, EOFError, ConnectionResetError):
                                            # Reconcile this assigned attempt after owned cleanup below.
                                            pass
                        if not any(slot.busy is not None for slot in slots) and stop_error is not None:
                            break
                        active = [slot.connection for slot in slots if slot.connection is not None]
                        wait(active, timeout=5.)
                        for slot in list(slots):
                            exited = False
                            # Drain queued notifications before evaluating the deadline. A
                            # terminal notification can arrive alongside EOF or timeout.
                            while slot.connection is not None and slot.connection.poll():
                                try:
                                    message = slot.connection.recv()
                                except (EOFError, ConnectionResetError):
                                    exited = True
                                    break
                                if not isinstance(message, Mapping) or message.get('type') not in ('started', 'complete', 'duplicate', 'failed'):
                                    raise BatchError('worker sent an invalid job event')
                                identifier = message.get('job_id')
                                if identifier in finalized:
                                    if finalized[identifier] != (slot.process.pid, message.get('run')):
                                        raise BatchConflict('late worker event differs from its finalized attempt')
                                    events.append({'type': 'duplicate_event', 'job_id': identifier, 'slot': slot.number})
                                    continue
                                job = slot.busy
                                if job is None or identifier != job.job_id or message.get('run') != slot.run:
                                    raise BatchError(f'worker {slot.number} reported an attempt it does not own')
                                events.append({**message, 'slot': slot.number})
                                if message['type'] == 'started':
                                    if any(message.get(key) != value for key, value in slot.worker.items()):
                                        raise BatchConflict('started event differs from owned worker generation')
                                    continue
                                finalize(slot, message=message)
                                if message['type'] == 'failed':
                                    exited = True
                                    break
                            timed_out = (slot.busy is not None and slot.dispatched_at is not None
                                and execution.job_timeout_seconds is not None
                                and time.monotonic() - slot.dispatched_at >= execution.job_timeout_seconds)
                            if not exited and not timed_out and slot.process.poll() is None:
                                continue
                            # No live owned publisher may remain before absence is used
                            # to synthesize a transport failure. Preserve pinned cleanup.
                            _close_slots([slot], interrupt=True)
                            slots.remove(slot)
                            if slot.busy is not None:
                                code = slot.process.returncode
                                transport = {'error_type': 'JobTimeout' if timed_out else 'WorkerExited',
                                    'message': f'job exceeded {execution.job_timeout_seconds} seconds' if timed_out
                                               else f'worker exited with code {code}', 'exit_code': code}
                                events.append({'type': 'transport_failure', 'job_id': slot.busy.job_id,
                                               'slot': slot.number, **transport})
                                finalize(slot, synthetic=transport)
                            events.append({'type': 'worker_exit', 'slot': slot.number, 'pid': slot.process.pid})
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
