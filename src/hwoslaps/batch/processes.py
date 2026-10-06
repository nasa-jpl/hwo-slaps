"""Linux worker ownership: discover survivors and signal pinned verified identities.

The internal child-environment cookie is inherited by the actual worker's spawned
processes. It is lifecycle metadata, never configuration or scientific seed input.
No process environment is logged and no numeric PID/PGID is used for signaling.
"""
from __future__ import annotations

from contextlib import contextmanager
import os
from pathlib import Path
import select
import signal
import sys

from .state import BatchError


def open_pidfd(pid):
    """Open a real Linux pidfd, including Python builds lacking os.pidfd_open."""
    if hasattr(os, 'pidfd_open'):
        return os.pidfd_open(pid)
    if sys.platform != 'linux':
        raise BatchError('pidfd ownership requires Linux')
    import ctypes
    import operator
    pid = operator.index(pid)
    if ctypes.c_int(pid).value != pid:
        raise OverflowError('PID does not fit in the libc signed int argument')
    try:
        function = ctypes.CDLL(None, use_errno=True).pidfd_open
    except AttributeError as error:
        raise BatchError('neither Python nor libc exposes pidfd_open; no workers were started') from error
    function.argtypes = (ctypes.c_int, ctypes.c_uint)
    function.restype = ctypes.c_int
    descriptor = function(pid, 0)
    if descriptor < 0:
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error))
    return descriptor


def require_process_support():
    """Refuse unsupported runtime before creating any backend worker."""
    if (sys.platform != 'linux' or not Path('/proc/self/stat').is_file()
            or not hasattr(signal, 'pidfd_send_signal')):
        raise BatchError('batch runtime requires Linux /proc and pidfd identity-safe process signaling')
    try:
        boot_id()
        _stat(os.getpid())
        Path('/proc/self/environ').read_bytes()
        descriptor = open_pidfd(os.getpid())
        try:
            signal.pidfd_send_signal(descriptor, 0)
        finally:
            os.close(descriptor)
    except OSError as error:
        raise BatchError('pidfd ownership support is unavailable; no workers were started') from error


def boot_id():
    return Path('/proc/sys/kernel/random/boot_id').read_text().strip()


def _stat(pid):
    directory = Path('/proc') / str(pid)
    fields = (directory / 'stat').read_text().rsplit(')', 1)[1].split()
    return {'pid': pid, 'state': fields[0], 'process_group': int(fields[2]),
            'session_id': int(fields[3]), 'start_time': int(fields[19]), 'uid': directory.stat().st_uid}


def _cookie(pid, identity_env):
    prefix = identity_env.encode('ascii') + b'='
    environment = (Path('/proc') / str(pid) / 'environ').read_bytes()
    return next((entry[len(prefix):] for entry in environment.split(b'\0') if entry.startswith(prefix)), None)


def _exited(descriptor):
    poller = select.poll()
    poller.register(descriptor, select.POLLIN | select.POLLHUP | select.POLLERR)
    return bool(poller.poll(0))


def worker_record(pid, *, slot, device, identity, identity_env):
    """Capture at spawn before bootstrap release, and persist this same record at ready."""
    descriptor = open_pidfd(pid)
    try:
        info = _stat(pid)
        cookie = _cookie(pid, identity_env)
        if (_exited(descriptor) or info['state'] in ('Z', 'X') or info['process_group'] != pid
                or info['session_id'] != pid or info['uid'] != os.getuid() or cookie != identity.encode('ascii')):
            raise BatchError(f'new worker {pid} did not retain its verified process identity')
        return {**{key: info[key] for key in ('pid', 'process_group', 'session_id', 'start_time', 'uid')},
                'slot': slot, 'device': device, 'identity': identity, 'identity_env': identity_env, 'boot_id': boot_id()}
    finally:
        os.close(descriptor)


@contextmanager
def _group_handles(record):
    """Open pidfds before fresh verification; hold them until any signals are complete."""
    handles, foreign, ambiguous = [], [], []
    try:
        if record['boot_id'] != boot_id():
            yield ()
            return
        # A live numerical leader with another birth belongs to a reused session.
        try:
            leader = _stat(record['pid'])
        except FileNotFoundError:
            leader = None
        if leader is not None and record['start_time'] is not None and leader['start_time'] != record['start_time']:
            yield ()
            return
        for directory in Path('/proc').iterdir():
            if not directory.name.isdecimal():
                continue
            pid = int(directory.name)
            try:
                preliminary = _stat(pid)
            except (FileNotFoundError, ProcessLookupError):
                continue
            if preliminary['process_group'] != record['process_group'] or preliminary['state'] in ('Z', 'X'):
                continue
            try:
                descriptor = open_pidfd(pid)
            except ProcessLookupError:
                continue
            except PermissionError:
                ambiguous.append(pid)
                continue
            retain = False
            try:
                info = _stat(pid)
                cookie = _cookie(pid, record['identity_env'])
                if _exited(descriptor) or info['state'] in ('Z', 'X'):
                    continue
                if info['process_group'] != record['process_group']:
                    continue
                if (info['session_id'] != record['session_id'] or info['uid'] != record['uid']
                        or cookie != record['identity'].encode('ascii')):
                    foreign.append(pid)
                    continue
                if record['start_time'] is not None and info['start_time'] < record['start_time']:
                    ambiguous.append(pid)
                    continue
                handles.append((info, descriptor))
                retain = True
            except (FileNotFoundError, ProcessLookupError):
                continue
            except PermissionError:
                ambiguous.append(pid)
            finally:
                if not retain:
                    os.close(descriptor)
        if ambiguous or handles and foreign:
            raise BatchError(f'cannot prove exclusive ownership of worker group {record["process_group"]}; '
                             f'ambiguous/foreign pids={sorted(set(ambiguous + foreign))}. '
                             'No group signal was sent; inspect these processes before resuming.')
        # A clearly foreign/reused group has no matching cookie and is never signaled.
        yield tuple(handles)
    finally:
        for _, descriptor in handles:
            os.close(descriptor)


def group_members(record):
    """Actual active owned survivors, including after their original leader exits."""
    with _group_handles(record) as handles:
        return tuple(dict(info) for info, _ in handles)


def signal_owned(record, number):
    """Signal only verified, pinned members; callers rescan until the group drains."""
    with _group_handles(record) as handles:
        for info, descriptor in handles:
            try:
                signal.pidfd_send_signal(descriptor, number)
            except ProcessLookupError:
                pass
