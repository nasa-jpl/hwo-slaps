"""B7: a four-member JAX batch versus fresh direct preparations on the same device.

Each phase runs in its own fresh subprocess with persistent compilation caches off.
The batch wall includes controller/worker startup, preparation, writes and shutdown.
Per-job forecast time measures the actual completed forecast call, without preparation
or artifact serialization. CPU mode is a smoke record and cannot certify the GPU budget.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import signal

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
THREADS = ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS')


def _spec(config, cpu):
    from hwoslaps.batch import parse_batch
    from hwoslaps.config.schema import load_config
    base = load_config(config)
    lens = next(component for component in base.scene.lens.mass if 'einstein_radius' in component.values)
    return parse_batch({'name': 'B7', 'seed': 7, 'config': str(config),
        'population': {'seed': 3, 'count': 4, 'variables': {'radius': {'kind': 'uniform', 'low': .95, 'high': 1.05}},
                       'bind': {f'scene.lens.mass.{lens.name}.einstein_radius': 'radius'}},
        'forecast': {'masses_msun': [1e7, 1e8, 1e9]},
        'execution': {'engine': 'jax', 'devices': 'cpu' if cpu else [0], 'workers_per_device': 1,
                      'threads_per_worker': 1, 'preparation_cache_size': 2, 'memory_fraction': .75}}, base_dir=config.parent)


def _arrays(result):
    import numpy as np
    from hwoslaps.identity import array_digest
    names = ('masses_msun', 'positions_yx', 'fisher_raw', 'fisher_profiled', 'sigma_amplitude', 'q_asimov',
             'degradation', 'amplitude_hat', 'q_mismatch', 'z_mismatch', 'amplitude_spurious', 'q_spurious', 'z_spurious',
             'z_asimov', 'cell_areas_arcsec2', 'boundary')
    required = set(names[:7])
    arrays = {}
    for name in names:
        value = getattr(result, name)
        if value is None:
            if name in required:
                raise RuntimeError(f'B7 missing required science array {name}')
            arrays[name] = None
        else:
            if name in ('masses_msun', 'positions_yx', 'fisher_raw', 'fisher_profiled') and not np.all(np.isfinite(value)):
                raise RuntimeError(f'B7 nonfinite science array {name}')
            arrays[name] = array_digest(value)
    return arrays


def _worker_records(output, errors):
    records = []
    for path in (output / 'run/sessions').glob('*/workers.jsonl'):
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
                records.append(json.loads(line))
            except (ValueError, TypeError) as error:
                message = f'{path}: invalid complete ownership record: {error}'
                if message not in errors:
                    errors.append(message)
    return records


def _owned_phase(command, environment, log, output, deadline, device):
    from hwoslaps.batch.processes import boot_id, group_members, signal_owned, worker_record
    from hwoslaps.batch.runner import OWNER_ENV
    identity = os.urandom(32).hex()
    environment = {**environment, OWNER_ENV: identity}
    boot = boot_id()
    process = subprocess.Popen(command, env=environment, stdin=subprocess.PIPE, stdout=log,
                               stderr=subprocess.STDOUT, start_new_session=True)
    own = {'slot': 0, 'device': device, 'pid': process.pid, 'process_group': process.pid,
           'session_id': process.pid, 'identity': identity, 'identity_env': OWNER_ENV,
           'uid': os.getuid(), 'start_time': None, 'boot_id': boot}
    known, errors, blocked = {}, [], set()
    def records():
        required = {'slot', 'device', 'pid', 'process_group', 'session_id', 'identity',
                    'identity_env', 'uid', 'start_time', 'boot_id'}
        for record in _worker_records(output, errors):
            try:
                if not isinstance(record, dict) or set(record) != required or record['identity_env'] != OWNER_ENV:
                    raise ValueError('invalid ownership record shape or environment name')
                for field in ('pid', 'process_group', 'session_id', 'start_time', 'uid'):
                    if isinstance(record[field], bool) or not isinstance(record[field], int) or record[field] < 0:
                        raise ValueError(f'invalid ownership field {field}')
                if not isinstance(record['identity'], str) or not isinstance(record['boot_id'], str):
                    raise ValueError('invalid ownership identity or boot')
                known[(record['pid'], record['identity'])] = record
            except (ValueError, TypeError, KeyError) as error:
                message = str(error)
                if message not in errors:
                    errors.append(message)
        return [own, *known.values()]
    def owned_live(record):
        key = (record['pid'], record['identity'])
        if key in blocked:
            return False
        try:
            return bool(group_members(record))
        except Exception as error:
            blocked.add(key)
            errors.append(str(error))
            return False
    def pinned_signal(record, number):
        key = (record['pid'], record['identity'])
        if key in blocked:
            return
        try:
            signal_owned(record, number)
        except Exception as error:
            blocked.add(key)
            errors.append(str(error))
    try:
        own = worker_record(process.pid, slot=0, device=device, identity=identity, identity_env=OWNER_ENV)
        process.stdin.write(b'go\n')
        process.stdin.close()
        code = process.wait(timeout=max(0., deadline - time.monotonic()))
        if any(owned_live(record) for record in records()) or errors:
            raise RuntimeError('B7 phase returned with survivors or invalid ownership metadata')
        if code != 0:
            raise subprocess.CalledProcessError(code, command)
    except BaseException as original:
        previous_term = signal.signal(signal.SIGTERM, signal.SIG_IGN)
        try:
            process.stdin.close()
            for record in records():
                pinned_signal(record, signal.SIGTERM)
            grace = time.monotonic() + 35.
            while time.monotonic() < grace:
                process.poll()
                if not any(owned_live(record) for record in records()):
                    break
                time.sleep(.05)
            kill_deadline = time.monotonic() + 5.
            while True:
                for record in records():
                    pinned_signal(record, signal.SIGKILL)
                if not any(owned_live(record) for record in records()):
                    break
                if time.monotonic() >= kill_deadline:
                    errors.append('verified phase/worker descendants did not drain before the cleanup bound')
                    break
                time.sleep(.05)
            try:
                process.wait(timeout=5.)
            except subprocess.TimeoutExpired:
                errors.append('phase remains after ownership-safe cleanup; inspect recorded identities')
        finally:
            signal.signal(signal.SIGTERM, previous_term)
        if errors:
            raise RuntimeError('B7 ownership cleanup failed: ' + '; '.join(errors)) from original
        raise


def _phase(args):
    from hwoslaps.artifacts import write_json
    from hwoslaps.batch import open_batch, plan_batch, run_batch
    from hwoslaps.provenance import capture_provenance
    spec = _spec(args.config, args.cpu)
    plan = plan_batch(spec)
    if args.phase == 'direct':
        from hwoslaps.fisher.api import forecast, prepare_forecast
        rows = []
        start = time.perf_counter()
        oracle_s = 0.
        for job in plan.jobs:
            preparation_start = time.perf_counter()
            with prepare_forecast(job.config, execution=spec.execution.forecast) as prepared:
                preparation_s = time.perf_counter() - preparation_start
                import jax
                devices_now = jax.devices()
                if len(devices_now) != 1 or devices_now[0].platform != ('cpu' if args.cpu else 'gpu'):
                    raise RuntimeError('B7 actual prepared engine is on the wrong assigned device')
                run_start = time.perf_counter()
                result = forecast(prepared, masses_msun=job.parameters['masses_msun'])
                forecast_s = time.perf_counter() - run_start
                oracle_start = time.perf_counter()
                arrays = _arrays(result)
                oracle_s += time.perf_counter() - oracle_start
                rows.append({'job_id': job.job_id, 'config_digest': job.config_digest,
                             'prepare_s': preparation_s, 'forecast_s': forecast_s, 'arrays': arrays})
        wall_s = time.perf_counter() - start - oracle_s
        worker_start_s = 0.
        devices = []
        report = None
    else:
        start = time.perf_counter()
        report = run_batch(spec, args.output_dir / 'run', resume=False)
        wall_s = time.perf_counter() - start
        results = open_batch(args.output_dir / 'run')
        rows = []
        devices = []
        for job in results.jobs:
            if job.kind != 'forecast' or job.status != 'complete':
                continue
            marker = job.marker
            result = results.forecast(job.run_name, job.arm)
            rows.append({'job_id': job.job_id, 'config_digest': marker['preparation']['key'],
                         'prepare_s': marker['preparation']['prepare_s'],
                         'forecast_s': marker['timing']['forecast_s'], 'arrays': _arrays(result)})
            devices.append(marker['worker'])
        events = [json.loads(line) for line in (args.output_dir / 'run/events.jsonl').read_text().splitlines()]
        session = json.loads((args.output_dir / 'run/sessions/1/session.json').read_text())
        worker_start_s = min(event['time'] for event in events if event['type'] == 'worker_ready') - session['started']
    record = {'schema': 1, 'workload': 'B7', 'phase': args.phase, 'wall_s': wall_s, 'worker_start_s': worker_start_s,
              'rows': sorted(rows, key=lambda row: row['job_id']), 'workers': devices,
              'provenance': capture_provenance(), 'lane': 'cpu-smoke' if args.cpu else 'gpu',
              'report': None if report is None else report.to_mapping()}
    write_json(args.output_dir / (args.phase + '.json'), record)
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=ROOT / 'tests/fixtures/paper_parity/engine/p1_optical_matched.yaml')
    parser.add_argument('-o', '--output-dir', type=Path, required=True)
    parser.add_argument('--device', type=int, default=0, help='index of the parent visible allocation')
    parser.add_argument('--cpu', action='store_true', help='CPU smoke only; does not certify B7')
    parser.add_argument('--phase', choices=('direct', 'batch'), help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    args.config = args.config.expanduser().resolve()
    args.output_dir = args.output_dir.expanduser().resolve()
    if any(key in os.environ for key in ('JAX_COMPILATION_CACHE_DIR', 'JAX_ENABLE_COMPILATION_CACHE')):
        parser.error('persistent JAX compilation cache variables must be unset')
    if any(os.environ.get(key) != '1' for key in THREADS):
        parser.error('every BLAS thread variable must equal1')
    if args.phase:
        if sys.stdin.readline() != 'go\n':
            parser.error('benchmark phase requires its owning controller bootstrap')
        return _phase(args)
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        parser.error('output directory must be missing or empty')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    from hwoslaps.batch.processes import require_process_support
    require_process_support()
    environment = dict(os.environ)
    if args.cpu:
        environment.update(CUDA_VISIBLE_DEVICES='', JAX_PLATFORMS='cpu', JAX_ENABLE_X64='1')
    else:
        visible = environment.get('CUDA_VISIBLE_DEVICES')
        if visible is None or args.device < 0:
            parser.error('GPU benchmark requires an explicit assigned visible allocation and non-negative index')
        tokens = [token.strip() for token in visible.split(',')]
        if not all(tokens) or args.device >= len(tokens):
            parser.error('device index lies outside the assigned visible allocation')
        environment.pop('JAX_PLATFORMS', None)
        environment.update(CUDA_VISIBLE_DEVICES=tokens[args.device], JAX_ENABLE_X64='1',
                           XLA_PYTHON_CLIENT_PREALLOCATE='true', XLA_PYTHON_CLIENT_MEM_FRACTION='.7500')
    deadline = time.monotonic() + 540.  # Reserve the remaining minute for verified cleanup.
    def terminated(number, frame):
        raise InterruptedError(f'B7 controller received signal {number}')
    previous_term = signal.signal(signal.SIGTERM, terminated)
    try:
        for phase in ('direct', 'batch'):
            command = [sys.executable, str(Path(__file__).resolve()), '--phase', phase, '--config', str(args.config),
                       '--output-dir', str(args.output_dir)] + (['--cpu'] if args.cpu else [])
            with (args.output_dir / (phase + '.log')).open('wb') as log:
                _owned_phase(command, environment, log, args.output_dir, deadline,
                             'cpu' if args.cpu else environment['CUDA_VISIBLE_DEVICES'])
    except BaseException as error:
        from hwoslaps.artifacts import write_json
        write_json(args.output_dir / 'failure.json', {'schema': 1, 'workload': 'B7',
                   'error_type': type(error).__name__, 'message': str(error), 'budget_certified': False})
        raise
    finally:
        signal.signal(signal.SIGTERM, previous_term)
    direct = json.loads((args.output_dir / 'direct.json').read_text())
    batch = json.loads((args.output_dir / 'batch.json').read_text())
    if len(direct['rows']) != 4 or len(batch['rows']) != 4 or len({row['config_digest'] for row in direct['rows']}) != 4:
        from hwoslaps.artifacts import write_json
        write_json(args.output_dir / 'failure.json', {'schema': 1, 'workload': 'B7', 'budget_certified': False,
                   'message': 'B7 requires exactly four distinct complete forecast configurations'})
        return 3
    if len({row['job_id'] for row in direct['rows']}) != 4 or len({row['job_id'] for row in batch['rows']}) != 4:
        from hwoslaps.artifacts import write_json
        write_json(args.output_dir / 'failure.json', {'schema': 1, 'workload': 'B7', 'budget_certified': False,
                   'message': 'B7 requires four unique completed job rows'})
        return 3
    if [row['job_id'] for row in direct['rows']] != [row['job_id'] for row in batch['rows']]:
        from hwoslaps.artifacts import write_json
        write_json(args.output_dir / 'failure.json', {'schema': 1, 'workload': 'B7', 'budget_certified': False,
                   'message': 'B7 direct and batch selected different jobs'})
        return 3
    paired = []
    for expected, actual in zip(direct['rows'], batch['rows'], strict=True):
        if actual['arrays'] != expected['arrays'] or actual['config_digest'] != expected['config_digest']:
            from hwoslaps.artifacts import write_json
            write_json(args.output_dir / 'failure.json', {'schema': 1, 'workload': 'B7', 'budget_certified': False,
                       'message': f"B7 actual output arrays or configuration differ: {actual['job_id']}"})
            return 3
        paired.append({'job_id': actual['job_id'], 'direct_forecast_s': expected['forecast_s'],
                       'batch_forecast_s': actual['forecast_s'], 'ratio': actual['forecast_s'] / expected['forecast_s']})
    record = {'schema': 1, 'workload': 'B7', 'lane': direct['lane'], 'direct_wall_s': direct['wall_s'],
              'batch_wall_s': batch['wall_s'], 'worker_start_s': batch['worker_start_s'], 'pairs': paired,
              'actual_arrays_equal': True,
              'wall_within_budget': batch['wall_s'] <= 1.10 * direct['wall_s'] + 30.,
              'forecast_within_budget': all(row['ratio'] <= 1.05 for row in paired),
              'budget_certified': not args.cpu and batch['wall_s'] <= 1.10 * direct['wall_s'] + 30.
                                  and all(row['ratio'] <= 1.05 for row in paired)}
    from hwoslaps.artifacts import write_json
    write_json(args.output_dir / 'comparison.json', record)
    print(json.dumps(record, sort_keys=True))
    return 0 if args.cpu or record['budget_certified'] else 3


if __name__ == '__main__':
    raise SystemExit(main())
