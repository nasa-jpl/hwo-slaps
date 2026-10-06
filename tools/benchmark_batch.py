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
    from hwoslaps.identity import array_digest
    names = ('masses_msun', 'positions_yx', 'fisher_raw', 'fisher_profiled', 'amplitude_hat', 'amplitude_spurious')
    return {name: None if getattr(result, name) is None else array_digest(getattr(result, name)) for name in names}


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
        for job in plan.jobs:
            preparation_start = time.perf_counter()
            with prepare_forecast(job.config, execution=spec.execution.forecast) as prepared:
                preparation_s = time.perf_counter() - preparation_start
                run_start = time.perf_counter()
                result = forecast(prepared, masses_msun=job.parameters['masses_msun'])
                forecast_s = time.perf_counter() - run_start
                rows.append({'job_id': job.job_id, 'config_digest': job.config_digest,
                             'prepare_s': preparation_s, 'forecast_s': forecast_s, 'arrays': _arrays(result)})
        wall_s = time.perf_counter() - start
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
        return _phase(args)
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        parser.error('output directory must be missing or empty')
    args.output_dir.mkdir(parents=True, exist_ok=True)
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
        environment.update(CUDA_VISIBLE_DEVICES=tokens[args.device], JAX_PLATFORMS='cuda', JAX_ENABLE_X64='1')
    for phase in ('direct', 'batch'):
        command = [sys.executable, str(Path(__file__).resolve()), '--phase', phase, '--config', str(args.config),
                   '--output-dir', str(args.output_dir)] + (['--cpu'] if args.cpu else [])
        with (args.output_dir / (phase + '.log')).open('wb') as log:
            subprocess.run(command, env=environment, stdout=log, stderr=subprocess.STDOUT, check=True)
    direct = json.loads((args.output_dir / 'direct.json').read_text())
    batch = json.loads((args.output_dir / 'batch.json').read_text())
    if [row['job_id'] for row in direct['rows']] != [row['job_id'] for row in batch['rows']]:
        raise RuntimeError('B7 direct and batch selected different jobs')
    paired = []
    for expected, actual in zip(direct['rows'], batch['rows'], strict=True):
        if actual['arrays'] != expected['arrays'] or actual['config_digest'] != expected['config_digest']:
            raise RuntimeError(f"B7 actual output arrays or configuration differ: {actual['job_id']}")
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
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
