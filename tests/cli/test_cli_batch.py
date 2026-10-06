"""Actual CLI plan/run/status wiring with inherited source binding."""
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from hwoslaps.artifacts import write_yaml

pytestmark = pytest.mark.backend


def _cli(arguments):
    environment = dict(os.environ)
    source = Path(__file__).resolve().parents[2] / 'src'
    environment['PYTHONPATH'] = str(source) + (os.pathsep + environment['PYTHONPATH'] if environment.get('PYTHONPATH') else '')
    return subprocess.run([sys.executable, '-m', 'hwoslaps', *arguments], env=environment,
                          capture_output=True, text=True, timeout=240)


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
