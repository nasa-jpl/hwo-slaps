"""Two assigned GPU devices preserve actual forecast and nonlinear case results."""
from dataclasses import replace
import os

import pytest

from hwoslaps.batch import open_batch, run_batch

pytestmark = pytest.mark.xtx_multi_gpu


def test_multi_device_run_equals_single_worker_run(tiny_batch_spec, tmp_path):
    tokens = [token.strip() for token in os.environ['CUDA_VISIBLE_DEVICES'].split(',')]
    assert len(tokens) >= 2 and len(set(tokens[:2])) == 2, 'this owner requires two actually assigned devices'
    family = {'trials': {'kind': 'forecast_argmax', 'masses_msun': [1e8]},
              'inject': True, 'noise': False, 'fit': {'mode': 'fixed_template'},
              'sampler': {'n_live_smooth': 50, 'n_live_subhalo_fixed': 50, 'n_eff': 200,
                          'n_shell': 1, 'f_live': .01, 'discard_exploration': True,
                          'use_jax': True, 'number_of_cores': 1}}
    spec = tiny_batch_spec(nonlinear={'n': family}, execution={'engine': 'jax', 'devices': [0, 1], 'workers_per_device': 1})
    multi, single = tmp_path / 'multi', tmp_path / 'single'
    two = run_batch(spec, multi)
    one = run_batch(replace(spec, execution=replace(spec.execution, devices=(0,))), single)
    assert two.counts['completed'] == one.counts['completed'] == 4
    many, serial = open_batch(multi), open_batch(single)
    markers = [job.marker for job in many.jobs]
    assert {marker['worker']['device'] for marker in markers} == set(tokens[:2])
    for member in many.members:
        first = many.forecast(member['run_name'], 'base')
        second = serial.forecast(member['run_name'], 'base')
        for name in ('fisher_raw', 'fisher_profiled', 'amplitude_hat', 'amplitude_spurious'):
            left, right = getattr(first, name), getattr(second, name)
            assert right is None if left is None else left.tobytes() == right.tobytes()
    parallel_cases = {record.job_id: case for record, case in many.cases()}
    serial_cases = {record.job_id: case for record, case in serial.cases()}
    assert parallel_cases.keys() == serial_cases.keys()
    for name, case in parallel_cases.items():
        expected = serial_cases[name]
        for role in ('smooth', 'subhalo'):
            assert case.role(role).status == expected.role(role).status == 'success'
            assert case.role(role).acceptance_status is expected.role(role).acceptance_status
            assert case.role(role).log_likelihood == expected.role(role).log_likelihood
            assert case.role(role).sampler.log_likelihood_max == expected.role(role).sampler.log_likelihood_max
        assert case.forecast_reference.to_mapping() == expected.forecast_reference.to_mapping()
