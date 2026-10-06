"""Canonical plans and independently derived named stream identities."""
from copy import deepcopy
from dataclasses import replace
import hashlib

import numpy as np
import pytest

from hwoslaps.batch import BatchExecution, parse_batch, plan_batch
from hwoslaps.fisher.api import Execution


_FAMILY = {'trials': {'kind': 'explicit', 'explicit': [{'members': 'all', 'mass_msun': 1e8,
                                                     'position_yx': [.1, .2]}]},
           'inject': True, 'noise': True, 'replicates': 2, 'fit': {'mode': 'fixed_template'}}


def _seed(entropy, name, *indices):
    words = np.frombuffer(hashlib.sha256(name.encode()).digest(), dtype='<u4')
    return int(np.random.SeedSequence(entropy, spawn_key=(*indices, *(int(word) for word in words)))
               .generate_state(1, dtype=np.uint64)[0])


def test_plan_enumerates_job_ids_in_canonical_order(tiny_batch_spec):
    spec = tiny_batch_spec(arms=[{'name': 'a'}, {'name': 'b'}],
                          simulate={'inject': True, 'noise': True, 'replicates': 2},
                          nonlinear={'n': deepcopy(_FAMILY)})
    plan = plan_batch(spec)
    expected = []
    for member in ('system_000000', 'system_000001'):
        for arm in ('a', 'b'):
            prefix = f'members/{member}/{arm}/'
            expected.extend(prefix + tail for tail in ('simulate/r000', 'simulate/r001', 'forecast',
                'nonlinear/n/m1.000000e+08_y+0.100000_x+0.200000/r000/a0',
                'nonlinear/n/m1.000000e+08_y+0.100000_x+0.200000/r001/a0'))
    assert [job.job_id for job in plan.jobs] == expected
    assert len(plan.jobs) == 20


def test_job_seeds_follow_the_documented_names(tiny_batch_spec):
    spec = tiny_batch_spec(arms=[{'name': 'a'}, {'name': 'b'}], simulate={'inject': True, 'noise': True},
                          nonlinear={'n': deepcopy(_FAMILY)})
    original = plan_batch(spec)
    for job in original.jobs:
        replicate = job.parameters.get('replicate', 0)
        if job.kind == 'simulate':
            assert job.seeds['noise'] == _seed(7, 'batch/simulate/noise', job.member.index, replicate)
        elif job.kind == 'nonlinear':
            trial = 'm1.000000e+08_y+0.100000_x+0.200000'
            assert job.seeds['noise'] == _seed(7, f'batch/nonlinear/n/noise/{trial}', job.member.index, replicate)
            assert job.seeds['sampler'] == _seed(7, f'batch/nonlinear/n/sampler/{job.arm}/{trial}', job.member.index, replicate, 0)
    mapping = spec.to_mapping()
    mapping['arms'].append({'name': 'later'})
    mapping['nonlinear']['later'] = deepcopy(_FAMILY)
    extended = {job.job_id: job for job in plan_batch(parse_batch(mapping, base_dir=spec.base_dir)).jobs}
    for job in original.jobs:
        assert extended[job.job_id].seeds == job.seeds
        assert extended[job.job_id].digest() == job.digest()
    other_execution = BatchExecution(Execution('reference', reference_workers=3, batch_size=7),
                                     workers_per_device=4, preparation_cache_size=1)
    changed = plan_batch(replace(spec, execution=other_execution))
    assert [job.digest() for job in changed.jobs] == [job.digest() for job in original.jobs]
    jax = plan_batch(replace(spec, execution=replace(other_execution, forecast=Execution('jax'))))
    assert [job.digest() for job in jax.jobs if job.kind == 'simulate'] == [job.digest() for job in original.jobs if job.kind == 'simulate']
    assert [job.digest() for job in jax.jobs if job.kind != 'simulate'] != [job.digest() for job in original.jobs if job.kind != 'simulate']


def test_plan_refuses_rounded_trial_collisions_and_keeps_configured_positions(tiny_batch_spec):
    family = deepcopy(_FAMILY)
    family['trials']['explicit'].append({'members': 'all', 'mass_msun': 1e8, 'position_yx': [.10000001, .2]})
    with pytest.raises(ValueError, match='rounded trial identity'):
        plan_batch(tiny_batch_spec(nonlinear={'n': family}))
    configured = plan_batch(tiny_batch_spec(nonlinear={'n': {**deepcopy(_FAMILY), 'trials': {'kind': 'configured'}}}))
    for job in configured.jobs:
        if job.kind == 'nonlinear':
            assert job.parameters['mass_msun'] == 1e8
            assert job.parameters['position_yx'] == (.1, .2)
