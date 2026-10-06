"""Batch-owned schema domains and effective-input round trips."""
from copy import deepcopy

import pytest

from hwoslaps.batch import parse_batch, plan_batch
from hwoslaps.config.checks import ConfigError


_FAMILY = {'trials': {'kind': 'explicit', 'explicit': [{'members': 'all', 'mass_msun': 1e8,
                                                     'position_yx': [.1, .2]}]},
           'inject': True, 'noise': False, 'fit': {'mode': 'fixed_template'}}
_RETRY = {'acceptance': {'smooth': ['accepted'], 'subhalo': ['accepted']},
          'require_retained_state': False, 'stationarity_tolerance': None}


@pytest.mark.parametrize('defect,path', [
    ('root_unknown', 'unknown'), ('execution_unknown', 'execution.unknown'), ('duplicate_arm', 'arms'),
    ('arm_seed', 'overrides.seed'), ('unnoisy_replicates', 'replicates'), ('missing_forecast', 'forecast'),
    ('bad_member', 'members'), ('bad_position', 'trials'), ('sampler_seed', 'sampler.seed'),
    ('refine_without_jax', 'refine'), ('retry_missing_stationarity', 'stationarity_tolerance'),
    ('retry_zero_stationarity', 'stationarity_tolerance'), ('retry_bad_role', 'acceptance.smooth'),
    ('devices_duplicate', 'execution.devices'), ('negative_memory', 'execution.memory_fraction'),
    ('trial_unknown', 'trials.unknown'), ('unknown_arm', 'arms'), ('no_family', 'at least one'),
])
def test_batch_spec_rejects_invalid_specs(tiny_batch_spec, defect, path):
    valid = tiny_batch_spec(nonlinear={'n': deepcopy(_FAMILY)})
    mapping = valid.to_mapping()
    if defect == 'root_unknown':
        mapping['unknown'] = True
    elif defect == 'execution_unknown':
        mapping['execution']['unknown'] = 1
    elif defect == 'duplicate_arm':
        mapping['arms'] *= 2
    elif defect == 'arm_seed':
        mapping['arms'][0]['overrides'] = {'seed': 3}
    elif defect == 'unnoisy_replicates':
        mapping['nonlinear']['n']['replicates'] = 2
    elif defect == 'missing_forecast':
        mapping['config']['forecast'] = None
    elif defect == 'bad_member':
        mapping['nonlinear']['n']['trials']['explicit'][0]['members'] = 2
    elif defect == 'bad_position':
        mapping['nonlinear']['n']['trials']['explicit'][0]['position_yx'] = [2., 0.]
    elif defect == 'sampler_seed':
        mapping['nonlinear']['n']['sampler']['seed'] = 1
    elif defect == 'refine_without_jax':
        mapping['nonlinear']['n']['refine'] = {}
    elif defect.startswith('retry_'):
        mapping['nonlinear']['n']['retry'] = deepcopy(_RETRY)
        if defect == 'retry_missing_stationarity':
            del mapping['nonlinear']['n']['retry']['stationarity_tolerance']
        elif defect == 'retry_zero_stationarity':
            mapping['nonlinear']['n']['retry']['stationarity_tolerance'] = 0.
        else:
            mapping['nonlinear']['n']['retry']['acceptance']['smooth'] = ['unknown']
    elif defect == 'devices_duplicate':
        mapping['execution']['devices'] = [0, 0]
    elif defect == 'negative_memory':
        mapping['execution']['memory_fraction'] = -.1
    elif defect == 'trial_unknown':
        mapping['nonlinear']['n']['trials']['unknown'] = 1
    elif defect == 'unknown_arm':
        mapping['forecast']['arms'] = ['absent']
    else:
        mapping.update(simulate=None, forecast=None, nonlinear={})
    with pytest.raises(ConfigError, match=path):
        plan_batch(parse_batch(mapping, base_dir=valid.base_dir))


def test_batch_spec_effective_mapping_round_trips(tiny_batch_spec):
    first = tiny_batch_spec(simulate={'inject': True, 'noise': True, 'replicates': 2},
                            nonlinear={'n': {**deepcopy(_FAMILY), 'retry': deepcopy(_RETRY)}})
    second = parse_batch(first.to_mapping(), base_dir=first.base_dir)
    assert second == first
    assert second.digest() == first.digest()
    assert second.to_mapping() == first.to_mapping()
