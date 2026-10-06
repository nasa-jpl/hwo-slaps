"""Paired optical directions and an actual separate forecast-arm reference."""
from copy import deepcopy
from dataclasses import replace

import numpy as np
import pytest

from hwoslaps.batch import open_batch, parse_batch, plan_batch, run_batch
from hwoslaps.config.checks import ConfigError
from hwoslaps.fisher.api import forecast, prepare_forecast
from hwoslaps.inference.result import ForecastReference
from hwoslaps.seeding import derived_seed


@pytest.mark.parametrize('fit_directions,forecast_directions,valid', [(2, 2, True), (2, None, True),
                                                                    (None, 2, False), (2, 3, False)])
def test_direction_expansion_pairs_actual_configs(optical_batch_spec, fit_directions, forecast_directions, valid):
    draw = {'kind': 'knowledge_error', 'draw': {'prior': {'packaged': 'jwst_wss_drift_v1'},
            'amplitude_rms_nm': 5., 'seed': 0, 'family': 'global'}}
    arms = [{'name': 'fit', 'directions': fit_directions, 'overrides': {'psf': {'model': draw}}},
            {'name': 'fore', 'directions': forecast_directions, 'overrides': {'psf': {'model': {
                **draw, 'draw': {**draw['draw'], 'amplitude_rms_nm': 10.}}}}}]
    family = {'arms': ['fit'], 'forecast_arm': 'fore', 'trials': {'kind': 'explicit', 'explicit': [
        {'mass_msun': 1e8, 'position_yx': [.1, .2]}]}, 'inject': False, 'noise': False, 'fit': {'mode': 'fixed_template'}}
    spec = optical_batch_spec(population={'count': 1}, arms=arms,
                              forecast={'masses_msun': [1e8], 'arms': ['fore']}, nonlinear={'n': family})
    if not valid:
        with pytest.raises(ConfigError, match='direction'):
            plan_batch(spec)
        return
    plan = plan_batch(spec)
    fit = [arm for arm in plan.arm_configs if arm.base_name == 'fit']
    fore = [arm for arm in plan.arm_configs if arm.base_name == 'fore']
    for arm in fit:
        if arm.direction is not None:
            expected = derived_seed(spec.seed, 'batch/psf.model/direction', arm.member.index, arm.direction)
            assert arm.config.psf.model.draw.seed == expected
            if forecast_directions is not None:
                reference = next(other for other in fore if other.direction == arm.direction)
                assert reference.config.psf.model.draw.seed == expected
        job = next(job for job in plan.jobs if job.kind == 'nonlinear' and job.arm == arm.name)
        assert job.parameters['forecast_arm'] == ('fore' if forecast_directions is None else f'fore/d{arm.direction}')
    mapping = spec.to_mapping()
    mapping['arms'][0]['overrides']['psf']['model']['draw']['amplitude_rms_nm'] = 25.
    changed = plan_batch(parse_batch(mapping, base_dir=spec.base_dir))
    assert [arm.config.psf.model.draw.seed for arm in changed.arm_configs] == [arm.config.psf.model.draw.seed for arm in plan.arm_configs]


@pytest.mark.backend
def test_nonlinear_reference_uses_the_actual_999_forecast_arm(optical_batch_spec, tmp_path):
    base = optical_batch_spec(population={'count': 1})
    optical = deepcopy(base.to_mapping()['config']['psf']['truth'])
    wide = {**optical, 'kernel_shape': [999, 999]}
    narrow = {**optical, 'kernel_shape': [51, 51]}
    family = {'arms': ['fit'], 'forecast_arm': 'fore', 'trials': {'kind': 'explicit', 'explicit': [
        {'mass_msun': 1e8, 'position_yx': [.1, .2]}]}, 'inject': True, 'noise': False,
        'fit': {'mode': 'fixed_template'}, 'sampler': {'n_live_smooth': 50, 'n_live_subhalo_fixed': 50,
        'n_eff': 200, 'n_shell': 1, 'f_live': .01, 'discard_exploration': True}}
    mapping = base.to_mapping()
    mapping['arms'] = [{'name': 'fore', 'overrides': {'psf': {'model': wide}}},
                       {'name': 'fit', 'overrides': {'psf': {'model': narrow}}}]
    mapping['forecast'] = {'masses_msun': [1e8], 'arms': ['fore']}
    mapping['nonlinear'] = {'n': family}
    mapping['execution']['workers_per_device'] = 1
    spec = parse_batch(mapping, base_dir=base.base_dir)
    plan = plan_batch(spec)
    root = tmp_path / 'paired'
    report = run_batch(spec, root)
    assert report.counts['completed'] == 2 and report.counts['failed'] == 0
    record, case = next(open_batch(root).cases())
    assert case.smooth.status == case.subhalo.status == 'success'
    reference_config = next(arm.config for arm in plan.arm_configs if arm.name == 'fore')
    with prepare_forecast(reference_config, execution=spec.execution.forecast) as prepared:
        expected = ForecastReference.from_result(forecast(prepared, masses_msun=[1e8], positions=[[.1, .2]]),
                                                 mass_index=0, position_index=0)
    assert case.forecast_reference.to_mapping() == expected.to_mapping()
    assert case.forecast_reference.config_digest != case.observation.config_digest
    assert record.marker['preparation']['key'] == case.observation.config_digest
    assert record.marker['forecast_reference_preparation']['key'] == expected.config_digest
