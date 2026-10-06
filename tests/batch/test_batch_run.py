"""Actual worker outputs, cache neutrality and resumable completion authority."""
from collections import Counter
from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path

import numpy as np
import pytest

from hwoslaps.batch import (BatchConflict, BatchIncomplete, BatchLocked, open_batch, plan_batch, run_batch)
from hwoslaps.config.loading import read_yaml
from hwoslaps.config.schema import parse_config
from hwoslaps.fisher.api import forecast, prepare_forecast
from hwoslaps.simulation import simulate
from hwoslaps.scene.cosmology import Cosmology
from hwoslaps.scene.subhalo import configured_injection

pytestmark = pytest.mark.backend


def _equal_arrays(left, right):
    for name in ('masses_msun', 'positions_yx', 'fisher_raw', 'fisher_profiled', 'amplitude_hat', 'amplitude_spurious'):
        a, b = getattr(left, name), getattr(right, name)
        if a is None:
            assert b is None
        else:
            assert a.dtype == b.dtype and a.shape == b.shape and a.tobytes() == b.tobytes(), name


def _direct_forecasts(spec, root):
    results = open_batch(root)
    jobs = {job.job_id: job for job in plan_batch(spec).jobs}
    for record in results.jobs:
        if record.kind != 'forecast' or record.status != 'complete':
            continue
        job = jobs[record.job_id]
        configuration = parse_config(read_yaml(root / 'members' / record.run_name / record.arm / 'effective_config.yaml'))
        with prepare_forecast(configuration, execution=spec.execution.forecast) as prepared:
            expected = forecast(prepared, masses_msun=job.parameters['masses_msun'])
        _equal_arrays(expected, results.forecast(record.run_name, record.arm))


def test_batch_artifacts_equal_direct_api(tiny_batch_spec, tmp_path):
    spec = tiny_batch_spec(simulate={'inject': True, 'noise': True, 'replicates': 2},
                          arms=[{'name': 'a'}, {'name': 'long', 'overrides': {'observation': {'exposure_time_s': 1800.}}}])
    root = tmp_path / 'batch'
    report = run_batch(spec, root, resume=False)
    assert report.counts['completed'] == 12 and report.counts['failed'] == 0
    _direct_forecasts(spec, root)
    results = open_batch(root)
    for record in results.jobs:
        if record.kind != 'simulate':
            continue
        config = parse_config(read_yaml(root / 'members' / record.run_name / record.arm / 'effective_config.yaml'))
        halo = configured_injection(config.scene, Cosmology(config.cosmology), seed=config.seed)
        expected = simulate(config, subhalo=halo, noise_seed=record.marker['seeds']['noise'])
        actual = results.observations(record.run_name, record.arm)[int(record.path.name[1:])]
        for name in ('data_adu', 'expected_adu', 'noise_map_adu', 'light_rate_e_per_s'):
            assert getattr(actual, name).tobytes() == getattr(expected, name).tobytes()
    members = [member.to_mapping() for member in plan_batch(spec).members]
    assert list(results.members) == members


def test_preparation_reuse_is_neutral(tiny_batch_spec, tmp_path):
    spec = tiny_batch_spec(forecast={'masses_msun': [1e7, 1e8]}, execution={'workers_per_device': 1},
        arms=[{'name': 'a'}, {'name': 'a_copy', 'overrides': {'observation': {'exposure_time_s': 900.}}},
              {'name': 'long', 'overrides': {'observation': {'exposure_time_s': 1800.}}}])
    root = tmp_path / 'cached'
    report = run_batch(spec, root)
    _direct_forecasts(spec, root)
    records = [job.marker['preparation'] for job in open_batch(root).jobs]
    misses = Counter(record['key'] for record in records if not record['cache_hit'])
    assert all(count == 1 for count in misses.values()) and len(misses) == 4
    assert report.counts['prepared'] == 4
    assert sum(record['cache_hit'] for record in records) == 2
    # A smaller actual cache must close evictions while retaining fresh direct bytes.
    smaller = replace(spec, execution=replace(spec.execution, preparation_cache_size=1))
    evicted = tmp_path / 'evicted'
    run_batch(smaller, evicted)
    _direct_forecasts(smaller, evicted)


def test_resume_runs_only_missing_jobs_and_verifies_artifacts(tiny_batch_spec, tmp_path):
    spec = tiny_batch_spec()
    root = tmp_path / 'resumed'
    first = run_batch(spec, root, select='members/system_000000/*')
    assert first.counts['completed'] == 1 and first.counts['not_selected'] == 1
    marker_path = root / 'members/system_000000/base/forecast/complete.json'
    original = marker_path.read_bytes()
    second = run_batch(spec, root)
    assert second.counts['completed'] == 1 and second.counts['skipped'] == 1
    assert marker_path.read_bytes() == original
    third = run_batch(spec, root, verify=True)
    assert third.counts['skipped'] == 2 and third.counts['completed'] == 0
    assert all(len(list(job.path.glob('run_*'))) == 1 for job in open_batch(root).jobs)
    extended = tiny_batch_spec(population={'count': 3})
    fourth = run_batch(extended, root)
    assert fourth.counts['completed'] == 1 and fourth.counts['skipped'] == 2
    changed = tiny_batch_spec(forecast={'masses_msun': [1e7, 1e8]})
    with pytest.raises(BatchConflict, match='job digest'):
        run_batch(changed, root)
    marker = json.loads(original)
    artifact = marker_path.parent / marker['artifacts']['forecast']['path']
    content = artifact.read_bytes()
    artifact.write_bytes(bytes([content[0] ^ 1]) + content[1:])
    with pytest.raises(BatchConflict, match='SHA-256'):
        run_batch(extended, root, verify=True)
    artifact.write_bytes(content)


def test_failed_jobs_are_recorded_and_rerun(tiny_batch_spec, tmp_path):
    bad = tmp_path / 'covariance.npy'
    np.save(bad, np.eye(2))
    spec = tiny_batch_spec(population={'count': 1}, arms=[{'name': 'good'}, {'name': 'bad',
        'overrides': {'forecast': {'noise_covariance': str(bad)}}}])
    assert len(plan_batch(spec).jobs) == 2  # The trigger is a build-time shape error.
    root = tmp_path / 'failure'
    with pytest.raises(BatchIncomplete) as failed:
        run_batch(spec, root)
    assert failed.value.report.counts['failed'] == 1 and failed.value.report.counts['completed'] == 1
    bad_job = root / 'members/system_000000/bad/forecast'
    assert not (bad_job / 'complete.json').exists()
    failure = json.loads((bad_job / 'run_001/failure.json').read_text())
    assert failure['error_type'] == 'ValueError'
    assert 'holds shape (2, 2), expected (1600, 1600)' in failure['message']
    assert 'Traceback' in failure['traceback']
    with pytest.raises(BatchIncomplete) as again:
        run_batch(spec, root)
    assert again.value.report.counts['skipped'] == 1 and again.value.report.counts['failed'] == 1
    assert (bad_job / 'run_002/failure.json').exists()


def test_second_controller_is_locked_out(tiny_batch_spec, tmp_path):
    import fcntl
    root = tmp_path / 'locked'
    root.mkdir()
    with (root / 'batch.lock').open('w') as stream:
        fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(BatchLocked):
            run_batch(tiny_batch_spec(), root)
    assert not (root / 'sessions').exists()


def test_resume_refuses_malformed_completion_and_worker_records(tiny_batch_spec, tmp_path):
    spec = tiny_batch_spec(population={'count': 1}, execution={'workers_per_device': 1})
    root = tmp_path / 'malformed'
    first = run_batch(spec, root)
    assert first.counts['completed'] == 1 and first.counts['failed'] == 0
    marker_path = root / 'members/system_000000/base/forecast/complete.json'
    marker_bytes = marker_path.read_bytes()
    marker = json.loads(marker_bytes)
    artifact = marker_path.parent / marker['artifacts']['forecast']['path']
    artifact_bytes = artifact.read_bytes()
    for defect, expected in [('invalid_json', 'invalid JSON'), ('schema', 'schema/kind'),
                             ('run', 'invalid run directory'), ('escape', 'escapes its claimed run'),
                             ('size', 'invalid recorded byte size'), ('sha', 'invalid recorded SHA-256'),
                             ('record', 'invalid artifact record')]:
        changed = deepcopy(marker)
        if defect == 'schema':
            changed['schema'] = 2
        elif defect == 'run':
            changed['run'] = '../outside'
        elif defect == 'escape':
            changed['artifacts']['forecast']['path'] = '../outside.npz'
        elif defect == 'size':
            changed['artifacts']['forecast']['bytes'] = True
        elif defect == 'sha':
            changed['artifacts']['forecast']['sha256'] = 'not-a-digest'
        elif defect == 'record':
            del changed['artifacts']['forecast']['sha256']
        try:
            marker_path.write_text('{invalid' if defect == 'invalid_json' else json.dumps(changed))
            with pytest.raises(BatchConflict, match=expected):
                run_batch(spec, root)
        finally:
            marker_path.write_bytes(marker_bytes)
        assert artifact.read_bytes() == artifact_bytes
    worker_path = root / 'sessions/1/workers.jsonl'
    worker_bytes = worker_path.read_bytes()
    record = json.loads(worker_bytes)
    for defect, expected in [('invalid_json', 'invalid worker record'), ('shape', 'invalid worker record'),
                             ('name', 'invalid ownership metadata name'), ('pid', 'invalid worker pid')]:
        changed = deepcopy(record)
        if defect == 'shape':
            changed['unknown'] = 1
        elif defect == 'name':
            changed['identity_env'] = 'A_FOREIGN_METADATA_NAME'
        elif defect == 'pid':
            changed['pid'] = True
        try:
            worker_path.write_text('{invalid\n' if defect == 'invalid_json' else json.dumps(changed) + '\n')
            with pytest.raises(BatchConflict, match=expected):
                run_batch(spec, root)
        finally:
            worker_path.write_bytes(worker_bytes)
        assert marker_path.read_bytes() == marker_bytes and artifact.read_bytes() == artifact_bytes
    resumed = run_batch(spec, root, verify=True)
    assert resumed.counts['skipped'] == 1 and resumed.counts['completed'] == 0


def test_batch_nonlinear_job_equals_direct_api(tiny_batch_spec, tmp_path):
    from hwoslaps.identity import array_digest
    from hwoslaps.inference.api import validate_nonlinear
    from hwoslaps.inference.result import ForecastReference
    from hwoslaps.inference.settings import FitSpec, SamplerSettings
    from hwoslaps.batch.jobs import case_id
    family = {'trials': {'kind': 'explicit', 'explicit': [{'members': 0, 'mass_msun': 1e8,
                                                        'position_yx': [.1, .2]}]},
              'inject': True, 'noise': True, 'fit': {'mode': 'fixed_template'},
              'sampler': {'n_live_smooth': 50, 'n_live_subhalo_fixed': 50, 'n_eff': 200,
                          'n_shell': 1, 'f_live': .01, 'discard_exploration': True, 'number_of_cores': 1}}
    spec = tiny_batch_spec(population={'count': 1}, nonlinear={'n': family}, execution={'workers_per_device': 1})
    root = tmp_path / 'nonlinear'
    report = run_batch(spec, root)
    assert report.counts['completed'] == 2 and report.counts['failed'] == 0
    records = list(open_batch(root).cases())
    assert len(records) == 1
    record, actual = records[0]
    planned = next(job for job in plan_batch(spec).jobs if job.kind == 'nonlinear')
    with prepare_forecast(planned.config, execution=spec.execution.forecast) as prepared:
        trial = prepared.hypothesis(1e8, (.1, .2))
        observed = simulate(prepared, subhalo=trial, noise_seed=record.marker['seeds']['noise'])
        assert array_digest(observed.data_adu) == actual.observation.data_digest
        expected_q = forecast(prepared, masses_msun=[1e8], positions=[[.1, .2]])
        reference = ForecastReference.from_result(expected_q, mass_index=0, position_index=0)
        assert actual.forecast_reference.to_mapping() == reference.to_mapping()
        expected = validate_nonlinear(prepared, trial, observed, fit=FitSpec.from_mapping(family['fit']),
            sampler=SamplerSettings.from_mapping(family['sampler']), sampler_seed=record.marker['seeds']['sampler'],
            output_dir=tmp_path / 'direct-fit', forecast_reference=reference, case_id=case_id(planned.job_id))
    assert actual.case_id == expected.case_id == case_id(planned.job_id)
    assert '/' not in actual.case_id
    for role in ('smooth', 'subhalo'):
        assert actual.role(role).status == expected.role(role).status == 'success'
        assert actual.role(role).acceptance_status is expected.role(role).acceptance_status
        assert actual.role(role).log_likelihood == expected.role(role).log_likelihood
        assert actual.role(role).sampler.log_likelihood_max == expected.role(role).sampler.log_likelihood_max


def test_preparation_refuses_real_input_swap_between_cache_key_and_capture(tiny_batch_spec, monkeypatch):
    import hwoslaps.fisher.api as api
    from hwoslaps.batch.worker import PreparationCache
    spec = tiny_batch_spec(population={'count': 1})
    config = plan_batch(spec).jobs[0].config
    path = Path(config.psf.truth.path)
    original = path.read_bytes()
    real_prepare = api.prepare_forecast
    kernel = np.zeros((7, 7))
    kernel[3, 3] = 1.
    def actual_preparation_from_changed_file(value, **keywords):
        np.save(path, kernel)
        try:
            return real_prepare(value, **keywords)
        finally:
            path.write_bytes(original)
    monkeypatch.setattr(api, 'prepare_forecast', actual_preparation_from_changed_file)
    cache = PreparationCache(2, spec.execution.forecast)
    try:
        with pytest.raises(BatchConflict, match='actual preparation inputs changed'):
            cache.get(config)
        assert cache.keys == ()
        assert path.read_bytes() == original
    finally:
        cache.close()
