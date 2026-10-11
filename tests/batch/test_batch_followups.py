"""Artifact-selected trials: positive-amplitude eligibility and validated retry transport."""
from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path

import numpy as np
import pytest

from hwoslaps.artifacts import save_forecast
from hwoslaps.batch import BatchConflict, BatchError, parse_batch, plan_batch
from hwoslaps.batch.jobs import FollowupFailure, follow_ups
from hwoslaps.batch.state import claim_run_dir, publish_marker
from hwoslaps.fisher.positions import explicit_positions
from hwoslaps.fisher.result import ForecastResult
from hwoslaps.identity import file_digest


def test_configured_retry_uses_carried_trial_without_resolving_geometry(tiny_batch_spec, monkeypatch):
    from hwoslaps.batch.jobs import trial_id
    from hwoslaps.batch.worker import resolve_configured_trial
    spec = tiny_batch_spec()
    position = (.2718281828459045, -.3141592653589793)
    carried = {'mass_msun': 1e8, 'position_yx': list(position), 'trial_id': trial_id(1e8, position)}
    def forbidden(*args, **kwargs):
        raise AssertionError('retry attempted to resolve a critical curve on this host')
    monkeypatch.setattr('hwoslaps.scene.subhalo.configured_injection', forbidden)
    parameters = {'mass_msun': 1e8, 'resolved_trial': carried}
    assert resolve_configured_trial(parameters, spec.base, None) == carried
    parameters['resolved_trial'] = {**carried, 'trial_id': 'another'}
    with pytest.raises(BatchConflict, match='carried configured trial'):
        resolve_configured_trial(parameters, spec.base, None)


@pytest.mark.parametrize('kind,radius,amplitudes,expected', [
    ('forecast_positions', None, [-2., 3., 2.], [(0., .1), (0., .2), (0., .3)]),
    ('forecast_argmax', None, [-2., 3., 2.], [(0., .2)]),
    ('forecast_argmax', .2, [-2., 3., 2.], [(0., .2)]),
    ('forecast_argmax', None, [-2., 0., -1.], None),
    ('forecast_argmax', .15, [-2., 3., 2.], None),
    ('forecast_argmax', .05, [-2., 3., 2.], None),
    ('forecast_argmax', None, [np.nan, np.nan, np.nan], None),
])
def test_follow_ups_from_completed_forecasts(tiny_batch_spec, tmp_path, kind, radius, amplitudes, expected):
    selector = {'kind': kind, 'masses_msun': [1e8]}
    if kind == 'forecast_argmax':
        selector['aperture_radius_arcsec'] = radius
    family = {'trials': selector, 'inject': True, 'noise': False, 'fit': {'mode': 'fixed_template'}}
    spec = tiny_batch_spec(population={'count': 1}, nonlinear={'n': family})
    plan = plan_batch(spec)
    job = next(job for job in plan.jobs if job.kind == 'forecast')
    positions = explicit_positions([[0., .1], [0., .2], [0., .3]], (0., 0.))
    profiled = np.array([[4., 1., 1.]]) if np.all(np.isfinite(amplitudes)) else np.zeros((1, 3))
    result = ForecastResult(np.array([1e8]), positions, np.array([[4., 1., 1.]]), profiled,
                            np.array([amplitudes]), np.ones((1, 3)), 'knowledge_error', job.config.to_mapping(), {})
    root = tmp_path / 'followup'
    job_dir = root / job.job_id
    run = claim_run_dir(job_dir)
    path = save_forecast(result, run / 'forecast.npz')
    publish_marker(job_dir, {'schema': 1, 'job_id': job.job_id, 'job_digest': job.digest(), 'kind': 'forecast',
        'run': run.name, 'artifacts': {'forecast': {'path': str(path.relative_to(job_dir)),
        'bytes': path.stat().st_size, 'sha256': file_digest(path)}}})
    if expected is None:
        failures = follow_ups(plan, job, root)
        assert len(failures) == 1 and isinstance(failures[0], FollowupFailure)
        assert failures[0].source_job_id == job.job_id
        assert failures[0].family == 'n' and failures[0].mass_msun == 1e8
        terminal = root / failures[0].job_id / 'failure.json'
        from hwoslaps.artifacts import write_json
        write_json(terminal, failures[0].to_mapping())
        assert follow_ups(plan, job, root) == failures
        record = json.loads(terminal.read_text())
        record['source_job_id'] = 'another/member/forecast'
        terminal.write_text(json.dumps(record))
        with pytest.raises(BatchConflict, match='terminal follow-up identity changed'):
            follow_ups(plan, job, root)
    else:
        following = follow_ups(plan, job, root)
        assert [tuple(item.parameters['position_yx']) for item in following] == expected
        assert all(item.parameters['mass_msun'] == 1e8 and item.parameters['attempt'] == 0 for item in following)


def test_argmax_filters_the_fit_domain_before_ranking(tiny_batch_spec, tmp_path):
    family = {'arms': ['fit'], 'forecast_arm': 'wide', 'trials': {'kind': 'forecast_argmax', 'masses_msun': [1e8]},
              'inject': True, 'noise': False, 'fit': {'mode': 'fixed_template'}}
    spec = tiny_batch_spec(population={'count': 1}, arms=[{'name': 'wide'}, {'name': 'fit',
        'overrides': {'forecast': {'positions': {'kind': 'explicit', 'positions_yx': [[0., .2]]}}}}], nonlinear={'n': family})
    plan = plan_batch(spec)
    job = next(job for job in plan.jobs if job.kind == 'forecast' and job.arm == 'wide')
    positions = explicit_positions([[0., .1], [0., .3]], (0., 0.))
    result = ForecastResult(np.array([1e8]), positions, np.array([[4., 100.]]), np.array([[4., 100.]]),
                            np.ones((1, 2)), np.ones((1, 2)), 'knowledge_error', job.config.to_mapping(), {})
    job_dir = tmp_path / 'domain' / job.job_id
    run = claim_run_dir(job_dir)
    path = save_forecast(result, run / 'forecast.npz')
    publish_marker(job_dir, {'schema': 1, 'job_id': job.job_id, 'job_digest': job.digest(), 'kind': 'forecast',
        'run': run.name, 'artifacts': {'forecast': {'path': str(path.relative_to(job_dir)),
        'bytes': path.stat().st_size, 'sha256': file_digest(path)}}})
    selected = follow_ups(plan, job, tmp_path / 'domain')
    assert len(selected) == 1 and selected[0].parameters['position_yx'] == (0., .1)


def test_missing_requested_mismatch_metric_is_a_terminal_followup(tiny_batch_spec, tmp_path):
    family = {'trials': {'kind': 'forecast_argmax', 'masses_msun': [1e8], 'metric': 'q_mismatch'},
              'inject': True, 'noise': False, 'fit': {'mode': 'fixed_template'}}
    plan = plan_batch(tiny_batch_spec(population={'count': 1}, nonlinear={'n': family}))
    job = next(job for job in plan.jobs if job.kind == 'forecast')
    positions = explicit_positions([[0., .1]], (0., 0.))
    result = ForecastResult(np.array([1e8]), positions, np.ones((1, 1)), np.ones((1, 1)),
                            None, None, 'matched', job.config.to_mapping(), {})
    root = tmp_path / 'missing-metric'
    job_dir = root / job.job_id
    run = claim_run_dir(job_dir)
    path = save_forecast(result, run / 'forecast.npz')
    publish_marker(job_dir, {'schema': 1, 'job_id': job.job_id, 'job_digest': job.digest(), 'kind': 'forecast',
        'run': run.name, 'artifacts': {'forecast': {'path': str(path.relative_to(job_dir)),
        'bytes': path.stat().st_size, 'sha256': file_digest(path)}}})
    failures = follow_ups(plan, job, root)
    assert len(failures) == 1 and isinstance(failures[0], FollowupFailure)


@pytest.mark.parametrize('metric,support,expected_index', [('q_asimov', False, 1), ('q_mismatch', False, 0), ('q_asimov', True, 0)])
def test_argmax_metric_member_radius_and_image_support_are_independent(tiny_batch_spec, tmp_path, metric, support, expected_index):
    family = {'trials': {'kind': 'forecast_argmax', 'masses_msun': [1e8], 'metric': metric,
                        'aperture_einstein_radius_multiple': .5}, 'inject': True, 'noise': False,
              'fit': {'mode': 'fixed_template'}}
    spec = tiny_batch_spec(population={'count': 1}, nonlinear={'n': family})
    plan = plan_batch(spec)
    radius = plan.arm_configs[0].config.scene.einstein_radius()
    if support:
        grid = plan.arm_configs[0].config.scene.grid
        margin = grid.shape[1] * grid.pixel_scale_arcsec / 2 - .2 * radius
        trials = replace(spec.nonlinear[0].trials, support_margin_arcsec=margin)
        spec = replace(spec, nonlinear=(replace(spec.nonlinear[0], trials=trials),))
        plan = plan_batch(spec)
    job = next(job for job in plan.jobs if job.kind == 'forecast')
    points = [(0., scale * radius) for scale in (.1, .3, .7)]
    positions = explicit_positions(points, (0., 0.))
    result = ForecastResult(np.array([1e8]), positions, np.array([[1., 4., 9.]]), np.array([[1., 4., 9.]]),
                            np.array([[10., 1., 1.]]), np.ones((1, 3)), 'knowledge_error', job.config.to_mapping(), {})
    root = tmp_path / 'geometry'
    job_dir = root / job.job_id
    run = claim_run_dir(job_dir)
    path = save_forecast(result, run / 'forecast.npz')
    publish_marker(job_dir, {'schema': 1, 'job_id': job.job_id, 'job_digest': job.digest(), 'kind': 'forecast',
        'run': run.name, 'artifacts': {'forecast': {'path': str(path.relative_to(job_dir)),
        'bytes': path.stat().st_size, 'sha256': file_digest(path)}}})
    chosen = follow_ups(plan, job, root)
    assert len(chosen) == 1 and chosen[0].parameters['position_yx'] == points[expected_index]


@pytest.mark.backend
def test_unusable_member_followup_is_terminal_and_other_member_finishes(tiny_batch_spec, tmp_path, monkeypatch):
    from hwoslaps.batch import BatchIncomplete, open_batch, run_batch
    from hwoslaps.batch.frozen import persist_plan
    from hwoslaps.batch.state import source_revision
    from hwoslaps.provenance import capture_provenance
    family = {'trials': {'kind': 'forecast_argmax', 'masses_msun': [1e8]}, 'inject': True, 'noise': False,
              'fit': {'mode': 'fixed_template'}, 'sampler': {'n_live_smooth': 50, 'n_live_subhalo_fixed': 50,
              'n_eff': 200, 'n_shell': 1, 'f_live': .01, 'discard_exploration': True, 'number_of_cores': 1}}
    spec = tiny_batch_spec(nonlinear={'n': family}, execution={'workers_per_device': 1})
    plan = plan_batch(spec)
    root = tmp_path / 'terminal'
    persist_plan(root, plan)
    for job in plan.jobs:
        positions = explicit_positions([[0., .1]], (0., 0.))
        amplitude = -1. if job.member.index == 0 else 1.
        result = ForecastResult(np.array([1e8]), positions, np.ones((1, 1)), np.ones((1, 1)),
                                np.array([[amplitude]]), np.ones((1, 1)), 'knowledge_error', job.config.to_mapping(), {})
        job_dir = root / job.job_id
        run = claim_run_dir(job_dir)
        path = save_forecast(result, run / 'forecast.npz')
        publish_marker(job_dir, {'schema': 1, 'job_id': job.job_id, 'job_digest': job.digest(), 'kind': 'forecast',
            'run': run.name, 'source_revision': source_revision(capture_provenance()), 'artifacts': {'forecast': {
                'path': str(path.relative_to(job_dir)), 'bytes': path.stat().st_size, 'sha256': file_digest(path)}}})
    with pytest.raises(BatchIncomplete) as failure:
        run_batch(spec, root)
    assert failure.value.report.counts['failed'] == 1 and failure.value.report.counts['completed'] == 1
    failed = next(job for job in open_batch(root).jobs if job.kind == 'followup')
    assert failed.status == 'failed' and failed.family == 'n' and failed.source_job_id.endswith('/forecast')
    original = (failed.path / 'failure.json').read_bytes()
    def forbidden(*args, **kwargs):
        raise AssertionError('terminal resume attempted to redraw the population')
    monkeypatch.setattr('hwoslaps.batch.jobs.iter_population_members', forbidden)
    with pytest.raises(BatchIncomplete) as resumed:
        run_batch(spec, root)
    assert resumed.value.report.counts['failed'] == 1 and resumed.value.report.counts['skipped'] == 3
    assert (failed.path / 'failure.json').read_bytes() == original
    assert not (root / 'sessions/2/workers.jsonl').exists()


@pytest.mark.backend
def test_real_worker_executes_configured_critical_curve_retry(tiny_batch_spec, tmp_path):
    import math
    from hwoslaps.batch import open_batch, run_batch
    from hwoslaps.identity import mapping_digest
    family = {'trials': {'kind': 'configured'}, 'inject': True, 'noise': False,
              'fit': {'mode': 'fixed_template'},
              'sampler': {'n_live_smooth': 50, 'n_live_subhalo_fixed': 50, 'n_eff': 40,
                          'n_shell': 1, 'f_live': .1, 'n_like_max': 50000,
                          'discard_exploration': True, 'number_of_cores': 1},
              # No refinement is requested, so neither role can attain this status.
              'retry': {'acceptance': {'smooth': ['accepted_repeatable_profile'],
                                      'subhalo': ['accepted_repeatable_profile']},
                        'require_retained_state': False, 'stationarity_tolerance': None,
                        'sampler': {'n_eff': 40, 'n_like_max': 50000}}}
    initial = tiny_batch_spec(population=None, forecast=None, nonlinear={'critical': family},
                              execution={'devices': 'cpu', 'workers_per_device': 1, 'job_timeout_seconds': 120.})
    mapping = initial.to_mapping()
    mapping['config']['scene']['lens']['mass']['mass']['einstein_radius'] = .25
    mapping['config']['scene']['injection']['position'] = {
        'kind': 'angle', 'radius': 'critical_curve', 'angle_deg': 45.}
    spec = parse_batch(mapping, base_dir=initial.base_dir)
    blueprint, = plan_batch(spec).jobs
    assert blueprint.parameters['configured_trial'] and blueprint.parameters['position_yx'] is None
    assert blueprint.parameters['resolved_trial'] is None and blueprint.seeds == {}
    root = tmp_path / 'critical-curve-retry'
    report = run_batch(spec, root)
    assert report.counts['completed'] == 2 and report.counts['failed'] == 0
    batch = open_batch(root)
    records = sorted((job for job in batch.jobs if job.kind == 'nonlinear'), key=lambda job: job.attempt)
    assert [job.attempt for job in records] == [0, 1]
    assert all(job.status == 'complete' for job in records)
    first_trial = records[0].marker['summary']['trial']
    assert records[1].marker['summary']['trial'] == first_trial
    assert first_trial['mass_msun'] == 1e8
    y, x = first_trial['position_yx']
    centre_y, centre_x = spec.base.scene.lens_centre
    assert .1 < math.hypot(y - centre_y, x - centre_x) < .4
    assert math.degrees(math.atan2(y - centre_y, x - centre_x)) == pytest.approx(45., abs=1e-10)
    assert records[0].marker['seeds']['noise'] == records[1].marker['seeds']['noise'] is None
    assert records[0].marker['seeds']['sampler'] != records[1].marker['seeds']['sampler']
    assert records[0].marker['summary']['retry']['status'] != 'accepted'
    _, first, retry = next(batch.case_pairs(family='critical'))
    assert first.hypothesis == retry.hypothesis
    assert first.observation == retry.observation
    assert mapping_digest(first.data) == mapping_digest(retry.data)
    assert first.fit == retry.fit and first.models == retry.models
    for record, case in zip(records, (first, retry), strict=True):
        assert case.sampler_seed == record.marker['seeds']['sampler']
        assert case.hypothesis.position_yx_arcsec == tuple(first_trial['position_yx'])
        assert all(case.role(role).status == 'success' and case.role(role).log_likelihood is not None
                   for role in ('smooth', 'subhalo'))


@pytest.mark.backend
@pytest.mark.parametrize('tolerance,retry_needed', [(None, False), (1e-4, True)])
def test_retry_followup_consumes_actual_typed_stationarity_verdict(tiny_batch_spec, tmp_path, tolerance, retry_needed):
    from hwoslaps.analysis.nonlinear import case_status
    from hwoslaps.artifacts import load_case_snapshot, save_case
    from hwoslaps.batch.jobs import case_id
    from hwoslaps.batch.state import RetryVerdict
    from hwoslaps.identity import mapping_digest
    from hwoslaps.inference.result import CaseResult, ObservationRecord, RefineOutcome, RoleFit, RoleStatus
    from hwoslaps.scene.cosmology import Cosmology
    from hwoslaps.scene.halos import make_halo
    policy = {'acceptance': {'smooth': ['accepted_repeatable_profile'], 'subhalo': ['accepted_repeatable_profile']},
              'require_retained_state': False, 'stationarity_tolerance': tolerance}
    family = {'trials': {'kind': 'explicit', 'explicit': [{'mass_msun': 1e8, 'position_yx': [.1, .2]}]},
              'inject': False, 'noise': False, 'fit': {'mode': 'fixed_template'}, 'retry': policy}
    spec = tiny_batch_spec(population={'count': 1}, nonlinear={'n': family})
    plan = plan_batch(spec)
    job = next(job for job in plan.jobs if job.kind == 'nonlinear')
    typed = spec.nonlinear[0]
    cosmology = Cosmology(job.config.cosmology)
    halo = make_halo(job.config.scene.subhalo, 1e8, (.1, .2), redshift=.2, source_redshift=.6, cosmology=cosmology)
    roles = []
    for role, likelihood, gradient in [('smooth', -3., 1e-6), ('subhalo', -1., 1e-3)]:
        refinement = RefineOutcome(RoleStatus.ACCEPTED, likelihood, (0.,), {}, {}, gradient, True)
        roles.append(RoleFit(role, 'search', 'success', likelihood, likelihood, ('x',), 1, None,
                             refinement, None, None, None, 0.))
    case = CaseResult(case_id(job.job_id), halo, ObservationRecord('expected', None, job.config_digest, 'd' * 64, None),
                      {'shape': [40, 40]}, typed.fit, typed.sampler, job.seeds['sampler'], None,
                      {'smooth': 's', 'subhalo': 'h'}, *roles, 4., 4., None, None, None,
                      ('x',), job.config.comparison_digest(), {})
    root = tmp_path / 'retry'
    job_dir = root / job.job_id
    run = claim_run_dir(job_dir)
    path = save_case(case, run / 'case.json')
    validated, sha = load_case_snapshot(path)
    status = case_status(validated, acceptance=typed.retry.acceptance, require_retained_state=False,
                         stationarity_tolerance=tolerance)
    verdict = RetryVerdict(status, mapping_digest(typed.retry.to_mapping()), sha)
    publish_marker(job_dir, {'schema': 1, 'job_id': job.job_id, 'job_digest': job.digest(), 'kind': 'nonlinear',
        'run': run.name, 'artifacts': {'case': {'path': str(path.relative_to(job_dir)), 'bytes': path.stat().st_size,
        'sha256': sha}}, 'summary': {'retry': verdict.to_mapping()}})
    following = follow_ups(plan, job, root)
    assert len(following) == int(retry_needed)
    if following:
        assert following[0].parameters['attempt'] == 1
        assert following[0].seeds['noise'] == job.seeds['noise']
        assert following[0].seeds['sampler'] != job.seeds['sampler']
        assert following[0].parameters['retry_case_sha256'] == sha
    original = path.read_bytes()
    path.write_bytes(original.replace(b'"q_signed": 4.0', b'"q_signed": 5.0'))
    if path.read_bytes() == original:
        raise AssertionError('actual case-byte corruption control did not change its input')
    with pytest.raises(BatchConflict, match='validated case changed'):
        follow_ups(plan, job, root)
    path.write_bytes(original)


@pytest.mark.backend
def test_actual_worker_passes_stationarity_policy_to_real_case_classifier(tiny_batch_spec, tmp_path, monkeypatch):
    import hashlib
    import sys
    import hwoslaps.analysis.nonlinear as classifier
    import hwoslaps.batch.runner as runner
    import hwoslaps.batch.worker as worker
    from hwoslaps.batch import open_batch, run_batch
    tolerance = 1e-4
    statuses = ['sampler_only', 'verified_zero_residual_anchor', 'accepted_repeatable_profile']
    family = {'trials': {'kind': 'explicit', 'explicit': [{'mass_msun': 1e8, 'position_yx': [.1, .2]}]},
              'inject': True, 'noise': False, 'fit': {'mode': 'fixed_template'},
              'sampler': {'n_live_smooth': 50, 'n_live_subhalo_fixed': 50, 'n_eff': 200, 'n_shell': 1,
                          'f_live': .01, 'discard_exploration': True, 'number_of_cores': 1},
              'retry': {'acceptance': {'smooth': statuses, 'subhalo': statuses},
                        'require_retained_state': False, 'stationarity_tolerance': tolerance}}
    spec = tiny_batch_spec(population={'count': 1}, forecast=None, nonlinear={'n': family},
                           execution={'workers_per_device': 1})
    observed = tmp_path / 'actual-classifier-call.json'
    entry = tmp_path / 'actual-worker-entry.py'
    expected = {name: {'path': str(Path(module.__file__).resolve()),
                       'sha256': hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest()}
                for name, module in [('worker', worker), ('classifier', classifier)]}
    entry.write_text(f'''
if __name__ == '__main__':
    import hashlib, json
    from pathlib import Path
    import hwoslaps.batch.worker as worker
    import hwoslaps.analysis.nonlinear as classifier
    binding = {{}}
    for name, module in [('worker', worker), ('classifier', classifier)]:
        actual = Path(module.__file__).resolve()
        binding[name] = {{'path': str(actual), 'sha256': hashlib.sha256(actual.read_bytes()).hexdigest()}}
    assert binding == {expected!r}
    real_status = classifier.case_status
    def observed_status(case, **keywords):
        assert keywords.get('stationarity_tolerance') == {tolerance!r}, 'actual worker must pass its stationarity tolerance'
        assert all(case.role(role).status == 'success' for role in ('smooth', 'subhalo'))
        result = real_status(case, **keywords)
        Path({str(observed)!r}).write_text(json.dumps({{'binding': binding, 'case_id': case.case_id,
            'stationarity_tolerance': keywords['stationarity_tolerance'], 'status': result.status,
            'reasons': list(result.reasons)}}))
        return result
    classifier.case_status = observed_status
    raise SystemExit(worker.main())
''')
    real_popen = runner.subprocess.Popen
    def actual_worker_entry(command, *arguments, **keywords):
        if isinstance(command, (list, tuple)) and list(command) == [sys.executable, '-m', 'hwoslaps.batch.worker']:
            command = [sys.executable, str(entry)]
        return real_popen(command, *arguments, **keywords)
    monkeypatch.setattr(runner.subprocess, 'Popen', actual_worker_entry)
    root = tmp_path / 'stationarity-worker'
    report = run_batch(spec, root)
    assert report.counts['completed'] == 1 and report.counts['failed'] == 0
    record, actual = next(open_batch(root).cases())
    call = json.loads(observed.read_text())
    assert call['binding'] == expected and call['case_id'] == actual.case_id
    assert call['stationarity_tolerance'] == tolerance
    assert call['status'] == record.marker['summary']['retry']['status']
    assert call['reasons'] == record.marker['summary']['retry']['reasons']
