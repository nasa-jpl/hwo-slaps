"""Artifact-selected trials: positive-amplitude eligibility and validated retry transport."""
from copy import deepcopy
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from hwoslaps.artifacts import save_forecast
from hwoslaps.batch import BatchConflict, BatchError, parse_batch, plan_batch
from hwoslaps.batch.jobs import follow_ups
from hwoslaps.batch.state import claim_run_dir, publish_marker
from hwoslaps.fisher.positions import explicit_positions
from hwoslaps.fisher.result import ForecastResult
from hwoslaps.identity import file_digest


@pytest.mark.parametrize('kind,radius,amplitudes,expected', [
    ('forecast_positions', None, [-2., 3., 2.], [(0., .1), (0., .2), (0., .3)]),
    ('forecast_argmax', None, [-2., 3., 2.], [(0., .2)]),
    ('forecast_argmax', .2, [-2., 3., 2.], [(0., .2)]),
    ('forecast_argmax', None, [-2., 0., -1.], None),
    ('forecast_argmax', .15, [-2., 3., 2.], None),
    ('forecast_argmax', None, [np.nan, np.nan, np.nan], None),
])
def test_follow_ups_from_completed_forecast_artifacts(tiny_batch_spec, tmp_path, kind, radius, amplitudes, expected):
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
        with pytest.raises(BatchError, match=job.job_id):
            follow_ups(plan, job, root)
    else:
        following = follow_ups(plan, job, root)
        assert [tuple(item.parameters['position_yx']) for item in following] == expected
        assert all(item.parameters['mass_msun'] == 1e8 and item.parameters['attempt'] == 0 for item in following)


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
