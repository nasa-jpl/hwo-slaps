"""Result names cannot alias path separators or hide retry attempts."""
from dataclasses import replace

import pytest

from hwoslaps.artifacts import save_case, write_json, write_yaml
from hwoslaps.batch import open_batch, plan_batch
from hwoslaps.batch.jobs import case_id
from hwoslaps.batch.state import claim_run_dir, publish_marker
from hwoslaps.identity import file_digest, text_digest


def test_result_grammar_captures_family_and_attempt_even_with_reserved_run_and_arm(tmp_path):
    root = tmp_path / 'grammar'
    write_yaml(root / 'sessions/1/batch_spec.yaml', {'name': 'grammar'})
    for attempt in (0, 1):
        path = root / f'members/nonlinear/nonlinear/nonlinear/target/trial/r003/a{attempt}/run_001'
        write_json(path / 'failure.json', {'error_type': 'probe'})
    jobs = open_batch(root).jobs
    assert [(job.run_name, job.arm, job.family, job.trial, job.replicate, job.attempt) for job in jobs] == [
        ('nonlinear', 'nonlinear', 'target', 'trial', 3, 0), ('nonlinear', 'nonlinear', 'target', 'trial', 3, 1)]


@pytest.mark.backend
def test_cases_default_to_first_attempt_and_pairs_keep_both_archives(tiny_batch_spec, tmp_path):
    from hwoslaps.inference.result import CaseResult, ObservationRecord, RefineOutcome, RoleFit, RoleStatus
    from hwoslaps.scene.cosmology import Cosmology
    from hwoslaps.scene.halos import make_halo
    family = {'trials': {'kind': 'explicit', 'explicit': [{'mass_msun': 1e8, 'position_yx': [.1, .2]}]},
              'inject': False, 'noise': False, 'fit': {'mode': 'fixed_template'}}
    spec = tiny_batch_spec(population={'count': 1}, arms=[{'name': 'nonlinear'}], nonlinear={'target': family})
    original = next(job for job in plan_batch(spec).jobs if job.kind == 'nonlinear')
    config = original.config
    halo = make_halo(config.scene.subhalo, 1e8, (.1, .2), redshift=.2, source_redshift=.6, cosmology=Cosmology(config.cosmology))
    roles = [RoleFit(role, 'search', 'success', value, value, ('x',), 1, None,
                    RefineOutcome(RoleStatus.ACCEPTED, value, (0.,), {}, {}, 0., True), None, None, None, 0.)
             for role, value in [('smooth', -3.), ('subhalo', -1.)]]
    case = CaseResult(case_id(original.job_id), halo, ObservationRecord('expected', None, original.config_digest, 'd' * 64, None),
                      {'shape': [40, 40]}, spec.nonlinear[0].fit, spec.nonlinear[0].sampler, original.seeds['sampler'], None,
                      {'smooth': 's', 'subhalo': 'h'}, *roles, 4., 4., None, None, None,
                      ('x',), config.comparison_digest(), {})
    root = tmp_path / 'pairs'
    write_yaml(root / 'sessions/1/batch_spec.yaml', spec.to_mapping())
    for attempt in (0, 1):
        identifier = original.job_id[:-1] + str(attempt)
        job_dir = root / identifier
        run = claim_run_dir(job_dir)
        path = save_case(replace(case, case_id=case_id(identifier)), run / 'case.json')
        publish_marker(job_dir, {'schema': 1, 'job_id': identifier, 'job_digest': text_digest(identifier), 'kind': 'nonlinear',
                                'run': run.name, 'source_revision': 'test', 'artifacts': {'case': {
                                    'path': str(path.relative_to(job_dir)), 'bytes': path.stat().st_size, 'sha256': file_digest(path)}}})
    batch = open_batch(root)
    assert [job.attempt for job, _ in batch.cases(family='target')] == [0]
    assert [job.attempt for job, _ in batch.cases(family='target', attempt=None)] == [0, 1]
    pairs = list(batch.case_pairs(family='target'))
    assert len(pairs) == 1 and pairs[0][0].family == 'target'
    assert pairs[0][1].case_id != pairs[0][2].case_id
