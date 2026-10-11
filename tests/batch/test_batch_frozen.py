"""Frozen cohort inputs survive host draw differences and reject persisted tampering."""
from copy import deepcopy
from dataclasses import replace

import pytest

from hwoslaps.batch import BatchConflict, parse_batch, plan_batch
from hwoslaps.batch.frozen import persist_plan, restored_plan
from hwoslaps.artifacts import write_json, write_yaml


@pytest.mark.parametrize('legacy', [False, True])
def test_resume_uses_persisted_values_without_drawing_again(tiny_batch_spec, tmp_path, monkeypatch, legacy):
    spec = tiny_batch_spec()
    original = plan_batch(spec)
    output = tmp_path / 'frozen'
    persist_plan(output, original)
    if legacy:
        session = output / 'sessions' / '1'
        write_yaml(session / 'batch_spec.yaml', spec.to_mapping())
        write_json(session / 'session.json', {'spec_digest': original.spec_digest, 'population_digest': original.population_digest})
        (output / 'frozen_plan.yaml').unlink()
        (output / 'frozen_plan.json').unlink()
        for member in original.members:
            (output / 'members' / member.run_name / 'member_config.yaml').unlink()
    def forbidden(*args, **kwargs):
        raise AssertionError('resume attempted another host-dependent population draw')
    monkeypatch.setattr('hwoslaps.batch.jobs.iter_population_members', forbidden)
    resumed = restored_plan(output, spec)
    assert [member.to_mapping() for member in resumed.members] == [member.to_mapping() for member in original.members]
    assert [job.digest() for job in resumed.jobs] == [job.digest() for job in original.jobs]
    assert [arm.config.to_mapping() for arm in resumed.arm_configs] == [arm.config.to_mapping() for arm in original.arm_configs]


@pytest.mark.parametrize('file', ['member.json', 'member_config.yaml', 'effective_config.yaml'])
def test_frozen_resume_rejects_even_semantically_equivalent_file_changes(tiny_batch_spec, tmp_path, file):
    spec = tiny_batch_spec()
    plan = plan_batch(spec)
    output = tmp_path / 'tamper'
    persist_plan(output, plan)
    root = output / 'members' / plan.members[0].run_name
    path = root / file if file != 'effective_config.yaml' else root / plan.arm_configs[0].name / file
    path.write_bytes(path.read_bytes() + b'\n')
    with pytest.raises(BatchConflict, match='bytes changed'):
        restored_plan(output, spec)


def test_frozen_science_identity_allows_runtime_placement_but_rejects_changed_inputs(tiny_batch_spec, tmp_path):
    spec = tiny_batch_spec()
    plan = plan_batch(spec)
    output = tmp_path / 'identity'
    persist_plan(output, plan)
    resumed = restored_plan(output, replace(spec, execution=replace(spec.execution, workers_per_device=1)))
    assert [job.digest() for job in resumed.jobs] == [job.digest() for job in plan.jobs]
    mapping = deepcopy(spec.to_mapping())
    mapping['population']['seed'] += 1
    with pytest.raises(BatchConflict, match='scientific specification'):
        restored_plan(output, parse_batch(mapping, base_dir=spec.base_dir))


@pytest.mark.parametrize('missing', ['frozen_plan.yaml', 'frozen_plan.json'])
def test_one_sided_frozen_storage_is_not_downgraded_to_legacy(tiny_batch_spec, tmp_path, missing):
    spec = tiny_batch_spec()
    output = tmp_path / 'one-sided'
    persist_plan(output, plan_batch(spec))
    (output / missing).unlink()
    retained = {path.name: path.read_bytes() for path in output.glob('frozen_plan.*')}
    with pytest.raises(BatchConflict, match='both be present'):
        restored_plan(output, spec)
    assert {path.name: path.read_bytes() for path in output.glob('frozen_plan.*')} == retained


def test_legacy_resume_rejects_matching_caller_and_prior_spec_tampering(tiny_batch_spec, tmp_path):
    spec = tiny_batch_spec()
    plan = plan_batch(spec)
    output = tmp_path / 'legacy-receipt'
    persist_plan(output, plan)
    for path in output.glob('frozen_plan.*'):
        path.unlink()
    session = output / 'sessions/1'
    write_json(session / 'session.json', {'spec_digest': plan.spec_digest, 'population_digest': plan.population_digest})
    mapping = spec.to_mapping()
    mapping['seed'] += 1
    write_yaml(session / 'batch_spec.yaml', mapping)
    requested = parse_batch(mapping, base_dir=spec.base_dir)
    with pytest.raises(BatchConflict, match='legacy session specification'):
        restored_plan(output, requested)


def test_catalogue_relocation_keeps_frozen_values_but_changed_bytes_conflict(tiny_batch_spec, tmp_path, monkeypatch):
    catalog = tmp_path / 'original.csv'
    catalog.write_text('radius\n0.6\n')
    spec = tiny_batch_spec(population={'count': 1, 'catalog': {'path': str(catalog), 'columns': {'radius': 'radius'}},
                                     'variables': {}})
    plan = plan_batch(spec)
    output = tmp_path / 'catalogue-frozen'
    persist_plan(output, plan)
    moved = tmp_path / 'moved.csv'
    catalog.rename(moved)
    mapping = spec.to_mapping()
    mapping['population']['catalog']['path'] = str(moved)
    requested = parse_batch(mapping, base_dir=spec.base_dir)
    def forbidden(*args, **kwargs):
        raise AssertionError('catalogue relocation caused a population redraw')
    monkeypatch.setattr('hwoslaps.batch.jobs.iter_population_members', forbidden)
    assert [job.digest() for job in restored_plan(output, requested).jobs] == [job.digest() for job in plan.jobs]
    moved.write_text('radius\n0.7\n')
    with pytest.raises(BatchConflict, match='catalogue bytes'):
        restored_plan(output, requested)


def test_frozen_job_rebuilds_reference_from_verified_arm_config(tiny_batch_spec, tmp_path):
    family = {'trials': {'kind': 'explicit', 'explicit': [{'mass_msun': 1e8, 'position_yx': [.1, .2]}]},
              'inject': False, 'noise': False, 'fit': {'mode': 'fixed_template'}}
    spec = tiny_batch_spec(nonlinear={'n': family})
    original = plan_batch(spec)
    output = tmp_path / 'reference'
    persist_plan(output, original)
    resumed = restored_plan(output, spec)
    planned = next(job for job in original.jobs if job.kind == 'nonlinear')
    restored = next(job for job in resumed.jobs if job.job_id == planned.job_id)
    assert restored.parameters['forecast_config_mapping'] == planned.parameters['forecast_config_mapping']
    assert restored.digest() == planned.digest()
