"""Frozen cohort inputs survive host draw differences and reject persisted tampering."""
from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path

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


def test_append_draws_only_new_absolute_indices_and_preserves_all_previous_bytes(tiny_batch_spec, tmp_path, monkeypatch):
    import hwoslaps.population.sampling as sampling
    spec = tiny_batch_spec(population={'start': 10, 'count': 2})
    original = plan_batch(spec)
    output = tmp_path / 'append'
    persist_plan(output, original)
    initial_bytes = {path.relative_to(output): path.read_bytes() for path in output.rglob('*') if path.is_file()}
    actual = sampling.iter_population_members
    calls = []
    def added_only(base, population, count, **kwargs):
        calls.append((kwargs['start'], count))
        assert (kwargs['start'], count) in {(12, 1), (13, 1)}
        return actual(base, population, count, **kwargs)
    monkeypatch.setattr(sampling, 'iter_population_members', added_only)
    monkeypatch.setattr('hwoslaps.batch.jobs.iter_population_members', added_only)
    requested = replace(spec, population=replace(spec.population, count=3))
    extended = restored_plan(output, requested)
    assert [member.index for member in extended.members] == [10, 11, 12]
    assert [job.digest() for job in extended.jobs[:len(original.jobs)]] == [job.digest() for job in original.jobs]
    persist_plan(output, extended)
    first_append_bytes = {path.relative_to(output): path.read_bytes() for path in output.rglob('*') if path.is_file()}
    requested = replace(spec, population=replace(spec.population, count=4))
    extended_twice = restored_plan(output, requested)
    assert [member.index for member in extended_twice.members] == [10, 11, 12, 13]
    persist_plan(output, extended_twice)
    assert calls == [(12, 1), (13, 1)]
    for collection in (initial_bytes, first_append_bytes):
        assert all((output / path).read_bytes() == data for path, data in collection.items())
    def forbidden(*args, **kwargs):
        raise AssertionError('same-count resume tried to redraw a frozen member')
    monkeypatch.setattr(sampling, 'iter_population_members', forbidden)
    monkeypatch.setattr('hwoslaps.batch.jobs.iter_population_members', forbidden)
    assert [job.digest() for job in restored_plan(output, requested).jobs] == [job.digest() for job in extended_twice.jobs]
    before = {path.relative_to(output): path.read_bytes() for path in output.rglob('*') if path.is_file()}
    persist_plan(output, restored_plan(output, requested))
    assert before == {path.relative_to(output): path.read_bytes() for path in output.rglob('*') if path.is_file()}


@pytest.mark.parametrize('damage', ['yaml-tamper', 'missing-yaml', 'missing-seal', 'gap', 'resealed-prefix'])
def test_append_revisions_fail_closed_on_tamper_gaps_and_incomplete_pairs(tiny_batch_spec, tmp_path, damage):
    from hwoslaps.config.loading import read_yaml
    from hwoslaps.identity import file_digest
    spec = tiny_batch_spec()
    output = tmp_path / 'revision-integrity'
    persist_plan(output, plan_batch(spec))
    requested = replace(spec, population=replace(spec.population, count=3))
    persist_plan(output, restored_plan(output, requested))
    path, seal = output / 'frozen_plan_r000001.yaml', output / 'frozen_plan_r000001.json'
    if damage == 'yaml-tamper':
        path.write_bytes(path.read_bytes() + b'\n')
    elif damage == 'missing-yaml':
        path.unlink()
    elif damage == 'missing-seal':
        seal.unlink()
    elif damage == 'gap':
        path.rename(output / 'frozen_plan_r000002.yaml')
        seal.rename(output / 'frozen_plan_r000002.json')
    else:
        record = read_yaml(path)
        record['jobs'][0]['parameters']['masses_msun'] = [1e7]
        path.unlink()
        write_yaml(path, record)
        authority = json.loads(seal.read_text())
        authority['sha256'] = file_digest(path)
        seal.write_text(json.dumps(authority))
    before = {item.relative_to(output): item.read_bytes() for item in output.rglob('*') if item.is_file()}
    with pytest.raises(BatchConflict, match='frozen.plan'):
        restored_plan(output, requested)
    assert before == {item.relative_to(output): item.read_bytes() for item in output.rglob('*') if item.is_file()}


@pytest.mark.parametrize('change', ['shrink', 'seed', 'partial-member'])
def test_extension_rejects_other_science_changes_and_unsealed_member_directories(tiny_batch_spec, tmp_path, monkeypatch, change):
    import hwoslaps.population.sampling as sampling
    spec = tiny_batch_spec()
    output = tmp_path / 'extension-conflict'
    persist_plan(output, plan_batch(spec))
    requested = replace(spec, population=replace(spec.population, count=3))
    if change == 'shrink':
        requested = replace(spec, population=replace(spec.population, count=1))
    elif change == 'seed':
        requested = replace(requested, population=replace(requested.population, seed=spec.population.seed + 1))
    else:
        (output / 'members/system_000002').mkdir()
    def forbidden(*args, **kwargs):
        raise AssertionError('a conflicting request began drawing new members')
    monkeypatch.setattr(sampling, 'iter_population_members', forbidden)
    monkeypatch.setattr('hwoslaps.batch.jobs.iter_population_members', forbidden)
    with pytest.raises(BatchConflict):
        restored_plan(output, requested)
    assert not (output / 'frozen_plan_r000001.yaml').exists()


@pytest.mark.parametrize('append,boundary', [(append, boundary) for append, count in ((False, 8), (True, 5))
                                           for boundary in range(count + 1)])
def test_transaction_interruption_at_each_live_publication_boundary(tiny_batch_spec, tmp_path, monkeypatch, append, boundary):
    import hwoslaps.batch.frozen as frozen
    spec = tiny_batch_spec()
    output = tmp_path / 'transaction'
    original = plan_batch(spec)
    if append:
        persist_plan(output, original)
        # A committed scientific product must survive every interrupted append.
        write_json(output / original.jobs[0].job_id / 'result-receipt.json', {'accepted': True})
    before = {path.relative_to(output): path.read_bytes() for path in output.rglob('*') if path.is_file()}
    requested = replace(spec, population=replace(spec.population, count=3)) if append else spec
    plan = restored_plan(output, requested)
    actual_link, actual_rename = frozen.os.link, frozen.os.rename
    published = []
    def link(source, destination, *args, **kwargs):
        result = actual_link(source, destination, *args, **kwargs)
        relative = Path(destination).relative_to(output)
        if not relative.parts[0].startswith('.'):
            published.append(relative)
            if len(published) == boundary:
                raise KeyboardInterrupt('after live publication')
        return result
    def rename(source, destination, *args, **kwargs):
        result = actual_rename(source, destination, *args, **kwargs)
        if Path(destination).name == '.frozen-plan-pending' and boundary == 0:
            raise KeyboardInterrupt('after transaction preparation')
        return result
    with monkeypatch.context() as fault:
        fault.setattr(frozen.os, 'link', link)
        fault.setattr(frozen.os, 'rename', rename)
        with pytest.raises(KeyboardInterrupt):
            persist_plan(output, plan)
    committed = boundary == (5 if append else 8)
    # Rollback makes the prior cohort usable; an already committed seal is kept.
    resumed = restored_plan(output, requested if committed or not append else spec)
    assert len(resumed.members) == (len(plan.members) if committed else len(original.members))
    assert all((output / path).read_bytes() == data for path, data in before.items())
    assert not (output / '.frozen-plan-pending').exists()
    persist_plan(output, restored_plan(output, requested))
    assert [job.digest() for job in restored_plan(output, requested).jobs] == [job.digest() for job in plan.jobs]


def test_interrupted_transaction_refuses_modified_or_unexpected_live_files(tiny_batch_spec, tmp_path, monkeypatch):
    import hwoslaps.batch.frozen as frozen
    output = tmp_path / 'changed-transaction'
    spec = tiny_batch_spec()
    original = frozen.os.link
    def interrupted(source, destination, *args, **kwargs):
        result = original(source, destination, *args, **kwargs)
        if Path(destination) == output / 'members/system_000000/member.json':
            raise KeyboardInterrupt
        return result
    with monkeypatch.context() as fault:
        fault.setattr(frozen.os, 'link', interrupted)
        with pytest.raises(KeyboardInterrupt):
            persist_plan(output, plan_batch(spec))
    product = output / 'members/system_000000/unexpected-result.bin'
    product.write_bytes(b'preserve evidence')
    before = {path.relative_to(output): path.read_bytes() for path in output.rglob('*') if path.is_file()}
    with pytest.raises(BatchConflict, match='unexpected files or job output'):
        restored_plan(output, spec)
    assert before == {path.relative_to(output): path.read_bytes() for path in output.rglob('*') if path.is_file()}
