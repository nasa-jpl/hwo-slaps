"""Persist the realized plan so resume never redraws a population or its trials."""
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
import re
import os
import shutil
import tempfile

from ..artifacts import write_json, write_yaml
from ..config.loading import read_yaml
from ..config.schema import parse_config
from ..identity import file_digest, mapping_digest, native_ready
from ..seeding import member_seed
from .jobs import BatchPlan, Job, MemberRecord, PlannedArm, build_plan, plan_batch
from .state import BatchConflict, read_json


def _retire_publication(pending):
    # Remove the recovery authority atomically before disposable stage cleanup.
    # An interruption during rmtree must not leave a half-deleted live journal.
    retired = pending.with_name('.frozen-stage-retired-' + os.urandom(16).hex())
    os.rename(pending, retired)
    shutil.rmtree(retired)


def plan_history(output, *, _uncommitted=None):
    """Read immutable revisions; every append retains the full previous realization."""
    root = output / 'frozen_plan.yaml'
    seal = output / 'frozen_plan.json'
    revisions = {}
    for path in output.glob('frozen_plan_r*'):
        if path == _uncommitted:
            continue
        match = re.fullmatch(r'frozen_plan_r([0-9]{6})\.(yaml|json)', path.name)
        if match is None:
            raise BatchConflict(f'unknown frozen-plan revision file: {path}')
        revisions.setdefault(int(match[1]), set()).add(match[2])
    root_exists = root.exists() and root != _uncommitted
    if root_exists != seal.exists() or revisions and not root_exists:
        raise BatchConflict('persisted frozen plan and its byte-identity seal must both be present')
    if not root_exists:
        return []
    if set(revisions) != set(range(1, len(revisions) + 1)) or any(kinds != {'yaml', 'json'} for kinds in revisions.values()):
        raise BatchConflict('frozen-plan revisions must be contiguous and paired with their seals')
    history = []
    for path in [root, *(output / f'frozen_plan_r{number:06d}.yaml' for number in sorted(revisions))]:
        sha = file_digest(path)
        expected = {'schema': 1, 'sha256': sha}
        if history:
            expected['previous_sha256'] = history[-1][2]
        if read_json(path.with_suffix('.json')) != expected:
            raise BatchConflict(f'persisted frozen plan bytes or revision link changed: {path}')
        record = read_yaml(path)
        if history:
            prior = history[-1][1]
            if len(record['members']) <= len(prior['members']):
                raise BatchConflict('frozen-plan revision must append population members')
            for key in ('members', 'arms', 'jobs'):
                if mapping_digest({'items': record[key][:len(prior[key])]}) != mapping_digest({'items': prior[key]}):
                    raise BatchConflict(f'frozen-plan revision changed previous {key}')
            for key in ('schema', 'base_config_digest', 'population_digest', 'deferred'):
                if record[key] != prior[key]:
                    raise BatchConflict(f'frozen-plan revision changed previous {key}')
        history.append((path, record, sha))
    return history


def _recover_publication(output):
    """Undo only the checked, uncommitted writes of our interrupted transaction.

    The seal is the commit point. A transaction is staged completely before any
    visible write, and jobs cannot start until persist_plan returns. Arbitrary
    orphan files, changed bytes and results in an uncommitted cohort conflict.
    """
    pending = output / '.frozen-plan-pending'
    if not pending.exists():
        return
    if pending.is_symlink() or not pending.is_dir():
        raise BatchConflict('invalid frozen-plan publication directory')
    manifest = read_json(pending / 'transaction.json')
    if set(manifest) != {'schema', 'previous_sha256', 'plan', 'files'} or manifest['schema'] != 1:
        raise BatchConflict('invalid frozen-plan publication transaction')
    name = manifest['plan']
    if not isinstance(name, str) or re.fullmatch(r'frozen_plan(?:_r[0-9]{6})?\.yaml', name) is None:
        raise BatchConflict('invalid frozen-plan transaction target')
    target = output / name
    files = manifest['files']
    if (not isinstance(files, dict) or name not in files or str(Path(name).with_suffix('.json')) not in files):
        raise BatchConflict('incomplete frozen-plan publication transaction')
    for relative, sha in files.items():
        path = Path(relative)
        if (path.is_absolute() or '..' in path.parts or not path.parts
                or relative not in (name, str(Path(name).with_suffix('.json')))
                and (len(path.parts) < 3 or path.parts[0] != 'members'
                     or path.name not in ('member.json', 'member_config.yaml', 'effective_config.yaml'))):
            raise BatchConflict('invalid frozen-plan transaction file path')
        staged = pending / path
        if (staged.is_symlink() or not staged.is_file() or file_digest(staged) != sha
                or not staged.resolve().is_relative_to(pending.resolve())):
            raise BatchConflict(f'changed staged frozen-plan bytes: {staged}')
        destination = output / path
        if (not destination.resolve().is_relative_to(output.resolve())
                or destination.exists() and (destination.is_symlink() or file_digest(destination) != sha)):
            raise BatchConflict(f'changed uncommitted frozen-plan bytes: {destination}')
    committed = target.with_suffix('.json').exists()
    history = plan_history(output, _uncommitted=None if committed else target)
    prior = history[:-1] if committed else history
    previous = prior[-1][2] if prior else None
    if previous != manifest['previous_sha256']:
        raise BatchConflict('frozen-plan transaction does not extend the committed history')
    expected_name = f'frozen_plan_r{len(prior):06d}.yaml' if prior else 'frozen_plan.yaml'
    protected = {entry['run_name'] for _, record, _ in prior for entry in record['members']}
    if name != expected_name or any(Path(path).parts[0] == 'members' and Path(path).parts[1] in protected for path in files):
        raise BatchConflict('frozen-plan transaction overlaps committed storage')
    if committed:
        if not history or history[-1][0] != target or any(not (output / path).is_file() for path in files):
            raise BatchConflict('committed frozen-plan transaction has missing files')
    else:
        # Check every affected member subtree before deleting anything. No job or
        # unrelated file may have been produced from an uncommitted realization.
        roots = {output / Path(path).parts[0] / Path(path).parts[1]
                 for path in files if Path(path).parts[0] == 'members'}
        for root in roots:
            if any(path.is_symlink() or path.is_file() and str(path.relative_to(output)) not in files
                   for path in root.rglob('*')):
                raise BatchConflict(f'uncommitted frozen cohort contains unexpected files or job output: {root}')
        for relative in reversed(files):
            path = output / relative
            path.unlink(missing_ok=True)
            parent = path.parent
            while parent != output:
                try:
                    parent.rmdir()
                except OSError:
                    break
                parent = parent.parent
    _retire_publication(pending)


def count_spec(spec, count):
    if spec.population is None:
        if count != 1:
            raise BatchConflict('a single-config batch cannot append population members')
        return spec
    return replace(spec, population=replace(spec.population, count=count))


def science_spec(spec):
    mapping = spec.to_mapping()
    mapping.pop('execution')
    if spec.population is not None:
        mapping['population'] = {**spec.population.to_mapping(), **spec.population.spec.identity_mapping()}
    return mapping


def member_names(spec):
    if spec.population is None:
        return [(0, spec.base.run_name, spec.base.seed)]
    population = spec.population
    return [(index, f'{population.name_prefix}_{index:06d}', member_seed(population.seed, index))
            for index in range(population.start, population.start + population.count)]


def bound_config(spec, record):
    """Replay bindings of already accepted values; evaluate no draws or hooks."""
    mapping = spec.base.to_mapping()
    mapping.update(run_name=record['run_name'], seed=record['seed'])
    if spec.population is not None:
        for path, reference in spec.population.spec.bind.items():
            segments = path.split('.')
            container = mapping
            for segment in segments[:-1]:
                key = int(segment) if isinstance(container, list) or segment not in container and segment.isdecimal() else segment
                container = container[key]
            segment = segments[-1]
            key = int(segment) if isinstance(container, list) or segment not in container and segment.isdecimal() else segment
            container[key] = deepcopy(reference.get(record['values']))
    return parse_config(mapping, base_dir=spec.base_dir)


def persisted_members(output, spec):
    members = []
    expected_names = member_names(spec)
    found = {path.parent.name for path in (output / 'members').glob('*/member.json')}
    if found != {name for _, name, _ in expected_names}:
        raise BatchConflict('persisted population indices/names differ from the requested frozen cohort')
    for index, name, seed in expected_names:
        record = read_json(output / 'members' / name / 'member.json')
        if (type(record['index']) is not int or type(record['seed']) is not int
                or type(record['attempt']) is not int or record['attempt'] < 0
                or spec.population is not None and record['attempt'] >= spec.population.spec.max_attempts):
            raise BatchConflict(f'invalid persisted population stream indices: {name}')
        if (record['index'], record['run_name'], record['seed']) != (index, name, seed):
            raise BatchConflict(f'persisted population seed/index/name changed: {name}')
        config_path = output / 'members' / name / 'member_config.yaml'
        config = parse_config(read_yaml(config_path)) if config_path.exists() else bound_config(spec, record)
        if config.digest() != record['config_digest']:
            raise BatchConflict(f'persisted population member configuration changed: {name}')
        members.append(MemberRecord(config=config, **record))
    if spec.population is not None and spec.population.spec.catalog is not None:
        captured = {member.catalog_sha256 for member in members}
        if len(captured) != 1 or file_digest(spec.population.spec.catalog.path) not in captured:
            raise BatchConflict('catalogue bytes differ from the frozen cohort identity')
    return tuple(members)


def persist_plan(output, plan):
    _recover_publication(output)
    history = plan_history(output)
    identity = mapping_digest(science_spec(plan.spec))
    if history:
        prior = history[-1][1]
        if (len(plan.members) < len(prior['members'])
                or prior['spec_identity'] != mapping_digest(science_spec(count_spec(plan.spec, len(prior['members']))))):
            raise BatchConflict('only an increased population count can append to the frozen plan')
    output.mkdir(parents=True, exist_ok=True)
    staging = tempfile.TemporaryDirectory(prefix='.frozen-stage-', dir=output)
    stage = Path(staging.name)
    files = {}
    def publish(path, mapping, writer):
        relative = path.relative_to(output)
        staged = stage / relative
        writer(staged, mapping)
        files[str(relative)] = file_digest(staged)
        return staged
    entries = []
    for member in plan.members:
        root = output / 'members' / member.run_name
        paths = {'record': root / 'member.json', 'config': root / 'member_config.yaml'}
        if paths['record'].exists():
            if read_json(paths['record']) != member.to_mapping():
                raise BatchConflict(f'persisted population member changed: {paths["record"]}')
        else:
            paths['record'] = publish(paths['record'], member.to_mapping(), write_json)
        if not paths['config'].exists():
            paths['config'] = publish(paths['config'], member.config.to_mapping(), write_yaml)
        entries.append({'index': member.index, 'run_name': member.run_name,
                        'record_sha256': file_digest(paths['record']), 'config_sha256': file_digest(paths['config'])})
    arms = []
    for arm in plan.arm_configs:
        path = output / 'members' / arm.member.run_name / arm.name / 'effective_config.yaml'
        if path.exists():
            if parse_config(read_yaml(path)).digest() != arm.config_digest:
                raise BatchConflict(f'persisted effective configuration changed: {path}')
        else:
            path = publish(path, arm.config.to_mapping(), write_yaml)
        arms.append({'member_index': arm.member.index, 'name': arm.name, 'base_name': arm.base_name,
                     'direction': arm.direction, 'config_digest': arm.config_digest, 'sha256': file_digest(path)})
    record = {'schema': 1, 'spec_identity': identity,
              'base_config_digest': plan.spec.base.digest(), 'spec_digest': plan.spec_digest,
              'population_digest': plan.population_digest, 'file_digests': dict(plan.file_digests),
              'catalog_path': None if plan.spec.population is None or plan.spec.population.spec.catalog is None
                              else str(plan.spec.population.spec.catalog.path),
              'members': entries, 'arms': arms, 'deferred': dict(plan.deferred),
              'jobs': [{'job_id': job.job_id, 'kind': job.kind, 'member_index': job.member.index,
                        'arm': job.arm, 'config_digest': job.config_digest, 'preparation_key': job.preparation_key,
                        'parameters': {key: value for key, value in job.parameters.items()
                                       if key != 'forecast_config_mapping'}, 'seeds': job.seeds} for job in plan.jobs]}
    if history:
        prior = history[-1][1]
        if record['spec_identity'] == prior['spec_identity']:
            staging.cleanup()
            return
        for key in ('members', 'arms', 'jobs'):
            if mapping_digest({'items': record[key][:len(prior[key])]}) != mapping_digest({'items': prior[key]}):
                raise BatchConflict(f'append changed previous frozen {key}')
        if (len(record['members']) <= len(prior['members']) or record['spec_identity'] != mapping_digest(science_spec(plan.spec))
                or prior['spec_identity'] != mapping_digest(science_spec(count_spec(plan.spec, len(prior['members']))))):
            raise BatchConflict('only an increased population count can append to the frozen plan')
        path = output / f'frozen_plan_r{len(history):06d}.yaml'
    else:
        path = output / 'frozen_plan.yaml'
    staged_plan = publish(path, native_ready(record), write_yaml)
    seal = {'schema': 1, 'sha256': file_digest(staged_plan)}
    if history:
        seal['previous_sha256'] = history[-1][2]
    publish(path.with_suffix('.json'), seal, write_json)
    write_json(stage / 'transaction.json', {'schema': 1, 'previous_sha256': history[-1][2] if history else None,
                                          'plan': path.name, 'files': files})
    pending = output / '.frozen-plan-pending'
    os.rename(stage, pending)
    # Link exact staged bytes without replacing a file. The final seal is last.
    # Keep the journal on any interruption so recovery can check every write.
    for relative in files:
        destination = output / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        try:
            os.link(pending / relative, destination)
        except FileExistsError as error:
            raise BatchConflict(f'frozen-plan publication destination already exists: {destination}') from error
    _retire_publication(pending)
    staging.cleanup()


def restored_plan(output, spec):
    """Return the checked frozen realization, or plan the first session."""
    _recover_publication(output)
    history = plan_history(output)
    if not history:
        if not any((output / 'members').glob('*/member.json')):
            return plan_batch(spec)
        # Upgrade existing completed cohorts without redrawing accepted values.
        previous = sorted((item for item in (output / 'sessions').glob('*/session.json') if item.parent.name.isdecimal()),
                          key=lambda item: int(item.parent.name))
        if not previous:
            raise BatchConflict('legacy persisted cohort has no session specification to verify')
        prior = read_yaml(previous[-1].parent / 'batch_spec.yaml')
        receipt = read_json(previous[-1])
        recorded_digest = mapping_digest({'spec': prior, 'config_digest': parse_config(prior['config']).digest(),
                                          'population_digest': receipt['population_digest']})
        if recorded_digest != receipt['spec_digest']:
            raise BatchConflict('legacy session specification or consumed input identity changed')
        prior.pop('execution')
        if prior['population'] is not None and prior['population'].get('catalog') is not None:
            prior['population']['catalog'].pop('path', None)
        for family in prior['nonlinear'].values():
            if family['trials']['kind'] == 'forecast_argmax':
                for key in ('aperture_einstein_radius_multiple', 'support_margin_arcsec', 'metric'):
                    family['trials'].setdefault(key, None)
        if mapping_digest(prior) != mapping_digest(science_spec(spec)):
            raise BatchConflict('requested scientific specification differs from the legacy frozen cohort')
        members = persisted_members(output, spec)
        configurations = []
        for member in members:
            for arm in spec.arms:
                for direction in (None,) if arm.directions is None else range(1, arm.directions + 1):
                    name = arm.name if direction is None else f'{arm.name}/d{direction}'
                    config = parse_config(read_yaml(output / 'members' / member.run_name / name / 'effective_config.yaml'))
                    configurations.append(PlannedArm(member, name, arm.name, direction, config, config.digest()))
        return build_plan(spec, frozen_members=members, frozen_arms=tuple(configurations))
    record = history[-1][1]
    requested = spec
    old_count = len(record['members'])
    requested_count = 1 if spec.population is None else spec.population.count
    if requested_count < old_count:
        raise BatchConflict('requested population count cannot shrink the frozen cohort')
    spec = count_spec(spec, old_count)
    if (any(item['schema'] != 1 or item['spec_identity'] != mapping_digest(science_spec(count_spec(spec, len(item['members']))))
            for _, item, _ in history) or record['base_config_digest'] != spec.base.digest()):
        raise BatchConflict('requested scientific specification or consumed inputs differ from the frozen plan')
    if any((output / 'members' / name).exists() for _, name, _ in member_names(requested)[old_count:]):
        raise BatchConflict('unsealed added-member directories conflict with population extension')
    for filename, digest in record['file_digests'].items():
        if filename != record['catalog_path'] and file_digest(filename) != digest:
            raise BatchConflict(f'consumed input bytes differ from the frozen plan: {filename}')
    for member in record['members']:
        root = output / 'members' / member['run_name']
        if (file_digest(root / 'member.json') != member['record_sha256']
                or file_digest(root / 'member_config.yaml') != member['config_sha256']):
            raise BatchConflict(f'persisted member/config bytes changed: {root}')
    members = persisted_members(output, spec)
    by_index = {member.index: member for member in members}
    configurations = []
    for arm in record['arms']:
        member = by_index[arm['member_index']]
        config_path = output / 'members' / member.run_name / arm['name'] / 'effective_config.yaml'
        if file_digest(config_path) != arm['sha256']:
            raise BatchConflict(f'persisted effective configuration bytes changed: {config_path}')
        config = parse_config(read_yaml(config_path))
        if config.digest() != arm['config_digest']:
            raise BatchConflict(f'persisted effective configuration identity changed: {config_path}')
        configurations.append(PlannedArm(member, arm['name'], arm['base_name'], arm['direction'], config, arm['config_digest']))
    by_arm = {(arm.member.index, arm.name): arm for arm in configurations}
    jobs = []
    for item in record['jobs']:
        arm = by_arm[item['member_index'], item['arm']]
        if item['config_digest'] != arm.config_digest:
            raise BatchConflict(f'frozen job configuration differs from its arm: {item["job_id"]}')
        parameters = deepcopy(item['parameters'])
        if item['kind'] == 'nonlinear':
            reference = by_arm[item['member_index'], parameters['forecast_arm']]
            if reference.config_digest != parameters['forecast_config_digest']:
                raise BatchConflict(f'frozen forecast reference differs from its arm: {item["job_id"]}')
            parameters['forecast_config_mapping'] = reference.config.to_mapping()
        jobs.append(Job(item['job_id'], item['kind'], arm.member, arm.name, arm.config, arm.config_digest,
                        item['preparation_key'], parameters, item['seeds'],
                        None if item['kind'] == 'simulate' else spec.execution.forecast.engine))
    current = BatchPlan(spec, members, tuple(jobs), record['deferred'], tuple(configurations),
                     spec.captured_digest(base_config_digest=record['base_config_digest'],
                                          population_digest=record['population_digest']),
                     record['population_digest'], record['file_digests'])
    if requested_count == old_count:
        return current
    from ..population.sampling import iter_population_members
    population = requested.population
    additions = tuple(MemberRecord(item.index, item.run_name, item.seed, item.attempt, item.values, item.config,
                                   item.config.digest(), item.catalog_sha256,
                                   population.spec.captured_digest(catalog_sha256=item.catalog_sha256))
                      for item in iter_population_members(requested.base, population.spec, requested_count - old_count,
                          seed=population.seed, start=population.start + old_count, name_prefix=population.name_prefix,
                          base_dir=requested.base_dir))
    extra = build_plan(requested, frozen_members=additions,
                       member_indices={index for index, _, _ in member_names(requested)})
    if extra.population_digest != current.population_digest:
        raise BatchConflict('added members differ from the frozen population identity')
    files = {path: sha for path, sha in current.file_digests.items() if path != record['catalog_path']}
    for path, sha in extra.file_digests.items():
        if path in files and files[path] != sha:
            raise BatchConflict(f'consumed input changed while extending the population: {path}')
        files[path] = sha
    return BatchPlan(requested, (*members, *additions), (*jobs, *extra.jobs), current.deferred,
                     (*configurations, *extra.arm_configs),
                     requested.captured_digest(base_config_digest=record['base_config_digest'], population_digest=current.population_digest),
                     current.population_digest, files)
