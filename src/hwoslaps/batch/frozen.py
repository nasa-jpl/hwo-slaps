"""Persist the realized plan so resume never redraws a population or its trials."""
from copy import deepcopy
from pathlib import Path

from ..artifacts import write_json, write_yaml
from ..config.loading import read_yaml
from ..config.schema import parse_config
from ..identity import file_digest, mapping_digest, native_ready
from ..seeding import member_seed
from .jobs import BatchPlan, Job, MemberRecord, PlannedArm, build_plan, plan_batch
from .state import BatchConflict, read_json


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
    entries = []
    for member in plan.members:
        root = output / 'members' / member.run_name
        paths = {'record': root / 'member.json', 'config': root / 'member_config.yaml'}
        if paths['record'].exists():
            if read_json(paths['record']) != member.to_mapping():
                raise BatchConflict(f'persisted population member changed: {paths["record"]}')
        else:
            write_json(paths['record'], member.to_mapping())
        if not paths['config'].exists():
            write_yaml(paths['config'], member.config.to_mapping())
        entries.append({'index': member.index, 'run_name': member.run_name,
                        'record_sha256': file_digest(paths['record']), 'config_sha256': file_digest(paths['config'])})
    arms = []
    for arm in plan.arm_configs:
        path = output / 'members' / arm.member.run_name / arm.name / 'effective_config.yaml'
        if path.exists():
            if parse_config(read_yaml(path)).digest() != arm.config_digest:
                raise BatchConflict(f'persisted effective configuration changed: {path}')
        else:
            write_yaml(path, arm.config.to_mapping())
        arms.append({'member_index': arm.member.index, 'name': arm.name, 'base_name': arm.base_name,
                     'direction': arm.direction, 'config_digest': arm.config_digest, 'sha256': file_digest(path)})
    record = {'schema': 1, 'spec_identity': mapping_digest(science_spec(plan.spec)),
              'base_config_digest': plan.spec.base.digest(), 'spec_digest': plan.spec_digest,
              'population_digest': plan.population_digest, 'file_digests': dict(plan.file_digests),
              'catalog_path': None if plan.spec.population is None or plan.spec.population.spec.catalog is None
                              else str(plan.spec.population.spec.catalog.path),
              'members': entries, 'arms': arms, 'deferred': dict(plan.deferred),
              'jobs': [{'job_id': job.job_id, 'kind': job.kind, 'member_index': job.member.index,
                        'arm': job.arm, 'config_digest': job.config_digest, 'preparation_key': job.preparation_key,
                        'parameters': {key: value for key, value in job.parameters.items()
                                       if key != 'forecast_config_mapping'}, 'seeds': job.seeds} for job in plan.jobs]}
    path = output / 'frozen_plan.yaml'
    if path.exists():
        return
    write_yaml(path, native_ready(record))
    write_json(output / 'frozen_plan.json', {'schema': 1, 'sha256': file_digest(path)})


def restored_plan(output, spec):
    """Return the checked frozen realization, or plan the first session."""
    path = output / 'frozen_plan.yaml'
    seal = output / 'frozen_plan.json'
    if path.exists() != seal.exists():
        raise BatchConflict('persisted frozen plan and its byte-identity seal must both be present')
    if not path.exists():
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
    authority = read_json(seal)
    if authority != {'schema': 1, 'sha256': file_digest(path)}:
        raise BatchConflict('persisted frozen plan bytes changed')
    record = read_yaml(path)
    if (record['schema'] != 1 or record['spec_identity'] != mapping_digest(science_spec(spec))
            or record['base_config_digest'] != spec.base.digest()):
        raise BatchConflict('requested scientific specification or consumed inputs differ from the frozen plan')
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
    return BatchPlan(spec, members, tuple(jobs), record['deferred'], tuple(configurations),
                     spec.captured_digest(base_config_digest=record['base_config_digest'],
                                          population_digest=record['population_digest']),
                     record['population_digest'], record['file_digests'])
