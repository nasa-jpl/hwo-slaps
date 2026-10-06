"""Deterministic batch plans and follow-ups derived from winning completed artifacts."""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from ..analysis.reductions import aperture_selection
from ..artifacts import load_forecast
from ..config.checks import ConfigError
from ..config.schema import EngineConfig
from ..fisher.positions import explicit_positions, grid_positions, ring_positions
from ..identity import file_digest, json_ready, mapping_digest, text_digest
from ..inference.settings import RefineSettings, SamplerSettings
from ..population import iter_population_members
from ..scene.cosmology import Cosmology
from ..scene.subhalo import configured_injection
from ..seeding import derived_seed
from .spec import BatchExecution, BatchSpec
from .state import BatchConflict, BatchError, RetryVerdict, read_marker, verify_marker


@dataclass(frozen=True)
class MemberRecord:
    index: int
    run_name: str
    seed: int
    attempt: int
    values: Mapping[str, Any]
    config: EngineConfig
    config_digest: str
    catalog_sha256: str | None = None
    population_digest: str | None = None

    def to_mapping(self):
        return json_ready({'index': self.index, 'run_name': self.run_name, 'seed': self.seed,
                           'attempt': self.attempt, 'values': self.values, 'config_digest': self.config_digest,
                           'catalog_sha256': self.catalog_sha256, 'population_digest': self.population_digest})


@dataclass(frozen=True)
class PlannedArm:
    member: MemberRecord
    name: str
    base_name: str
    direction: int | None
    config: EngineConfig
    config_digest: str


@dataclass(frozen=True)
class JobPayload:
    job_id: str
    kind: str
    job_dir: str
    job_digest: str
    config_mapping: Mapping[str, Any]
    config_digest: str
    parameters: Mapping[str, Any]
    seeds: Mapping[str, int | None]
    session: int
    execution: Mapping[str, Any]

    def to_mapping(self):
        return json_ready(self.__dict__)


@dataclass(frozen=True)
class Job:
    job_id: str
    kind: str
    member: MemberRecord
    arm: str
    config: EngineConfig
    config_digest: str
    preparation_key: str | None
    parameters: Mapping[str, Any]
    seeds: Mapping[str, int | None]
    engine: str | None

    def digest(self):
        parameters = {key: value for key, value in self.parameters.items()
                      if key not in ('forecast_config_mapping', 'retry_case_path')}
        record = {'kind': self.kind, 'job_id': self.job_id, 'config_digest': self.config_digest,
                  'parameters': parameters, 'seeds': self.seeds}
        if self.engine is not None:
            record['engine'] = self.engine
        return mapping_digest(record)

    def payload(self, output_dir: Path, session: int, execution: BatchExecution) -> JobPayload:
        return JobPayload(self.job_id, self.kind, str(output_dir / self.job_id), self.digest(),
                          self.config.to_mapping(), self.config_digest, json_ready(self.parameters),
                          dict(self.seeds), session, json_ready(execution.to_mapping()))


@dataclass(frozen=True)
class BatchPlan:
    spec: BatchSpec
    members: tuple[MemberRecord, ...]
    jobs: tuple[Job, ...]
    deferred: Mapping[str, str]
    arm_configs: tuple[PlannedArm, ...]
    spec_digest: str
    population_digest: str | None
    file_digests: Mapping[str, str]

    def to_mapping(self):
        return {'members': len(self.members), 'arms': len(self.spec.arms),
                'static_jobs': {kind: sum(job.kind == kind for job in self.jobs)
                                for kind in ('simulate', 'forecast', 'nonlinear')},
                'deferred': dict(self.deferred)}


def trial_id(mass_msun: float, position_yx: tuple[float, float]) -> str:
    y, x = position_yx
    return f'm{mass_msun:.6e}_y{y + 0.0:+.6f}_x{x + 0.0:+.6f}'


def case_id(job_id: str) -> str:
    """Globally specific deterministic basename required by the actual nonlinear API."""
    return 'batch_' + text_digest(job_id)


def _participates(family, arm: PlannedArm) -> bool:
    return family is not None and (family.arms is None or arm.base_name in family.arms)


def _domain(config):
    layout = config.forecast.positions
    centre = config.scene.lens_centre
    if layout.kind == 'grid':
        return grid_positions(centre, spacing_arcsec=layout.spacing_arcsec,
                              half_width_arcsec=layout.half_width_arcsec, annulus=layout.annulus).domain_radius_arcsec
    if layout.kind == 'explicit':
        return explicit_positions(layout.positions_yx, centre).domain_radius_arcsec
    if layout.radius == 'critical_curve':
        return None
    radius = config.scene.einstein_radius() if layout.radius == 'einstein_radius' else layout.radius
    return ring_positions(centre, count=layout.count, radius_arcsec=radius,
                          offset_arcsec=layout.offset_arcsec).domain_radius_arcsec


def _check_position(config, position, path):
    domain = _domain(config)
    if domain is not None and np.hypot(*(np.asarray(position) - config.scene.lens_centre)) > domain + 1e-12:
        raise ConfigError(path, 'trial lies outside the configured forecast domain')


def _reference_arm(plan, family, arm):
    target = arm.base_name if family.forecast_arm is None else family.forecast_arm
    candidates = [item for item in plan.arm_configs if item.member.index == arm.member.index and item.base_name == target]
    if not candidates:
        raise ConfigError(f'nonlinear.{family.name}.forecast_arm', 'forecast arm is absent')
    if len(candidates) == 1 and candidates[0].direction is None:
        return candidates[0]
    fit_count = sum(item.member.index == arm.member.index and item.base_name == arm.base_name
                    for item in plan.arm_configs)
    if arm.direction is None or fit_count != len(candidates):
        raise ConfigError(f'nonlinear.{family.name}.forecast_arm', 'expanded fit/forecast arms need equal direction counts')
    matching = [item for item in candidates if item.direction == arm.direction]
    if len(matching) != 1:
        raise ConfigError(f'nonlinear.{family.name}.forecast_arm', 'expanded arms require a matching direction')
    return matching[0]


def _job(plan, arm, kind, suffix, parameters, seeds):
    job_id = f'members/{arm.member.run_name}/{arm.name}/{suffix}'
    engine = None if kind == 'simulate' else plan.spec.execution.forecast.engine
    return Job(job_id, kind, arm.member, arm.name, arm.config, arm.config_digest,
               None if kind == 'simulate' else arm.config_digest, parameters, seeds, engine)


def _nonlinear_jobs(plan, arm, family, mass, position, *, attempt=0, resolved_trial=None, retry_source=None):
    reference = _reference_arm(plan, family, arm)
    identifier = 'configured' if position is None else trial_id(mass, position)
    sampler, refine = family.sampler, family.refine
    if attempt:
        sampler = SamplerSettings.from_mapping({**sampler.to_mapping(), **family.retry.sampler})
        if refine is not None:
            refine = RefineSettings.from_mapping({**refine.to_mapping(), **family.retry.refine})
    parameters = {'family': family.name, 'mass_msun': mass, 'position_yx': position,
                  'inject': family.inject, 'noise': family.noise, 'attempt': attempt,
                  'fit': family.fit.to_mapping(), 'sampler': sampler.to_mapping(),
                  'refine': None if refine is None else refine.to_mapping(),
                  'retry': None if family.retry is None else family.retry.to_mapping(),
                  'forecast_arm': reference.name, 'forecast_config_digest': reference.config_digest,
                  'forecast_config_mapping': reference.config.to_mapping()}
    if position is None:
        parameters.update(configured_trial=True, batch_seed=plan.spec.seed, member_index=arm.member.index,
                          sampler_arm=arm.name, resolved_trial=resolved_trial)
    if retry_source is not None:
        parameters.update(retry_source)
    for replicate in range(family.replicates):
        seed_identifier = identifier if resolved_trial is None else resolved_trial['trial_id']
        seeds = {} if position is None and resolved_trial is None else {
                 'noise': derived_seed(plan.spec.seed, f'batch/nonlinear/{family.name}/noise/{seed_identifier}',
                                       arm.member.index, replicate) if family.noise else None,
                 'sampler': derived_seed(plan.spec.seed, f'batch/nonlinear/{family.name}/sampler/{arm.name}/{seed_identifier}',
                                         arm.member.index, replicate, attempt)}
        yield _job(plan, arm, 'nonlinear', f'nonlinear/{family.name}/{identifier}/r{replicate:03d}/a{attempt}',
                   {**parameters, 'replicate': replicate}, seeds)


def plan_batch(spec: BatchSpec) -> BatchPlan:
    base_identity = spec.base.capture_identity()
    manifest = dict(base_identity['file_digests'])
    def config_identity(config):
        record = config.capture_identity()
        for path, digest in record['file_digests'].items():
            if path in manifest and manifest[path] != digest:
                raise ConfigError('config', f'the same referenced file changed during planning: {path}')
            manifest[path] = digest
        return record['config_digest']
    population_digest = None
    if spec.population is None:
        base = spec.base
        members = (MemberRecord(0, base.run_name, base.seed, 0, {}, base, base_identity['config_digest']),)
    else:
        population = spec.population
        members = tuple(MemberRecord(member.index, member.run_name, member.seed, member.attempt,
                                     member.values, member.config, config_identity(member.config), member.catalog_sha256,
                                     population.spec.captured_digest(catalog_sha256=member.catalog_sha256))
                        for member in iter_population_members(spec.base, population.spec, population.count,
                            seed=population.seed, start=population.start, name_prefix=population.name_prefix,
                            base_dir=spec.base_dir))
        captured = {member.population_digest for member in members}
        if len(captured) != 1:
            raise ConfigError('population', 'one planned cohort must have one captured population identity')
        population_digest = captured.pop()
        if population.spec.catalog is not None:
            path = str(population.spec.catalog.path)
            digest = members[0].catalog_sha256
            if path in manifest and manifest[path] != digest:
                raise ConfigError('population.catalog', f'catalog snapshot conflicts with another planned input: {path}')
            manifest[path] = digest
    configurations = []
    for member in members:
        for arm in spec.arms:
            config = member.config.replace(arm.overrides, base_dir=spec.base_dir)
            if arm.directions is not None and config.psf.model.kind != 'knowledge_error':
                raise ConfigError(f'arms.{arm.name}.directions', 'requires a knowledge_error model PSF')
            for direction in (None,) if arm.directions is None else range(1, arm.directions + 1):
                effective = config if direction is None else config.replace({'psf': {'model': {'draw': {
                    'seed': derived_seed(spec.seed, 'batch/psf.model/direction', member.index, direction)}}}})
                name = arm.name if direction is None else f'{arm.name}/d{direction}'
                if (spec.forecast is not None or spec.nonlinear) and effective.forecast is None:
                    raise ConfigError(f'members.{member.run_name}.arms.{name}.forecast', 'required for forecast/nonlinear jobs')
                configurations.append(PlannedArm(member, name, arm.name, direction, effective, config_identity(effective)))
    spec_digest = spec.captured_digest(base_config_digest=base_identity['config_digest'], population_digest=population_digest)
    plan = BatchPlan(spec, members, (), {}, tuple(configurations), spec_digest, population_digest, manifest)
    member_indices = {member.index for member in members}
    for family in spec.nonlinear:
        for index, trial in enumerate(family.trials.explicit):
            if trial.members != 'all' and not set(trial.members) <= member_indices:
                raise ConfigError(f'nonlinear.{family.name}.trials.explicit[{index}].members', 'member index outside population')
    jobs, deferred = [], {}
    for arm in configurations:
        if _participates(spec.simulate, arm):
            family = spec.simulate
            if family.inject and arm.config.scene.injection is None:
                raise ConfigError('simulate.inject', 'requires scene.injection in every participating configuration')
            for replicate in range(family.replicates):
                seed = derived_seed(spec.seed, 'batch/simulate/noise', arm.member.index, replicate) if family.noise else None
                jobs.append(_job(plan, arm, 'simulate', f'simulate/r{replicate:03d}',
                                 {'noise': family.noise, 'inject': family.inject, 'replicate': replicate}, {'noise': seed}))
        if _participates(spec.forecast, arm):
            jobs.append(_job(plan, arm, 'forecast', 'forecast', {'masses_msun': spec.forecast.masses_msun}, {}))
        for family in spec.nonlinear:
            if not _participates(family, arm):
                continue
            reference = _reference_arm(plan, family, arm)
            if family.trials.kind.startswith('forecast_'):
                if not _participates(spec.forecast, reference) or not set(family.trials.masses_msun) <= set(spec.forecast.masses_msun):
                    raise ConfigError(f'nonlinear.{family.name}.trials', 'forecast trials require this arm and every mass in the forecast family')
                deferred[family.name] = 'after forecast'
                continue
            if family.retry is not None:
                deferred[family.name] = 'after first attempt'
            if family.trials.kind == 'configured':
                placement = arm.config.scene.injection
                if placement is None:
                    raise ConfigError(f'nonlinear.{family.name}.trials', 'configured trials require scene.injection')
                if getattr(placement.position, 'radius', None) == 'critical_curve':
                    trials = [(placement.mass_msun, None)]
                else:
                    halo = configured_injection(arm.config.scene, Cosmology(arm.config.cosmology), seed=arm.config.seed)
                    trials = [(halo.mass_msun, halo.position_yx_arcsec)]
            else:
                trials = [(trial.mass_msun, trial.position_yx) for trial in family.trials.explicit
                          if trial.members == 'all' or arm.member.index in trial.members]
            for mass, position in trials:
                if position is not None:
                    _check_position(arm.config, position, f'nonlinear.{family.name}.trials')
                    _check_position(reference.config, position, f'nonlinear.{family.name}.forecast_arm')
                jobs.extend(_nonlinear_jobs(plan, arm, family, mass, position))
    seen = set()
    for job in jobs:
        if job.job_id in seen:
            raise ConfigError('nonlinear.trials', f'two jobs have the same rounded trial identity: {job.job_id}')
        seen.add(job.job_id)
    return BatchPlan(spec, members, tuple(jobs), deferred, tuple(configurations), spec_digest, population_digest, manifest)


def follow_ups(plan: BatchPlan, job: Job, output_dir: Path) -> tuple[Job, ...]:
    job_dir = Path(output_dir) / job.job_id
    marker = read_marker(job_dir)
    if marker is None:
        raise BatchError(f'completed marker is required for follow-ups of {job.job_id}')
    if marker['job_digest'] != job.digest():
        raise BatchConflict(f'winning marker has a different job digest: {job.job_id}')
    verify_marker(job_dir, marker, digests=False)
    if job.kind == 'forecast':
        result = load_forecast(job_dir / marker['artifacts']['forecast']['path'])
        jobs = []
        for arm in plan.arm_configs:
            if arm.member.index != job.member.index:
                continue
            for family in plan.spec.nonlinear:
                if not _participates(family, arm) or not family.trials.kind.startswith('forecast_'):
                    continue
                if _reference_arm(plan, family, arm).name != job.arm:
                    continue
                selection = np.ones(len(result.positions_yx), dtype=bool)
                if family.trials.aperture_radius_arcsec is not None:
                    selection = aperture_selection(result, centre_yx=arm.config.scene.lens_centre,
                                                   radius_arcsec=family.trials.aperture_radius_arcsec)
                for mass in family.trials.masses_msun:
                    row = list(result.masses_msun).index(mass)
                    if family.trials.kind == 'forecast_positions':
                        indices = range(len(result.positions_yx))
                    else:
                        metric = getattr(result, result.detection_metric)[row]
                        eligible = selection & np.isfinite(metric)
                        if result.detection_metric == 'q_mismatch':
                            eligible &= result.amplitude_hat[row] > 0
                        if not np.any(eligible):
                            raise BatchError(f'no eligible forecast_argmax node for {job.job_id}, mass {mass}')
                        indices = (int(np.argmax(np.where(eligible, metric, -np.inf))),)
                    for index in indices:
                        position = tuple(float(value) for value in result.positions_yx[index])
                        _check_position(arm.config, position, f'nonlinear.{family.name}.trials')
                        jobs.extend(_nonlinear_jobs(plan, arm, family, mass, position))
        identifiers = [followup.job_id for followup in jobs]
        if len(identifiers) != len(set(identifiers)):
            raise BatchError(f'forecast positions have colliding rounded trial identities for {job.job_id}')
        return tuple(jobs)
    if job.kind == 'nonlinear' and job.parameters['attempt'] == 0:
        family = next(family for family in plan.spec.nonlinear if family.name == job.parameters['family'])
        if family.retry is None:
            return ()
        retry = family.retry
        raw_verdict = marker['summary'].get('retry')
        artifact = marker['artifacts']['case']
        case_path = job_dir / artifact['path']
        if not isinstance(raw_verdict, Mapping):
            raise BatchConflict(f'missing typed retry verdict: {job.job_id}')
        verdict = RetryVerdict.from_mapping(raw_verdict)
        if (verdict.policy_digest != mapping_digest(retry.to_mapping())
                or verdict.case_sha256 != artifact['sha256'] or file_digest(case_path) != artifact['sha256']):
            raise BatchConflict(f'retry verdict or its validated case changed: {job.job_id}')
        if verdict.status.status == 'accepted':
            return ()
        arm = next(arm for arm in plan.arm_configs if arm.member.index == job.member.index and arm.name == job.arm)
        unresolved = job.parameters['position_yx'] is None
        resolved_trial = marker['summary'].get('trial') if unresolved else None
        if unresolved and (not isinstance(resolved_trial, Mapping) or
                           resolved_trial.get('mass_msun') != job.parameters['mass_msun']):
            raise BatchConflict(f'missing or inconsistent resolved trial for {job.job_id}')
        source = {'retry_case_path': str(case_path.resolve()), 'retry_case_sha256': artifact['sha256'],
                  'retry_source_job_id': job.job_id}
        return tuple(followup for followup in _nonlinear_jobs(plan, arm, family, job.parameters['mass_msun'],
                           None if unresolved else tuple(job.parameters['position_yx']), attempt=1,
                           resolved_trial=resolved_trial, retry_source=source)
                     if followup.parameters['replicate'] == job.parameters['replicate'])
    return ()
