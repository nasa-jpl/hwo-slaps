"""Typed batch inputs, reusing the configuration, population and fit schemas."""
from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from ..analysis.nonlinear import RoleAcceptance
from ..config.checks import (Boolean, ConfigError, Integer, Key, ListOf, MapOf, Nullable,
                             Pair, Real, Table, Text, Union, Variants)
from ..config.loading import read_yaml
from ..config.schema import EngineConfig, ROOT_TABLE, compose_config, resolve_config
from ..fisher.api import Execution
from ..identity import json_ready, mapping_digest
from ..inference.result import RoleStatus
from ..inference.settings import FitSpec, RefineSettings, SamplerSettings
from ..population.sampling import POPULATION_TABLE, PopulationSpec


_POSITIVE = Real(min=0., min_open=True)
_COUNT = Integer(min=1)
_NAME = Text(pattern=r'[A-Za-z0-9_.-]+')
_ARMS = Nullable(ListOf(_NAME, min_length=1, unique=True))
_MASSES = ListOf(_POSITIVE, min_length=1, unique=True)


def _native_keys(native, normalized):
    """Keep normalized portable leaves without coercing scientific mapping keys."""
    if isinstance(native, Mapping):
        return {(key if isinstance(key, str) else int(key)): _native_keys(
                    value, normalized[key if isinstance(key, str) else str(int(key))])
                for key, value in native.items()}
    if isinstance(native, (list, tuple)):
        return [_native_keys(value, normalized[index]) for index, value in enumerate(native)]
    return normalized


@dataclass(frozen=True)
class _Mapping:
    def accepts(self, value):
        return isinstance(value, Mapping)

    def describe(self):
        return 'mapping'

    def __call__(self, value, path):
        if not self.accepts(value):
            raise ConfigError(path, 'must be a mapping')
        return deepcopy(dict(value))


@dataclass(frozen=True)
class _Settings:
    kind: type

    def accepts(self, value):
        return isinstance(value, Mapping)

    def describe(self):
        return 'mapping'

    def __call__(self, value, path):
        return self.kind.from_mapping(value, path=path).to_mapping()


@dataclass(frozen=True)
class BatchExecution:
    forecast: Execution = field(default_factory=Execution)
    devices: Literal['cpu'] | tuple[int, ...] = 'cpu'
    workers_per_device: int = 1
    threads_per_worker: int = 1
    preparation_cache_size: int = 2
    memory_fraction: float = .75
    training_workers: int = 1

    def __post_init__(self):
        if not isinstance(self.forecast, Execution):
            raise ConfigError('execution', 'forecast must be an Execution')
        values = _EXECUTION_TABLE.read(self.to_mapping(), 'execution')
        object.__setattr__(self, 'devices', 'cpu' if values['devices'] == 'cpu' else tuple(values['devices']))

    @classmethod
    def from_mapping(cls, mapping, *, path='execution'):
        values = _EXECUTION_TABLE.read(mapping, path)
        forecast = Execution(values.pop('engine'), values.pop('reference_workers'), values.pop('batch_size'), False)
        return cls(forecast=forecast, **values)

    def to_mapping(self):
        return {'engine': self.forecast.engine, 'reference_workers': self.forecast.reference_workers,
                'batch_size': self.forecast.batch_size, 'devices': self.devices,
                'workers_per_device': self.workers_per_device, 'threads_per_worker': self.threads_per_worker,
                'preparation_cache_size': self.preparation_cache_size, 'memory_fraction': self.memory_fraction,
                'training_workers': self.training_workers}


_EXECUTION_TABLE = Table((
    Key('engine', Text(choices=('reference', 'jax')), 'forecast engine', 'reference'),
    Key('reference_workers', _COUNT, 'reference workers inside each batch worker', 1),
    Key('batch_size', _COUNT, 'JAX position batch size', 16),
    Key('devices', Union(Text(choices=('cpu',)), ListOf(Integer(min=0), min_length=1, unique=True)),
        'CPU or indices of the parent visible devices', 'cpu'),
    Key('workers_per_device', _COUNT, 'workers sharing each assigned device', 1),
    Key('threads_per_worker', _COUNT, 'BLAS threads per batch worker', 1),
    Key('preparation_cache_size', _COUNT, 'bounded preparations retained per worker', 2),
    Key('memory_fraction', Real(min=0., min_open=True, max=1.), 'total preallocated fraction per device', .75),
    Key('training_workers', _COUNT, 'backend emulator-training workers per batch worker', 1),
))


@dataclass(frozen=True)
class Arm:
    name: str
    overrides: Mapping[str, Any]
    directions: int | None = None

    def to_mapping(self):
        return {'name': self.name, 'overrides': deepcopy(dict(self.overrides)), 'directions': self.directions}


@dataclass(frozen=True)
class PopulationBlock:
    spec: PopulationSpec
    seed: int
    count: int
    start: int
    name_prefix: str

    def to_mapping(self):
        return {**self.spec.to_mapping(), 'seed': self.seed, 'count': self.count,
                'start': self.start, 'name_prefix': self.name_prefix}


@dataclass(frozen=True)
class SimulateFamily:
    inject: bool
    noise: bool
    replicates: int = 1
    arms: tuple[str, ...] | None = None

    def to_mapping(self):
        return {'inject': self.inject, 'noise': self.noise, 'replicates': self.replicates, 'arms': self.arms}


@dataclass(frozen=True)
class ForecastFamily:
    masses_msun: tuple[float, ...]
    arms: tuple[str, ...] | None = None

    def to_mapping(self):
        return {'masses_msun': self.masses_msun, 'arms': self.arms}


@dataclass(frozen=True)
class ExplicitTrial:
    members: Literal['all'] | tuple[int, ...]
    mass_msun: float
    position_yx: tuple[float, float]

    def to_mapping(self):
        return {'members': self.members, 'mass_msun': self.mass_msun, 'position_yx': self.position_yx}


@dataclass(frozen=True)
class TrialSelector:
    kind: Literal['explicit', 'configured', 'forecast_positions', 'forecast_argmax']
    explicit: tuple[ExplicitTrial, ...] = ()
    masses_msun: tuple[float, ...] = ()
    aperture_radius_arcsec: float | None = None

    def to_mapping(self):
        if self.kind == 'explicit':
            return {'kind': self.kind, 'explicit': [trial.to_mapping() for trial in self.explicit]}
        if self.kind == 'configured':
            return {'kind': self.kind}
        return {'kind': self.kind, 'masses_msun': self.masses_msun, **(
            {'aperture_radius_arcsec': self.aperture_radius_arcsec} if self.kind == 'forecast_argmax' else {})}


@dataclass(frozen=True)
class RetryPolicy:
    acceptance: RoleAcceptance
    require_retained_state: bool
    stationarity_tolerance: float | None
    sampler: Mapping[str, Any]
    refine: Mapping[str, Any]

    def to_mapping(self):
        return {'acceptance': self.acceptance.to_mapping(), 'require_retained_state': self.require_retained_state,
                'stationarity_tolerance': self.stationarity_tolerance, 'sampler': dict(self.sampler), 'refine': dict(self.refine)}


@dataclass(frozen=True)
class NonlinearFamily:
    name: str
    trials: TrialSelector
    inject: bool
    noise: bool
    replicates: int
    fit: FitSpec
    sampler: SamplerSettings
    refine: RefineSettings | None
    retry: RetryPolicy | None
    arms: tuple[str, ...] | None = None
    forecast_arm: str | None = None

    def to_mapping(self):
        return {'arms': self.arms, 'forecast_arm': self.forecast_arm, 'trials': self.trials.to_mapping(),
                'inject': self.inject, 'noise': self.noise, 'replicates': self.replicates,
                'fit': self.fit.to_mapping(), 'sampler': self.sampler.to_mapping(),
                'refine': None if self.refine is None else self.refine.to_mapping(),
                'retry': None if self.retry is None else self.retry.to_mapping()}


@dataclass(frozen=True)
class BatchSpec:
    name: str
    seed: int
    base: EngineConfig
    base_dir: Path
    population: PopulationBlock | None
    arms: tuple[Arm, ...]
    simulate: SimulateFamily | None
    forecast: ForecastFamily | None
    nonlinear: tuple[NonlinearFamily, ...]
    execution: BatchExecution

    def to_mapping(self):
        native = {'name': self.name, 'seed': self.seed, 'config': self.base.to_mapping(), 'overrides': {},
                           'population': None if self.population is None else self.population.to_mapping(),
                           'arms': [arm.to_mapping() for arm in self.arms],
                           'simulate': None if self.simulate is None else self.simulate.to_mapping(),
                           'forecast': None if self.forecast is None else self.forecast.to_mapping(),
                           'nonlinear': {family.name: family.to_mapping() for family in self.nonlinear},
                           'execution': self.execution.to_mapping()}
        return _native_keys(native, json_ready(native))

    def digest(self):
        return self.captured_digest(base_config_digest=self.base.digest(),
            population_digest=None if self.population is None else self.population.spec.digest())

    def captured_digest(self, *, base_config_digest, population_digest):
        """Identity from the inputs captured by a plan, without reopening their paths."""
        return mapping_digest({'spec': self.to_mapping(), 'config_digest': base_config_digest,
                               'population_digest': population_digest})


_ARM_TABLE = Table((Key('name', _NAME, 'arm name'), Key('overrides', _Mapping(), 'configuration overrides', {}),
                    Key('directions', Nullable(_COUNT), 'paired knowledge-error directions', None)))
_POPULATION_TABLE = POPULATION_TABLE.extend((
    Key('seed', Integer(min=0), 'population stream seed'), Key('count', _COUNT, 'member count'),
    Key('start', Integer(min=0), 'first absolute population index', 0),
    Key('name_prefix', _NAME, 'population member name prefix', 'system'),
))
_SIMULATE_TABLE = Table((Key('inject', Boolean(), 'inject the configured scene hypothesis'),
                        Key('noise', Boolean(), 'draw detector noise'),
                        Key('replicates', _COUNT, 'independent noise replicates', 1),
                        Key('arms', _ARMS, 'participating arm names', None)))
_FORECAST_TABLE = Table((Key('masses_msun', _MASSES, 'forecast masses'),
                        Key('arms', _ARMS, 'participating arm names', None)))
_EXPLICIT_TABLE = Table((Key('members', Union(Text(choices=('all',)), Integer(min=0),
                                            ListOf(Integer(min=0), min_length=1, unique=True)), 'member indices', 'all'),
                         Key('mass_msun', _POSITIVE, 'trial halo mass'),
                         Key('position_yx', Pair(Real()), 'trial position in arcseconds')))
_TRIALS_TABLE = Variants('kind', {
    'explicit': Table((Key('explicit', ListOf(_EXPLICIT_TABLE, min_length=1), 'trials'),)),
    'configured': Table(()),
    'forecast_positions': Table((Key('masses_msun', _MASSES, 'trial masses present in forecast'),)),
    'forecast_argmax': Table((Key('masses_msun', _MASSES, 'trial masses present in forecast'),
                              Key('aperture_radius_arcsec', Nullable(_POSITIVE), 'closed selection disc', None))),
})
_RETRY_TABLE = Table((
    Key('acceptance', Table(tuple(Key(role, ListOf(Text(choices=tuple(value.value for value in RoleStatus)),
                                                  min_length=1, unique=True), 'accepted role outcomes')
                                 for role in ('smooth', 'subhalo'))), 'accepted outcomes'),
    Key('require_retained_state', Boolean(), 'require searched roles to retain backend state'),
    Key('stationarity_tolerance', Nullable(_POSITIVE), 'None is the paper rule; positive bounds projected gradient'),
    Key('sampler', _Mapping(), 'sampler overrides for attempt1', {}),
    Key('refine', _Mapping(), 'refinement overrides for attempt1', {}),
))
_NONLINEAR_TABLE = Table((
    Key('arms', _ARMS, 'participating arm names', None),
    Key('forecast_arm', Nullable(_NAME), 'arm supplying trials and signed forecast references', None),
    Key('trials', _TRIALS_TABLE, 'trial selection'), Key('inject', Boolean(), 'inject the trial'),
    Key('noise', Boolean(), 'draw detector noise'), Key('replicates', _COUNT, 'noise replicates', 1),
    Key('fit', _Settings(FitSpec), 'nonlinear fit'), Key('sampler', _Settings(SamplerSettings), 'sampler settings', {}),
    Key('refine', Nullable(_Settings(RefineSettings)), 'bounded refinement', None),
    Key('retry', Nullable(_RETRY_TABLE), 'one follow-up attempt', None),
))
BATCH_TABLE = Table((
    Key('name', _NAME, 'batch name'), Key('seed', Integer(min=0), 'batch noise/sampler/direction entropy'),
    Key('config', Union(ROOT_TABLE, Text(), ListOf(Text(), min_length=1)), 'configuration files or effective mapping'),
    Key('overrides', _Mapping(), 'base configuration overlay', {}),
    Key('population', Nullable(_POPULATION_TABLE), 'member population', None),
    Key('arms', ListOf(_ARM_TABLE, min_length=1), 'configuration arms', [{'name': 'base'}]),
    Key('simulate', Nullable(_SIMULATE_TABLE), 'simulation family', None),
    Key('forecast', Nullable(_FORECAST_TABLE), 'forecast family', None),
    Key('nonlinear', MapOf(_NAME, _NONLINEAR_TABLE), 'named nonlinear families', {}),
    Key('execution', _EXECUTION_TABLE, 'worker placement and runtime settings', {}),
))


def _arms(values):
    return None if values is None else tuple(values)


def parse_batch(mapping: Mapping[str, Any], *, base_dir) -> BatchSpec:
    directory = Path(base_dir).expanduser().resolve()
    if isinstance(mapping, Mapping) and isinstance(mapping.get('config'), Mapping):
        def resolve_path(filename, check):
            path = Path(filename).expanduser()
            return str((path if path.is_absolute() else directory / path).resolve())
        mapping = {**mapping, 'config': ROOT_TABLE.transform_paths(mapping['config'], resolve_path)}
    values = BATCH_TABLE.read(mapping, '')
    source = values['config']
    if isinstance(source, Mapping):
        composed = ROOT_TABLE.merge(source, values['overrides'])
    else:
        paths = [source] if isinstance(source, str) else source
        paths = [Path(path).expanduser() if Path(path).expanduser().is_absolute()
                 else directory / Path(path).expanduser() for path in paths]
        composed = compose_config(paths, overrides=values['overrides'], base_dir=directory)
    base = resolve_config(composed, base_dir=directory)
    arms = tuple(Arm(**record) for record in values['arms'])
    names = [arm.name for arm in arms]
    if len(set(names)) != len(names):
        raise ConfigError('arms', 'arm names must be unique')
    for index, arm in enumerate(arms):
        if arm.name in ('.', '..'):
            raise ConfigError(f'arms[{index}].name', 'must be a safe path component')
        for reserved in ('run_name', 'seed'):
            if reserved in arm.overrides:
                raise ConfigError(f'arms[{index}].overrides.{reserved}', 'member identity cannot be overridden')
    population = None
    if values['population'] is not None:
        record = dict(values['population'])
        controls = {key: record.pop(key) for key in ('seed', 'count', 'start', 'name_prefix')}
        population = PopulationBlock(PopulationSpec.from_mapping(record, base_dir=directory), **controls)
    simulate = None if values['simulate'] is None else SimulateFamily(**{**values['simulate'], 'arms': _arms(values['simulate']['arms'])})
    forecast = None if values['forecast'] is None else ForecastFamily(tuple(values['forecast']['masses_msun']), _arms(values['forecast']['arms']))
    families = []
    for name, record in values['nonlinear'].items():
        if name in ('.', '..'):
            raise ConfigError(f'nonlinear.{name}', 'must be a safe path component')
        selector = record['trials']
        explicit = tuple(ExplicitTrial('all' if trial['members'] == 'all' else (
            (trial['members'],) if isinstance(trial['members'], int) else tuple(trial['members'])),
            trial['mass_msun'], tuple(trial['position_yx'])) for trial in selector.get('explicit', ()))
        trials = TrialSelector(selector['kind'], explicit, tuple(selector.get('masses_msun', ())), selector.get('aperture_radius_arcsec'))
        fit = FitSpec.from_mapping(record['fit'], path=f'nonlinear.{name}.fit')
        sampler = SamplerSettings.from_mapping(record['sampler'], path=f'nonlinear.{name}.sampler')
        refine = None if record['refine'] is None else RefineSettings.from_mapping(record['refine'], path=f'nonlinear.{name}.refine')
        retry = None
        if record['retry'] is not None:
            retry_record = dict(record['retry'])
            retry_record['acceptance'] = RoleAcceptance.from_mapping(retry_record['acceptance'], path=f'nonlinear.{name}.retry.acceptance')
            retry = RetryPolicy(**retry_record)
            retry_sampler = SamplerSettings.from_mapping({**sampler.to_mapping(), **retry.sampler}, path=f'nonlinear.{name}.retry.sampler')
            if refine is None and retry.refine:
                raise ConfigError(f'nonlinear.{name}.retry.refine', 'requires a family refine block')
            if refine is not None:
                RefineSettings.from_mapping({**refine.to_mapping(), **retry.refine}, path=f'nonlinear.{name}.retry.refine')
                if not retry_sampler.use_jax:
                    raise ConfigError(f'nonlinear.{name}.retry.sampler.use_jax', 'refinement requires use_jax')
        if refine is not None and not sampler.use_jax:
            raise ConfigError(f'nonlinear.{name}.refine', 'requires sampler.use_jax')
        families.append(NonlinearFamily(name, trials, record['inject'], record['noise'], record['replicates'],
                                        fit, sampler, refine, retry, _arms(record['arms']), record['forecast_arm']))
    if simulate is None and forecast is None and not families:
        raise ConfigError('', 'at least one of simulate, forecast or nonlinear is required')
    for path, family in [('simulate', simulate), ('forecast', forecast)] + [(f'nonlinear.{family.name}', family) for family in families]:
        if family is None:
            continue
        if family.arms is not None and not set(family.arms) <= set(names):
            raise ConfigError(f'{path}.arms', 'must name configured arms')
        if isinstance(family, (SimulateFamily, NonlinearFamily)) and not family.noise and family.replicates != 1:
            raise ConfigError(f'{path}.replicates', 'must be1 when noise is false')
        if isinstance(family, NonlinearFamily) and family.forecast_arm is not None and family.forecast_arm not in names:
            raise ConfigError(f'{path}.forecast_arm', 'must name a configured arm')
    return BatchSpec(values['name'], values['seed'], base, directory, population, arms, simulate, forecast,
                     tuple(families), BatchExecution.from_mapping(values['execution']))


def load_batch_spec(path) -> BatchSpec:
    filename = Path(path).expanduser().resolve()
    return parse_batch(read_yaml(filename), base_dir=filename.parent)
