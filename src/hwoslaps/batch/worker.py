"""One owned subprocess: bounded preparation cache, backend session and atomic jobs."""
from __future__ import annotations

from collections import OrderedDict
from contextlib import contextmanager, ExitStack
import json
import logging
from multiprocessing.connection import Client
import os
from pathlib import Path
import sys
import time
import traceback

from ..config.checks import Integer
from ..config.schema import parse_config
from ..identity import file_digest, json_ready, mapping_digest
from ..provenance import capture_provenance
from ..seeding import derived_seed
from .jobs import JobPayload, case_id, trial_id
from .spec import BatchExecution
from .state import (BatchConflict, RetryVerdict, claim_run_dir, publish_marker, read_marker, source_revision,
                    verify_marker, write_failure)


class PreparationCache:
    """LRU of actual preparations, closed before replacement and checked before reuse."""

    def __init__(self, size, execution):
        self.size = Integer(min=1)(size, 'preparation_cache_size')
        self.execution = execution
        self._items = OrderedDict()
        self.prepared_count = 0

    def get(self, config):
        from ..fisher.api import prepare_forecast
        key = config.digest()
        if key in self._items:
            prepared = self._items.pop(key)
            try:
                if prepared.record['config_digest'] != key:
                    raise BatchConflict('cached preparation identity differs from its key')
                prepared.validate_identity()
            except BaseException:
                prepared.close()
                raise
            self._items[key] = prepared
            return prepared, True, 0.
        if len(self._items) == self.size:
            _, old = self._items.popitem(last=False)
            old.close()
        start = time.perf_counter()
        prepared = prepare_forecast(config, execution=self.execution)
        duration = time.perf_counter() - start
        self.prepared_count += 1
        if prepared.record['config_digest'] != key:
            prepared.close()
            raise BatchConflict('actual preparation inputs changed after the cache key was captured')
        self._items[key] = prepared
        return prepared, False, duration

    def close(self):
        with ExitStack() as closers:
            for prepared in self._items.values():
                closers.callback(prepared.close)
            self._items.clear()

    @property
    def keys(self):
        return tuple(self._items)


@contextmanager
def _job_log(path):
    sys.stdout.flush()
    sys.stderr.flush()
    saved = (os.dup(1), os.dup(2))
    handler = logging.FileHandler(path)
    handler.setFormatter(logging.Formatter('%(asctime)s %(levelname)s %(name)s: %(message)s'))
    logger = logging.getLogger()
    previous = logger.level
    logger.setLevel(logging.DEBUG)
    with path.open('ab', buffering=0) as stream:
        os.dup2(stream.fileno(), 1)
        os.dup2(stream.fileno(), 2)
        logger.addHandler(handler)
        try:
            yield
        finally:
            sys.stdout.flush()
            sys.stderr.flush()
            logger.removeHandler(handler)
            handler.close()
            logger.setLevel(previous)
            for descriptor, restore in zip((1, 2), saved, strict=True):
                os.dup2(restore, descriptor)
                os.close(restore)


def _preparation(prepared, hit, duration):
    return {'key': prepared.record['config_digest'], 'cache_hit': hit, 'prepare_s': duration,
            'identity': json_ready(prepared.record)}


def run_job(payload, cache, session, *, connection, worker, revision):
    from ..analysis.nonlinear import RoleAcceptance, case_status
    from ..artifacts import load_case_snapshot, save_case, save_forecast, save_observation
    from ..fisher.api import forecast
    from ..inference.api import validate_nonlinear
    from ..inference.result import ForecastReference
    from ..inference.settings import FitSpec, RefineSettings, SamplerSettings
    from ..scene.cosmology import Cosmology
    from ..scene.subhalo import configured_injection
    from ..simulation import simulate

    job_dir = Path(payload.job_dir)
    run_dir = claim_run_dir(job_dir)
    try:
        connection.send({'type': 'started', 'job_id': payload.job_id, 'run': run_dir.name, **worker})
    except (BrokenPipeError, ConnectionResetError, EOFError):
        # A received job still publishes its result before this orphan exits.
        pass
    start, before = time.perf_counter(), cache.prepared_count
    with _job_log(run_dir / 'job.log'):
        try:
            config = parse_config(payload.config_mapping)
            if config.digest() != payload.config_digest:
                raise BatchConflict(f'configuration bytes changed for {payload.job_id}')
            parameters = payload.parameters
            seeds = dict(payload.seeds)
            summary, preparation, reference_preparation = {}, None, None
            forecast_s, validated_case_sha = None, None
            if payload.kind == 'simulate':
                halo = configured_injection(config.scene, Cosmology(config.cosmology), seed=config.seed) if parameters['inject'] else None
                result = simulate(config, subhalo=halo, noise_seed=seeds['noise'])
                if result.config_digest != payload.config_digest:
                    raise BatchConflict(f'actual simulated inputs differ from the planned job: {payload.job_id}')
                artifacts = {'observation': save_observation(result, run_dir / 'observation.npz')}
            else:
                prepared, hit, duration = cache.get(config)
                if prepared.record['config_digest'] != payload.config_digest:
                    raise BatchConflict(f'actual preparation differs from the planned job: {payload.job_id}')
                preparation = _preparation(prepared, hit, duration)
                if payload.kind == 'forecast':
                    forecast_start = time.perf_counter()
                    result = forecast(prepared, masses_msun=parameters['masses_msun'])
                    if result.provenance['config_digest'] != payload.config_digest:
                        raise BatchConflict(f'actual forecast inputs differ from the planned job: {payload.job_id}')
                    forecast_s = time.perf_counter() - forecast_start
                    artifacts = {'forecast': save_forecast(result, run_dir / 'forecast.npz')}
                elif payload.kind == 'nonlinear':
                    if parameters.get('configured_trial'):
                        resolved = configured_injection(config.scene, prepared.scene.cosmology, seed=config.seed)
                        actual_trial = {'mass_msun': resolved.mass_msun, 'position_yx': resolved.position_yx_arcsec,
                                        'trial_id': trial_id(resolved.mass_msun, resolved.position_yx_arcsec)}
                        carried = parameters['resolved_trial']
                        if carried is not None and json_ready(carried) != json_ready(actual_trial):
                            raise BatchConflict(f'configured retry resolved to another physical trial: {payload.job_id}')
                        if resolved.mass_msun != parameters['mass_msun']:
                            raise BatchConflict(f'configured trial mass changed: {payload.job_id}')
                        hypothesis = prepared.hypothesis(resolved.mass_msun, resolved.position_yx_arcsec)
                        replicate, member = parameters['replicate'], parameters['member_index']
                        identifier = actual_trial['trial_id']
                        # The blueprint remains immutable; only actual run seeds are resolved here.
                        actual_seeds = {'noise': derived_seed(parameters['batch_seed'],
                            f"batch/nonlinear/{parameters['family']}/noise/{identifier}", member, replicate)
                            if parameters['noise'] else None,
                            'sampler': derived_seed(parameters['batch_seed'],
                            f"batch/nonlinear/{parameters['family']}/sampler/{parameters['sampler_arm']}/{identifier}",
                            member, replicate, parameters['attempt'])}
                        if seeds and seeds != actual_seeds:
                            raise BatchConflict(f'configured retry seeds differ from the carried trial: {payload.job_id}')
                        seeds = actual_seeds
                    else:
                        hypothesis = prepared.hypothesis(parameters['mass_msun'], tuple(parameters['position_yx']))
                        actual_trial = {'mass_msun': hypothesis.mass_msun, 'position_yx': hypothesis.position_yx_arcsec,
                                        'trial_id': trial_id(hypothesis.mass_msun, hypothesis.position_yx_arcsec)}
                    if parameters['attempt']:
                        first_path = Path(parameters['retry_case_path'])
                        first, first_sha = load_case_snapshot(first_path)
                        if first_sha != parameters['retry_case_sha256']:
                            raise BatchConflict(f'first retry case artifact changed: {first_path}')
                        if (first.case_id != case_id(parameters['retry_source_job_id']) or first.hypothesis != hypothesis
                                or first.observation.config_digest != payload.config_digest
                                or first.observation.noise_seed != seeds['noise']):
                            raise BatchConflict(f'retry is not the same physical case: {payload.job_id}')
                    reference_config = parse_config(parameters['forecast_config_mapping'])
                    if reference_config.digest() != parameters['forecast_config_digest']:
                        raise BatchConflict(f'forecast reference bytes changed for {payload.job_id}')
                    reference, reference_hit, reference_duration = cache.get(reference_config)
                    if reference.record['config_digest'] != parameters['forecast_config_digest']:
                        raise BatchConflict(f'actual reference preparation differs from its plan: {payload.job_id}')
                    reference_preparation = _preparation(reference, reference_hit, reference_duration)
                    q = forecast(reference, masses_msun=[hypothesis.mass_msun], positions=[hypothesis.position_yx_arcsec])
                    if q.provenance['config_digest'] != parameters['forecast_config_digest']:
                        raise BatchConflict(f'actual reference forecast differs from its plan: {payload.job_id}')
                    forecast_reference = ForecastReference.from_result(q, mass_index=0, position_index=0)
                    # A size-one cache may have evicted the fit preparation while forecasting.
                    if reference is not prepared:
                        prepared, restored_hit, restored_duration = cache.get(config)
                        if prepared.record['config_digest'] != payload.config_digest:
                            raise BatchConflict(f'restored fit preparation differs from its plan: {payload.job_id}')
                        preparation = _preparation(prepared, hit and restored_hit, duration + restored_duration)
                    observation = simulate(prepared, subhalo=hypothesis if parameters['inject'] else None,
                                           noise_seed=seeds['noise'])
                    if observation.config_digest != payload.config_digest:
                        raise BatchConflict(f'actual nonlinear observation differs from its plan: {payload.job_id}')
                    refine = parameters['refine']
                    result = validate_nonlinear(prepared, hypothesis, observation,
                        fit=FitSpec.from_mapping(parameters['fit']), sampler=SamplerSettings.from_mapping(parameters['sampler']),
                        sampler_seed=seeds['sampler'], output_dir=run_dir / 'fit',
                        refine=None if refine is None else RefineSettings.from_mapping(refine), session=session,
                        forecast_reference=forecast_reference, case_id=case_id(payload.job_id))
                    if (result.observation.config_digest != payload.config_digest
                            or result.forecast_reference.config_digest != parameters['forecast_config_digest']):
                        raise BatchConflict(f'actual case input identities differ from their plans: {payload.job_id}')
                    artifacts = {'case': save_case(result, run_dir / 'case.json')}
                    summary = {'role_statuses': {role: result.role(role).acceptance_status.value for role in ('smooth', 'subhalo')},
                               'q_signed': result.q_signed, 'forecast_q': forecast_reference.q, 'trial': actual_trial}
                    retry = parameters['retry']
                    if retry is not None:
                        # Keep the full physical archive validator in this backend-owning process.
                        validated, validated_case_sha = load_case_snapshot(artifacts['case'])
                        status = case_status(validated, acceptance=RoleAcceptance.from_mapping(retry['acceptance']),
                                             require_retained_state=retry['require_retained_state'],
                                             stationarity_tolerance=retry['stationarity_tolerance'])
                        summary['retry'] = RetryVerdict(status, mapping_digest(retry), validated_case_sha).to_mapping()
                else:
                    raise ValueError(f'unknown job kind {payload.kind!r}')
            records = {name: {'path': str(path.relative_to(job_dir)), 'bytes': path.stat().st_size,
                              'sha256': validated_case_sha if name == 'case' and validated_case_sha is not None
                              else file_digest(path)} for name, path in artifacts.items()}
            marker = {'schema': 1, 'job_id': payload.job_id, 'job_digest': payload.job_digest, 'kind': payload.kind,
                      'run': run_dir.name, 'artifacts': records, 'seeds': seeds, 'session': payload.session,
                      'worker': worker, 'source_revision': revision, 'timing': {'run_s': time.perf_counter() - start},
                      'summary': summary}
            if preparation is not None:
                marker['preparation'] = preparation
            if reference_preparation is not None:
                marker['forecast_reference_preparation'] = reference_preparation
            if forecast_s is not None:
                marker['timing']['forecast_s'] = forecast_s
            try:
                publish_marker(job_dir, marker)
            except FileExistsError:
                winner = read_marker(job_dir)
                if winner is None or winner['job_digest'] != payload.job_digest:
                    raise BatchConflict(f'duplicate publication has a different winning job digest: {payload.job_id}')
                verify_marker(job_dir, winner, digests=False)
                return {'type': 'duplicate', 'job_id': payload.job_id, 'marker': winner,
                        'prepared': cache.prepared_count - before, 'cache_keys': cache.keys}
            return {'type': 'complete', 'job_id': payload.job_id, 'marker': marker,
                    'prepared': cache.prepared_count - before, 'cache_keys': cache.keys}
        except Exception as error:
            logging.getLogger(__name__).exception('job failed: %s', payload.job_id)
            failure = {'schema': 1, 'job_id': payload.job_id, 'job_digest': payload.job_digest,
                       'session': payload.session, 'error_type': type(error).__name__, 'message': str(error),
                       'traceback': traceback.format_exc(), 'worker': worker, 'exit_code': None}
            write_failure(run_dir, failure)
            return {'type': 'failed', 'job_id': payload.job_id, 'failure': failure,
                    'run': run_dir.name, 'prepared': cache.prepared_count - before}


def main():
    from threadpoolctl import threadpool_limits
    from ..inference.backend import BackendSession

    line = sys.stdin.readline()
    if not line:
        return 0
    startup = json.loads(line)
    try:
        connection = Client(startup['address'], family='AF_UNIX', authkey=bytes.fromhex(startup['authkey']))
    except (FileNotFoundError, ConnectionRefusedError):
        return 0
    execution = BatchExecution.from_mapping(startup['execution'])
    worker = {'slot': startup['slot'], 'device': startup['device'], 'device_kind': 'cpu', 'pid': os.getpid()}
    cache = PreparationCache(execution.preparation_cache_size, execution.forecast)
    try:
        with threadpool_limits(limits=execution.threads_per_worker), BackendSession(training_workers=execution.training_workers) as session:
            if startup['device'] != 'cpu':
                import jax
                devices = jax.devices()
                if len(devices) != 1 or devices[0].platform != 'gpu':
                    connection.send({'type': 'fatal', 'message': 'assigned GPU worker must see exactly one GPU', **worker})
                    return 1
                worker['device_kind'] = devices[0].device_kind
            provenance = capture_provenance()
            connection.send({'type': 'ready', **worker, 'provenance': provenance})
            while True:
                try:
                    message = connection.recv()
                except (EOFError, ConnectionResetError):
                    return 0
                if message['type'] == 'stop':
                    return 0
                if message['type'] != 'job':
                    raise ValueError(f"unknown worker message {message['type']!r}")
                result = run_job(JobPayload(**message['payload']), cache, session, connection=connection,
                                 worker=worker, revision=source_revision(provenance))
                try:
                    connection.send(result)
                except (BrokenPipeError, ConnectionResetError, EOFError):
                    return 0
                if result['type'] == 'failed':
                    return 0
    finally:
        cache.close()
        connection.close()


if __name__ == '__main__':
    raise SystemExit(main())
