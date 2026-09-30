"""Run one null500 repeated-control case through the unchanged Stage 3 worker.

The spec is a clone of the system's canonical standard noisy_control spec with
only the arm (noisy_control_rK) and the case identity changed; the route
derives the frozen noise and sampler seeds from that arm's declaration. This
wrapper checks the derived seeds against the amendment before the payload is
receipted. For a spec carrying jax_n_batch_override = 8 (sys0279 only) it also
applies the validated batch-8 likelihood evaluation and proves scalar parity
before each fresh search.
"""
import hashlib
import json
import os
import sys
from pathlib import Path


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def payload_path(spec):
    return Path(spec['case_output']) / f"nonlinear_validation_{spec['arm']}.json"


def check_payload(spec, payload):
    from hwoslaps.campaign.system_ids import bare_system_id
    expected = spec['null500']
    observed = dict(noise_seed=payload['noise_seed'], noise_replicate=payload['noise_replicate'],
                    noise_spawn_key=payload['noise_spawn_key'], sampler_seed=payload['sampler_seed'],
                    seed_spawn_key=payload['seed_spawn_key'], arm=payload['arm'],
                    system_id=bare_system_id(payload['system_id']),
                    rung=payload['rung'], positions_artifact_sha256=payload['positions_artifact_sha256'])
    wanted = dict(noise_seed=expected['noise_seed'], noise_replicate=expected['noise_replicate'],
                  noise_spawn_key=expected['noise_spawn_key'], sampler_seed=expected['sampler_seed'],
                  seed_spawn_key=expected['sampler_seed_spawn_key'], arm=spec['arm'], system_id=spec['system_id'],
                  rung=expected['control_rung'], positions_artifact_sha256=expected['control_positions_sha256'])
    for name, value in wanted.items():
        if observed[name] != value:
            raise ValueError(f'null500 identity mismatch for {name}: {observed[name]!r} != {value!r}')
    if payload['arm_declaration'].get('dataset_kind') != 'noisy' or payload['arm_declaration'].get('subhalo_in_truth') is not False:
        raise ValueError('null500 case must be noisy data without a subhalo')


def install_batch8(route, spec):
    import numpy as np
    from autofit.non_linear.fitness import Fitness
    from hwoslaps.modeling.nonlinear.fresh_profile import FreshProfileRunner

    original_settings = route._v7_sampler_settings

    def settings(protocol, release):
        result = original_settings(protocol, release)
        result['jax_n_batch'] = 8
        protocol['fit']['jax_n_batch'] = 8
        return result

    original_run_model = FreshProfileRunner.run_model
    reference = read(spec['null500']['batch8_parity_reference'])

    def checked_run_model(self, *, model, analysis, role, **kwargs):
        assert self.settings.jax_n_batch == 8
        record = reference['fresh_profile_records'][role]
        lo = np.array([p.lower_limit for p in model.priors_ordered_by_id])
        hi = np.array([p.upper_limit for p in model.priors_ordered_by_id])
        rng = np.random.default_rng(20260923)
        vectors = np.array([record['incumbent']['physical_vector'], record['candidate_best_vector'],
                            *(lo + (.1 + .8 * rng.random((6, len(lo)))) * (hi - lo))])
        scalar = Fitness(model=model, analysis=analysis)
        vmapped = Fitness(model=model, analysis=analysis, use_jax_vmap=True, batch_size=8)
        direct = np.array([float(scalar(v)) for v in vectors])
        batch8 = np.asarray(vmapped(vectors))
        batch4 = np.concatenate([np.asarray(vmapped(vectors[i:i + 4])) for i in range(0, 8, 4)])
        errors = dict(batch8=float(np.max(np.abs(batch8 - direct))), batch4=float(np.max(np.abs(batch4 - direct))))
        if not np.all(np.isfinite(direct)) or max(errors.values()) > 1e-4:
            raise ValueError(f'batch-8 likelihood parity failed for {role}: {errors}')
        proof_path = Path(spec['output']) / 'LIKELIHOOD_PARITY.json'
        proof = read(proof_path) if proof_path.exists() else {}
        proof[role] = dict(vectors=vectors.tolist(), scalar=direct.tolist(), batch8=batch8.tolist(),
                           batch4=batch4.tolist(), max_error=errors, passed=True)
        proof_path.write_text(json.dumps(proof, indent=2) + '\n')
        result = original_run_model(self, model=model, analysis=analysis, role=role, **kwargs)
        assert result.jax_n_batch_effective == 8
        return result

    route._v7_sampler_settings = settings
    FreshProfileRunner.run_model = checked_run_model


def install_checks(route, spec):
    original_main = route.main

    def main(*args, **kwargs):
        original_main(*args, **kwargs)
        path = payload_path(spec)
        payload = read(path)
        check_payload(spec, payload)
        if spec['null500'].get('jax_n_batch_override') == 8:
            if not all(payload['case'][role + '_fit']['jax_n_batch_effective'] == 8 for role in ('smooth', 'subhalo')):
                raise ValueError('sys0279 role did not run at batch 8')
            payload['null500_batch8'] = dict(
                batch_size=8, original_batch_size=32, amendment=spec['null500']['amendment'],
                reason='Batch 32 gives a wrong likelihood on the 1074x1074 sys0279 geometry (recovery 2026-09-19); '
                       'batches 1, 2, 4 and 8 agree with scalar evaluation. Both roles parity-checked before search.',
                parity=read(Path(spec['output']) / 'LIKELIHOOD_PARITY.json'))
            path.write_text(json.dumps(payload, indent=2, allow_nan=False) + '\n')

    route.main = main


def main():
    spec_path = Path(sys.argv[1]).resolve()
    spec = read(spec_path)
    if sha(__file__) != spec['hashes'][str(Path(__file__).resolve())]:
        raise ValueError('null500 worker hash does not match the spec binding')
    if sha(spec['null500']['amendment']) != spec['null500']['amendment_sha256']:
        raise ValueError('null500 amendment hash mismatch')
    worktree = Path(spec['null500']['worktree'])
    sys.path[:0] = [str(worktree / 'src'), str(worktree / 'scripts')]
    os.environ['HWOSLAPS_RELEASE_CASE_ID'] = spec['case_id']
    import run_nonlinear_validation as route
    import run_nonlinear_production as worker
    if spec['null500'].get('jax_n_batch_override') == 8:
        install_batch8(route, spec)
    elif 'jax_n_batch_override' in spec['null500']:
        raise ValueError('only the declared sys0279 batch-8 override is supported')
    install_checks(route, spec)
    return worker.main([str(spec_path)])


if __name__ == '__main__':
    raise SystemExit(main())
