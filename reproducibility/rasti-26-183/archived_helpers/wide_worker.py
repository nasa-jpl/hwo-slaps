"""Run one wide-prior case through the unchanged Stage 3 worker.

The spec is a copy of the case's canonical production spec with a new output
namespace, approval and a ``wide`` block. Only three things change, each bound
to the frozen PANEL_FREEZE.json variant named in the spec:

- the nonlinear prior half-widths, passed as ``priors_config`` to the fit (the
  Einstein-radius half-width is a fraction of the truth value);
- the freed-search subhalo log10(M200) range;
- the sampler live points / n_eff and the pre-committed sampler seed.

Images, noise, PSF, mask, likelihood, refinement and acceptance are unchanged.
After the route writes its payload the worker checks the data identity against
the original payload, the effective sampler settings, the resolved widths and
the realized prior box of every fitted model, and records them under
``wide_priors``.
"""
import copy
import hashlib
import json
import os
import sys
from pathlib import Path

DECLARED_MASS_RANGE = (6.0, 9.7)
SUBHALO_WINDOW = 0.15


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def payload_path(spec):
    suffix = '' if spec.get('direction') is None else '_dir' + str(spec['direction'])
    return Path(spec['case_output']) / f"nonlinear_validation_{spec['arm']}{suffix}.json"


def prior_widths(variant, full_config):
    """Resolve the variant's declared widths into model-builder keys for one case."""
    widths = variant['widths']
    theta_e = float(full_config['lensing']['lens_galaxy']['mass']['einstein_radius'])
    return {
        'lens_centre_sigma_arcsec': float(widths['lens_centre_arcsec']),
        'lens_einstein_radius_sigma': float(widths['einstein_radius_frac']) * theta_e,
        'lens_ell_comps_sigma': float(widths['lens_ell_comps']),
        'source_centre_sigma_arcsec': float(widths['source_centre_arcsec']),
        'image_flux_scale_frac_sigma': float(widths['image_flux_scale_frac']),
        'image_size_scale_frac_sigma': float(widths['image_size_scale_frac']),
    }


def live_for(variant, mode):
    return variant['sampler'][{'smooth': 'n_live_smooth', 'freed': 'n_live_subhalo_search',
                               'fixed_template': 'n_live_subhalo_fixed'}[mode]]


def batch_parity(model, analysis, role, spec, record):
    """Prove the batch-8 likelihood matches scalar evaluation before a sys0279 search (as null500 did)."""
    import numpy as np
    from autofit.non_linear.fitness import Fitness
    reference = read(spec['wide']['original_payload'])['fresh_profile_records'][role]
    lo = np.array([p.lower_limit for p in model.priors_ordered_by_id])
    hi = np.array([p.upper_limit for p in model.priors_ordered_by_id])
    rng = np.random.default_rng(20260926)
    vectors = np.array([reference['incumbent']['physical_vector'], reference['candidate_best_vector'],
                        *(lo + (.1 + .8*rng.random((6, len(lo))))*(hi - lo))])
    batch = int(spec['wide']['jax_n_batch_override'])
    scalar = Fitness(model=model, analysis=analysis)
    vmapped = Fitness(model=model, analysis=analysis, use_jax_vmap=True, batch_size=batch)
    direct = np.array([float(scalar(v)) for v in vectors])
    batched = np.asarray(vmapped(vectors))
    error = float(np.max(np.abs(batched - direct)))
    if not np.all(np.isfinite(direct)) or error > 1e-4:
        raise ValueError(f'batch-{batch} likelihood parity failed for {role}: {error}')
    record.setdefault('batch_parity', {})[role] = dict(vectors=vectors.tolist(), scalar=direct.tolist(),
                                                       batched=batched.tolist(), max_error=error, batch_size=batch)


def install_overrides(route, spec, panel, record):
    """Install the variant's sampler, seed, prior-width and mass-range overrides."""
    from hwoslaps.modeling.nonlinear import mass_mapping, psf_mismatch
    from hwoslaps.modeling.nonlinear.fresh_profile import FreshProfileRunner

    wide = spec['wide']
    variant = panel['variants'][wide['variant']]
    planned = next(c for c in panel['panel'] if c['case_id'] == spec['case_id'])
    assert planned['runs'][wide['variant']] == wide['sampler_seed'], 'sampler seed is not the frozen one'
    assert wide['sampler_seed'] != planned['original_sampler_seed']
    sampler = variant['sampler']
    mass_range = tuple(float(v) for v in variant['log10_m200_range'])

    original_settings = route._v7_sampler_settings

    def settings(protocol, release):
        result = original_settings(protocol, release)
        result.update(sampler)
        result['sampler_contract'] = dict(result['sampler_contract'])
        result['sampler_contract']['n_eff'] = sampler['n_eff']
        result['sampler_contract']['n_live_by_fit_mode'] = {
            'smooth': sampler['n_live_smooth'], 'freed': sampler['n_live_subhalo_search'],
            'fixed_template': sampler['n_live_subhalo_fixed']}
        for name in ('n_live_smooth', 'n_live_subhalo_search', 'n_live_subhalo_fixed', 'maxcall'):
            protocol['fit'][name] = sampler[name]
        if wide.get('jax_n_batch_override') is not None:
            result['jax_n_batch'] = int(wide['jax_n_batch_override'])
            protocol['fit']['jax_n_batch'] = int(wide['jax_n_batch_override'])
        return result

    route._v7_sampler_settings = settings
    route.derive_sampler_seed = lambda entropy, index, arm_index: wide['sampler_seed']

    original_case = psf_mismatch.run_psf_mismatch_case

    def run_case(validator, observation, full_config, trial, **kwargs):
        assert kwargs.get('priors_config') is None, 'production route unexpectedly set priors_config'
        widths = prior_widths(variant, full_config)
        record['priors_config'] = widths
        record['truth_einstein_radius'] = float(full_config['lensing']['lens_galaxy']['mass']['einstein_radius'])
        return original_case(validator, observation, full_config, trial, priors_config=widths, **kwargs)

    psf_mismatch.run_psf_mismatch_case = run_case

    original_context = mass_mapping.build_mass_mapping_context

    def context(full_config, log10_m200_range=(6.0, 8.5)):
        declared = tuple(float(v) for v in log10_m200_range)
        assert declared == DECLARED_MASS_RANGE, f'unexpected declared mass range {declared}'
        record['declared_log10_m200_range'] = list(declared)
        record['log10_m200_range'] = list(mass_range)
        return original_context(full_config, log10_m200_range=mass_range)

    mass_mapping.build_mass_mapping_context = context

    original_run_model = FreshProfileRunner.run_model

    def run_model(self, *, model, analysis, role, **kwargs):
        priors = model.priors_ordered_by_id
        paths = list(model.unique_prior_paths)
        assert len(paths) == len(priors), (role, len(paths), len(priors))
        box = dict(paths=['.'.join(str(p) for p in path) for path in paths],
                   lower=[float(p.lower_limit) for p in priors], upper=[float(p.upper_limit) for p in priors])
        record.setdefault('realized_boxes', {})[role] = box
        check_boxes(dict(priors_config=record['priors_config'], realized_boxes={role: box}), variant)
        if wide.get('jax_n_batch_override') is not None:
            assert self.settings.jax_n_batch == int(wide['jax_n_batch_override'])
            batch_parity(model, analysis, role, spec, record)
        return original_run_model(self, model=model, analysis=analysis, role=role, **kwargs)

    FreshProfileRunner.run_model = run_model
    return variant


def check_boxes(record, variant):
    """The realized prior box must be the declared truth-centred box, clipped only where the builder clips."""
    widths = record['priors_config']
    mass_range = [float(v) for v in variant['log10_m200_range']]
    checks = {}
    for role, box in record['realized_boxes'].items():
        rows = []
        for path, lo, hi in zip(box['paths'], box['lower'], box['upper']):
            half = (hi - lo) / 2.0
            parts = path.split('.')
            want = None
            if parts[:3] == ['galaxies', 'lens', 'mass']:
                if parts[3] == 'centre':
                    want = widths['lens_centre_sigma_arcsec']
                elif parts[3] == 'einstein_radius':
                    want = widths['lens_einstein_radius_sigma']
                elif parts[3] == 'ell_comps' and lo > -0.9 + 1e-9 and hi < 0.9 - 1e-9:
                    want = widths['lens_ell_comps_sigma']
                elif parts[3] != 'ell_comps':
                    raise AssertionError('unexpected lens parameter ' + path)
            elif parts[:3] == ['galaxies', 'source', 'light']:
                if parts[3] == 'centre':
                    want = widths['source_centre_sigma_arcsec']
                elif parts[3] in ('flux_scale', 'size_scale'):
                    frac = widths['image_%s_frac_sigma' % parts[3]]
                    if lo > 0:
                        assert abs(hi / lo - (1 + frac) / (1 - frac)) <= 1e-9, (role, path, lo, hi)
                else:
                    raise AssertionError('unexpected source parameter ' + path)
            elif parts[:3] == ['galaxies', 'lens', 'subhalo']:
                if parts[3] == 'centre':
                    want = SUBHALO_WINDOW
                elif parts[3] == 'log10_m200':
                    assert [lo, hi] == mass_range, (role, path, lo, hi)
                else:
                    raise AssertionError('unexpected subhalo parameter ' + path)
            else:
                raise AssertionError('unexpected parameter ' + path)
            if want is not None:
                assert abs(half - want) <= 1e-9 * max(1.0, want), (role, path, half, want)
            rows.append(dict(path=path, lower=lo, upper=hi, half_width=half, checked_half_width=want))
        checks[role] = rows
    return checks


def install_checks(route, spec, variant, record):
    original_main = route.main

    def main(*args, **kwargs):
        original_main(*args, **kwargs)
        path = payload_path(spec)
        payload = read(path)
        wide = spec['wide']
        assert payload['sampler_seed'] == wide['sampler_seed']
        for name, expected in wide['data_identity'].items():
            assert payload.get(name) == expected, 'data identity mismatch: ' + name
        roles = {}
        for role in ('smooth', 'subhalo'):
            rec = payload['fresh_profile_records'][role]
            fit = payload['case'][role + '_fit']
            sampled = rec.get('candidate_acceptance_status') != 'verified_zero_residual_anchor'
            if sampled:
                assert fit['n_eff_effective'] == variant['sampler']['n_eff'], (role, fit['n_eff_effective'])
                mode = 'smooth' if role == 'smooth' else payload['case']['fit_mode']
                assert fit['n_live_effective'] == live_for(variant, mode), (role, fit['n_live_effective'])
            support = rec.get('candidate_start_agreement', {}).get('supporting_original_start_indices', [])
            roles[role] = dict(status=rec.get('candidate_acceptance_status'), supporting_starts=len(support),
                               log_likelihood_max=fit.get('log_likelihood_max'),
                               raw_sampler_log_likelihood_max=rec.get('fresh_search_summary', {}).get('log_likelihood_max'),
                               n_like_max_reached=fit.get('n_like_max_reached'),
                               jax_n_batch_effective=fit.get('jax_n_batch_effective'))
            if wide.get('jax_n_batch_override') is not None:
                assert fit.get('jax_n_batch_effective') == int(wide['jax_n_batch_override']), (role, fit.get('jax_n_batch_effective'))
        Path(spec['output'], 'wide_record.json').write_text(json.dumps(record, indent=1) + '\n')
        boxes = check_boxes(record, variant)
        if payload['case']['fit_mode'] == 'freed':
            assert record.get('log10_m200_range') == [float(v) for v in variant['log10_m200_range']]
        payload['wide_priors'] = dict(
            variant=wide['variant'], variant_declaration=variant, panel=wide['panel'], panel_sha256=wide['panel_sha256'],
            priors_config=record['priors_config'], truth_einstein_radius=record['truth_einstein_radius'],
            declared_log10_m200_range=record.get('declared_log10_m200_range'),
            log10_m200_range=record.get('log10_m200_range'), realized_boxes=boxes, roles=roles,
            sampler_seed=wide['sampler_seed'], original_spec=wide['original_spec'], batch_parity=record.get('batch_parity'),
            tangent_comparator_note=('likelihood_matched_tangent, when present, builds its models with the production '
                                     'default widths and the widened mass context; it is not a wide-prior quantity'),
            original_payload=wide['original_payload'], original_payload_sha256=wide['original_payload_sha256'],
            baseline_payload=wide['baseline_payload'], baseline_q=wide['baseline_q'],
            base_seed_entropy=payload.pop('seed_entropy'), base_seed_spawn_key=payload.pop('seed_spawn_key'),
            canonical_production_eligible=False)
        payload['fit_settings'] = dict(payload['fit_settings'])
        payload['fit_settings']['declared_log10_m200_range'] = payload['fit_settings'].get('log10_m200_range')
        if record.get('log10_m200_range') is not None:
            payload['fit_settings']['log10_m200_range'] = record['log10_m200_range']
        for name in ('n_live_smooth', 'n_live_subhalo_search', 'n_live_subhalo_fixed', 'maxcall'):
            payload['fit_settings'][name] = variant['sampler'][name]
        payload['release_protocol'] = copy.deepcopy(payload['release_protocol'])
        payload['release_protocol']['sampler'].update(variant['sampler'])
        payload['release_protocol']['engineering_override_receipt'] = wide['panel']
        path.write_text(json.dumps(payload, indent=2, allow_nan=False) + '\n')

    route.main = main


def main():
    spec_path = Path(sys.argv[1]).resolve()
    spec = read(spec_path)
    if sha(__file__) != spec['hashes'][str(Path(__file__).resolve())]:
        raise ValueError('wide worker hash does not match the spec binding')
    if sha(spec['wide']['panel']) != spec['wide']['panel_sha256']:
        raise ValueError('panel hash mismatch')
    worktree = Path(spec['wide']['worktree'])
    sys.path[:0] = [str(worktree / 'src'), str(worktree / 'scripts')]
    os.environ['HWOSLAPS_RELEASE_CASE_ID'] = spec['case_id']
    if spec.get('expected_case_sha256'):
        os.environ['HWOSLAPS_EXPECTED_CASE_SHA256'] = spec['expected_case_sha256']
    import run_nonlinear_validation as route
    import run_nonlinear_production as worker
    panel = read(spec['wide']['panel'])
    record = {}
    variant = install_overrides(route, spec, panel, record)
    install_checks(route, spec, variant, record)
    return worker.main([str(spec_path)])


if __name__ == '__main__':
    raise SystemExit(main())
