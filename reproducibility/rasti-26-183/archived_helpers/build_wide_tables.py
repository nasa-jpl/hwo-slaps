"""Canonical tables for the paper's nonlinear comparisons from the broad-prior campaign.

Usage (xtx, this directory): python build_wide_tables.py PLAN [PLAN ...]
PLANs are the campaign plans whose slam1 runs count (first passes) plus retry plans (variant slam1_retry).

For each of the 1,179 paper cases the row starts from the existing canonical row (02_CANONICAL_CASES, or the
null500 overlay for the 315 repeated controls), keeps the prior-independent columns (catalog, trial, Fisher
statistics, PSF labels), and replaces every fit-derived column with values extracted, by the unchanged handoff
extractor, from the broad-prior payload. original_* is the broad-prior first pass; final_* is the first pass
or, for an unresolved first pass, a second pass that is accepted and keeps both role maxima within 0.1 logL of
the first pass (appendix rule). Writes tables/02_CANONICAL_CASES_part01..05_of_05.txt and
tables/TOP50_REPEATED_VIEW.csv, plus WIDE_TABLES_SUMMARY.json.
"""
import csv
import hashlib
import json
import math
import sys
from collections import OrderedDict
from pathlib import Path

B = Path(__file__).resolve().parent
S3 = B.parent
HANDOFF_TABLES = Path('/data/home/gvassilakis/prior_census_20260926/handoff_tables')
EXTRACTOR = Path('/data/home/gvassilakis/prior_census_20260926/extract_payload_scalars.py')
NULL500_OVERLAY = Path('/data/home/gvassilakis/prior_census_20260926/NULL500_CANONICAL_OVERLAY.txt')
NA = 'NA'
THRESHOLD = 10.0
TOLERANCE = 0.1
OUT = B/'tables'


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def cell(v):
    if v is None:
        return NA
    if isinstance(v, float):
        return NA if math.isnan(v) else repr(v)
    if isinstance(v, (list, dict)):
        return json.dumps(v, separators=(',', ':'))
    return str(v)


def read_tsv(paths):
    header, rows = None, []
    for p in paths:
        lines = [l for l in open(p) if not l.startswith('#')]
        this = lines[0].rstrip('\n').split('\t')
        assert header in (None, this), p
        header = this
        rows.extend(csv.DictReader(lines, delimiter='\t'))
    return header, rows


def load_extractor():
    source = EXTRACTOR.read_text()
    assert source.endswith('\nmain()\n')
    namespace = {'__name__': 'handoff_extract_payload_scalars'}
    exec(compile(source[:-len('main()\n')], str(EXTRACTOR), 'exec'), namespace)
    return namespace['extract']


def payload_path(plan, key, spec):
    suffix = '' if spec.get('direction') is None else '_dir' + str(spec['direction'])
    name = f"nonlinear_validation_{spec['arm']}{suffix}.json"
    for base in (B/f'task_{plan}'/'attempts'/key, Path('/nfs/cam/gvassilakis/nonlinear_c963160_20260918/wide_priors_20260926')/plan/'attempts'/key):
        if (base/'case'/name).exists():
            return base/'case'/name
    return None


def runs(plans):
    """(case_id, variant) -> payload path for COMPLETE attempts."""
    out = {}
    for plan in plans:
        records = {r['key']: r for r in json.load(open(B/f'PREPARED_{plan}.json'))['records']}
        manifest = {j['key']: j for j in json.load(open(B/f'manifest_{plan}.json'))['jobs']}
        ledger = json.load(open(B/f'task_{plan}/state/budget.json'))['attempts']
        for key, a in ledger.items():
            r = records[key]
            if a['status'] != 'COMPLETE' or r['variant'] not in ('slam1', 'slam1_retry'):
                continue
            p = payload_path(plan, key, json.load(open(manifest[key]['spec'])))
            assert p is not None, (plan, key)
            assert (r['case_id'], r['variant']) not in out, (r['case_id'], r['variant'])
            out[(r['case_id'], r['variant'])] = p
    return out


def role(c, name, key):
    return c['roles'][name][key]


def fit_columns(prefix, c):
    """original_* (prefix 'original') columns from one payload record."""
    sq = c['metric_signed_delta_log_l']
    q = None if sq is None else 2.0*sq
    return OrderedDict([
        (f'{prefix}_numerical_status', c['numerical_status_payload']), (f'{prefix}_q_signed', q),
        (f'{prefix}_q_clipped', None if q is None else max(0.0, q)),
        (f'{prefix}_marginal_flag', c['marginal_q_flag_payload']), (f'{prefix}_profile_decision', c['profile_decision_payload']),
        (f'{prefix}_h0_acceptance', role(c, 'smooth', 'acceptance_status')), (f'{prefix}_h1_acceptance', role(c, 'subhalo', 'acceptance_status')),
        (f'{prefix}_h0_support', role(c, 'smooth', 'support_supporting_count')), (f'{prefix}_h1_support', role(c, 'subhalo', 'support_supporting_count')),
        (f'{prefix}_payload_sha256', c['payload_sha256']), (f'{prefix}_payload_path', c['payload_path']),
        (f'{prefix}_code_revision', c['code_revision_git_hash'])])


def final_columns(f, source, first):
    fsq = f['metric_signed_delta_log_l']
    fq = None if fsq is None else 2.0*fsq
    status = f['numerical_status_payload']
    detection = (fq >= THRESHOLD) if status == 'accepted' else None
    changed = class_changed = None
    if source != 'wide_slam1':
        changed = 'unresolved_to_' + ('detection' if detection else 'non_detection')
        q0 = first['metric_signed_delta_log_l']
        class_changed = None if q0 is None or fq is None else (2*q0 >= THRESHOLD) != (fq >= THRESHOLD)
    return OrderedDict([
        ('final_source', source), ('final_numerical_status', status), ('final_payload_sha256', f['payload_sha256']),
        ('final_payload_path', f['payload_path']), ('final_code_revision', f['code_revision_git_hash']),
        ('final_jax_n_batch', f['jax_n_batch_setting']),
        ('final_log_l_h0', f['metric_log_l_smooth']), ('final_log_l_h1', f['metric_log_l_subhalo']),
        ('final_q_signed', fq), ('final_q_clipped', None if fq is None else max(0.0, fq)), ('q_threshold', THRESHOLD),
        ('final_marginal_flag', abs(fq - THRESHOLD) < 1.0 if status == 'accepted' else None), ('final_detection', detection),
        ('final_delta_log_evidence', f['delta_log_evidence']), ('evidence_status', 'provisional'),
        ('final_h0_acceptance', role(f, 'smooth', 'acceptance_status')), ('final_h1_acceptance', role(f, 'subhalo', 'acceptance_status')),
        ('final_h0_support_count', role(f, 'smooth', 'support_supporting_count')),
        ('final_h1_support_count', role(f, 'subhalo', 'support_supporting_count')),
        ('final_h0_original_start_count', role(f, 'smooth', 'support_original_start_count')),
        ('final_h1_original_start_count', role(f, 'subhalo', 'support_original_start_count')),
        ('final_h0_incumbent_saved_logL', role(f, 'smooth', 'incumbent_saved_log_likelihood')),
        ('final_h0_incumbent_direct_logL', role(f, 'smooth', 'incumbent_direct_log_likelihood')),
        ('final_h0_incumbent_passed', role(f, 'smooth', 'incumbent_passed')),
        ('final_h1_incumbent_saved_logL', role(f, 'subhalo', 'incumbent_saved_log_likelihood')),
        ('final_h1_incumbent_direct_logL', role(f, 'subhalo', 'incumbent_direct_log_likelihood')),
        ('final_h1_incumbent_passed', role(f, 'subhalo', 'incumbent_passed')),
        ('final_h0_profile_improvement_logL', role(f, 'smooth', 'profile_improvement_over_sampler_logL')),
        ('final_h1_profile_improvement_logL', role(f, 'subhalo', 'profile_improvement_over_sampler_logL')),
        ('final_h0_tighter_repeat_change_logL', role(f, 'smooth', 'tighter_repeat_change_logL')),
        ('final_h1_tighter_repeat_change_logL', role(f, 'subhalo', 'tighter_repeat_change_logL')),
        ('final_h0_sampler_log_evidence', role(f, 'smooth', 'sampler_log_evidence')),
        ('final_h1_sampler_log_evidence', role(f, 'subhalo', 'sampler_log_evidence')),
        ('final_h0_n_live', role(f, 'smooth', 'sampler_n_live_effective')), ('final_h1_n_live', role(f, 'subhalo', 'sampler_n_live_effective')),
        ('final_h0_n_eff', role(f, 'smooth', 'sampler_n_eff_effective')), ('final_h1_n_eff', role(f, 'subhalo', 'sampler_n_eff_effective')),
        ('recovered_log10_m200_ml', f['recovery_log10_m200_ml']), ('recovered_log10_m200_p16', f['recovery_log10_m200_p16']),
        ('recovered_log10_m200_p50', f['recovery_log10_m200_p50']), ('recovered_log10_m200_p84', f['recovery_log10_m200_p84']),
        ('recovered_centre_y_arcsec', f['recovery_centre_ml_y']), ('recovered_centre_x_arcsec', f['recovery_centre_ml_x']),
        ('recovered_mass_at_lower_bound', f['recovery_mass_at_lower_bound']),
        ('recovered_mass_at_upper_bound', f['recovery_mass_at_upper_bound']),
        ('procedure_version', f['procedure_version']), ('objective_version', f['objective_version']),
        ('worker_elapsed_s', f['worker_exit_elapsed_s']), ('fit_pair_s', (f['timings'] or {}).get('fit_pair_s')),
        ('cuda_visible_devices', f['cuda_visible_devices']),
        ('detection_flag_changed_by_recovery', changed), ('q_classification_changed_by_recovery', class_changed)])


def promote(first, retry):
    """Second pass counts only if accepted and neither role maximum falls more than 0.1 logL below the first pass."""
    if retry is None or retry['numerical_status_payload'] != 'accepted':
        return False
    for r in ('smooth', 'subhalo'):
        key = {'smooth': 'metric_log_l_smooth', 'subhalo': 'metric_log_l_subhalo'}[r]
        if first[key] is not None and retry[key] < first[key] - TOLERANCE:
            return False
    return True


def main():
    plans = sys.argv[1:]
    extract = load_extractor()
    header, canon = read_tsv(sorted(HANDOFF_TABLES.glob('02_CANONICAL_CASES_part*.txt')))
    h2, overlay = read_tsv([NULL500_OVERLAY])
    assert h2 == header
    base = {r['case_id']: r for r in canon + overlay}
    selection = json.load(open(B/'SELECTION_redo3.json'))['cases']
    assert len(selection) == 1179
    have = runs(plans)
    rows, missing, promoted, summary = [], [], [], {}
    for c in selection:
        cid = c['case_id']
        row = OrderedDict((h, base[cid][h]) for h in header)
        first_path = have.get((cid, 'slam1'))
        if first_path is None:
            missing.append(cid)
            row.update(final_source='not_run', final_numerical_status='not_run')
            rows.append(row)
            continue
        first = extract(str(first_path))
        wide = json.load(open(first_path))['wide_priors']
        assert wide['variant'] == 'slam1'
        retry_path = have.get((cid, 'slam1_retry'))
        retry = extract(str(retry_path)) if retry_path else None
        use_retry = first['numerical_status_payload'] != 'accepted' and promote(first, retry)
        f = retry if use_retry else first
        if use_retry:
            promoted.append(cid)
        row.update(sampler_seed=first['sampler_seed'], sampler_seed_spawn_key=None,
                   tangent_computed=first['tangent_computed'], tangent_q=first['tangent_q'],
                   tangent_q_free_background_only=first['tangent_q_with_free_background_only'],
                   tangent_q_finite_prior_box=first['tangent_q_with_finite_prior_box'],
                   tangent_derivatives_stable=first['tangent_derivatives_stable'],
                   tangent_half_step_q_difference=first['tangent_half_step_q_difference'],
                   catalog_archived_v5_q_fit=None, catalog_archived_v5_delta_log_evidence=None,
                   h1_evidence_claim=True, release_freeze_sha256=first['release_freeze_sha256'],
                   consumed_freeze_sha256=first['consumed_freeze_sha256'])
        row.update(fit_columns('original', first))
        row.update(final_columns(f, 'wide_slam1_retry' if use_retry else 'wide_slam1', first))
        for k in ('h1_anchor_acceptance_status', 'h1_anchor_chi2', 'h1_anchor_sampler_executed', 'h1_anchor_evidence_claim'):
            row[k] = None
        assert list(row) == header
        rows.append({h: cell(v) if not isinstance(v, str) else v for h, v in row.items()})
    rows.sort(key=lambda r: r['case_id'])
    OUT.mkdir(exist_ok=True)
    preamble = [f'02_CANONICAL_CASES (broad-prior campaign): the {len(rows)} nonlinear comparisons reported in the paper.',
                'Fit-derived columns come from the broad-prior payloads (SLaM-informed uniform boxes centred on the input values, '
                'log10 M200 6 to 11, 300/600 live points, n_eff 2000); prior-independent columns from the canonical handoff rows.',
                'final_* = first pass, or an accepted second pass (600/1200, n_eff 4000, fresh seed) that keeps both role maxima within 0.1 '
                'logL of the first pass. final_detection = accepted AND final_q_signed >= 10.']
    n = 5
    for i in range(n):
        part = rows[i::n] if False else rows[i*len(rows)//n:(i + 1)*len(rows)//n]
        with open(OUT/f'02_CANONICAL_CASES_part{i + 1:02d}_of_{n:02d}.txt', 'w') as fh:
            fh.writelines('# ' + l + '\n' for l in preamble)
            fh.write('\t'.join(header) + '\n')
            fh.writelines('\t'.join(r[h] if isinstance(r[h], str) else cell(r[h]) for h in header) + '\n' for r in part)
    top50 = set(json.load(open(B/'SELECTION_DRAFT_redo.json'))['top50'])
    view_rows = [r for r in rows if r['system_id'] in top50 and (r['arm'] == 'noisy_control' or r['arm'].startswith('noisy_control_r'))
                 and r['case_kind'] == 'standard' and r['campaign'] != 'psf_knowledge_nonlinear_v1']
    with open(OUT/'TOP50_REPEATED_VIEW.csv', 'w', newline='') as fh:
        w = csv.writer(fh)
        w.writerow(['case_id', 'system_id', 'arm', 'final_numerical_status', 'final_q_signed', 'final_detection'])
        for r in view_rows:
            w.writerow([r['case_id'], r['system_id'], r['arm'], r['final_numerical_status'], r['final_q_signed'], r['final_detection']])
    status = {}
    for r in rows:
        status[r['final_numerical_status']] = status.get(r['final_numerical_status'], 0) + 1
    summary = dict(plans=plans, rows=len(rows), missing=len(missing), promoted_second_pass=promoted, status=status,
                   repeated_view_rows=len(view_rows), extractor_sha256=sha(EXTRACTOR))
    (B/'WIDE_TABLES_SUMMARY.json').write_text(json.dumps(summary, indent=1) + '\n')
    print(json.dumps({k: v for k, v in summary.items() if k != 'promoted_second_pass'}), 'promoted', len(promoted))


if __name__ == '__main__':
    main()
