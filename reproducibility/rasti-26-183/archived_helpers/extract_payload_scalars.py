"""Read-only extraction of scalar evidence from saved v7 nonlinear payloads.

Runs on xtx. Reads the 1,601 canonical original payloads named in
FINAL_RECONCILIATION.json (production root on NFS, canary root on NFS), the 26
selected recovery payloads, and each attempt's worker_exit.json. Writes one JSON
document to stdout. Nothing is written on the remote host. No lensing runtime is
imported.
"""
import json, os, sys, hashlib, glob

REC = "/nfs/cam/gvassilakis/nonlinear_c963160_20260918/production_standard8_20260918_harvest/recovery_reconciliation/FINAL_RECONCILIATION.json"
PROD_DATA = "/data/home/gvassilakis/nonlinear_release_20260917/stage3/production_standard8_20260918/attempts/"
PROD_NFS = "/nfs/cam/gvassilakis/nonlinear_c963160_20260918/production_standard8_20260918/attempts/"
CAN_NFS = "/nfs/cam/gvassilakis/nonlinear_c963160_20260918/canary/attempts/"

def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()

def g(d, *ks, default=None):
    for k in ks:
        if not isinstance(d, dict) or k not in d:
            return default
        d = d[k]
    return d

def resolve(p):
    if os.path.exists(p):
        return p
    if p.startswith(PROD_DATA):
        q = PROD_NFS + p[len(PROD_DATA):]
        if os.path.exists(q):
            return q
    return None

def role_record(fp, fit):
    """fp = fresh_profile_records[role]; fit = case.<role>_fit."""
    out = {}
    if fp is None and fit is None:
        return None
    fp = fp or {}
    fit = fit or {}
    runs = fp.get("runs") or []
    orig = [r for r in runs if (r.get("start_provenance") or {}).get("original_start")]
    finite_orig = [r for r in orig if r.get("observed_best_half_chi2") is not None]
    best_half = fp.get("candidate_best_half_chi2")
    best_run_half = min((float(r["observed_best_half_chi2"]) for r in runs if r.get("observed_best_half_chi2") is not None), default=None)
    rep = fp.get("tighter_repeat") or {}
    rep_half = rep.get("observed_best_half_chi2")
    out.update({
        "acceptance_status": fp.get("candidate_acceptance_status"),
        "procedure_version": fp.get("procedure_version"),
        "candidate_best_log_likelihood": fp.get("candidate_best_log_likelihood"),
        "candidate_best_chi2": fp.get("candidate_best_chi2"),
        "candidate_best_half_chi2": best_half,
        "candidate_direct_log_likelihood_error": fp.get("candidate_direct_log_likelihood_error"),
        "candidate_scalar_residual_error": fp.get("candidate_scalar_residual_error"),
        "candidate_gradient_valid": fp.get("candidate_best_gradient_valid"),
        "candidate_projected_gradient_l2": g(fp, "candidate_best_projected_gradient", "l2"),
        "candidate_projected_gradient_linf": g(fp, "candidate_best_projected_gradient", "linf"),
        "candidate_not_worse_than_incumbent": fp.get("candidate_not_worse_than_incumbent"),
        "support_original_start_count": g(fp, "candidate_start_agreement", "original_start_count"),
        "support_finite_start_count": g(fp, "candidate_start_agreement", "finite_start_count"),
        "support_supporting_count": (len(g(fp, "candidate_start_agreement", "supporting_original_start_indices", default=[]) or [])
                                     if fp.get("candidate_start_agreement") else None),
        "support_supporting_indices": g(fp, "candidate_start_agreement", "supporting_original_start_indices"),
        "support_passed": g(fp, "candidate_start_agreement", "support_passed"),
        "support_tolerance_logL": g(fp, "candidate_start_agreement", "support_tolerance_log_likelihood"),
        "support_minimum": g(fp, "candidate_start_agreement", "minimum_supporting_original_starts"),
        "incumbent_saved_log_likelihood": g(fp, "incumbent", "saved_log_likelihood"),
        "incumbent_direct_log_likelihood": g(fp, "incumbent", "direct_log_likelihood"),
        "incumbent_saved_log_likelihood_error": g(fp, "incumbent", "saved_log_likelihood_error"),
        "incumbent_direct_log_likelihood_error": g(fp, "incumbent", "direct_log_likelihood_error"),
        "incumbent_matches_current_search": g(fp, "incumbent", "matches_current_search"),
        "incumbent_candidate_not_worse": g(fp, "incumbent", "candidate_not_worse"),
        "incumbent_scalar_consistent": g(fp, "incumbent", "scalar_consistent"),
        "incumbent_passed": g(fp, "incumbent", "passed"),
        "incumbent_half_chi2": g(fp, "incumbent", "half_chi2"),
        "incumbent_tolerance": g(fp, "incumbent", "tolerance"),
        "sampler_log_likelihood_max": g(fp, "fresh_search_summary", "log_likelihood_max"),
        "sampler_log_evidence": g(fp, "fresh_search_summary", "log_evidence"),
        "sampler_n_live_effective": g(fp, "fresh_search_summary", "n_live_effective"),
        "sampler_n_eff_effective": g(fp, "fresh_search_summary", "n_eff_effective"),
        "sampler_f_live_effective": g(fp, "fresh_search_summary", "f_live_effective"),
        "sampler_n_like_max_reached": g(fp, "fresh_search_summary", "n_like_max_reached"),
        "sampler_status": g(fp, "fresh_search_summary", "status"),
        "sampler_runtime_s": g(fp, "fresh_search_summary", "runtime_s"),
        "sampler_jax_n_batch_effective": g(fp, "fresh_search_summary", "jax_n_batch_effective"),
        "sampler_n_free_parameters": g(fp, "fresh_search_summary", "n_free_parameters"),
        "sampler_search_internal_retained": g(fp, "fresh_search_summary", "search_internal_retained"),
        "sampler_search_internal_dill_bytes": g(fp, "fresh_search_summary", "search_internal_payload", "files", "search_internal.dill", "bytes"),
        "sampler_search_internal_dill_sha256": g(fp, "fresh_search_summary", "search_internal_payload", "files", "search_internal.dill", "sha256"),
        "profile_improvement_over_sampler_logL": (
            None if fp.get("candidate_best_log_likelihood") is None or g(fp, "fresh_search_summary", "log_likelihood_max") is None
            else float(fp["candidate_best_log_likelihood"]) - float(g(fp, "fresh_search_summary", "log_likelihood_max"))),
        "n_runs": len(runs),
        "n_original_starts_finite": len(finite_orig),
        "runs_observed_best_half_chi2": [r.get("observed_best_half_chi2") for r in runs],
        "runs_original_start_flags": [bool((r.get("start_provenance") or {}).get("original_start")) for r in runs],
        "runs_success": [r.get("success") for r in runs],
        "runs_message": [r.get("message") for r in runs],
        "runs_nit": [r.get("nit") for r in runs],
        "best_run_observed_half_chi2": best_run_half,
        "tighter_repeat_observed_best_half_chi2": rep_half,
        "tighter_repeat_change_logL": (None if rep_half is None or best_run_half is None else abs(float(rep_half) - float(best_run_half))),
        "tighter_repeat_success": rep.get("success"),
        "tighter_repeat_status": rep.get("status"),
        "tighter_repeat_message": rep.get("message"),
        "tighter_repeat_evaluations": rep.get("evaluation_count"),
        "anchor_chi2": fp.get("anchor_chi2"),
        "anchor_sampler_executed": fp.get("sampler_executed"),
        "anchor_evidence_claim": fp.get("evidence_claim"),
        "fit_status": fit.get("status"),
        "fit_log_likelihood_max": fit.get("log_likelihood_max"),
        "fit_log_evidence": fit.get("log_evidence"),
        "fit_log_likelihood_extraction_method": fit.get("log_likelihood_extraction_method"),
        "fit_search_engine": fit.get("search_engine"),
        "fit_n_live": fit.get("n_live"),
        "fit_mode": fit.get("fit_mode"),
        "fit_warnings": fit.get("warnings"),
    })
    return out

def extract(path):
    d = json.load(open(path))
    case = d.get("case") or {}
    met = case.get("metric") or {}
    tr = case.get("trial") or {}
    dm = case.get("dataset_metadata") or {}
    sr = case.get("subhalo_recovery") or {}
    fpr = d.get("fresh_profile_records") or {}
    lmt = d.get("likelihood_matched_tangent") or fpr.get("likelihood_matched_tangent") or g(case, "diagnostics", "likelihood_matched_tangent")
    bfq = d.get("bracket_fisher_q") or {}
    rc = d.get("retention_contract") or {}
    rec = {
        "payload_path": path,
        "payload_sha256": sha(path),
        "schema_version": d.get("schema_version"),
        "system_id": d.get("system_id"), "arm": d.get("arm"), "tier": d.get("tier"), "rung": d.get("rung"),
        "artifact": d.get("artifact"),
        "code_revision_git_hash": g(d, "code_revision", "git_hash"),
        "release_freeze_sha256": d.get("release_freeze_sha256"),
        "consumed_freeze_sha256": d.get("consumed_freeze_sha256"),
        "procedure_version": d.get("procedure_version"),
        "objective_version": d.get("objective_version") or dm.get("objective_version"),
        "staged_config_hash": d.get("staged_config_hash"),
        "positions_artifact_sha256": d.get("positions_artifact_sha256"),
        "source_asset_sha256": d.get("source_asset_sha256"),
        "kernel_sha256": d.get("kernel_sha256"), "truth_kernel_sha256": d.get("truth_kernel_sha256"),
        "fit_psf_delta": d.get("fit_psf_delta"),
        "measured_truth_total_rms_nm": d.get("measured_truth_total_rms_nm"),
        "psf_fit_label": dm.get("psf_fit_label"), "psf_truth_label": dm.get("psf_truth_label"),
        "psf_fit_sha256": dm.get("psf_fit_sha256"),
        "dataset_kind": dm.get("dataset_kind"), "mask_name": dm.get("mask_name"),
        "n_unmasked_pixels": dm.get("n_unmasked_pixels") or d.get("n_unmasked_pixels"),
        "image_shape": d.get("image_shape"),
        "background_treatment": dm.get("background_treatment"),
        "generation_sub_size": dm.get("generation_sub_size"), "blurring_sub_size": dm.get("blurring_sub_size"),
        "noise_seed": d.get("noise_seed"), "noise_replicate": d.get("noise_replicate"),
        "noise_spawn_key": d.get("noise_spawn_key"),
        "sampler_seed": d.get("sampler_seed"), "seed_entropy": d.get("seed_entropy"), "seed_spawn_key": d.get("seed_spawn_key"),
        "cuda_visible_devices": d.get("cuda_visible_devices"),
        "jax_n_batch_setting": g(d, "fit_settings", "jax_n_batch"),
        "q_fit_payload": d.get("q_fit"),
        "numerical_status_payload": d.get("numerical_status"),
        "profile_decision_payload": d.get("profile_decision"),
        "marginal_q_flag_payload": d.get("marginal_q_flag"),
        "smooth_status_payload": d.get("smooth_status"), "subhalo_status_payload": d.get("subhalo_status"),
        "profile_role_statuses": d.get("profile_role_statuses"),
        "sampler_pair_status": d.get("sampler_pair_status"),
        "delta_log_evidence": d.get("delta_log_evidence"),
        "delta_log_likelihood": d.get("delta_log_likelihood"),
        "artifact_completeness_status": d.get("artifact_completeness_status"),
        "retention_roles_complete": {k: (v or {}).get("complete") for k, v in (rc.get("roles") or {}).items()} if isinstance(rc, dict) else None,
        "censored": d.get("censored"), "quality_flags": d.get("quality_flags") or case.get("quality_flags"),
        "timings": d.get("timings"),
        "metric_log_l_smooth": met.get("log_l_smooth"), "metric_log_l_subhalo": met.get("log_l_subhalo"),
        "metric_signed_delta_log_l": met.get("signed_delta_log_l"), "metric_delta_log_l": met.get("delta_log_l"),
        "metric_q": met.get("q"), "metric_clip_negative_q": met.get("clip_negative_q"),
        "metric_detected_scdd_local": met.get("detected_scdd_local"),
        "metric_threshold_q": met.get("threshold_q"), "metric_convention": met.get("convention"),
        "fisher_q_matched_case": case.get("fisher_q"),
        "fisher_z_case": case.get("fisher_z"),
        "log_l_fixed_template_point": g(case, "diagnostics", "log_l_fixed_template_point"),
        "trial_mass_msun": tr.get("mass_msun"), "trial_position_yx_arcsec": tr.get("position_yx_arcsec"),
        "trial_concentration": tr.get("concentration"), "trial_concentration_model": tr.get("concentration_model"),
        "trial_model": tr.get("model"), "trial_profile_class": tr.get("profile_class"),
        "trial_kappa_s": tr.get("kappa_s"), "trial_scale_radius_arcsec": tr.get("scale_radius_arcsec"),
        "trial_lens_redshift": tr.get("lens_redshift"), "trial_source_redshift": tr.get("source_redshift"),
        "trial_fisher_q": tr.get("fisher_q"),
        "trial_metadata_source": g(tr, "metadata", "source"),
        "recovery_log10_m200_ml": sr.get("log10_m200_ml"), "recovery_log10_m200_p16": sr.get("log10_m200_p16"),
        "recovery_log10_m200_p50": sr.get("log10_m200_p50"), "recovery_log10_m200_p84": sr.get("log10_m200_p84"),
        "recovery_centre_ml_x": sr.get("centre_ml_x"), "recovery_centre_ml_y": sr.get("centre_ml_y"),
        "recovery_concentration_ml": sr.get("concentration_ml"),
        "recovery_mass_at_lower_bound": sr.get("mass_at_lower_bound"), "recovery_mass_at_upper_bound": sr.get("mass_at_upper_bound"),
        "recovery_n_samples": sr.get("n_samples"), "recovery_pdf_converged": sr.get("pdf_converged"),
        "tangent_computed": None if lmt is None else lmt.get("computed"),
        "tangent_q": None if lmt is None else lmt.get("q"),
        "tangent_q_with_free_background_only": None if lmt is None else lmt.get("q_with_free_background_only"),
        "tangent_q_with_finite_prior_box": None if lmt is None else lmt.get("q_with_finite_prior_box"),
        "tangent_derivatives_stable": None if lmt is None else lmt.get("derivatives_stable"),
        "tangent_half_step_q_difference": None if lmt is None else lmt.get("half_step_q_difference"),
        "tangent_rank": None if lmt is None else lmt.get("rank"),
        "tangent_bounded_solver_success": None if lmt is None else lmt.get("bounded_solver_success"),
        "tangent_fixed_h1_reference_chi2": None if lmt is None else lmt.get("fixed_h1_reference_chi2"),
        "tangent_n_pixels": None if lmt is None else lmt.get("n_pixels"),
        "tangent_background": None if lmt is None else lmt.get("background"),
        "tangent_position_policy": None if lmt is None else lmt.get("position_policy"),
        "bracket_fisher_q_production_at_position": bfq.get("q_f_production_at_position"),
        "bracket_fisher_log10_m200": bfq.get("log10_m200"),
        "bracket_fisher_position_yx_arcsec": bfq.get("position_yx_arcsec"),
        "bracket_fisher_kernel_shape_native": bfq.get("kernel_shape_native"),
        "bracket_fisher_evaluator": bfq.get("evaluator"),
        "h1_anchor_source": g(d, "h1_anchor", "anchor_source"),
        "h1_anchor_sampler_executed": g(d, "h1_anchor", "sampler_executed"),
        "h1_anchor_evidence_claim": g(d, "h1_anchor", "evidence_claim"),
        "roles": {
            "smooth": role_record(fpr.get("smooth"), case.get("smooth_fit")),
            "subhalo": role_record(fpr.get("subhalo"), case.get("subhalo_fit")),
        },
    }
    att = os.path.dirname(os.path.dirname(path))
    we = os.path.join(att, "worker_exit.json")
    if os.path.exists(we):
        w = json.load(open(we))
        rec["worker_exit_status"] = w.get("status"); rec["worker_exit_elapsed_s"] = w.get("elapsed_s")
        rec["worker_exit_spec_sha256"] = w.get("spec_sha256"); rec["worker_exit_config_sha256"] = w.get("config_sha256")
        rec["worker_exit_positions_sha256"] = w.get("positions_sha256")
    pr = os.path.join(att, "production_run.json")
    if os.path.exists(pr):
        p = json.load(open(pr))
        rec["production_run_status"] = p.get("status"); rec["production_run_case_id"] = p.get("case_id")
        rec["production_run_archived_state_imported"] = p.get("archived_state_imported")
        rec["production_run_release_freeze_sha256"] = p.get("release_freeze_sha256")
    return rec

def main():
    rec = json.load(open(REC))
    out = {"source_reconciliation": REC, "source_reconciliation_sha256": sha(REC), "cases": [], "recoveries": [], "errors": []}
    for row in rec["rows"]:
        cid = row["case_id"]
        p = resolve(row["original_payload"])
        entry = {"case_id": cid, "reconciliation_original_payload": row["original_payload"],
                 "reconciliation_original_payload_sha256": row.get("original_payload_sha256"),
                 "reconciliation_status": row.get("status"), "reconciliation_original_status": row.get("original_status"),
                 "reconciliation_q": row.get("q"), "reconciliation_profile_decision": row.get("profile_decision")}
        if p is None:
            out["errors"].append({"case_id": cid, "missing": row["original_payload"]}); out["cases"].append(entry); continue
        try:
            e = extract(p); e.update(entry); e["payload_hash_matches_reconciliation"] = (e["payload_sha256"] == row.get("original_payload_sha256"))
            out["cases"].append(e)
        except Exception as ex:
            out["errors"].append({"case_id": cid, "path": p, "error": repr(ex)}); out["cases"].append(entry)
        sr = row.get("selected_recovery")
        if sr:
            rp = resolve(sr["payload"])
            rentry = {"case_id": cid, "recovery_campaign": sr.get("campaign"), "recovery_payload": sr["payload"], "recovery_q_reconciliation": sr.get("q"), "confirmation": sr.get("confirmation")}
            if rp is None:
                out["errors"].append({"case_id": cid, "missing_recovery": sr["payload"]}); out["recoveries"].append(rentry); continue
            try:
                e = extract(rp); e.update(rentry); out["recoveries"].append(e)
            except Exception as ex:
                out["errors"].append({"case_id": cid, "path": rp, "error": repr(ex)}); out["recoveries"].append(rentry)
    # sys0279 independent repeats (not population members) for the comparison record
    r4 = "/data/home/gvassilakis/nonlinear_release_20260917/stage3/sys0279_recovery_20260919/task_r4/attempts"
    out["sys0279_r4_attempts"] = []
    if os.path.isdir(r4):
        for a in sorted(os.listdir(r4)):
            for f in glob.glob(os.path.join(r4, a, "case", "nonlinear_validation_*.json")):
                try:
                    e = extract(f); e["attempt_key"] = a; out["sys0279_r4_attempts"].append(e)
                except Exception as ex:
                    out["errors"].append({"attempt": a, "path": f, "error": repr(ex)})
    json.dump(out, sys.stdout)

main()
