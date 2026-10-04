# HWO-SLAPS: canonical writing assessment

Updated 2026-09-19. Purpose: compact entry point for fresh writing agents. Read this before the older outline, plans, or result summaries. This is a synthesis of the completed results, GPT Pro's final assessment, the historical Fable-review resolution, and Codex's independent scalar checks. It is not a new scientific analysis or authorization to run experiments.

Writing update 2026-09-21: the outline's Sections 3–7, number bank and evidence
map now reflect this assessment and the literature guide. The additional
selection/PSF context below was checked against existing saved reductions;
no scientific fits or manuscript-prose edits were made.

## Decision and scientific argument

Proceed with a **Fisher mass-reach forecast paper with targeted nonlinear likelihood validation and conditional PSF-knowledge results**. The completed experiment supports writing after the reporting corrections below. It does not require another census or automatic attempts to resolve every control.

The argument is: sensitivity varies substantially across lens configurations; selecting informative targets improves forecast reach; nonlinear fits support the Fisher predictions on those selected targets while revealing failures elsewhere; PSF knowledge errors alter recovery and control behavior within the tested residual family.

Keep the distinction between measurements and interpretation. Selection is associated with a well-performing regime; it has not been shown to cause Fisher accuracy. The title should give mass reach and selection appropriate weight. “PSF stability requirements” needs qualification because static modeling residuals do not directly specify temporal stability or an engineering allocation.

## Verified result bank

All mass entries are log10(M200/Msun). Fisher medians use the frozen censoring-aware inverse empirical CDF, not an averaged even-sample median. M10/M50 denote 10%/50% sensitive aperture coverage, not noisy recovery probabilities. M_best is favorable-position Fisher reach.

| Fisher cohort | N | M_best | M10 | M50 |
|---|---:|---:|---:|---:|
| Full surviving ensemble | 965 | 8.622 | 9.112 | 9.356 |
| Top 50 by R | 50 | 7.509 | 8.116 | 8.542 |
| Original top 12 by R | 12 | 7.189 | 7.818 | 8.302 |

Full-ensemble censored counts: 142/269/377 for M_best/M10/M50; retain them in denominators. Top-50 versus ensemble M10 gain is approximately 1 dex. These are 965 configurations of five fixed galaxy templates, not 965 independent morphologies. Top50 contains no smooth-disk template.

Author decision 2026-09-22: present R as a physically motivated selection
score and report the reach of its selected samples. SNR-only comparisons and
component ablations are outside the paper's scope, with no deferred writing
task or requirement to justify R through them. The approximately one-dex gain
is the comparison between the R-selected top50 and the full ensemble.

Nonlinear statistic: signed q = 2(logL_H1 − logL_H0); detection requires numerical acceptance and q >= 10. State any display clipping separately. Final maxima come from profiling after fresh sampling. Evidence and posterior estimates remain distinct.

Author decision 2026-09-23 (final design): the paper reports the top 50 by R
and the validation sample, now the first 100 of all 965 survivors in SHA-256
order of the system ID (7 are also in the top 50; 3 come from the former
reference set), plus ten no-subhalo draws on each top-50 system. The legacy
44 reference systems and null590 are out of the paper; their archives remain.
Reduction: `scratch/figure_builds/results_20260922/final_design_tables.py`.

| Nonlinear view | Upper Asimov detections | Noisy injection detections | Nominal control exceedances |
|---|---:|---:|---:|
| Top50 | 50/50 | 38/50 | 1/48 accepted; 2 unresolved |
| Validation 100 | 76/88 | 63/88 | 0/97 accepted; 3 unresolved |
| Top50 plus validation (143) | 119/131 | 96/131 | 1/139 accepted; 4 unresolved |
| Selected12 | 12/12 | 11/12 | 0/12 |

Detection denominators are production-Fisher-positive upper rungs.

- Final campaign: 1,601 unique canonical cases; 1,588 accepted, 13 unresolved. All unresolved cases are noisy controls. Recovery upgraded 26 outcomes, not the population size. Views overlap; never sum their denominators. The mixed total of 486 detections is not a recovery fraction. These campaign totals include the legacy views.
- Of the 143 upper Asimov cases, 131 are production-Fisher positive and 119 of these pass nonlinear q. All 12 misses lie outside top50, all at log mass >= 9. The severe example is sys0109 (qF 12.937 → q 0.0919); sys0882 is a legacy reference system and is out of the paper. Small admissible H0 residuals can bound the achievable Asimov contrast; these failures need not be attributed to missed H1 maxima.
- Top50 upper-Asimov median q−qF: +0.412 against production Fisher, −0.102 against support-matched Fisher. Production uses 999×999 PSF support; nonlinear/support-matched calculations use 51×51. Equal support alone does not match every likelihood convention. The fully likelihood-matched tangent agrees within 0.0423 in q on the selected12's 24 upper/lower Asimov cases.
- Top-50 repeated controls (`scratch/null500_top50_20260923/RESULTS.md`): 500 draws, 493 accepted, 7 unresolved, 3 accepted exceedances in 3 of 50 systems; q95 6.73, q99 8.83. Ten draws reuse each lens in a targeted local search. Missing-outcome sensitivity is 3/500–10/500 (0.6%–2.0%), not a confidence interval or unconditional false-positive bound. Every originally unresolved draw had the same single recovery pass (the 9 new ones on 2026-09-23, `scratch/null500_recovery9_20260923`: 5 accepted, 4 still unresolved, q unchanged). Top-12 subset (PSF comparator): 1/119 accepted, 1 unresolved, q95 6.33, q99 8.10. Retain the declared q=10 screen.

## PSF result and its scope

The separate correctly modeled 0–65 nm sweep gives cohort-median increases of
0.159/0.145/0.132 dex for M_best/M10/M50. Author decision 2026-09-23: the sweep
is summarized with the Table 3 inverse-empirical-CDF median, so its 35 nm point
equals the top-12 row of Table 3 (per-lens values are identical to v6). These
are differences of cohort medians computed before rounding, not median paired
shifts; per-lens 0–65 nm increases are 0.08–0.21 (M_best), 0.12–0.26 (M10) and
0.11–0.21 dex (M50). The archived harvest's averaged-median values
(0.120/0.163/0.126) are superseded for writing. Source:
`rasti/campaigns/psf_sweep_v1/harvest/` in the neighboring HWO checkout, linked
as E5 in the outline; Figure 7 is rebuilt by
`scratch/figure_builds/results_20260922/build_superbit_style.py` (`quality`).

The truth PSF stays fixed; amplitudes describe fitting-wavefront residual RMS in the declared drift-shaped family. The selected12 are reused across three directions and amplitudes.

| Residual RMS | Injected detections | Accepted control exceedances | Lost nominal detections |
|---|---:|---:|---:|
| 2 nm | 33/36 | 0/35; 1 unresolved | 0 |
| 5 nm | 32/36 | 0/36 | 1 |
| 10 nm | 32/36 | 0/35; 1 unresolved | 1 |
| 20 nm | 28/36 | 1/36 | 5 |

The same condition, sys0069 direction 1, is lost at 5 and 10 nm. At 20 nm all three sys0069 directions and sys0533 directions 2/3 are lost. sys0429 is already nominally undetected; no new detections are gained. Some mismatched q values increase, which does not establish improved sensitivity.

Write “similar aggregate recovery at 2–10 nm, with one loss already at 5 nm,” not “unchanged through 10 nm.” Author decision 2026-09-23: Table 5 reports the retained-location fraction K (share of the correct-PSF sensitive cells still sensitive with the wrong PSF; same 0.9/0.1 gate on Q10[K] and Q90[F]), because the declared total-area ratio R lets cells gained at new positions offset cells lost. Location-gate median passing amplitude: 5 nm for M10 (range 2–10) and 5 nm for M50 (2–10). The declared R gate gives 10 nm (5–20) and 5 nm (2–10) and is described in 5.2. Table 5 medians use the inverse empirical CDF; its earlier R cells were averaged medians. Reduction: `scratch/figure_builds/results_20260922/psf_retention_table.py`. These Fisher tolerances remain separate conditional forecasts. The nonlinear point tests do not validate nonlinear area tolerances. Identical seeds do not imply identical Poisson noise when the truth image changes.

## Required reporting corrections; no new fits implied

1. **Crossings:** in the final design (top50 plus validation 100), 131 paired rungs include 8 changed-position pairs and 34 without an upward nonlinear bracket. Require identical coordinates and q_below < 10 <= q_top for both nonlinear and the named Fisher comparator. Valid production comparisons: 94 overall/40 top50/10 selected12; median shifts −0.01237/−0.01237/−0.01287 dex. Support-matched: 87/39/10; +0.00373/+0.00261/+0.00243 dex. (The legacy 187-system values were 170 rungs, 116 and 105.) These are conditional two-point log-q/log-mass interpolations, not precision thresholds or population corrections. Report excluded categories. Retire the figure transferring these shifts to M_best/M10/M50.
2. **Upward tests:** all 36 physical-mass tests at +0.1/+0.2/+0.3 dex detect (q=13.97–33.51). They demonstrate detection at tested higher masses, not onset “right-censored above +0.3 dex.” Ten selected12 systems already have valid same-position brackets. Additional endpoints for sys0264/sys0536 matter only if every selected lens must have an individual onset estimate.
3. **Recovered parameters:** exported ML mass/center come from the sampler before profiling replaces the likelihood maximum. Label them as sampler estimates, or extract saved candidate_best_vector with verified parameter order. Keep posterior quantiles separate. Do not write profiled mass-bias claims from the existing sampler column.
4. **Evidence/acceptance:** preserve the six-case evidence-audit HOLD; it does not invalidate stable q results or certify Bayes factors. The 0.1-logL repeatability gate is not a guaranteed global q error bound. Keep unresolved controls unclassified. Regenerate publication figures from final tables; GPT Pro reported four uploaded PNG hash mismatches, with underlying text data matching.

## Writing map and boundaries

Editorial update, 2026-09-21: Sections 3–7 of the writing outline now
follow the [whole-paper literature comparison](LITERATURE_WRITING_GUIDE.md).
Organize methods around the scientific questions and reported quantities;
place detailed numerical checks and auxiliary experiments according to what
they establish. Neither a particular test nor a terminology checklist should
dictate the narrative. The result bank and claim boundaries above are unchanged.

Retain the seven agreed sections:

1. Introduction: heterogeneous reach, target selection, two PSF questions.
2. Simulation suite: completed 965 ensemble, top50 comparison, top12 PSF tier. Replace the historical 165-system narrative. Ranking uses perfect-PSF 51×51 expected images; science forecasts use science35 999×999.
3. Metrics: Fisher statistic → spatial thresholds/censoring → executed nonlinear method, comparator conventions and controls.
4. Sensitivity: ensemble reach → selection depth → selected-cohort agreement and wider-cohort failures.
5. PSF: correctly modeled degradation → area response to knowledge errors → paired nonlinear outcomes.
6. Discussion: implications and limitations of the adopted model family.
7. Conclusions: selection gain, bounded validation, conditional PSF sensitivity.

The results outline and number bank have been refreshed; earlier author-drafting
notes in Sections 1–2 remain context, not scientific status. Much of the drafted
introduction/methods remains usable. Keep compute history, hashes, canaries and
exhaustive checks in supporting records. Preserve author prose unless editing
is requested. The revised outline itself has not yet received GPT Pro review.

Essential limits: foreground-free images; known internal source morphology and rotation with center/flux/size freedom; five templates; restricted macro model; local subhalo-position/mass priors; untruncated lens-plane NFW with the adopted Moline prescription; monochromatic PSF; finite, dependent noise/direction samples. No nonlinear M10/M50, global-search false-positive probability, calibrated Bayes-factor detection, population yield, or particle constraint follows from these runs.

## Evidence and review precedence

Final primary outcomes and declarations outrank old plans/status files. GPT Pro's 19 September corrections supersede conflicting prose in the original result summaries; Codex independently reproduced canonical counts, q reconstruction, cohort outcomes, valid crossing counts/medians and paired PSF losses from the text tables. This was a scalar verification, not a fresh fit or full remote archive audit.

Fable's historical review, as recorded in the 17 September resolution, usefully emphasized executable declarations, sampling/objective identity, explicit case bindings and valid reuse. Some proposed mechanisms required correction. It is a preproduction review, not separate final-results certification; do not resurrect its historical blockers from dated notes. The completed harvest and latest review establish current writing status.

Read further only for the claim being drafted (paths relative to this repository):

- Final review: `/Users/vassig/Downloads/HWO_Nonlinear_Final_Assessment_2026-09-19/REVIEW.md`; estimator trace and audit outputs in that directory.
- Final primary accounting: `scratch/nonlinear_bulk_20260918/final_harvest/FINAL_SUMMARY.md` and `FINAL_RECONCILIATION.json`.
- Canonical scalar tables: `scratch/nonlinear_final_review_20260919/HWO_Nonlinear_Evidence_Handoff_2026-09-19/tables/`. Its prose summaries contain the corrected errors above.
- Fisher: `scratch/v6_full_pool/production_ab_20260915/REPORT.md`; numerical harvest under `evidence/v6_fisher_production_20260915/harvest/`.
- Fable resolution: `scratch/nonlinear_release_plan_20260917/FABLE_REVIEW_RESOLUTION.md`.
- Manuscript/outline: `scratch/rasti_manuscript/main.tex`, `WRITING_OUTLINE.md`.

Additional fits are claim-dependent future work, not a prerequisite to this bounded paper. Update this assessment when verified results or author decisions change; do not accumulate competing “canonical” summaries.
