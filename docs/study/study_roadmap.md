# HWO-SLAPS: current RASTI study

Updated 2026-09-05 from the local code, archived campaign reviews, and the
2026-09-04 paper rulings and PSF launch record. This is the current human
orientation page; older roadmap versions remain in Git history.

## Read first

The paper asks how subhalo mass reach varies across an HWO lens sample,
and how much PSF knowledge error can be tolerated under a declared model.
Selection and source-template dependence explain the mass-reach distribution.
The known-PSF amplitude sweep and the knowledge-error experiment address
separate optical questions.

The executable design is [design_freeze_v1.yaml](../../configs/design/design_freeze_v1.yaml),
currently amendment v5. Its staged campaign versions and bound artifacts
specify each experiment. This page summarizes them; it does not amend them.
See [the venue plan](venue_plan.md) for manuscript structure and publication
boundaries. Read only the relevant spec or code path below for a focused task.

## Final study design

- Observation: one 2000 s visit, EAC1/HRI UVIS, 0.00716 arcsec pixels,
  throughput 0.21, and SEI v0.1.9 detector inputs. The production truth
  PSF is the frozen science35 realization at 35 nm RMS, modeled correctly.
- Population: 1000 SIE lenses with five COSMOS source templates; 965 survive
  the declared cuts. The reference 48 are a template-stratified design
  sample. The selected 12 are ranked from all survivors, with five golden
  members inside them. One reference/selected overlap gives 59 unique systems.
- Validation ladders: 100 independently selected validation members plus
  six additional historical control systems, 106 distinct additional systems.
- Subhalos: lens-plane untruncated NFW, mass convention M200, with the frozen
  concentration relation. The 0.05 arcsec position grid lies inside 2 theta_E.
- Estimands: M_best is the first position crossing; M10 and M50 reach 10%
  and 50% of the aperture area. Uncrossed thresholds at log10(M200/Msun)=9.5
  are right-censored. Censoring-aware summaries are primary.
- Statistics: q_F >= 10 is the Fisher screening convention. Nonlinear fits
  report q_fit and Delta log Z separately. A Fisher-equivalent sqrt(q)
  does not establish a calibrated significance for a freed search.

## Results and campaign status

These are recorded artifact states, not a live cluster monitor.

| Component | Verified or last recorded state | Interpretation |
|---|---|---|
| Reference and selected ladders | Complete, 48 + 12 campaign members | Median log10 M10: reference 9.151 with censoring, selected 7.848; about 1.30 dex contrast |
| Known-PSF amplitude sweep | Complete; selected 12 at 0, 20, 35, 50, 65 nm | Median thresholds shift by 0.12–0.16 dex along this scaled wavefront realization |
| Validation ladders and selector analysis | Complete; comparison uses the unenriched validation 100 | Report mass reach under the physically motivated R selection; practical ranking from noisy images remains unvalidated |
| Nonlinear validation v1 | Archived review CLEAN, 245/245 fit pairs | Targeted crossing tests, controls, and golden-five bridge/replicate arms |
| Nonlinear null extension | Archived review CLEAN, 531/531 new pairs | With 59 original controls: 3/590 exceed q_fit >= 10; 0/590 exceed Delta log Z > 5 |
| Nonlinear validation-100 | Archived review CLEAN, 389/389 pairs | At the first positive rung, 61/89 recovered in each injected arm; below-rung Asimov consistency 87/89 |
| PSF knowledge-error block | Launch recorded 2026-09-04 22:10 UTC; completion not verified here | 588 Fisher jobs / 1764 maps, then 288 nonlinear pairs; read both campaign reviews before quoting results |

The knowledge block uses the selected 12, paired drift-prior residuals at
0, 1, 2, 5, 10, 20, 35 nm, eight Fisher directions, and each system's
M_best/M10/M50 bracket-top rungs. Nonlinear injected and control arms use
2, 5, 10, 20 nm and three directions. The 35 nm point is an endpoint anchor.
The Fisher gates are 90% area retention and a spurious-area ratio <= 0.1,
using directional quantiles and a 33-cell denominator floor. These area
ratios are distinct from nonlinear control exceedance frequencies.

## Claim boundaries

- This is a foreground-free information-ceiling forecast: lens light is
  omitted, and nonlinear source fits reuse the truth image asset with
  limited transformations. It is not unrestricted source reconstruction.
- The reference ensemble is a conditional design distribution. The selected
  tier is an idealized no-subhalo truth-proxy selection, not a validated
  selector for noisy survey images. Five templates support template-dependent
  comparisons, not a universal morphology law.
- Nonlinear rendering and fitting use a 51x51 kernel with support-matched
  Fisher comparisons; production Fisher ladders use 999x999 kernels.
- The null calibrates the declared targeted, truth-centred search. Its pooled
  per-draw interval assumes independence; the recorded per-system summary
  must accompany it. It is not a blind-search false-positive rate.
- PSF tolerances are conditional on the selected tier, residual shape, sampled
  directions, and gates. They are not a final HWO engineering budget or a
  temporal stability specification. No quality/knowledge interaction campaign
  or continuous PSF-nuisance marginalization is claimed.
- Counts/yields and Despali-equivalent threshold claims are withheld.
  SPIE numerical results cannot validate RASTI: the brightness-bug ruling
  makes the submitted SPIE paper citation-only heritage.

## Remaining paper work

- Verify and archive the PSF knowledge-error harvests when available, then
  derive their figures and conditional conclusions.
- Draft the approved manuscript skeleton with production results and the
  completed nonlinear/selector analyses; the two nonlinear extension harvests
  have landed, so that earlier drafting gate is satisfied.
- Complete nonlinear and knowledge-error figures and use M200 consistently
  on mass axes. Resolve the co-reader correspondence and submission package.
- Check figure/table provenance, include the campaign-driver snapshots needed
  for reproduction, and complete co-reader/JPL review.

The September 2 prune already removed the unused count-fold, PSF-bank,
flexible-macro, synthetic-source, and diagnostic physics branches. Its
execution spec is history, not an outstanding task. A broad FisherDetector
split and a rasti-to-main PR remain deferred; this page authorizes neither.

## Code map: load the path needed for the task

| Task | Entry points and implementation |
|---|---|
| Render a scene and observation | `runner.py`, `pipeline.py`; `src/hwoslaps/{lensing,psf,observation}/` |
| Statistical profiling and mismatch algebra | `src/hwoslaps/modeling/fisher_core.py`; `tests/test_fisher_core.py` |
| Construct nuisance images and sensitivity maps | `modeling/fisher_detector.py`, `modeling/fisher_grid_jax.py`; grid-map and detector tests |
| Population, selection, and mass ladders | `campaign/{design_freeze,stage0,ladder,ladder_walk}.py`, `analysis/{selection_score,rank_stability}.py`; `scripts/run_ladder.py` |
| Nonlinear model and data conventions | `modeling/nonlinear/{autolens_model_builder,dataset_builder,mass_mapping,autolens_runner}.py` |
| Nonlinear campaign identity and results | `scripts/{generate_nonlinear_validation_campaign,run_nonlinear_validation,harvest_nonlinear_validation}.py` |
| PSF knowledge-error campaign | `scripts/{generate_psf_knowledge_campaign,run_psf_knowledge_map,harvest_psf_knowledge}.py`, `scripts/psf_knowledge_launch.sh`; `tests/test_psf_knowledge.py` |

Unqualified implementation paths in the table are under `src/hwoslaps/`.
The production execution layer also includes historical `t12_drivers` on
xtx; those are not versioned package APIs. The documented choice to keep
that layer separate does not remove the need to archive reproduction inputs.

## Local records and operational boundary

- Manuscript: `scratch/rasti_manuscript/`, a separate Git repository. The
  approved starting point is the Aug 25 skeleton plus the author's title edit;
  the Aug 26 production draft is stashed and must not be restored implicitly.
- Archived data: `../rasti/campaigns/`, including nonlinear `harvest/review.json`
  and `reporting_v1/selector_validation/selector_validation.md`.
- Current campaign root on xtx: `/data/home/gvassilakis/hwo-slaps-campaigns`.
  The September 4 launch used GPUs 0–3, one worker per GPU. Check live state
  before operational work; do not move a checkout beneath an active fleet.
- Tests and experiments run on xtx. Local environment repair is not part of
  paper preparation. Long GPU launches require the author's explicit scope.
- `scratch/README.md` indexes local records. Dated briefs and build reports
  describe their original stage; they are not additional current roadmaps.
