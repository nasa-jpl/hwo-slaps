# Study roadmap: HWO-SLAPS PSF stability and dark-matter subhalo detectability

## Scientific spine

Core question:

> Given an HWO-like strong-lensing scene, which PSF error modes and stability amplitudes materially degrade the detectability of low-mass dark-matter subhalos?

This should remain one integrated study with two maturity levels:

- **SPIE:** a controlled first forecast and proceedings-quality demonstration.
- **RASTI:** the calibrated, broader archival study that turns the forecast into requirement-style conclusions.

The two science axes are inseparable:

1. **Mass reach:** how low in subhalo mass HWO can recover, summarized by local detection metrics, ring/sensitivity maps, and detectable-area proxies.
2. **PSF stability:** how that mass reach degrades under physically meaningful PSF perturbations, summarized by PSF-mode coupling, degradation curves, and tolerance-style plots.

The SCDD anchor is the smooth-model versus subhalo-model log-likelihood criterion. Use the SCDD convention that a halo is detectable when

$$
\Delta \log \mathcal{L} > 5
$$

which corresponds to

$$
q = 2\Delta \log \mathcal{L} > 10, \qquad Z > \sqrt{10}
$$

under the usual likelihood-ratio convention.

## Current state

The four core HWO-SLAPS modules are in a good position for a bounded SPIE study:

- **Lensing:** deterministic strong-lens scenes with optional PointMass, SIS, and NFW subhalo injection.
- **PSF:** segmented HWO-style PSFs with segment-level and global aberration modes.
- **Observation:** PSF convolution and detector-noise simulation.
- **Modeling:** Fisher/Asimov subhalo detectability with local and Einstein-ring map modes.

The missing layer is not the core physics module layer. The SPIE-specific
study layer has been moved out of the main repo into the sibling personal
archive `../spie/`. That archive keeps the canonical config, study manifests,
study scripts, packaged results, run provenance, manuscript source, and poster
source used for the SPIE submission. For RASTI, reusable pieces still need to
be promoted from the archived scripts and packaged outputs into supported,
journal-grade analysis tooling:

- canonical SPIE/SCDD configs,
- sweep manifests,
- cross-run aggregation,
- publication figures,
- detectable-area or detectable-ring summaries,
- reproducibility metadata,
- PyAutoLens nonlinear evidence validation for SPIE,
- discrete PSF-bank marginalized nonlinear validation,
- broader scene/source/PSF-nuisance validation for RASTI.

## Stage 0: one-week internal-review priority

Goal:

Produce a credible internal-review poster package quickly without trying to complete the full paper.

Minimum internal-review deliverables:

- [x] One canonical SCDD-like baseline scene.
- [x] One perfect-PSF mass sweep.
- [x] One PSF perturbation family with an amplitude ladder.
- [x] One ring-map / detectable-ring-fraction demonstration.
- [x] One clear table of metric definitions: `q_F`, `Z_F`, `Delta log L_F,equiv`, and threshold pass/fail.
- [x] One limitations box that says the SPIE version is a first forecast, not final HWO engineering requirements.

Recommended poster logic:

1. SCDD asks for PSF stability because low-mass subhalo detectability is PSF limited.
2. HWO-SLAPS maps optics perturbations into a lensing detection statistic.
3. Perfect-PSF simulations define mass reach.
4. Perturbed-PSF simulations show how mass reach or detectable-ring fraction degrades.
5. PyAutoLens nonlinear validation calibrates the controlled SPIE forecast.
6. The RASTI study will expand this into calibrated requirement curves.

## Stage 1: SPIE-level codebase hardening

Goal:

Make the current pipeline produce repeatable proceedings-level results.

Status note: the checked-off study-runner and aggregation items below describe
the submitted SPIE archive, not supported main-branch APIs retained in this
merge branch.

Checklist:

- [x] Create a canonical SCDD/SPIE baseline config, separate from `configs/master_config.yaml`.
- [x] Use lens redshift `z_l = 0.2` and source redshift `z_s = 0.6` for the SCDD-anchored run, unless a different source redshift is explicitly justified in the manuscript.
- [x] Set the canonical grid, pixel scale, wavelength, aperture, exposure/noise convention, subhalo model, concentration model, and source morphology in one named config.
- [x] Use the SCDD anchor masses for the headline SPIE grid: `1e7`, `10^7.25`, `10^7.5`, `10^7.75`, and `1e8 Msun`.
- [x] Treat masses below `1e7 Msun` as exploratory unless calibrated by
      nonlinear fits. The archived canonical-scene `10^6.5` and
      `10^6.75 Msun` points have PSF-bank marginalized nonlinear validation
      and should be described as calibrated for that setup, not as final HWO
      requirements.
- [x] Define a perfect-PSF baseline.
- [x] Define one required PSF perturbation family for SPIE.
- [x] Define one optional second PSF perturbation family for SPIE.
- [x] State the amplitude units unambiguously for the Stage 0 PSF family: OPD nm, mirror-surface nm, outgoing-beam microradians, or coefficient RMS.
- [x] Archive the SPIE manifest format in `../spie/study_scripts/spie_study/`.
- [x] Archive the SPIE study-runner and validation workflow through `../spie/`
      plus git history at the archived `spie` commit.
- [x] Package cross-run aggregation in `../spie/data_package/spie_draft_results/`.
- [x] Record at least: run name, config hash, git hash if available, mass, subhalo model, subhalo position, PSF family, PSF mode, PSF amplitude, seed, local `q_F`, local `Z_F`, `Delta log L_F,equiv`, threshold pass/fail, map median `Z_F`, map max `Z_F`, detectable-ring fraction, profiling degradation, nuisance count, and PSF quality metrics.
- [x] Store SPIE-needed PSF diagnostics: Strehl, WFE/OPD RMS if available, kernel shape, kernel sum, kernel peak, and FWHM.
- [x] Add remaining diagnostic extras: raw peak ratio before clipping and a simple kernel-difference norm relative to the perfect PSF.
- [x] Archive SPIE plotting scripts in `../spie/study_scripts/spie_study/` and
      `../spie/manuscript/`.
- [x] Archive provenance beyond `config_used.yaml`: git hash, config hash,
      package versions, Python version, and command line.
- [x] Make the output path portable. Avoid absolute user-specific paths in canonical configs.

Minimum SPIE code outputs:

- canonical config archived at `../spie/configs/scdd_spie_baseline.yaml`,
- sweep manifests archived at `../spie/study_scripts/spie_study/`,
- generated per-run configs and run summaries archived under
  `../spie/provenance/run_summaries/`,
- aggregate results tables archived under
  `../spie/data_package/spie_draft_results/csv/`,
- figure-generation scripts archived in `../spie/study_scripts/spie_study/`
  and `../spie/manuscript/`,
- reproducibility metadata archived under `../spie/provenance/`.

## Stage 2: SPIE-level study

Goal:

Run a bounded study that supports the SPIE abstract without overclaiming final HWO requirements.

### Minimum scope for the internal review / poster

- One canonical strong-lens scene.
- Perfect-PSF mass sweep at the SCDD anchor masses.
- One PSF perturbation family with one amplitude ladder.
- Fisher/Asimov local detectability for all runs.
- One Fisher ring-map around the Einstein ring, reported as a detectable-ring fraction rather than a full 2D detectable area.
- No final engineering requirement language.

### Nominal SPIE proceedings scope

- One canonical strong-lens scene.
- Masses: `1e7`, `10^7.25`, `10^7.5`, `10^7.75`, and `1e8 Msun` as the SCDD-anchored mass ladder.
- Optional low masses: `10^6.5` and `10^6.75 Msun` have archived
  canonical-scene PSF-bank marginalized nonlinear validation; `1e6 Msun`
  remains exploratory unless separately validated.
- Perfect-PSF baseline.
- One or two PSF families:
  - segment piston or selected segment hexike modes,
  - low-order global Zernikes.
- One amplitude ladder per PSF family.
- Fisher/Asimov forecasts for the headline sweep.
- PyAutoLens local-search nonlinear evidence validation.
- Matched-PSF and wrong-PSF no-subhalo control grids.
- Full noisy PyAutoLens local-search validation.
- High-`n_live` noisy threshold-disagreement checks.
- Discrete PSF-bank marginalized nonlinear validation.
- PSF-bank marginalized mass-completeness curve for `10^6.5`,
  `10^6.75`, `10^7`, `10^7.25`, and `10^7.5 Msun`.

### Stretch scope only if the above is already working

- A second source morphology.
- A second lens/source geometry.
- A small seed set.
- A sparse 2D map rather than ring-only positions.

## Detection-metric convention

Use one notation consistently across plots, docs, and manuscript:

- Profiled Fisher statistic: $q_F \equiv \Delta \chi^2_F$.
- Fisher-equivalent local significance: $Z_F = \sqrt{q_F}$.
- Fisher-equivalent SCDD likelihood metric: $\Delta \log \mathcal{L}_{F,\mathrm{equiv}} = q_F / 2$.
- SCDD threshold: $\Delta \log \mathcal{L} > 5$, equivalent to $q_F > 10$ and $Z_F > \sqrt{10}$.
- Nonlinear validation statistic: $q_{\mathrm{fit}} = 2(\log \mathcal{L}_{\mathrm{subhalo}} - \log \mathcal{L}_{\mathrm{smooth}})$.

Do not call the Fisher value a nonlinear likelihood ratio. Call it a Fisher-equivalent or Asimov forecast. The current PyAutoLens validation supports using it as a calibrated screening metric for the controlled matched-PSF SPIE study, not as a universal requirement metric.

## SPIE study questions

- [x] Does detectability improve monotonically with subhalo mass in the perfect-PSF baseline?
- [x] Does the perfect-PSF mass floor qualitatively match the SCDD expectation near `10^7` to `10^7.25 Msun` for an HWO-like high-resolution setup?
- [x] Which selected PSF modes couple most strongly to subhalo-like residuals?
- [x] At what approximate PSF amplitude does the local detection statistic or detectable-ring fraction degrade materially?
- [x] What fraction of the sampled Einstein ring remains above the SCDD threshold?
- [x] Does `q_F` track `q_fit` for the PyAutoLens local-search validation grid?
- [x] Does the nonlinear PSF-bank marginalized evidence produce a coherent
      mass-completeness curve?
- [x] Can HWO-SLAPS produce preliminary requirement-style curves from optics perturbations to subhalo detectability?

## SPIE study figures

Required:

- [x] Example lensing scene and subhalo residual.
- [x] Example perfect and perturbed PSF diagnostic.
- [x] Detection statistic versus subhalo mass.
- [x] Detection degradation or detectable-ring fraction versus PSF amplitude.
- [x] PSF-mode coupling or tolerance-style plot.
- [x] Fisher ring-map or detectable-ring-fraction figure.

Validation figures:

- [x] Fisher-versus-nonlinear calibration plot showing `q_F` against `q_fit`, with the SCDD threshold `q=10`.
- [x] No-subhalo plus PSF-mismatch false-positive diagnostic.
- [x] Noisy PyAutoLens pilot diagnostic.
- [x] Full noisy Fisher-versus-nonlinear validation diagnostic.
- [x] PSF-bank marginalized mass-completeness curve.

## SPIE claim boundary

Use:

> We present HWO-SLAPS, an end-to-end framework that converts HWO-like PSF perturbations into preliminary low-mass subhalo detectability forecasts. We show first mass-reach and PSF-mode degradation curves, calibrated against PyAutoLens nonlinear evidence for the controlled SPIE grid.

Also supported for SPIE:

> In the canonical scene, discrete PSF-bank marginalized PyAutoLens evidence
> gives a steep nonlinear mass-completeness curve: `3e6 Msun` subhalos are
> generally not recovered, `5-10e6 Msun` spans the transition region, and
> `>= 1.8e7 Msun` is recovered at near-saturated completeness.

Avoid:

> We derive final HWO engineering requirements.

Avoid overextending the validation:

> The Fisher statistic is universally calibrated across all scenes, source morphologies, PSF nuisance models, and noisy realizations.

## PyAutoLens validation scope

### SPIE status

PyAutoLens nonlinear modeling became a **core SPIE validation layer** rather
than a stretch goal. The controlled SPIE study is archived in `../spie/` and
has:

- a full matched-PSF local-search evidence grid,
- full matched-PSF no-subhalo controls,
- full wrong-PSF no-subhalo controls,
- `n_live=800` convergence checks,
- a compact noisy PyAutoLens pilot,
- full noisy injected-subhalo and no-subhalo control grids,
- high-`n_live` noisy threshold-disagreement reruns,
- discrete PSF-bank marginalized model comparison,
- amplitude-matched `1e7 Msun` PSF-bank no-subhalo controls,
- a five-point PSF-bank marginalized mass-completeness curve.

Minimum validation output:

- table of `q_F`, `q_fit`, evidence, and pass/fail relative to `q=10`,
- matched-PSF false-positive control summary,
- wrong-PSF false-positive control summary,
- noisy-pilot robustness summary,
- full noisy-validation summary,
- PSF-bank marginalized control summary,
- PSF-bank marginalized mass-completeness summary,
- statement that the Fisher metric is calibrated as a screening forecast for
  the controlled matched-PSF setup.

Current SPIE validation anchors:

- Asimov matched-PSF local search: `211/370` injected detections and `0/63`
  matched-control false positives.
- Asimov wrong-PSF controls: `49/63` false positives by `q_fit >= 10`, with
  `47/63` also passing `Delta logZ > 5`.
- Full noisy local search: `217/370` injected detections versus `215/370`
  Fisher-threshold forecasts, with `0/63` matched-control false positives and
  `48/63` wrong-PSF false positives by `q_fit >= 10`.
- PSF-bank `1e7 Msun` ensemble: paired injected detections `222/289` by
  `q_fit_psf_profile >= 10` and `209/289` by `Delta logZ_psf_marg > 5`, with
  matched no-subhalo controls `0/289`.
- PSF-bank mass completeness by `Delta logZ_psf_marg > 5`: `2.8%` at
  `3.16e6 Msun`, `30.1%` at `5.62e6 Msun`, `72.3%` at `1e7 Msun`, `97.2%`
  at `1.78e7 Msun`, and `100.0%` at `3.16e7 Msun`.

### RASTI recommendation

PyAutoLens nonlinear validation remains mandatory for RASTI, but the RASTI task
is now expansion rather than first validation. It should span additional scenes,
positions, PSF states, source morphologies, and no-subhalo false-positive cases,
replace or extend the discrete PSF bank with a broader nuisance treatment where
feasible, and fit or bound a calibration relation such as

$$
q_{\mathrm{fit}} \simeq \alpha q_F
$$

or a monotonic mapping with uncertainty.

## Known SPIE limitations to state explicitly

- The SPIE source model may be smoother than the clumpy Ly-alpha emitter source assumed in the SCDD sensitivity discussion.
- Ring maps are a lightweight sensitivity-map surrogate, not a replacement for full 2D detectable-area maps.
- Lens-plane subhalos are modeled first; line-of-sight halos are deferred or treated as a caveat.
- Lens galaxy light and lens-light subtraction residuals are not yet part of the controlled SPIE forecast unless specifically added.
- Fisher/Asimov statistics are local forecasts. The controlled SPIE validation
  supports them as screening metrics, and the archived discrete PSF-bank
  validation gives the strongest SPIE canonical-scene nonlinear mass reach.
  Final requirement claims still require broader nonlinear validation,
  source-scene variation, full 2D maps, and expanded PSF-nuisance treatment.

## Acceptance checks before using SPIE plots

- [x] Perfect-PSF `q_F` increases with subhalo mass for the same position and setup.
- [x] The chosen detection threshold is shown on every relevant plot.
- [x] `Delta log L_F,equiv = q_F / 2` is computed consistently.
- [x] A zero-aberration PSF run is reproducible.
- [x] PSF amplitude units are stated in the figure captions.
- [x] The code records enough provenance to rerun every plotted point.
- [x] No-subhalo PSF-mismatch case does not produce unexplained threshold-level detections, or the issue is highlighted as a result rather than hidden.
- [x] PSF-bank marginalized controls remain clean for the canonical `1e7 Msun`
      no-subhalo ensemble.

## Stage 3: SPIE manuscript and poster

Goal:

Submit a real full-length SPIE proceedings manuscript and poster, then archive
the exact submitted artifacts and supporting results outside the main repo.

Status update, 2026-07-06:

The SPIE paper and poster have been submitted. The submitted manuscript source,
compiled PDF, final poster PDF/PPTX, manuscript figures, packaged result CSVs,
plot products, provenance, and reproduction instructions are archived under
`../spie/`. The main repo should treat those files as submitted study artifacts,
not supported package APIs.

Manuscript spine:

- [x] Motivation from the SCDD PSF-stability future-work case.
- [x] HWO-SLAPS pipeline overview.
- [x] Fisher/Asimov detection metric and SCDD threshold mapping.
- [x] Canonical experiment setup.
- [x] First mass-reach forecast.
- [x] First PSF-mode degradation forecast.
- [x] PyAutoLens nonlinear validation status.
- [x] PSF-bank marginalized nonlinear mass-completeness result.
- [x] Limitations and path to the RASTI study.

Submitted manuscript archive:

- source and compiled PDF: `../spie/manuscript/main.tex` and
  `../spie/manuscript/main.pdf`,
- final manuscript figures: `../spie/manuscript/figures/`,
- complete manuscript working copy:
  `../spie/SPIE_Proceeding_HWO_SLAPS_Draft_work/`.

Poster status update, 2026-06-12:

The final poster is locked as `SPIE_Poster_JPL_FINAL.pdf` / `SPIE_Poster_JPL_FINAL.pptx`. It uses the final five-figure structure: canonical lensing scene, segmented-pupil/hexike PSF assumptions, noisy PyAutoLens validation, PSF-bank mass completeness, and PSF-model-error false-positive controls. The poster leads with PSF-bank marginalized mass completeness and the matched-versus-over-idealized-perfect-PSF false-positive stress test.

Final poster spine:

- [x] SCDD/HWO science motivation and science question.
- [x] Pipeline overview paragraph.
- [x] Example lensing scene, injected subhalo, and fractional residual.
- [x] Segmented-pupil / hexike PSF assumptions figure.
- [x] Fisher/Asimov, `q_fit`, and Bayesian-evidence detection convention.
- [x] Noisy PyAutoLens/Nautilus validation plot.
- [x] PSF-bank marginalized mass-completeness curve.
- [x] PSF-model-error false-positive control plot.
- [x] Conclusions and RASTI-scope next steps.

Deferred from poster to manuscript/RASTI: standalone pipeline schematic, PSF-amplitude degradation curve, full detectable-ring/2D sensitivity map, and continuous PSF-nuisance false-positive study.

SPIE deadline targets:

- [x] Poster PDF deadline: 10 June 2026.
- [x] Manuscript deadline: 17 June 2026.
- [ ] Conference week: 5-10 July 2026.

## Stage 4: RASTI-level codebase

Goal:

Make the repo journal-grade for a calibrated archival study.

Checklist:

- [ ] Promote reusable study tooling into a supported analysis layer, likely `src/hwoslaps/analysis/`.
- [ ] Add a supported sweep-manifest parser.
- [ ] Add a supported study aggregator.
- [ ] Add requirement-curve generation.
- [ ] Add publication figure-generation scripts.
- [ ] Add full 2D detectable-area support, not only local injected-position or Einstein-ring significance.
- [ ] Promote the SPIE metric convention into supported analysis outputs: `q_F`, `Z_F`, `Delta log L_F,equiv`, and SCDD-threshold pass/fail.
- [ ] Extend PyAutoLens nonlinear calibration across more masses, positions,
      PSF states, scenes, and false-positive cases.
- [ ] Fit or bound the calibration relation between `q_F` and `q_fit` with
      uncertainty across those broader conditions.
- [ ] Add clumpy/cuspy source-realism stress tests.
- [ ] Add continuous or broader PSF-nuisance marginalized false-positive tests
      beyond the archived discrete PSF-bank canonical-scene validation.
- [ ] Add lens-light and lens-subtraction residual stress tests.
- [ ] Add line-of-sight halo approximation or caveat handling.
- [ ] Add stronger provenance: code hash, config hash, dependency versions, manifest, and output validation checks.
- [ ] Add tests for manifest expansion, aggregation schema, deterministic reruns, and required artifact presence.

Minimum RASTI code outputs:

- supported analysis module,
- full study manifest,
- reproducible run command sequence,
- aggregated study table,
- requirement-curve outputs,
- validation outputs,
- publication-ready figure scripts.

## Stage 5: RASTI-level study

Goal:

Run the full science study and derive calibrated PSF-stability and mass-reach implications.

Required science expansions:

This is the minimum expansion from the submitted SPIE artifact toward a defensible
RASTI paper. The goal is to show that the PSF-stability conclusion is not an
artifact of one lensing scene, one monochromatic static PSF, one pixel scale, or
one targeted nonlinear fit.

**[L] Lensing scene upgrades:**

- L1: Expand from the canonical SPIE scene to many lens/source/subhalo
  configurations, including a mass and position ladder for subhalos.
- L2: Add source-morphology complexity beyond smooth elliptical profiles, including
  clumpy or irregular source structure.
- L3: Add lens-model realism and stress tests for lens light, subtraction residuals,
  external shear, and line-of-sight structure or an explicit caveat when those
  effects remain out of scope.

**[P] PSF model upgrades:**

- P1: Replace the static single-PSF assumption with time-varying or exposure-varying
  PSFs when the observation model includes multiple visits or coadds.
- P2: Use the EAC1 pupil PSD or covariance model if it becomes available; otherwise,
  use documented parametric priors on PSF mode coefficients.
- P3: Sweep a broad segmented and global mode basis, including segment piston,
  segment tip/tilt, segment hexikes, global Zernikes, and selected mixed modes.
- P4: Treat the distribution or weight function over modes as a tunable study knob,
  not only as independent one-mode perturbations.
- P5: Include a perfect-PSF reference arm and a matched-truth-PSF control arm in
  every ensemble grid manifest, alongside the marginalized-PSF cases.

**[O] Observation model upgrades:**

- O1: Replace monochromatic draws with realistic bandpasses or a documented
  approximation to bandpass-integrated images.
- O2: Add HWO observing assumptions for exposure time, visit cadence, roll angle or
  sky orientation diversity, and coaddition when those are needed for a claim.
- O3: Sweep pixel scale because the high-resolution imager sampling is not fixed.
- O4: Record detector noise, background, and throughput as explicit config fields
  so any requirement curve can be regenerated under changed architecture inputs.

**[D] Detection and inference upgrades:**

- D1: Move from targeted detection tests to a broader nonlinear search over subhalo
  mass, position, and nuisance parameters.
- D2: Include PSF uncertainty in the nonlinear inference through PSF posterior
  samples, coefficient priors, or controlled nuisance marginalization.
- D3: Validate Fisher predictions against nonlinear fits across more than one scene,
  PSF family, mass scale, and signal-to-noise regime.
- D4: Quantify false positives where PSF mismatch, source complexity, or lens-model
  mismatch can mimic substructure.
- D5: Report detectable-area maps and mass-floor curves rather than only
  single-location detection statistics.

**[S] Overall study and comparison upgrades:**

- S1: Convert the one-scene SPIE proof of concept into a reproducible ensemble study
  with manifests, provenance, aggregated tables, and regenerable figures.
- S3: Include JWST/Webb and alternate HWO architecture comparisons, such as EAC2 or
  later EAC concepts, where they are needed to support an HWO-specific claim.
- S4: Mark every physical and instrument parameter as frozen or swept in the
  config/manifest schema.

**[R] Requirements and science endpoints:**

- R1: Define the requirement-statement format before running sweeps: the number
  an engineer receives, for example tolerable unmodeled WFE per mode family in
  nm RMS at fixed completeness and false-positive rate.
- R2: Run a graded PSF-knowledge-error sweep: generate with the truth PSF, fit
  with a PSF wrong by a controlled amount `delta`, and sweep `delta` per mode
  family to locate where completeness degrades and false positives rise. This
  is the requirement-grade form of the D4 PSF-mismatch test.
- R3: Fold the 2D sensitivity maps with subhalo mass functions under competing
  dark-matter models (CDM plus WDM suppression) to forecast expected detections
  per lens and the number of lenses needed to discriminate models.

Tiered scope ranking:

This ranking replaces the earlier flat recommended-scope list. It ranks only
pipeline/study work by the depth it receives; process and writing conventions
are not ranked (former L4 and S2 were process notes and have been removed —
this ranking itself does S2's job). Non-binary items are split into lettered
subitems whose minimum and extended forms can land in different tiers. Every
item is broken into check-offable deliverables — codebase features or defined
sweeps/runs — checked off when the change lands or the run completes and is
aggregated. Tier definitions:

- **T0 enabling tooling:** built first so the sweeps can run; not a depth
  choice.
- **T1 core:** a headline claim dies without it; must be complete for the
  31 August submission no matter what.
- **T2 referee-proofing:** cheap relative to the predictable objection it
  defuses; do when T1 is on schedule.
- **T3 stretch:** strengthens the paper but nothing depends on it; only if
  ahead of schedule after T1 and T2.
- **T4 deferred:** out of scope; the deliverable is the named caveat in the
  limitations and future-work text.

Cost tags: `(A)` mainly agent time, `(C)` mainly compute (front-load while xtx
access lasts), `(G)` mainly George time (batch into review sessions).
`[ext]` marks items blocked on external input; chase these in week 1 or they
default to T4.

**T0 — enabling tooling (week 1):**

- S1 (A): supported analysis layer; every other item runs through it.
  - [ ] Build `src/hwoslaps/analysis/`: sweep-manifest parser, study
        aggregator, provenance capture, and figure-generation entry points.
  - [ ] Encode the schema requirements: detector noise, background, and
        throughput as explicit config fields (O4); every parameter marked
        frozen or swept (S4); perfect-PSF reference arm and matched-truth-PSF
        control arm mandatory in every grid manifest (P5).

**T1 — core (the paper on 31 August):**

Workstream 1 — mass reach:

- L1a (C): structured scene ensemble. The archived canonical-scene curve
  covers `10^6.5` to `10^7.5 Msun`; broaden it, do not repeat it.
  - [ ] Define 3-4 lens/source ensemble configurations beyond the canonical
        scene.
  - [ ] Run the ensemble Fisher sweep: scenes crossed with the mass ladder
        (`1e6` to `1e8 Msun`, `10^0.25` steps in the core range) and
        near-ring positions, aggregated into one results table.
- L2a (C): clumpy sources.
  - [ ] Implement one or two clumpy/irregular source models.
  - [ ] Run their injected-subhalo and no-subhalo ensemble grids.
- L3a (A): external shear.
  - [ ] Add external shear to the base lens model in generating and fitting
        configs.
- D5 (C): 2D sensitivity maps; the main GPU burn and the input to R3.
  - [ ] Implement 2D Fisher sensitivity maps over subhalo position.
  - [ ] Run the map grid across ensemble scenes, masses, and PSF states;
        produce thresholded detectable-area and mass-floor curves.
- D4a (C): no-subhalo false-positive controls along every T1 axis.
  - [ ] Run matched-PSF no-subhalo control grids across the ensemble.
  - [ ] Run clumpy-source no-subhalo data against smooth-source fit models.
- R3 (A/G): dark-matter fold.
  - [ ] Implement the fold of the D5 maps with CDM and WDM subhalo mass
        functions.
  - [ ] Produce expected-detections-per-lens and lenses-to-discriminate
        forecasts.

Workstream 2 — PSF requirements:

- R1 (G): requirement-statement format; gates which sweeps are needed.
  - [ ] Write the requirement-metric definition (tolerable unmodeled WFE per
        mode family at fixed completeness and false-positive budget) into
        this doc and the canonical configs after advisor input, week 1.
- P2 (A/G) `[ext]`: PSF coefficient priors; feed D2a and R2.
  - [ ] Request the EAC1 pupil PSD or covariance model from HWO/JPL contacts.
  - [ ] Freeze documented parametric priors on mode coefficients (the default
        if no PSD arrives by mid-July).
- P3 (C): mode-family sweep.
  - [ ] Add segment-piston and segment tip/tilt PSF families alongside the
        existing hexike, global-Zernike, and combined families.
  - [ ] Run the family-by-amplitude-ladder Fisher grid.
- R2 (C): graded PSF-knowledge-error sweep; anchors the requirement curves.
  - [ ] Implement delta-mismatch fitting: generate with the truth PSF, fit
        with a PSF wrong by a controlled `delta`.
  - [ ] Run the `delta` ladder per mode family; produce completeness and
        false-positive degradation curves.
- D2a (C): prior-sampled PSF nuisance bank.
  - [ ] Extend the PSF bank from 4 scaled candidates to roughly 16-32 draws
        from the P2 priors with log-sum-exp marginalization.
  - [ ] Rerun the bank-marginalized completeness and control grids with the
        new bank.

Workstream 3 — calibration:

- D1a (C): freed nonlinear searches.
  - [ ] Free the subhalo mass (`log M200`) and widen the position window in
        the nonlinear fit model.
  - [ ] Rerun the nonlinear validation grid with the freed model.
- D3 (C/G): Fisher-to-nonlinear calibration; expect `alpha` to shift for
  clumpy sources.
  - [ ] Select a stratified validation subset across scene, source type, PSF
        family, mass, and S/N.
  - [ ] Fit `q_fit ~ alpha q_F` per regime and report the calibration with
        uncertainty.

**T2 — referee-proofing:**

- O1 (A/C): bandpass.
  - [ ] Implement few-wavelength quadrature (3-5 wavelengths across one HRI
        band) in the PSF/observation model.
  - [ ] Run the monochromatic-versus-bandpass comparison and document the
        approximation error.
- O3 (C): pixel scale.
  - [ ] Rerun the core Fisher grids at 2-3 HRI-plausible pixel scales with a
        few nonlinear anchors.
- O2a (A): S/N scaling.
  - [ ] Derive the exposure-time/S-N scaling of the requirement curves and
        report all curves at a stated fixed S/N.
- L3b (C/G): lens-light stress test; degrades to a T4 caveat if the schedule
  collapses.
  - [ ] Add lens-galaxy light and an imperfect-subtraction model to the
        canonical scene.
  - [ ] Run one injected-subhalo plus no-subhalo stress grid against it.

**T3 — stretch:**

- P4 (A/C): mode weighting; collapses into P2 if a real PSD model arrives.
  - [ ] Run one or two alternative mode-weighting functions beyond the
        fiducial prior.
- S3b (C) `[ext]`: EAC contrast case.
  - [ ] Obtain an EAC2 or alternate-EAC pupil definition.
  - [ ] Run one Fisher-only contrast grid at a fixed scene.
- P1a (C): PSF drift demo, only if T1 and T2 lock early.
  - [ ] Run a single two-epoch PSF-drift demonstration.

**T4 — deferred; each item checks off when its caveat lands in the
limitations/future-work text:**

- [ ] L1b: population-level random lens ensembles (conclusions stay
      structured-ensemble level).
- [ ] L2b: systematic source-morphology survey beyond the clumpy variants.
- [ ] L3c: line-of-sight halos (lens-plane-only is conservative for expected
      counts).
- [ ] P1b: full time-varying or exposure-varying PSF treatment.
- [ ] O2b `[ext]`: visit cadence, roll diversity, and coaddition.
- [ ] D1b: fully blind whole-image subhalo searches (Fisher maps cover
      position dependence).
- [ ] D2b: continuous or hierarchical PSF-posterior inference in the
      nonlinear fits.
- [ ] D4b: broader lens-model-mismatch false-positive survey (cite He et al.
      2022).
- [ ] S3a: quantitative JWST comparison (cite existing literature).

RASTI study questions:

- [ ] What mass floor is reachable under ideal PSF assumptions?
- [ ] How does each PSF mode degrade that mass floor?
- [ ] Which PSF modes are most dangerous for subhalo detection?
- [ ] What PSF tolerance preserves the science case at a chosen degradation budget?
- [ ] How robust are the conclusions to source morphology?
- [ ] How robust are the conclusions to detection-threshold choice?
- [ ] How often can PSF mismatch mimic a subhalo detection?
- [ ] How does broader nonlinear validation modify the Fisher requirement curves?

RASTI study figures:

- [ ] Ideal-PSF sensitivity map or detectable-area map.
- [ ] Detectable area versus mass.
- [ ] Mass floor versus PSF mode amplitude.
- [ ] PSF-mode ranking / tolerance plot.
- [ ] Fisher-versus-fit validation plot.
- [ ] False-positive / PSF-mismatch diagnostic.
- [ ] Source-realism comparison.
- [ ] Requirement-curve summary figure.

## Stage 6: RASTI manuscript

Goal:

Submit the expanded archival paper to the RASTI HWO Special Issue.

Manuscript spine:

- [ ] HWO dark-matter subhalo science case.
- [ ] Detection metric and sensitivity-map formalism.
- [ ] HWO-SLAPS implementation.
- [ ] Fisher forecast validation.
- [ ] Ideal-PSF mass reach.
- [ ] PSF-mode degradation study.
- [ ] Requirement translation.
- [ ] Limitations and mission-design implications.

RASTI deadline target:

- [ ] Submission deadline: 31 August 2026.

## Timeline

Reference date: 17 May 2026.

- Now to internal review: lock one canonical config, run minimum mass/PSF sweeps, make poster-grade figures.
- Late May: finish SPIE aggregation and nominal study grid.
- Late May to early June: rerun final SPIE grid, generate figures, and write manuscript.
- 10 June 2026: SPIE poster PDF deadline.
- 17 June 2026: SPIE manuscript deadline.
- Late June to conference: poster polish and result cleanup.
- 5-10 July 2026: SPIE Astronomical Telescopes + Instrumentation.
- July: build RASTI-grade analysis and validation layer.
- Late July to mid-August: run full RASTI sweeps.
- Mid to late August: write RASTI manuscript and finalize figures.
- 31 August 2026: RASTI HWO Special Issue deadline.

## Immediate next decisions

1. Decide which SPIE archive components should be promoted into supported RASTI
   analysis tooling.
2. Decide the first RASTI scene/source expansion beyond the canonical SPIE
   setup.
3. Decide how to replace or extend the discrete PSF-bank validation with a
   broader PSF-nuisance treatment.
4. Decide which SPIE figures should seed RASTI figures and which should remain
   archived proceedings-only artifacts.
