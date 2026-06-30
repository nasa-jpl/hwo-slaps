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

The missing layer is not the core physics module layer. For SPIE, the first
study layer now exists; for RASTI, that layer still needs to be promoted from
scripts and packaged outputs into supported, journal-grade analysis tooling:

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

Checklist:

- [x] Create a canonical SCDD/SPIE baseline config, separate from `configs/master_config.yaml`.
- [x] Use lens redshift `z_l = 0.2` and source redshift `z_s = 0.6` for the SCDD-anchored run, unless a different source redshift is explicitly justified in the manuscript.
- [x] Set the canonical grid, pixel scale, wavelength, aperture, exposure/noise convention, subhalo model, concentration model, and source morphology in one named config.
- [x] Use the SCDD anchor masses for the headline SPIE grid: `1e7`, `10^7.25`, `10^7.5`, `10^7.75`, and `1e8 Msun`.
- [x] Treat masses below `1e7 Msun` as exploratory unless calibrated by nonlinear fits. The canonical-scene `10^6.5` and `10^6.75 Msun` points now have PSF-bank marginalized nonlinear validation and should be described as calibrated for that setup, not as final HWO requirements.
- [x] Define a perfect-PSF baseline.
- [x] Define one required PSF perturbation family for SPIE.
- [x] Define one optional second PSF perturbation family for SPIE.
- [x] State the amplitude units unambiguously for the Stage 0 PSF family: OPD nm, mirror-surface nm, outgoing-beam microradians, or coefficient RMS.
- [x] Add a lightweight study manifest format in `scratch/study` or `analysis_manifests/`.
- [x] Add a lightweight study runner that expands the manifest into per-run configs and run directories.
- [x] Add cross-run aggregation to `results.csv` or `results.jsonl`.
- [x] Record at least: run name, config hash, git hash if available, mass, subhalo model, subhalo position, PSF family, PSF mode, PSF amplitude, seed, local `q_F`, local `Z_F`, `Delta log L_F,equiv`, threshold pass/fail, map median `Z_F`, map max `Z_F`, detectable-ring fraction, profiling degradation, nuisance count, and PSF quality metrics.
- [x] Store SPIE-needed PSF diagnostics: Strehl, WFE/OPD RMS if available, kernel shape, kernel sum, kernel peak, and FWHM.
- [x] Add remaining diagnostic extras: raw peak ratio before clipping and a simple kernel-difference norm relative to the perfect PSF.
- [x] Add SPIE plotting scripts for required figures.
- [x] Add provenance beyond `config_used.yaml`: git hash, config hash, package versions, Python version, and command line.
- [x] Make the output path portable. Avoid absolute user-specific paths in canonical configs.

Minimum SPIE code outputs:

- canonical config,
- sweep manifest,
- generated per-run configs,
- per-run logs and config snapshots,
- one aggregate results table,
- one figure-generation command or script,
- one reproducibility summary file.

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
- Optional low masses: `10^6.5` and `10^6.75 Msun` now have canonical-scene PSF-bank marginalized nonlinear validation; `1e6 Msun` remains exploratory unless separately validated.
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

PyAutoLens nonlinear modeling is now a **core SPIE validation layer** rather
than a stretch goal. The controlled SPIE study has:

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
  supports them as screening metrics, and the discrete PSF-bank validation gives
  the current strongest canonical-scene nonlinear mass reach. Final requirement
  claims still require broader nonlinear validation, source-scene variation,
  full 2D maps, and expanded PSF-nuisance treatment.

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

Submit a real full-length SPIE proceedings manuscript and prepare the poster.

Manuscript spine:

- [ ] Motivation from the SCDD PSF-stability future-work case.
- [ ] HWO-SLAPS pipeline overview.
- [ ] Fisher/Asimov detection metric and SCDD threshold mapping.
- [ ] Canonical experiment setup.
- [ ] First mass-reach forecast.
- [ ] First PSF-mode degradation forecast.
- [ ] PyAutoLens nonlinear validation status.
- [ ] PSF-bank marginalized nonlinear mass-completeness result.
- [ ] Limitations and path to the RASTI study.

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
- [ ] Manuscript deadline: 17 June 2026.
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
      beyond the current discrete PSF-bank canonical-scene validation.
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

Recommended scope:

- Mass ladder: `1e6`, `10^6.5`, `10^6.75`, `1e7`, `10^7.25`, `10^7.5`, `10^7.75`, and `1e8 Msun`. The canonical-scene PSF-bank curve already covers `10^6.5` through `10^7.5`; RASTI should broaden that result rather than merely repeat it.
- Full 2D sensitivity maps over subhalo positions around the lensed arcs.
- Multiple PSF mode families:
  - segment piston,
  - segment tip/tilt,
  - segment hexikes,
  - global Zernikes,
  - selected combinations.
- Multiple amplitudes per PSF family.
- Perfect-PSF and perturbed-PSF comparisons.
- Expanded Fisher-versus-PyAutoLens validation suite.
- Expanded false-positive analysis with broader PSF nuisance and noisy ensembles.
- Detectable-area and mass-floor curves derived from thresholded 2D maps.
- Source-morphology stress tests.
- Lens-light and subtraction-residual stress tests.
- Time-varying PSF or drift tests, if motivated by HWO observing assumptions.

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

1. Decide which PyAutoLens validation figures are central versus backup for the SPIE poster.
2. Decide whether the PSF-degradation figure should use nominal amplitude, measured WFE, or both.
3. Decide how prominently to feature the wrong-PSF false-positive result versus the Fisher degradation curves.
4. Decide which limitations move into the poster itself and which are reserved for the proceedings manuscript.
