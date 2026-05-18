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

The missing layer is not the core physics module layer. The missing layer is the **study layer**:

- canonical SPIE/SCDD configs,
- sweep manifests,
- cross-run aggregation,
- publication figures,
- detectable-area or detectable-ring summaries,
- reproducibility metadata,
- sparse nonlinear validation for SPIE,
- broader nonlinear validation for RASTI.

## Stage 0: one-week internal-review priority

Goal:

Produce a credible internal-review poster package quickly without trying to complete the full paper.

Minimum internal-review deliverables:

- [ ] One canonical SCDD-like baseline scene.
- [ ] One perfect-PSF mass sweep.
- [ ] One PSF perturbation family with an amplitude ladder.
- [ ] One ring-map / detectable-ring-fraction demonstration.
- [ ] One clear table of metric definitions: `q_F`, `Z_F`, `Delta log L_F,equiv`, and threshold pass/fail.
- [ ] One limitations box that says the SPIE version is a first forecast, not final HWO engineering requirements.

Recommended poster logic:

1. SCDD asks for PSF stability because low-mass subhalo detectability is PSF limited.
2. HWO-SLAPS maps optics perturbations into a lensing detection statistic.
3. Perfect-PSF simulations define mass reach.
4. Perturbed-PSF simulations show how mass reach or detectable-ring fraction degrades.
5. The RASTI study will expand this into calibrated requirement curves.

## Stage 1: SPIE-level codebase hardening

Goal:

Make the current pipeline produce repeatable proceedings-level results.

Checklist:

- [ ] Create a canonical SCDD/SPIE baseline config, separate from `configs/master_config.yaml`.
- [ ] Use lens redshift `z_l = 0.2` and source redshift `z_s = 0.6` for the SCDD-anchored run, unless a different source redshift is explicitly justified in the manuscript.
- [ ] Set the canonical grid, pixel scale, wavelength, aperture, exposure/noise convention, subhalo model, concentration model, and source morphology in one named config.
- [ ] Use the SCDD anchor masses for the headline SPIE grid: `1e7`, `10^7.25`, `10^7.5`, `10^7.75`, and `1e8 Msun`.
- [ ] Treat masses below `1e7 Msun` as exploratory extrapolation unless calibrated by nonlinear fits.
- [ ] Define a perfect-PSF baseline.
- [ ] Define one required PSF perturbation family for SPIE and one optional second family.
- [ ] State the amplitude units unambiguously for every PSF family: OPD nm, mirror-surface nm, outgoing-beam microradians, or coefficient RMS.
- [ ] Add a lightweight study manifest format in `scratch/study` or `analysis_manifests/`.
- [ ] Add a lightweight study runner that expands the manifest into per-run configs and run directories.
- [ ] Add cross-run aggregation to `results.csv` or `results.jsonl`.
- [ ] Record at least: run name, config hash, git hash if available, mass, subhalo model, subhalo position, PSF family, PSF mode, PSF amplitude, seed, local `q_F`, local `Z_F`, `Delta log L_F,equiv`, threshold pass/fail, map median `Z_F`, map max `Z_F`, detectable-ring fraction, profiling degradation, nuisance count, and PSF quality metrics.
- [ ] Store PSF diagnostics: Strehl, raw peak ratio before clipping, WFE/OPD RMS if available, kernel shape, kernel sum, and a simple kernel-difference norm relative to the perfect PSF.
- [ ] Add SPIE plotting scripts for required figures.
- [ ] Add provenance beyond `config_used.yaml`: git hash, config hash, package versions, Python version, and command line.
- [ ] Make the output path portable. Avoid absolute user-specific paths in canonical configs.

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
- Optional exploratory masses: `1e6`, `10^6.5`, and `10^6.75 Msun`, clearly labeled as extrapolations.
- Perfect-PSF baseline.
- One or two PSF families:
  - segment piston or selected segment hexike modes,
  - low-order global Zernikes.
- One amplitude ladder per PSF family.
- Fisher/Asimov forecasts for the headline sweep.
- Sparse PyAutoLens-JAX nonlinear validation if time permits.
- One no-subhalo plus PSF-mismatch false-positive demonstration if time permits.

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

Do not call the Fisher value a nonlinear likelihood ratio. Call it a Fisher-equivalent or Asimov forecast until the PyAutoLens validation supports a calibration.

## SPIE study questions

- [ ] Does detectability improve monotonically with subhalo mass in the perfect-PSF baseline?
- [ ] Does the perfect-PSF mass floor qualitatively match the SCDD expectation near `10^7` to `10^7.25 Msun` for an HWO-like high-resolution setup?
- [ ] Which selected PSF modes couple most strongly to subhalo-like residuals?
- [ ] At what approximate PSF amplitude does the local detection statistic or detectable-ring fraction degrade materially?
- [ ] What fraction of the sampled Einstein ring remains above the SCDD threshold?
- [ ] Does `q_F` track `q_fit` for the sparse nonlinear validation subset, if that subset is completed?
- [ ] Can HWO-SLAPS produce requirement-style curves from optics perturbations to subhalo detectability?

## SPIE study figures

Required:

- [ ] Example lensing scene and subhalo residual.
- [ ] Example perfect and perturbed PSF diagnostic.
- [ ] Detection statistic versus subhalo mass.
- [ ] Detection degradation or detectable-ring fraction versus PSF amplitude.
- [ ] PSF-mode coupling or tolerance-style plot.
- [ ] Fisher ring-map or detectable-ring-fraction figure.

Recommended if time permits:

- [ ] Fisher-versus-nonlinear calibration plot showing `q_F` against `q_fit`, with the SCDD threshold `q=10`.
- [ ] No-subhalo plus PSF-mismatch false-positive diagnostic.

## SPIE claim boundary

Use:

> We present HWO-SLAPS, an end-to-end framework that converts HWO-like PSF perturbations into preliminary low-mass subhalo detectability forecasts. We show first mass-reach and PSF-mode degradation curves, with sparse nonlinear validation planned or demonstrated for selected cases.

Avoid:

> We derive final HWO engineering requirements.

Avoid unless the validation is actually complete:

> The Fisher statistic is fully calibrated to nonlinear PyAutoLens evidence or likelihood-ratio recovery.

## PyAutoLens validation scope

### SPIE recommendation

PyAutoLens nonlinear modeling is in scope for SPIE only as a **small validation subset**. It should not block the poster or proceedings if the Fisher study is already producing coherent results.

Suggested SPIE subset:

- perfect PSF at `10^7.25` and `10^7.75 Msun`, one position;
- one perturbed-PSF case at the same position;
- optional no-subhalo PSF-mismatch false-positive case.

Minimum validation output:

- table of `q_F`, `q_fit`, and pass/fail relative to `q=10`,
- qualitative statement that the fast metric is a screening/forecast metric,
- no claim of full calibration.

### RASTI recommendation

PyAutoLens nonlinear validation is mandatory for RASTI. It should span masses, positions, PSF states, and no-subhalo false-positive cases, and should fit or bound a calibration relation such as

$$
q_{\mathrm{fit}} \simeq \alpha q_F
$$

or a monotonic mapping with uncertainty.

## Known SPIE limitations to state explicitly

- The SPIE source model may be smoother than the clumpy Ly-alpha emitter source assumed in the SCDD sensitivity discussion.
- Ring maps are a lightweight sensitivity-map surrogate, not a replacement for full 2D detectable-area maps.
- Lens-plane subhalos are modeled first; line-of-sight halos are deferred or treated as a caveat.
- Lens galaxy light and lens-light subtraction residuals are not yet part of the controlled SPIE forecast unless specifically added.
- Fisher/Asimov statistics are local forecasts and require nonlinear validation before final requirement claims.

## Acceptance checks before using SPIE plots

- [ ] Perfect-PSF `q_F` increases with subhalo mass for the same position and setup.
- [ ] The chosen detection threshold is shown on every relevant plot.
- [ ] `Delta log L_F,equiv = q_F / 2` is computed consistently.
- [ ] A zero-aberration PSF run is reproducible.
- [ ] PSF amplitude units are stated in the figure captions.
- [ ] The code records enough provenance to rerun every plotted point.
- [ ] No-subhalo PSF-mismatch case does not produce unexplained threshold-level detections, or the issue is highlighted as a result rather than hidden.

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
- [ ] Sparse validation status.
- [ ] Limitations and path to the RASTI study.

Poster spine:

- [ ] One-sentence SCDD motivation.
- [ ] Pipeline schematic.
- [ ] Example lens + PSF + subhalo residual.
- [ ] Detection significance versus mass.
- [ ] PSF amplitude versus degradation or detectable-ring fraction.
- [ ] Next-step validation box.

SPIE deadline targets:

- [ ] Poster PDF deadline: 10 June 2026.
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
- [ ] Add mandatory PyAutoLens-JAX nonlinear-fit calibration across masses, positions, PSF states, and false-positive cases.
- [ ] Fit or bound the calibration relation between `q_F` and `q_fit`.
- [ ] Add clumpy/cuspy source-realism stress tests.
- [ ] Add no-subhalo plus PSF-mismatch false-positive tests.
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

- Mass ladder: `1e6`, `10^6.5`, `10^6.75`, `1e7`, `10^7.25`, `10^7.5`, `10^7.75`, and `1e8 Msun`.
- Full 2D sensitivity maps over subhalo positions around the lensed arcs.
- Multiple PSF mode families:
  - segment piston,
  - segment tip/tilt,
  - segment hexikes,
  - global Zernikes,
  - selected combinations.
- Multiple amplitudes per PSF family.
- Perfect-PSF and perturbed-PSF comparisons.
- Fisher-versus-fit validation subset.
- False-positive analysis.
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
- [ ] How does full nonlinear validation modify the Fisher requirement curves?

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

1. Pick the single PSF family for the first internal-review sweep.
2. Lock the canonical SCDD baseline config.
3. Decide whether SPIE PyAutoLens validation is minimum, nominal, or stretch.
4. Decide the degradation budget to show on the first tolerance-style plot, for example fractional retained Fisher information, detectable-ring fraction, or mass-floor shift.
