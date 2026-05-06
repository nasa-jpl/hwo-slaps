# Study roadmap

## Scientific spine

Core question:

> Given an HWO-like strong-lensing scene, how do PSF error mode and amplitude degrade low-mass dark-matter subhalo detectability?

The project should stay one coherent study with two maturity levels:

- SPIE: controlled first forecast and proceedings-quality demonstration.
- RASTI: calibrated, broader archival science study.

## Current state

The four core HWO-SLAPS modules are already usable for a bounded SPIE study:

- Lensing: subhalo injection with PointMass, SIS, and NFW support.
- PSF: segmented HWO-style PSFs with segment-level and global aberration modes.
- Observation: PSF convolution and detector-noise simulation.
- Modeling: Fisher/Asimov subhalo detectability with local and map modes.

The missing layer is study infrastructure:

- Canonical SPIE/SCDD configs.
- Sweep manifests.
- Cross-run aggregation.
- Publication plots.
- Detectable-area summaries.
- Reproducibility metadata.
- RASTI-grade validation and analysis.

## Stage 1: SPIE-level codebase

Goal:

Make the current pipeline produce repeatable proceedings-level results.

Checklist:

- [ ] Create a canonical SCDD/SPIE baseline config.
- [ ] Use lens redshift `z_l = 0.2`.
- [ ] Choose source redshift deliberately: either SCDD-like `z_s = 0.6` or an explicitly justified alternative.
- [ ] Fix one canonical lens/source/subhalo geometry.
- [ ] Define a perfect-PSF baseline.
- [ ] Define one or two PSF perturbation families for SPIE.
- [ ] Add a lightweight study manifest format in `scratch/study`.
- [ ] Add a lightweight study runner that expands the manifest into configs and run directories.
- [ ] Add cross-run aggregation to `results.csv` or `results.jsonl`.
- [ ] Record run name, mass, PSF mode, amplitude, seed, local `Z`, map median `Z`, map max `Z`, degradation, and detectable fraction/area where available.
- [ ] Add SPIE plotting scripts for the required figures.
- [ ] Add basic provenance beyond `config_used.yaml`, ideally git commit/hash and environment summary.

Minimum SPIE code outputs:

- A canonical config.
- A sweep manifest.
- Generated per-run configs.
- Per-run logs and config snapshots.
- One aggregate results table.
- One figure-generation command or script.

## Stage 2: SPIE-level study

Goal:

Run a bounded study that supports the submitted SPIE abstract without overclaiming final HWO requirements.

Recommended scope:

- One canonical strong-lens scene.
- Masses: `10^7`, `10^7.5`, and `10^8 Msun`.
- Perfect-PSF baseline.
- One or two PSF families:
  - segment piston or segment hexike,
  - low-order global Zernikes.
- One amplitude ladder per PSF family.
- Fisher-only forecasts, framed as first quantitative forecasts.
- Optional small seed set if runtime allows.

SPIE study questions:

- [ ] Does detectability improve with subhalo mass as expected?
- [ ] Which selected PSF modes couple most strongly to subhalo-like signals?
- [ ] At what approximate PSF amplitude does detection degrade materially?
- [ ] Can HWO-SLAPS produce requirement-style curves from optics perturbations to subhalo detectability?

SPIE study figures:

- [ ] Example lensing scene and subhalo perturbation.
- [ ] Example PSF / aberration diagnostic.
- [ ] Detection significance versus subhalo mass.
- [ ] Detection significance or degradation versus PSF amplitude.
- [ ] PSF-mode coupling or tolerance plot.
- [ ] Optional sensitivity-map or ring-map figure.

SPIE claim boundary:

Use:

> We present a framework and first quantitative forecasts.

Avoid:

> We derive final HWO engineering requirements.

## Stage 3: SPIE manuscript and poster

Goal:

Submit a real full-length SPIE proceedings manuscript and prepare the poster.

Manuscript spine:

- [ ] Motivation from the SCDD PSF-stability future-work case.
- [ ] HWO-SLAPS pipeline overview.
- [ ] Fisher/Asimov detection metric.
- [ ] Canonical experiment setup.
- [ ] First mass-reach and PSF-mode forecasts.
- [ ] Limitations and path to the RASTI study.

Poster spine:

- [ ] Pipeline schematic.
- [ ] Example lens + PSF + subhalo residual.
- [ ] Detection significance versus mass.
- [ ] PSF amplitude versus degradation or tolerance.

SPIE deadline targets:

- [ ] Poster PDF deadline: 10 June 2026.
- [ ] Manuscript deadline: 17 June 2026.
- [ ] Poster presentation: 8 July 2026.

## Stage 4: RASTI-level codebase

Goal:

Make the repo journal-grade for a calibrated archival study.

Checklist:

- [ ] Promote reusable study tooling from `scratch/study` into a real analysis layer, likely `src/hwoslaps/analysis/`.
- [ ] Add a supported sweep-manifest parser.
- [ ] Add a supported study aggregator.
- [ ] Add requirement-curve generation.
- [ ] Add publication figure-generation scripts.
- [ ] Add full detectable-area support, not only local injected-position significance.
- [ ] Map Fisher `Z` / `Delta chi^2` to the SCDD-style `Delta log L > 5` criterion.
- [ ] Add sparse nonlinear fit validation cases if feasible.
- [ ] Add source-realism stress tests, especially clumpy or cuspy source morphology.
- [ ] Add false-positive tests: no subhalo plus PSF mismatch.
- [ ] Add stronger provenance: code hash, config hash, dependency versions, manifest, and output validation checks.
- [ ] Add tests for manifest expansion, aggregation schema, deterministic reruns, and required artifact presence.

Minimum RASTI code outputs:

- Supported analysis module.
- Full study manifest.
- Reproducible run command sequence.
- Aggregated study table.
- Requirement-curve outputs.
- Validation outputs.
- Publication-ready figure scripts.

## Stage 5: RASTI-level study

Goal:

Run the full science study and derive calibrated PSF-stability and mass-reach implications.

Recommended scope:

- Mass ladder: `10^7`, `10^7.25`, `10^7.5`, `10^7.75`, and `10^8 Msun`.
- Multiple subhalo positions or full sensitivity maps.
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
- Detectable-area and mass-floor curves.
- Source-morphology stress test.

RASTI study questions:

- [ ] What mass floor is reachable under ideal PSF assumptions?
- [ ] How does each PSF mode degrade that mass floor?
- [ ] Which PSF modes are most dangerous for subhalo detection?
- [ ] What PSF tolerance preserves the science case?
- [ ] How robust are the conclusions to source morphology?
- [ ] How robust are the conclusions to the detection-threshold choice?
- [ ] How often can PSF mismatch mimic a subhalo detection?

RASTI study figures:

- [ ] Ideal-PSF sensitivity map or detectable-area map.
- [ ] Detectable area versus mass.
- [ ] Mass floor versus PSF mode amplitude.
- [ ] PSF-mode ranking / tolerance plot.
- [ ] Fisher-versus-fit validation plot.
- [ ] False-positive / PSF-mismatch diagnostic.
- [ ] Source-realism comparison.

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

Reference date: 6 May 2026.

- Now to mid-May: lock SPIE config, runner, aggregation, and first plots.
- Mid to late May: run SPIE study grid and inspect failure modes.
- Late May to early June: rerun final SPIE grid, generate figures, and write manuscript.
- 10 June 2026: SPIE poster PDF deadline.
- 17 June 2026: SPIE manuscript deadline.
- Late June to 8 July: poster polish and result cleanup.
- 8 July 2026: SPIE poster presentation.
- July: build RASTI-grade analysis and validation layer.
- Late July to mid-August: run full RASTI sweeps.
- Mid to late August: write RASTI manuscript and finalize figures.
- 31 August 2026: RASTI HWO Special Issue deadline.

## Immediate next decisions

- [ ] Decide the SPIE canonical source redshift.
- [ ] Decide the SPIE source morphology.
- [ ] Decide the two SPIE PSF families.
- [ ] Decide the SPIE amplitude ladders.
- [ ] Decide whether SPIE includes a small seed ensemble or single deterministic scene.
- [ ] Decide whether the SPIE result includes a map/detectable-area figure or only local/ring detectability.
