# Migrating from the RASTI code

The submitted implementation is available at the git tag `rasti-26-183-submitted`
(commit `41621de`). Its [code and provenance record](https://github.com/nasa-jpl/hwo-slaps/tree/rasti-26-183-submitted/reproducibility/rasti-26-183)
remains in Git history. This package has a new configuration schema and result format;
use the tagged implementation to read old products or run the original campaign scripts.

## Entry points

| Submitted RASTI code | This version |
|---|---|
| `python runner.py --config FILE`, calling `hwoslaps.pipeline.run_enhanced_pipeline` | `hwoslaps forecast CONFIG... --masses M... -o DIR` or `hwoslaps simulate CONFIG...` |
| `Pipeline.run(config)` selecting simulation or detection from `modeling.enabled` | Explicit `prepare_forecast`, `forecast`, and `simulate` calls |
| `hwoslaps.modeling.fisher_detector.FisherDetector` and its grid-map methods | `PreparedForecast` and `forecast`, returning a `ForecastResult` |
| `scripts/run_ladder.py` and the campaign generation, launch, and harvest scripts | Configurable population and batch jobs, followed by explicit analysis |
| Pipeline-driven plots | Plotting functions that return Matplotlib axes; callers choose what to plot and save |

The old `campaign`, `lensing`, `modeling`, `psf`, and `pipeline` modules are removed.
Their reusable calculations now live in `scene`, `optics`, `fisher`, `inference`,
`analysis`, and `batch`. The old study scripts and frozen campaign files are not
installed alongside the new engine.

## Configuration

| Submitted RASTI configuration | This version |
|---|---|
| `global_seed` | `seed` for scene choices; detector noise and samplers have separate seeds |
| `lensing.grid`, `lensing.lens_galaxy`, `lensing.source_galaxy` | `scene.grid`, named `scene.lens` components, and named `scene.source.light` components |
| `lensing.subhalo` combines model and injection choices | `scene.subhalo` describes the hypothesis; optional `scene.injection` specifies an injected mass and position |
| `psf.telescope`, high-resolution/kernel settings, and wavefront coefficients | `psf.truth`, with a pupil and optical settings or an external kernel |
| `modeling.fit_psf` | `psf.model` |
| `observation.detector` and `observation.exposure_time` | `instrument.detector` and `observation.exposure_time_s` |
| `modeling.fisher` mask, parameter, and position choices | `forecast.mask`, `forecast.nuisances`, and `forecast.positions` |
| Detection thresholds in study configuration | Explicit `q_threshold` arguments in analysis |

Paths are relative to the file that declares them. Unknown keys are errors. The new
schema does not translate old YAML automatically; start with a current example and
transfer the physical choices explicitly.

The old observation layer applied `observation.throughput` to rendered source samples.
Current literal profile rates are already detected rates. Alternatively, use AB
photometry with a collecting area, a bandpass including detector quantum efficiency,
and a source SED. Copying an old intensity without its throughput and pixel-sampling
convention can change the detected brightness. Sky and dark-current rates are already
detected and must not be multiplied by throughput again.

## Forecasts and analysis

`prepare_forecast` builds reusable scene, observation, PSF and nuisance state.
`forecast(prepared, masses_msun=[...])` evaluates a chosen mass list. `summarize`
reduces those maps at your threshold, and `mass_reach` finds crossings in the supplied
mass sequence. These replace the detector and ladder interfaces, rather than aliasing
them.

The paper used refined rungs for its best-position mass reach, but coarse-lattice
rungs for its 10% and 50% area estimands. Select those rung sets explicitly when
reproducing the paper: `mass_reach` does not infer a ladder's coarse/refined history.
Nonmonotone curves return `non_monotonic`; the paper's ladder instead used its first
upward crossing. Bounds and sampled crossings also have explicit status values.

For the submitted Table 5, the retained-sensitive-area statistic is
`KnowledgeErrorAreas.retention` (K), the intersection with correct-PSF detections.
`detected_area_ratio` is the different total-area ratio used by the pre-submission
harvester. See [PSF knowledge error](guide/psfs.md#psf-knowledge-error).

## Simulations and fits

Use `simulate(prepared, subhalo=..., noise_seed=...)`; `None` omits the halo or the
noise draw. The returned observation records whether it is expected or noisy.
`validate_nonlinear` takes explicit fit, sampler, and optional refinement settings.
Failed, unresolved, and accepted nondetection outcomes remain distinct.

An attached forecast reference now uses `comparison_policy="strict"` by default.
The fit rejects differing or unassessed scientific inputs before starting a search.
For an intentional comparison across PSF arms, noise choices, or nuisance constraints,
pass `comparison_policy="diagnostic"` and inspect `result.data["forecast_comparison"]`.
`detection_agreement` also defaults to strict inclusion and reports each excluded
comparison's differences. Its diagnostic mode must be requested separately.
The truth-preserving `comparison_digest` keeps its existing meaning.

Sampler records now capture requested and achieved stopping criteria, exploration
completion, and the stop reason. Role `quality` records distinguish sampling status,
posterior quality, and evidence quality from likelihood optimization acceptance.
The backend's historical `pdf_converged` field is only a sample-weight usability
heuristic; recovery also exposes it as `pdf_usable`. Incomplete runs retain finite
evidence and usable quantiles as `provisional`. `scientific_outputs` excludes these
values by default; `include_provisional=True` makes their inclusion explicit.
Meeting sampler criteria does not establish that every posterior mode was found.

Every supported scene parameter is free by default unless listed in
`forecast.nuisances.fixed`. This includes an Image component's `rotation_deg`, which
the paper held fixed. For a component named `galaxy`, include this entry alongside
any other fixed parameters:

```yaml
forecast:
  nuisances:
    fixed: [source.light.galaxy.rotation_deg]
```

The nonlinear fit uses the forecast's free scene parameters. Its default prior boxes
and sampler settings are not the adopted paper protocol. The [paper recipe](guide/nonlinear.md#the-settings-used-in-the-rasti-paper)
gives `slam1`, its retry settings, batch size and fixed-rotation configuration. Both
forecast-mask and explicit-mask fits remove the PSF border, so their fitted pixels
may still differ from the forecast mask.

Random placement, direction and sampler streams now use named identifiers. A seed
value alone does not reproduce the old campaign's realization. Random injection
scatter is specified in arcseconds rather than pixels, and polar placement is about
the configured lens centre. Preserve explicit positions and archived seeds when an
exact historical realization is required. Injected and control nonlinear families
receive distinct noise seeds; sharing a Poisson seed would not establish paired noise.

## Command line

Configuration files are positional and combine in order:

```bash
hwoslaps forecast scene.yaml instrument.yaml forecast.yaml --masses 1e7 1e8 -o out/run
hwoslaps simulate scene.yaml instrument.yaml --noise-seed 11 -o out/noisy
```

`--smooth` simulates without a subhalo; `--expected` omits the noise draw.
[Populations and batches](guide/batches.md) describes the replacement campaign interface.
The archived parity fixtures remain unchanged evidence from the tagged implementation.
Current-engine tests retain exact checks on unchanged inputs and use independent
residual-space and likelihood oracles for corrected numerical behavior. They report
old-to-new changes rather than claiming paper reproduction. Forecast schema 3 and case
schema 2 retain numerical validity, precision bounds, and sampling-quality records;
loaders accept their current schemas only.
