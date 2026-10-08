# API overview

The most common functions and classes can be imported from the top-level package:

```python
from hwoslaps import load_config, prepare_forecast, forecast, summarize, mass_reach
```

This page groups the public API by task. Each name links to its full reference,
generated from the source code. [All modules](api/index.rst) lists every public name.

## Configuration

| Name | Purpose |
|---|---|
| {py:func}`~hwoslaps.config.schema.load_config` | Read and combine YAML files into a configuration |
| {py:func}`~hwoslaps.config.schema.parse_config` | Build a configuration from a Python mapping |
| {py:class}`~hwoslaps.config.schema.EngineConfig` | A validated, immutable configuration; `replace`, `digest`, `to_mapping` |
| {py:class}`~hwoslaps.config.checks.ConfigError` | Raised for an invalid configuration, with the path of the bad key |

## Forecasts

| Name | Purpose |
|---|---|
| {py:func}`~hwoslaps.fisher.api.prepare_forecast` | Build the scene, PSFs, observation and nuisances once |
| {py:func}`~hwoslaps.fisher.api.forecast` | Evaluate subhalo masses and positions |
| {py:class}`~hwoslaps.fisher.api.Execution` | Choose the engine, CPU workers and GPU batch size |
| {py:class}`~hwoslaps.fisher.api.PreparedForecast` | A prepared forecast; also supplies `hypothesis(mass, position)` |
| {py:class}`~hwoslaps.fisher.result.ForecastResult` | Statistics by mass and position, with provenance |

## Summaries

| Name | Purpose |
|---|---|
| {py:func}`~hwoslaps.analysis.reductions.summarize` | Largest *q*, detected fractions and areas per mass |
| {py:func}`~hwoslaps.analysis.reductions.aperture_selection` | Select the positions inside a circle |
| {py:func}`~hwoslaps.analysis.reach.mass_reach` | The mass at which a summary quantity crosses a target |
| {py:func}`~hwoslaps.analysis.reach.adaptive_mass_reach` | Find the crossing by bisection |
| {py:func}`~hwoslaps.analysis.knowledge_error.knowledge_error_areas` | Compare a matched and a mismatched forecast |
| {py:func}`~hwoslaps.analysis.knowledge_error.knowledge_error_tolerance` | The largest PSF error amplitude passing two gates |
| {py:func}`~hwoslaps.analysis.binomial.clopper_pearson` | Exact binomial confidence intervals |
| {py:func}`~hwoslaps.analysis.selection.rank_pool`, {py:class}`~hwoslaps.analysis.selection.RankingPolicy` | Rank a pool of lenses by image features, with your own cuts and weights |

## Observations

| Name | Purpose |
|---|---|
| {py:func}`~hwoslaps.simulation.simulate` | Expected or noisy images, with or without a subhalo |
| {py:class}`~hwoslaps.observation.observation.Observation` | A detector image with its noise map, light and records |
| {py:class}`~hwoslaps.scene.halos.Halo` | A subhalo of a given type, mass and position |

## Nonlinear fits

| Name | Purpose |
|---|---|
| {py:func}`~hwoslaps.inference.api.validate_nonlinear` | Fit smooth and subhalo models to an observation |
| {py:func}`~hwoslaps.inference.api.prepare_case` | Build the fit data, models and likelihood without running a search |
| {py:class}`~hwoslaps.inference.settings.FitSpec` | Fit mode, mask, prior boxes and mass support |
| {py:class}`~hwoslaps.inference.settings.SamplerSettings` | Nautilus settings and the JAX likelihood |
| {py:class}`~hwoslaps.inference.settings.RefineSettings` | Gradient refinement and its acceptance checks |
| {py:class}`~hwoslaps.inference.settings.PriorWidths`, {py:class}`~hwoslaps.inference.settings.BoxRule` | Prior box widths by parameter kind |
| {py:class}`~hwoslaps.inference.settings.MassSupport`, {py:class}`~hwoslaps.inference.settings.PixelMask` | The freed-mass range, and a custom fitted-pixel mask |
| {py:class}`~hwoslaps.inference.result.ForecastReference` | The forecast value at a fitted point, for comparisons |
| {py:class}`~hwoslaps.inference.result.CaseResult` | The two role fits, `q_signed` and recovery |
| {py:class}`~hwoslaps.analysis.nonlinear.ClassificationRule` | A detection rule: threshold, marginal band and accepted statuses |
| {py:func}`~hwoslaps.analysis.nonlinear.classify_case` | Apply a detection rule to a case |
| {py:func}`~hwoslaps.analysis.nonlinear.detection_agreement` | Compare classified cases with their forecasts |

## Batches and populations

| Name | Purpose |
|---|---|
| {py:func}`~hwoslaps.batch.spec.load_batch_spec` | Read a batch file |
| {py:func}`~hwoslaps.batch.jobs.plan_batch` | List the members and jobs of a batch |
| {py:func}`~hwoslaps.batch.runner.run_batch` | Run or resume a batch |
| {py:func}`~hwoslaps.batch.results.open_batch` | Read a batch's records and load its products |
| {py:func}`~hwoslaps.population.sampling.sample_population` | Draw population members without running a batch |

## Saving and loading

| Name | Purpose |
|---|---|
| {py:func}`~hwoslaps.artifacts.save_forecast`, {py:func}`~hwoslaps.artifacts.load_forecast` | Forecast files (`.npz`) |
| {py:func}`~hwoslaps.artifacts.save_observation`, {py:func}`~hwoslaps.artifacts.load_observation` | Observation files (`.npz`) |
| {py:func}`~hwoslaps.artifacts.save_case`, {py:func}`~hwoslaps.artifacts.load_case` | Nonlinear case files (JSON) |

## Plotting

| Name | Purpose |
|---|---|
| {py:func}`~hwoslaps.plotting.forecast.plot_statistic_map` | A statistic over positions at one mass |
| {py:func}`~hwoslaps.plotting.forecast.plot_detection_map` | Detections over positions at one mass |
| {py:func}`~hwoslaps.plotting.forecast.plot_mass_curve` | A summary quantity against mass |
| {py:func}`~hwoslaps.plotting.forecast.plot_knowledge_error` | Knowledge-error area ratios against mass |
| {py:func}`~hwoslaps.plotting.observation.plot_observation` | An observation image |
| {py:func}`~hwoslaps.plotting.optics.plot_kernel`, {py:func}`~hwoslaps.plotting.optics.plot_pupil` | A PSF kernel or telescope pupil |

## Lower-level building blocks

The `scene`, `optics`, `spectra` and `observation` packages expose the pieces that
`prepare_forecast` assembles: pupils, wavefronts, PSF providers, bandpasses, spectra,
lens and source profiles and halo models. They are useful for inspecting a
configuration or building a new workflow. See [All modules](api/index.rst).

```{toctree}
:hidden:

api/index
```
