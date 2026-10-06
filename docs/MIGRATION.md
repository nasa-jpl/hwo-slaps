# Migration to the current engine

> The typed owners, public exports, CLI and artifact transport have passed scoped validation. Batch execution and final combined validation remain open. Old study files and archives have no compatibility loader.

## Configuration and ownership

Construct `EngineConfig` through `config.schema.load_config` or `parse_config`. The previous raw-mapping `config.loading.load_config` wrapper has been removed. Replace inputs through `config.replace(overrides)`; direct construction or `dataclasses.replace` bypasses the supported factory.

The final configuration reference is generated from owning tables, not maintained as another manual schema. Supply detector, observation, scene and PSFs through their current sections. Execution settings and analysis thresholds belong to operation arguments. Asset paths resolve against their declaring files. A changed discriminator replaces the old alternative instead of leaving keys from both kinds.

| Previous input or call | Current route |
|---|---|
| `global_seed` | `seed` for named scene streams; detector noise has a separate seed |
| `lensing.lens_galaxy`, `source_galaxy` | Named mass/light components in `scene.lens` and `scene.source` |
| `telescope`, `imaging` | `psf.truth`, `instrument` and `observation` |
| `modeling.fit_psf` | `psf.model`, including matched/kernel/optical/knowledge-error choices |
| Repository-root PSF prior path | Packaged prior name or a path supplied with the configuration |
| Fisher `mask_mode`, annulus settings | `forecast.mask` |
| Fixed nuisance subset | `forecast.nuisances.fixed` holds the complement by names/patterns |
| Background/PSF nuisance switches | Current nuisance background/wavefront tables |
| Fisher covariance file | `forecast.noise_covariance` |
| Engine/worker/batch size in map science keys | `Execution` or operation/batch execution arguments |
| Configured detection threshold | Required caller analysis threshold |
| Injection used as implicit forecast mass | Required `forecast(..., masses_msun=...)` |
| `prepared.trial(...)`, `SubhaloTrial` | `prepared.hypothesis(...)` returning `Halo` |
| `forecast(..., masses=..., domain_positions=...)` | `masses_msun` and optional positions; domain fixed at preparation |
| `summarize_forecast` with supplied areas/boundaries | `analysis.reductions.summarize` using PositionSet geometry |
| Array-only `mass_reach` | `crossing` or `mass_reach(summary, quantity=..., target=..., interpolation=...)` |
| `simulate(..., trial=..., seed=..., sample_noise=...)` | `simulate(..., subhalo=..., noise_seed=...)`; None means expectation |
| Fit `dataset_kind` separate from observation | Actual `Observation.kind` controls expected/noisy likelihood |
| `NonlinearSearchSettings`, `FreshProfileSettings` | `SamplerSettings`, `RefineSettings`; sampler seed separate |
| One undifferentiated subhalo recovery | Sampler and refined estimates recorded separately |
| Selection's fixed two-feature helper | `RankingPolicy` with caller cuts, terms/logs/weights and top-k |
| Automatic plot registry and saving wrappers | Current product plot functions returning Axes; caller saves |

Current nuisance names identify plane, component role/name and parameter, for example `lens.mass.<name>.<parameter>` and `source.light.<name>.<parameter>`. Wavefront Zernikes use `psf.zernikes[n]`. Image rotation is a normal nuisance unless fixed. Schema defaults and allowed alternatives come from the generated reference after final integration.

## Fits, masks and comparisons

`validate_nonlinear` receives the observation it actually fits, a `FitSpec`, sampler settings/seed and optional refinement. The default mask is all pixels minus the PSF border, independent of a forecast mask. Python custom masks are recorded self-contained for current case persistence; their record codec does not extend the YAML configuration grammar.

The fitted kernel is copied into the backend dataset so its normalization cannot mutate the prepared kernel. Named PSF support is a property of the effective configuration, and a smaller fit-support arm is a scientifically different comparison. Mask differences, comparison digests and unfitted forecast nuisances are reported rather than silently equated.

A null/no-subhalo control now fits the actual expected/noisy observation supplied. It remains a control and is excluded from injected-detection agreement. A trial must match the configured halo recipe; a non-null observation must inject that trial. Simulation's ability to create a deliberate alternate truth-model study is retained.

Signed q, positive-amplitude forecast detection, threshold/marginal choices and accepted-role policy are separate. `stationarity_tolerance=None` preserves repeatable-profile acceptance; a positive value additionally checks the projected gradient. Failed or unresolved cases do not become measured nondetections. Gradient-only exact-centre/circular-shape refusals and sampling diagnostics are described in [SCIENCE](SCIENCE.md).

## Files, commands and reproducibility

The CLI uses positional configuration files, `--engine`, required masses/output directory, and `--noise-seed N` or `--expected`. `--smooth` requests the no-subhalo control. Final batch validation remains open. `python -m hwoslaps` is the module entry point; old runner/config selectors do not define the new workflow.

Current forecast/observation/case artifacts keep their current schemas, typed values and scientific identities. Historical NPZ, case paths, draw IDs and configuration digests are not interchangeable with current records. Retain the old study in git when reproduction needs the submitted implementation. New calculations must record current configuration/file/kernel/mask/noise identities and execution provenance.

Public import boundaries, wheel entry points, isolated installation and generated CONFIG.md equality have passed their scoped checks. Study-specific cohorts, thresholds, time limits and telescope assumptions remain caller/example inputs.
