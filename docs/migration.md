# Migrating from the RASTI code

The code used for the RASTI paper is kept at the git tag `rasti-26-183-submitted`.
This version reorganizes the package around one configuration file, one forecast
interface and one set of result files. Parity tests check that forecasts of the paper's
test scenes match the RASTI code exactly; the interfaces differ. Old result files cannot be read by
this version, so keep the tagged code if you need to reproduce or reread old products.

## Configuration

| RASTI code | This version |
|---|---|
| `config.loading.load_config` returning a dictionary | `load_config` returning a validated `EngineConfig` |
| `global_seed` | `seed` for the scene; detector noise has its own seed |
| `lensing.lens_galaxy`, `lensing.source_galaxy` | Named components under `scene.lens` and `scene.source` |
| `telescope`, `imaging` | `psf.truth`, `instrument` and `observation` |
| `modeling.fit_psf` | `psf.model` |
| PSF priors given as repository paths | Packaged prior names, or a path relative to the configuration file |
| Fisher `mask_mode` and annulus settings | `forecast.mask` |
| A list of free nuisance parameters | `forecast.nuisances.fixed`, listing the parameters to hold fixed |
| Background and PSF nuisance switches | `forecast.nuisances.background_offset` and `forecast.nuisances.wavefront` |
| Engine, workers and batch size in the configuration | `Execution`, or command-line options |
| A detection threshold in the configuration | A `q_threshold` argument wherever detections are counted |

Paths inside a configuration are now relative to the file that contains them, and
misspelled keys are errors.

## Forecasts and analysis

| RASTI code | This version |
|---|---|
| The injected mass used as the forecast mass | `forecast(prepared, masses_msun=[...])` |
| `prepared.trial(...)`, `SubhaloTrial` | `prepared.hypothesis(mass, position)`, returning a `Halo` |
| `forecast(..., masses=..., domain_positions=...)` | `forecast(..., masses_msun=..., positions=...)`; the position domain is set at preparation |
| `summarize_forecast` with supplied areas | `summarize`, which takes areas from the grid |
| Array-based `mass_reach` | `mass_reach(summary, quantity=..., target=..., interpolation=...)` |
| A fixed two-feature selection helper | `RankingPolicy` and `rank_pool` with your own cuts and weights |
| Plotting functions that save files | Plotting functions that return Matplotlib axes |

Nuisance parameter names now follow the configuration, for example
`lens.mass.main.einstein_radius` or `source.light.disk.centre_x`.

## Simulations and fits

| RASTI code | This version |
|---|---|
| `simulate(..., trial=..., seed=..., sample_noise=...)` | `simulate(..., subhalo=..., noise_seed=...)`, with `None` for no subhalo or no noise |
| A fit's `dataset_kind` set separately from the data | Taken from the observation being fitted |
| `NonlinearSearchSettings`, `FreshProfileSettings` | `SamplerSettings`, `RefineSettings`, with the sampler seed as a separate argument |
| A single subhalo recovery | Separate sampler and refined estimates |
| A fit mask taken from the forecast | All pixels except a PSF border by default; `mask="forecast_mask_minus_psf_border"` to match the forecast |

The default nonlinear prior boxes and sampler settings are narrower than those used for
the paper's adopted runs. [Nonlinear fits](guide/nonlinear.md#the-settings-used-in-the-rasti-paper)
shows how to set the paper's values.

## Command line

The command line takes configuration files as positional arguments:

```bash
hwoslaps forecast scene.yaml instrument.yaml forecast.yaml --masses 1e7 1e8 -o out/run
hwoslaps simulate scene.yaml instrument.yaml --noise-seed 11 -o out/noisy
```

`--smooth` simulates without a subhalo, and `--expected` without noise. Campaigns that
used the RASTI orchestration scripts now use [batches](guide/batches.md).
