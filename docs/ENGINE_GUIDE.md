# Engine guide

> Draft: the typed Python owners below are source-checked. Final exports, CLI/batch integration, example runs and generated configuration-reference validation remain pending.

## Configuration and preparation

`EngineConfig` holds a scene, cosmology, truth/model PSFs, instrument, exposure and optional forecast setup. Create it through `config.schema.load_config`, `parse_config` or `resolve_config`. A direct constructor and `dataclasses.replace` are unsupported; use `config.replace(overrides)` so key and cross-section checks run together. The configuration is immutable, and `to_mapping()` returns an independent effective mapping.

Configuration files compose in order. Asset paths belong to the file that supplies them. Changing an alternative's `kind` or `type` replaces that alternative; scalar/list overrides replace values, while named components preserve their ordering. Misspelled keys fail with their path. The final `docs/CONFIG.md` will be generated once through `hwoslaps reference` from the real owning tables; this guide does not duplicate its key inventory.

```python
from hwoslaps.config.schema import load_config

config = load_config(["scene.yaml", "instrument.yaml", "forecast.yaml"])
longer = config.replace({"observation": {"exposure_time_s": 1800.0}})
```

`config_digest` includes scientific inputs and referenced file bytes. The run label is bookkeeping. `comparison_digest` removes only `psf.model`; it is the required pairing identity for PSF knowledge-error reductions. Masks, nuisance names and truth kernel bindings are also checked. Input files or kernels changed after preparation require a new preparation.

A `PreparedForecast` owns the smooth expectation, kernels, pixel selection, nuisance design, workspace and template engine. Use its context manager to release resources. `Execution` selects `reference` or `jax`, reference-worker count and batch size. Device visibility belongs to the execution environment, for example `CUDA_VISIBLE_DEVICES`; it is not an instrument property.

## Forecasts and reductions

```python
from hwoslaps.fisher.api import Execution, forecast, prepare_forecast
from hwoslaps.analysis.reductions import aperture_selection, summarize
from hwoslaps.analysis.reach import mass_reach

with prepare_forecast(config, execution=Execution(engine="reference", reference_workers=1)) as prepared:
    result = forecast(prepared, masses_msun=[1e7, 1e8, 1e9])
selection = aperture_selection(result, centre_yx=(0.0, 0.0), radius_arcsec=1.0)
summary = summarize(result, q_threshold=10.0, selection=selection)
reach = mass_reach(summary, quantity="detectable_fraction", target=0.1, interpolation="linear")
```

`ForecastResult` arrays have axes `(mass, position)`. It holds raw/profiled information and, for PSF mismatch, fitted data/bias amplitudes. Its properties expose matched, mismatch and spurious statistics. Positions are `(y, x)` arcseconds; lattice metadata lives at `result.positions.grid`. Grid cell areas equal spacing squared. A subset that drops a boundary node has unknown boundary clipping, rather than a false unclipped flag.

`summarize` records the supplied threshold and selected metric. Mismatch detections require finite positive amplitudes as well as finite `q >= threshold`. Mismatch `q_max` zeroes nonpositive fitted amplitudes. Areas use detection count times cell area. Mass reach records sampled/bracketed results or bounds; it does not extrapolate. `interpolation="linear"` interpolates values in log mass, while `"log"` interpolates their logarithm. Choose it to suit the quantity and check nonmonotonic curves.

## Expected, noisy and null observations

Simulation takes the halo to inject and the detector-noise seed as separate arguments. Configuration `seed` drives scene randomness. `noise_seed=None` returns the expectation; an integer requests one noisy realization. The actual `Observation.kind` controls expected/noisy fitting.

```python
from hwoslaps.simulation import simulate

with prepare_forecast(config) as prepared:
    trial = prepared.hypothesis(1e8, (0.4, -0.6))
    expected = simulate(prepared, subhalo=trial, noise_seed=None)
    noisy = simulate(prepared, subhalo=trial, noise_seed=11)
    null = simulate(prepared, subhalo=None, noise_seed=11)
```

The null observation is a valid fit/control input. It is excluded from like-for-like injected-detection agreement. Simulation can deliberately inject another halo recipe for a truth-model study; inference refuses a trial recipe different from the configured hypothesis and refuses a non-null observation injecting a different trial. Treat these different studies as different comparisons.

Detector data/noise are ADU. Source/lens light maps are detected electrons per second per native pixel, split by plane. `sampling` is measured on the fiducial smooth scene once and retained for injected/noisy observations. It is a diagnostic; [SCIENCE](SCIENCE.md) explains its tested range and nonenforcement.

## PSF questions and chromatic weights

For PSF quality, vary truth and model together. For knowledge error, keep truth fixed and vary `psf.model`. External detector kernels enter through the same truth/model configuration and must match detector angular sampling. A kernel's support, bytes and sampling are part of its identity; copying it at the dataset boundary prevents backend normalization from changing the prepared object.

`knowledge_error_areas` separates retention, total detected-area ratio R, full-domain spurious area and in-selection spurious ratio F. Supply the aperture/selection and reference count floor. `knowledge_error_tolerance` requires one common keyed `(member, direction)` cohort at every amplitude and both maps. Eligibility comes from the reference floor, and nonfinite eligible values refuse. Clopper-Pearson confidence is supplied by the caller; pooled directions from one system carry the nominal label.

Each light SED group receives its effective kernel. Spectral weights store finite common-scale bin integrals and `log_rate_scale`; use `normalized` for relative weights. These values describe `throughput*fnu*dlnlambda`, not absolute detected photon counts. Absolute exposure rates come from the photometric normalization. Unit-sum support truncation and chromatic convergence limits are stated in [SCIENCE](SCIENCE.md).

Nonlinear imaging currently needs one distinct fitted model kernel. Chromatic truth may have multiple bound kernels when the fitted model uses a shared kernel, such as a monochromatic/kernel model. A matched model with multiple distinct kernels cannot be represented by this nonlinear likelihood. Preserve each truth group's binding in records.

## Nonlinear comparisons

`validate_nonlinear` takes a prepared forecast, physical `Halo`, actual `Observation`, `FitSpec`, `SamplerSettings`, separate `sampler_seed` and output directory. Optional `RefineSettings` requires a JAX analysis. The mask defaults to all native pixels minus the PSF border; choose the forecast-mask option or a Python `PixelMask` when the intended comparison requires it. Custom masks have a self-contained case record, distinct from the configuration grammar.

```python
from hwoslaps.inference.api import validate_nonlinear
from hwoslaps.inference.settings import FitSpec, SamplerSettings

with prepare_forecast(config) as prepared:
    trial = prepared.hypothesis(1e8, (0.4, -0.6))
    observation = simulate(prepared, subhalo=trial, noise_seed=None)
    case = validate_nonlinear(prepared, trial, observation,
        fit=FitSpec(mode="fixed_template"), sampler=SamplerSettings(use_jax=False),
        sampler_seed=11, output_dir="out/cases")
```

This is a source-checked call shape, without a promised sampler runtime or convergence outcome. `fixed_template` fixes the physical halo; `local_search` frees its position; `freed` also frees mass inside a supplied `MassSupport`. Use actual records to identify fitted parameters, masks and priors. Forecast background/wavefront nuisances are not automatically fitted by inference.

`q_signed = 2*(logL_subhalo-logL_smooth)` may be negative. `q_clipped` is a separate display value. Classification needs a complete caller `ClassificationRule`, including acceptance statuses and a nullable or positive stationarity tolerance. Failed/incomplete/unresolved cases remain distinct from accepted nondetections. Agreement reports mask/comparison differences and unfitted forecast nuisances. Retries may change sampler/refinement settings but preserve the physical case and comparison inputs.

Sampler recovery, weighted sampler quantiles and refined recovery are separate fields. Requested `use_jax` and the effective constructed search fields are also separate records: an inactive backend default such as `use_jax_vmap=True` does not alone identify the executed likelihood path. Repeatable-profile acceptance does not certify stationarity or a global optimum; inspect all six gates, projected gradient and repeat diagnostics. Some exact centres/circular mass shapes have unsupported gradients while value-only sampling remains available; see [SCIENCE](SCIENCE.md).

## Populations, batches, artifacts and plots

Population members use named per-member streams, so extending a pool or changing chunking does not redefine existing members. A ranking policy supplies cuts, weighted standardized terms and top-k size. No library paper cohort, threshold or instrument selection policy is inferred.

Batch source is defined and its integration/runtime checks are still pending. The source interfaces plan jobs, run/resume into an output directory and open metadata. A run report distinguishes completed, skipped, failed, duplicate, not-selected and orphaned jobs, preparations and revision counts. Resume verifies the recorded case/policy identity; a changed policy conflicts rather than silently reclassifying old outcomes. Full scientific case reading can require a backend, while controller/metadata imports stay separate. The current source signature is `run_batch(spec, output_dir, resume=True, execution=None, select=None, verify=False, require_single_revision=False)`. `open_batch` reads metadata; product accessors load forecasts, observations and cases. Final acceptance still needs the integrated producer and its tests.

Current artifacts record arrays, kernels, effective configuration, input hashes, code/environment provenance and schemas. Old study archives have no compatibility loader. Final CLI/artifact transport and generated-reference checks are pending integration. Keep the data, mask, kernel, covariance and nuisance span fixed when making a numerical comparison.

The plotting producer returns Axes and leaves saving to callers. It uses current lattice geometry, retains sparse/floor gaps and labels expected source S/N. Shared public helpers `plotting.axes.axes_or_new` and `pixel_extent` support those consumers; final plot runtime proof is pending.

## Staged command examples

These command forms come from the current staged CLI/example sources and are unexecuted in this draft. Use new output directories and the final integrated input assets.

```bash
hwoslaps validate configs/minimal.yaml
hwoslaps forecast configs/minimal.yaml --masses 1e7 1e8 1e9 -o out/minimal
hwoslaps simulate configs/minimal.yaml --smooth --noise-seed 11 -o out/null
python examples/hwo_reference/run.py --quick --q-threshold 10 --seed 11 --output out/hwo_quick
python examples/monolithic_illustrative/run.py --q-threshold 10 --output out/monolithic
python examples/chromatic/run.py --q-threshold 10 --output out/chromatic
hwoslaps batch plan examples/population/batch.yaml
hwoslaps batch run examples/population/batch.yaml -o out/population --devices cpu --select 'members/system_000000/*'
hwoslaps batch status out/population
```

Repeating the same batch run command resumes missing jobs. `--fresh` refuses an existing batch, `--verify` checks completed artifact hashes, and `--require-single-revision` enforces the recorded revision constraint. These source-defined forms are unexecuted here. The example budgets are acceptance targets, not measured runtimes. Chromatic sampling/support variants and arms have no convergence claim until their actual product comparisons pass.
