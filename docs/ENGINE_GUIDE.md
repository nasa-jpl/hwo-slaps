# Engine API and research workflows

The engine separates scene/instrument preparation, statistical evaluation,
experiment policy, and output. Configuration files compose in order; mappings
merge recursively while lists/scalars replace. A relative file path belongs to
the YAML file declaring it. Python mappings use the caller's directory or an
explicit `base_dir`. The input mapping is copied.

## Prepared forecasting

```python
from hwoslaps.config import load_config
from hwoslaps import prepare_forecast, forecast, summarize_forecast, mass_reach

config = load_config(["scene.yaml", "instrument.yaml", "forecast.yaml"])
prepared = prepare_forecast(config, backend="jax")  # or reference
result = forecast(prepared, masses=[1e7, 1e8, 1e9], positions=positions_yx)
summary = summarize_forecast(result, q_threshold=10, cell_areas_arcsec2=cell_areas)
reach = mass_reach(result.masses_msun, summary.detectable_fraction, target=0.1)
```

`PreparedForecast` contains the smooth scene, expected observation, truth and
fitted PSFs, and reusable nuisance/projection workspace. Preparation makes no
detector-noise draws. A context owns backend caches; use one per worker rather
than sharing it concurrently between threads.

The returned `ForecastResult` has mass-by-position arrays for q, raw/profiled
information, amplitude uncertainty, degradation, and optional mismatch/control
diagnostics. JAX mass retargeting reuses compiled products. Sparse evaluated
positions may supply `domain_positions` separately so the numerical support
remains the full declared domain.

Summaries default to the actual mismatch statistic when a fitted-PSF error is
present, otherwise the matched Asimov statistic. Positive-amplitude rules are
applied centrally by `result.detections(threshold)`. Use `metric="q_asimov"` only
when matched-template power is deliberately the question; `metric="q_spurious"`
selects the no-subhalo expected-image control statistic. Undefined diagnostics
remain visible and reductions reject consumed nonfinite values.

A fraction counts selected positions. A physical angular area requires explicit
quadrature cell areas; arbitrary sparse samples do not automatically define an
aperture or boundary. `selection` and `boundary` are caller-supplied masks.
`mass_reach` returns a sampled/bracketed crossing, a below/above-range bound, or a
nonmonotone status. It does not extrapolate censored or nonmonotone curves.
`adaptive_mass_reach` can refine a declared finite bracket with a bounded number
of evaluations.

## Instruments and PSF questions

YAML supports the existing optical provider, or `psf.provider: kernel` with
`psf.kernel.path` and `pixel_scale_arcsec`. NPY or NPZ kernels are detector-sampled
responses; an NPZ defaults to the `kernel` array or an explicit `array_key`.
A `fit_kernel` describes a different model PSF. File input paths and normalized
response hashes remain in the effective configuration for replay.

```python
from hwoslaps.psf import DetectorPSF

truth = DetectorPSF.from_array(calibrated_kernel, pixel_scale_arcsec)
model = DetectorPSF.from_array(model_kernel, pixel_scale_arcsec)
prepared = prepare_forecast(config, psf=truth, fit_psf=model)
result = forecast(prepared)
```

Angular sampling must match the scene. Normalization is explicit; no resampling
or invented pupil metadata occurs. Optical coefficient nuisance modes require
an optical provider with a declared wavefront basis. External kernels support
mass/position and actual-kernel mismatch forecasts without that extra model.

For PSF quality, generate and fit with the same provider/kernel at each chosen
quality. For PSF knowledge error, hold the truth observation/provider fixed and
vary the model kernel. Compare `detections` over the same masses and positions;
retained positions are the intersection with the correct-PSF detection mask.
Spurious-area ratios use the no-subhalo control over the entire declared domain,
not only previously sensitive positions. Undefined zero-denominator ratios need
an explicit caller policy. No fixed tolerance, amplitude list or retention gate
is embedded in the engine.

## Simulation and nonlinear comparisons

```python
from hwoslaps import simulate, validate_nonlinear

trial = prepared.trial(mass_msun=1e8, position_yx=(0.1, 0.2))
expected_injection = simulate(prepared, trial=trial, sample_noise=False)
noisy_injection = simulate(prepared, trial=trial, seed=42)
noisy_control = simulate(prepared, trial=trial, injected=False, seed=43)
check = validate_nonlinear(
    prepared, trial, observation=noisy_injection, dataset_kind="noisy",
    fit_mode="fixed_template", output_dir="new-fit",
)
```

`simulate` respects the configuration's subhalo flag unless a trial or explicit
`injected` value overrides it. A trial makes injection explicit; `injected=False`
produces a null control with the same observing setup. Seeds identify random
streams, not an assertion that two different Poisson means share identical noise.

With no `observation`, nonlinear validation generates the declared injection.
Pass a control observation explicitly to test a false detection. Choose
`dataset_kind="noisy"` to fit the measured image; `"asimov"` fits the expected
image stored with the observation. The current
consistent sampling contract, kernel identity, units, optimizer and acceptance
checks are preserved. Freed fits require explicit mass support; no mass-prior
range is guessed. Profile refinement requires a differentiable JAX backend and
explicit search settings. Fitting always requires an output directory.

## Morphology and populations

Image source profiles accept prepared assets with declared flux, angular scale,
position, rotation and size transformation. `scripts/prepare_source_image.py`
provides explicit preparation/normalization inputs; no fixed galaxy bank is
shipped. Analytic and image sources use the same simulation and forecast API.

`iter_population_configs` provides independent declared parameter distributions,
deterministic member/parameter streams, and unique integer noise seeds. A
population can also be a caller-owned table or iterator of correlated scenes.
The engine imposes no survey population or follow-up selection model.

Selection utilities measure electron SNR, angular gradient power and relative
complexity. Cuts, feature weights and top-k must be supplied; there are no
parent/selected/golden tiers. Ranking noisy survey data is a scientific question
for a study, not a claimed property of a noise-free example.

## Output and execution

`save_forecast_result`/`load_forecast_result` preserve explicit arrays and JSON
metadata in a versioned NPZ without pickle. They atomically refuse overwrite.
The CLI writes the effective configuration, provenance, log and result; Python
calculations return data and do not manage campaign fleets, deadline watchers,
release approvals, report tables or plotting.

Optional plotting utilities consume results/components explicitly. Additional
pupil/model families, chromatic rendering and flexible source reconstruction
need their own physical implementation and validation. The API does not turn
unsupported science into a configuration option.
