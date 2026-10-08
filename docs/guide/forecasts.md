# Forecasts

A forecast has two steps. `prepare_forecast` builds everything that does not depend on
the subhalo: the smooth scene, the PSFs, the expected observation, the noise model and
the nuisance parameters. `forecast` then evaluates trial subhalos.

```python
from hwoslaps import forecast, load_config, prepare_forecast

config = load_config("configs/minimal.yaml")

with prepare_forecast(config) as prepared:
    coarse = forecast(prepared, masses_msun=[1e7, 1e8, 1e9])
    fine = forecast(prepared, masses_msun=[2e7, 3e7, 5e7])
```

Preparation is the slow part, so prepare once and call `forecast` as often as you need.
Use the `with` block, or call `prepared.close()`, to release worker processes and
device memory.

`prepare_forecast` also accepts a path or a list of paths directly:

```python
with prepare_forecast(["scene.yaml", "instrument.yaml", "forecast.yaml"]) as prepared:
    ...
```

## Masses

`masses_msun` is a list of positive masses in solar masses. The mass convention
depends on the subhalo type in `scene.subhalo`:

| Type | Mass |
|---|---|
| `PointMass` | Total mass |
| `SIS` | $M_{200c}$ |
| `NFW` | $M_{200c}$, with the concentration set in the configuration |
| `TNFW` | $M_{200c}$ of the parent NFW halo; the truncated total mass is recorded separately |

The NFW concentration is set in `scene.subhalo.concentration`. The minimal example uses
the Moliné et al. (2017) relation for subhalos, `{kind: moline2017_eq7, x_sub: 1.0}`, where
`x_sub` is the subhalo's distance from the host centre in units of the host virial radius. That
relation is calibrated for $M_{200c}$ between 10⁶ and 10¹² M☉, and HWO-SLAPS raises an error
for masses outside that range. For other masses, use a fixed concentration,
`{kind: fixed, value: 15.0}`, or a power law in mass and redshift (`kind: power_law`; see
the [configuration reference](../configuration.md)).

## Trial positions

`forecast.positions` sets where trial subhalos are placed. Positions are `(y, x)` in
arcseconds.

`grid`
: A square lattice centred on the lens.

  ```yaml
  forecast:
    positions: {kind: grid, spacing_arcsec: 0.05, half_width_arcsec: 1.5}
  ```

  Add `annulus: {inner_arcsec: 0.5, outer_arcsec: 1.5}` to keep only the lattice nodes in
  a ring around the lens. Grid results carry cell areas, so they support detectable
  areas and maps.

`ring`
: Equally spaced positions on a circle around the lens centre. The radius can be a
  number, `einstein_radius` or `critical_curve`, plus an optional offset.

  ```yaml
  forecast:
    positions: {kind: ring, count: 36, radius: einstein_radius}
  ```

`explicit`
: A list of positions.

  ```yaml
  forecast:
    positions: {kind: explicit, positions_yx: [[0.0, 1.0], [1.0, 0.0]]}
  ```

You can also pass positions to `forecast` directly. They must lie inside the region
covered by the prepared positions:

```python
result = forecast(prepared, masses_msun=[1e7], positions=[[0.0, 1.0], [1.0, 0.0]])
```

## Pixel masks

`forecast.mask` chooses the detector pixels that enter the statistic.

| `kind` | Pixels used |
|---|---|
| `all_pixels` | Every pixel of the image |
| `psf_border` | Every pixel at least half a PSF width from the image edge |
| `annulus` | Pixels between `inner_arcsec` and `outer_arcsec` from the lens centre (or from the grid centre with `about: grid`) |
| `source_snr` | Pixels where the expected lensed-source signal-to-noise is above `snr_min` |

`prepared.mask` is the resulting boolean image.

## Nuisance parameters

By default, every parameter of the lens mass, lens light and source light is profiled,
along with a constant background offset in ADU. `prepared.nuisances.names` lists them:

```python
print(prepared.nuisances.names)
```

```text
('lens.mass.main.centre_y', 'lens.mass.main.centre_x', 'lens.mass.main.einstein_radius',
 'lens.mass.main.ell_comp_1', 'lens.mass.main.ell_comp_2', 'source.light.disk.centre_y',
 'source.light.disk.centre_x', 'source.light.disk.ell_comp_1', 'source.light.disk.ell_comp_2',
 'source.light.disk.intensity', 'source.light.disk.effective_radius',
 'observation.background_offset_adu')
```

The `forecast.nuisances` section changes this:

```yaml
forecast:
  nuisances:
    fixed: ["source.light.disk.intensity", "lens.mass.main.centre_*"]
    priors: {lens.mass.main.einstein_radius: 0.01}
    steps: {position: 1.0e-4}
    background_offset: false
```

`fixed`
: Parameter names, or patterns with `*`, to hold at their true values. A fixed
  parameter cannot absorb any of the subhalo signal, so fixing parameters raises *q*.

`priors`
: Gaussian standard deviations, in the parameter's own units, for named parameters.

`steps`
: Finite-difference steps used to compute each nuisance column, by parameter kind
  (`position`, `einstein_radius`, `ellipticity`, `slope`, `multipole`, `shear`,
  `amplitude`, `size`, `sersic_index`, `orientation`) or by full name. The defaults
  are 10⁻³ for geometric parameters, a relative step of 1% for amplitudes and sizes,
  and 0.1 degrees for orientations.

`background_offset`
: Whether to profile a constant offset in ADU. On by default.

`wavefront`
: Wavefront modes of the model PSF to profile, for PSF studies. See
  [PSFs and PSF errors](psfs.md#wavefront-nuisances).

## Correlated noise

By default the noise is independent from pixel to pixel. To forecast with correlated
noise, give a covariance matrix over the full image as a `.npy` file:

```yaml
forecast:
  noise_covariance: covariance.npy
```

The matrix is square, with one row per pixel of the full image. Simulated noise and
nonlinear fits still use independent pixels.

## Engines and execution

`Execution` chooses how templates are computed. It is not part of the configuration
digest. The reference and JAX engines agree to a few parts per million.

```python
from hwoslaps import Execution

execution = Execution(engine="jax", batch_size=16)
with prepare_forecast(config, execution=execution) as prepared:
    result = forecast(prepared, masses_msun=[1e7, 1e8, 1e9])
```

`engine="reference"`
: The default. NumPy and PyAutoLens on the CPU. `reference_workers` sets the number of
  worker processes. With more than one worker, run your script under an
  `if __name__ == "__main__":` guard, because workers are started as new processes.

`engine="jax"`
: JAX on a GPU, or on the CPU if no GPU is visible. `batch_size` sets how many
  positions are evaluated together. Set `JAX_ENABLE_X64=1` and choose the device with
  `CUDA_VISIBLE_DEVICES` before Python starts.

The command-line equivalents are `--engine`, `--reference-workers` and `--batch-size`.

## The forecast result

`forecast` returns a `ForecastResult`. Its arrays have shape `(masses, positions)` and
are read-only.

| Attribute | Meaning |
|---|---|
| `masses_msun` | The evaluated masses |
| `positions_yx` | The evaluated positions, shape `(positions, 2)` |
| `q_asimov`, `z_asimov` | The detection statistic and its square root |
| `fisher_raw` | Information on the amplitude before profiling the nuisances |
| `degradation` | The fraction of the raw information that survives profiling |
| `sigma_amplitude` | The 1σ uncertainty on the subhalo amplitude |
| `q_mismatch`, `q_spurious` | Mismatch statistics, when the model PSF differs from the truth |
| `amplitude_hat`, `amplitude_spurious` | The fitted amplitudes behind them |
| `detection_metric` | `q_asimov`, or `q_mismatch` when the model PSF differs |
| `cell_areas_arcsec2`, `boundary` | Cell areas and edge flags for grid positions |
| `config` | The complete configuration as a dictionary |
| `provenance` | Digests, PSF and photometry records, nuisance names, engine and version |

`result.detections(q_threshold=10.0)` returns a boolean array of the same shape.
[Summaries and mass reach](analysis.md) reduces it to numbers.

## Other things a preparation gives you

| Attribute or method | What it is |
|---|---|
| `prepared.observation` | The expected smooth observation (an `Observation`) |
| `prepared.positions` | The prepared trial positions |
| `prepared.mask` | The boolean pixel mask |
| `prepared.nuisances.names` | The profiled parameters, in order |
| `prepared.config` | A copy of the configuration |
| `prepared.hypothesis(mass, (y, x))` | A `Halo` of the configured subhalo type, for simulations and fits |

If a file that the configuration references, such as a PSF kernel, changes on disk
after preparation, `forecast` raises an error. Prepare again to pick up the change.
