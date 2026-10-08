# Observations

`simulate` makes a detector image of a configuration, with or without a subhalo and
with or without noise. Simulations use the same scene, PSF and detector model as the
forecast, so they can be fitted, plotted or saved alongside it.

```python
from hwoslaps import load_config, prepare_forecast, simulate

config = load_config("configs/minimal.yaml")

with prepare_forecast(config) as prepared:
    subhalo = prepared.hypothesis(1e9, (0.0, 1.0))         # mass, (y, x)

    expected = simulate(prepared, subhalo=subhalo, noise_seed=None)
    noisy = simulate(prepared, subhalo=subhalo, noise_seed=11)
    control = simulate(prepared, subhalo=None, noise_seed=11)
```

`simulate` has two required keyword arguments:

`subhalo`
: The subhalo to add, as a `Halo`, or `None` for the smooth lens.

`noise_seed`
: An integer draws one noisy image with that seed. `None` returns the expected image,
  with no noise.

The scene's own random choices, such as a random subhalo position, use the
configuration's `seed`. Detector noise uses `noise_seed`. The two never share a
random stream, so you can redraw the noise without changing the scene.

## Subhalos

`prepared.hypothesis(mass_msun, (y, x))` returns a `Halo` of the type and concentration
set in `scene.subhalo`, at the lens redshift. It is the same subhalo the forecast
evaluates at that mass and position.

A configuration can also describe one injected subhalo in `scene.injection`:

```yaml
scene:
  injection:
    mass_msun: 1.0e8
    position: {kind: angle, angle_deg: 90.0}     # on the Einstein radius, straight up
```

The position can be `direct` (a fixed `(y, x)` centre), `angle` (a position angle on
a circle around the lens), or `random` (a random offset around a radius, drawn from
the configuration's `seed`).

## Smooth controls

To test for false detections, fit a smooth observation that contains noise:

```python
control = simulate(prepared, subhalo=None, noise_seed=11)
```

Use a noisy control, not the expected smooth image. A noise-free smooth image is fitted
perfectly by the smooth model, so it cannot show how often noise alone looks like a
subhalo.

## From the command line

`hwoslaps simulate` writes an observation file. Without `--smooth`, it uses the
subhalo in `scene.injection`:

```bash
hwoslaps simulate configs/minimal.yaml --noise-seed 11 -o out/injected
hwoslaps simulate configs/minimal.yaml --smooth --noise-seed 11 -o out/control
hwoslaps simulate configs/minimal.yaml --expected -o out/expected
```

Each command writes `observation.npz`, `effective_config.yaml`, `provenance.json` and
`run.log`.

## What an observation contains

| Attribute | Meaning |
|---|---|
| `kind` | `"expected"` or `"noisy"` |
| `data_adu` | The image, in ADU |
| `expected_adu` | The noise-free image, in ADU |
| `noise_map_adu` | The standard deviation of each pixel, from the expected image |
| `light_rate_e_per_s` | The convolved light of all planes, in detected electrons per second per pixel |
| `light_rate_by_plane_e_per_s` | The same, split into `"source"` and, with lens light, `"lens"` |
| `noise_seed`, `subhalo` | What was drawn and injected |
| `exposure`, `grid`, `psfs` | The exposure, pixel grid and PSF kernels used |
| `photometry` | How magnitudes became detected rates, when the configuration uses them |
| `sampling` | The pixel sampling diagnostic, described below |
| `config_digest` | The digest of the configuration |

An expected observation can draw noisy copies of itself:

```python
first = expected.draw(1)
second = expected.draw(2)
```

## Noise model

Each pixel's variance in electrons squared is

$$
\sigma^2 = \max(\text{light}, 0)\,t + \text{sky}\,t + \text{dark}\,t + N\,r^2,
$$

for exposure time $t$, $N$ exposures and read noise $r$ per exposure. The image and
noise map are converted to ADU with the detector gain. Throughput is applied once, to
the light; sky and dark rates are given as detected rates. The detector model has no
saturation, cosmic rays, interpixel capacitance or flat-field errors.

## Photometry

When light is given as an AB magnitude, `observation.photometry` records the
collecting area, the bandpass, the sky rate and each component's detected rate. For
the [HWO reference](../examples/hwo.md):

```python
print(prepared.observation.photometry.to_mapping())
```

## Pixel sampling

hwoslaps evaluates light on an oversampled grid, bins it to detector pixels, and then
convolves it with a pixel-integrated PSF. This is exact when the light is nearly
constant across each detector pixel. `observation.sampling` reports how far it is from
that, as the relative variation of light within a pixel, per light group:

```python
print(prepared.observation.sampling)     # {'source': 0.32274029790875214}
```

In the tests that calibrate this check, values below about 0.06 kept the error in the
subhalo signal information below one percent. hwoslaps does not enforce a limit.
For larger values, or a new kind of scene, compare against a configuration with finer
pixels before relying on the result. The minimal example uses coarse pixels on
purpose; the HWO reference has a value of about 0.024.
