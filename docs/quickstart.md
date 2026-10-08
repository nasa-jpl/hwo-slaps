# Quickstart

This page runs a complete forecast on a small example lens. It takes a few seconds on
a laptop CPU. Run the commands from the repository root, inside the environment
you created in [Installation](installation.md).

## The example configuration

`configs/minimal.yaml` describes an isothermal lens at redshift 0.2, an exponential
source at redshift 0.6, a small detector image and a simple PSF kernel. Its values are
illustrative; they describe no particular telescope.

```{literalinclude} ../configs/minimal.yaml
:language: yaml
:caption: configs/minimal.yaml
```

The sections are:

`cosmology`, `scene`
: The lens, the source and the kind of subhalo to forecast (an NFW halo here).

`psf`
: The point spread function. Here it is a kernel stored in a `.npy` file.

`instrument`, `observation`
: Detector gain, read noise and dark current, then the exposure time and sky level.

`forecast`
: Where to place trial subhalos (a 13 × 13 grid, 0.2 arcsec apart) and which pixels
  to use (all of them).

[Configuration files](guide/configuration.md) explains the format.

## Run a forecast from the command line

Check the file, then forecast five subhalo masses between 10⁶ and 10⁸ solar masses:

```bash
hwoslaps validate configs/minimal.yaml
hwoslaps forecast configs/minimal.yaml --masses 1e6 3e6 1e7 3e7 1e8 -o out/quickstart
```

The first command prints the configuration's digest. The second prints the path of
its output directory when it finishes. PyAutoLens may also print notices
about its JAX settings while it loads; they need no action.

The output directory must not exist yet. hwoslaps writes four files into it:

| File | Contents |
|---|---|
| `forecast.npz` | The forecast: statistics for every mass and position, with the configuration and provenance. |
| `effective_config.yaml` | The complete configuration used, with every default filled in. |
| `provenance.json` | The command, software versions and input file hashes. |
| `run.log` | The log of the run. |

## Read the result in Python

```python
from hwoslaps import load_forecast, mass_reach, summarize

result = load_forecast("out/quickstart/forecast.npz")
print(result.masses_msun)        # [1.e+06 3.e+06 1.e+07 3.e+07 1.e+08]
print(result.q_asimov.shape)     # (5, 169): one row per mass, one column per position
```

`result.q_asimov[i, j]` is the detection statistic *q* for mass `i` at position `j`.
Larger is more detectable; *q* is approximately the square of the detection
significance in standard deviations. [How hwoslaps works](concepts.md) defines it.

To turn the map into numbers, choose a detection threshold and summarize. The threshold
is always your choice; hwoslaps has no default. This example uses *q* ≥ 10:

```python
summary = summarize(result, q_threshold=10.0)
print(summary.q_max.round(1))                # [   1.5    9.    57.3  283.8 1459.7]
print(summary.detectable_fraction.round(2))  # [0.   0.   0.68 0.91 0.96]
```

`q_max` is the largest *q* over all positions at each mass. `detectable_fraction` is
the fraction of trial positions where a subhalo of that mass would be detected.

The **mass reach** is the mass at which a summary quantity crosses a target. Here,
the smallest mass whose best position reaches *q* = 10:

```python
reach = mass_reach(summary, quantity="q_max", target=10.0, interpolation="log")
print(reach.status, f"{reach.mass_msun:.3g}")  # bracketed 3.21e+06
```

`bracketed` means the crossing lies between two of your masses (3 × 10⁶ and 10⁷), so
the reach is interpolated. If no mass reached the target, the status would say so
instead of extrapolating. See [Summaries and mass reach](guide/analysis.md).

## Plot the result

```python
import matplotlib.pyplot as plt
from hwoslaps.plotting import plot_mass_curve, plot_statistic_map

fig, (left, right) = plt.subplots(1, 2, figsize=(10, 4))
plot_statistic_map(result, "q_asimov", mass_index=2, ax=left)
plot_mass_curve(summary, "q_max", reach=reach, ax=right)
right.set_yscale("log")
fig.savefig("quickstart.png")
```

```{figure} _static/quickstart.png
:alt: Left, a map of q for a 10^7 solar-mass subhalo, high along the ring and low at the centre. Right, q_max rising with subhalo mass on log axes, crossing q equals 10 at about 3 times 10^6 solar masses.
:width: 100%

Left: *q* for a 10⁷ M☉ subhalo at each trial position. The statistic is high where
the lensed arc is bright and falls to zero inside the ring. Right: the largest *q*
at each mass, with the target *q* = 10 and the interpolated mass reach.
```

Plotting functions return Matplotlib axes, so you can add labels, colorbars or
other artists before saving.

## Do the same entirely in Python

The command line is a thin wrapper around the Python interface. This script runs the
same forecast without leaving Python:

```python
from hwoslaps import forecast, load_config, mass_reach, prepare_forecast, summarize

config = load_config("configs/minimal.yaml")

with prepare_forecast(config) as prepared:
    result = forecast(prepared, masses_msun=[1e6, 3e6, 1e7, 3e7, 1e8])

summary = summarize(result, q_threshold=10.0)
reach = mass_reach(summary, quantity="q_max", target=10.0, interpolation="log")
print(f"mass reach: {reach.mass_msun:.3g} solar masses")
```

`prepare_forecast` does the expensive work once: it builds the smooth lens model,
the PSF, the expected observation and the noise model. `forecast` can then be called
as many times as you like with different masses or positions. The `with` block
releases the prepared resources when you are done.

## Next steps

- [How hwoslaps works](concepts.md) explains what *q* measures and why some lens and
  source parameters are marginalized.
- [Configuration files](guide/configuration.md) shows how to change the lens,
  instrument and exposure, and how to combine several files.
- [The HWO reference example](examples/hwo.md) runs the same steps for a realistic
  Habitable Worlds Observatory instrument.
