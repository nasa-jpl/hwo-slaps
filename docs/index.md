# hwoslaps

hwoslaps forecasts how well a telescope can detect dark-matter subhalos in
galaxy-scale strong gravitational lenses. You describe a lens, a source, an
optical system and an exposure in a configuration file. hwoslaps then predicts
the detection significance of a subhalo of a given mass at every position
around the lensed arc.

```{figure} _static/hwo-reference.png
:alt: Left, a simulated Einstein ring observed with the HWO reference telescope. Right, a map of the forecast detection statistic q for a 10^8 solar-mass subhalo, which is largest on the brightest parts of the arc.
:width: 100%

A simulated HWO observation of a lensed ring (left) and the forecast detection
statistic for a 10⁸ M☉ NFW subhalo at each position (right).
```

With hwoslaps you can:

- map the detection statistic of a subhalo over mass and position;
- find the smallest detectable subhalo mass for a lens and an instrument;
- compare telescopes, detectors, exposure times and source morphologies;
- measure how errors in the model point spread function (PSF) bias detections;
- simulate expected and noisy observations of the same configuration;
- check the forecast against full nonlinear lens-model fits with PyAutoLens and Nautilus;
- run populations of lenses as resumable batches on CPUs or GPUs.

## Where to start

New users should read these in order:

1. [Installation](installation.md) sets up the package and its scientific dependencies.
2. [Quickstart](quickstart.md) runs a first forecast from the command line and from Python.
3. [How hwoslaps works](concepts.md) explains the statistic and the steps behind it.

## User guide

| Page | Covers |
|---|---|
| [Configuration files](guide/configuration.md) | The file format, combining files, overrides and digests |
| [Forecasts](guide/forecasts.md) | Masses, trial positions, pixel masks, nuisance parameters and engines |
| [Summaries and mass reach](guide/analysis.md) | Detection thresholds, apertures, areas, mass reach and plots |
| [Observations](guide/observations.md) | Expected and noisy images, smooth controls and the noise model |
| [PSFs and PSF errors](guide/psfs.md) | Kernel and optical PSFs, wavefront errors, knowledge error and chromatic PSFs |
| [Nonlinear fits](guide/nonlinear.md) | Checking a forecast with PyAutoLens and Nautilus fits |
| [Populations and batches](guide/batches.md) | Running many lenses as one resumable batch |
| [Saving and loading](guide/saving.md) | Result files and the provenance they carry |

## Examples

Each example is a set of configuration files and a driver script in the repository's
`examples/` directory.

| Example | Shows |
|---|---|
| [HWO reference](examples/hwo.md) | The RASTI paper's HWO set-up: segmented telescope, AB photometry and saved products |
| [Monolithic telescope](examples/monolithic.md) | A circular telescope with obscuration and spiders, lens light and a signal-to-noise mask |
| [Chromatic PSF](examples/chromatic.md) | Broadband PSFs for sources of different colours, with a convergence check |
| [Kernel PSFs](examples/kernel_psf.md) | PSF kernel files and PSF knowledge-error areas |
| [Population batch](examples/population.md) | A population of lenses run as a resumable batch, with a nonlinear fit |

Only the HWO reference reproduces a published set-up. The other examples use
illustrative values chosen to exercise features of hwoslaps.

```{toctree}
:hidden:
:caption: Getting started

installation
quickstart
concepts
```

```{toctree}
:hidden:
:caption: User guide

guide/configuration
guide/forecasts
guide/analysis
guide/observations
guide/psfs
guide/nonlinear
guide/batches
guide/saving
```

```{toctree}
:hidden:
:caption: Examples

examples/hwo
examples/monolithic
examples/chromatic
examples/kernel_psf
examples/population
```

```{toctree}
:hidden:
:caption: Reference

api_overview
configuration
command_line
conventions
migration
```

```{toctree}
:hidden:
:caption: Development

development
```
