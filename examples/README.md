# Examples

Run commands from the repository root in the installed science environment. The HWO example
uses a pinned instrument reference; the other examples illustrate engine features. Choose a
new output directory for each run. Threshold 10 in the commands is a caller choice.
Thresholds have no driver default.

| Example | What it shows | Inputs | Budget | Measured runtime |
|---|---|---|---|---|
| [HWO reference](hwo_reference/README.md) | AB normalization, optical PSF and paper overlays | SEI v0.1.9 and a study band | CPU quick: 120 s; one-GPU full: 600 s | Pending |
| [Monolithic instrument](monolithic_illustrative/README.md) | Obscuration, spiders, shear, Sersic lens light and source-S/N mask | Illustrative round values | CPU: 120 s | Pending |
| [Chromatic](chromatic/README.md) | Two source SEDs and a monochromatic model arm | SEI curves with an assumed filter | One GPU: 600 s per run | Pending |
| [Kernel PSF](kernel_psf/README.md) | Matched and mismatched kernels, knowledge-error areas | Illustrative | CPU pair: 120 s | Pending |
| [Population](population/README.md) | Named member streams, forecasts and nonlinear batch jobs | Illustrative minimal scene | Selected member on CPU: 900 s | Pending |

Every forecast driver prints and saves the actual light-group values of
`Observation.sampling` in `run.json`. This is relative within-pixel variation of lensed
light at the configured oversampling. It does not by itself establish convergence of a
detection statistic. The chromatic example has separate wavelength and support checks.
Runtime and sampling values for these exact inputs await measurement.
