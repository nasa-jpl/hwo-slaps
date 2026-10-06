# Examples

Run commands from the repository root in the installed science environment. The HWO example
uses a pinned instrument reference; the other examples illustrate engine features. Choose a
new output directory for each run. Threshold 10 in the commands is a caller choice.
Thresholds have no driver default.

| Example | What it shows | Inputs | Budget | Measured runtime |
|---|---|---|---|---|
| [HWO reference](hwo_reference/README.md) | AB normalization, optical PSF and paper overlays | SEI v0.1.9 and a study band | CPU quick: 120 s; one-GPU full: 600 s | CPU 75.3120 s; GPU 55.3556 s |
| [Monolithic instrument](monolithic_illustrative/README.md) | Obscuration, spiders, shear, Sersic lens light and source-S/N mask | Illustrative round values | CPU: 120 s | 24.7548 s |
| [Chromatic](chromatic/README.md) | Two source SEDs and a monochromatic model arm | SEI curves with an assumed filter | One GPU: 600 s per run | Default grid 137.4071 s; all six convergence products within budget |
| [Kernel PSF](kernel_psf/README.md) | Matched and mismatched kernels, knowledge-error areas | Illustrative | CPU pair: 120 s | 8.8843 s |
| [Population](population/README.md) | Named member streams, forecasts and nonlinear batch jobs | Illustrative minimal scene | Selected member on CPU: 900 s | Pending |

Every forecast driver prints and saves the actual light-group values of
`Observation.sampling` in `run.json`. This is relative within-pixel variation of lensed
light at the configured oversampling. It does not by itself establish convergence of a
detection statistic. The chromatic example has separate wavelength and support checks.
Runtimes were measured on XTX on 2026-10-06 with one BLAS thread per process. The HWO
quick and monolithic drivers use eight CPU workers; GPU runs use one NVIDIA B200.
Individual READMEs record sampling values and the scope of the convergence comparison.
