# hwoslaps

hwoslaps computes the profiled linear-Gaussian statistic of a dark-matter subhalo over masses and positions in a strong-lens image. It supports studies of mass reach, source morphology, PSF quality, PSF knowledge error and chromatic imaging. Lens and source parameters, background and supported wavefront modes can enter the nuisance model. AutoLens and Nautilus provide nonlinear comparisons under specified fit bounds.

> Validation in progress: the public CLI, isolated installation, generated reference and four example families have passed their scoped checks. Batch execution and final combined validation remain open.

## Start here

- [Engine guide](docs/ENGINE_GUIDE.md)
- [Scientific conventions and limits](docs/SCIENCE.md)
- [Migration](docs/MIGRATION.md)
- [Configuration reference](docs/CONFIG.md), generated from the owning tables
- [Test instructions](tests/README.md), pending final test-tooling reconciliation

The Python core supplies configuration and array reductions. Rendering and nonlinear fitting need the supported science stack. The CPU installer and import-origin checks passed in an isolated XTX environment; GPU calculations were validated separately in the existing science environment. Choose the installer mode for your hardware:

```bash
python -m pip install .
bash install.sh --cpu
# For a CUDA installation:
bash install.sh --gpu
```

The quick start uses positional configuration files and a new output directory:

```bash
hwoslaps validate configs/minimal.yaml
hwoslaps forecast configs/minimal.yaml --masses 1e7 1e8 1e9 -o out/minimal
```

Use the typed configuration owner and supply the analysis threshold yourself. Masses are solar masses, positions are `(y, x)` arcseconds, and `interpolation` selects how values cross a target in log mass.

```python
from hwoslaps.config.schema import load_config
from hwoslaps.fisher.api import prepare_forecast, forecast
from hwoslaps.analysis.reductions import summarize
from hwoslaps.analysis.reach import mass_reach

config = load_config("configs/minimal.yaml")
with prepare_forecast(config) as prepared:
    result = forecast(prepared, masses_msun=[1e7, 1e8, 1e9])
summary = summarize(result, q_threshold=10.0)  # a caller choice
reach = mass_reach(summary, quantity="detectable_fraction", target=0.1, interpolation="linear")
```

## Examples

The examples distinguish reproduction inputs from illustrative instrument choices. These runtimes were measured on XTX on 2026-10-06; each linked README records the inputs and numerical limits.

| Example | Purpose | Input label | Execution status |
|---|---|---|---|
| [HWO reference](examples/hwo_reference/README.md) | Paper HWO pupil, SEI throughput and source/sky derivations | Reproduction targets | CPU quick 75.31 s; GPU full 55.36 s |
| [Monolithic instrument](examples/monolithic_illustrative/README.md) | A configurable monolithic instrument | Illustrative, no Euclid measurement claim | CPU 24.75 s |
| [Chromatic](examples/chromatic/README.md) | Multiple SED groups and a monochromatic fitted PSF comparison | Finite-support approximation | GPU grid 137.41 s; six-product convergence comparison passed |
| [Kernel PSF](examples/kernel_psf/README.md) | External matched/mismatched kernels and area reductions | Illustrative detector kernels | CPU pair 8.88 s |
| [Population](examples/population/README.md) | Member streams and resumable forecast/nonlinear jobs | Illustrative population | Batch validation pending |

A threshold of 10 in an example command is a reader choice, not a package detection rule. The generated products record the chosen threshold, input identities and execution settings.

The submitted RASTI implementation is retained at tag `rasti-26-183-submitted` (commit `41621de`). Paper fixtures in `tests/parity/` pin reference CPU and JAX GPU quantities; those named anchors establish their own numerical reproduction, not accuracy for every scene or calibrated blind-search significance. The final citation text and assembled package validation remain pending.

## Copyright

Copyright 2025, by the California Institute of Technology. ALL RIGHTS RESERVED. United States Government Sponsorship acknowledged. Any commercial use must be negotiated with the Office of Technology Transfer at the California Institute of Technology.

This software may be subject to U.S. export control laws. By accepting this software, the user agrees to comply with all applicable U.S. export laws and regulations. User has the responsibility to obtain export licenses, or other export authority as may be required before exporting such information to foreign countries or providing access to foreign persons.

## Authors

Georgios Vassilakis (JPL)
