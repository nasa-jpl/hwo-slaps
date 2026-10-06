# hwoslaps

hwoslaps computes the profiled linear-Gaussian statistic of a dark-matter subhalo over masses and positions in a strong-lens image. It supports studies of mass reach, source morphology, PSF quality, PSF knowledge error and chromatic imaging. Lens and source parameters, background and supported wavefront modes can enter the nuisance model. AutoLens and Nautilus provide nonlinear comparisons under specified fit bounds.

> Draft documentation: final CLI, example, installation and generated-reference checks are pending. The commands below describe the staged interfaces; no new example runtime or chromatic convergence result is claimed.

## Start here

- [Engine guide](docs/ENGINE_GUIDE.md)
- [Scientific conventions and limits](docs/SCIENCE.md)
- [Migration](docs/MIGRATION.md)
- [Configuration reference](docs/CONFIG.md), generated from the final owning tables when integration is ready
- [Test instructions](tests/README.md), pending final test-tooling reconciliation

The Python core supplies configuration and array reductions. Rendering and nonlinear fitting need the supported science stack. These installation routes are defined by the package/installer and still need final environment verification:

```bash
python -m pip install .
bash install.sh --gpu
```

The staged quick start uses positional configuration files and a new output directory:

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

## Examples awaiting execution

The current example sources distinguish reproduction inputs from illustrative instrument choices. Their final README and runtime records are pending.

| Example | Purpose | Input label | Execution status |
|---|---|---|---|
| `examples/hwo_reference/` | Paper HWO pupil, SEI throughput and source/sky derivations | Reproduction targets | Unexecuted current drivers |
| `examples/monolithic_illustrative/` | A configurable monolithic instrument | Illustrative, no Euclid measurement claim | Unexecuted |
| `examples/chromatic/` | Multiple SED groups and a monochromatic fitted PSF comparison | Approximation with required convergence study | Unexecuted; no convergence claim |
| `examples/kernel_psf/` | External matched/mismatched kernels and area reductions | Illustrative detector kernels | Generator and driver unexecuted |
| `examples/population/` | Member streams and resumable forecast/nonlinear jobs | Illustrative population | Source defined; producer/runtime checks pending |

A threshold of 10 in an example command is a reader choice, not a package detection rule. The generated products record the chosen threshold, input identities and execution settings.

The submitted RASTI implementation is retained at tag `rasti-26-183-submitted` (commit `41621de`). Paper fixtures in `tests/parity/` pin reference CPU and JAX GPU quantities; those named anchors establish their own numerical reproduction, not accuracy for every scene or calibrated blind-search significance. The final citation text and assembled package validation remain pending.

## Copyright

Copyright 2025, by the California Institute of Technology. ALL RIGHTS RESERVED. United States Government Sponsorship acknowledged. Any commercial use must be negotiated with the Office of Technology Transfer at the California Institute of Technology.

This software may be subject to U.S. export control laws. By accepting this software, the user agrees to comply with all applicable U.S. export laws and regulations. User has the responsibility to obtain export licenses, or other export authority as may be required before exporting such information to foreign countries or providing access to foreign persons.

## Authors

Georgios Vassilakis (JPL)
