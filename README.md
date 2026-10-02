# HWO-SLAPS

A configurable strong-lensing pipeline for subhalo sensitivity forecasts,
source-morphology comparisons, instrument/PSF experiments, and targeted nonlinear
validation. The scientific path is explicit: prepare a scene and observation,
evaluate masses and positions, reduce the result, and choose how to save it.

```python
from hwoslaps import prepare_forecast, forecast, summarize_forecast, mass_reach
from hwoslaps.config import load_config

prepared = prepare_forecast(load_config("configs/master_config.yaml"))
result = forecast(prepared, masses=[1e7, 3e7, 1e8, 3e8, 1e9])
summary = summarize_forecast(result, q_threshold=10)
reach = mass_reach(result.masses_msun, summary.detectable_fraction, target=0.1)
result.save_npz("forecast.npz")
```

Masses are solar masses; positions are `(y, x)` arcseconds. Thresholds, area
fractions, population distributions, selection policy, PSFs and execution
settings are explicit inputs. Source assets can be analytic or image based.
The reference renderer and validated JAX acceleration share the same numerical
contracts. External detector-sampled PSFs use the same public forecasting path.

Run the small, explicitly synthetic example in the supported backend environment:

```bash
python examples/quickstart.py --output-dir new-example --backend jax
```

It creates a detector-integrated Gaussian kernel, computes a mass bank, reports
sensitive fractions/censoring, and saves replayable inputs and results.

## Start here

- [Engine API and research workflows](docs/ENGINE_GUIDE.md)
- [Scientific conventions and limits](docs/SCIENCE.md)
- [Migration to the explicit API](docs/engineering/MIGRATION.md)
- [Testing and ownership](tests/README.md)

The base package provides configuration, array-level statistics, populations,
and result I/O. Scene rendering and nonlinear inference require the supported
scientific backend; `install.sh` describes its pinned developer environment.
In that environment, install this checkout with
`python -m pip install -e . --no-deps`.

```bash
hwoslaps validate -c configs/master_config.yaml
hwoslaps forecast -c configs/master_config.yaml --output-dir new-results --masses 1e7 1e8 1e9
hwoslaps simulate -c configs/master_config.yaml --output-dir new-observation
```

Output directories and files refuse overwrite. Python calculations create no
plots or result artifacts automatically. Use `simulate` for injected/noisy/null
observations, `prepare_forecast` for the smooth expectation, and
`validate_nonlinear` for explicitly requested fitting.

Study reproduction machinery, fixed cohorts, production asset banks, release
controllers, and historical replay routes have been removed. Git history retains
the submitted implementation; new studies use ordinary configurations and
iterators over the same engine API.

## Copyright

Copyright 2025, by the California Institute of Technology. ALL RIGHTS RESERVED. United States Government Sponsorship acknowledged. Any commercial use must be negotiated with the Office of Technology Transfer at the California Institute of Technology.

This software may be subject to U.S. export control laws. By accepting this software, the user agrees to comply with all applicable U.S. export laws and regulations. User has the responsibility to obtain export licenses, or other export authority as may be required before exporting such information to foreign countries or providing access to foreign persons.

## Authors

Georgios Vassilakis (JPL)
