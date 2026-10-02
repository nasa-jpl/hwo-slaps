# HWO-SLAPS

HWO-SLAPS simulates strong-lensing observations and forecasts subhalo sensitivity.
The engine separates scene construction, optical response, detector noise,
Fisher profiling, nonlinear comparisons, and campaign execution. Instrument
parameters, source assets, inference settings, and populations are explicit
inputs rather than assumptions from one paper.

## Start here

- [Engine API, configuration, and population recipes](docs/ENGINE_GUIDE.md)
- [Migration from the submitted RASTI branch](docs/engineering/MIGRATION.md)
- [Preserved RASTI study](studies/rasti/README.md)
- [Paper-code checkpoint](reproducibility/rasti-26-183/README.md)

This branch is a first refactoring pass. It preserves the existing physical
models and optimized kernels; the interfaces do not imply support for every
telescope pupil or unrestricted source reconstruction. The submitted code
remains at the `rasti-26-183-submitted` tag.

## Run a configuration

Use the existing science environment (see `install.sh`), then install this
checkout with `python -m pip install -e . --no-deps`. The command is:

```bash
hwoslaps -c configs/master_config.yaml --output-dir outputs
```

For source checkouts, `python runner.py` accepts the same arguments. Repeat
`-c` to compose instrument, scene, and forecast fragments in order. Later
mappings merge recursively; lists and scalar values replace earlier values.
Relative file paths belong to the YAML file declaring them. Use
`--base-dir .` explicitly when replaying old repository-relative inputs.

```bash
hwoslaps -c configs/master_config.yaml -c configs/examples/observation_override.yaml --validate-only
```

Validation creates no outputs and initializes no scientific backend.

## Python API

```python
from hwoslaps.config import load_config
from hwoslaps import run_pipeline

config = load_config("configs/master_config.yaml")
result = run_pipeline(config)
```

Use `run_with_artifacts` when a resolved configuration snapshot, log, and
provenance record are required. Individual lensing, PSF, and observation
functions also accept explicit seeds and detector sampling without a complete
pipeline configuration. The guide describes their supported contracts.

## Development

The reusable package is under `src/hwoslaps`. The source-only `studies/rasti`
namespace contains frozen paper population contracts and execution recipes;
it is excluded from the wheel and imported only by reproduction tools/tests.
The core package must not import this study namespace.

```bash
python -m pytest -q tests/
```

GPU tests carry the `xtx_gpu` marker and require the existing pinned backend.
See [validation](docs/engineering/validation.md) for this cleanup session's
actual checks and remaining concerns.

## Copyright

Copyright 2025, by the California Institute of Technology. ALL RIGHTS RESERVED. United States Government Sponsorship acknowledged. Any commercial use must be negotiated with the Office of Technology Transfer at the California Institute of Technology.

This software may be subject to U.S. export control laws. By accepting this software, the user agrees to comply with all applicable U.S. export laws and regulations. User has the responsibility to obtain export licenses, or other export authority as may be required before exporting such information to foreign countries or providing access to foreign persons.

## Authors

Georgios Vassilakis (JPL)
