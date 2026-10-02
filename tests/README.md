# Scientific contract tests

The default test harness does not configure optional backends, alter Numba JIT,
or add study/script directories. Core tests use the package through ordinary
imports. CI routing is defined in `.github/workflows/tests.yml`.

A focused dependency-light check is:

```bash
python -m pytest -q tests/test_config_loading.py tests/test_population.py \
  tests/test_forecast_artifacts.py tests/test_cli.py tests/test_package_boundaries.py
```

The package-boundary keeper builds a wheel in a temporary project, imports it
outside the checkout, blocks optional backend imports, and invokes the actual
console entry point from wheel metadata. It needs the standard build tools
(setuptools, wheel, pip) but installs no wheel or optional backend.

Use the supported backend environment for the complete scientific suite:

```bash
python tools/run_backend_tests.py tests -q -m 'not xtx_gpu'
```

That explicit launcher prepares AutoArray configuration and isolates generated
logs in a temporary working directory. Its default Numba policy matches the
validated reference test lane. Compiled backend checks opt in explicitly:

```bash
python tools/run_backend_tests.py --numba-jit enabled tests -q
python tools/run_backend_tests.py --require-gpu --numba-jit enabled tests -q -m xtx_gpu
```

GPU CI requires a manually dispatched job on a configured GPU runner. The
launcher fails if CUDA is unavailable rather than reporting a passing skipped
GPU lane. Test results belong to their captured source snapshot; keep source
immutable during each run.
