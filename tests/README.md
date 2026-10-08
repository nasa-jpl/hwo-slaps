# Tests

Tests are grouped by package. Run them from the repository root:

```bash
# Configuration, reductions and file formats, with the scientific libraries blocked:
python tools/run_core_tests.py tests -q

# Everything that runs on a CPU:
python tools/run_backend_tests.py tests -q -m 'not xtx_gpu and not xtx_multi_gpu'

# Tests that need one GPU, and tests that need two:
CUDA_VISIBLE_DEVICES=0 JAX_ENABLE_X64=1 python tools/run_backend_tests.py \
    --require-gpu --numba-jit enabled tests -q -m 'xtx_gpu and not xtx_multi_gpu'
CUDA_VISIBLE_DEVICES=0,1 JAX_ENABLE_X64=1 python tools/run_backend_tests.py \
    --require-gpu --numba-jit enabled tests -q -m xtx_multi_gpu
```

The core runner blocks the scientific libraries and checks that none was imported, so
configuration handling and result loading keep working without them. It sets its own
marker expression. The backend runner supplies the pinned AutoArray configuration in a
temporary directory. GPU runs fail if the requested CUDA runtime is missing. Batch tests
need Linux.

Markers and packaging metadata are defined in `pyproject.toml`. The installed-package
tests build a wheel without build isolation, unpack it into a temporary directory and run
the command from outside the source tree; they never install into the current
environment.

## Paper parity

`tests/parity/` compares the forecast, optics and nonlinear likelihood with fixtures
computed by the RASTI code. The fixtures and their inputs come from
`tests/scripts/generate_paper_parity.py`. To regenerate only the inputs and compare them
with the committed ones:

```bash
python tests/scripts/generate_paper_parity.py --inputs-only --out /path/to/new-inputs
```

Regenerating the expected values requires a checkout of the tagged RASTI code and a GPU.

## Writing tests

- Test one behaviour in one place. Extend an existing test or parameter table before
  adding a new file.
- Take expected values from an independent source: closed-form physics, the parity
  fixtures, or a comparison between the reference and JAX engines.
- Import the real package and the real scientific libraries. Do not add fake backend
  modules, broad skips, or production code that exists only for tests.
- A missing optional dependency should be explicit; a broken pinned dependency is a
  failure.
- When a test injects a fault to check that another test catches it, restore the
  original file afterwards.
