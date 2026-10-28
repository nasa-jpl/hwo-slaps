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

The `test` extra supplies the backend-free test runner. The `scientific-test` extra also
declares the independent interpolation and high-precision numerical oracles; `all`
includes it. Install the scientific stack with `install.sh` so the checked AutoArray
patches are applied in the selected environment.

Pull requests run a compact pinned CPU scientific job covering nuisance projection,
halo values, singular integration, padded rendering, Image gradients, low-dimensional
inference, posterior-quality reporting, and public geometry/spectral/kernel contracts.
The complete scientific and hardware suites remain explicit integration and release
checks. Independent analytic and high-precision oracles validate deliberate changes
from the historical numerical policy.

Markers and packaging metadata are defined in `pyproject.toml`. The installed-package
tests build a wheel without build isolation, unpack it into a temporary directory and run
the command from outside the source tree; they never install into the current
environment.

## Paper parity

`tests/parity/` compares five synthetic forecast scenes, optical kernels and fixed-vector
nonlinear likelihoods with values computed by the tagged RASTI code. A separate CPU
reference check uses the archived sys0043 production inputs: a COSMOS source, a 522 x 522
image at 0.00716 arcsec/pixel, the 999 x 999 science35 kernel and HWO detector noise.
The production oracle is a new evaluation of those inputs using the submitted code;
historical per-node forecast arrays were not archived.

`tests/scripts/generate_paper_parity.py` generates these tagged-code oracles. This command
writes only the synthetic inputs; it does not compare them with committed fixtures:

```bash
python tests/scripts/generate_paper_parity.py --inputs-only --out /path/to/new-inputs
```

Regenerating the original expected values requires a checkout of the tagged RASTI code
and a GPU. The separate `--production-anchor` mode requires the original configuration
and job receipt and runs on the CPU. Its oracle records input and submitted-source hashes.
The `*_8fa6209` fixtures instead preserve continuity with an intermediate engine revision;
they do not establish submitted-paper nonlinear refinement parity under the slam1 boxes.

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
