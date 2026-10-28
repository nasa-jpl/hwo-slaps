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

`tests/parity/` keeps historical fixtures and manifests byte-for-byte, with a separate
SHA-256 seal. Unchanged optics, prior, mask and asset identities remain exact. Corrected
rendering, halo evaluation, Image geometry and nuisance projection use independent
current-policy oracles rather than the old output arrays.

The forecast oracle applies pivoted augmented Householder QR to live whitened nuisance
columns and Gaussian-prior rows, then measures residual norms and signed inner products.
Every archived mass is tested on a declared centre/opposite-corner subset of the five
synthetic grids. The production control uses all three archived sys0043 positions and
its original COSMOS, 522 x 522 detector, 999 x 999 kernel and HWO inputs. Each process
shares a checked preparation across engines. Receipts attached to the JUnit XML report
cohort indices, historical/current values, changed input digests and actual execution.

Current fixed-vector likelihoods are checked against independent Gaussian arithmetic;
JAX gradients are checked against finite differences. Current fit/refinement runs use
an explicit per-role likelihood-call cap of 12,000. Archived intermediate-engine maxima
are independently rescored on current data as comparison candidates, and provisional
posterior/evidence outputs keep their quality labels. No sampler budget is raised or
run relaunched automatically by these tests.

`tests/scripts/generate_paper_parity.py` remains the historical tagged-code generator.
Its `--inputs-only` mode writes inputs without comparing committed fixtures. Its
`--production-anchor` mode recorded a new submitted-code evaluation of archived inputs;
those arrays were not historical per-node campaign outputs. Regenerating expected arrays
with the current implementation would not supply an independent oracle.

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
