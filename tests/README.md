# Scientific contract tests

The test lanes use current package owners through ordinary imports. Run all science
in the supported XTX environment; never install or execute the scientific stack on
the local Mac. Source snapshots and receipts identify the tested code.

```bash
python tools/run_core_tests.py tests -q
python tools/run_backend_tests.py tests -q -m 'not xtx_gpu and not xtx_multi_gpu'
python tools/run_backend_tests.py --require-gpu --numba-jit enabled tests -q -m 'xtx_gpu and not xtx_multi_gpu'
```

The core launcher blocks actual scientific backend loading, permits harmless
optional-dependency discovery and verifies that none loaded after execution. Its
marker expression is fixed. The backend launcher supplies the pinned AutoArray
configuration in a temporary directory. GPU lanes require assigned devices and fail
when the requested CUDA runtime is absent. Runtime defaults, backend flags and JIT
settings are not changed by ordinary fixtures.

Markers and packaging metadata have one owner, pyproject.toml. Installed-boundary
tests build a real wheel with no build isolation, extract it into a temporary
directory and invoke the command from outside the source checkout. They assert
the actual imported wheel origin, package data and backend-free validation. They
do not install a wheel or modify the current environment.

Paper input generation is separate from paper execution:

```bash
python tests/scripts/generate_paper_parity.py --inputs-only --out /path/to/new-inputs
```

Compare the five generated engine YAML files and synthetic assets with their
committed inputs before using a regenerated anchor. Reference execution requires
the extracted submitted-paper tree and explicit work/GPU arguments. No source
or numerical change is certified by a snapshot from another head.

Retained old suites and their required source remain until their exact declarations
are reconciled against actual keepers. A mention inventory does not authorize
retirement. The final full core/CPU/GPU/static gates run after the reviewed retirement
manifest is applied; provisional retained dependencies are never hidden by weaker
test selection or import exceptions.
