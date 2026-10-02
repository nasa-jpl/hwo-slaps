# Cleanup validation, 2 October 2026

The reference is the submitted `rasti` commit
`41621de8ca861e425e1caddb7a599d5c1032751f`. Validation uses separate
baseline and candidate source snapshots on XTX. The existing checkout and
installed dependencies were left unchanged.

## Runtime and isolation

- Python: `/data/home/gvassilakis/Software/miniconda3/envs/hwo-slaps/bin/python`
  (3.11.15).
- NumPy 1.26.4, SciPy 1.17.1, JAX/JAXlib/CUDA plugin 0.4.38,
  PyAutoArray/PyAutoFit/PyAutoGalaxy/PyAutoConf 2026.5.14.2,
  Nautilus 1.0.5, scikit-learn 1.8.0, pytest 9.0.3.
- PyAutoLens source: `10bfea51ea95fb02087147190da562e186535d7f`; HCIPy
  source: `cc853b392463c33f02db6d20ce16dce0f7d10e2e`.
- Snapshot root: `/data/home/gvassilakis/forecast-engine-cleanup-20261002/`.
- CPU suite: GPUs hidden, `JAX_PLATFORMS=cpu`, `NUMBA_DISABLE_JIT=1`,
  four BLAS/OpenMP threads, affinity 32–39.
- GPU canaries: one idle B200, `CUDA_VISIBLE_DEVICES=1` for the candidate (GPU 0 for the baseline),
  `JAX_PLATFORMS=cuda`, four BLAS/OpenMP threads, affinity 40–47. GPU 1 was idle before the candidate
  run and returned to 0 MiB after it completed.

The snapshots contain shallow Git metadata for the real reference commit.
Five initial baseline provenance assertions failed because the archive lacked
`.git`; all five passed after metadata was restored, with no source changes.
These were snapshot setup failures, not baseline scientific failures.

## Results

| Check | Baseline | Candidate |
| --- | --- | --- |
| Full CPU suite | 1,986 passed; 15 skipped | 2,107 passed; 15 skipped; 1 strict xfail (212.88 s) |
| GPU numerical canaries | 13 passed in 21.49 s | 13 passed in 20.56 s |
| GPU worker/spawn contracts | 4 passed in 7.16 s | 4 passed in 7.09 s |
| Wheel contents and CLI entry point | Not applicable | Built; 79 entries; smoke passed |

The final candidate has no unexpected failures and adds 121 passing contracts.
The expected scheduler failure is described below. Of the 15 CPU skips, 13 require an available JAX GPU backend; two omit
inapplicable concentration-relation combinations for SIS and PointMass.
The GPU passes total 17 tests per snapshot and exercise all 13
GPU-dependent skips as well as four numerical parity cases also covered on CPU.

The CPU baseline covers all 2,001 collected tests: 1,981 passed in the initial
214.76 s run, followed by the five provenance tests passing in 1.99 s.

The GPU canaries compare Fisher JAX templates, maps, PSF mismatch maps, and
retargeted engines with reference calculations. They also check nonlinear
JAX/NumPy fitness parity for the supported model families, float64 output,
x64 setup in a fresh interpreter, recovery from nonfinite dynamic vectors,
and exact serial-versus-pooled Nautilus emulator weights. Four further
worker/spawn checks cover reference-worker overrides, override rejection,
JAX worker routing, and serial/parallel PSF mismatch map agreement.

The built wheel excludes `studies/` and `scratch/` and exposes
`hwoslaps = hwoslaps.cli:main`. A fresh interpreter imported the wheel from
outside the checkout while a finder actively rejected imports of AutoLens,
AutoGalaxy, AutoArray, AutoFit, HCIPy, and JAX. Engine, CLI, population sampling,
and nonlinear settings imports all passed. No package was installed.

The 248-file code, recipe, configuration, and test-input manifest matched
the local source and remote snapshot at the test freeze:
`5c7fb7c80ff97ac6cc227a53e775c19f28604a59974b6b0ee4d0126a01c13750`.
The final formatted source manifest is
`b9221618424d9c2bbc5839df3a7e06546516bcd4478bb517eef0bee28f32f591`.
Only `population.py` and `test_population.py` differ, through whitespace and
docstring wrapping. Their executable abstract syntax trees match the tested
files exactly after removing docstrings. The 14 population tests passed again
in 0.36 s after formatting. Focused Flake8 passed for the population module,
its tests, `setup.py`, and the package initializer. Flake8 ran in the existing
local `hwo-slaps` environment because the XTX environment lacks it.

The wheel was built from the tested snapshot before these formatting edits.
Its SHA-256 is
`c142c956603d91c854c2af44c046f3cddceec0879a9b684c32c3469093246860`.

Raw receipts are `baseline_cpu.log`, `baseline_provenance.log`,
`baseline_gpu.log`, `baseline_gpu_workers.log`, `candidate_cpu_final.log`,
`candidate_gpu_final.log`, `candidate_gpu_workers.log`,
`wheel_build.log`, `wheel_smoke.log`, and `source_manifest_final.json` under the
remote snapshot root.

## Repeating the checks

From the candidate snapshot, run:

```bash
CUDA_VISIBLE_DEVICES='' JAX_PLATFORMS=cpu NUMBA_DISABLE_JIT=1 \
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 \
PYTHONPATH=src:. taskset -c 32-39 \
/data/home/gvassilakis/Software/miniconda3/envs/hwo-slaps/bin/python \
-m pytest tests -q --disable-warnings --tb=short
```

GPU canaries use `tests/test_fisher_grid_map.py`'s JAX reference and retarget
checks, `tests/test_nautilus_training_pool.py::test_pooled_weights_match_serial`,
and the `xtx_gpu` nonlinear x64, dtype, model-parity, and nonfinite-recovery
checks in `tests/test_nonlinear_jax_port.py`.

## Limits

These checks establish regression coverage for the tested contracts and tiny
numerical fixtures. They do not establish full-campaign throughput, production
checkpoint equivalence, or scientific validity for a new telescope or
population. No production fits or campaigns were launched. Correctness concerns
identified during cleanup are recorded separately from refactoring changes.
The strict expected failure in `tests/test_fisher_runtime.py` records the
pre-existing supervised scheduler bug that can omit unscheduled inputs when
an entire pending batch completes. See [Fisher cleanup](fisher-cleanup.md).
