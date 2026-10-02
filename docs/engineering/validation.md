# Final engine validation — 2026-10-02

The final source passed the full supported XTX CPU and CUDA lanes, the
backend-denied core CI lane, clean-wheel checks, and a real public API-to-CLI
replay. No tests, runtime probes, benchmarks, package installations, or
production fits ran locally after the user's explicit execution restriction.
No merge or push was performed by validation.

## Exact source and runtime

The baseline is `bf73e2ce1b05377d6b5e9c206fd1b30d2c88ec7f`, materialized from
`git archive` with real shallow Git objects and a clean worktree. The final
immutable remote source is:

`/data/home/gvassilakis/forecast-engine-platform-20261002/candidate-final-10`

The final payload contains 184 audited source, test, configuration, package,
CI, example and engineering-documentation files. It contains no credentials,
symlinks, unrelated scratch probes, study tree, or reproduction archive.

- Archive SHA-256: `c5cbd3a2711dade528fa1349d960bc01ba05ef6fca5cab6feb162f235cea5c0a`.
- Tested Python/test/package manifest SHA-256: `aec5638a9a1138085f5d8a0bf079211c9764c3569ee3ade5c6d4bc4792b66e61`.
- Final local manifest SHA-256: `8e267173b6377cff5e905c369643ca68744ac7bd155fa10f59869f4f3abc5f6b`.

The sole post-snapshot code/test byte difference removes two trailing EOF
bytes from `tests/test_nonlinear_profile_calibration.py`. Its executable AST
is identical, and its exact final bytes passed both tests separately. Every
other tested Python/package file and every loaded configuration matches the
local final source byte for byte. Documentary ledger updates are excluded from
that executable-source comparison.

Runtime: Python 3.11.15; NumPy 1.26.4; SciPy 1.17.1; JAX/JAXlib/CUDA plugin
0.4.38; AutoArray, AutoFit and AutoGalaxy 2026.5.14.2; AutoLens 1.0.dev0;
Nautilus 1.0.5; pytest 9.0.3. Hardware: NVIDIA B200. The CPU backend launcher
uses the existing supported configuration with Numba disabled. The strict GPU
CI command uses `--require-gpu --numba-jit enabled`, x64 and explicit device
allocation.

## Completed gates

All receipt paths below are under:

`/data/home/gvassilakis/forecast-engine-platform-20261002/receipts/`

| Gate | Actual result | Receipt |
| --- | --- | --- |
| Pinned baseline CPU | 2,107 passed, 2 skipped, 1 known scheduling xfail, 13 GPU cases deselected | `baseline_bf73e2c_cpu.xml`, `.log` |
| Baseline GPU | 13 marked + 11 additional existing Fisher cases passed | `baseline_bf73e2c_gpu.xml`, `baseline_bf73e2c_fisher_gpu.xml` |
| Final full CPU | 1,380 passed, 2 skipped, 13 GPU cases deselected; 87.13 s | `final_10_cpu.xml`, `.log` |
| Final strict GPU CI | 13 passed; 23.69 s | `final_10_gpu.xml`, `.log` |
| Final extra Fisher GPU | 14 passed; 13.21 s | `final_10_fisher_gpu.xml`, `.log` |
| Exact minimal core CI | 343 passed, zero skips; 5.96 s | `final_10_core_ci.xml`, `.json`, `.log` |
| Exact final EOF bytes | 2 passed; 0.62 s | `final_10_eof.xml`, `.log` |
| Clean wheel | 73 distribution files; no study/test/reproduction payload; all public imports quiet, actual NPZ round trip | `final_10_wheel.json`, `final_10_wheel_smoke.log` |
| Foreign-directory wheel CLI | Help and validation exited 0 with optional backends denied | `final_10_wheel_foreign_cli.json` |
| Real wheel forecast replay | CLI validation and seven-mass forecast exited 0; 7 × 169 finite arrays identical to public quickstart | `final_10_installed_cli_replay.json`, command logs |

The two CPU skips are exactly the pre-existing power-law concentration cases
for SIS and PointMass, which have no concentration relation. There are no new
skips, failures, errors, missing test files, or xfails. The scheduling xfail was
repaired and is now an ordinary passing regression.

The minimal CI command reads the actual 18-file workflow list and actively
blocks Astropy, PyAuto packages, HCIPy, JAX, Matplotlib, Nautilus, Torch and
TensorFlow through inherited `sitecustomize`/MetaPathFinder policy. It does not
merely inspect import strings or rely on those packages being absent.

## Preservation and retirement accounting

Baseline per-file/case state was recorded before pruning: all 95 test files,
2,123 unique cases, no missing files. Final accounting records all 75 test
files. There are 33 deliberately retired files and 13 new files. Every retired
file is accounted for in the owner ledgers; the obsolete Pipeline file's three
declarations have explicit public API/config/CLI keeper or retired-wrapper
mappings. The detailed case receipts are `baseline_per_file_cases.json` and
`final_per_file_cases.json`. The independent review did not treat reduced test
counts as proof of preserved behavior.

Seven deliberate faults were caught at their intended owner assertions in
separate disposable XTX copies: discarded covariance off-diagonals, omitted
throughput, open rather than closed aperture boundary, wrong Image flux path,
JAX raw rather than profiled reduction, doubled SIS magnification, and default
validation using the null image rather than the injected trial. Original source
bytes were restored and SHA-256 verified after every mutation. Receipts:
`keeper_mutation_controls.json`, `extra_keeper_mutation_controls.json` and
`control_*.log`. No shared source or active test snapshot was mutated.

The three baseline defects have explicit failing controls and passing final
regressions: posterior-start separation serialized as the default; scheduler
batch exhaustion dropped six of ten inputs; actual smooth scenes with a null
subhalo mass failed trial construction for NFW, SIS and PointMass. Controls:
`control_settings_baseline.*`, `control_scheduler_baseline.*`, and
`control_smooth_trial_baseline.*`. The fixes are isolated in commit `2f82967`.

The changed-fit-file identity canary silently rebound the calibration in the
old source and now rejects it (`control_fit_replay_final06.log` versus
`fit_replay_final10.log`). Matched observation PSFs accept only original bytes
or the exact documented AutoLens sum-normalization, demonstrated by
`matched_psf_normalization_control08.json`; approximate comparisons are not used.

The current nonlinear backend cannot evaluate a 1 × 1 fitted kernel because
its empty blurring grid indexes a zero-length sampler. A direct upstream-only
control reproduced that failure and a 3 × 3 supported case succeeded. The API
now rejects that unsupported fitted-kernel domain early. Valid injection and
shared-ROI contracts use actual supported kernels; no likelihood, assertions,
or backend exception were bypassed.

## Bounded numerical and performance evidence

The same 61 × 61 scene, 11 × 11 kernel, 81 candidate positions and 10^8 solar
mass subhalo were evaluated through reference and JAX engines. Final and
baseline arrays are bit-identical within each backend:

- Reference q-array SHA-256: `e7dd526b51181f97a0e74c30c1ea8ef7c8694fafa10152e1a5fd569566ca7077`.
- JAX q-array SHA-256: `4b45e24cd17f8a5688c6a0f1c1cff0e1b6fcfd2a06aeb0476c7772f1a653af76`.

Reference/JAX agreement passed relative tolerance 10^-6; maximum absolute
backend difference was 1.9265 × 10^-5 and was unchanged from baseline. Three
warm evaluations gave baseline medians 0.820 s reference and 3.785 ms JAX;
final equivalent numerical code gave 0.870 s reference and 3.404 ms JAX.
Cold JAX times were 1.244 s and 1.163 s. These short, concurrent canaries do not
establish production-scale timing equivalence or a universal speedup. Receipts:
`benchmark_baseline.json` and `benchmark_candidate_final08.json`. Later changes
only touch API guards, exact kernel-identity acceptance and tests; the three
numerical performance owners are unchanged.

The standalone example ran with the real public API and enabled JIT, without
test-harness setup or fits. The final built-wheel CLI replayed its saved clean
configuration into a new directory. Mass, position and q arrays were exactly
equal, with maximum q difference zero. The example's 10% area reach was below
the sampled mass range and remained explicitly censored rather than being
extrapolated.
