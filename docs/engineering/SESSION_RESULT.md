# Subhalo forecasting platform result

The cleanup branch now implements explicit simulation, prepared forecasting,
spatial/mass reductions, independent populations/selection policy, portable
results, and current optional nonlinear validation. Study/reproduction trees,
fixed cohorts/assets, release/fleet tools and historical replay are deleted.
The submitted implementation remains in Git history.

Public operations are `prepare_forecast`, `forecast`, `simulate`,
`summarize_forecast`, `mass_reach`, `adaptive_mass_reach`, result I/O and optional
`validate_nonlinear`. A prepared context has copy-on-read scientific settings,
truth/model kernel identities, a smooth expected observation and reusable
projection/backend state. Injection and null controls are explicit.

External detector kernels and the optical provider use the same forecasting
path. Model calibration files are pinned as well as truth files. Nonlinear
validation shares the prepared likelihood mask with necessary PSF-border
intersection; custom masks are explicit. Default mismatch detection uses its
actual statistic and positive fitted amplitude.

Independent physics/covariance oracles, actual package-import tests, real
reference/JAX mass-bank parity, sampling/model checks and controlled defects
replace study fixtures and duplicate wrapper layers. Controlled keeper faults
were caught in isolated XTX copies. Three
baseline-red controls established scheduler input loss, lost serialized settings
and smooth-scene trial construction. Additional candidate controls caught stale
model-kernel replay and preparation side effects. Exact receipts and their
source snapshots are authoritative in [validation](validation.md).

All runtime tests, GPU canaries, controlled faults, wheel checks, CLI operations
and examples ran on XTX. No production campaign or external publication was
performed. Scientific assumptions and supported-model limits are documented in
[SCIENCE](../SCIENCE.md); the API/workflows are in [ENGINE_GUIDE](../ENGINE_GUIDE.md).

Physical Python line counts include tracked source, tests and tools; generated
and ignored artifacts are excluded.

| Tree | Package source | Tests | All Python |
| --- | ---: | ---: | ---: |
| `main` (`ff49264`) | 12,889 | 5,717 | 18,685 |
| Submitted `rasti` (`41621de`) | 42,157 | 37,778 | 97,804 |
| Platform candidate | 29,108 | 22,921 | 53,865 |

The candidate removes 43,939 Python lines overall from the submitted tree and
retains the larger scientific feature set relative to `main`. The study and
reproduction trees, dead benchmark configuration and obsolete scratch launcher
are absent. Three reproduced correctness repairs are isolated in commit
`2f82967`; the platform/API migration follows separately.

The source/test retirement and keeper ledgers in this directory document the
preservation review. This is a reusable engine candidate for RASTI review; it
is not a claim that new instruments/populations are scientifically validated
or that a blind-search significance calibration is available.
