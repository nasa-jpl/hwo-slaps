# Forecast engine cleanup result

Branch: `refactor/forecast-engine-20261002`, based on submitted RASTI checkpoint
`41621de`. Work was performed in an isolated managed checkout. The original
RASTI checkout, manuscript repository, and local uncommitted documentation were
preserved.

## Delivered

The installed engine now separates reusable simulation/inference primitives from
frozen RASTI designs, cohort rules, release adapters, and campaign supervision.
Eight campaign/controller modules, the rank-stability study harness, and 31
study scripts moved into source-only `studies/rasti`. Its 22,774 Python lines
are preserved; the wheel excludes this layer and core imports never point to it.

The package's Python source decreased from 42,157 to 32,498 lines; root scripts
from 16,645 to 4,277. These are boundary/extraction counts, not measures of code
quality or evidence that every remaining component is clean.

- Portable YAML composition, explicit file-local paths, installed CLI, lazy
  public imports, and a programmatic pipeline API.
- Shared artifact/provenance contracts; existing run directories fail rather
  than silently overwrite earlier products; grid export can be disabled.
- Generic independent population recipes, stable per-member streams, collision-
  free integer noise seeds, and fail-closed numeric support checks.
- Standalone scene, PSF, and observation APIs; empirical detector PSF kernels
  with explicit sampling; shared detector expectations and caller-owned RNGs.
- Configurable selection floors and existing arbitrary image-source assets.
- Fisher geometry, nuisance planning, and runtime supervision extracted from
  rendering; detector reduced from 3,177 to 2,680 lines.
- Nonlinear profile settings/calibration separated from execution; fresh-profile
  module reduced from 1,706 to 986 lines; training-worker settings explicit.

## Evidence and preservation

Final XTX full CPU validation: 2,107 passed, 15 skipped, one strict expected
failure for an inherited scheduling defect. The submitted baseline has 1,986
passing tests and 15 skips. GPU parity and worker-routing checks: 17 passed on
both baseline and candidate. Any additional final worker-routing checks are
recorded in [validation](validation.md), the authoritative receipt summary.
The built wheel passed content/entry-point checks and fresh imports from outside
the checkout with optional scientific dependencies actively blocked.

All 36 original configuration and submission-reproducibility files remain
byte-identical. The accelerated Fisher JAX implementation, custom nonlinear
experimental device/persistent kernels, mass-model calculations, and image-source
implementation remain byte-identical. Extracted scientific routines passed
AST/output parity checks. The historical checkpoint checker passes in the
original RASTI checkout for all 1,179 recorded comparisons.

## Remaining work

This is a substantial first refactor, not a declaration that the entire package
is production-ready for arbitrary telescopes. Existing physical model families
are preserved. Full external-PSF integration into optical nuisance forecasting,
additional pupil/source/macro-model families, chromatic rendering, and correlated
population recipes require their own implementations and scientific validation.

The supervised Fisher iterator defect and profile-settings serialization
omission are recorded in [correctness debt](CORRECTNESS_DEBT.md) for separate
corrections. No published-result impact was inferred and no production campaign
or fit was rerun. Full campaign throughput was not re-benchmarked.

Start with [the engine guide](../ENGINE_GUIDE.md),
[the migration guide](MIGRATION.md), and [RASTI reproduction](../../studies/rasti/README.md).
