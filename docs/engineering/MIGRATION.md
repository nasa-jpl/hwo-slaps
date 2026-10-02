# Migration from the RASTI checkpoint

Base: `rasti-26-183-submitted` (`41621de`). This cleanup permits breaking
interfaces while preserving scientific calculations and archived inputs.

## Core entry points

Install the engine from the checkout to obtain the `hwoslaps` console command.
`runner.py` is a thin source-checkout wrapper. Repeated `-c` arguments compose
configuration fragments. `--validate-only` checks the final configuration
without creating output files or loading scientific backends.

Relative source-asset, covariance, and output paths now belong to the YAML
file declaring them. Python mappings use the current directory or `base_dir`.
For historical repository-relative YAML, use `--base-dir` with the repository
root. Configuration snapshots, provenance hashes, and pipeline inputs now use
the same resolved mapping. The defaults of physical and accelerated routes
remain unchanged. Existing artifact run directories now fail rather than
overwrite a snapshot while leaving previous science products beside it.

`hwoslaps.run_pipeline` is the programmatic engine entry; use
`hwoslaps.run_with_artifacts` for captured outputs and provenance. The existing
`run_enhanced_pipeline` name remains a deliberate convenience export.
Individual scene/PSF/observation functions accept explicit component inputs
instead of requiring a whole pipeline mapping.

## Study namespace

Frozen RASTI population, design, ladder-generation, release, and supervision
contracts have moved from `hwoslaps.campaign` and nonlinear execution adapters
to `studies.rasti`. The generic executor and pure adaptive-ladder routines
remain in the installed package. Reproduction commands and import mappings
are listed in `studies/rasti/README.md`.

The study namespace is excluded from installed wheels. Run reproduction tools
from a source checkout. The original design files, source assets, numerical
settings, and `reproducibility/rasti-26-183` records are preserved. The submitted
tag retains the original layout for exact historical reproduction.

## Numerical and runtime concerns

Numerical/physics corrections are outside this refactor. The engineering
reports record identified concerns and checks; a separate correction should
have its own tests and scientific assessment. In particular, a confirmed
existing scheduling issue in the Fisher supervised iterator is represented by
an expected-failure regression rather than silently changed in this pass.

The new generic population API describes independent distributions and has its
own reproducibility contract. It does not replace or reproduce the frozen
RASTI sampler. Unsupported telescope/source implementations remain explicit
extension work rather than unvalidated configuration options.

## Checkpoint verification

The immutable `verify_checkpoint.py` checks the historical fitting-engine file
hash. It passes in the original RASTI checkout; it intentionally refuses the
refactored fitting-engine layout. Run that checker from the submitted tag for
historical verification. The archive itself and original configs remain
byte-identical; the cleanup has separate parity and regression validation.
