# RASTI reproduction

This source-only study preserves the fixed design, cohort selection, release
contracts, execution gates, and paper-specific reporting used for the submitted
RASTI study. These modules are excluded from the installed `hwoslaps` package.
The reusable engine retains immutable campaign execution, adaptive mass-ladder
primitives, instrument modelling, lensing, and inference.

## Inputs and historical reference

The submitted checkpoint is `41621de`. The immutable release bundle remains at
`reproducibility/rasti-26-183/`; original design declarations, observation
references, and source inputs remain under `configs/`. This refactor does not
rewrite those scientific inputs or change their digests. Historical release
declarations may name `scripts/...`; active executable routes translate those
paths to `studies/rasti/scripts/...`.

The study package intentionally retains the original fixed population and
provenance checks. It is a reproduction adapter, not the specification for new
studies. Use the engine's configurable APIs for new instruments and populations.

## Commands

Run these commands from the repository checkout with its scientific environment.
Every executable bootstraps the checkout's source paths; no separate study
installation is required.

```sh
python studies/rasti/scripts/generate_stage0_campaign.py --help
python studies/rasti/scripts/generate_ladder_campaign.py --help
python studies/rasti/scripts/generate_nonlinear_release_catalog.py --help
python studies/rasti/scripts/prepare_nonlinear_execution.py --help
python studies/rasti/scripts/harvest_nonlinear_production.py --help
```

Preparation remains separate from launch. Approval receipts, bound artifact
digests, source revision checks, fail-closed GPU admission, and immutable harvest
identities retain their original requirements. Commands do not bypass those
requirements after relocation.

## Import migration

`hwoslaps.campaign.design_freeze`, `stage0`, `ladder`, `release_catalog`,
`release_dependencies`, `production_harvest`, and `execution_prepare` now live
under `studies.rasti.campaign`. The fixed Stage 3 controller formerly named
`hwoslaps.modeling.nonlinear.profile_execution` also lives there. Study Fisher
calibration and release settings adapters live in `profile_adapters`.

`hwoslaps.campaign.s1_lite`, `_common`, `system_ids`, `ladder_walk`, and the shared
`artifacts.file_sha256` remain reusable engine primitives. Study helpers must
import the engine; engine modules must not import this study.
