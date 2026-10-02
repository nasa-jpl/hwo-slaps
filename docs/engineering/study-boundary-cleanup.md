# Study boundary cleanup

The installed package no longer owns the RASTI design-freeze schema, fixed
cohort generators, release-catalog materialization, production harvest counts,
or the fixed Stage 3 admission controller. Those contracts now live in the
source-only `studies/rasti/campaign` package, with 31 study executables under
`studies/rasti/scripts`. Packaging still discovers only `src/`.

This is an intentionally breaking namespace and command-path change. No
compatibility shim remains in the installed engine. The original submission
bundle and scientific config bytes remain intact. The study adapter translates
historical release entrypoint paths when producing executable routes.

The pure adaptive mass-ladder policy and estimands remain in the engine, as do
the immutable S1 executor, system identifiers, shared configuration helpers,
and a streamed SHA-256 utility. Study defaults no longer enter `_common`: the
RASTI freeze path resolver belongs to its design adapter.

## Validation targets

```sh
python -m pytest tests/test_design_freeze.py tests/test_campaign_stage0.py \
  tests/test_campaign_ladder.py tests/test_campaign_s1_lite.py \
  tests/test_release_catalog.py tests/test_release_dependencies.py \
  tests/test_execution_prepare.py tests/test_production_harvest.py \
  tests/test_run_ladder.py tests/test_bulk_launch.py \
  tests/test_production_cli.py tests/test_stage3_cli.py tests/test_profile_replay.py
```

Local collection in the existing `hwo-slaps` environment reached a historical
AutoArray compatibility failure (`autoarray.decorators` is absent); the pinned
XTX environment is the authoritative numerical test runtime. Validation
receipts and any unresolved failures are recorded in `validation.md`.

No correctness corrections or scientific fits were authorized by this cleanup.
Frozen science checks and approval gates remain in the study code.
