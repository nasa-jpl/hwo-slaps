# Nonlinear engine cleanup

The nonlinear package now separates profile optimization, calibration diagnostics,
and the submitted study's release protocol. Likelihoods, prior construction,
optimizer defaults, acceptance gates, and custom acceleration remain unchanged.

## Module boundaries

- `profile_settings.py` owns the immutable `FreshProfileSettings` schema. It can
  be imported without AutoLens, AutoFit, AutoArray, JAX, or plotting libraries.
- `fresh_profile.py` owns current-search start selection, normalized coordinates,
  best-finite retention, local optimization, and paired-fit orchestration.
- `profile_calibration.py` owns likelihood-matched tangents and verified
  zero-residual anchors. These diagnostics do not change sampler or profile state.
- `studies/rasti/campaign/profile_adapters.py` owns release-declaration parsing,
  the fixed 999-pixel Fisher adapter, and stage-3 bracket materialization.
- The bounded stage-3 supervisor moved to
  `studies/rasti/campaign/profile_execution.py`. Its memory registry and execution
  policy describe the RASTI/B200 campaign rather than a portable engine contract.

All existing nonlinear package exports resolve through an explicit lazy module
map. Existing imports of `FreshProfileSettings`, `ZeroResidualAnchorRunner`, and
`likelihood_matched_tangent` from `fresh_profile` continue to work.

## Deliberate migration

Release parsing changes from `FreshProfileSettings.from_release_protocol(release)`
to `studies.rasti.campaign.profile_adapters.profile_settings_from_release_protocol(release)`.
The study-specific `evaluate_established_fisher_q`, `materialize_bracket_case`, and
`materialize_bracket_case_from_files` functions moved into that same adapter.
New studies should configure profile settings directly rather than imitate a
RASTI release declaration or its fixed PSF geometry.

`NonlinearSearchSettings.nautilus_training_workers` now makes emulator-training
parallelism explicit. A positive integer overrides
`HWOSLAPS_NAUTILUS_TRAINING_WORKERS`; `None` preserves the previous environment
resolution and serial default. The sampler's process pool and random streams are
unchanged. Emulator training still runs serially when sampler cores exceed one.
Requested and effective counts continue to appear in each fit summary.

The existing spawn pool, scoped Nautilus patch, compatibility patch, and optional
persistent-preparation cache remain available. Their lifecycle should eventually
be expressed through backend objects; this pass does not change their behavior.

## Validation evidence

AST comparison confirmed unchanged bodies for 20 optimization, start-selection,
runner, validator, and tangent units after extraction. New subprocess tests verify
that the package and profile settings do not import execution backends. New pool
contracts cover invalid worker counts, explicit precedence, environment changes,
and forwarding the configured count into the actual fit scope.

The initial local focused run passed 74 tests; two bracket-rendering tests failed
because the local AutoArray installation lacks `autoarray.decorators`. This is a
runtime mismatch, so the pinned XTX validation receipts are the acceptance source
for real rendering and GPU checks. The final focused local run passed 75 tests
with those two rendering cases and the GPU parity case excluded. New module,
adapter, and test lint checks passed.

## Correctness concern retained for separate work

`FreshProfileSettings.to_dict()` omits
`start_separation_posterior_sigma`. An explicit value of 2.0 reconstructs as the
1.0 default through `from_mapping(to_dict())`. This can change start selection
when settings are reused from serialized metadata. The omission predates this
cleanup; no correction was included because it could change scientific execution.
A separate change should define an authoritative round-trip schema and add a
nondefault-value regression contract.
