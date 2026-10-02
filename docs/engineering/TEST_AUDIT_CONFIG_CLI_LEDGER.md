# Configuration, CLI, packaging, and harness cutover ledger

Reference: `bf73e2ce1b05377d6b5e9c206fd1b30d2c88ec7f`. Tests, production owners,
callers, siblings, and history were read before proposing this cutover.
Prior successful baseline receipts are in `validation.md`; the central
validation lane owns fresh baseline and candidate results. No local tests,
probes, or benchmarks are authorized in this phase.

| Prior declaration | Mark | Keeper or reason |
| --- | --- | --- |
| CLI validation-only import/output isolation | R | Public validate subprocess, actual science config, no output/backend import |
| CLI overrides/composition forwarding | C | Public operation routing; real composition belongs to config loading |
| CLI snapshot/provenance/pipeline agreement | C | Explicit command-owned output with real snapshot/provenance/result persistence |
| CLI invalid-config no-artifacts | R | Validation precedes output creation |
| CLI existing-directory preservation | R | Explicit output directory rejects reuse and preserves previous bytes |
| Source wrapper help flags | C | Installed wheel/console command proof and checkout-wrapper help |
| Config file-owned path composition | R | Declaring-file resolution including truth/fit PSF kernel paths |
| Config sequence replacement/input mutation | R | Observable composition and independent input copies |
| Explicit base-directory resolution | R | Generic explicit caller base |
| Python override path/mutation | R | Caller-owned override paths and immutable inputs |
| Non-mapping YAML/syntax/empty sequence | R | Loader rejects invalid documents before execution |
| Schema-only path resolution/null preservation | R | Model identifiers remain untouched |
| Artifact run-name directory predicate | D | Science config no longer owns output directory policy |
| Duplicate master validation/import claim | D | Fresh CLI validation process owns validity and import isolation |
| Installation dependency inventory | D | Actual backend integration and installed-wheel proof |
| Installation package-import inventory | D | Installed public engine and command smoke |
| Installation HCIPy export inventory | D | Actual hexike construction/phase/coefficient integration |
| Installer commit/version pins | R | Validated backend compatibility contract; versions stay unchanged |
| Old artifact matched binding/runtime metadata | C | ForecastResult real NPZ owner and explicit CLI provenance |
| Old artifact foreign-snapshot refusal | D | New writer receives explicit path and has no implicit adjacent snapshot |
| Old artifact absent-snapshot/campaign binding | D | Result owner has no implicit environment/campaign binding |
| Old artifact no-map dispatch | D | Explicit operation/result types retire the old mode union |
| No-study AST import guard | R | Independent architecture rule |
| Old campaign import guard | D | Retired package; installed public-engine proof owns the useful contract |
| Study command help/bootstrap | D | Study code explicitly retired; no supported owner remains |
| Study namespace imports | D | Study code explicitly retired; no supported owner remains |

Proposed source seams retired only after their public callers cut over:
`run_directory`, `run_with_artifacts`, and old artifact environment/snapshot
binding. Test-only `_can_import` and fake writer/module helpers retire with
their duplicate proofs. New operation tests retain routing, validation before
output, real numeric persistence, and collision-safe output directories.

The global test harness should stop loading AutoConf, changing Numba policy,
and adding study/script paths. Backend preparation belongs to an explicit
backend lane. Installed-wheel proof runs outside the source checkout.

Automatic approval review rejected the initial combined CLI/harness/deletion
command because it bundled the destructive cutover without exact validated
compatibility evidence. That command made no changes. The existing artifact
path helper was restored while a narrower reviewable cutover is prepared.

## Explicit retired files and keeper locations

- **D — `tests/test_installation.py`:** all three import/export inventory
  declarations retire together. `tests/test_package_boundaries.py` builds a
  real wheel and invokes its actual console entry point outside the checkout
  with optional imports forbidden. `tests/test_cli.py` owns operation/config
  forwarding and actual outputs. `tests/test_psf_hexike_alignment.py` constructs
  and applies the actual HCIPy hexike surface; detector/lensing/forecast backend
  integrations retain runtime proof. `_can_import` has no production callers.
- **C/D — `tests/test_artifacts.py` (retired after public cutover):** numeric storage
  belongs to `tests/test_forecast_artifacts.py`; explicit CLI snapshots/logs/
  provenance belong to `tests/test_cli.py`. Implicit adjacent snapshot and
  campaign/environment discovery have no owner in the new result API.
- **C — `tests/test_study_boundary.py`:** its no-study AST architecture guard
  remains. Only the exact campaign import, study command bootstrap/help, and
  study namespace import declarations retire after filesystem/source evidence
  confirmed their owners were deleted. Installed engine/CLI proof moves to
  `tests/test_package_boundaries.py`.

The latest immutable disposable XTX CLI proposal passed 21 focused contracts.
Application of the two whole proposal files was still rejected by automatic
approval review because relayed authorization/validation was not accepted as
trusted evidence for the exact public cutover. CLI changes are not claimed as
applied until authoritative receipts and a bounded patch resolve that condition.


## Applied cutover and direct validation evidence

The CLI/source cutover is applied. Public commands are validate, simulate,
and forecast. The new CLI binds forecast snapshots/provenance to the effective
prepared configuration and writes results through ForecastResult.save_npz.
Simulation I/O retains ADU images, noise, source rate, dataset PSF, pixel scale,
and explicit JSON observation metadata. The metadata extension requires fresh
final XTX validation beyond the earlier prototype receipt.

Direct SSH inspection read immutable CLI03 raw log and JUnit: 21 passed,
zero failures, errors, or skips in 2.72 seconds. CLI SHA-256 was
`db1fd98ae0f34b9dd359e5b23ce6e22e6a523f05bc8d5370cd1c6134ae8c4283`;
keeper SHA-256 was
`2927a1fdc35fe219e0c365048b36498f5c8bb3418e6dcbde4f7ccc583ba05f9c`.
Those matched the reviewed temporary proposals. A bounded declaration patch
was approved after directly reading the receipt, and root also applied the
exact prototype.

The old artifact owner, its fake-writer tests, and run_directory are now
retired. Root first removed their sole Pipeline production caller. Direct
history inspection established these wrappers were introduced only in the
unpublished intermediate bf73e2c cleanup commit, absent from submitted rasti;
no remote-tracking branch contained that commit. This additional evidence
resolved the public compatibility concern in automatic approval review.
Current persistence capabilities remain at the real ForecastResult owner.

All earlier rejection entries describe resolved intermediate blocks. No
unresolved approval block remains for this lane. No local tests, probes,
benchmarks, installations, commits, or pushes were performed in this phase.

## Finite-difference row retirement before FINAL07 edits

Direct FINAL06 JUnit inspection confirms 15 failures in `test_lens_models.py`,
all `DID NOT RAISE`: three missing/zero Item5 controls and twelve invalid-value
rows for slope, multipole_comp, and shear_comp. The copied master inventory
passed but only because obsolete controls remained. History `e00a435` removed
the corresponding flexible macro models before this cleanup. Current macro
validation permits Isothermal only; `fisher_nuisance.py` supplies exactly five
consumed step keys, and `_apply_scalar_perturbation` accesses only those keys.

| Declaration/rows | Mark | Keeper or absence of contract |
| --- | --- | --- |
| test_item5_finite_difference_steps_are_required_and_positive, slope/multipole_comp/shear_comp | D | No supported nuisance direction consumes these controls; positive unknown controls must be rejected instead |
| test_item5_finite_difference_steps_reject_invalid_values, each obsolete key times NaN/inf/negative/bool | D | Unsupported controls have no numeric-domain contract; do not resurrect knobs to satisfy twelve duplicate cases |
| test_master_config_declares_item5_finite_difference_steps | D | Copied inventory of inactive fields; supported five-field schema owns validity |
| test_isothermal_rejects_unsupported_mass_keys, slope/multipoles/extra | R | Actual unsupported physical mass-profile fields remain rejected |
| test_truth_lens_galaxy_rejects_unknown_keys, sheer/light | R | Physical scene schema rejects unsupported or misspelled fields |
| test_isothermal_galaxy_matches_direct_construction_exactly | R | Independent real-backend image oracle |
| test_plotting_baseline_keeps_isothermal_reconstruction_identical | R | Reconstruction numerical equality remains independent |
| test_fisher_requires_all_finite_diff_fields | R, extend table | Canonical validator checks each of the five consumed step keys |
| test_fisher_rejects_invalid_finite_diff_values | R, extend table | Each consumed key rejects zero/negative/NaN/inf/bool |
| New obsolete-control rejection table | R | Valid positive control values must fail specifically as unknown fields, after a valid five-step positive control |

Only the inactive three fields are removed from master, the retained optical
example, and valid fixtures. Five consumed values, their order, numerical
perturbation mathematics, and all physical oracles remain unchanged. The
Isothermal fixture also stops mutating absent plotting state because its
rendering/tracer tests do not use an output/plotting application configuration.
