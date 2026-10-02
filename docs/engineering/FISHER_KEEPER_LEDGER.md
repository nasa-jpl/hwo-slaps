# Fisher test ownership and engine cutover

Audit baseline: `bf73e2c`. The root validation owner pinned all 95 baseline
files and 2,123 unique cases on XTX before retirement. This lane ran no local
tests, control probes, or benchmarks. OpenClaw's supplied test-audit skill and
campaign rules were applied to the contracts below; line counts are physical
lines, not complexity claims.

## Production ownership

The old installed Fisher surface had 6,400 lines across eight files; its
Fisher/routing test surface had 5,873 lines. The statistical algebra and custom
JAX kernels are retained. RASTI's only non-test consumer of the old
`prepare_ladder_grid_selection` / `compute_ladder_summary` APIs was
`studies/rasti/scripts/run_ladder.py`. Those APIs, their identity registry, and
their ladder result classes are retired. General geometry remains, with the
selection class named `SpatialSelection`.

The production boundary is now `FisherDetector.evaluate_positions` /
`evaluate_masses` and `ForecastResult`: compact mass-by-position arrays with
matched, mismatch, and spurious-amplitude diagnostics. Explicit complete-domain
positions keep sparse evaluation from shrinking interpolation bounds. A mass
change reuses compiled scene/convolution/projection products. The deterministic
mean renderer delegates to `observation.forward`, which owns detector units.
Supplied detector truth and fit kernels do not regenerate an optical pupil;
optical delta mode retains its existing identity gate.

## Declaration decisions

`R` retains an independent contract; `F` repairs its oracle; `C` moves the
contract to its canonical owner; `D` retires an obsolete or non-behavioral test.

| Former declaration or layer | Decision and canonical keeper | Evidence and risk |
|---|---|---|
| `test_sparse_nonlinear_calibration_qfit_definition_matches_scdd_threshold` | D; nonlinear likelihood-metric owner remains | Test only computes literal arithmetic and never calls production; introduced by `eb7b5b5`. No production regression is detectable. |
| `test_master_config_uses_scdd_baseline_redshifts` | D; redshift/scene validation remains | Fixed example values 0.2/0.6 are study defaults, not a general engine contract; introduced by `eb7b5b5`. |
| `test_no_nuisance_directions_make_profiled_information_equal_raw` | D; core no-nuisance oracle and real detector runtime remain | Manually supplies `nuisance_images=None`, duplicating algebra without exercising detector dispatch; introduced by `a076195`. |
| Nuisance selector declarations in `test_fisher_nuisance_subset_and_mask` | C to `test_fisher_nuisance` | All built-in/default/explicit/invalid-selector cases call the pure planner/selector directly. Real runtime none/all/lens-only cases remain as wiring proofs. Removes the `__new__` and private-wrapper selector harness, while retaining independent mask coverage. |
| `test_grid_runtime_provenance_separates_requested_and_effective_workers` | C to real `test_grid_map_worker_env_override_matches_serial` result assertion | Pure runtime policy is already independently owned; real result must still record effective worker count from the environment. Introduced by `f6f0aea`. |
| `test_dense_covariance_matches_explicit_whitening` | F, same owner/name | Former expected path reused the same Whitener/workspace. It now computes inverse-covariance Schur algebra independently using `numpy.linalg.solve`; introduced by `d0218bf`. |
| `test_compute_asimov_from_images_matches_manual_vector_call` | F, same owner/name | Former oracle reused adapter flatten/stack and core; now literal selected vectors and nonuniform noise independently determine the projection. Introduced by `d0218bf`. |
| Old sparse ladder selection tests | C to general geometry tests | Closed boundaries, row-major selection, complete-domain coordinates, uniqueness and immutability remain at the geometry owner. Detector identity-registration restrictions are intentionally retired. |
| Old `_rung_metrics` literal/compact-output tests | C to `test_forecast_results` | Explicit selected positions, supplied quadrature areas, separate boundary diagnostics, signed amplitudes and consumed nonfinite failure have independently hand-computed oracles. Ladder-only output types and policy names are retired. |
| Old sparse/dense numerical and combined A/B tests | C to `test_forecast_evaluation` and retained `test_fisher_radial_lookup` | Real reference/JAX sparse mass evaluation, full/tail batches 7/8, engine reuse/rebuild parity and full-domain radial bounds remain executable. No study-runner or frozen-tier fixture is retained. |
| Old runner artifact/observer/default tier gate tests | D | Production study runner is deliberately removed; general forecast NPZ round trips live in `test_forecast_artifacts`. |
| Fixed adaptive ladder walk/policy tests | C/D to `test_mass_reach` | General crossings, zero-log-bracket failure, finite unique axes, range censoring, arbitrary targets and bounded no-repeat refinement remain. Fixed M10/M50/two-zero-rung/coarse/refine/descent policy is intentionally retired. |

The new result/reduction tests protect observable estimands, rather than replay
production formatting. Area is absent without declared cell areas, boundary
containment is absent without a boundary, and undefined consumed diagnostics
raise rather than fabricating zero area. Nonmonotone mass curves do not assert
a unique crossing; unbracketed curves return explicit bounds without
extrapolation.

## Scheduler defect, kept separate from science changes

The baseline owner regression was run on the untouched snapshot with
`--runxfail`: it returned only four of ten inputs when the entire pending batch
completed. Root receipts are `control_scheduler_baseline.log` and XML. The
owner now tracks input exhaustion independently of pending futures, and the
same regression is no longer marked xfail. This patch changes task completeness
only and should be committed separately from the engine API cutover.

## Required focused XTX proof

Run the covariance and image-adapter owners, nuisance planner plus real
detector runtime, real serial/parallel map provenance, forecast result and
artifact owners, mass reach, and reference/JAX evaluation. Keep radial lookup,
source guard, retarget, mismatch, actual spawned-worker failure/order and
PSF-fit/scan parity suites. XTX is the only test/probe/benchmark host.

The validation owner records actual receipts and final declaration
reconciliation; this document does not claim unexecuted candidate tests pass.

## Final public API cutover

`generator_fisher.perform_fisher_detection` and its package export are removed
with the legacy mode-dispatch Pipeline. Its sole production consumer was that
Pipeline; the actual prepared forecast owns public dispatch, and real position
evaluation is covered by `test_forecast_evaluation`.

The following six declarations in `test_fisher_grid_map` are retired with the
implicit adjacent-file/campaign-binding output policy:

- `test_pipeline_populates_grid_map_provenance`
- `test_pipeline_omits_config_hash_without_snapshot`
- `test_pipeline_snapshot_hash_rejects_bool_int_alias`
- `test_pipeline_snapshot_hash_accepts_yaml_sequence_roundtrip`
- `test_pipeline_rejects_foreign_grid_map_snapshot`
- `test_pipeline_binds_exact_resolved_snapshot`

All stages in these tests were replaced by one stub object and a synthetic
result. Metadata preservation is now an explicit `ForecastResult` NPZ
contract. Before removing the cases, the two independent typed-hash and
tuple/list YAML-canonicalization assertions were transferred to the existing
`test_provenance::test_config_hash_is_stable_and_key_order_insensitive` owner;
the production key-sorted YAML hash convention remains unchanged.

`test_generator_dispatches_grid_map` moves to the real prepared forecast and
position-evaluation boundary. `_stub_pipeline_grid_result` is removed. The
delta identity assertions formerly attached to
`test_fisher_detection_transports_delta_provenance` now live in
`test_fisher_forecast_npz_preserves_delta_provenance`: a real detector evaluates
positions, saves and reloads the new artifact, and checks the same scientific
wire payload, truth/model kernel identities and revision information. Only the
obsolete console-summary formatting assertion is retired.

The new provenance also identifies actual scalar nuisance names/priors and
loaded image-source identity. Inactive optical PSF basis descriptions are not
built for a detector kernel; the genuine external-kernel integration test
exposed this branch on XTX and its production owner was repaired. Active
optical derivative/scan behavior and delta identity checks remain.

## Exact retired-file declaration reconciliation

The following are all 17 declarations in retired
`tests/test_fisher_aperture_perimeter.py`. `geometry` means
`tests/test_fisher_geometry.py`, `results` means `tests/test_forecast_results.py`,
and `evaluation` means `tests/test_forecast_evaluation.py`.

| Exact old declaration | Disposition and stronger remaining owner |
|---|---|
| `test_selection_preserves_off_centre_square_and_exact_closed_union` | C: geometry `test_off_centre_aperture_uses_its_own_coordinates_on_translated_lattice` plus `test_translated_lattice_keeps_coordinates_and_row_major_positions`. |
| `test_selection_includes_exact_radius_and_edge_overlap_without_duplicates` | C: geometry `test_compact_selection_contains_aperture_and_original_perimeter_once` and exact-boundary case. |
| `test_selection_honours_nextafter_closed_radius_boundary` | C: geometry `test_aperture_boundary_uses_squared_distance_without_tolerance`; the old test never exercised exact equality. |
| `test_selection_rejects_zero_inside_geometry` | C: geometry invalid-radius table and `test_aperture_without_any_lattice_node_is_rejected`. |
| `test_selection_rejects_annulus_because_ladder_requires_full_square` | C/D: geometry rejects a genuinely restricted layout in `test_annulus_uses_closed_boundaries_and_retains_position_indices`; the declaration-only ladder policy is retired. |
| `test_rung_metrics_keeps_perimeter_only_detection_and_skipped_maximum` | C: results `test_spatial_summary_uses_explicit_selection_weights_and_separate_boundary` hand oracle. |
| `test_rung_metrics_no_detections_and_all_inside_are_hand_checkable` | C: results `test_uniform_detection_limits_have_literal_area_and_fraction`. |
| `test_rung_metrics_consumed_nonfinite_fails_but_skipped_nonfinite_is_irrelevant` | C: results `test_nonfinite_consumed_statistics_are_reported_without_fabricating_zero_area`. |
| `test_rung_data_contract_exposes_compact_consumed_arrays` | C/D: result shape/required-statistic guard and real forecast-artifact axis round trip retain compact-output contracts; the obsolete rung data type is retired. |
| `test_specialized_summary_matches_dense_reduction_on_consumed_nodes` | C: independent spatial-result hand oracle plus evaluation `test_sparse_mass_bank_matches_dense_reference_and_reuses_accelerated_products`. |
| `test_specialized_summary_rejects_consumed_nonfinite_and_unsupported_mismatch` | C/D: consumed-nonfinite result guard remains; the old restriction forbidding mismatch evaluation is intentionally removed. |
| `test_specialized_summary_rejects_selection_with_dropped_perimeter_node` | D: detector-owned identity-registered complete-perimeter selection is deliberately retired. Boundary diagnostics now describe an explicitly supplied spatial sample. |
| `test_prepared_selection_arrays_are_immutable_and_annulus_is_rejected` | C: geometry `test_compact_selection_contains_aperture_and_original_perimeter_once` and restricted-layout guard. |
| `test_specialized_summary_rejects_nonfinite_edge_or_aperture_value` | C: results consumed-nonfinite guard covers both selected and boundary positions. |
| `test_specialized_summary_handles_no_detection_and_all_inside` | C: results uniform-zero/full detection table. |
| `test_specialized_summary_handles_aperture_containing_every_lattice_node` | C: result full-position uniform-detection area/fraction table; no aperture-only special execution path remains. |
| `test_jax_dense_and_ladder_match_each_consumed_q_and_radial_table` | C: evaluation real sparse/dense reference and JAX retarget/rebuild parity with an explicit complete domain; retained radial-lookup and grid-engine bounds guards. |

All three declarations in retired `tests/test_fisher_ab_integration.py`:

| Exact old declaration | Disposition and stronger remaining owner |
|---|---|
| `test_compact_affine_matches_dense_general_interpolation` | C: evaluation full/tail batch sizes 7/8 against real reference rendering; retained `test_fisher_radial_lookup` owns affine versus general interpolation at every knot, midpoint and actual NFW table. |
| `test_runner_writes_readable_artifacts_and_calls_observer` | D/C: obsolete study tier/observer loop is retired; general artifact readability, atomic publication and numerical array preservation live in `test_forecast_artifacts`. |
| `test_default_main_rejects_full_pool_before_initializing` | D: the frozen study-tier command and its admission rule are deliberately removed. |

Default detection and reach now use `q_mismatch` when an actual mismatched
data/model statistic exists. `q_asimov` remains an explicit matched-template
diagnostic override, and `q_spurious` explicitly selects a null control. One
`ForecastResult.detections` owner applies positive signed amplitudes and strict
threshold validation; the spatial reducer consumes that owner. Literal signed
and null-control cases in `test_forecast_results` prove these distinct science
contracts without changing any numerical model.
