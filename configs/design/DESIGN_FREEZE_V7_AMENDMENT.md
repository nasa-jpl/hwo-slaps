# Design Freeze v7 nonlinear release amendment

**Status:** prepared, not ratified, not launched. The complete machine-readable
declaration is `design_freeze_v7.yaml`.

This additive amendment consumes the ratified v5 design at
`configs/design/design_freeze_v1.yaml`, SHA-256
`7c6302ef4217573c0e2def0094844941922a04f050bd4fccfc7889b1b4341f75`.
Detector, observing, source, lens, PSF, noise, mass-mapping and original
ladder science remain governed by that v5 artifact. The v7 amendment declares
fresh nonlinear search routing and retention policy.

The release catalog contains 1,601 unique case IDs: 1,453 archived case
specifications, 112 standard cases for the 28 top-50 systems without
historical nonlinear coverage, and 36 selected-12 H0 bracket declarations.
The standard count is 1,565. Views are selected12 standard (48), top50
standard (200 = 88 archived + 112 new), null590 (59 primary controls + 531
replicates), and PSF288 (12 systems × 8 delta arms × 3 directions). View
overlap is intentional; execution case IDs are unique and duplicate archive
signatures are rejected.

The C inventory supplies all 1,453 archived input records. Every config,
position, aggregate case artifact and source asset binding is hash checked
before catalog write. The v6 manifest supplies the 28 new ladder bindings;
the actual ladder NPZ and sibling config are required as generator inputs and
are checked for SHA-256, run name, config hash, campaign UUID, source asset and
recorded kernel shape.

Three new systems (`sys0013`, `sys0082`, `sys0987`) reuse completed,
preparation-only position artifacts with recorded hashes. The other 25 systems
have explicit position-producer tasks. A producer must complete and hash
`injection_position.json` before its four dependent standard case specs may be
materialized. No future position or artifact hash is inserted into the
catalog.

The 36 bracket cases use each selected system's existing upper-rung position
and declare +0.1, +0.2 and +0.3 dex in log10(M200). Physical subhalo
parameters are recomputed at generation time. Each bracket requires a fresh
H1 physical-truth zero-residual anchor before an H0-only profile. H1 Nautilus
and evidence claims are forbidden for these anchors.

Every ready case routes through the optimizer-owned
`scripts/run_nonlinear_production.py` case-spec interface. Fresh searching is
mandatory; archived fits, posterior samples, evidence, optimizer state and
checkpoints are identity references only. The sampler contract is n_eff=500,
n_live smooth/search/fixed = 100/200/100, n_shell=1,
discard_exploration=false, f_live=0.01 and raw sampler-internal retention true. The local
profile contract is eight separated current-search starts plus the ML
incumbent, normalized separation 0.05, bounded L-BFGS-B settings, a tighter
repeat, 0.1 support/repeat tolerances and no fallback solver.

The case route is locked behind an immutable approval sidecar naming the exact
v7 freeze SHA-256, generated catalog SHA-256, authorized scope and GPU limit.
The preparation catalog cannot be launched by changing a boolean. The CPU restamp path changes `stage0.code_revision` and verifies the remaining scientific configuration. New v6 cases additionally use the explicit 999-to-51 nonlinear fit-kernel conversion, preserving every other field. It emits reviewable pending-approval specs from a clean source revision; actual execution requires the later approval sidecar.

This amendment is an execution declaration. It does not claim a completed
campaign, area or census result, population inference, continuum convergence,
or any universal correction to historical products.
