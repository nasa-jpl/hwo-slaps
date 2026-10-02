# Nonlinear engine and test audit

Nonlinear fitting is an optional backend of the reusable forecasting engine.
The entry point is `validate_nonlinear(prepared, trial, settings, ...)`, exposed
through the public engine. It consumes the prepared truth/fit PSFs and requires
an explicit output directory because AutoFit writes search artifacts.

## Scientific behavior

With no observation argument, the engine simulates the declared subhalo trial.
For a null control, pass `prepared.observation` explicitly. `dataset_kind` selects
Asimov or noisy data. Pixel support defaults to the actual prepared Fisher mask,
intersected with the fitted PSF safe border; an explicit `mask_bool_use` overrides
the ROI and is identified as `custom_minus_psf_border` in result metadata.
The generated and fitted light-profile and external
blurring grids use the same declared sampling; `consistent_sampling_v2` is the
sole supported builder objective. External AutoLens datasets may have no bound
rendering metadata; datasets built by this package always record and check it.

Freed fits require an explicit `MassMappingContext`, including the physical
mass-prior support. The new public entry point chooses no mass range. Existing
model priors, sampler defaults, x64 guards, emulator-pool ordering, normalized
profile optimization, best-finite retention, and independent-start acceptance
gates retain their numerical implementation. Providing `profile_settings`
selects current-search local refinement after the sampler. Refinement requires
explicit `NonlinearSearchSettings(use_jax=True)`; an incompatible analysis is
rejected before rendering or starting an expensive sampler.

A provided detector PSF uses the real `kernel` mode: its declared digest,
shape, and sampling are checked before fitting. The validator binds the fitted
kernel, metadata, and case label. AutoLens may normalize its kernel; the executor
checks that normalization against the supplied kernel and records the digest
of the kernel actually fitted. Neither optics regeneration nor a study launch
protocol is required. Existing delta/explicit optical mismatch paths retain
their own configuration identity guards.

The tested AutoArray/AutoLens runtime cannot evaluate a nonlinear fit with a
1x1 PSF: its empty blurring grid reaches `OverSampler.sub_is_uniform`, which
indexes the nonexistent first sub-size. A direct dependency-only XTX control
reproduces this with a 1x1 kernel and evaluates a 3x3 kernel successfully. The
public nonlinear entry point rejects a 1x1 **fit** kernel before simulation or
sampling. This does not restrict simulation/Fisher identity kernels, nor a 1x1
truth kernel with a wider mismatched fit kernel. No dependency is monkeypatched,
and the nonlinear likelihood is unchanged. Matched observation kernels retain
an exact identity check: shapes must match and bytes must equal the supplied
kernel or its exact AutoLens sum-normalization. The existing
`fitted_kernel_sha256` contract implements that known transform; no approximate
array comparison admits a different kernel.

## Retired owners and retained tests

Archived-vector replay, historical revision allow-lists and identity schemas,
study-specific single-GPU admission, globally patched persistent preparation
caches, the replay-only least-squares runner, fixed study benchmark anchors,
and their study/protocol tests are removed. They remain in Git history.
`linearized_comparator` moved to `profile_calibration.py`; its independent
nuisance projection, finite bounds and background-effect tests moved with it.

The fresh-profile test module now imports the engine directly. Removing study
adapters therefore cannot silently skip the entire scientific regression suite.
Reusable builder and JAX-family tests use small generic scenes and synthetic
image assets created by tests rather than paper source-bank files.

Low-value copied lazy-export inventory and dataclass identity assertions were
removed in favor of actual fresh-interpreter import and JSON/CSV export owners.
The unused `profile_likelihood_q` wrapper retired with its only tests; paired
fits use the retained signed/clipped likelihood-ratio metric owner.

The per-declaration audit ledger records retention, repaired assertions,
consolidation and retirement against the pinned pre-cutover source. Retired
protocol tests were valid for their old owners; their retirement is not evidence
that those tests were junk.

## Separate correctness repair and metadata break

`FreshProfileSettings.to_dict()` previously omitted
`start_separation_posterior_sigma`. The actual-owner baseline control converts an
explicit 2.0 to the 1.0 default on reload. The serialization repair adds the
missing field; a nondefault round-trip regression protects the public settings
contract. This does not change optimizer defaults or current in-memory settings.
Control/candidate runtime proof is run on XTX, not locally.

The existing physical-trial helper also assumed that an available
`subhalo_mass` attribute was numeric. Real smooth `LensingData` carries that
attribute as `None`. It now recomputes profile scales from the declared mass and
configuration for a smooth reference; matching injected truth still reuses its
scales. A real generated smooth-scene regression covers NFW, SIS and PointMass,
with baseline-red/candidate-green control requested on XTX.

The optimizer provenance identifier is now `normalized_lbfgsb_v2` rather than
a RASTI release name. This intentionally changes metadata; objective arithmetic,
optimizer tolerances and acceptance math are unchanged.

## Validation

The baseline gate records all test files at `bf73e2c` on the pinned XTX runtime.
Focused and full candidate results, CPU/GPU parity, scientific-method preservation
checks, and mutation/control receipts are recorded by the coordinated validation
lane. No production fits are launched by this cleanup. Local work is limited to
file edits and static inspection after the user's execution restriction.
