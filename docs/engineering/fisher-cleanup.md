# Fisher engine cleanup

The detector now delegates geometry planning, scalar nuisance planning, and
worker policy to modules that can be imported without AutoLens or HCIPy.
`fisher_detector.py` falls from 3,177 to 2,680 lines. The extraction preserves
rendering, statistical reductions, JAX kernels, numerical operation order,
result persistence, and the existing detector entry points.

## Reusable interfaces

- `hwoslaps.modeling.fisher_geometry.build_grid_layout(grid_config, centre_yx)`
  constructs a square lattice or closed annulus from explicit sky coordinates.
  It has no dependency on a scene, telescope, renderer, or execution backend.
- `select_aperture_and_perimeter(layout, centre_arcsec, radius_arcsec)` produces
  the compact aperture/perimeter selection. It preserves the full square's
  coordinates, exact closed-boundary convention, row-major position order,
  and immutable selection arrays. The detector retains selection ownership
  checks and caches; the geometry module performs no evaluation.
- `hwoslaps.modeling.fisher_nuisance.build_scalar_nuisance_specs` accepts the
  source-light schema, prior mapping, and background flag explicitly.
  `select_scalar_nuisances` can select any supplied scalar specification list
  in canonical order. Custom direction names therefore do not require
  constructing a detector to plan a subset.
- `hwoslaps.modeling.fisher_runtime.grid_num_workers` and
  `grid_runtime_provenance` accept a runtime override explicitly, keeping
  worker policy separate from hashed numerical configuration.

For example, geometry can be planned before constructing any forward model:

```python
from hwoslaps.modeling.fisher_geometry import (
    build_grid_layout,
    select_aperture_and_perimeter,
)

layout = build_grid_layout(
    {"spacing_arcsec": 0.1, "half_width_arcsec": 1.0},
    centre_yx=(0.25, -0.25),
)
selection = select_aperture_and_perimeter(
    layout, centre_arcsec=(0.25, -0.25), radius_arcsec=0.5
)
```

`FisherLadderGridSelection` and `FisherLadderRungData` remain available from
`fisher_detector` as deliberate exports. Their canonical implementation now
lives in `fisher_geometry`. Existing detector configuration and NPZ output
schemas are unchanged. The old D-F7 study label was removed from a geometry
error message. Existing stub-based tests now discover the pure modules through
the modeling package path; they retain their original assertions.

## Validation

Against original commit `41621de8ca861e425e1caddb7a599d5c1032751f`:

- An isolated comparison executes the original methods and the extracted
  interfaces: all 30 grid/aperture cases and 42 nuisance cases agree exactly.
  The cases cover translated lattices, multiple spacings, annuli, apertures,
  parametric and image sources, background flags, prior mappings, and every
  supported selector form used in the comparison.
- The other 65 detector methods have identical abstract syntax trees.
- `fisher_core.py`, `fisher_adapter.py`, `fisher_grid_jax.py`,
  `generator_fisher.py`, and `utils_fisher.py` remain byte-identical.
- The local CPU suite passes 82 tests with one strict expected failure in
  13.04 seconds, using the existing `hwo-slaps` Python 3.11 environment.
  It comprises `test_fisher_geometry.py`, `test_fisher_nuisance.py`,
  `test_fisher_runtime.py`, `test_fisher_runtime_guards.py`,
  `test_fisher_nuisance_subset_and_mask.py`, and
  `test_fisher_detector_psf_semantics.py`.
- Flake8 passes for the three new modules and their three new test files.
  Existing scientific and integrated runtime tests are also assigned to the
  central XTX validation pass; that pass owns its CPU/GPU receipts.

## Confirmed engineering correctness issue, deferred

The original supervised process scheduler can omit unscheduled input after an
entire pending batch completes. Its loop condition is
`while pending or next_submit == 0`; after a completed batch empties `pending`,
it can exit even when the input iterator contains more work. A deterministic
executor with already-completed futures returns only `[0, 1, 2, 3]` for ten
inputs and two workers. This can occur with real workers when the pending
batch finishes before the parent resumes.

`test_supervised_map_must_not_drop_inputs_when_entire_batch_finishes` records
this issue as a strict expected failure. The extraction preserves the original
scheduler; it does not claim to repair it. A separate correctness change
should track input exhaustion explicitly and test simultaneous batch completion,
empty input, ordering, worker failure, and generator cancellation.

## Remaining boundaries

The detector still owns forward-model construction, PSF nuisance semantics,
mask construction, template evaluation, and scientific reductions. Its
built-in scalar specification builder still follows the existing source-light
schema. Generalizing that builder to additional parameter schemas and
separating the remaining responsibilities require subsequent changes with
scientific parity checks. The optimized kernels were retained in full.
