# RASTI-26-183 code checkpoint

This directory records the code and nonlinear campaign settings associated with
*Point spread function requirements for dark matter subhalo detection with the
Habitable Worlds Observatory*, submitted on 30 September 2026. The surrounding
`rasti` commit is the paper-code checkpoint. Future development can continue on
the branch while this commit remains the reference.

## Code and settings

- Fisher production used commit
  `47590a3e63cbe3ed01eaa7e2b468c044dadf49ea`, retained in this branch's history.
- Of the 1,179 adopted nonlinear comparisons, 1,177 record commit
  `5ddc87f528878e567577ba9010d68f10cf1d521d` and two record
  `c9631605a1651d3fd7edae2aee8079411291d073`. Both revisions are retained. Their
  `src/hwoslaps/modeling/nonlinear/fresh_profile.py` files are identical.
- `archived_helpers/wide_worker.py` is the exact campaign adapter bound by all
  1,179 adopted job specifications. It applies the wider priors and sampler
  settings on top of the committed fitting engine.
- `nonlinear_settings.json` records those settings. The paper uses the `slam1`
  variant and, for 13 adopted second passes, `slam1_retry`. These override the
  earlier defaults in `configs/design/design_freeze_v7.yaml`.
- `archived_helpers/build_wide_tables.py` and `extract_payload_scalars.py` preserve
  the table reduction code. `null500_worker.py` preserves the repeated-control
  helper referenced by the original input bindings. All four are byte-identical
  copies of the archived scripts.

`nonlinear_cases.csv` records case identifiers, trial quantities, PSF and input
hashes, noise and sampler seeds, adopted revisions, payload hashes, and numerical
status. It includes all 1,179 comparisons: 1,174 accepted and five unresolved.
`sampler_seed` is the adopted fit's seed; `first_pass_sampler_seed` and
`second_pass_sampler_seed` record both frozen alternatives. A second-pass seed
does not imply that the second pass was run or adopted. `base_seed_entropy` is a
separate campaign input. `NA` denotes an unavailable or inapplicable field.
`original_payload_sha256` refers to the broad-prior first pass.

`provenance.json` records the original artifact hashes and verification results.
`environment_observed.json` records package versions and editable dependency
revisions inspected on the campaign host on 30 September; it is not a complete
environment lockfile. The submitted manuscript's separate repository revision
is also recorded in the provenance file.

## Checking and using the checkpoint

From the repository root, this check uses only the Python standard library and
does not launch fits or use a GPU:

```sh
python3 reproducibility/rasti-26-183/verify_checkpoint.py
```

The checkpoint was checked against the archived payloads, job specifications,
worker bindings, and frozen seeds. The 56 CPU tests listed in `provenance.json`
also passed on the original campaign host with GPUs disabled.

The archived helpers preserve their original filesystem layout and input
bindings. Running them requires the corresponding configurations, source
assets, PSF inputs, and campaign manifests. The table reducer additionally
requires the original handoff tables, selection files, and fit payloads. These
study-specific materials and generated outputs remain available upon reasonable
request, as stated in the manuscript. The CSV and settings file here are
provenance records, not executable job specifications. The scalar extractor has
an unguarded command-line entry point; it should not be imported directly.
