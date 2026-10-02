# Correctness concerns deferred from the refactor

These concerns predate the cleanup. They were reproduced or compared against
the submitted checkpoint and retained for a separate correction with its own
validation. They are not claims of a measured effect on the published results.

## Fisher supervised iterator can drop unscheduled inputs

`hwoslaps.modeling.fisher_runtime.supervised_ordered_map` retains the original
loop condition. If all pending futures complete together, the loop can terminate
while the input iterator still has unscheduled items. A deterministic fake-pool
regression sends ten inputs with two workers and returns only four. The strict
expected-failure test in `tests/test_fisher_runtime.py` records this defect.
The extracted loop is AST-identical to the submitted implementation.

A separate fix should distinguish input exhaustion from an empty pending batch,
verify all ordered results, and retain failure supervision. This refactor does
not establish whether the published campaign encountered the triggering schedule.

## Serialized profile settings omit a start-selection parameter

`FreshProfileSettings.to_dict` omits `start_separation_posterior_sigma`.
A nondefault value of 2.0 reconstructs as 1.0 through
`from_mapping(to_dict())`. Both methods preserve the original executable logic.
A separate change should define a complete round-trip settings contract and
assess whether any saved execution configuration depended on reconstruction.

## New-interface issues resolved during review

Independent peer review identified a collision in the new population API's
32-bit member seeds (seed 2, members 110170 and 111187), out-of-support nonfinite
draws, and artifact directory overwrites. The new API now uses injective integer
seed pairing, rejects nonfinite draws, and refuses existing artifact run
directories. Regression tests cover these cases. These changes affect new
interfaces, not the frozen RASTI sampling or scientific calculations.
