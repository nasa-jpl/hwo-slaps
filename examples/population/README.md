# Population batch

This illustrative population varies source size, Einstein radius and source redshift around
the minimal scene. Its distributions describe no lens survey. Named member streams preserve
existing draws when count changes, when the population is partitioned, or when an unrelated
variable is added. Member names and seeds depend on absolute index.

The matched arm forecasts 1e8 and 1e9 solar masses. One noisy injected `fixed_template`
nonlinear job runs on member 0. Compact sampler settings illustrate the workflow and
establish no posterior-convergence claim.

Batch execution requires Linux with readable `/proc` and pidfd support for owned worker
cleanup. Planning, imports and metadata readers remain portable. Use the supported Linux
science environment for the run command below.

```bash
hwoslaps batch plan examples/population/batch.yaml
hwoslaps batch run examples/population/batch.yaml -o out/population --devices cpu --select 'members/system_000000/*'
hwoslaps batch status out/population
```

Run the same command again to resume completed jobs. `--fresh` refuses an existing batch;
`--verify` checks completed artifact hashes. Selection uses the shown job-id glob.
Other members remain available for a later run with a different selection.

The commented knowledge-error arm needs optical truth, because the minimal scene has a
fixed kernel. `directions` assigns a direction per member and index, shared by amplitude
arms. A nonlinear family can name `forecast_arm` when its reference comes from another
arm, for example a 999-pixel forecast PSF paired with 51-pixel fit support.

Read a batch with `hwoslaps.batch.open_batch(output_dir)`, and products with
`hwoslaps.artifacts.load_forecast(path)` or `load_case(path)`. Batch records contain
completed paths, member values, loaded catalog identity when present, source revisions
and job seeds.

Budget: 900 s CPU for the selected member. Measured runtime and light-group sampling:
pending. Forecast provenance and saved observations retain actual sampling mappings.
Scientific interpretation requires case acceptance and sampler-convergence checks.
