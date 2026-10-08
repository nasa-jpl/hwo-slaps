# Population batch

Draws eight lenses around the minimal configuration, forecasts each at 10^8 and 10^9
solar masses, and fits one injected subhalo in the first. The distributions are
illustrative. Batches run on Linux.

```bash
hwoslaps batch plan examples/population/batch.yaml
hwoslaps batch run examples/population/batch.yaml -o out/population --devices cpu --select 'members/system_000000/*'
hwoslaps batch status out/population
```

The two jobs of member 0 take about five minutes on one CPU core. Run `batch run` again
without `--select` to add the other members. See `docs/examples/population.md` and
`docs/guide/batches.md` in the handbook.
