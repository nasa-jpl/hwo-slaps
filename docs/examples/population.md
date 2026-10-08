# Population batch

This example draws a population of eight lenses around the minimal configuration,
forecasts each one, and fits one injected subhalo with a nonlinear model. It shows the
batch file format, resumable execution and how to read the results. The distributions
are illustrative; they describe no lens survey.

## The batch file

```{literalinclude} ../../examples/population/batch.yaml
:language: yaml
:caption: examples/population/batch.yaml
```

Each member draws a source size, an Einstein radius and a source redshift, and `bind`
writes them into the configuration. The forecast family runs every member at 10⁸ and
10⁹ M☉. The nonlinear family `injected` fits member 0 with a 10⁹ M☉ subhalo at
`(y, x) = (1.0, 0.0)`, in noisy data, using a fixed template and small sampler settings.

## Running it

Batches run on Linux. Plan the batch, run the jobs of member 0 on the CPU, and check
the result:

```bash
hwoslaps batch plan examples/population/batch.yaml
hwoslaps batch run examples/population/batch.yaml -o out/population --devices cpu \
    --select 'members/system_000000/*'
hwoslaps batch status out/population
```

The two jobs of member 0 take about five minutes on one CPU core, most of it in the
nonlinear fit. Run the same `batch run` command without `--select` to add the other
seven members; the finished jobs are skipped.

## Reading the results

```python
from hwoslaps.batch import open_batch

batch = open_batch("out/population")
result = batch.forecast("system_000000", "matched")
for job, case in batch.cases(family="injected"):
    print(job.job_id, case.q_signed, case.smooth.acceptance_status)
```

With no refinement, both roles of the fit are `sampler_only`. That is enough to show
the workflow; a scientific comparison needs refined fits and converged sampling, as
described in [Nonlinear fits](../guide/nonlinear.md).

## Knowledge-error arms

The batch file contains a commented-out arm that adds a 10 nm PSF knowledge error with
eight directions per member. It needs an optical truth PSF, because the minimal
configuration's kernel PSF has no wavefront to perturb. Combine it with an optical
instrument, such as the [HWO reference](hwo.md), to run a PSF tolerance study across a
population.
