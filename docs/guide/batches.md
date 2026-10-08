# Populations and batches

A batch runs many forecasts, simulations and nonlinear fits from one YAML file. It can
draw a population of lenses, apply several configuration variants to each, and spread
the jobs over CPUs or GPUs. Every finished job writes a completion marker, so an
interrupted batch resumes where it stopped.

## A batch file

This example draws eight lenses from distributions, forecasts each at two masses, and
fits one injected subhalo in the first lens:

```{literalinclude} ../../examples/population/batch.yaml
:language: yaml
:caption: examples/population/batch.yaml
```

The parts are:

`name`, `seed`
: The batch name and the seed for the random choices the batch makes itself: detector
  noise, sampler seeds and PSF error directions. The population has its own seed.

`config`
: The base configuration: one file, a list of files combined in order, or an inline
  mapping. `overrides` can change it for the whole batch.

`population`
: Optional. Random variables and the configuration keys they set (`bind`). Each member
  is one lens. Without a population, the batch has a single member.

`arms`
: Variants applied to every member, each with its own `overrides`. The default is a
  single arm called `base`.

`forecast`, `simulate`, `nonlinear`
: The jobs to run for every member and arm. Each family can be limited to some arms
  with `arms:`.

`execution`
: Where to run: the forecast engine, the devices and the workers per device.

## Populations

Each variable has a distribution `kind`:

| Kinds | Draws |
|---|---|
| `constant`, `choice` | A fixed value, or one of a list with optional weights |
| `uniform`, `log_uniform` | Between `low` and `high` |
| `normal`, `truncated_normal` | With `mean` and `std`, optionally between `low` and `high` |
| `lognormal`, `truncated_lognormal` | With `median` and `sigma_ln`, optionally truncated |
| `vector`, `polar_offset` | A list of values, or a `(y, x)` position from a radius and angle |
| `ell_comps`, `shear_components`, `multipole_components` | Lens and source shape components from physical parameters |

Parameters can refer to earlier variables with `{var: name}`, so one variable can depend
on another. Gaussian copulas correlate variables, and `max_attempts` redraws members that
produce an invalid configuration. See the [configuration reference](../configuration.md)
for every option.

`bind` maps configuration paths to variables:

```yaml
bind:
  scene.source.light.disk.effective_radius: r_eff
  scene.lens.mass.main.einstein_radius: theta_e
```

`population.seed` sets the draws, `count` the number of members, and `start` the index of
the first member (0 by default). Every member has its own random stream, named by its
index, so adding members or changing `count` or `start` never changes the members that
already exist.

## Arms

Arms run the same members under different configurations, for example a matched and a
mismatched PSF:

```yaml
arms:
  - {name: matched}
  - name: ke_10nm
    overrides: {psf: {model: {kind: knowledge_error,
                 draw: {prior: {packaged: jwst_wss_drift_v1}, amplitude_rms_nm: 10.0, seed: 0}}}}
    directions: 8
```

`directions: 8` runs the arm eight times per member, each with a different PSF error
pattern. Each direction's seed is derived from the batch seed, the member and the
direction number, and replaces the `seed` written in the arm.

## Job families

`forecast`
: Forecasts each member and arm at `masses_msun`.

`simulate`
: Simulates observations: `inject` adds the configured `scene.injection`, `noise` draws
  noise, and `replicates` sets how many noise draws to make.

`nonlinear`
: Named families of nonlinear fits. Each family chooses its trials, whether to inject
  the trial and add noise, and the `fit`, `sampler` and `refine` settings described in
  [Nonlinear fits](nonlinear.md).

The `trials` of a nonlinear family can be:

| `kind` | Trials |
|---|---|
| `explicit` | Listed masses and positions, for all or some members |
| `configured` | The `scene.injection` of each member |
| `forecast_positions` | Every forecast position, at the listed masses |
| `forecast_argmax` | The position of the largest forecast *q* at each listed mass, optionally inside an aperture |

Trials taken from a forecast run after that forecast finishes. If the fit and the
forecast use different arms, for example a smaller PSF kernel for the fit, name the
forecast's arm with `forecast_arm`. With `forecast_argmax`, each member's forecast needs at
least one position inside the fit arm's region with a finite *q* (and, for a mismatched PSF,
a positive amplitude). If a member has none, the batch stops, and it stops at the same
member on every resume until you change the batch file.

`retry` gives a nonlinear family one follow-up attempt with different sampler or
refinement settings, for cases whose roles did not reach an accepted status.

## Running a batch

Batches run on Linux, because the runner uses `/proc` and process file descriptors to
track and clean up its workers. Planning a batch and reading its results work anywhere.

Check the batch first. `plan` validates every member and prints the job list:

```bash
hwoslaps batch plan examples/population/batch.yaml
```

Then run it:

```bash
hwoslaps batch run examples/population/batch.yaml -o out/population --devices cpu
```

On GPUs, list the device indices. Each worker gets one GPU:

```bash
hwoslaps batch run batch.yaml -o out/campaign --devices 0,1,2,3
```

| Option | Effect |
|---|---|
| `--devices` | `cpu`, or comma-separated GPU indices among those visible to the process |
| `--workers-per-device` | Workers sharing each device |
| `--select GLOB` | Run only jobs whose identifier matches, for example `'members/system_000000/*'` |
| `--fresh` | Start a new batch, and fail if the output directory already holds one |
| `--verify` | Recheck the hashes of completed artifacts before skipping them |
| `--require-single-revision` | Fail if completed jobs came from different source revisions |

Running the same command again resumes: completed jobs are skipped and the rest run.
A batch whose specification changed in a way that alters a completed job's identity
stops with a conflict instead of mixing results.

Run long batches inside `tmux` or `screen`. If the controlling process is killed, its
workers finish their current jobs and exit. A new `batch run` on the same directory
waits for them, and logs their process IDs in case you want to stop them sooner.

## Reading a batch

```bash
hwoslaps batch status out/population
```

prints the number of jobs of each kind and status, and lists failed jobs with their log
directories. In Python, `open_batch` reads the same information and loads products:

```python
from hwoslaps.batch import open_batch

batch = open_batch("out/population")
for member in batch.members:
    print(member["run_name"], member["index"])

result = batch.forecast("system_000000", "matched")    # a ForecastResult
observations = batch.observations("system_000000", "matched")
for job, case in batch.cases(family="injected"):
    print(job.job_id, case.q_signed)
```

Each job has its own directory under `members/<member>/<arm>/`, holding its artifacts,
`complete.json` and the log of every attempt.
