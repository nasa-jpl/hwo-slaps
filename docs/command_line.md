# Command line

Installing hwoslaps adds the `hwoslaps` command. `python -m hwoslaps` is equivalent.

| Command | Does |
|---|---|
| `hwoslaps validate CONFIG...` | Combine and check configuration files, and print the digest |
| `hwoslaps forecast CONFIG... --masses M... -o DIR` | Run a forecast and save it |
| `hwoslaps simulate CONFIG... (--noise-seed N \| --expected) -o DIR` | Simulate an observation and save it |
| `hwoslaps reference [SECTION]` | Print the configuration reference |
| `hwoslaps batch plan SPEC` | Validate a batch file and print its jobs |
| `hwoslaps batch run SPEC -o DIR` | Run or resume a batch |
| `hwoslaps batch status DIR` | Summarize a batch's completed and failed jobs |

## Common options

`CONFIG...`
: One or more configuration files, combined in order.

`--set PATH=VALUE`
: Change one configuration value, for example `--set observation.exposure_time_s=4000`.
  The value is read as YAML. Repeat the option to change several values.

`-o DIR`, `--output-dir DIR`
: The output directory. For `forecast` and `simulate` it must not exist yet. `batch run`
  creates it, or resumes the batch already in it.

`--log-level {DEBUG,INFO,WARNING}`
: Placed before the command, for example `hwoslaps --log-level WARNING forecast ...`.
  Sets what is printed; the full log is always written to `run.log`.

## validate

```bash
hwoslaps validate configs/minimal.yaml
hwoslaps validate configs/minimal.yaml --print
```

`--print` writes the complete configuration, with every default filled in, instead of
the digest.

## forecast

```bash
hwoslaps forecast configs/minimal.yaml --masses 1e7 1e8 1e9 -o out/minimal
```

| Option | Meaning |
|---|---|
| `--masses` | Subhalo masses in solar masses (required) |
| `--engine {reference,jax}` | The forecast engine (default `reference`) |
| `--reference-workers N` | CPU worker processes for the reference engine (default 1) |
| `--batch-size N` | Positions per JAX evaluation (default 16) |
| `--progress` | Show a progress bar |

Writes `forecast.npz`, `effective_config.yaml`, `provenance.json` and `run.log`.

## simulate

```bash
hwoslaps simulate configs/minimal.yaml --noise-seed 11 -o out/injected
hwoslaps simulate configs/minimal.yaml --smooth --noise-seed 11 -o out/control
hwoslaps simulate configs/minimal.yaml --expected -o out/expected
```

| Option | Meaning |
|---|---|
| `--noise-seed N` | Draw detector noise with this seed |
| `--expected` | Write the expected image, without noise |
| `--smooth` | Leave out the subhalo; without it, `scene.injection` is used |

Exactly one of `--noise-seed` and `--expected` is required. Writes `observation.npz`,
`effective_config.yaml`, `provenance.json` and `run.log`.

## reference

```bash
hwoslaps reference scene.subhalo
```

Prints the [configuration reference](configuration.md), or the part of it at or below
a section path.

## batch

```bash
hwoslaps batch plan examples/population/batch.yaml
hwoslaps batch run examples/population/batch.yaml -o out/population --devices cpu
hwoslaps batch status out/population
```

| `batch run` option | Meaning |
|---|---|
| `--devices` | `cpu`, or comma-separated GPU indices |
| `--workers-per-device N` | Workers sharing each device |
| `--select GLOB` | Run only the jobs whose identifier matches |
| `--fresh` | Start a new batch, and fail if the output directory already holds one |
| `--verify` | Recheck the hashes of completed artifacts |
| `--require-single-revision` | Fail if completed jobs came from different source revisions |

`batch run` exits with status 3 when some jobs failed, after printing its report.
See [Populations and batches](guide/batches.md).

## Full help text

The help printed by each command:

<!-- GENERATED_COMMAND_LINE -->
