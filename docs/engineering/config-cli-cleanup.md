# Configuration, CLI, and pipeline cleanup

The pipeline now accepts portable composed configurations, and the installed
CLI owns run capture. Scientific generators and Fisher routing retain their
existing numerical behavior. This change introduces explicit path semantics
and a reusable Python entry point; it does not add new scientific models.

## Interfaces

`hwoslaps.config.load_config(path_or_paths, overrides=None, base_dir=None)`
loads one YAML mapping or composes an ordered sequence. Later mappings merge
recursively; lists and scalars replace whole values. The loader copies input
data and validates the final configuration. Each file resolves its declared
paths before composition, so a source fragment and an instrument fragment
can live in different directories.

The path fields are `plotting.output_dir`,
`lensing.source_galaxy.light.asset_path`, and
`modeling.fisher.covariance_path`. Identifiers and arbitrary strings are not
interpreted as paths. Python overrides resolve against the caller's working
directory unless `base_dir` is explicit. Absolute paths remain absolute.

`run_pipeline(config_or_paths, overrides=None, base_dir=None,
save_grid_maps=True)` accepts a mapping or YAML files and returns the existing
observation/Fisher result types. `Pipeline.run(mapping)` retains its existing
direct-mapping entry behavior. The historical `run_enhanced_pipeline` name
remains a thin wrapper, with an explicit `base_dir` escape for old inputs.

`hwoslaps.cli.run_with_artifacts` captures a resolved configuration snapshot,
log, and provenance record. These records describe the same configuration
passed to `Pipeline`; the old runner loaded the file twice and hashed the
unresolved document while the pipeline used a resolved output path.
`runner.py` delegates to the installed CLI implementation.

The grid-map writer lives in `hwoslaps.artifacts`, rather than embedding YAML,
campaign metadata, directory creation, and NPZ persistence in the scientific
pipeline. `save_grid_maps=False` retains a grid forecast in memory. Plotting
and high-resolution PSF export remain controlled by their existing switches.

## Usage and migration

```sh
hwoslaps -c configs/master_config.yaml -c my_instrument.yaml -c my_scene.yaml \
  --output-dir outputs --run-name pilot --validate-only

hwoslaps -c configs/master_config.yaml --base-dir . --output-dir outputs
```

The second command retains repository-relative paths from historical YAML.
Without `--base-dir`, a relative path belongs to the directory of the file
declaring it. A CLI `--output-dir` override always belongs to the caller's
working directory, including when `--base-dir` is supplied.

```python
from hwoslaps.config import load_config
from hwoslaps.pipeline import run_pipeline

config = load_config(["instrument.yaml", "scene.yaml", "forecast.yaml"])
result = run_pipeline(config, save_grid_maps=False)
```

Run names must be one nonempty directory component for artifact capture;
nested output organization belongs in `plotting.output_dir`. Validation-only
does not import the scientific runtime or create output directories.
Artifact capture requires a previously unused run directory. Repeated runs
must choose a new run name or output root; prior snapshots, logs, provenance,
and arrays are never silently overwritten. Directory creation is exclusive
so two simultaneous invocations cannot claim the same run directory.

New snapshots contain fully resolved paths. Grid maps bind only to an exact
adjacent snapshot; the old implicit repository-root normalization has been
removed. Replaying an old relative snapshot through the new CLI with
`--base-dir` produces a new resolved snapshot, while the original archive can
remain intact in its historical output directory.

## Validation

Local validation on 2026-10-02 used:

```sh
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python -m pytest -q -p no:cacheprovider \
  tests/test_config_loading.py tests/test_cli.py tests/test_artifacts.py \
  tests/test_config_validation_observation.py \
  tests/test_config_validation_fisher.py \
  tests/test_lensing_config_validation_strictness.py
```

Result: **260 passed in 2.68 seconds**. These cover composition, declaration-local
paths, explicit historical replay, input isolation, malformed YAML, invalid
run names, dependency-free CLI validation, consistent snapshot/provenance
hashes, collision preservation, exact grid-map snapshot binding, and standalone
artifact metadata.
Flake8 passes for the changed configuration, CLI, artifact, pipeline, wrapper,
and new test modules. The master YAML also passes the CLI validation command.

`tests/test_provenance.py` and `tests/test_pipeline_fisher_routing.py` require
the scientific environment. The local collection attempt failed because
`autoarray` is absent; this is an environment limitation, not a numerical
validation result. These targets, including the new programmatic-entry test,
were sent to the integration agent for XTX validation. No fits, kernel
benchmarks, or GPU runs were launched by this subtask.

## Remaining limitations

The scientific schema still uses its existing dictionary sections and
validation rules. Outputs remain configured under `plotting.output_dir` even
when plotting is disabled. Moving output configuration to an execution
section and separating population generation from one-system simulation need
a later coordinated schema change.

Recursive merging preserves unspecified keys. When switching a polymorphic
light-profile type, callers must construct a valid replacement block rather
than retain keys from a different type; the strict scientific validator will
reject unsupported keys. No inheritance or plugin framework was introduced.

Historical study-specific generators still control their frozen campaign
contracts. Their existing archive bytes and scientific assumptions are
outside this subtask. No scientific correctness findings were changed.
