# Using the forecasting engine

## Configuration composition

`hwoslaps.config.load_config` reads one YAML file or an ordered list of files.
Mappings merge recursively; scalar and list values replace the earlier value.
Each file's declared source-asset, covariance, and output paths are resolved
against that file before merging. Programmatic overrides resolve against the
caller's directory, or an explicit `base_dir`. Input mappings are copied.

```python
from hwoslaps.config import load_config

config = load_config(
    ["configs/master_config.yaml", "configs/examples/observation_override.yaml"],
    overrides={"plotting": {"output_dir": "outputs/example"}},
)
```

Validation checks the final composed engine configuration. It does not infer
units or turn an unsupported physical model into a supported one. Existing
field units are explicit in the example configuration: angular coordinates in
arcseconds, wavelength and pupil geometry in metres, wavefront coefficients in
nanometres, exposure in seconds, and detector counts in electrons.

## Simulation and forecasting

`run_pipeline` accepts a configuration mapping, a YAML path, or a sequence of
YAML paths. Its standard and detection modes retain the existing numerical
routes. `run_with_artifacts` additionally writes a resolved snapshot, log, and
provenance record from the same configuration passed to the engine.

For in-memory grid results, pass `save_grid_maps=False` to `run_pipeline` and
also disable `plotting.enabled` and `psf.hres_psf.save_highres_psf_npy` when no
other configured exports are wanted. `run_with_artifacts` always captures a run.
Existing run artifact directories are rejected; choose a new run identity.

Individual module entry points support independently constructed components:

```python
from hwoslaps.lensing import generate_lensing_system
from hwoslaps.psf import generate_psf_system
from hwoslaps.observation import generate_observation

scene = generate_lensing_system(config["lensing"], seed=123, run_name="example")
psf = generate_psf_system(config["psf"], target_pixel_scale=0.00716)
observation = generate_observation(
    scene, psf, config["observation"], noise_seed=456, run_name="example"
)
```

Keep the detector sampling consistent with the scene grid. Explicit component
inputs make that dependency visible; they do not change the physical models.
Fisher geometry, nuisance specifications, and worker supervision are separate
from rendering and accelerated grid evaluation. Nonlinear optimizer settings
and calibration diagnostics are separate from the sampler and likelihood.

## External instruments and image-level forecasts

`hwoslaps.psf.DetectorPSF.from_array` accepts an empirical or externally rendered
PSF already integrated at the detector's angular sampling. It can be used with
`generate_observation` without inventing segmented-pupil metadata. Its constructor
copies and validates the kernel and records normalization provenance.

The image-level Fisher adapters accept expected smooth/subhalo images, noise
weights, and caller-supplied nuisance derivatives independently of the telescope:

```python
from hwoslaps.modeling.fisher_adapter import compute_asimov_from_images

forecast = compute_asimov_from_images(
    smooth_mean_image=smooth_e,
    subhalo_mean_image=perturbed_e,
    sigma_image=sigma_e,
    nuisance_images=derivative_images,
)
```

Images and noise must use consistent units and sampling; here they are electrons.
The caller defines the nuisance responses for its scene model. Omitting them
holds the smooth model fixed and changes the scientific question. The optical
PSF nuisance and full sensitivity-grid pipeline still use generated optical
PSFs; wiring external detector kernels into those paths requires further work.

## Population recipes

`hwoslaps.population` samples named independent distributions without importing
AutoLens, HCIPy, JAX, or the RASTI study. It supports `constant`, `choice`,
`uniform`, `log_uniform`, `normal`, `truncated_normal`, and `lognormal`.
Normal distributions use `mean` and `std`; lognormal uses `median` and
`sigma_ln`, the standard deviation in natural-log space. Choices accept
optional non-negative weights, including source asset names or whole blocks.

```python
import yaml
from hwoslaps import iter_population_configs

with open("configs/examples/population.yaml") as stream:
    parameters = yaml.safe_load(stream)

for member in iter_population_configs(config, parameters, 100, seed=42):
    # Pass to run_pipeline, run_with_artifacts, or an explicit job executor.
    print(member["run_name"], member["global_seed"])
```

Parameter names are existing dotted configuration paths. Lists or mappings
can be selected/replaced as complete values. Typographical errors, overlapping
parent/child paths, and overrides of generated run identity are rejected.
The sampler validates the resulting physical configuration by default.

Each parameter and member has a separate deterministic stream. Extending the
population, adding parameters, changing mapping order, or sampling a chunk
with `start` preserves existing parameter draws. Noise seeds and run names
are derived from member identity. Noise seeds use an injective integer pairing
of the population seed and member index, avoiding 32-bit birthday collisions.
Nonfinite numeric draws fail with the parameter name instead of entering a
forecast configuration. The base configuration and global NumPy
random state are unchanged.

These are independent distributions, not an observational selection model.
Correlated lens/source properties, conditional redshift bounds, and survey
selection must be encoded in an explicit caller recipe. The frozen RASTI
population remains separate and retains its original sampling rules.

## Campaign execution and extension limits

`hwoslaps.campaign` retains immutable manifest execution, artifact validation,
and adaptive mass-ladder primitives. A job configuration is a complete engine
input. The executor owns process lifecycle and receipts; forecasting modules
own scientific results. The RASTI builders are source-only examples of using
these primitives, with their original paper constraints intact.

Changing exposures, detector parameters, supported source assets, nuisance
settings, selection thresholds, or distributions uses existing engine inputs.
Additional pupil families, broadband/chromatic image formation, correlated
population recipes, and flexible source reconstruction need physical
implementations and validation before they can be declared supported.
