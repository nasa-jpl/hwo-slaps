# Physics component contracts

The simulation components can now run without a complete campaign configuration.
Existing pipeline calls retain their seeded numerical paths. The change separates
scene randomness, detector randomness, and detector sampling so a new population
or instrument need not borrow a RASTI campaign container.

```python
from hwoslaps.lensing import generate_lensing_system
from hwoslaps.observation import generate_observation, detector_moments
from hwoslaps.psf import generate_psf_system, DetectorPSF

scene = generate_lensing_system(lensing_config, seed=123, run_name="example")
optical_psf = generate_psf_system(psf_config, target_pixel_scale=scene.pixel_scale)
observation = generate_observation(
    scene, optical_psf, observation_config, noise_seed=456, run_name="example"
)
```

Explicit scene and detector seeds override their corresponding values in
`full_config`. Stored provenance records the seed actually used. A PSF detector
scale supplied both explicitly and through `full_config.lensing.grid.pixel_scale`
must agree; no resampling is inferred. PSF configuration snapshots are copied so
later caller edits cannot change recorded provenance.

## External detector PSFs

`DetectorPSF.from_array(values, pixel_scale_arcsec, normalize=True, config=None)`
wraps an external or empirical PSF for observation simulation. Inputs must
already describe response integrated over detector pixels at the declared
angular sampling. The constructor copies the array, rejects even support,
negative or nonfinite values, and nonpositive total flux. Sum normalization is
explicit and records the input flux, normalization choice, support, and angular
sampling in `config.detector_psf`. With `normalize=False`, unit flux is required.

```python
detector_psf = DetectorPSF.from_array(
    calibrated_kernel, scene.pixel_scale,
    config={"run_name": "calibration", "source": "instrument-calibration"},
)
observation = generate_observation(
    scene, detector_psf, observation_config, noise_seed=456, run_name="example"
)
```

This interface enables another telescope's sampled PSF without inventing pupil
segments, wavefront diagnostics, or aberration priors. The full pipeline and
Fisher PSF nuisance machinery still require their existing generated optical
PSF provenance; accepting an external detector kernel there remains follow-on
work.

## Detector expectations and randomness

`detector_moments(source_eps, exposure_time, detector_config)` provides the common
expectation for noise generation and noise maps. Source rates are electrons per
second. Sky and dark rates are detected electrons per second per pixel. Gain is
electrons per ADU and read noise is electrons per exposure. Throughput scales
the source before this contract; it does not scale already detected backgrounds.

The returned object exposes source, sky, dark, and expected electrons, plus the
expected variance in electrons squared. Variance is evaluated only when needed
so drawing an image does not allocate an unnecessary variance map. The legacy
arithmetic order and Poisson-then-Gaussian draw order are preserved.

`apply_detector_noise(..., rng=numpy_generator)` accepts a caller-owned local
stream for population members and replicate forecasts. `seed` and `rng` are
mutually exclusive. Detector mappings may be read-only mapping objects. Importing
these detector functions no longer loads the optional imaging dependencies.

## Selection and study boundary

`apply_floor_cuts` and `rank_pool` accept `theta_e_min_arcsec` and `arc_snr_min`
as keyword arguments. Both thresholds must be finite and nonnegative, and cuts
remain strict. Historical defaults reproduce the submitted selection rule;
other surveys can supply their own floors and tier sizes.

The frozen T4 noise-seed rank-stability harness moved to
`studies/rasti/analysis/rank_stability.py`. Its study script and tests now import
that location. It is no longer exported from `hwoslaps.analysis`. Reusable array
observables and deterministic ranking remain in the engine.

## Validation and remaining scope

Local dependency-light verification: 132 tests passed across detector contracts,
selection statistics, and the migrated rank-stability harness. All edited
runtime modules parse successfully. New integration contracts compare standalone
lensing, optical PSF, observation, and external-kernel interfaces against the
legacy outputs exactly; those require the XTX PyAutoLens/HCIPy environment.

No scientific model or numerical default changed in this pass. The current
optical simulator remains monochromatic with a hexagonally segmented pupil.
Scene generation supports its existing isothermal macro lens and exponential or
image-asset source morphology. External image assets already provide a route to
more complex source shapes; a validated profile registry, broader macro models,
bandpass integration, and correlated detector noise remain future extensions.
