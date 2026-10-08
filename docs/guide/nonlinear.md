# Nonlinear fits

A forecast is a linear approximation. To check it, HWO-SLAPS fits a simulated
observation with full lens models: once with a smooth lens (the **smooth role**) and
once with a subhalo (the **subhalo role**). It uses PyAutoLens for the model
and the Nautilus nested sampler for the search, and can then polish each best fit with
a gradient optimizer. The result is

$$
q_\mathrm{signed} = 2\left(\ln L_\mathrm{subhalo} - \ln L_\mathrm{smooth}\right),
$$

which can be compared with the forecast's *q* at the same mass and position.

Fits take minutes to hours each, depending on the image size and sampler settings.
Use them to check a forecast at selected points, and use a GPU with the JAX likelihood
for anything beyond a small test.

## A first fit

```python
from hwoslaps import (FitSpec, SamplerSettings, load_config, prepare_forecast,
                      simulate, validate_nonlinear)

config = load_config("configs/minimal.yaml")

with prepare_forecast(config) as prepared:
    trial = prepared.hypothesis(1e9, (0.0, 1.0))
    observation = simulate(prepared, subhalo=trial, noise_seed=None)

    case = validate_nonlinear(
        prepared, trial, observation,
        fit=FitSpec(mode="fixed_template"),
        sampler=SamplerSettings(n_live_smooth=50, n_live_subhalo_fixed=50),
        sampler_seed=1,
        output_dir="out/fits",
    )

print(case.q_signed, case.smooth.acceptance_status, case.subhalo.acceptance_status)
```

`validate_nonlinear` needs:

- a prepared forecast, which supplies the scene, the model PSF and the parameter list;
- the trial subhalo, usually from `prepared.hypothesis`;
- an observation of the same configuration, from `simulate`;
- a `FitSpec` (what to fit), `SamplerSettings` (how to search) and a sampler seed;
- an output directory, under which each case gets its own subdirectory.

The sampler seed is separate from the scene seed and the noise seed, so the same data
can be refitted with a different search.

## Fit modes

`FitSpec.mode` sets what the subhalo role may vary.

| Mode | The subhalo role |
|---|---|
| `fixed_template` | The subhalo is fixed at the trial mass and position. |
| `local_search` | The subhalo position is free within a small window (0.03 arcsec) around the trial position. |
| `freed` | The position is free within 0.15 arcsec and the mass is free within a `MassSupport`. |

```python
from hwoslaps.inference import MassSupport

fit = FitSpec(mode="freed", mass_support=MassSupport(log10_mass_min=6.0, log10_mass_max=11.0))
```

`fixed_template` is the closest to the forecast and the fastest. `freed` is the closest
to a real search.

In every mode, the lens and source parameters are free in both roles, inside uniform
boxes centred on their true values.

## Prior boxes

`PriorWidths` sets the half width of each parameter's box by kind, as a `BoxRule`.
The defaults are narrow:

| Key | Default half width |
|---|---|
| `lens.position` | 0.005 arcsec |
| `lens.einstein_radius` | 0.01 arcsec |
| `lens.ellipticity` | 0.02 |
| `source.position` | 0.01 arcsec |
| `source.ellipticity` | 0.05 |
| `source.amplitude`, `lens.amplitude` | 50% of the true value |
| `source.size`, `lens.size` | 30% of the true value |

`hwoslaps reference fit` lists every key.

### The settings used in the RASTI paper

The nonlinear fits in the RASTI paper used wider boxes and more live points than the
defaults. To reproduce them, build the settings explicitly:

```python
from hwoslaps import FitSpec, RefineSettings, SamplerSettings
from hwoslaps.inference import BoxRule, MassSupport, PriorWidths
from hwoslaps.inference.settings import DEFAULT_BOX_RULES

rules = dict(DEFAULT_BOX_RULES) | {
    "lens.position": BoxRule(0.05),
    "lens.einstein_radius": BoxRule(0.25, fractional=True),
    "lens.ellipticity": BoxRule(0.2, clip=(-0.9, 0.9)),
    "source.position": BoxRule(0.05),
}
paper_widths = PriorWidths(rules=tuple(sorted(rules.items())))

fit = FitSpec(mode="freed", prior_widths=paper_widths,
              mass_support=MassSupport(log10_mass_min=6.0, log10_mass_max=11.0))
sampler = SamplerSettings(n_live_smooth=300, n_live_subhalo_fixed=300, n_live_subhalo_search=600,
                          n_eff=2000, n_like_max=2_000_000, f_live=0.01, n_shell=1,
                          discard_exploration=False, retain_search_internal=True,
                          use_jax=True)
refine = RefineSettings()
```

For an image (pixelized) source, the paper also widened the size scale to 50%:
add `"source.size": BoxRule(0.5, fractional=True)` for those cases.

## Masks

The fit uses every pixel except a border half a PSF wide, by default. To fit exactly
the forecast's pixels, use the forecast mask instead:

```python
fit = FitSpec(mode="fixed_template", mask="forecast_mask_minus_psf_border")
```

For any other set of pixels, pass a boolean image with `True` for fitted pixels, as
`mask=PixelMask(pixels)` (`PixelMask` is in `hwoslaps.inference`). The mask is recorded
with the result, and a comparison with the forecast reports any difference between the
two.

## Refinement

The sampler's best point is close to, but not exactly at, the maximum likelihood.
`RefineSettings` adds a multistart L-BFGS-B optimization of each role, starting from
several well-separated sampler points, followed by a tighter repeat from the best one.
Refinement needs the JAX likelihood:

```python
case = validate_nonlinear(prepared, trial, observation, fit=fit,
                          sampler=SamplerSettings(use_jax=True), refine=RefineSettings(),
                          sampler_seed=1, output_dir="out/fits")
```

A refined role is **accepted** when several independent starts agree, the tighter repeat
agrees, the gradient is finite, and the likelihood is consistent with a direct
evaluation and with the sampler. Otherwise it is **unresolved**: its best value is kept,
but it is not trusted for classification.

Each role's `acceptance_status` is one of:

| Status | Meaning |
|---|---|
| `accepted_repeatable_profile` | Refinement converged and passed every check |
| `unresolved_optimization` | Refinement ran but did not pass every check |
| `sampler_only` | No refinement was requested; the value is the sampler's maximum |
| `verified_zero_residual_anchor` | The role was evaluated at the truth and fits the data exactly |
| `failed` | The search failed; there is no likelihood |

Without `refine`, both roles are `sampler_only`, which is enough for exploring but not
for the paper's classification rule below. `verified_zero_residual_anchor` occurs only with
`FitSpec(h1="truth_anchor")`, which evaluates the subhalo role at the true subhalo instead
of searching, for checks on noise-free data.

## The result

`validate_nonlinear` returns a `CaseResult`:

| Attribute | Meaning |
|---|---|
| `q_signed` | $2(\ln L_\mathrm{subhalo} - \ln L_\mathrm{smooth})$, which can be negative |
| `q_clipped` | `max(q_signed, 0)`, for display |
| `delta_log_evidence` | The difference in log evidence, when both searches finished |
| `smooth`, `subhalo` | Each role's `acceptance_status`, likelihood, sampler record and refinement record |
| `recovery` | For `freed` fits, the recovered subhalo mass and position |
| `fit`, `sampler`, `refine`, `sampler_seed` | The settings used |
| `observation`, `hypothesis` | What was fitted |
| `forecast_reference` | The forecast at the same mass and position, if supplied |

Save it with `save_case` and read it back with `load_case`.

A subhalo role with a fixed halo does not contain the smooth model as a special case,
and a freed halo with a minimum mass does not either. `q_signed` is therefore a
likelihood difference, not a statistic with a known distribution, and it can be
negative.

## Classifying cases

`classify_case` applies a detection rule to a result. The rule is always yours. This is
the rule the RASTI paper used:

```python
from hwoslaps.analysis import ClassificationRule, RoleAcceptance, classify_case

accepted = {"accepted_repeatable_profile", "verified_zero_residual_anchor"}
rule = ClassificationRule(
    q_threshold=10.0,
    marginal_half_width=1.0,
    acceptance=RoleAcceptance(smooth=accepted, subhalo=accepted),
    require_retained_state=True,
    retry_log_likelihood_tolerance=0.1,
    stationarity_tolerance=None,
)
outcome = classify_case(case, rule)
print(outcome.status, outcome.detected, outcome.marginal)
```

`outcome.status` is one of:

| Status | Meaning |
|---|---|
| `accepted` | Both roles have an acceptance status the rule allows |
| `unresolved` | A role finished, but its acceptance status is not allowed by the rule, or it fails the optional stationarity check |
| `failed` | A role's search failed |
| `incomplete` | The rule requires the sampler's saved state, and it was not kept |

For an `accepted` case, `detected` says whether `q_signed` reaches the threshold, and
`marginal` flags values within `marginal_half_width` of it. Other cases have
`detected = None`; they are never counted as non-detections.

`require_retained_state=True` needs the sampler's internal files to be kept, which
`SamplerSettings(retain_search_internal=True)` does. Without them, the case is
`incomplete`.

`stationarity_tolerance` adds an optional requirement that the projected gradient at
each refined maximum be small. `None` leaves it out, as in the paper.

`retry_log_likelihood_tolerance` matters when a case was fitted twice, for example by a
batch `retry`. `select_attempt` uses the retry only if it is accepted and neither role's
log-likelihood is lower than in the first attempt by more than this amount.

## Comparing with the forecast

Pass `forecast_reference` to record the forecast's value at the same point. Build it from
a forecast result with `ForecastReference.from_result(result, mass_index=i, position_index=j)`
(`ForecastReference` is in `hwoslaps.inference`). Then use `detection_agreement` to compare a set of classified cases with their forecasts. For a
fair comparison:

- fit noise-free data (`noise_seed=None`) with a matched PSF;
- fit the forecast's pixels (`mask="forecast_mask_minus_psf_border"`);
- keep the same free lens and source parameters.

The forecast also profiles a background offset that the fit does not include; set
`forecast.nuisances.background_offset: false` to remove it from the forecast. The comparison
reports this and any other difference between the two set-ups.

To estimate false detections, fit smooth controls with noise:
`simulate(prepared, subhalo=None, noise_seed=seed)` for many seeds. A noise-free smooth
image is fitted exactly by the smooth model and tells you nothing about noise.

## Reusing a backend session

Each call starts and stops the PyAutoLens backend unless you pass a `BackendSession`.
When running many fits in one process, open one session and pass it to every call:

```python
from hwoslaps.inference import BackendSession

with BackendSession() as session:
    for seed in range(10):
        ...
        case = validate_nonlinear(..., session=session)
```

For many fits across lenses and GPUs, use a [batch](batches.md) instead.
