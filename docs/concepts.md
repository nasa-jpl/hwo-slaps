# How HWO-SLAPS works

A dark-matter subhalo inside a lens galaxy adds a small deflection to the light of the
background source. In the image, this shows up as a slight distortion of the lensed
arc near the subhalo. HWO-SLAPS asks one question: for a given lens, instrument and
exposure, is that distortion large enough to detect?

The answer depends on three things:

- **The size of the distortion**, which grows with the subhalo's mass and depends on
  how much bright, structured arc lies near it.
- **The noise** in each pixel, from the source, sky, detector dark current and read noise.
- **What a smooth lens model can absorb.** A small shift of the lens or source
  parameters can mimic part of a subhalo's signal. Only the part that no smooth model
  can reproduce counts as evidence for a subhalo.

## From configuration to forecast

Every forecast follows the same steps.

```{raw} html
<figure class="pipeline">
<svg viewBox="0 0 760 150" role="img" aria-label="Steps of a forecast: prepare_forecast builds the scene, PSF and observation once; forecast evaluates q for each mass and position; summarize and mass_reach reduce the result.">
  <g font-family="var(--font-stack)" text-anchor="middle">
    <rect x="16" y="20" width="416" height="118" rx="10" class="pipeline-group"/>
    <g font-size="12" font-family="var(--font-stack--monospace)" class="pipeline-note">
      <text x="224" y="42">prepare_forecast()</text>
      <text x="496" y="42">forecast()</text>
      <text x="682" y="42">summarize()</text>
    </g>
    <g class="pipeline-box">
      <rect x="28" y="56" width="120" height="64" rx="8"/>
      <rect x="164" y="56" width="120" height="64" rx="8"/>
      <rect x="300" y="56" width="120" height="64" rx="8"/>
      <rect x="436" y="56" width="120" height="64" rx="8"/>
      <rect x="616" y="56" width="132" height="64" rx="8"/>
    </g>
    <g font-size="15" font-weight="600" class="pipeline-title">
      <text x="88" y="84">Scene</text>
      <text x="224" y="84">PSF</text>
      <text x="360" y="84">Observation</text>
      <text x="496" y="84">Forecast</text>
      <text x="682" y="84">Summaries</text>
    </g>
    <g font-size="11.5" class="pipeline-note">
      <text x="88" y="104">lens, source</text>
      <text x="224" y="104">truth and model</text>
      <text x="360" y="104">image, noise</text>
      <text x="496" y="104">q for each trial</text>
      <text x="682" y="104">areas, mass reach</text>
    </g>
    <g class="pipeline-arrow" stroke-width="2" fill="none">
      <path d="M148 88 H156"/><path d="M284 88 H292"/><path d="M420 88 H428"/><path d="M556 88 H608"/>
    </g>
    <g class="pipeline-head">
      <polygon points="163,88 155,83.5 155,92.5"/><polygon points="299,88 291,83.5 291,92.5"/>
      <polygon points="435,88 427,83.5 427,92.5"/><polygon points="615,88 607,83.5 607,92.5"/>
    </g>
  </g>
</svg>
</figure>
```

1. **Scene.** The lens mass and light, the source light and the cosmology are
   assembled on an oversampled pixel grid. Lensing uses PyAutoLens profiles.
2. **PSF.** The point spread function either comes from a kernel file or is computed
   from a telescope pupil and wavefront with HCIPy. Two PSFs are involved: the
   *truth* PSF that makes the data, and the *model* PSF that the analysis assumes.
   They are the same unless you choose otherwise.
3. **Observation.** The lensed light is binned to detector pixels, convolved with the
   truth PSF and converted to detector counts. Sky, dark current, read noise and gain
   give the expected image and its noise map, both in ADU.
4. **Forecast.** For each trial subhalo mass and position, HWO-SLAPS computes how the
   expected image would change if that subhalo were present, removes the part that
   the smooth model can mimic, and reports the detection statistic *q*.
5. **Summaries.** You choose a detection threshold. HWO-SLAPS then counts detectable
   positions, measures areas and finds the smallest detectable mass.

Steps 1 to 3 happen once, in `prepare_forecast`. Step 4 happens in `forecast`, as
many times as you like. Step 5 works on saved results and needs no scientific
dependencies.

## The detection statistic

Near the smooth model, the expected image $\mu$ changes linearly with small changes in
the model:

$$
\mu = \mu_0 + A\,s + J\,\eta .
$$

Here $\mu_0$ is the smooth image, $s$ is the **subhalo template** (the change in the
image caused by the subhalo), $A$ is its amplitude, and the columns of $J$ are the
changes caused by small shifts $\eta$ of the nuisance parameters, such as the lens
Einstein radius or the source position. All images are divided by the per-pixel noise
before they are combined.

If the nuisance parameters were known exactly, the information on $A$ would be
$s^\top s$. Because they must be fitted at the same time, the part of $s$ that lies
along the columns of $J$ is lost. The information that remains is

$$
F = s^\top s - (s^\top J)\,\left(J^\top J + P\right)^{-1}(J^\top s),
$$

where $P$ holds optional Gaussian prior precisions on the nuisance parameters. Fitting
the nuisance parameters at the same time as the subhalo amplitude, and keeping their best
values, is called *profiling* them.

The template is computed for a subhalo of the full physical mass, so the subhalo is
present when $A = 1$. The **detection statistic** is then

$$
q = F .
$$

HWO-SLAPS calls this `q_asimov`, because it is the value of the likelihood-ratio
statistic for noise-free ("Asimov") data that contain the subhalo. Its square root,
`z_asimov`, is the local significance in standard deviations for a subhalo whose
mass and position are known. A threshold such as *q* ≥ 10 is a choice you make
when you summarize; HWO-SLAPS never applies one for you.

*q* is a local, linear summary. It does not account for searching many positions,
and it assumes the smooth lens and source are described by the configured parametric
profiles. [Nonlinear checks](#nonlinear-checks) below describe how to test it with full
lens-model fits.

## Nuisance parameters

By default every free parameter of the lens mass, the lens light and the source light
is profiled, along with a constant background offset. You can:

- hold parameters fixed by name or pattern, for example `source.light.disk.intensity`
  or `lens.mass.main.*`;
- give parameters Gaussian priors, which make them easier to separate from the subhalo;
- add wavefront modes of the model PSF as nuisances, for PSF studies.

The list of profiled parameters is recorded with every result. See
[Forecasts](guide/forecasts.md#nuisance-parameters).

## When the model PSF is wrong

If the analysis assumes a PSF that differs from the true one, the data contain a
residual even without a subhalo. HWO-SLAPS then reports two further statistics, both
fitted with the model PSF:

`q_mismatch`
: The data contain the subhalo and the PSF error. The subhalo amplitude is fitted
  freely. A detection requires both *q* at or above your threshold and a positive fitted
  amplitude, because a large *q* from a negative amplitude is not a subhalo.

`q_spurious`
: The data contain only the PSF error. Any detection is a false positive caused
  by the PSF error.

Comparing `q_mismatch` and `q_spurious` with the matched result tells you how much
PSF error a detection survives. See [PSFs and PSF errors](guide/psfs.md).

## Positions, masks and areas

`forecast.positions` sets where trial subhalos are placed: a square grid, a ring at
the Einstein radius, or a list of positions. `forecast.mask` sets which detector
pixels enter the statistic. On a grid, each position stands for a square cell, so
counting detectable positions gives a **detectable area** in square arcseconds.

The **mass reach** is the mass at which a summary quantity, such as the largest *q*,
reaches a target. HWO-SLAPS interpolates between the masses you evaluated and reports a
bound, never an extrapolation, when the target lies outside them.

(nonlinear-checks)=
## Nonlinear checks

The forecast linearizes the model around the smooth lens. To test it, HWO-SLAPS can fit
a simulated observation with PyAutoLens and the Nautilus sampler, once without a
subhalo and once with one, and report

$$
q_\mathrm{signed} = 2\left(\ln L_\mathrm{subhalo} - \ln L_\mathrm{smooth}\right)
$$

from the two best fits. For noise-free data, a matched PSF, the same pixels and the
same free parameters, this value approaches `q_asimov` when the subhalo is small.
The fits are much slower than a forecast, so they are best used to check a forecast
at a few masses and positions. See [Nonlinear fits](guide/nonlinear.md).

## Conventions

- Masses are in solar masses. NFW and SIS subhalos use $M_{200c}$; a point mass uses
  its total mass.
- Positions are written as `(y, x)` in arcseconds, measured from the grid centre.
- Images are in ADU. Light rates are in detected electrons per second per pixel.
- Wavefront coefficients are optical path differences in nanometres.

[Conventions and limits](conventions.md) lists every unit and the approximations
behind each step.
