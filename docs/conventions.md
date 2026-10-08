# Conventions and limits

This page collects the units, coordinate conventions and approximations behind every
step, and the limits of what a forecast or fit can tell you.

## Units

| Quantity | Unit |
|---|---|
| Subhalo mass | Solar masses: total mass for `PointMass`; $M_{200c}$ for `SIS`, `NFW` and `TNFW` |
| Positions | Arcseconds, written `(y, x)`, measured from the centre of the image grid |
| Distances | Megaparsecs where a record names them |
| Pupil size, wavelength | Metres and nanometres, as named in each key |
| Wavefront coefficients | Nanometres of optical path difference |
| Light, sky and dark rates | Detected electrons per second; image rates are per detector pixel |
| Image data and noise | ADU; the gain is electrons per ADU |
| Exposure time | Total seconds over all exposures |
| Angles | Degrees, counter-clockwise from +x toward +y |
| *q*, significances, ratios | Dimensionless |

## Coordinates

- Positions are `(y, x)` in arcseconds, with the origin at the centre of the image grid.
- In detector images, row 0 has the largest *y*, and columns increase in *x*. Image-source
  assets are the exception: their row 0 is the bottom row.
- In plots and trial-position grids, *y* increases upward and *x* to the right.
- Ellipticities use the PyAutoLens components $(e_1, e_2) = (f \sin 2\phi, f \cos 2\phi)$,
  with $f = (1-b/a)/(1+b/a)$ for axis ratio $b/a$ and major-axis angle $\phi$ from +x.

## Lens and source models

Lens mass profiles are `Isothermal`, `PowerLaw` and `ExternalShear`, with optional m = 3
and m = 4 multipoles on the first two. Light profiles are `Exponential`, `Sersic` and
`Image`. The `Image` profile uses a fixed pixelized galaxy whose position, brightness,
size and rotation are free; its pixel values are not. None of these models is a free-form
source reconstruction, so a forecast is conditional on the source being described by
the chosen profiles.

Multipole amplitudes must keep the total convergence positive. hwoslaps checks a
sufficient condition at the configured values and, for nonlinear fits, at the corners of
each prior box. The check can reject some valid combinations. It does not change other
parameters to make a combination pass.

## Subhalos

| Type | Mass | Profile |
|---|---|---|
| `PointMass` | Total mass | A point mass |
| `SIS` | $M_{200c}$ | A singular isothermal sphere |
| `NFW` | $M_{200c}$ | An untruncated NFW profile, with a fixed, power-law or Moliné et al. (2017) concentration |
| `TNFW` | $M_{200c}$ of the parent NFW | An NFW profile with the Baltz, Marshall and Oguri (2009) *n* = 2 truncation; the total truncated mass is recorded |

The Moliné relation is applied at the lens redshift and only within its calibrated
range of $10^6$ to $10^{12}$ M☉. A subhalo off the lens plane is placed in the angular
coordinates of its own plane.

Line-of-sight halos, if configured, are drawn once from the configuration's `seed` and
held fixed in both the smooth and subhalo models. hwoslaps applies no subhalo mass
function of its own.

## Cosmology

A named flat ΛCDM cosmology (such as `Planck15`) or custom flat ΛCDM parameters set the
distances. Halo scales use the critical density

$$
\rho_\mathrm{crit}(z) = \frac{3 H(z)^2}{8\pi G}, \qquad
H(z) = H_0 \sqrt{\Omega_m (1+z)^3 + 1 - \Omega_m},
$$

which leaves out radiation and massive neutrinos even when the distance calculation
includes them. The RASTI paper used the same convention.

## Photometry and detector

An AB magnitude is converted to a detected rate with the bandpass throughput, the
collecting area, the spectrum and the photon energy. Throughput is applied once, to
the light. Sky and dark rates given in electrons per second are already detected rates
and are not multiplied by the throughput.

Each pixel's variance in electrons squared is

$$
\sigma^2 = \max(L, 0)\,t + S\,t + D\,t + N r^2
$$

for light rate $L$, sky rate $S$, dark rate $D$, exposure time $t$, $N$ exposures and
read noise $r$. Images and noise maps are divided by the gain to give ADU. The detector
model has no saturation, cosmic rays, interpixel capacitance or flat-field errors.

## Rendering and pixel sampling

Light is evaluated on a grid oversampled by `over_sample_size`, binned to detector
pixels, and convolved with a PSF that is already integrated over a detector pixel. This
is exact when the light is constant within each detector pixel and approximate
otherwise. Increasing the oversampling cannot recover structure lost in the binning.

`Observation.sampling` measures the relative variation of the smooth scene's light
within detector pixels, for each light group. In the calibration tests, values below
0.063 kept the error in the subhalo signal information below 1% compared with a finer
reference. The value is reported, not enforced. Some of the paper's test scenes exceed it, and
small values do not guarantee small errors for every subhalo position. For a new kind of
scene, compare against finer pixels.

## Optics

Optical PSFs are propagated from the pupil and wavefront with HCIPy and integrated over
detector pixels. Kernels must have odd dimensions, and every wavelength must be sampled
finely enough that the PSF is neither aliased nor under-sampled at the detector scale.
Kernel files must be sampled at the detector pixel scale.

Kernels are normalized to unit sum on their finite support. The fraction of the PSF's
total power that lies inside the support is recorded as its captured fraction.

For a chromatic PSF, each wavelength's kernel is normalized on its support before the
photon-weighted sum. If wavelength $k$ captures a fraction $f_k$ of its power, this
scales its contribution by $1/f_k$, moving power from the wings into the kernel by a
different amount at each wavelength. Check convergence in the number of wavelengths and
the kernel size, as the [chromatic example](examples/chromatic.md) does.

Spectral weights are stored as relative integrals of throughput × *f*ν × dlnλ with a
common scale factor. They set the relative weight of each wavelength, not absolute
counts; absolute rates come from the photometry.

## The linear forecast

The forecast linearizes the expected image around the smooth model and profiles a
Gaussian likelihood over the nuisance parameters ([How hwoslaps works](concepts.md)).
Results depend on the parametric lens and source model, the pixel mask, the noise
model, the PSF kernels and the finite-difference steps.

`q_asimov` is the profiled information on the subhalo amplitude, equal to the
likelihood-ratio statistic for noise-free data containing the subhalo. With a
mismatched model PSF, `q_mismatch` fits a free amplitude $\hat a$ to data made with the truth
PSF, giving $q_\mathrm{mismatch} = \hat a^2 F$, and `q_spurious` fits an amplitude to the PSF
error alone. A large *q*
from a negative fitted amplitude is not a subhalo, so mismatch detections also require a
positive amplitude. *q* and $\sqrt{q}$ are local summaries for a subhalo of known mass
and position; they do not include the cost of searching many positions.

The nuisance fit uses a pseudo-inverse that drops eigen-directions smaller than
$\max(10^{-12}, p\,\epsilon)$ times the largest eigenvalue, where $p$ is the number of nuisance
parameters and $\epsilon$ is the double-precision machine epsilon. The result is unchanged if every
nuisance column is rescaled by the same factor, but it depends on the relative units of
individual columns. Each result records the nuisance rank and the condition number so
you can see when this cutoff acts.

Forecasts can use a dense noise covariance. Simulations draw independent pixel noise,
and nonlinear fits use independent pixels, so `validate_nonlinear` does not accept a
forecast prepared with a dense covariance.

## Areas and mass reach

On a grid, each position stands for a square cell of side equal to the grid spacing, so
a detected area is the number of detected positions times the spacing squared. If the
detected region touches the edge of the grid, the area could be larger on a wider grid,
and `boundary_detectable` says so. A subset of positions that drops edge nodes cannot
tell whether the edge was reached, and reports `None`.

Mass reach interpolates in log mass between the two evaluated masses on either side of
the target. It reports a bound when the target is outside the evaluated range, and no
value when the curve is not monotonic. It never extrapolates.

## PSF knowledge error

`knowledge_error_areas` requires the two forecasts to share their comparison digest,
positions, pixel mask, nuisance list and truth PSF. Only the model PSF may differ. Ratios
are `NaN` when the reference detects fewer positions than the chosen floor.

`knowledge_error_tolerance` applies its gates to one set of keys (for example the
directions of one lens at one mass) that must be present at every amplitude. Keys below
the floor can be listed as ineligible; eligible keys with missing or non-finite values
raise an error rather than changing the set.

Clopper-Pearson intervals are exact for independent trials with one success
probability. Directions of the same lens are not independent, so intervals pooled over
them are approximate. Population statements need one outcome per independently drawn
lens.

## Nonlinear fits

`fixed_template` fixes the subhalo at the trial mass and position with unit amplitude;
`local_search` frees its position; `freed` frees its position and mass within the given
support. These differ from the free-amplitude mismatch forecast. With the same pixels,
noise, PSF and free parameters, in the linear regime,

$$
q_\mathrm{fixed} = (2\hat a - 1) F, \qquad q_\mathrm{mismatch} - q_\mathrm{fixed} = (\hat a - 1)^2 F,
$$

so for matched noise-free data ($\hat a = 1$) a fixed-template fit approaches
`q_asimov`.

The smooth and subhalo models are not nested: a fixed subhalo is not a special case of
the smooth model, and a freed subhalo with a positive minimum mass is not either. So
$q_\mathrm{signed}$ can be negative, and no nested-model significance formula applies.

The default forecast also profiles a background offset that the fit does not include.
Comparisons with forecasts record this, along with any difference in pixel masks or
configuration.

Refinement accepts a role only when six checks pass: enough separated starts agree, a
tighter repeat agrees, the gradient at the best point is finite, the residual and scalar
likelihoods agree, a direct evaluation agrees, and the sampler's best point is
consistent. These checks establish a repeatable optimum. They do not require the
optimizer to report success or the gradient to be small, and they do not prove a global
maximum. `ClassificationRule.stationarity_tolerance` adds an optional bound on the
projected gradient.

## Gradients at special points

Refinement needs gradients of the likelihood. A few profiles have no well-defined
gradient at exact special values, and hwoslaps raises an error there instead of
returning a wrong gradient:

- an `Isothermal` lens whose free ellipticity is exactly zero;
- a circular `PowerLaw` lens with slope other than 2;
- an `Exponential` or `Sersic` (index ≥ 1) light profile with a grid point exactly at its
  centre.

Small non-zero ellipticities, fixed circular profiles and sampling without refinement
all work. When a role fails for this reason, offset the centre slightly or fix the
parameter.

## Nonlinear PSFs

Nonlinear fits use one model PSF kernel. A chromatic truth with several kernels can be
fitted with one common model kernel file, or a monochromatic model at one wavelength.
Fitted kernels must be at least 3 × 3 pixels.
