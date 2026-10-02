# Scientific conventions and limits

The engine forecasts how a declared subhalo changes a strongly lensed image and
how much of that perturbation can be absorbed by nuisance adjustments. The fast
statistic profiles a linear Gaussian likelihood; optional nonlinear comparisons
allow finite parameter changes under explicitly declared bounds.

- Coordinates are `(y, x)` arcseconds. Wavelength/pupil geometry are metres;
  wavefront coefficients are nanometres as documented by their provider.
- Source/sky/dark rates are electrons per second; detector data and noise maps
  are ADU, with explicit gain in electrons per ADU. Throughput scales source
  flux once; detected backgrounds are not multiplied by throughput again.
- NFW and SIS mass parameters use M200; PointMass is the point mass. The current
  NFW is untruncated and concentration prescription is explicit. There is no
  implicit subhalo abundance or line-of-sight population.
- Existing macro-lens/source implementations and Planck15 remain the supported
  physical models. Image templates allow complex morphology; internal source
  structure is fixed in the current targeted fits while supported transformations
  can adjust. This is not unrestricted source reconstruction.
- The current scene does not include lens-galaxy light or subtraction residuals.
  Its information reach is conditional on that foreground-free model.
- The optical provider is monochromatic and hexagonally segmented. External
  detector kernels permit another instrument's response without claiming its
  pupil, spectral integration or wavefront priors are modeled here.
- q and sqrt(q) are local model statistics. A threshold is an explicit screening
  choice; it is not a calibrated significance for a freed or blind search.
  No-subhalo expected-image susceptibility, sensitive area, noisy recovery, and
  empirical false-positive frequencies are different quantities.
- PSF quality and PSF knowledge error are distinct: change truth/model together
  for quality; hold truth fixed and change the fitted model for knowledge error.
  A mismatch can suppress or increase apparent signal. Positive amplitude and
  the declared threshold are both required for the mismatch detection rule.
- Area fractions need a declared domain and sampling. Censored/nonmonotone mass
  curves stay explicit. No target-selection fraction, mass threshold, PSF budget,
  population size or special source tier is a package rule.

Current numerical conventions and optimized kernels are protected by independent
physics/projection oracles, reference/JAX parity, real process tests, and explicit
input/output identity checks. New model families or statistical questions should
add their independent scientific proof at the owning boundary.
