# Halo values and singular pixels

All halo consumers use the same spherical deflection formulas. At every positive
radius, PointMass and SIS use their analytic profiles without an angular floor.
NFW and BMO n=1 TNFW use cancellation-free central expressions and regular series
near radius/scale-radius equal to one. The NFW/TNFW vector deflection has the
removable value zero at its centre. This finite value does not establish a finite
derivative with respect to the halo centre: geometry gradients at the true cusp
remain unsupported. A mass derivative at a fixed centre is a separate operation.

A PointMass or SIS has no central vector deflection value. Its exact central ray
therefore returns a nonfinite value and cannot be used as an ordinary brightness
sample. Rendering integrates lensed source brightness over the native pixel
rectangles instead. The `native-brightness-adaptive-v1` policy splits each
rectangle at the physical halo coordinate and uses interior Gauss nodes of order
8, 16, and 32 on each rectangle. The larger of the two successive differences
estimates its error. The algorithm subdivides the rectangle with the largest
error until the summed error meets relative tolerance `2e-6` plus absolute
tolerance `1e-10` in native brightness units. Refinement stops at 1024 leaf
rectangles or depth 20. The halo and detector stay fixed; unresolved integrals raise
`SingularIntegrationError`.

For a configuration with a PointMass/SIS hypothesis, every source-plane native
pixel receives its full brightness integral in both smooth and injected scenes
and in both nonlinear model roles. This shared operator prevents a change in
quadrature from appearing as a mass-independent halo signal. Fixed singular
perturbers also activate this policy. It costs more than ordinary uniform native
sampling; there is no finite-radius core or moving integration boundary.
Agreement of successive rules cannot exclude arbitrarily small source features
that fall between all initial nodes. Independent source and detector sampling
checks remain necessary for such inputs.

The corrected images enter the existing native PSF convolution. This procedure
integrates brightness before convolution; it does not average deflections or
introduce a softened halo. It currently supports two-plane scenes with the
singular halos in the lens plane. Other geometries raise a capability error.
An off-plane halo coordinate belongs to its own traced plane; it cannot be used
as an image-plane quadrature split without validating that mapping and the light
in every intervening plane. Ordinary off-plane NFW/TNFW forecasts retain their
own-plane coordinate convention.

Reference forecasts, truth scenes, and NumPy nonlinear fits use this operator.
A requested JAX forecast containing PointMass/SIS runs the shared reference
quadrature and records `kind=reference`, `requested_kind=jax`,
`jax_accelerated=false`, and the integration policy. It must not be counted as a
CUDA forecast benchmark. NFW/TNFW forecasts retain analytic JAX acceleration.
Traced JAX nonlinear PointMass/SIS fits raise
`SingularIntegrationCapabilityError`: their integrated likelihood derivatives
have not been validated. NumPy fits evaluate the integrated model directly.

Independent density-cylinder integrals, 100-digit closed forms, native
centre/edge/corner controls, positive-radius NumPy/JAX comparisons, and separate
CPU/CUDA kernel tests cover these distinctions. Pixel-integral convergence and
engine agreement remain separate checks.
