# Scientific conventions and limits

> Draft: the conventions below follow inspected producer source and named evidence. Final assembled release, example/convergence and remaining performance acceptance are incomplete.

## Units, coordinates and models

| Quantity | Convention |
|---|---|
| Subhalo mass | Solar masses; PointMass uses point mass, SIS/NFW use M200c |
| Truncated NFW mass | M200c of the parent NFW profile; total truncated mass is recorded separately |
| Image/hypothesis position | `(y, x)` arcseconds, origin at grid centre |
| Native detector pixels | Row zero has largest y; columns increase x |
| Hypothesis lattice | Increasing y/x coordinates with cell area equal to spacing squared |
| Pupil size, distance and wavelength | Metres; lensing distance records use Mpc where named |
| Wavefront coefficients | Nanometres of optical path difference |
| Light, sky and dark rates | Detected electrons per second; image rates are per native pixel |
| Data and noise standard deviation | ADU; gain is electrons per ADU |
| Exposure time | Total seconds across the recorded exposure count |
| Image orientation parameters | Degrees |
| q, significance, ratios and within-pixel variation | Dimensionless |

Lens mass families include Isothermal, PowerLaw and ExternalShear, with supported m=3/m=4 multipoles. Exponential, Sersic and Image components can supply source and lens light. Sersic normalization uses the backend polynomial for its half-light constant. Image structure is fixed internally; supported position, flux, size and rotation transformations can vary. These finite-dimensional fits do not provide unrestricted source reconstruction. Image rotation is profiled by default; reproduce a fixed paper template by fixing its rotation parameter.

NFW is untruncated. TNFW uses the BMO n=2 truncation shape with the parent-mass convention and a recorded total mass. Concentration/truncation prescriptions are configuration inputs; the Moline relation is restricted to its calibrated context at the lens redshift. No abundance law or additional line-of-sight population is inferred. A specified perturber realization is held fixed in both fit hypotheses.

A halo off the main lens plane is positioned in angular coordinates of its own plane. For a background halo those coordinates differ from its image-plane position. Truth and fitted plane assembly follow the same convention; support checks use the hypothesis's own-plane position.

Multipole bounds require sufficient positive-convergence conditions. At slope gamma and evaluated axis ratio q, the angular lower-bound condition is `sum(k_m) < 2*(3-gamma)*q**(gamma-1)/(1+q)`. Isothermal evaluates its circular clamp `q <= 0.99999`. The fit-box sufficient bound is tighter and uses the worst q/slope/amplitude corners. These sufficient conditions may reject a physically valid combination; the code refuses an invalid box rather than silently narrowing another parameter.

Cosmology can use a named flat realization or custom flat-LCDM inputs. The retained halo critical-density convention computes `H(z)=H0*sqrt(Om0*(1+z)**3+1-Om0)` and `rho_crit=3*H(z)**2/(8*pi*G)`. This convention omits radiation and massive-neutrino terms from this H(z), even when distance parameters carry them. It remains deferred correction C10. Named backend/astropy distance differences are recorded; a named realization is not silently replaced by default Planck15.

## Optics, photons and detector response

Optical providers propagate the pupil and wavefront into detector-integrated kernels. Odd kernel support and the optical Nyquist check are required. External detector kernels use their recorded angular sampling and bytes. A unit-sum kernel conserves its represented support; its captured fraction records how much physical power lies on that support when available. Kernel propagation and forecast host algebra run with one BLAS thread for reproducible operation order.

For chromatic imaging, each source/lens SED group uses photon-weighted node kernels. `SpectralWeights.rates` stores finite bin integrals with one whole-band offset, `log_rate_scale`. The mathematical measure is `rates * exp(log_rate_scale)` for `throughput*fnu*dlnlambda`; normalized weights are `rates/rates.sum()`. These common-scale values are arbitrary SED-shape integrals, not absolute detected electron counts. Normalization does not need an unrepresentable common exponential, and bins are not normalized independently. The one-node effective kernel preserves the input object and its bytes.

AB flux conversion, collecting area, throughput and photon energy determine absolute detected rates. Throughput acts once on light; a supplied detected sky/dark rate is not multiplied by throughput again. Exposure variance in electrons squared is `max(light_rate,0)*t + sky_rate*t + dark_rate*t + exposure_count*read_noise**2`; its ADU variance divides by gain squared. Mean and noisy draws preserve the recorded operation order. The detector model omits saturation, cosmic rays, interpixel capacitance and flat-field errors.

Chromatic node kernels are normalized on their finite support, preserving the monochromatic paper route. If physical node power P_k has captured fraction f_k, the effective represented kernel contains weighted terms P_k/f_k. This moves off-support power into the support with a wavelength-dependent effect. Chromatic accuracy therefore needs actual wavelength/support convergence checks. The authored six-product example comparison is unexecuted, and no blanket chromatic convergence claim is made. Tests `test_sed_bin_weights_follow_independent_photon_equations` and `test_arbitrary_table_normalization_preserves_relative_weights_and_effective_kernel` protect bin arithmetic/scaling; they do not establish accuracy for every scene.

## Native-pixel sampling diagnostic

The current imaging route bins oversampled light to native pixels, then convolves with a pixel-integrated kernel. It is exact for light constant within a native pixel and approximate otherwise. Increasing source oversampling cannot restore information removed by that binning.

`Observation.sampling` records the fiducial smooth scene's relative within-pixel variation per light group. Preparation computes it once; injected/noisy observations retain it. `MAX_NATIVE_SAMPLING_VARIATION=0.063` is the diagnostic threshold from `test_native_pixel_convolution_meets_the_sampling_budget`, whose tested budget is at most 1e-2 relative error in raw subhalo-template information against a finer reference.

The threshold is not enforced by `observe`. Recorded P1/P2/P4 variation is 0.10130071018061558, P3 is 0.13814890239604488, and the paper science scene is 0.02448039444770317. P1-P4 exceed the threshold and remain parity inputs. The oracle includes 48 lensed rows below threshold at 0.015 arcsec pixels, with maximum information error 0.006043. Some original 0.03 arcsec P1 rows change template information by about 0.9-1.6 percent despite small smooth-image errors. The paper scene's lower diagnostic is not a direct measurement of its exact subhalo-template error. Inspect the recorded diagnostic and run a finer sampling comparison for a new scene.

## Linear forecast and nuisance projection

The forecast profiles a linear-Gaussian amplitude/template model. Its nuisance span follows registry parameters, optional background and supported wavefront derivatives, with the configured steps and priors. Results remain conditional on that parametric source/light model and on their pixel mask, covariance, kernels and finite-difference choices.

`q_asimov` equals the profiled information for matched expected unit-amplitude data. `q_mismatch=a_hat**2*F` optimizes a free model-template amplitude on truth-PSF data. `q_spurious` optimizes an amplitude on the truth-minus-model PSF-bias residual. Large squared statistics of negative amplitudes do not indicate positive subhalos: detection requires finite `q >= T` and, for mismatch/spurious metrics, a finite amplitude greater than zero. T is a caller input. q and sqrt(q) are local model summaries without a calibrated blind-search false-positive interpretation.

The retained Gram pseudo-inverse cutoff is `max(rcond,p*eps)*max(abs(eigenvalues))`. It is invariant under uniform scaling of nuisance columns and depends on individual column units. The same span with `J=diag(1,1e-7)` can retain rank one/F=1, while `J=I` gives rank two/F=0; `test_gram_cutoff_depends_on_individual_column_units` holds that behavior. Provenance records nuisance rank and Gram condition number. Column scaling with transformed priors remains deferred C13, so no unit-invariant cutoff is claimed.

Dense forecast covariance is supported by the forecast whiteners. Simulation draws independent detector-pixel noise, and the nonlinear likelihood uses its diagonal noise map; `prepare_case` refuses dense covariance. `ForecastReference.noise_model` records diagonal or the covariance file identity.

## Areas, reach and PSF knowledge-error cohorts

Grid areas use detected-cell count times spacing squared over a specified selection. Missing lattice boundary nodes make boundary clipping unknown. Mass reach stays sampled/bracketed/bounded or nonmonotonic; it does not extrapolate. Value-linear and log-value interpolation are different choices in log mass.

Knowledge-error retention is the intersection with correct-PSF detections divided by reference area. The paper's R uses all mismatch detections in the selection, and F uses spurious detections in that selection. Full-domain spurious area is reported separately. Ratios are NaN below the reference count floor. Pair inputs must match comparison digest, geometry, mask, nuisance order and complete truth binding; only the model PSF may differ scientifically.

Tolerance reductions use one common keyed `(member,direction)` cohort across all amplitudes and both statistics. Eligibility comes from the reference floor. Missing keys or nonfinite eligible values refuse; ineligible keys remain counted as excluded. The submitted paper gate table reproduces 1.0, 5.0 and None under this cohort rule in `test_tolerance_reproduces_the_paper_gate_table` and the exact submitted-function control. That table reproduction does not silently substitute a different cohort for another study.

Clopper-Pearson intervals are exact for independent binomial trials at one success probability. Multiple directions of one system are clustered; their pooled result is labeled `clopper_pearson_nominal`. Population statements need one outcome per independently drawn system. First separation is strict lower-control-bound greater than upper-null-bound, with a separate flag for whether all sampled larger controls also separate. Neither a monotonic transition nor an abundance law is assumed.

## Nonlinear comparisons, refinement and gradients

`fixed_template` fits a physical halo at amplitude one; `local_search` varies position, and `freed` varies position/mass in its support. These alternatives differ from the free-amplitude mismatch forecast. With shared pixels, covariance, kernel and nuisance span in the linear regime, `q_fixed=(2*a_hat-1)*F` and `q_mismatch-q_fixed=(a_hat-1)**2*F`. Matched expected fixed-template comparisons approach q_asimov under those shared conditions. The exercised linear-regime keepers require F in [4,9], mismatch amplitude displacement in [.05,.2], and a 5-percent approximation bound; no unrecorded exact residual gap is claimed.

H0 and a fixed halo H1 are not generally nested, and a freed halo with positive minimum mass does not contain H0. `q_signed=2*(max(logL_H1)-max(logL_H0))` can be negative. `q_clipped` is separate. Nested-model significance formulas and global-optimum guarantees do not follow. The default forecast can additionally profile a background column that inference does not fit. Its conservative forecast-side interpretation is restricted to matched expected data; mismatch optimizes a different amplitude alternative and has no such general ordering. Different fit/forecast masks, comparison digests and unfitted background/wavefront nuisances are recorded diagnostics. Null controls and foreign-injection cases are excluded from like-for-like agreement, while physical truth-model studies retain their separate interpretation.

The six refinement gates check separated-start support, tighter-repeat agreement, finite retained gradient, residual/scalar consistency, direct likelihood consistency and sampler-incumbent consistency. They preserve historical repeatable-profile acceptance. They do not require solver success or a small projected gradient. `projected_gradient_linf` measures the unit-box half-chi-square gradient with outward active-bound components removed, using bound tolerance 1e-10. A caller's positive `stationarity_tolerance` adds that requirement; null preserves the historical rule.

Recorded B4 projected gradients are 0.0009443779736/3.0637435756e-10 for smooth/subhalo in B4a and 0.0003954641619/0.0001663024547 in B4b. These values demonstrate why accepted repeatability and stationary solutions are separate claims. `test_classification_rule_mapping_is_strict` and the classifier stationarity table protect that choice. Sampler maximum, weighted sampler quantiles and refined recovery are reported separately.

Inference uses project-owned differentials where circular light geometry is regular. The Exponential adapter supports a free circular ellipticity away from coincident source-centre samples. Its exactly coincident sample has a translation cusp and refuses the selected gradient path; finite value-only sampling remains available. The Sersic brightness differential at its centre is zero for n<1, while active geometry at n>=1 refuses; index-only directions remain distinguished. Ordinary nonround derivatives and backend primal values are preserved within their tested numerical bounds.

A free Isothermal ellipticity at exactly (0,0) refuses gradients because of the inherited circular clamp. Small nonzero ellipticities and entirely fixed circles are distinct supported cases. A free circular PowerLaw at slope different from two has the `(2-slope)*abs(ellipticity)` normalization cusp and refuses that gradient; slope two has a regular Cartesian differential away from an exactly coincident active mass-centre sample. These refusals are gradient-only, named errors, without silently freezing extra parameters or a global backend patch. The owner tests include `test_coincident_light_cusp_is_recorded_by_refinement`, `test_free_circular_isothermal_refuses_only_the_origin_gradient` and the circular-PowerLaw geometry tests.

## Evidence range and unresolved decisions

Paper CPU/GPU fixtures and the N1/B4 owners pin their specified data, likelihood/gradient/fit quantities. They verify those inputs and operation paths. They do not calibrate nonlinear search significance, prove every fit reaches a global optimum, or validate every new source/covariance/optical resolution. Nonlinear imaging needs one distinct model kernel; multi-kernel chromatic truth is retained. AutoArray's 1x1 model-PSF fit is refused; use at least 3x3 support. JAX lensing retains its implemented plane limits.

Deferred C10 keeps the critical-density H(z) convention above. C12 reserves halo-arithmetic unification because paper freed-fit values can move by one or two ulp. C13 reserves the scaled Gram cutoff. These are distinct from the evidenced project gradient/profile corrections already applied.

The original Sersic B5 target is steady throughput at least 0.80 of B3. The recorded case gives 3261.602649861229 nodes/s against confirmation 4245.567371029686 nodes/s, ratio 0.768237, and misses that target. Its finite scientific arrays and successful children do not turn the performance comparison into a pass. A proposed overhead optimization is unmeasured/unmerged, and human disposition remains pending. Current chromatic examples, final installation/export/configuration-reference checks and full release acceptance are also pending.
