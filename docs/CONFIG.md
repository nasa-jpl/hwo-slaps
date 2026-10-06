## top level

Engine configuration. Section-local rules run before these cross-section rules.

| key | value | default | unit | meaning |
|---|---|---|---|---|
| `run_name` | name of letters, digits, '.', '_', '-' | `run` |  | run label, excluded from scientific identity |
| `seed` | integer >= 0 | required |  | root of named scene-randomness streams; never a noise seed |
| `cosmology` | mapping, see `cosmology` | required |  | cosmology of the lensing scene |
| `scene` | mapping, see `scene` | required |  | smooth scene, subhalo hypothesis and injected halo |
| `psf` | mapping, see `psf` | required |  | truth and model point spread functions |
| `instrument` | mapping, see `instrument` | required |  | instrument and detector noise parameters |
| `observation` | mapping, see `observation` | required |  | exposure and sky background |
| `forecast` | null or mapping, see `forecast` | `null` |  | Fisher forecast inputs |

- every kernel file is sampled at scene.grid.pixel_scale_arcsec (X1)
- optical truth and model nodes are neither aliased nor under-resolved at the scene pixel scale (X4)
- wavelength_samples requires a bandpass and component SEDs (X5)
- AB source or sky inputs require a bandpass and a collecting area (X6)
- nuisance names resolve; wavefront modes need a model basis and existing segments (X7); an Einstein-radius ring needs exactly one radius (X8)

## cosmology

The cosmology of distances, critical densities and halo scales.

| key | value | default | unit | meaning |
|---|---|---|---|---|
| `name` | null or non-empty text | `null` |  | named astropy flat Lambda-CDM realization (Planck15 preserves the paper backend) |
| `flat_lcdm` | null or mapping, see `cosmology.flat_lcdm` | `null` |  | custom flat Lambda-CDM parameters |

- exactly one of `name`, `flat_lcdm` is set; write null to clear one

## cosmology.flat_lcdm

| key | value | default | unit | meaning |
|---|---|---|---|---|
| `H0` | number > 0 | required | km/s/Mpc | Hubble constant |
| `Om0` | number in (0, 1) | required |  | matter density fraction |
| `Ob0` | number >= 0 | `0.0` |  | baryon density fraction |
| `Tcmb0` | number >= 0 | `0.0` | K | CMB temperature |
| `Neff` | number >= 0 | `3.046` |  | effective number of neutrino species |
| `m_nu_eV` | list of 3 items, each number >= 0 | `[0.0, 0.0, 0.0]` | eV | three neutrino masses |

- Ob0 must not exceed Om0

## scene

The lensing scene.

| key | value | default | unit | meaning |
|---|---|---|---|---|
| `grid` | mapping, see `scene.grid` | required |  | image grid |
| `lens` | mapping, see `scene.lens` | required |  | lens galaxy |
| `source` | mapping, see `scene.source` | required |  | source galaxy |
| `subhalo` | mapping by `type` (PointMass, SIS, NFW, TNFW), see `scene.subhalo` | required |  | the detection hypothesis: forecasts and fits test this halo family |
| `injection` | null or mapping, see `scene.injection` | `null` |  | the subhalo hwoslaps simulate and batch simulate jobs inject |
| `perturbers` | mapping, see `scene.perturbers` | `{}` |  | fixed perturbing halos |

- source.redshift > lens.redshift
- component names unique within a galaxy and not reserved
- halo redshifts in (0, source.redshift); moline2017_eq7 only at the lens redshift and in its mass range
- radius: einstein_radius needs exactly one lens mass component with an Einstein radius

## scene.grid

| key | value | default | unit | meaning |
|---|---|---|---|---|
| `shape` | pair of positive integers | required | pixels | (ny, nx) |
| `pixel_scale_arcsec` | number > 0 | required | arcsec | pixel side |
| `over_sample_size` | integer >= 1 | required |  | sub-pixels per axis for light rendering (the paper used 4) |

## scene.lens

| key | value | default | unit | meaning |
|---|---|---|---|---|
| `redshift` | number > 0 | required |  | lens redshift |
| `mass` | named components (at least 1), see `scene.lens.mass.<name>` | required |  | mass components; this order is the nuisance order |
| `light` | named components, see `scene.lens.light.<name>` | `{}` |  | lens-plane light components |

## scene.lens.mass.<name> (type: Isothermal)

Selected by `type: Isothermal`.

A mass component; its parameters are profiled in registry order unless fixed.

Singular isothermal ellipsoid (al.mp.Isothermal); AutoGalaxy evaluates it at q <= 0.99999.

| key | value | default | unit | meaning |
|---|---|---|---|---|
| `centre` | pair, each number | required | arcsec | (y, x) centre |
| `einstein_radius` | number > 0 | required | arcsec | AutoLens Einstein radius of the SIE |
| `ell_comps` | pair (e1, e2) with sqrt(e1^2 + e2^2) < 1 | required |  | elliptical components (f sin 2 phi, f cos 2 phi), f = (1 - q) / (1 + q), q the axis ratio and phi the major-axis angle counter-clockwise from +x; hypot below 0.999 |
| `multipoles` | null or mapping, see `scene.lens.mass.<name>.multipoles` | `null` |  | multipoles linked to the base profile |

- hypot(ell_comps) < 0.999
- positive base plus multipole convergence

## scene.lens.mass.<name>.multipoles

| key | value | default | unit | meaning |
|---|---|---|---|---|
| `m3` | null or pair, each number | `null` |  | Cartesian third-order multipole components |
| `m4` | null or pair, each number | `null` |  | Cartesian fourth-order multipole components |

- at least one multipole order

## scene.lens.mass.<name> (type: PowerLaw)

Selected by `type: PowerLaw`.

A mass component; its parameters are profiled in registry order unless fixed.

Elliptical power-law lens mass.

| key | value | default | unit | meaning |
|---|---|---|---|---|
| `centre` | pair, each number | required | arcsec | (y, x) centre |
| `einstein_radius` | number > 0 | required | arcsec | AutoLens Einstein radius of the SIE |
| `ell_comps` | pair (e1, e2) with sqrt(e1^2 + e2^2) < 1 | required |  | elliptical components (f sin 2 phi, f cos 2 phi), f = (1 - q) / (1 + q), q the axis ratio and phi the major-axis angle counter-clockwise from +x; hypot below 0.999 |
| `multipoles` | null or mapping, see `scene.lens.mass.<name>.multipoles` | `null` |  | multipoles linked to the base profile |
| `slope` | number in (1, 3) | required |  | three-dimensional density slope |

- hypot(ell_comps) < 0.999
- positive base plus multipole convergence

## scene.lens.mass.<name> (type: ExternalShear)

Selected by `type: ExternalShear`.

A mass component; its parameters are profiled in registry order unless fixed.

| key | value | default | unit | meaning |
|---|---|---|---|---|
| `gamma_1` | number | required |  | external shear about the image-plane origin |
| `gamma_2` | number | required |  | external shear about the image-plane origin |

- hypot(shear) < 1

## scene.lens.light.<name> (type: Exponential)

Selected by `type: Exponential`.

A light component; its parameters are profiled in registry order unless fixed.

Exponential (Sersic n = 1) light profile (al.lp.Exponential).

| key | value | default | unit | meaning |
|---|---|---|---|---|
| `centre` | pair, each number | required | arcsec | (y, x) centre |
| `ell_comps` | pair (e1, e2) with sqrt(e1^2 + e2^2) < 1 | required |  | elliptical components (f sin 2 phi, f cos 2 phi), f = (1 - q) / (1 + q), q the axis ratio and phi the major-axis angle counter-clockwise from +x; hypot below 0.999 |
| `effective_radius` | number > 0 | required | arcsec | circularized half-light radius |
| `intensity` | null or number > 0 | `null` |  | surface brightness at the effective radius, detected e-/s per pixel sample |
| `flux` | null or mapping, see `scene.lens.light.<name>.flux` | `null` |  | intrinsic photometric normalization |
| `sed` | null or mapping by `kind` (flat_fnu, flat_flambda, power_law, table), see `scene.lens.light.<name>.sed` | `null` |  | fixed spectral shape |

- exactly one of `intensity`, `flux` is set; write null to clear one
- hypot(ell_comps) < 0.999
- reference-band flux requires an SED

## scene.lens.light.<name>.flux

| key | value | default | unit | meaning |
|---|---|---|---|---|
| `rate_e_per_s` | null or number > 0 | `null` | e-/s | intrinsic unlensed detected rate |
| `ab_mag` | null or number | `null` |  | AB magnitude in the instrument or reference band |
| `reference_band` | null or mapping by `kind` (top_hat, table, product), see `scene.lens.light.<name>.flux.reference_band` | `null` |  | band in which the AB magnitude is specified |

- exactly one of `rate_e_per_s`, `ab_mag` is set; write null to clear one
- reference band requires AB magnitude

## scene.lens.light.<name>.flux.reference_band (kind: top_hat)

Selected by `kind: top_hat`.

| key | value | default | unit | meaning |
|---|---|---|---|---|
| `min_nm` | number > 0 | required | nm | lower wavelength |
| `max_nm` | number > 0 | required | nm | upper wavelength |
| `throughput` | number in (0, 1] | required |  | electrons per entrance-pupil photon |
| `label` | null or non-empty text | `null` |  | bandpass label |

- wavelengths are ordered

## scene.lens.light.<name>.flux.reference_band (kind: table)

Selected by `kind: table`.

| key | value | default | unit | meaning |
|---|---|---|---|---|
| `path` | path to an existing .yaml, .yml, .csv or .npz file | required |  | spectral table |
| `wavelength_key` | non-empty text | required |  | wavelength column or array |
| `value_key` | non-empty text | required |  | value column or array |
| `wavelength_unit` | one of: nm, angstrom, um, m | required |  | wavelength unit |
| `power` | integer >= 1 | `1` |  | number of identical surfaces |
| `label` | null or non-empty text | `null` |  | bandpass label |
| `support_nm` | pair, each number > 0 | required | nm | band support (low, high) |

- support is ordered

## scene.lens.light.<name>.flux.reference_band (kind: product)

Selected by `kind: product`.

| key | value | default | unit | meaning |
|---|---|---|---|---|
| `support_nm` | pair, each number > 0 | required | nm | band support (low, high) |
| `factors` | list of at least 1 items, each mapping selected by kind: table, top_hat, constant, see `scene.lens.light.<name>.flux.reference_band.factors[i]` | required |  | response factors |
| `label` | null or non-empty text | `null` |  | bandpass label |

- support is ordered and contained in each top-hat factor

## scene.lens.light.<name>.flux.reference_band.factors[i] (kind: table)

Selected by `kind: table`.

| key | value | default | unit | meaning |
|---|---|---|---|---|
| `path` | path to an existing .yaml, .yml, .csv or .npz file | required |  | spectral table |
| `wavelength_key` | non-empty text | required |  | wavelength column or array |
| `value_key` | non-empty text | required |  | value column or array |
| `wavelength_unit` | one of: nm, angstrom, um, m | required |  | wavelength unit |
| `power` | integer >= 1 | `1` |  | number of identical surfaces |
| `label` | null or non-empty text | `null` |  | bandpass label |

## scene.lens.light.<name>.flux.reference_band.factors[i] (kind: top_hat)

Selected by `kind: top_hat`.

| key | value | default | unit | meaning |
|---|---|---|---|---|
| `min_nm` | number > 0 | required | nm | lower wavelength |
| `max_nm` | number > 0 | required | nm | upper wavelength |
| `throughput` | number in (0, 1] | required |  | electrons per entrance-pupil photon |
| `label` | null or non-empty text | `null` |  | bandpass label |

- wavelengths are ordered

## scene.lens.light.<name>.flux.reference_band.factors[i] (kind: constant)

Selected by `kind: constant`.

| key | value | default | unit | meaning |
|---|---|---|---|---|
| `value` | number in (0, 1] | required |  | constant factor |

## scene.lens.light.<name>.sed (kind: flat_fnu)

Selected by `kind: flat_fnu`.

No keys.

## scene.lens.light.<name>.sed (kind: flat_flambda)

Selected by `kind: flat_flambda`.

No keys.

## scene.lens.light.<name>.sed (kind: power_law)

Selected by `kind: power_law`.

| key | value | default | unit | meaning |
|---|---|---|---|---|
| `index` | number | required |  | index of f_nu proportional to nu**index |

## scene.lens.light.<name>.sed (kind: table)

Selected by `kind: table`.

| key | value | default | unit | meaning |
|---|---|---|---|---|
| `path` | path to an existing .yaml, .yml, .csv or .npz file | required |  | spectral table |
| `wavelength_key` | non-empty text | required |  | wavelength column or array |
| `value_key` | non-empty text | required |  | value column or array |
| `wavelength_unit` | one of: nm, angstrom, um, m | required |  | wavelength unit |
| `quantity` | one of: fnu, flambda | required |  | spectral density convention |
| `frame` | one of: observed, rest | `observed` |  | wavelength frame |

## scene.lens.light.<name> (type: Sersic)

Selected by `type: Sersic`.

A light component; its parameters are profiled in registry order unless fixed.

Sersic light with circularized effective radius.

| key | value | default | unit | meaning |
|---|---|---|---|---|
| `centre` | pair, each number | required | arcsec | (y, x) centre |
| `ell_comps` | pair (e1, e2) with sqrt(e1^2 + e2^2) < 1 | required |  | elliptical components (f sin 2 phi, f cos 2 phi), f = (1 - q) / (1 + q), q the axis ratio and phi the major-axis angle counter-clockwise from +x; hypot below 0.999 |
| `effective_radius` | number > 0 | required | arcsec | circularized half-light radius |
| `intensity` | null or number > 0 | `null` |  | surface brightness at the effective radius, detected e-/s per pixel sample |
| `sersic_index` | number in [0.36, 8] | required |  | Sersic index of the rendered Ciotti-Bertin series |
| `flux` | null or mapping, see `scene.lens.light.<name>.flux` | `null` |  | intrinsic photometric normalization |
| `sed` | null or mapping by `kind` (flat_fnu, flat_flambda, power_law, table), see `scene.lens.light.<name>.sed` | `null` |  | fixed spectral shape |

- exactly one of `intensity`, `flux` is set; write null to clear one
- hypot(ell_comps) < 0.999
- reference-band flux requires an SED

## scene.lens.light.<name> (type: Image)

Selected by `type: Image`.

A light component; its parameters are profiled in registry order unless fixed.

Pixelized source: a unit-integral asset evaluated by bilinear interpolation with a one-pixel zero pad.

| key | value | default | unit | meaning |
|---|---|---|---|---|
| `asset_path` | path to an existing .npz file | required |  | prepared image asset (format version 1) |
| `centre` | pair, each number | required | arcsec | (y, x) centre |
| `rotation_deg` | number | `0.0` | deg | counter-clockwise rotation of the image on the sky |
| `total_flux` | null or number > 0 | `null` |  | integral of the image at unit flux and size scales, e-/s per pixel sample times arcsec^2 |
| `flux_scale` | number > 0 | `1.0` |  | brightness multiplier |
| `size_scale` | number > 0 | `1.0` |  | magnification of the image at fixed surface brightness |
| `flux` | null or mapping, see `scene.lens.light.<name>.flux` | `null` |  | intrinsic photometric normalization |
| `sed` | null or mapping by `kind` (flat_fnu, flat_flambda, power_law, table), see `scene.lens.light.<name>.sed` | `null` |  | fixed spectral shape |

- exactly one of `total_flux`, `flux` is set; write null to clear one
- reference-band flux requires an SED

## scene.source

| key | value | default | unit | meaning |
|---|---|---|---|---|
| `redshift` | number > 0 | required |  | source redshift, behind the lens |
| `light` | named components (at least 1), see `scene.source.light.<name>` | required |  | source light components |

## scene.source.light.<name> (type: Exponential)

Selected by `type: Exponential`.

A light component; its parameters are profiled in registry order unless fixed.

Exponential (Sersic n = 1) light profile (al.lp.Exponential).

| key | value | default | unit | meaning |
|---|---|---|---|---|
| `centre` | pair, each number | required | arcsec | (y, x) centre |
| `ell_comps` | pair (e1, e2) with sqrt(e1^2 + e2^2) < 1 | required |  | elliptical components (f sin 2 phi, f cos 2 phi), f = (1 - q) / (1 + q), q the axis ratio and phi the major-axis angle counter-clockwise from +x; hypot below 0.999 |
| `effective_radius` | number > 0 | required | arcsec | circularized half-light radius |
| `intensity` | null or number > 0 | `null` |  | surface brightness at the effective radius, detected e-/s per pixel sample |
| `flux` | null or mapping, see `scene.source.light.<name>.flux` | `null` |  | intrinsic photometric normalization |
| `sed` | null or mapping by `kind` (flat_fnu, flat_flambda, power_law, table), see `scene.source.light.<name>.sed` | `null` |  | fixed spectral shape |

- exactly one of `intensity`, `flux` is set; write null to clear one
- hypot(ell_comps) < 0.999
- reference-band flux requires an SED

## scene.source.light.<name>.flux

| key | value | default | unit | meaning |
|---|---|---|---|---|
| `rate_e_per_s` | null or number > 0 | `null` | e-/s | intrinsic unlensed detected rate |
| `ab_mag` | null or number | `null` |  | AB magnitude in the instrument or reference band |
| `reference_band` | null or mapping by `kind` (top_hat, table, product), see `scene.source.light.<name>.flux.reference_band` | `null` |  | band in which the AB magnitude is specified |

- exactly one of `rate_e_per_s`, `ab_mag` is set; write null to clear one
- reference band requires AB magnitude

## scene.source.light.<name>.flux.reference_band (kind: top_hat)

Selected by `kind: top_hat`.

| key | value | default | unit | meaning |
|---|---|---|---|---|
| `min_nm` | number > 0 | required | nm | lower wavelength |
| `max_nm` | number > 0 | required | nm | upper wavelength |
| `throughput` | number in (0, 1] | required |  | electrons per entrance-pupil photon |
| `label` | null or non-empty text | `null` |  | bandpass label |

- wavelengths are ordered

## scene.source.light.<name>.flux.reference_band (kind: table)

Selected by `kind: table`.

| key | value | default | unit | meaning |
|---|---|---|---|---|
| `path` | path to an existing .yaml, .yml, .csv or .npz file | required |  | spectral table |
| `wavelength_key` | non-empty text | required |  | wavelength column or array |
| `value_key` | non-empty text | required |  | value column or array |
| `wavelength_unit` | one of: nm, angstrom, um, m | required |  | wavelength unit |
| `power` | integer >= 1 | `1` |  | number of identical surfaces |
| `label` | null or non-empty text | `null` |  | bandpass label |
| `support_nm` | pair, each number > 0 | required | nm | band support (low, high) |

- support is ordered

## scene.source.light.<name>.flux.reference_band (kind: product)

Selected by `kind: product`.

| key | value | default | unit | meaning |
|---|---|---|---|---|
| `support_nm` | pair, each number > 0 | required | nm | band support (low, high) |
| `factors` | list of at least 1 items, each mapping selected by kind: table, top_hat, constant, see `scene.source.light.<name>.flux.reference_band.factors[i]` | required |  | response factors |
| `label` | null or non-empty text | `null` |  | bandpass label |

- support is ordered and contained in each top-hat factor

## scene.source.light.<name>.flux.reference_band.factors[i] (kind: table)

Selected by `kind: table`.

| key | value | default | unit | meaning |
|---|---|---|---|---|
| `path` | path to an existing .yaml, .yml, .csv or .npz file | required |  | spectral table |
| `wavelength_key` | non-empty text | required |  | wavelength column or array |
| `value_key` | non-empty text | required |  | value column or array |
| `wavelength_unit` | one of: nm, angstrom, um, m | required |  | wavelength unit |
| `power` | integer >= 1 | `1` |  | number of identical surfaces |
| `label` | null or non-empty text | `null` |  | bandpass label |

## scene.source.light.<name>.flux.reference_band.factors[i] (kind: top_hat)

Selected by `kind: top_hat`.

| key | value | default | unit | meaning |
|---|---|---|---|---|
| `min_nm` | number > 0 | required | nm | lower wavelength |
| `max_nm` | number > 0 | required | nm | upper wavelength |
| `throughput` | number in (0, 1] | required |  | electrons per entrance-pupil photon |
| `label` | null or non-empty text | `null` |  | bandpass label |

- wavelengths are ordered

## scene.source.light.<name>.flux.reference_band.factors[i] (kind: constant)

Selected by `kind: constant`.

| key | value | default | unit | meaning |
|---|---|---|---|---|
| `value` | number in (0, 1] | required |  | constant factor |

## scene.source.light.<name>.sed (kind: flat_fnu)

Selected by `kind: flat_fnu`.

No keys.

## scene.source.light.<name>.sed (kind: flat_flambda)

Selected by `kind: flat_flambda`.

No keys.

## scene.source.light.<name>.sed (kind: power_law)

Selected by `kind: power_law`.

| key | value | default | unit | meaning |
|---|---|---|---|---|
| `index` | number | required |  | index of f_nu proportional to nu**index |

## scene.source.light.<name>.sed (kind: table)

Selected by `kind: table`.

| key | value | default | unit | meaning |
|---|---|---|---|---|
| `path` | path to an existing .yaml, .yml, .csv or .npz file | required |  | spectral table |
| `wavelength_key` | non-empty text | required |  | wavelength column or array |
| `value_key` | non-empty text | required |  | value column or array |
| `wavelength_unit` | one of: nm, angstrom, um, m | required |  | wavelength unit |
| `quantity` | one of: fnu, flambda | required |  | spectral density convention |
| `frame` | one of: observed, rest | `observed` |  | wavelength frame |

## scene.source.light.<name> (type: Sersic)

Selected by `type: Sersic`.

A light component; its parameters are profiled in registry order unless fixed.

Sersic light with circularized effective radius.

| key | value | default | unit | meaning |
|---|---|---|---|---|
| `centre` | pair, each number | required | arcsec | (y, x) centre |
| `ell_comps` | pair (e1, e2) with sqrt(e1^2 + e2^2) < 1 | required |  | elliptical components (f sin 2 phi, f cos 2 phi), f = (1 - q) / (1 + q), q the axis ratio and phi the major-axis angle counter-clockwise from +x; hypot below 0.999 |
| `effective_radius` | number > 0 | required | arcsec | circularized half-light radius |
| `intensity` | null or number > 0 | `null` |  | surface brightness at the effective radius, detected e-/s per pixel sample |
| `sersic_index` | number in [0.36, 8] | required |  | Sersic index of the rendered Ciotti-Bertin series |
| `flux` | null or mapping, see `scene.source.light.<name>.flux` | `null` |  | intrinsic photometric normalization |
| `sed` | null or mapping by `kind` (flat_fnu, flat_flambda, power_law, table), see `scene.source.light.<name>.sed` | `null` |  | fixed spectral shape |

- exactly one of `intensity`, `flux` is set; write null to clear one
- hypot(ell_comps) < 0.999
- reference-band flux requires an SED

## scene.source.light.<name> (type: Image)

Selected by `type: Image`.

A light component; its parameters are profiled in registry order unless fixed.

Pixelized source: a unit-integral asset evaluated by bilinear interpolation with a one-pixel zero pad.

| key | value | default | unit | meaning |
|---|---|---|---|---|
| `asset_path` | path to an existing .npz file | required |  | prepared image asset (format version 1) |
| `centre` | pair, each number | required | arcsec | (y, x) centre |
| `rotation_deg` | number | `0.0` | deg | counter-clockwise rotation of the image on the sky |
| `total_flux` | null or number > 0 | `null` |  | integral of the image at unit flux and size scales, e-/s per pixel sample times arcsec^2 |
| `flux_scale` | number > 0 | `1.0` |  | brightness multiplier |
| `size_scale` | number > 0 | `1.0` |  | magnification of the image at fixed surface brightness |
| `flux` | null or mapping, see `scene.source.light.<name>.flux` | `null` |  | intrinsic photometric normalization |
| `sed` | null or mapping by `kind` (flat_fnu, flat_flambda, power_law, table), see `scene.source.light.<name>.sed` | `null` |  | fixed spectral shape |

- exactly one of `total_flux`, `flux` is set; write null to clear one
- reference-band flux requires an SED

## scene.subhalo (type: PointMass)

Selected by `type: PointMass`.

PointMass (point_mass), SIS or NFW (M200c), or TNFW (M200c_parent) halo.

| key | value | default | unit | meaning |
|---|---|---|---|---|
| `redshift` | null or number > 0 | `null` |  | redshift of the hypothesis; null is the lens redshift. Positions of an off-plane halo are angular positions in its own plane |

## scene.subhalo (type: SIS)

Selected by `type: SIS`.

PointMass (point_mass), SIS or NFW (M200c), or TNFW (M200c_parent) halo.

| key | value | default | unit | meaning |
|---|---|---|---|---|
| `redshift` | null or number > 0 | `null` |  | redshift of the hypothesis; null is the lens redshift. Positions of an off-plane halo are angular positions in its own plane |

## scene.subhalo (type: NFW)

Selected by `type: NFW`.

PointMass (point_mass), SIS or NFW (M200c), or TNFW (M200c_parent) halo.

| key | value | default | unit | meaning |
|---|---|---|---|---|
| `concentration` | mapping by `kind` (moline2017_eq7, power_law, fixed), see `scene.subhalo.concentration` | required |  | concentration-mass relation |
| `redshift` | null or number > 0 | `null` |  | redshift of the hypothesis; null is the lens redshift. Positions of an off-plane halo are angular positions in its own plane |

## scene.subhalo.concentration (kind: moline2017_eq7)

Selected by `kind: moline2017_eq7`.

Concentration-mass relation of an NFW halo.

Moline et al. (2017), eq. 7: subhalos at the lens redshift, M200 in [1e6, 1e12] Msun.

| key | value | default | unit | meaning |
|---|---|---|---|---|
| `x_sub` | number in (0, 1.5] | required |  | host-centric radius of the subhalo in units of the host virial radius |
| `h` | null or number > 0 | `null` |  | reduced Hubble constant of the relation's mass unit 1e8 / h Msun; null takes H0 / 100 of the cosmology |

## scene.subhalo.concentration (kind: power_law)

Selected by `kind: power_law`.

Concentration-mass relation of an NFW halo.

c200 = c0 (M200 / mass_pivot_msun)^mass_slope (1 + z)^redshift_slope.

| key | value | default | unit | meaning |
|---|---|---|---|---|
| `c0` | number > 0 | required |  | concentration at the pivot mass and z = 0 |
| `mass_pivot_msun` | number > 0 | required | Msun | pivot mass |
| `mass_slope` | number | required |  | exponent of M200 / mass_pivot_msun |
| `redshift_slope` | number | required |  | exponent of 1 + z |

## scene.subhalo.concentration (kind: fixed)

Selected by `kind: fixed`.

Concentration-mass relation of an NFW halo.

| key | value | default | unit | meaning |
|---|---|---|---|---|
| `value` | number > 0 | required |  | concentration c200 |

## scene.subhalo (type: TNFW)

Selected by `type: TNFW`.

PointMass (point_mass), SIS or NFW (M200c), or TNFW (M200c_parent) halo.

| key | value | default | unit | meaning |
|---|---|---|---|---|
| `concentration` | mapping by `kind` (moline2017_eq7, power_law, fixed), see `scene.subhalo.concentration` | required |  | parent NFW concentration-mass relation |
| `truncation` | mapping by `kind` (tau, overdensity), see `scene.subhalo.truncation` | required |  | BMO truncation radius |
| `redshift` | null or number > 0 | `null` |  | redshift of the hypothesis; null is the lens redshift. Positions of an off-plane halo are angular positions in its own plane |

## scene.subhalo.truncation (kind: tau)

Selected by `kind: tau`.

| key | value | default | unit | meaning |
|---|---|---|---|---|
| `tau` | number > 0 | required |  | r_t / r_s |

## scene.subhalo.truncation (kind: overdensity)

Selected by `kind: overdensity`.

| key | value | default | unit | meaning |
|---|---|---|---|---|
| `overdensity` | number > 0 | required |  | parent NFW mean enclosed density in units of rho_crit |

## scene.injection

| key | value | default | unit | meaning |
|---|---|---|---|---|
| `mass_msun` | number > 0 | required | Msun | subhalo mass in the hypothesis mass definition |
| `position` | mapping by `kind` (direct, angle, random), see `scene.injection.position` | required |  | placement of the subhalo |

## scene.injection.position (kind: direct)

Selected by `kind: direct`.

Where the injected subhalo sits.

| key | value | default | unit | meaning |
|---|---|---|---|---|
| `centre` | pair, each number | required | arcsec | (y, x) position of the subhalo in its own plane |

## scene.injection.position (kind: angle)

Selected by `kind: angle`.

Where the injected subhalo sits.

| key | value | default | unit | meaning |
|---|---|---|---|---|
| `angle_deg` | number | required | deg | position angle about the lens centre, from +x toward +y |
| `radius` | one of: einstein_radius, critical_curve or number > 0 | `einstein_radius` | arcsec | einstein_radius, critical_curve, or a radius about the lens centre |
| `offset_arcsec` | number | `0.0` | arcsec | added to the radius |

## scene.injection.position (kind: random)

Selected by `kind: random`.

Where the injected subhalo sits.

| key | value | default | unit | meaning |
|---|---|---|---|---|
| `radius` | one of: einstein_radius, critical_curve or number > 0 | `einstein_radius` | arcsec | einstein_radius, critical_curve, or a radius about the lens centre |
| `scatter_arcsec` | number > 0 | required | arcsec | half width of the uniform radial offset |

## scene.perturbers

| key | value | default | unit | meaning |
|---|---|---|---|---|
| `halos` | list, each mapping selected by type: PointMass, SIS, NFW, TNFW, see `scene.perturbers.halos[i]` | `[]` |  | fixed perturbing halos, in order |
| `populations` | list, each mapping selected by type: PointMass, SIS, NFW, TNFW, see `scene.perturbers.populations[i]` | `[]` |  | drawn halo populations, in order |

## scene.perturbers.halos[i] (type: PointMass)

Selected by `type: PointMass`.

PointMass (point_mass), SIS or NFW (M200c), or TNFW (M200c_parent) halo.

| key | value | default | unit | meaning |
|---|---|---|---|---|
| `mass_msun` | number > 0 | required | Msun | halo mass in its mass definition |
| `centre` | pair, each number | required | arcsec | (y, x) position in the halo's own plane |
| `redshift` | null or number > 0 | `null` |  | halo redshift; null is the lens redshift |

## scene.perturbers.halos[i] (type: SIS)

Selected by `type: SIS`.

PointMass (point_mass), SIS or NFW (M200c), or TNFW (M200c_parent) halo.

| key | value | default | unit | meaning |
|---|---|---|---|---|
| `mass_msun` | number > 0 | required | Msun | halo mass in its mass definition |
| `centre` | pair, each number | required | arcsec | (y, x) position in the halo's own plane |
| `redshift` | null or number > 0 | `null` |  | halo redshift; null is the lens redshift |

## scene.perturbers.halos[i] (type: NFW)

Selected by `type: NFW`.

PointMass (point_mass), SIS or NFW (M200c), or TNFW (M200c_parent) halo.

| key | value | default | unit | meaning |
|---|---|---|---|---|
| `concentration` | mapping by `kind` (moline2017_eq7, power_law, fixed), see `scene.perturbers.halos[i].concentration` | required |  | concentration-mass relation |
| `mass_msun` | number > 0 | required | Msun | halo mass in its mass definition |
| `centre` | pair, each number | required | arcsec | (y, x) position in the halo's own plane |
| `redshift` | null or number > 0 | `null` |  | halo redshift; null is the lens redshift |

## scene.perturbers.halos[i].concentration (kind: moline2017_eq7)

Selected by `kind: moline2017_eq7`.

Concentration-mass relation of an NFW halo.

Moline et al. (2017), eq. 7: subhalos at the lens redshift, M200 in [1e6, 1e12] Msun.

| key | value | default | unit | meaning |
|---|---|---|---|---|
| `x_sub` | number in (0, 1.5] | required |  | host-centric radius of the subhalo in units of the host virial radius |
| `h` | null or number > 0 | `null` |  | reduced Hubble constant of the relation's mass unit 1e8 / h Msun; null takes H0 / 100 of the cosmology |

## scene.perturbers.halos[i].concentration (kind: power_law)

Selected by `kind: power_law`.

Concentration-mass relation of an NFW halo.

c200 = c0 (M200 / mass_pivot_msun)^mass_slope (1 + z)^redshift_slope.

| key | value | default | unit | meaning |
|---|---|---|---|---|
| `c0` | number > 0 | required |  | concentration at the pivot mass and z = 0 |
| `mass_pivot_msun` | number > 0 | required | Msun | pivot mass |
| `mass_slope` | number | required |  | exponent of M200 / mass_pivot_msun |
| `redshift_slope` | number | required |  | exponent of 1 + z |

## scene.perturbers.halos[i].concentration (kind: fixed)

Selected by `kind: fixed`.

Concentration-mass relation of an NFW halo.

| key | value | default | unit | meaning |
|---|---|---|---|---|
| `value` | number > 0 | required |  | concentration c200 |

## scene.perturbers.halos[i] (type: TNFW)

Selected by `type: TNFW`.

PointMass (point_mass), SIS or NFW (M200c), or TNFW (M200c_parent) halo.

| key | value | default | unit | meaning |
|---|---|---|---|---|
| `concentration` | mapping by `kind` (moline2017_eq7, power_law, fixed), see `scene.perturbers.halos[i].concentration` | required |  | parent NFW concentration-mass relation |
| `truncation` | mapping by `kind` (tau, overdensity), see `scene.perturbers.halos[i].truncation` | required |  | BMO truncation radius |
| `mass_msun` | number > 0 | required | Msun | halo mass in its mass definition |
| `centre` | pair, each number | required | arcsec | (y, x) position in the halo's own plane |
| `redshift` | null or number > 0 | `null` |  | halo redshift; null is the lens redshift |

## scene.perturbers.halos[i].truncation (kind: tau)

Selected by `kind: tau`.

| key | value | default | unit | meaning |
|---|---|---|---|---|
| `tau` | number > 0 | required |  | r_t / r_s |

## scene.perturbers.halos[i].truncation (kind: overdensity)

Selected by `kind: overdensity`.

| key | value | default | unit | meaning |
|---|---|---|---|---|
| `overdensity` | number > 0 | required |  | parent NFW mean enclosed density in units of rho_crit |

## scene.perturbers.populations[i] (type: PointMass)

Selected by `type: PointMass`.

PointMass (point_mass), SIS or NFW (M200c), or TNFW (M200c_parent) halo.

| key | value | default | unit | meaning |
|---|---|---|---|---|
| `mass_function` | mapping by `kind` (power_law), see `scene.perturbers.populations[i].mass_function` | required |  | population mass distribution and count |
| `spatial` | mapping by `kind` (uniform_disk, uniform_annulus), see `scene.perturbers.populations[i].spatial` | required |  | population positions in their own plane |
| `redshift` | null or number > 0 | `null` |  | population redshift; null is the lens redshift |

## scene.perturbers.populations[i].mass_function (kind: power_law)

Selected by `kind: power_law`.

| key | value | default | unit | meaning |
|---|---|---|---|---|
| `slope` | number | required |  | slope of dN/dM proportional to M**slope |
| `mass_min_msun` | number > 0 | required | Msun | minimum population mass |
| `mass_max_msun` | number > 0 | required | Msun | maximum population mass |
| `count` | null or integer >= 0 | `null` |  | fixed number of halos |
| `expected_count` | null or number > 0 | `null` |  | mean Poisson count |

- exactly one of `count`, `expected_count` is set; write null to clear one
- mass_max_msun exceeds mass_min_msun

## scene.perturbers.populations[i].spatial (kind: uniform_disk)

Selected by `kind: uniform_disk`.

| key | value | default | unit | meaning |
|---|---|---|---|---|
| `radius_arcsec` | number > 0 | required | arcsec | disc radius about the configured lens centre |

## scene.perturbers.populations[i].spatial (kind: uniform_annulus)

Selected by `kind: uniform_annulus`.

| key | value | default | unit | meaning |
|---|---|---|---|---|
| `inner_arcsec` | number >= 0 | required | arcsec | inner radius about the configured lens centre |
| `outer_arcsec` | number > 0 | required | arcsec | outer radius |

- outer_arcsec exceeds inner_arcsec

## scene.perturbers.populations[i] (type: SIS)

Selected by `type: SIS`.

PointMass (point_mass), SIS or NFW (M200c), or TNFW (M200c_parent) halo.

| key | value | default | unit | meaning |
|---|---|---|---|---|
| `mass_function` | mapping by `kind` (power_law), see `scene.perturbers.populations[i].mass_function` | required |  | population mass distribution and count |
| `spatial` | mapping by `kind` (uniform_disk, uniform_annulus), see `scene.perturbers.populations[i].spatial` | required |  | population positions in their own plane |
| `redshift` | null or number > 0 | `null` |  | population redshift; null is the lens redshift |

## scene.perturbers.populations[i] (type: NFW)

Selected by `type: NFW`.

PointMass (point_mass), SIS or NFW (M200c), or TNFW (M200c_parent) halo.

| key | value | default | unit | meaning |
|---|---|---|---|---|
| `concentration` | mapping by `kind` (moline2017_eq7, power_law, fixed), see `scene.perturbers.populations[i].concentration` | required |  | concentration-mass relation |
| `mass_function` | mapping by `kind` (power_law), see `scene.perturbers.populations[i].mass_function` | required |  | population mass distribution and count |
| `spatial` | mapping by `kind` (uniform_disk, uniform_annulus), see `scene.perturbers.populations[i].spatial` | required |  | population positions in their own plane |
| `redshift` | null or number > 0 | `null` |  | population redshift; null is the lens redshift |

## scene.perturbers.populations[i].concentration (kind: moline2017_eq7)

Selected by `kind: moline2017_eq7`.

Concentration-mass relation of an NFW halo.

Moline et al. (2017), eq. 7: subhalos at the lens redshift, M200 in [1e6, 1e12] Msun.

| key | value | default | unit | meaning |
|---|---|---|---|---|
| `x_sub` | number in (0, 1.5] | required |  | host-centric radius of the subhalo in units of the host virial radius |
| `h` | null or number > 0 | `null` |  | reduced Hubble constant of the relation's mass unit 1e8 / h Msun; null takes H0 / 100 of the cosmology |

## scene.perturbers.populations[i].concentration (kind: power_law)

Selected by `kind: power_law`.

Concentration-mass relation of an NFW halo.

c200 = c0 (M200 / mass_pivot_msun)^mass_slope (1 + z)^redshift_slope.

| key | value | default | unit | meaning |
|---|---|---|---|---|
| `c0` | number > 0 | required |  | concentration at the pivot mass and z = 0 |
| `mass_pivot_msun` | number > 0 | required | Msun | pivot mass |
| `mass_slope` | number | required |  | exponent of M200 / mass_pivot_msun |
| `redshift_slope` | number | required |  | exponent of 1 + z |

## scene.perturbers.populations[i].concentration (kind: fixed)

Selected by `kind: fixed`.

Concentration-mass relation of an NFW halo.

| key | value | default | unit | meaning |
|---|---|---|---|---|
| `value` | number > 0 | required |  | concentration c200 |

## scene.perturbers.populations[i] (type: TNFW)

Selected by `type: TNFW`.

PointMass (point_mass), SIS or NFW (M200c), or TNFW (M200c_parent) halo.

| key | value | default | unit | meaning |
|---|---|---|---|---|
| `concentration` | mapping by `kind` (moline2017_eq7, power_law, fixed), see `scene.perturbers.populations[i].concentration` | required |  | parent NFW concentration-mass relation |
| `truncation` | mapping by `kind` (tau, overdensity), see `scene.perturbers.populations[i].truncation` | required |  | BMO truncation radius |
| `mass_function` | mapping by `kind` (power_law), see `scene.perturbers.populations[i].mass_function` | required |  | population mass distribution and count |
| `spatial` | mapping by `kind` (uniform_disk, uniform_annulus), see `scene.perturbers.populations[i].spatial` | required |  | population positions in their own plane |
| `redshift` | null or number > 0 | `null` |  | population redshift; null is the lens redshift |

## scene.perturbers.populations[i].truncation (kind: tau)

Selected by `kind: tau`.

| key | value | default | unit | meaning |
|---|---|---|---|---|
| `tau` | number > 0 | required |  | r_t / r_s |

## scene.perturbers.populations[i].truncation (kind: overdensity)

Selected by `kind: overdensity`.

| key | value | default | unit | meaning |
|---|---|---|---|---|
| `overdensity` | number > 0 | required |  | parent NFW mean enclosed density in units of rho_crit |

## psf

| key | value | default | unit | meaning |
|---|---|---|---|---|
| `truth` | mapping by `kind` (kernel, kernel_cube, optical), see `psf.truth` | required |  | the PSF that makes the data |
| `model` | mapping by `kind` (matched, kernel, optical, wavefront, knowledge_error, monochromatic), see `psf.model` | `{}` |  | the PSF the analysis assumes |

- optical, wavefront, knowledge_error and monochromatic models need an optical truth; segment hexikes and segment draws need a hex-segmented pupil with those segments

## psf.truth (kind: kernel)

Selected by `kind: kernel`.

| key | value | default | unit | meaning |
|---|---|---|---|---|
| `path` | path to an existing .npy or .npz file | required |  | detector kernel file |
| `array_key` | null or non-empty text | `null` |  | member of a .npz file (null reads kernel); null for .npy |
| `pixel_scale_arcsec` | number > 0 | required | arcsec | angular sampling of the kernel |
| `normalize` | true or false | `true` |  | divide the kernel by its sum; false requires a sum within 1e-10 of one |
| `file_sha256` | null or lowercase SHA-256 hex digest | `null` |  | SHA-256 of the file bytes, checked before the file is read |

- a .npy file takes no array_key

## psf.truth (kind: kernel_cube)

Selected by `kind: kernel_cube`.

| key | value | default | unit | meaning |
|---|---|---|---|---|
| `path` | path to an existing .npz file | required |  | kernels and wavelengths_m members in one .npz snapshot |
| `pixel_scale_arcsec` | number > 0 | required | arcsec | angular sampling of every cube slice |
| `normalize` | true or false | `true` |  | divide each slice by its sum; false preserves unit-kernel bytes |
| `file_sha256` | null or lowercase SHA-256 hex digest | `null` |  | SHA-256 of the bytes decoded for both cube members |

## psf.truth (kind: optical)

Selected by `kind: optical`.

| key | value | default | unit | meaning |
|---|---|---|---|---|
| `pupil` | mapping by `kind` (hex_segmented, circular), see `psf.truth.pupil` | required |  | pupil geometry and sampling |
| `focal_length_m` | number > 0 | required | m | effective focal length |
| `wavelength_nm` | null or number > 0 | `null` | nm | wavelength of the monochromatic kernel |
| `wavelength_samples` | null or integer >= 1 | `null` |  | number of caller-supplied bandpass nodes |
| `detector_oversampling` | integer >= 1 | required |  | sub-samples per detector pixel side for the pixel integral (paper 3) |
| `kernel_shape` | pair of odd positive integers | required | pixels | kernel support (ny, nx), both odd |
| `wavefront` | mapping, see `psf.truth.wavefront` | `{}` |  | truth wavefront coefficients |
| `draw` | null or mapping, see `psf.truth.draw` | `null` |  | truth wavefront drawn from a prior at an exact RMS |

- exactly one of `wavelength_nm`, `wavelength_samples` is set; write null to clear one
- draw excludes listed wavefront coefficients

## psf.truth.pupil (kind: hex_segmented)

Selected by `kind: hex_segmented`.

| key | value | default | unit | meaning |
|---|---|---|---|---|
| `diameter_m` | number > 0 | required | m | side of the square pupil grid; a circular aperture fills it, a segmented aperture must fit inside |
| `pixels` | integer >= 1 | required |  | pupil samples per side |
| `supersampling` | integer >= 1 | required |  | sub-samples per pupil pixel side when the aperture is evaluated |
| `obscuration_ratio` | number in [0, 1) | `0.0` |  | central obscuration diameter as a fraction of diameter_m |
| `spiders` | null or mapping, see `psf.truth.pupil.spiders` | `null` |  | spiders from the centre outward |
| `rings` | integer >= 0 | required |  | rings of hexagonal segments around the centre |
| `segment_point_to_point_m` | number > 0 | required | m | segment vertex-to-vertex size |
| `gap_m` | number >= 0 | required | m | gap between adjacent segments |
| `central_segment` | true or false | `true` |  | whether the central segment is present |

- rings >= 1 without the central segment; every segment vertex lies inside the pupil grid

## psf.truth.pupil.spiders

| key | value | default | unit | meaning |
|---|---|---|---|---|
| `count` | integer >= 1 | required |  | number of spiders, at equal angles |
| `width_m` | number > 0 | required | m | full width of each spider |
| `angle_deg` | number | `0.0` | deg | direction of the first spider, from +x toward +y |

## psf.truth.pupil (kind: circular)

Selected by `kind: circular`.

| key | value | default | unit | meaning |
|---|---|---|---|---|
| `diameter_m` | number > 0 | required | m | side of the square pupil grid; a circular aperture fills it, a segmented aperture must fit inside |
| `pixels` | integer >= 1 | required |  | pupil samples per side |
| `supersampling` | integer >= 1 | required |  | sub-samples per pupil pixel side when the aperture is evaluated |
| `obscuration_ratio` | number in [0, 1) | `0.0` |  | central obscuration diameter as a fraction of diameter_m |
| `spiders` | null or mapping, see `psf.truth.pupil.spiders` | `null` |  | spiders from the centre outward |

## psf.truth.wavefront

Wavefront coefficients in nm of optical path difference; a listed zero is an entry.

| key | value | default | unit | meaning |
|---|---|---|---|---|
| `segment_hexikes` | mapping of integer >= 0 to mapping with at least 1 entries of integer >= 1 to number | `{}` | nm | segment index -> {Noll index: coefficient}; hex-segmented pupils only |
| `zernikes` | mapping of integer >= 1 to number | `{}` | nm | Noll index -> global Zernike coefficient |

## psf.truth.draw

| key | value | default | unit | meaning |
|---|---|---|---|---|
| `prior` | mapping, see `psf.truth.draw.prior` | required |  | mode-weight prior: exactly one of packaged, path, power_law |
| `amplitude_rms_nm` | number >= 0 | required | nm | piston-removed OPD RMS of the draw over the illuminated pupil |
| `seed` | integer >= 0 | required |  | seed of the draw's random numbers |
| `family` | one of: combined, global, segment | `combined` |  | combined (segment hexikes and global Zernikes), global or segment |

## psf.truth.draw.prior

| key | value | default | unit | meaning |
|---|---|---|---|---|
| `packaged` | null or one of: jwst_wss_static_v1, jwst_wss_drift_v1 | `null` |  | a prior table shipped with hwoslaps |
| `path` | null or path to an existing .yaml file | `null` |  | a prior table file |
| `power_law` | null or mapping, see `psf.truth.draw.prior.power_law` | `null` |  | a radial-order power-law prior |

- exactly one of `packaged`, `path`, `power_law` is set; write null to clear one

## psf.truth.draw.prior.power_law

| key | value | default | unit | meaning |
|---|---|---|---|---|
| `alpha` | number >= 0 | required |  | power-law index of the weights in radial order |
| `global_nolls` | null or pair, each integer >= 4 | required |  | inclusive global Zernike Noll range, or null for none |
| `segment_nolls` | null or pair, each integer >= 1 | required |  | inclusive segment hexike Noll range, or null for none |
| `segment_variance_fraction` | number in [0, 1] | required |  | share of a combined draw's variance on segments |

- at least one side; each range has lo <= hi

## psf.model (kind: matched)

Selected by `kind: matched` (the default).

No keys.

## psf.model (kind: kernel)

Selected by `kind: kernel`.

| key | value | default | unit | meaning |
|---|---|---|---|---|
| `path` | path to an existing .npy or .npz file | required |  | detector kernel file |
| `array_key` | null or non-empty text | `null` |  | member of a .npz file (null reads kernel); null for .npy |
| `pixel_scale_arcsec` | number > 0 | required | arcsec | angular sampling of the kernel |
| `normalize` | true or false | `true` |  | divide the kernel by its sum; false requires a sum within 1e-10 of one |
| `file_sha256` | null or lowercase SHA-256 hex digest | `null` |  | SHA-256 of the file bytes, checked before the file is read |

- a .npy file takes no array_key

## psf.model (kind: optical)

Selected by `kind: optical`.

| key | value | default | unit | meaning |
|---|---|---|---|---|
| `pupil` | mapping by `kind` (hex_segmented, circular), see `psf.model.pupil` | required |  | pupil geometry and sampling |
| `focal_length_m` | number > 0 | required | m | effective focal length |
| `wavelength_nm` | null or number > 0 | `null` | nm | wavelength of the monochromatic kernel |
| `wavelength_samples` | null or integer >= 1 | `null` |  | number of caller-supplied bandpass nodes |
| `detector_oversampling` | integer >= 1 | required |  | sub-samples per detector pixel side for the pixel integral (paper 3) |
| `kernel_shape` | pair of odd positive integers | required | pixels | kernel support (ny, nx), both odd |
| `wavefront` | mapping, see `psf.model.wavefront` | `{}` |  | truth wavefront coefficients |
| `draw` | null or mapping, see `psf.model.draw` | `null` |  | truth wavefront drawn from a prior at an exact RMS |

- exactly one of `wavelength_nm`, `wavelength_samples` is set; write null to clear one
- draw excludes listed wavefront coefficients

## psf.model.pupil (kind: hex_segmented)

Selected by `kind: hex_segmented`.

| key | value | default | unit | meaning |
|---|---|---|---|---|
| `diameter_m` | number > 0 | required | m | side of the square pupil grid; a circular aperture fills it, a segmented aperture must fit inside |
| `pixels` | integer >= 1 | required |  | pupil samples per side |
| `supersampling` | integer >= 1 | required |  | sub-samples per pupil pixel side when the aperture is evaluated |
| `obscuration_ratio` | number in [0, 1) | `0.0` |  | central obscuration diameter as a fraction of diameter_m |
| `spiders` | null or mapping, see `psf.model.pupil.spiders` | `null` |  | spiders from the centre outward |
| `rings` | integer >= 0 | required |  | rings of hexagonal segments around the centre |
| `segment_point_to_point_m` | number > 0 | required | m | segment vertex-to-vertex size |
| `gap_m` | number >= 0 | required | m | gap between adjacent segments |
| `central_segment` | true or false | `true` |  | whether the central segment is present |

- rings >= 1 without the central segment; every segment vertex lies inside the pupil grid

## psf.model.pupil.spiders

| key | value | default | unit | meaning |
|---|---|---|---|---|
| `count` | integer >= 1 | required |  | number of spiders, at equal angles |
| `width_m` | number > 0 | required | m | full width of each spider |
| `angle_deg` | number | `0.0` | deg | direction of the first spider, from +x toward +y |

## psf.model.pupil (kind: circular)

Selected by `kind: circular`.

| key | value | default | unit | meaning |
|---|---|---|---|---|
| `diameter_m` | number > 0 | required | m | side of the square pupil grid; a circular aperture fills it, a segmented aperture must fit inside |
| `pixels` | integer >= 1 | required |  | pupil samples per side |
| `supersampling` | integer >= 1 | required |  | sub-samples per pupil pixel side when the aperture is evaluated |
| `obscuration_ratio` | number in [0, 1) | `0.0` |  | central obscuration diameter as a fraction of diameter_m |
| `spiders` | null or mapping, see `psf.model.pupil.spiders` | `null` |  | spiders from the centre outward |

## psf.model.wavefront

Wavefront coefficients in nm of optical path difference; a listed zero is an entry.

| key | value | default | unit | meaning |
|---|---|---|---|---|
| `segment_hexikes` | mapping of integer >= 0 to mapping with at least 1 entries of integer >= 1 to number | `{}` | nm | segment index -> {Noll index: coefficient}; hex-segmented pupils only |
| `zernikes` | mapping of integer >= 1 to number | `{}` | nm | Noll index -> global Zernike coefficient |

## psf.model.draw

| key | value | default | unit | meaning |
|---|---|---|---|---|
| `prior` | mapping, see `psf.model.draw.prior` | required |  | mode-weight prior: exactly one of packaged, path, power_law |
| `amplitude_rms_nm` | number >= 0 | required | nm | piston-removed OPD RMS of the draw over the illuminated pupil |
| `seed` | integer >= 0 | required |  | seed of the draw's random numbers |
| `family` | one of: combined, global, segment | `combined` |  | combined (segment hexikes and global Zernikes), global or segment |

## psf.model.draw.prior

| key | value | default | unit | meaning |
|---|---|---|---|---|
| `packaged` | null or one of: jwst_wss_static_v1, jwst_wss_drift_v1 | `null` |  | a prior table shipped with hwoslaps |
| `path` | null or path to an existing .yaml file | `null` |  | a prior table file |
| `power_law` | null or mapping, see `psf.model.draw.prior.power_law` | `null` |  | a radial-order power-law prior |

- exactly one of `packaged`, `path`, `power_law` is set; write null to clear one

## psf.model.draw.prior.power_law

| key | value | default | unit | meaning |
|---|---|---|---|---|
| `alpha` | number >= 0 | required |  | power-law index of the weights in radial order |
| `global_nolls` | null or pair, each integer >= 4 | required |  | inclusive global Zernike Noll range, or null for none |
| `segment_nolls` | null or pair, each integer >= 1 | required |  | inclusive segment hexike Noll range, or null for none |
| `segment_variance_fraction` | number in [0, 1] | required |  | share of a combined draw's variance on segments |

- at least one side; each range has lo <= hi

## psf.model (kind: wavefront)

Selected by `kind: wavefront`.

| key | value | default | unit | meaning |
|---|---|---|---|---|
| `wavefront` | null or mapping, see `psf.model.wavefront` | `null` |  | coefficients replacing the truth coefficients |
| `offset` | null or mapping, see `psf.model.offset` | `null` |  | coefficients added to the truth coefficients |

- exactly one of `wavefront`, `offset` is set; write null to clear one

## psf.model.offset

Wavefront coefficients in nm of optical path difference; a listed zero is an entry.

| key | value | default | unit | meaning |
|---|---|---|---|---|
| `segment_hexikes` | mapping of integer >= 0 to mapping with at least 1 entries of integer >= 1 to number | `{}` | nm | segment index -> {Noll index: coefficient}; hex-segmented pupils only |
| `zernikes` | mapping of integer >= 1 to number | `{}` | nm | Noll index -> global Zernike coefficient |

## psf.model (kind: knowledge_error)

Selected by `kind: knowledge_error`.

| key | value | default | unit | meaning |
|---|---|---|---|---|
| `draw` | mapping, see `psf.model.draw` | required |  | knowledge-error draw added to the truth coefficients |

## psf.model (kind: monochromatic)

Selected by `kind: monochromatic`.

| key | value | default | unit | meaning |
|---|---|---|---|---|
| `wavelength_nm` | null or number > 0 | `null` |  | model wavelength; null uses each group's photon-weighted mean |

## instrument

the instrument

| key | value | default | unit | meaning |
|---|---|---|---|---|
| `name` | null or non-empty text | `null` |  | instrument label, recorded only |
| `detector` | mapping, see `instrument.detector` | required |  | detector noise parameters |
| `bandpass` | null or mapping by `kind` (top_hat, table, product), see `instrument.bandpass` | `null` |  | system throughput including detector quantum efficiency |
| `collecting_area_m2` | null or number > 0 | `null` | m^2 | photon-collecting area |

## instrument.detector

detector noise parameters

| key | value | default | unit | meaning |
|---|---|---|---|---|
| `gain_e_per_adu` | number | required | e-/ADU | detector gain; must be > 0 |
| `read_noise_e` | number | required | e- per pixel per exposure | read noise of one pixel in one exposure; must be >= 0 |
| `dark_current_e_per_s` | number | required | e-/s per pixel | dark current of one pixel; must be >= 0 |

- gain > 0, read noise >= 0 and dark current >= 0 (the Detector domain)

## instrument.bandpass (kind: top_hat)

Selected by `kind: top_hat`.

| key | value | default | unit | meaning |
|---|---|---|---|---|
| `min_nm` | number > 0 | required | nm | lower wavelength |
| `max_nm` | number > 0 | required | nm | upper wavelength |
| `throughput` | number in (0, 1] | required |  | electrons per entrance-pupil photon |
| `label` | null or non-empty text | `null` |  | bandpass label |

- wavelengths are ordered

## instrument.bandpass (kind: table)

Selected by `kind: table`.

| key | value | default | unit | meaning |
|---|---|---|---|---|
| `path` | path to an existing .yaml, .yml, .csv or .npz file | required |  | spectral table |
| `wavelength_key` | non-empty text | required |  | wavelength column or array |
| `value_key` | non-empty text | required |  | value column or array |
| `wavelength_unit` | one of: nm, angstrom, um, m | required |  | wavelength unit |
| `power` | integer >= 1 | `1` |  | number of identical surfaces |
| `label` | null or non-empty text | `null` |  | bandpass label |
| `support_nm` | pair, each number > 0 | required | nm | band support (low, high) |

- support is ordered

## instrument.bandpass (kind: product)

Selected by `kind: product`.

| key | value | default | unit | meaning |
|---|---|---|---|---|
| `support_nm` | pair, each number > 0 | required | nm | band support (low, high) |
| `factors` | list of at least 1 items, each mapping selected by kind: table, top_hat, constant, see `instrument.bandpass.factors[i]` | required |  | response factors |
| `label` | null or non-empty text | `null` |  | bandpass label |

- support is ordered and contained in each top-hat factor

## instrument.bandpass.factors[i] (kind: table)

Selected by `kind: table`.

| key | value | default | unit | meaning |
|---|---|---|---|---|
| `path` | path to an existing .yaml, .yml, .csv or .npz file | required |  | spectral table |
| `wavelength_key` | non-empty text | required |  | wavelength column or array |
| `value_key` | non-empty text | required |  | value column or array |
| `wavelength_unit` | one of: nm, angstrom, um, m | required |  | wavelength unit |
| `power` | integer >= 1 | `1` |  | number of identical surfaces |
| `label` | null or non-empty text | `null` |  | bandpass label |

## instrument.bandpass.factors[i] (kind: top_hat)

Selected by `kind: top_hat`.

| key | value | default | unit | meaning |
|---|---|---|---|---|
| `min_nm` | number > 0 | required | nm | lower wavelength |
| `max_nm` | number > 0 | required | nm | upper wavelength |
| `throughput` | number in (0, 1] | required |  | electrons per entrance-pupil photon |
| `label` | null or non-empty text | `null` |  | bandpass label |

- wavelengths are ordered

## instrument.bandpass.factors[i] (kind: constant)

Selected by `kind: constant`.

| key | value | default | unit | meaning |
|---|---|---|---|---|
| `value` | number in (0, 1] | required |  | constant factor |

## observation

the exposure

| key | value | default | unit | meaning |
|---|---|---|---|---|
| `exposure_time_s` | number > 0 | required | s | total exposure time of the summed exposures |
| `exposure_count` | integer >= 1 | `1` |  | number of equal exposures summed into the image; read noise enters once per exposure |
| `sky` | mapping, see `observation.sky` | required |  | the sky background |

## observation.sky

the sky background

| key | value | default | unit | meaning |
|---|---|---|---|---|
| `rate_e_per_s` | null or number >= 0 | `null` | e-/s per pixel | detected sky rate of one pixel |
| `ab_mag_per_arcsec2` | null or number | `null` | mag/arcsec^2 | AB sky surface brightness |
| `reference_band` | null or mapping by `kind` (top_hat, table, product), see `observation.sky.reference_band` | `null` |  | band of the supplied sky magnitude |
| `sed` | null or mapping by `kind` (flat_fnu, flat_flambda, power_law, table), see `observation.sky.sed` | `null` |  | sky spectral shape for a reference-band magnitude |

- exactly one of `rate_e_per_s`, `ab_mag_per_arcsec2` is set; write null to clear one
- reference sky magnitude and SED belong together

## observation.sky.reference_band (kind: top_hat)

Selected by `kind: top_hat`.

| key | value | default | unit | meaning |
|---|---|---|---|---|
| `min_nm` | number > 0 | required | nm | lower wavelength |
| `max_nm` | number > 0 | required | nm | upper wavelength |
| `throughput` | number in (0, 1] | required |  | electrons per entrance-pupil photon |
| `label` | null or non-empty text | `null` |  | bandpass label |

- wavelengths are ordered

## observation.sky.reference_band (kind: table)

Selected by `kind: table`.

| key | value | default | unit | meaning |
|---|---|---|---|---|
| `path` | path to an existing .yaml, .yml, .csv or .npz file | required |  | spectral table |
| `wavelength_key` | non-empty text | required |  | wavelength column or array |
| `value_key` | non-empty text | required |  | value column or array |
| `wavelength_unit` | one of: nm, angstrom, um, m | required |  | wavelength unit |
| `power` | integer >= 1 | `1` |  | number of identical surfaces |
| `label` | null or non-empty text | `null` |  | bandpass label |
| `support_nm` | pair, each number > 0 | required | nm | band support (low, high) |

- support is ordered

## observation.sky.reference_band (kind: product)

Selected by `kind: product`.

| key | value | default | unit | meaning |
|---|---|---|---|---|
| `support_nm` | pair, each number > 0 | required | nm | band support (low, high) |
| `factors` | list of at least 1 items, each mapping selected by kind: table, top_hat, constant, see `observation.sky.reference_band.factors[i]` | required |  | response factors |
| `label` | null or non-empty text | `null` |  | bandpass label |

- support is ordered and contained in each top-hat factor

## observation.sky.reference_band.factors[i] (kind: table)

Selected by `kind: table`.

| key | value | default | unit | meaning |
|---|---|---|---|---|
| `path` | path to an existing .yaml, .yml, .csv or .npz file | required |  | spectral table |
| `wavelength_key` | non-empty text | required |  | wavelength column or array |
| `value_key` | non-empty text | required |  | value column or array |
| `wavelength_unit` | one of: nm, angstrom, um, m | required |  | wavelength unit |
| `power` | integer >= 1 | `1` |  | number of identical surfaces |
| `label` | null or non-empty text | `null` |  | bandpass label |

## observation.sky.reference_band.factors[i] (kind: top_hat)

Selected by `kind: top_hat`.

| key | value | default | unit | meaning |
|---|---|---|---|---|
| `min_nm` | number > 0 | required | nm | lower wavelength |
| `max_nm` | number > 0 | required | nm | upper wavelength |
| `throughput` | number in (0, 1] | required |  | electrons per entrance-pupil photon |
| `label` | null or non-empty text | `null` |  | bandpass label |

- wavelengths are ordered

## observation.sky.reference_band.factors[i] (kind: constant)

Selected by `kind: constant`.

| key | value | default | unit | meaning |
|---|---|---|---|---|
| `value` | number in (0, 1] | required |  | constant factor |

## observation.sky.sed (kind: flat_fnu)

Selected by `kind: flat_fnu`.

No keys.

## observation.sky.sed (kind: flat_flambda)

Selected by `kind: flat_flambda`.

No keys.

## observation.sky.sed (kind: power_law)

Selected by `kind: power_law`.

| key | value | default | unit | meaning |
|---|---|---|---|---|
| `index` | number | required |  | index of f_nu proportional to nu**index |

## observation.sky.sed (kind: table)

Selected by `kind: table`.

| key | value | default | unit | meaning |
|---|---|---|---|---|
| `path` | path to an existing .yaml, .yml, .csv or .npz file | required |  | spectral table |
| `wavelength_key` | non-empty text | required |  | wavelength column or array |
| `value_key` | non-empty text | required |  | value column or array |
| `wavelength_unit` | one of: nm, angstrom, um, m | required |  | wavelength unit |
| `quantity` | one of: fnu, flambda | required |  | spectral density convention |
| `frame` | one of: observed, rest | `observed` |  | wavelength frame |

## forecast

Fisher forecast inputs; execution options belong to Execution.

| key | value | default | unit | meaning |
|---|---|---|---|---|
| `positions` | mapping by `kind` (grid, ring, explicit), see `forecast.positions` | required |  | where the subhalo hypothesis is evaluated |
| `mask` | mapping by `kind` (all_pixels, source_snr, annulus, psf_border), see `forecast.mask` | required |  | pixels used by the statistic |
| `nuisances` | mapping, see `forecast.nuisances` | `{}` |  | parameters profiled in the likelihood |
| `noise_covariance` | null or path to an existing .npy file | `null` |  | dense covariance over the full image |

## forecast.positions (kind: grid)

Selected by `kind: grid`.

| key | value | default | unit | meaning |
|---|---|---|---|---|
| `spacing_arcsec` | number > 0 | required | arcsec | lattice spacing |
| `half_width_arcsec` | number > 0 | required | arcsec | half width of the square lattice |
| `annulus` | null or mapping, see `forecast.positions.annulus` | `null` |  | retain nodes in this closed annulus |

- half_width_arcsec >= spacing_arcsec

## forecast.positions.annulus

| key | value | default | unit | meaning |
|---|---|---|---|---|
| `inner_arcsec` | number >= 0 | required | arcsec | inner radius of the closed annulus |
| `outer_arcsec` | number > 0 | required | arcsec | outer radius of the closed annulus |

- inner_arcsec < outer_arcsec

## forecast.positions (kind: ring)

Selected by `kind: ring`.

| key | value | default | unit | meaning |
|---|---|---|---|---|
| `count` | integer >= 1 | required |  | number of equally spaced positions |
| `radius` | one of: einstein_radius, critical_curve or number > 0 | `einstein_radius` | arcsec | radius about the lens centre |
| `offset_arcsec` | number | `0.0` | arcsec | offset added to the ring radius |

- a numeric radius plus offset is positive

## forecast.positions (kind: explicit)

Selected by `kind: explicit`.

| key | value | default | unit | meaning |
|---|---|---|---|---|
| `positions_yx` | list of at least 1 items, each pair, each number | required | arcsec | positions in (y, x) order |

- no duplicate position rows

## forecast.mask (kind: all_pixels)

Selected by `kind: all_pixels`.

No keys.

## forecast.mask (kind: source_snr)

Selected by `kind: source_snr`.

| key | value | default | unit | meaning |
|---|---|---|---|---|
| `snr_min` | number > 0 | required |  | minimum source-plane light signal-to-noise |

## forecast.mask (kind: annulus)

Selected by `kind: annulus`.

| key | value | default | unit | meaning |
|---|---|---|---|---|
| `inner_arcsec` | number >= 0 | required | arcsec | inner radius of the closed annulus |
| `outer_arcsec` | number > 0 | required | arcsec | outer radius of the closed annulus |
| `about` | one of: lens, grid | `lens` |  | centre of the annulus |

- inner_arcsec < outer_arcsec

## forecast.mask (kind: psf_border)

Selected by `kind: psf_border`.

No keys.

## forecast.nuisances

| key | value | default | unit | meaning |
|---|---|---|---|---|
| `fixed` | list, each non-empty text, no repeats | `[]` |  | scene parameter names or fnmatch patterns held fixed |
| `steps` | mapping of non-empty text to number > 0 | `{}` |  | finite-difference steps per kind or scene parameter name |
| `priors` | mapping of non-empty text to number > 0 | `{}` |  | Gaussian sigmas per scene parameter name |
| `background_offset` | true or false | `true` |  | profile a constant ADU offset |
| `wavefront` | null or mapping, see `forecast.nuisances.wavefront` | `null` |  | wavefront-mode nuisances |

## forecast.nuisances.wavefront

| key | value | default | unit | meaning |
|---|---|---|---|---|
| `modes` | mapping, see `forecast.nuisances.wavefront.modes` | required |  | wavefront families and modes to profile |
| `step_nm` | number > 0 or mapping of one of: segment_hexikes, zernikes to number > 0 | `1.0` | nm | central-difference step, scalar or per family |
| `prior_sigma_nm` | null or number > 0 or mapping of one of: segment_hexikes, zernikes to number > 0 | `null` | nm | Gaussian sigma, scalar or per family |

- family scales cover every selected family

## forecast.nuisances.wavefront.modes

| key | value | default | unit | meaning |
|---|---|---|---|---|
| `segment_hexikes` | null or mapping, see `forecast.nuisances.wavefront.modes.segment_hexikes` | `null` |  | segment hexike modes |
| `zernikes` | null or mapping, see `forecast.nuisances.wavefront.modes.zernikes` | `null` |  | global Zernike modes |

- at least one family; global Zernike Noll 1 is refused

## forecast.nuisances.wavefront.modes.segment_hexikes

| key | value | default | unit | meaning |
|---|---|---|---|---|
| `segments` | one of: all or list of at least 1 items, each integer >= 0, no repeats | required |  | segment indices, or all: every active segment of the model pupil |
| `nolls` | list of at least 1 items, each integer >= 1, no repeats | required |  | hexike Noll indices on each listed segment |

## forecast.nuisances.wavefront.modes.zernikes

| key | value | default | unit | meaning |
|---|---|---|---|---|
| `nolls` | list of at least 1 items, each integer >= 1, no repeats | required |  | global Zernike Noll indices (Noll 1 is refused) |

## population

| key | value | default | unit | meaning |
|---|---|---|---|---|
| `variables` | mapping of text matching [A-Za-z_][A-Za-z0-9_]* to mapping selected by kind: constant, choice, uniform, log_uniform, normal, truncated_normal, lognormal, truncated_lognormal, vector, polar_offset, ell_comps, shear_components, multipole_components, function, see `population.variables.<key>` | required |  | ordered variables |
| `copulas` | mapping of text matching [A-Za-z_][A-Za-z0-9_]* to mapping, see `population.copulas.<key>` | `{}` |  | Gaussian copulas |
| `catalog` | null or mapping, see `population.catalog` | `null` |  | catalog |
| `bind` | mapping with at least 1 entries of non-empty text to text matching [A-Za-z_][A-Za-z0-9_]*(?:\[[0-9]+\])? | required |  | existing effective configuration paths |
| `max_attempts` | integer >= 1 | `1` |  | rejection limit |

## population.variables.<key> (kind: constant)

Selected by `kind: constant`.

| key | value | default | unit | meaning |
|---|---|---|---|---|
| `value` | any value | required |  | constant JSON-shaped value |

## population.variables.<key> (kind: choice)

Selected by `kind: choice`.

| key | value | default | unit | meaning |
|---|---|---|---|---|
| `values` | list of at least 1 items, each any value | required |  | choice values |
| `weights` | null or list, each number >= 0 | `null` |  | choice weights |

## population.variables.<key> (kind: uniform)

Selected by `kind: uniform`.

| key | value | default | unit | meaning |
|---|---|---|---|---|
| `low` | number or mapping, see `population.variables.<key>.low` | required |  | finite numeric parameter or earlier reference |
| `high` | number or mapping, see `population.variables.<key>.high` | required |  | finite numeric parameter or earlier reference |

## population.variables.<key>.low

| key | value | default | unit | meaning |
|---|---|---|---|---|
| `var` | text matching [A-Za-z_][A-Za-z0-9_]*(?:\[[0-9]+\])? | required |  | earlier variable, optionally indexed |

## population.variables.<key>.high

| key | value | default | unit | meaning |
|---|---|---|---|---|
| `var` | text matching [A-Za-z_][A-Za-z0-9_]*(?:\[[0-9]+\])? | required |  | earlier variable, optionally indexed |

## population.variables.<key> (kind: log_uniform)

Selected by `kind: log_uniform`.

| key | value | default | unit | meaning |
|---|---|---|---|---|
| `low` | number or mapping, see `population.variables.<key>.low` | required |  | finite numeric parameter or earlier reference |
| `high` | number or mapping, see `population.variables.<key>.high` | required |  | finite numeric parameter or earlier reference |

## population.variables.<key> (kind: normal)

Selected by `kind: normal`.

| key | value | default | unit | meaning |
|---|---|---|---|---|
| `mean` | number or mapping, see `population.variables.<key>.mean` | required |  | finite numeric parameter or earlier reference |
| `std` | number or mapping, see `population.variables.<key>.std` | required |  | finite numeric parameter or earlier reference |

## population.variables.<key>.mean

| key | value | default | unit | meaning |
|---|---|---|---|---|
| `var` | text matching [A-Za-z_][A-Za-z0-9_]*(?:\[[0-9]+\])? | required |  | earlier variable, optionally indexed |

## population.variables.<key>.std

| key | value | default | unit | meaning |
|---|---|---|---|---|
| `var` | text matching [A-Za-z_][A-Za-z0-9_]*(?:\[[0-9]+\])? | required |  | earlier variable, optionally indexed |

## population.variables.<key> (kind: truncated_normal)

Selected by `kind: truncated_normal`.

| key | value | default | unit | meaning |
|---|---|---|---|---|
| `mean` | number or mapping, see `population.variables.<key>.mean` | required |  | finite numeric parameter or earlier reference |
| `std` | number or mapping, see `population.variables.<key>.std` | required |  | finite numeric parameter or earlier reference |
| `low` | number or mapping, see `population.variables.<key>.low` | required |  | finite numeric parameter or earlier reference |
| `high` | number or mapping, see `population.variables.<key>.high` | required |  | finite numeric parameter or earlier reference |

## population.variables.<key> (kind: lognormal)

Selected by `kind: lognormal`.

| key | value | default | unit | meaning |
|---|---|---|---|---|
| `median` | number or mapping, see `population.variables.<key>.median` | required |  | finite numeric parameter or earlier reference |
| `sigma_ln` | number or mapping, see `population.variables.<key>.sigma_ln` | required |  | finite numeric parameter or earlier reference |

## population.variables.<key>.median

| key | value | default | unit | meaning |
|---|---|---|---|---|
| `var` | text matching [A-Za-z_][A-Za-z0-9_]*(?:\[[0-9]+\])? | required |  | earlier variable, optionally indexed |

## population.variables.<key>.sigma_ln

| key | value | default | unit | meaning |
|---|---|---|---|---|
| `var` | text matching [A-Za-z_][A-Za-z0-9_]*(?:\[[0-9]+\])? | required |  | earlier variable, optionally indexed |

## population.variables.<key> (kind: truncated_lognormal)

Selected by `kind: truncated_lognormal`.

| key | value | default | unit | meaning |
|---|---|---|---|---|
| `median` | number or mapping, see `population.variables.<key>.median` | required |  | finite numeric parameter or earlier reference |
| `sigma_ln` | number or mapping, see `population.variables.<key>.sigma_ln` | required |  | finite numeric parameter or earlier reference |
| `low` | number or mapping, see `population.variables.<key>.low` | required |  | finite numeric parameter or earlier reference |
| `high` | number or mapping, see `population.variables.<key>.high` | required |  | finite numeric parameter or earlier reference |

## population.variables.<key> (kind: vector)

Selected by `kind: vector`.

| key | value | default | unit | meaning |
|---|---|---|---|---|
| `of` | list of at least 2 items, each number or mapping | required |  | vector components |

## population.variables.<key>.of[i]

| key | value | default | unit | meaning |
|---|---|---|---|---|
| `var` | text matching [A-Za-z_][A-Za-z0-9_]*(?:\[[0-9]+\])? | required |  | earlier variable, optionally indexed |

## population.variables.<key> (kind: polar_offset)

Selected by `kind: polar_offset`.

| key | value | default | unit | meaning |
|---|---|---|---|---|
| `radius` | number or mapping, see `population.variables.<key>.radius` | required |  | radius |
| `angle_deg` | number or mapping, see `population.variables.<key>.angle_deg` | required |  | angle in degrees |
| `centre_y` | number or mapping, see `population.variables.<key>.centre_y` | `0.0` |  | centre y |
| `centre_x` | number or mapping, see `population.variables.<key>.centre_x` | `0.0` |  | centre x |

## population.variables.<key>.radius

| key | value | default | unit | meaning |
|---|---|---|---|---|
| `var` | text matching [A-Za-z_][A-Za-z0-9_]*(?:\[[0-9]+\])? | required |  | earlier variable, optionally indexed |

## population.variables.<key>.angle_deg

| key | value | default | unit | meaning |
|---|---|---|---|---|
| `var` | text matching [A-Za-z_][A-Za-z0-9_]*(?:\[[0-9]+\])? | required |  | earlier variable, optionally indexed |

## population.variables.<key>.centre_y

| key | value | default | unit | meaning |
|---|---|---|---|---|
| `var` | text matching [A-Za-z_][A-Za-z0-9_]*(?:\[[0-9]+\])? | required |  | earlier variable, optionally indexed |

## population.variables.<key>.centre_x

| key | value | default | unit | meaning |
|---|---|---|---|---|
| `var` | text matching [A-Za-z_][A-Za-z0-9_]*(?:\[[0-9]+\])? | required |  | earlier variable, optionally indexed |

## population.variables.<key> (kind: ell_comps)

Selected by `kind: ell_comps`.

| key | value | default | unit | meaning |
|---|---|---|---|---|
| `axis_ratio` | number or mapping, see `population.variables.<key>.axis_ratio` | required |  | minor/major axis ratio |
| `angle_deg` | number or mapping, see `population.variables.<key>.angle_deg` | required |  | major axis angle |

## population.variables.<key>.axis_ratio

| key | value | default | unit | meaning |
|---|---|---|---|---|
| `var` | text matching [A-Za-z_][A-Za-z0-9_]*(?:\[[0-9]+\])? | required |  | earlier variable, optionally indexed |

## population.variables.<key> (kind: shear_components)

Selected by `kind: shear_components`.

| key | value | default | unit | meaning |
|---|---|---|---|---|
| `magnitude` | number or mapping, see `population.variables.<key>.magnitude` | required |  | shear magnitude |
| `angle_deg` | number or mapping, see `population.variables.<key>.angle_deg` | required |  | shear angle |

## population.variables.<key>.magnitude

| key | value | default | unit | meaning |
|---|---|---|---|---|
| `var` | text matching [A-Za-z_][A-Za-z0-9_]*(?:\[[0-9]+\])? | required |  | earlier variable, optionally indexed |

## population.variables.<key> (kind: multipole_components)

Selected by `kind: multipole_components`.

| key | value | default | unit | meaning |
|---|---|---|---|---|
| `strength` | number or mapping, see `population.variables.<key>.strength` | required |  | multipole strength |
| `angle_deg` | number or mapping, see `population.variables.<key>.angle_deg` | required |  | multipole angle |
| `order` | integer >= 1 | required |  | multipole order |

## population.variables.<key>.strength

| key | value | default | unit | meaning |
|---|---|---|---|---|
| `var` | text matching [A-Za-z_][A-Za-z0-9_]*(?:\[[0-9]+\])? | required |  | earlier variable, optionally indexed |

## population.variables.<key> (kind: function)

Selected by `kind: function`.

| key | value | default | unit | meaning |
|---|---|---|---|---|
| `function` | text matching [A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*:[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)* | required |  | module:function |
| `inputs` | mapping of non-empty text to number or mapping or list, each number or mapping | `{}` |  | keyword inputs |

## population.variables.<key>.inputs.<key>

| key | value | default | unit | meaning |
|---|---|---|---|---|
| `var` | text matching [A-Za-z_][A-Za-z0-9_]*(?:\[[0-9]+\])? | required |  | earlier variable, optionally indexed |

## population.variables.<key>.inputs.<key>[i]

| key | value | default | unit | meaning |
|---|---|---|---|---|
| `var` | text matching [A-Za-z_][A-Za-z0-9_]*(?:\[[0-9]+\])? | required |  | earlier variable, optionally indexed |

## population.copulas.<key>

| key | value | default | unit | meaning |
|---|---|---|---|---|
| `variables` | list of at least 2 items, each text matching [A-Za-z_][A-Za-z0-9_]*, no repeats | required |  | coupled distributions |
| `correlation` | list of at least 2 items, each list of at least 2 items, each number | required |  | normal-score correlation matrix |

## population.catalog

| key | value | default | unit | meaning |
|---|---|---|---|---|
| `path` | non-empty text | required |  | CSV/NPZ path |
| `columns` | mapping of non-empty text to non-empty text | `{}` |  | numeric columns |
| `text_columns` | mapping of non-empty text to non-empty text | `{}` |  | text columns |

## batch

| key | value | default | unit | meaning |
|---|---|---|---|---|
| `name` | text matching [A-Za-z0-9_.-]+ | required |  | batch name |
| `seed` | integer >= 0 | required |  | batch noise/sampler/direction entropy |
| `config` | mapping, see `batch.config`, non-empty text, or list of at least 1 items, each non-empty text | required |  | configuration files or effective mapping |
| `overrides` | mapping | `{}` |  | base configuration overlay |
| `population` | null or mapping, see `batch.population` | `null` |  | member population |
| `arms` | list of at least 1 items, each mapping, see `batch.arms[i]` | `[{name: base, overrides: {}, directions: null}]` |  | configuration arms |
| `simulate` | null or mapping, see `batch.simulate` | `null` |  | simulation family |
| `forecast` | null or mapping, see `batch.forecast` | `null` |  | forecast family |
| `nonlinear` | mapping of text matching [A-Za-z0-9_.-]+ to mapping, see `batch.nonlinear.<key>` | `{}` |  | named nonlinear families |
| `execution` | mapping, see `batch.execution` | `{}` |  | worker placement and runtime settings |

## batch.config

Engine configuration. Section-local rules run before these cross-section rules.

| key | value | default | unit | meaning |
|---|---|---|---|---|
| `run_name` | name of letters, digits, '.', '_', '-' | `run` |  | run label, excluded from scientific identity |
| `seed` | integer >= 0 | required |  | root of named scene-randomness streams; never a noise seed |
| `cosmology` | mapping, see `batch.config.cosmology` | required |  | cosmology of the lensing scene |
| `scene` | mapping, see `batch.config.scene` | required |  | smooth scene, subhalo hypothesis and injected halo |
| `psf` | mapping, see `batch.config.psf` | required |  | truth and model point spread functions |
| `instrument` | mapping, see `batch.config.instrument` | required |  | instrument and detector noise parameters |
| `observation` | mapping, see `batch.config.observation` | required |  | exposure and sky background |
| `forecast` | null or mapping, see `batch.config.forecast` | `null` |  | Fisher forecast inputs |

- every kernel file is sampled at scene.grid.pixel_scale_arcsec (X1)
- optical truth and model nodes are neither aliased nor under-resolved at the scene pixel scale (X4)
- wavelength_samples requires a bandpass and component SEDs (X5)
- AB source or sky inputs require a bandpass and a collecting area (X6)
- nuisance names resolve; wavefront modes need a model basis and existing segments (X7); an Einstein-radius ring needs exactly one radius (X8)

## batch.config.cosmology

The cosmology of distances, critical densities and halo scales.

| key | value | default | unit | meaning |
|---|---|---|---|---|
| `name` | null or non-empty text | `null` |  | named astropy flat Lambda-CDM realization (Planck15 preserves the paper backend) |
| `flat_lcdm` | null or mapping, see `batch.config.cosmology.flat_lcdm` | `null` |  | custom flat Lambda-CDM parameters |

- exactly one of `name`, `flat_lcdm` is set; write null to clear one

## batch.config.cosmology.flat_lcdm

| key | value | default | unit | meaning |
|---|---|---|---|---|
| `H0` | number > 0 | required | km/s/Mpc | Hubble constant |
| `Om0` | number in (0, 1) | required |  | matter density fraction |
| `Ob0` | number >= 0 | `0.0` |  | baryon density fraction |
| `Tcmb0` | number >= 0 | `0.0` | K | CMB temperature |
| `Neff` | number >= 0 | `3.046` |  | effective number of neutrino species |
| `m_nu_eV` | list of 3 items, each number >= 0 | `[0.0, 0.0, 0.0]` | eV | three neutrino masses |

- Ob0 must not exceed Om0

## batch.config.scene

The lensing scene.

| key | value | default | unit | meaning |
|---|---|---|---|---|
| `grid` | mapping, see `batch.config.scene.grid` | required |  | image grid |
| `lens` | mapping, see `batch.config.scene.lens` | required |  | lens galaxy |
| `source` | mapping, see `batch.config.scene.source` | required |  | source galaxy |
| `subhalo` | mapping by `type` (PointMass, SIS, NFW, TNFW), see `batch.config.scene.subhalo` | required |  | the detection hypothesis: forecasts and fits test this halo family |
| `injection` | null or mapping, see `batch.config.scene.injection` | `null` |  | the subhalo hwoslaps simulate and batch simulate jobs inject |
| `perturbers` | mapping, see `batch.config.scene.perturbers` | `{}` |  | fixed perturbing halos |

- source.redshift > lens.redshift
- component names unique within a galaxy and not reserved
- halo redshifts in (0, source.redshift); moline2017_eq7 only at the lens redshift and in its mass range
- radius: einstein_radius needs exactly one lens mass component with an Einstein radius

## batch.config.scene.grid

| key | value | default | unit | meaning |
|---|---|---|---|---|
| `shape` | pair of positive integers | required | pixels | (ny, nx) |
| `pixel_scale_arcsec` | number > 0 | required | arcsec | pixel side |
| `over_sample_size` | integer >= 1 | required |  | sub-pixels per axis for light rendering (the paper used 4) |

## batch.config.scene.lens

| key | value | default | unit | meaning |
|---|---|---|---|---|
| `redshift` | number > 0 | required |  | lens redshift |
| `mass` | named components (at least 1), see `batch.config.scene.lens.mass.<name>` | required |  | mass components; this order is the nuisance order |
| `light` | named components, see `batch.config.scene.lens.light.<name>` | `{}` |  | lens-plane light components |

## batch.config.scene.lens.mass.<name> (type: Isothermal)

Selected by `type: Isothermal`.

A mass component; its parameters are profiled in registry order unless fixed.

Singular isothermal ellipsoid (al.mp.Isothermal); AutoGalaxy evaluates it at q <= 0.99999.

| key | value | default | unit | meaning |
|---|---|---|---|---|
| `centre` | pair, each number | required | arcsec | (y, x) centre |
| `einstein_radius` | number > 0 | required | arcsec | AutoLens Einstein radius of the SIE |
| `ell_comps` | pair (e1, e2) with sqrt(e1^2 + e2^2) < 1 | required |  | elliptical components (f sin 2 phi, f cos 2 phi), f = (1 - q) / (1 + q), q the axis ratio and phi the major-axis angle counter-clockwise from +x; hypot below 0.999 |
| `multipoles` | null or mapping, see `batch.config.scene.lens.mass.<name>.multipoles` | `null` |  | multipoles linked to the base profile |

- hypot(ell_comps) < 0.999
- positive base plus multipole convergence

## batch.config.scene.lens.mass.<name>.multipoles

| key | value | default | unit | meaning |
|---|---|---|---|---|
| `m3` | null or pair, each number | `null` |  | Cartesian third-order multipole components |
| `m4` | null or pair, each number | `null` |  | Cartesian fourth-order multipole components |

- at least one multipole order

## batch.config.scene.lens.mass.<name> (type: PowerLaw)

Selected by `type: PowerLaw`.

A mass component; its parameters are profiled in registry order unless fixed.

Elliptical power-law lens mass.

| key | value | default | unit | meaning |
|---|---|---|---|---|
| `centre` | pair, each number | required | arcsec | (y, x) centre |
| `einstein_radius` | number > 0 | required | arcsec | AutoLens Einstein radius of the SIE |
| `ell_comps` | pair (e1, e2) with sqrt(e1^2 + e2^2) < 1 | required |  | elliptical components (f sin 2 phi, f cos 2 phi), f = (1 - q) / (1 + q), q the axis ratio and phi the major-axis angle counter-clockwise from +x; hypot below 0.999 |
| `multipoles` | null or mapping, see `batch.config.scene.lens.mass.<name>.multipoles` | `null` |  | multipoles linked to the base profile |
| `slope` | number in (1, 3) | required |  | three-dimensional density slope |

- hypot(ell_comps) < 0.999
- positive base plus multipole convergence

## batch.config.scene.lens.mass.<name> (type: ExternalShear)

Selected by `type: ExternalShear`.

A mass component; its parameters are profiled in registry order unless fixed.

| key | value | default | unit | meaning |
|---|---|---|---|---|
| `gamma_1` | number | required |  | external shear about the image-plane origin |
| `gamma_2` | number | required |  | external shear about the image-plane origin |

- hypot(shear) < 1

## batch.config.scene.lens.light.<name> (type: Exponential)

Selected by `type: Exponential`.

A light component; its parameters are profiled in registry order unless fixed.

Exponential (Sersic n = 1) light profile (al.lp.Exponential).

| key | value | default | unit | meaning |
|---|---|---|---|---|
| `centre` | pair, each number | required | arcsec | (y, x) centre |
| `ell_comps` | pair (e1, e2) with sqrt(e1^2 + e2^2) < 1 | required |  | elliptical components (f sin 2 phi, f cos 2 phi), f = (1 - q) / (1 + q), q the axis ratio and phi the major-axis angle counter-clockwise from +x; hypot below 0.999 |
| `effective_radius` | number > 0 | required | arcsec | circularized half-light radius |
| `intensity` | null or number > 0 | `null` |  | surface brightness at the effective radius, detected e-/s per pixel sample |
| `flux` | null or mapping, see `batch.config.scene.lens.light.<name>.flux` | `null` |  | intrinsic photometric normalization |
| `sed` | null or mapping by `kind` (flat_fnu, flat_flambda, power_law, table), see `batch.config.scene.lens.light.<name>.sed` | `null` |  | fixed spectral shape |

- exactly one of `intensity`, `flux` is set; write null to clear one
- hypot(ell_comps) < 0.999
- reference-band flux requires an SED

## batch.config.scene.lens.light.<name>.flux

| key | value | default | unit | meaning |
|---|---|---|---|---|
| `rate_e_per_s` | null or number > 0 | `null` | e-/s | intrinsic unlensed detected rate |
| `ab_mag` | null or number | `null` |  | AB magnitude in the instrument or reference band |
| `reference_band` | null or mapping by `kind` (top_hat, table, product), see `batch.config.scene.lens.light.<name>.flux.reference_band` | `null` |  | band in which the AB magnitude is specified |

- exactly one of `rate_e_per_s`, `ab_mag` is set; write null to clear one
- reference band requires AB magnitude

## batch.config.scene.lens.light.<name>.flux.reference_band (kind: top_hat)

Selected by `kind: top_hat`.

| key | value | default | unit | meaning |
|---|---|---|---|---|
| `min_nm` | number > 0 | required | nm | lower wavelength |
| `max_nm` | number > 0 | required | nm | upper wavelength |
| `throughput` | number in (0, 1] | required |  | electrons per entrance-pupil photon |
| `label` | null or non-empty text | `null` |  | bandpass label |

- wavelengths are ordered

## batch.config.scene.lens.light.<name>.flux.reference_band (kind: table)

Selected by `kind: table`.

| key | value | default | unit | meaning |
|---|---|---|---|---|
| `path` | path to an existing .yaml, .yml, .csv or .npz file | required |  | spectral table |
| `wavelength_key` | non-empty text | required |  | wavelength column or array |
| `value_key` | non-empty text | required |  | value column or array |
| `wavelength_unit` | one of: nm, angstrom, um, m | required |  | wavelength unit |
| `power` | integer >= 1 | `1` |  | number of identical surfaces |
| `label` | null or non-empty text | `null` |  | bandpass label |
| `support_nm` | pair, each number > 0 | required | nm | band support (low, high) |

- support is ordered

## batch.config.scene.lens.light.<name>.flux.reference_band (kind: product)

Selected by `kind: product`.

| key | value | default | unit | meaning |
|---|---|---|---|---|
| `support_nm` | pair, each number > 0 | required | nm | band support (low, high) |
| `factors` | list of at least 1 items, each mapping selected by kind: table, top_hat, constant, see `batch.config.scene.lens.light.<name>.flux.reference_band.factors[i]` | required |  | response factors |
| `label` | null or non-empty text | `null` |  | bandpass label |

- support is ordered and contained in each top-hat factor

## batch.config.scene.lens.light.<name>.flux.reference_band.factors[i] (kind: table)

Selected by `kind: table`.

| key | value | default | unit | meaning |
|---|---|---|---|---|
| `path` | path to an existing .yaml, .yml, .csv or .npz file | required |  | spectral table |
| `wavelength_key` | non-empty text | required |  | wavelength column or array |
| `value_key` | non-empty text | required |  | value column or array |
| `wavelength_unit` | one of: nm, angstrom, um, m | required |  | wavelength unit |
| `power` | integer >= 1 | `1` |  | number of identical surfaces |
| `label` | null or non-empty text | `null` |  | bandpass label |

## batch.config.scene.lens.light.<name>.flux.reference_band.factors[i] (kind: top_hat)

Selected by `kind: top_hat`.

| key | value | default | unit | meaning |
|---|---|---|---|---|
| `min_nm` | number > 0 | required | nm | lower wavelength |
| `max_nm` | number > 0 | required | nm | upper wavelength |
| `throughput` | number in (0, 1] | required |  | electrons per entrance-pupil photon |
| `label` | null or non-empty text | `null` |  | bandpass label |

- wavelengths are ordered

## batch.config.scene.lens.light.<name>.flux.reference_band.factors[i] (kind: constant)

Selected by `kind: constant`.

| key | value | default | unit | meaning |
|---|---|---|---|---|
| `value` | number in (0, 1] | required |  | constant factor |

## batch.config.scene.lens.light.<name>.sed (kind: flat_fnu)

Selected by `kind: flat_fnu`.

No keys.

## batch.config.scene.lens.light.<name>.sed (kind: flat_flambda)

Selected by `kind: flat_flambda`.

No keys.

## batch.config.scene.lens.light.<name>.sed (kind: power_law)

Selected by `kind: power_law`.

| key | value | default | unit | meaning |
|---|---|---|---|---|
| `index` | number | required |  | index of f_nu proportional to nu**index |

## batch.config.scene.lens.light.<name>.sed (kind: table)

Selected by `kind: table`.

| key | value | default | unit | meaning |
|---|---|---|---|---|
| `path` | path to an existing .yaml, .yml, .csv or .npz file | required |  | spectral table |
| `wavelength_key` | non-empty text | required |  | wavelength column or array |
| `value_key` | non-empty text | required |  | value column or array |
| `wavelength_unit` | one of: nm, angstrom, um, m | required |  | wavelength unit |
| `quantity` | one of: fnu, flambda | required |  | spectral density convention |
| `frame` | one of: observed, rest | `observed` |  | wavelength frame |

## batch.config.scene.lens.light.<name> (type: Sersic)

Selected by `type: Sersic`.

A light component; its parameters are profiled in registry order unless fixed.

Sersic light with circularized effective radius.

| key | value | default | unit | meaning |
|---|---|---|---|---|
| `centre` | pair, each number | required | arcsec | (y, x) centre |
| `ell_comps` | pair (e1, e2) with sqrt(e1^2 + e2^2) < 1 | required |  | elliptical components (f sin 2 phi, f cos 2 phi), f = (1 - q) / (1 + q), q the axis ratio and phi the major-axis angle counter-clockwise from +x; hypot below 0.999 |
| `effective_radius` | number > 0 | required | arcsec | circularized half-light radius |
| `intensity` | null or number > 0 | `null` |  | surface brightness at the effective radius, detected e-/s per pixel sample |
| `sersic_index` | number in [0.36, 8] | required |  | Sersic index of the rendered Ciotti-Bertin series |
| `flux` | null or mapping, see `batch.config.scene.lens.light.<name>.flux` | `null` |  | intrinsic photometric normalization |
| `sed` | null or mapping by `kind` (flat_fnu, flat_flambda, power_law, table), see `batch.config.scene.lens.light.<name>.sed` | `null` |  | fixed spectral shape |

- exactly one of `intensity`, `flux` is set; write null to clear one
- hypot(ell_comps) < 0.999
- reference-band flux requires an SED

## batch.config.scene.lens.light.<name> (type: Image)

Selected by `type: Image`.

A light component; its parameters are profiled in registry order unless fixed.

Pixelized source: a unit-integral asset evaluated by bilinear interpolation with a one-pixel zero pad.

| key | value | default | unit | meaning |
|---|---|---|---|---|
| `asset_path` | path to an existing .npz file | required |  | prepared image asset (format version 1) |
| `centre` | pair, each number | required | arcsec | (y, x) centre |
| `rotation_deg` | number | `0.0` | deg | counter-clockwise rotation of the image on the sky |
| `total_flux` | null or number > 0 | `null` |  | integral of the image at unit flux and size scales, e-/s per pixel sample times arcsec^2 |
| `flux_scale` | number > 0 | `1.0` |  | brightness multiplier |
| `size_scale` | number > 0 | `1.0` |  | magnification of the image at fixed surface brightness |
| `flux` | null or mapping, see `batch.config.scene.lens.light.<name>.flux` | `null` |  | intrinsic photometric normalization |
| `sed` | null or mapping by `kind` (flat_fnu, flat_flambda, power_law, table), see `batch.config.scene.lens.light.<name>.sed` | `null` |  | fixed spectral shape |

- exactly one of `total_flux`, `flux` is set; write null to clear one
- reference-band flux requires an SED

## batch.config.scene.source

| key | value | default | unit | meaning |
|---|---|---|---|---|
| `redshift` | number > 0 | required |  | source redshift, behind the lens |
| `light` | named components (at least 1), see `batch.config.scene.source.light.<name>` | required |  | source light components |

## batch.config.scene.source.light.<name> (type: Exponential)

Selected by `type: Exponential`.

A light component; its parameters are profiled in registry order unless fixed.

Exponential (Sersic n = 1) light profile (al.lp.Exponential).

| key | value | default | unit | meaning |
|---|---|---|---|---|
| `centre` | pair, each number | required | arcsec | (y, x) centre |
| `ell_comps` | pair (e1, e2) with sqrt(e1^2 + e2^2) < 1 | required |  | elliptical components (f sin 2 phi, f cos 2 phi), f = (1 - q) / (1 + q), q the axis ratio and phi the major-axis angle counter-clockwise from +x; hypot below 0.999 |
| `effective_radius` | number > 0 | required | arcsec | circularized half-light radius |
| `intensity` | null or number > 0 | `null` |  | surface brightness at the effective radius, detected e-/s per pixel sample |
| `flux` | null or mapping, see `batch.config.scene.source.light.<name>.flux` | `null` |  | intrinsic photometric normalization |
| `sed` | null or mapping by `kind` (flat_fnu, flat_flambda, power_law, table), see `batch.config.scene.source.light.<name>.sed` | `null` |  | fixed spectral shape |

- exactly one of `intensity`, `flux` is set; write null to clear one
- hypot(ell_comps) < 0.999
- reference-band flux requires an SED

## batch.config.scene.source.light.<name>.flux

| key | value | default | unit | meaning |
|---|---|---|---|---|
| `rate_e_per_s` | null or number > 0 | `null` | e-/s | intrinsic unlensed detected rate |
| `ab_mag` | null or number | `null` |  | AB magnitude in the instrument or reference band |
| `reference_band` | null or mapping by `kind` (top_hat, table, product), see `batch.config.scene.source.light.<name>.flux.reference_band` | `null` |  | band in which the AB magnitude is specified |

- exactly one of `rate_e_per_s`, `ab_mag` is set; write null to clear one
- reference band requires AB magnitude

## batch.config.scene.source.light.<name>.flux.reference_band (kind: top_hat)

Selected by `kind: top_hat`.

| key | value | default | unit | meaning |
|---|---|---|---|---|
| `min_nm` | number > 0 | required | nm | lower wavelength |
| `max_nm` | number > 0 | required | nm | upper wavelength |
| `throughput` | number in (0, 1] | required |  | electrons per entrance-pupil photon |
| `label` | null or non-empty text | `null` |  | bandpass label |

- wavelengths are ordered

## batch.config.scene.source.light.<name>.flux.reference_band (kind: table)

Selected by `kind: table`.

| key | value | default | unit | meaning |
|---|---|---|---|---|
| `path` | path to an existing .yaml, .yml, .csv or .npz file | required |  | spectral table |
| `wavelength_key` | non-empty text | required |  | wavelength column or array |
| `value_key` | non-empty text | required |  | value column or array |
| `wavelength_unit` | one of: nm, angstrom, um, m | required |  | wavelength unit |
| `power` | integer >= 1 | `1` |  | number of identical surfaces |
| `label` | null or non-empty text | `null` |  | bandpass label |
| `support_nm` | pair, each number > 0 | required | nm | band support (low, high) |

- support is ordered

## batch.config.scene.source.light.<name>.flux.reference_band (kind: product)

Selected by `kind: product`.

| key | value | default | unit | meaning |
|---|---|---|---|---|
| `support_nm` | pair, each number > 0 | required | nm | band support (low, high) |
| `factors` | list of at least 1 items, each mapping selected by kind: table, top_hat, constant, see `batch.config.scene.source.light.<name>.flux.reference_band.factors[i]` | required |  | response factors |
| `label` | null or non-empty text | `null` |  | bandpass label |

- support is ordered and contained in each top-hat factor

## batch.config.scene.source.light.<name>.flux.reference_band.factors[i] (kind: table)

Selected by `kind: table`.

| key | value | default | unit | meaning |
|---|---|---|---|---|
| `path` | path to an existing .yaml, .yml, .csv or .npz file | required |  | spectral table |
| `wavelength_key` | non-empty text | required |  | wavelength column or array |
| `value_key` | non-empty text | required |  | value column or array |
| `wavelength_unit` | one of: nm, angstrom, um, m | required |  | wavelength unit |
| `power` | integer >= 1 | `1` |  | number of identical surfaces |
| `label` | null or non-empty text | `null` |  | bandpass label |

## batch.config.scene.source.light.<name>.flux.reference_band.factors[i] (kind: top_hat)

Selected by `kind: top_hat`.

| key | value | default | unit | meaning |
|---|---|---|---|---|
| `min_nm` | number > 0 | required | nm | lower wavelength |
| `max_nm` | number > 0 | required | nm | upper wavelength |
| `throughput` | number in (0, 1] | required |  | electrons per entrance-pupil photon |
| `label` | null or non-empty text | `null` |  | bandpass label |

- wavelengths are ordered

## batch.config.scene.source.light.<name>.flux.reference_band.factors[i] (kind: constant)

Selected by `kind: constant`.

| key | value | default | unit | meaning |
|---|---|---|---|---|
| `value` | number in (0, 1] | required |  | constant factor |

## batch.config.scene.source.light.<name>.sed (kind: flat_fnu)

Selected by `kind: flat_fnu`.

No keys.

## batch.config.scene.source.light.<name>.sed (kind: flat_flambda)

Selected by `kind: flat_flambda`.

No keys.

## batch.config.scene.source.light.<name>.sed (kind: power_law)

Selected by `kind: power_law`.

| key | value | default | unit | meaning |
|---|---|---|---|---|
| `index` | number | required |  | index of f_nu proportional to nu**index |

## batch.config.scene.source.light.<name>.sed (kind: table)

Selected by `kind: table`.

| key | value | default | unit | meaning |
|---|---|---|---|---|
| `path` | path to an existing .yaml, .yml, .csv or .npz file | required |  | spectral table |
| `wavelength_key` | non-empty text | required |  | wavelength column or array |
| `value_key` | non-empty text | required |  | value column or array |
| `wavelength_unit` | one of: nm, angstrom, um, m | required |  | wavelength unit |
| `quantity` | one of: fnu, flambda | required |  | spectral density convention |
| `frame` | one of: observed, rest | `observed` |  | wavelength frame |

## batch.config.scene.source.light.<name> (type: Sersic)

Selected by `type: Sersic`.

A light component; its parameters are profiled in registry order unless fixed.

Sersic light with circularized effective radius.

| key | value | default | unit | meaning |
|---|---|---|---|---|
| `centre` | pair, each number | required | arcsec | (y, x) centre |
| `ell_comps` | pair (e1, e2) with sqrt(e1^2 + e2^2) < 1 | required |  | elliptical components (f sin 2 phi, f cos 2 phi), f = (1 - q) / (1 + q), q the axis ratio and phi the major-axis angle counter-clockwise from +x; hypot below 0.999 |
| `effective_radius` | number > 0 | required | arcsec | circularized half-light radius |
| `intensity` | null or number > 0 | `null` |  | surface brightness at the effective radius, detected e-/s per pixel sample |
| `sersic_index` | number in [0.36, 8] | required |  | Sersic index of the rendered Ciotti-Bertin series |
| `flux` | null or mapping, see `batch.config.scene.source.light.<name>.flux` | `null` |  | intrinsic photometric normalization |
| `sed` | null or mapping by `kind` (flat_fnu, flat_flambda, power_law, table), see `batch.config.scene.source.light.<name>.sed` | `null` |  | fixed spectral shape |

- exactly one of `intensity`, `flux` is set; write null to clear one
- hypot(ell_comps) < 0.999
- reference-band flux requires an SED

## batch.config.scene.source.light.<name> (type: Image)

Selected by `type: Image`.

A light component; its parameters are profiled in registry order unless fixed.

Pixelized source: a unit-integral asset evaluated by bilinear interpolation with a one-pixel zero pad.

| key | value | default | unit | meaning |
|---|---|---|---|---|
| `asset_path` | path to an existing .npz file | required |  | prepared image asset (format version 1) |
| `centre` | pair, each number | required | arcsec | (y, x) centre |
| `rotation_deg` | number | `0.0` | deg | counter-clockwise rotation of the image on the sky |
| `total_flux` | null or number > 0 | `null` |  | integral of the image at unit flux and size scales, e-/s per pixel sample times arcsec^2 |
| `flux_scale` | number > 0 | `1.0` |  | brightness multiplier |
| `size_scale` | number > 0 | `1.0` |  | magnification of the image at fixed surface brightness |
| `flux` | null or mapping, see `batch.config.scene.source.light.<name>.flux` | `null` |  | intrinsic photometric normalization |
| `sed` | null or mapping by `kind` (flat_fnu, flat_flambda, power_law, table), see `batch.config.scene.source.light.<name>.sed` | `null` |  | fixed spectral shape |

- exactly one of `total_flux`, `flux` is set; write null to clear one
- reference-band flux requires an SED

## batch.config.scene.subhalo (type: PointMass)

Selected by `type: PointMass`.

PointMass (point_mass), SIS or NFW (M200c), or TNFW (M200c_parent) halo.

| key | value | default | unit | meaning |
|---|---|---|---|---|
| `redshift` | null or number > 0 | `null` |  | redshift of the hypothesis; null is the lens redshift. Positions of an off-plane halo are angular positions in its own plane |

## batch.config.scene.subhalo (type: SIS)

Selected by `type: SIS`.

PointMass (point_mass), SIS or NFW (M200c), or TNFW (M200c_parent) halo.

| key | value | default | unit | meaning |
|---|---|---|---|---|
| `redshift` | null or number > 0 | `null` |  | redshift of the hypothesis; null is the lens redshift. Positions of an off-plane halo are angular positions in its own plane |

## batch.config.scene.subhalo (type: NFW)

Selected by `type: NFW`.

PointMass (point_mass), SIS or NFW (M200c), or TNFW (M200c_parent) halo.

| key | value | default | unit | meaning |
|---|---|---|---|---|
| `concentration` | mapping by `kind` (moline2017_eq7, power_law, fixed), see `batch.config.scene.subhalo.concentration` | required |  | concentration-mass relation |
| `redshift` | null or number > 0 | `null` |  | redshift of the hypothesis; null is the lens redshift. Positions of an off-plane halo are angular positions in its own plane |

## batch.config.scene.subhalo.concentration (kind: moline2017_eq7)

Selected by `kind: moline2017_eq7`.

Concentration-mass relation of an NFW halo.

Moline et al. (2017), eq. 7: subhalos at the lens redshift, M200 in [1e6, 1e12] Msun.

| key | value | default | unit | meaning |
|---|---|---|---|---|
| `x_sub` | number in (0, 1.5] | required |  | host-centric radius of the subhalo in units of the host virial radius |
| `h` | null or number > 0 | `null` |  | reduced Hubble constant of the relation's mass unit 1e8 / h Msun; null takes H0 / 100 of the cosmology |

## batch.config.scene.subhalo.concentration (kind: power_law)

Selected by `kind: power_law`.

Concentration-mass relation of an NFW halo.

c200 = c0 (M200 / mass_pivot_msun)^mass_slope (1 + z)^redshift_slope.

| key | value | default | unit | meaning |
|---|---|---|---|---|
| `c0` | number > 0 | required |  | concentration at the pivot mass and z = 0 |
| `mass_pivot_msun` | number > 0 | required | Msun | pivot mass |
| `mass_slope` | number | required |  | exponent of M200 / mass_pivot_msun |
| `redshift_slope` | number | required |  | exponent of 1 + z |

## batch.config.scene.subhalo.concentration (kind: fixed)

Selected by `kind: fixed`.

Concentration-mass relation of an NFW halo.

| key | value | default | unit | meaning |
|---|---|---|---|---|
| `value` | number > 0 | required |  | concentration c200 |

## batch.config.scene.subhalo (type: TNFW)

Selected by `type: TNFW`.

PointMass (point_mass), SIS or NFW (M200c), or TNFW (M200c_parent) halo.

| key | value | default | unit | meaning |
|---|---|---|---|---|
| `concentration` | mapping by `kind` (moline2017_eq7, power_law, fixed), see `batch.config.scene.subhalo.concentration` | required |  | parent NFW concentration-mass relation |
| `truncation` | mapping by `kind` (tau, overdensity), see `batch.config.scene.subhalo.truncation` | required |  | BMO truncation radius |
| `redshift` | null or number > 0 | `null` |  | redshift of the hypothesis; null is the lens redshift. Positions of an off-plane halo are angular positions in its own plane |

## batch.config.scene.subhalo.truncation (kind: tau)

Selected by `kind: tau`.

| key | value | default | unit | meaning |
|---|---|---|---|---|
| `tau` | number > 0 | required |  | r_t / r_s |

## batch.config.scene.subhalo.truncation (kind: overdensity)

Selected by `kind: overdensity`.

| key | value | default | unit | meaning |
|---|---|---|---|---|
| `overdensity` | number > 0 | required |  | parent NFW mean enclosed density in units of rho_crit |

## batch.config.scene.injection

| key | value | default | unit | meaning |
|---|---|---|---|---|
| `mass_msun` | number > 0 | required | Msun | subhalo mass in the hypothesis mass definition |
| `position` | mapping by `kind` (direct, angle, random), see `batch.config.scene.injection.position` | required |  | placement of the subhalo |

## batch.config.scene.injection.position (kind: direct)

Selected by `kind: direct`.

Where the injected subhalo sits.

| key | value | default | unit | meaning |
|---|---|---|---|---|
| `centre` | pair, each number | required | arcsec | (y, x) position of the subhalo in its own plane |

## batch.config.scene.injection.position (kind: angle)

Selected by `kind: angle`.

Where the injected subhalo sits.

| key | value | default | unit | meaning |
|---|---|---|---|---|
| `angle_deg` | number | required | deg | position angle about the lens centre, from +x toward +y |
| `radius` | one of: einstein_radius, critical_curve or number > 0 | `einstein_radius` | arcsec | einstein_radius, critical_curve, or a radius about the lens centre |
| `offset_arcsec` | number | `0.0` | arcsec | added to the radius |

## batch.config.scene.injection.position (kind: random)

Selected by `kind: random`.

Where the injected subhalo sits.

| key | value | default | unit | meaning |
|---|---|---|---|---|
| `radius` | one of: einstein_radius, critical_curve or number > 0 | `einstein_radius` | arcsec | einstein_radius, critical_curve, or a radius about the lens centre |
| `scatter_arcsec` | number > 0 | required | arcsec | half width of the uniform radial offset |

## batch.config.scene.perturbers

| key | value | default | unit | meaning |
|---|---|---|---|---|
| `halos` | list, each mapping selected by type: PointMass, SIS, NFW, TNFW, see `batch.config.scene.perturbers.halos[i]` | `[]` |  | fixed perturbing halos, in order |
| `populations` | list, each mapping selected by type: PointMass, SIS, NFW, TNFW, see `batch.config.scene.perturbers.populations[i]` | `[]` |  | drawn halo populations, in order |

## batch.config.scene.perturbers.halos[i] (type: PointMass)

Selected by `type: PointMass`.

PointMass (point_mass), SIS or NFW (M200c), or TNFW (M200c_parent) halo.

| key | value | default | unit | meaning |
|---|---|---|---|---|
| `mass_msun` | number > 0 | required | Msun | halo mass in its mass definition |
| `centre` | pair, each number | required | arcsec | (y, x) position in the halo's own plane |
| `redshift` | null or number > 0 | `null` |  | halo redshift; null is the lens redshift |

## batch.config.scene.perturbers.halos[i] (type: SIS)

Selected by `type: SIS`.

PointMass (point_mass), SIS or NFW (M200c), or TNFW (M200c_parent) halo.

| key | value | default | unit | meaning |
|---|---|---|---|---|
| `mass_msun` | number > 0 | required | Msun | halo mass in its mass definition |
| `centre` | pair, each number | required | arcsec | (y, x) position in the halo's own plane |
| `redshift` | null or number > 0 | `null` |  | halo redshift; null is the lens redshift |

## batch.config.scene.perturbers.halos[i] (type: NFW)

Selected by `type: NFW`.

PointMass (point_mass), SIS or NFW (M200c), or TNFW (M200c_parent) halo.

| key | value | default | unit | meaning |
|---|---|---|---|---|
| `concentration` | mapping by `kind` (moline2017_eq7, power_law, fixed), see `batch.config.scene.perturbers.halos[i].concentration` | required |  | concentration-mass relation |
| `mass_msun` | number > 0 | required | Msun | halo mass in its mass definition |
| `centre` | pair, each number | required | arcsec | (y, x) position in the halo's own plane |
| `redshift` | null or number > 0 | `null` |  | halo redshift; null is the lens redshift |

## batch.config.scene.perturbers.halos[i].concentration (kind: moline2017_eq7)

Selected by `kind: moline2017_eq7`.

Concentration-mass relation of an NFW halo.

Moline et al. (2017), eq. 7: subhalos at the lens redshift, M200 in [1e6, 1e12] Msun.

| key | value | default | unit | meaning |
|---|---|---|---|---|
| `x_sub` | number in (0, 1.5] | required |  | host-centric radius of the subhalo in units of the host virial radius |
| `h` | null or number > 0 | `null` |  | reduced Hubble constant of the relation's mass unit 1e8 / h Msun; null takes H0 / 100 of the cosmology |

## batch.config.scene.perturbers.halos[i].concentration (kind: power_law)

Selected by `kind: power_law`.

Concentration-mass relation of an NFW halo.

c200 = c0 (M200 / mass_pivot_msun)^mass_slope (1 + z)^redshift_slope.

| key | value | default | unit | meaning |
|---|---|---|---|---|
| `c0` | number > 0 | required |  | concentration at the pivot mass and z = 0 |
| `mass_pivot_msun` | number > 0 | required | Msun | pivot mass |
| `mass_slope` | number | required |  | exponent of M200 / mass_pivot_msun |
| `redshift_slope` | number | required |  | exponent of 1 + z |

## batch.config.scene.perturbers.halos[i].concentration (kind: fixed)

Selected by `kind: fixed`.

Concentration-mass relation of an NFW halo.

| key | value | default | unit | meaning |
|---|---|---|---|---|
| `value` | number > 0 | required |  | concentration c200 |

## batch.config.scene.perturbers.halos[i] (type: TNFW)

Selected by `type: TNFW`.

PointMass (point_mass), SIS or NFW (M200c), or TNFW (M200c_parent) halo.

| key | value | default | unit | meaning |
|---|---|---|---|---|
| `concentration` | mapping by `kind` (moline2017_eq7, power_law, fixed), see `batch.config.scene.perturbers.halos[i].concentration` | required |  | parent NFW concentration-mass relation |
| `truncation` | mapping by `kind` (tau, overdensity), see `batch.config.scene.perturbers.halos[i].truncation` | required |  | BMO truncation radius |
| `mass_msun` | number > 0 | required | Msun | halo mass in its mass definition |
| `centre` | pair, each number | required | arcsec | (y, x) position in the halo's own plane |
| `redshift` | null or number > 0 | `null` |  | halo redshift; null is the lens redshift |

## batch.config.scene.perturbers.halos[i].truncation (kind: tau)

Selected by `kind: tau`.

| key | value | default | unit | meaning |
|---|---|---|---|---|
| `tau` | number > 0 | required |  | r_t / r_s |

## batch.config.scene.perturbers.halos[i].truncation (kind: overdensity)

Selected by `kind: overdensity`.

| key | value | default | unit | meaning |
|---|---|---|---|---|
| `overdensity` | number > 0 | required |  | parent NFW mean enclosed density in units of rho_crit |

## batch.config.scene.perturbers.populations[i] (type: PointMass)

Selected by `type: PointMass`.

PointMass (point_mass), SIS or NFW (M200c), or TNFW (M200c_parent) halo.

| key | value | default | unit | meaning |
|---|---|---|---|---|
| `mass_function` | mapping by `kind` (power_law), see `batch.config.scene.perturbers.populations[i].mass_function` | required |  | population mass distribution and count |
| `spatial` | mapping by `kind` (uniform_disk, uniform_annulus), see `batch.config.scene.perturbers.populations[i].spatial` | required |  | population positions in their own plane |
| `redshift` | null or number > 0 | `null` |  | population redshift; null is the lens redshift |

## batch.config.scene.perturbers.populations[i].mass_function (kind: power_law)

Selected by `kind: power_law`.

| key | value | default | unit | meaning |
|---|---|---|---|---|
| `slope` | number | required |  | slope of dN/dM proportional to M**slope |
| `mass_min_msun` | number > 0 | required | Msun | minimum population mass |
| `mass_max_msun` | number > 0 | required | Msun | maximum population mass |
| `count` | null or integer >= 0 | `null` |  | fixed number of halos |
| `expected_count` | null or number > 0 | `null` |  | mean Poisson count |

- exactly one of `count`, `expected_count` is set; write null to clear one
- mass_max_msun exceeds mass_min_msun

## batch.config.scene.perturbers.populations[i].spatial (kind: uniform_disk)

Selected by `kind: uniform_disk`.

| key | value | default | unit | meaning |
|---|---|---|---|---|
| `radius_arcsec` | number > 0 | required | arcsec | disc radius about the configured lens centre |

## batch.config.scene.perturbers.populations[i].spatial (kind: uniform_annulus)

Selected by `kind: uniform_annulus`.

| key | value | default | unit | meaning |
|---|---|---|---|---|
| `inner_arcsec` | number >= 0 | required | arcsec | inner radius about the configured lens centre |
| `outer_arcsec` | number > 0 | required | arcsec | outer radius |

- outer_arcsec exceeds inner_arcsec

## batch.config.scene.perturbers.populations[i] (type: SIS)

Selected by `type: SIS`.

PointMass (point_mass), SIS or NFW (M200c), or TNFW (M200c_parent) halo.

| key | value | default | unit | meaning |
|---|---|---|---|---|
| `mass_function` | mapping by `kind` (power_law), see `batch.config.scene.perturbers.populations[i].mass_function` | required |  | population mass distribution and count |
| `spatial` | mapping by `kind` (uniform_disk, uniform_annulus), see `batch.config.scene.perturbers.populations[i].spatial` | required |  | population positions in their own plane |
| `redshift` | null or number > 0 | `null` |  | population redshift; null is the lens redshift |

## batch.config.scene.perturbers.populations[i] (type: NFW)

Selected by `type: NFW`.

PointMass (point_mass), SIS or NFW (M200c), or TNFW (M200c_parent) halo.

| key | value | default | unit | meaning |
|---|---|---|---|---|
| `concentration` | mapping by `kind` (moline2017_eq7, power_law, fixed), see `batch.config.scene.perturbers.populations[i].concentration` | required |  | concentration-mass relation |
| `mass_function` | mapping by `kind` (power_law), see `batch.config.scene.perturbers.populations[i].mass_function` | required |  | population mass distribution and count |
| `spatial` | mapping by `kind` (uniform_disk, uniform_annulus), see `batch.config.scene.perturbers.populations[i].spatial` | required |  | population positions in their own plane |
| `redshift` | null or number > 0 | `null` |  | population redshift; null is the lens redshift |

## batch.config.scene.perturbers.populations[i].concentration (kind: moline2017_eq7)

Selected by `kind: moline2017_eq7`.

Concentration-mass relation of an NFW halo.

Moline et al. (2017), eq. 7: subhalos at the lens redshift, M200 in [1e6, 1e12] Msun.

| key | value | default | unit | meaning |
|---|---|---|---|---|
| `x_sub` | number in (0, 1.5] | required |  | host-centric radius of the subhalo in units of the host virial radius |
| `h` | null or number > 0 | `null` |  | reduced Hubble constant of the relation's mass unit 1e8 / h Msun; null takes H0 / 100 of the cosmology |

## batch.config.scene.perturbers.populations[i].concentration (kind: power_law)

Selected by `kind: power_law`.

Concentration-mass relation of an NFW halo.

c200 = c0 (M200 / mass_pivot_msun)^mass_slope (1 + z)^redshift_slope.

| key | value | default | unit | meaning |
|---|---|---|---|---|
| `c0` | number > 0 | required |  | concentration at the pivot mass and z = 0 |
| `mass_pivot_msun` | number > 0 | required | Msun | pivot mass |
| `mass_slope` | number | required |  | exponent of M200 / mass_pivot_msun |
| `redshift_slope` | number | required |  | exponent of 1 + z |

## batch.config.scene.perturbers.populations[i].concentration (kind: fixed)

Selected by `kind: fixed`.

Concentration-mass relation of an NFW halo.

| key | value | default | unit | meaning |
|---|---|---|---|---|
| `value` | number > 0 | required |  | concentration c200 |

## batch.config.scene.perturbers.populations[i] (type: TNFW)

Selected by `type: TNFW`.

PointMass (point_mass), SIS or NFW (M200c), or TNFW (M200c_parent) halo.

| key | value | default | unit | meaning |
|---|---|---|---|---|
| `concentration` | mapping by `kind` (moline2017_eq7, power_law, fixed), see `batch.config.scene.perturbers.populations[i].concentration` | required |  | parent NFW concentration-mass relation |
| `truncation` | mapping by `kind` (tau, overdensity), see `batch.config.scene.perturbers.populations[i].truncation` | required |  | BMO truncation radius |
| `mass_function` | mapping by `kind` (power_law), see `batch.config.scene.perturbers.populations[i].mass_function` | required |  | population mass distribution and count |
| `spatial` | mapping by `kind` (uniform_disk, uniform_annulus), see `batch.config.scene.perturbers.populations[i].spatial` | required |  | population positions in their own plane |
| `redshift` | null or number > 0 | `null` |  | population redshift; null is the lens redshift |

## batch.config.scene.perturbers.populations[i].truncation (kind: tau)

Selected by `kind: tau`.

| key | value | default | unit | meaning |
|---|---|---|---|---|
| `tau` | number > 0 | required |  | r_t / r_s |

## batch.config.scene.perturbers.populations[i].truncation (kind: overdensity)

Selected by `kind: overdensity`.

| key | value | default | unit | meaning |
|---|---|---|---|---|
| `overdensity` | number > 0 | required |  | parent NFW mean enclosed density in units of rho_crit |

## batch.config.psf

| key | value | default | unit | meaning |
|---|---|---|---|---|
| `truth` | mapping by `kind` (kernel, kernel_cube, optical), see `batch.config.psf.truth` | required |  | the PSF that makes the data |
| `model` | mapping by `kind` (matched, kernel, optical, wavefront, knowledge_error, monochromatic), see `batch.config.psf.model` | `{}` |  | the PSF the analysis assumes |

- optical, wavefront, knowledge_error and monochromatic models need an optical truth; segment hexikes and segment draws need a hex-segmented pupil with those segments

## batch.config.psf.truth (kind: kernel)

Selected by `kind: kernel`.

| key | value | default | unit | meaning |
|---|---|---|---|---|
| `path` | path to an existing .npy or .npz file | required |  | detector kernel file |
| `array_key` | null or non-empty text | `null` |  | member of a .npz file (null reads kernel); null for .npy |
| `pixel_scale_arcsec` | number > 0 | required | arcsec | angular sampling of the kernel |
| `normalize` | true or false | `true` |  | divide the kernel by its sum; false requires a sum within 1e-10 of one |
| `file_sha256` | null or lowercase SHA-256 hex digest | `null` |  | SHA-256 of the file bytes, checked before the file is read |

- a .npy file takes no array_key

## batch.config.psf.truth (kind: kernel_cube)

Selected by `kind: kernel_cube`.

| key | value | default | unit | meaning |
|---|---|---|---|---|
| `path` | path to an existing .npz file | required |  | kernels and wavelengths_m members in one .npz snapshot |
| `pixel_scale_arcsec` | number > 0 | required | arcsec | angular sampling of every cube slice |
| `normalize` | true or false | `true` |  | divide each slice by its sum; false preserves unit-kernel bytes |
| `file_sha256` | null or lowercase SHA-256 hex digest | `null` |  | SHA-256 of the bytes decoded for both cube members |

## batch.config.psf.truth (kind: optical)

Selected by `kind: optical`.

| key | value | default | unit | meaning |
|---|---|---|---|---|
| `pupil` | mapping by `kind` (hex_segmented, circular), see `batch.config.psf.truth.pupil` | required |  | pupil geometry and sampling |
| `focal_length_m` | number > 0 | required | m | effective focal length |
| `wavelength_nm` | null or number > 0 | `null` | nm | wavelength of the monochromatic kernel |
| `wavelength_samples` | null or integer >= 1 | `null` |  | number of caller-supplied bandpass nodes |
| `detector_oversampling` | integer >= 1 | required |  | sub-samples per detector pixel side for the pixel integral (paper 3) |
| `kernel_shape` | pair of odd positive integers | required | pixels | kernel support (ny, nx), both odd |
| `wavefront` | mapping, see `batch.config.psf.truth.wavefront` | `{}` |  | truth wavefront coefficients |
| `draw` | null or mapping, see `batch.config.psf.truth.draw` | `null` |  | truth wavefront drawn from a prior at an exact RMS |

- exactly one of `wavelength_nm`, `wavelength_samples` is set; write null to clear one
- draw excludes listed wavefront coefficients

## batch.config.psf.truth.pupil (kind: hex_segmented)

Selected by `kind: hex_segmented`.

| key | value | default | unit | meaning |
|---|---|---|---|---|
| `diameter_m` | number > 0 | required | m | side of the square pupil grid; a circular aperture fills it, a segmented aperture must fit inside |
| `pixels` | integer >= 1 | required |  | pupil samples per side |
| `supersampling` | integer >= 1 | required |  | sub-samples per pupil pixel side when the aperture is evaluated |
| `obscuration_ratio` | number in [0, 1) | `0.0` |  | central obscuration diameter as a fraction of diameter_m |
| `spiders` | null or mapping, see `batch.config.psf.truth.pupil.spiders` | `null` |  | spiders from the centre outward |
| `rings` | integer >= 0 | required |  | rings of hexagonal segments around the centre |
| `segment_point_to_point_m` | number > 0 | required | m | segment vertex-to-vertex size |
| `gap_m` | number >= 0 | required | m | gap between adjacent segments |
| `central_segment` | true or false | `true` |  | whether the central segment is present |

- rings >= 1 without the central segment; every segment vertex lies inside the pupil grid

## batch.config.psf.truth.pupil.spiders

| key | value | default | unit | meaning |
|---|---|---|---|---|
| `count` | integer >= 1 | required |  | number of spiders, at equal angles |
| `width_m` | number > 0 | required | m | full width of each spider |
| `angle_deg` | number | `0.0` | deg | direction of the first spider, from +x toward +y |

## batch.config.psf.truth.pupil (kind: circular)

Selected by `kind: circular`.

| key | value | default | unit | meaning |
|---|---|---|---|---|
| `diameter_m` | number > 0 | required | m | side of the square pupil grid; a circular aperture fills it, a segmented aperture must fit inside |
| `pixels` | integer >= 1 | required |  | pupil samples per side |
| `supersampling` | integer >= 1 | required |  | sub-samples per pupil pixel side when the aperture is evaluated |
| `obscuration_ratio` | number in [0, 1) | `0.0` |  | central obscuration diameter as a fraction of diameter_m |
| `spiders` | null or mapping, see `batch.config.psf.truth.pupil.spiders` | `null` |  | spiders from the centre outward |

## batch.config.psf.truth.wavefront

Wavefront coefficients in nm of optical path difference; a listed zero is an entry.

| key | value | default | unit | meaning |
|---|---|---|---|---|
| `segment_hexikes` | mapping of integer >= 0 to mapping with at least 1 entries of integer >= 1 to number | `{}` | nm | segment index -> {Noll index: coefficient}; hex-segmented pupils only |
| `zernikes` | mapping of integer >= 1 to number | `{}` | nm | Noll index -> global Zernike coefficient |

## batch.config.psf.truth.draw

| key | value | default | unit | meaning |
|---|---|---|---|---|
| `prior` | mapping, see `batch.config.psf.truth.draw.prior` | required |  | mode-weight prior: exactly one of packaged, path, power_law |
| `amplitude_rms_nm` | number >= 0 | required | nm | piston-removed OPD RMS of the draw over the illuminated pupil |
| `seed` | integer >= 0 | required |  | seed of the draw's random numbers |
| `family` | one of: combined, global, segment | `combined` |  | combined (segment hexikes and global Zernikes), global or segment |

## batch.config.psf.truth.draw.prior

| key | value | default | unit | meaning |
|---|---|---|---|---|
| `packaged` | null or one of: jwst_wss_static_v1, jwst_wss_drift_v1 | `null` |  | a prior table shipped with hwoslaps |
| `path` | null or path to an existing .yaml file | `null` |  | a prior table file |
| `power_law` | null or mapping, see `batch.config.psf.truth.draw.prior.power_law` | `null` |  | a radial-order power-law prior |

- exactly one of `packaged`, `path`, `power_law` is set; write null to clear one

## batch.config.psf.truth.draw.prior.power_law

| key | value | default | unit | meaning |
|---|---|---|---|---|
| `alpha` | number >= 0 | required |  | power-law index of the weights in radial order |
| `global_nolls` | null or pair, each integer >= 4 | required |  | inclusive global Zernike Noll range, or null for none |
| `segment_nolls` | null or pair, each integer >= 1 | required |  | inclusive segment hexike Noll range, or null for none |
| `segment_variance_fraction` | number in [0, 1] | required |  | share of a combined draw's variance on segments |

- at least one side; each range has lo <= hi

## batch.config.psf.model (kind: matched)

Selected by `kind: matched` (the default).

No keys.

## batch.config.psf.model (kind: kernel)

Selected by `kind: kernel`.

| key | value | default | unit | meaning |
|---|---|---|---|---|
| `path` | path to an existing .npy or .npz file | required |  | detector kernel file |
| `array_key` | null or non-empty text | `null` |  | member of a .npz file (null reads kernel); null for .npy |
| `pixel_scale_arcsec` | number > 0 | required | arcsec | angular sampling of the kernel |
| `normalize` | true or false | `true` |  | divide the kernel by its sum; false requires a sum within 1e-10 of one |
| `file_sha256` | null or lowercase SHA-256 hex digest | `null` |  | SHA-256 of the file bytes, checked before the file is read |

- a .npy file takes no array_key

## batch.config.psf.model (kind: optical)

Selected by `kind: optical`.

| key | value | default | unit | meaning |
|---|---|---|---|---|
| `pupil` | mapping by `kind` (hex_segmented, circular), see `batch.config.psf.model.pupil` | required |  | pupil geometry and sampling |
| `focal_length_m` | number > 0 | required | m | effective focal length |
| `wavelength_nm` | null or number > 0 | `null` | nm | wavelength of the monochromatic kernel |
| `wavelength_samples` | null or integer >= 1 | `null` |  | number of caller-supplied bandpass nodes |
| `detector_oversampling` | integer >= 1 | required |  | sub-samples per detector pixel side for the pixel integral (paper 3) |
| `kernel_shape` | pair of odd positive integers | required | pixels | kernel support (ny, nx), both odd |
| `wavefront` | mapping, see `batch.config.psf.model.wavefront` | `{}` |  | truth wavefront coefficients |
| `draw` | null or mapping, see `batch.config.psf.model.draw` | `null` |  | truth wavefront drawn from a prior at an exact RMS |

- exactly one of `wavelength_nm`, `wavelength_samples` is set; write null to clear one
- draw excludes listed wavefront coefficients

## batch.config.psf.model.pupil (kind: hex_segmented)

Selected by `kind: hex_segmented`.

| key | value | default | unit | meaning |
|---|---|---|---|---|
| `diameter_m` | number > 0 | required | m | side of the square pupil grid; a circular aperture fills it, a segmented aperture must fit inside |
| `pixels` | integer >= 1 | required |  | pupil samples per side |
| `supersampling` | integer >= 1 | required |  | sub-samples per pupil pixel side when the aperture is evaluated |
| `obscuration_ratio` | number in [0, 1) | `0.0` |  | central obscuration diameter as a fraction of diameter_m |
| `spiders` | null or mapping, see `batch.config.psf.model.pupil.spiders` | `null` |  | spiders from the centre outward |
| `rings` | integer >= 0 | required |  | rings of hexagonal segments around the centre |
| `segment_point_to_point_m` | number > 0 | required | m | segment vertex-to-vertex size |
| `gap_m` | number >= 0 | required | m | gap between adjacent segments |
| `central_segment` | true or false | `true` |  | whether the central segment is present |

- rings >= 1 without the central segment; every segment vertex lies inside the pupil grid

## batch.config.psf.model.pupil.spiders

| key | value | default | unit | meaning |
|---|---|---|---|---|
| `count` | integer >= 1 | required |  | number of spiders, at equal angles |
| `width_m` | number > 0 | required | m | full width of each spider |
| `angle_deg` | number | `0.0` | deg | direction of the first spider, from +x toward +y |

## batch.config.psf.model.pupil (kind: circular)

Selected by `kind: circular`.

| key | value | default | unit | meaning |
|---|---|---|---|---|
| `diameter_m` | number > 0 | required | m | side of the square pupil grid; a circular aperture fills it, a segmented aperture must fit inside |
| `pixels` | integer >= 1 | required |  | pupil samples per side |
| `supersampling` | integer >= 1 | required |  | sub-samples per pupil pixel side when the aperture is evaluated |
| `obscuration_ratio` | number in [0, 1) | `0.0` |  | central obscuration diameter as a fraction of diameter_m |
| `spiders` | null or mapping, see `batch.config.psf.model.pupil.spiders` | `null` |  | spiders from the centre outward |

## batch.config.psf.model.wavefront

Wavefront coefficients in nm of optical path difference; a listed zero is an entry.

| key | value | default | unit | meaning |
|---|---|---|---|---|
| `segment_hexikes` | mapping of integer >= 0 to mapping with at least 1 entries of integer >= 1 to number | `{}` | nm | segment index -> {Noll index: coefficient}; hex-segmented pupils only |
| `zernikes` | mapping of integer >= 1 to number | `{}` | nm | Noll index -> global Zernike coefficient |

## batch.config.psf.model.draw

| key | value | default | unit | meaning |
|---|---|---|---|---|
| `prior` | mapping, see `batch.config.psf.model.draw.prior` | required |  | mode-weight prior: exactly one of packaged, path, power_law |
| `amplitude_rms_nm` | number >= 0 | required | nm | piston-removed OPD RMS of the draw over the illuminated pupil |
| `seed` | integer >= 0 | required |  | seed of the draw's random numbers |
| `family` | one of: combined, global, segment | `combined` |  | combined (segment hexikes and global Zernikes), global or segment |

## batch.config.psf.model.draw.prior

| key | value | default | unit | meaning |
|---|---|---|---|---|
| `packaged` | null or one of: jwst_wss_static_v1, jwst_wss_drift_v1 | `null` |  | a prior table shipped with hwoslaps |
| `path` | null or path to an existing .yaml file | `null` |  | a prior table file |
| `power_law` | null or mapping, see `batch.config.psf.model.draw.prior.power_law` | `null` |  | a radial-order power-law prior |

- exactly one of `packaged`, `path`, `power_law` is set; write null to clear one

## batch.config.psf.model.draw.prior.power_law

| key | value | default | unit | meaning |
|---|---|---|---|---|
| `alpha` | number >= 0 | required |  | power-law index of the weights in radial order |
| `global_nolls` | null or pair, each integer >= 4 | required |  | inclusive global Zernike Noll range, or null for none |
| `segment_nolls` | null or pair, each integer >= 1 | required |  | inclusive segment hexike Noll range, or null for none |
| `segment_variance_fraction` | number in [0, 1] | required |  | share of a combined draw's variance on segments |

- at least one side; each range has lo <= hi

## batch.config.psf.model (kind: wavefront)

Selected by `kind: wavefront`.

| key | value | default | unit | meaning |
|---|---|---|---|---|
| `wavefront` | null or mapping, see `batch.config.psf.model.wavefront` | `null` |  | coefficients replacing the truth coefficients |
| `offset` | null or mapping, see `batch.config.psf.model.offset` | `null` |  | coefficients added to the truth coefficients |

- exactly one of `wavefront`, `offset` is set; write null to clear one

## batch.config.psf.model.offset

Wavefront coefficients in nm of optical path difference; a listed zero is an entry.

| key | value | default | unit | meaning |
|---|---|---|---|---|
| `segment_hexikes` | mapping of integer >= 0 to mapping with at least 1 entries of integer >= 1 to number | `{}` | nm | segment index -> {Noll index: coefficient}; hex-segmented pupils only |
| `zernikes` | mapping of integer >= 1 to number | `{}` | nm | Noll index -> global Zernike coefficient |

## batch.config.psf.model (kind: knowledge_error)

Selected by `kind: knowledge_error`.

| key | value | default | unit | meaning |
|---|---|---|---|---|
| `draw` | mapping, see `batch.config.psf.model.draw` | required |  | knowledge-error draw added to the truth coefficients |

## batch.config.psf.model (kind: monochromatic)

Selected by `kind: monochromatic`.

| key | value | default | unit | meaning |
|---|---|---|---|---|
| `wavelength_nm` | null or number > 0 | `null` |  | model wavelength; null uses each group's photon-weighted mean |

## batch.config.instrument

the instrument

| key | value | default | unit | meaning |
|---|---|---|---|---|
| `name` | null or non-empty text | `null` |  | instrument label, recorded only |
| `detector` | mapping, see `batch.config.instrument.detector` | required |  | detector noise parameters |
| `bandpass` | null or mapping by `kind` (top_hat, table, product), see `batch.config.instrument.bandpass` | `null` |  | system throughput including detector quantum efficiency |
| `collecting_area_m2` | null or number > 0 | `null` | m^2 | photon-collecting area |

## batch.config.instrument.detector

detector noise parameters

| key | value | default | unit | meaning |
|---|---|---|---|---|
| `gain_e_per_adu` | number | required | e-/ADU | detector gain; must be > 0 |
| `read_noise_e` | number | required | e- per pixel per exposure | read noise of one pixel in one exposure; must be >= 0 |
| `dark_current_e_per_s` | number | required | e-/s per pixel | dark current of one pixel; must be >= 0 |

- gain > 0, read noise >= 0 and dark current >= 0 (the Detector domain)

## batch.config.instrument.bandpass (kind: top_hat)

Selected by `kind: top_hat`.

| key | value | default | unit | meaning |
|---|---|---|---|---|
| `min_nm` | number > 0 | required | nm | lower wavelength |
| `max_nm` | number > 0 | required | nm | upper wavelength |
| `throughput` | number in (0, 1] | required |  | electrons per entrance-pupil photon |
| `label` | null or non-empty text | `null` |  | bandpass label |

- wavelengths are ordered

## batch.config.instrument.bandpass (kind: table)

Selected by `kind: table`.

| key | value | default | unit | meaning |
|---|---|---|---|---|
| `path` | path to an existing .yaml, .yml, .csv or .npz file | required |  | spectral table |
| `wavelength_key` | non-empty text | required |  | wavelength column or array |
| `value_key` | non-empty text | required |  | value column or array |
| `wavelength_unit` | one of: nm, angstrom, um, m | required |  | wavelength unit |
| `power` | integer >= 1 | `1` |  | number of identical surfaces |
| `label` | null or non-empty text | `null` |  | bandpass label |
| `support_nm` | pair, each number > 0 | required | nm | band support (low, high) |

- support is ordered

## batch.config.instrument.bandpass (kind: product)

Selected by `kind: product`.

| key | value | default | unit | meaning |
|---|---|---|---|---|
| `support_nm` | pair, each number > 0 | required | nm | band support (low, high) |
| `factors` | list of at least 1 items, each mapping selected by kind: table, top_hat, constant, see `batch.config.instrument.bandpass.factors[i]` | required |  | response factors |
| `label` | null or non-empty text | `null` |  | bandpass label |

- support is ordered and contained in each top-hat factor

## batch.config.instrument.bandpass.factors[i] (kind: table)

Selected by `kind: table`.

| key | value | default | unit | meaning |
|---|---|---|---|---|
| `path` | path to an existing .yaml, .yml, .csv or .npz file | required |  | spectral table |
| `wavelength_key` | non-empty text | required |  | wavelength column or array |
| `value_key` | non-empty text | required |  | value column or array |
| `wavelength_unit` | one of: nm, angstrom, um, m | required |  | wavelength unit |
| `power` | integer >= 1 | `1` |  | number of identical surfaces |
| `label` | null or non-empty text | `null` |  | bandpass label |

## batch.config.instrument.bandpass.factors[i] (kind: top_hat)

Selected by `kind: top_hat`.

| key | value | default | unit | meaning |
|---|---|---|---|---|
| `min_nm` | number > 0 | required | nm | lower wavelength |
| `max_nm` | number > 0 | required | nm | upper wavelength |
| `throughput` | number in (0, 1] | required |  | electrons per entrance-pupil photon |
| `label` | null or non-empty text | `null` |  | bandpass label |

- wavelengths are ordered

## batch.config.instrument.bandpass.factors[i] (kind: constant)

Selected by `kind: constant`.

| key | value | default | unit | meaning |
|---|---|---|---|---|
| `value` | number in (0, 1] | required |  | constant factor |

## batch.config.observation

the exposure

| key | value | default | unit | meaning |
|---|---|---|---|---|
| `exposure_time_s` | number > 0 | required | s | total exposure time of the summed exposures |
| `exposure_count` | integer >= 1 | `1` |  | number of equal exposures summed into the image; read noise enters once per exposure |
| `sky` | mapping, see `batch.config.observation.sky` | required |  | the sky background |

## batch.config.observation.sky

the sky background

| key | value | default | unit | meaning |
|---|---|---|---|---|
| `rate_e_per_s` | null or number >= 0 | `null` | e-/s per pixel | detected sky rate of one pixel |
| `ab_mag_per_arcsec2` | null or number | `null` | mag/arcsec^2 | AB sky surface brightness |
| `reference_band` | null or mapping by `kind` (top_hat, table, product), see `batch.config.observation.sky.reference_band` | `null` |  | band of the supplied sky magnitude |
| `sed` | null or mapping by `kind` (flat_fnu, flat_flambda, power_law, table), see `batch.config.observation.sky.sed` | `null` |  | sky spectral shape for a reference-band magnitude |

- exactly one of `rate_e_per_s`, `ab_mag_per_arcsec2` is set; write null to clear one
- reference sky magnitude and SED belong together

## batch.config.observation.sky.reference_band (kind: top_hat)

Selected by `kind: top_hat`.

| key | value | default | unit | meaning |
|---|---|---|---|---|
| `min_nm` | number > 0 | required | nm | lower wavelength |
| `max_nm` | number > 0 | required | nm | upper wavelength |
| `throughput` | number in (0, 1] | required |  | electrons per entrance-pupil photon |
| `label` | null or non-empty text | `null` |  | bandpass label |

- wavelengths are ordered

## batch.config.observation.sky.reference_band (kind: table)

Selected by `kind: table`.

| key | value | default | unit | meaning |
|---|---|---|---|---|
| `path` | path to an existing .yaml, .yml, .csv or .npz file | required |  | spectral table |
| `wavelength_key` | non-empty text | required |  | wavelength column or array |
| `value_key` | non-empty text | required |  | value column or array |
| `wavelength_unit` | one of: nm, angstrom, um, m | required |  | wavelength unit |
| `power` | integer >= 1 | `1` |  | number of identical surfaces |
| `label` | null or non-empty text | `null` |  | bandpass label |
| `support_nm` | pair, each number > 0 | required | nm | band support (low, high) |

- support is ordered

## batch.config.observation.sky.reference_band (kind: product)

Selected by `kind: product`.

| key | value | default | unit | meaning |
|---|---|---|---|---|
| `support_nm` | pair, each number > 0 | required | nm | band support (low, high) |
| `factors` | list of at least 1 items, each mapping selected by kind: table, top_hat, constant, see `batch.config.observation.sky.reference_band.factors[i]` | required |  | response factors |
| `label` | null or non-empty text | `null` |  | bandpass label |

- support is ordered and contained in each top-hat factor

## batch.config.observation.sky.reference_band.factors[i] (kind: table)

Selected by `kind: table`.

| key | value | default | unit | meaning |
|---|---|---|---|---|
| `path` | path to an existing .yaml, .yml, .csv or .npz file | required |  | spectral table |
| `wavelength_key` | non-empty text | required |  | wavelength column or array |
| `value_key` | non-empty text | required |  | value column or array |
| `wavelength_unit` | one of: nm, angstrom, um, m | required |  | wavelength unit |
| `power` | integer >= 1 | `1` |  | number of identical surfaces |
| `label` | null or non-empty text | `null` |  | bandpass label |

## batch.config.observation.sky.reference_band.factors[i] (kind: top_hat)

Selected by `kind: top_hat`.

| key | value | default | unit | meaning |
|---|---|---|---|---|
| `min_nm` | number > 0 | required | nm | lower wavelength |
| `max_nm` | number > 0 | required | nm | upper wavelength |
| `throughput` | number in (0, 1] | required |  | electrons per entrance-pupil photon |
| `label` | null or non-empty text | `null` |  | bandpass label |

- wavelengths are ordered

## batch.config.observation.sky.reference_band.factors[i] (kind: constant)

Selected by `kind: constant`.

| key | value | default | unit | meaning |
|---|---|---|---|---|
| `value` | number in (0, 1] | required |  | constant factor |

## batch.config.observation.sky.sed (kind: flat_fnu)

Selected by `kind: flat_fnu`.

No keys.

## batch.config.observation.sky.sed (kind: flat_flambda)

Selected by `kind: flat_flambda`.

No keys.

## batch.config.observation.sky.sed (kind: power_law)

Selected by `kind: power_law`.

| key | value | default | unit | meaning |
|---|---|---|---|---|
| `index` | number | required |  | index of f_nu proportional to nu**index |

## batch.config.observation.sky.sed (kind: table)

Selected by `kind: table`.

| key | value | default | unit | meaning |
|---|---|---|---|---|
| `path` | path to an existing .yaml, .yml, .csv or .npz file | required |  | spectral table |
| `wavelength_key` | non-empty text | required |  | wavelength column or array |
| `value_key` | non-empty text | required |  | value column or array |
| `wavelength_unit` | one of: nm, angstrom, um, m | required |  | wavelength unit |
| `quantity` | one of: fnu, flambda | required |  | spectral density convention |
| `frame` | one of: observed, rest | `observed` |  | wavelength frame |

## batch.config.forecast

Fisher forecast inputs; execution options belong to Execution.

| key | value | default | unit | meaning |
|---|---|---|---|---|
| `positions` | mapping by `kind` (grid, ring, explicit), see `batch.config.forecast.positions` | required |  | where the subhalo hypothesis is evaluated |
| `mask` | mapping by `kind` (all_pixels, source_snr, annulus, psf_border), see `batch.config.forecast.mask` | required |  | pixels used by the statistic |
| `nuisances` | mapping, see `batch.config.forecast.nuisances` | `{}` |  | parameters profiled in the likelihood |
| `noise_covariance` | null or path to an existing .npy file | `null` |  | dense covariance over the full image |

## batch.config.forecast.positions (kind: grid)

Selected by `kind: grid`.

| key | value | default | unit | meaning |
|---|---|---|---|---|
| `spacing_arcsec` | number > 0 | required | arcsec | lattice spacing |
| `half_width_arcsec` | number > 0 | required | arcsec | half width of the square lattice |
| `annulus` | null or mapping, see `batch.config.forecast.positions.annulus` | `null` |  | retain nodes in this closed annulus |

- half_width_arcsec >= spacing_arcsec

## batch.config.forecast.positions.annulus

| key | value | default | unit | meaning |
|---|---|---|---|---|
| `inner_arcsec` | number >= 0 | required | arcsec | inner radius of the closed annulus |
| `outer_arcsec` | number > 0 | required | arcsec | outer radius of the closed annulus |

- inner_arcsec < outer_arcsec

## batch.config.forecast.positions (kind: ring)

Selected by `kind: ring`.

| key | value | default | unit | meaning |
|---|---|---|---|---|
| `count` | integer >= 1 | required |  | number of equally spaced positions |
| `radius` | one of: einstein_radius, critical_curve or number > 0 | `einstein_radius` | arcsec | radius about the lens centre |
| `offset_arcsec` | number | `0.0` | arcsec | offset added to the ring radius |

- a numeric radius plus offset is positive

## batch.config.forecast.positions (kind: explicit)

Selected by `kind: explicit`.

| key | value | default | unit | meaning |
|---|---|---|---|---|
| `positions_yx` | list of at least 1 items, each pair, each number | required | arcsec | positions in (y, x) order |

- no duplicate position rows

## batch.config.forecast.mask (kind: all_pixels)

Selected by `kind: all_pixels`.

No keys.

## batch.config.forecast.mask (kind: source_snr)

Selected by `kind: source_snr`.

| key | value | default | unit | meaning |
|---|---|---|---|---|
| `snr_min` | number > 0 | required |  | minimum source-plane light signal-to-noise |

## batch.config.forecast.mask (kind: annulus)

Selected by `kind: annulus`.

| key | value | default | unit | meaning |
|---|---|---|---|---|
| `inner_arcsec` | number >= 0 | required | arcsec | inner radius of the closed annulus |
| `outer_arcsec` | number > 0 | required | arcsec | outer radius of the closed annulus |
| `about` | one of: lens, grid | `lens` |  | centre of the annulus |

- inner_arcsec < outer_arcsec

## batch.config.forecast.mask (kind: psf_border)

Selected by `kind: psf_border`.

No keys.

## batch.config.forecast.nuisances

| key | value | default | unit | meaning |
|---|---|---|---|---|
| `fixed` | list, each non-empty text, no repeats | `[]` |  | scene parameter names or fnmatch patterns held fixed |
| `steps` | mapping of non-empty text to number > 0 | `{}` |  | finite-difference steps per kind or scene parameter name |
| `priors` | mapping of non-empty text to number > 0 | `{}` |  | Gaussian sigmas per scene parameter name |
| `background_offset` | true or false | `true` |  | profile a constant ADU offset |
| `wavefront` | null or mapping, see `batch.config.forecast.nuisances.wavefront` | `null` |  | wavefront-mode nuisances |

## batch.config.forecast.nuisances.wavefront

| key | value | default | unit | meaning |
|---|---|---|---|---|
| `modes` | mapping, see `batch.config.forecast.nuisances.wavefront.modes` | required |  | wavefront families and modes to profile |
| `step_nm` | number > 0 or mapping of one of: segment_hexikes, zernikes to number > 0 | `1.0` | nm | central-difference step, scalar or per family |
| `prior_sigma_nm` | null or number > 0 or mapping of one of: segment_hexikes, zernikes to number > 0 | `null` | nm | Gaussian sigma, scalar or per family |

- family scales cover every selected family

## batch.config.forecast.nuisances.wavefront.modes

| key | value | default | unit | meaning |
|---|---|---|---|---|
| `segment_hexikes` | null or mapping, see `batch.config.forecast.nuisances.wavefront.modes.segment_hexikes` | `null` |  | segment hexike modes |
| `zernikes` | null or mapping, see `batch.config.forecast.nuisances.wavefront.modes.zernikes` | `null` |  | global Zernike modes |

- at least one family; global Zernike Noll 1 is refused

## batch.config.forecast.nuisances.wavefront.modes.segment_hexikes

| key | value | default | unit | meaning |
|---|---|---|---|---|
| `segments` | one of: all or list of at least 1 items, each integer >= 0, no repeats | required |  | segment indices, or all: every active segment of the model pupil |
| `nolls` | list of at least 1 items, each integer >= 1, no repeats | required |  | hexike Noll indices on each listed segment |

## batch.config.forecast.nuisances.wavefront.modes.zernikes

| key | value | default | unit | meaning |
|---|---|---|---|---|
| `nolls` | list of at least 1 items, each integer >= 1, no repeats | required |  | global Zernike Noll indices (Noll 1 is refused) |

## batch.population

| key | value | default | unit | meaning |
|---|---|---|---|---|
| `variables` | mapping of text matching [A-Za-z_][A-Za-z0-9_]* to mapping selected by kind: constant, choice, uniform, log_uniform, normal, truncated_normal, lognormal, truncated_lognormal, vector, polar_offset, ell_comps, shear_components, multipole_components, function, see `batch.population.variables.<key>` | required |  | ordered variables |
| `copulas` | mapping of text matching [A-Za-z_][A-Za-z0-9_]* to mapping, see `batch.population.copulas.<key>` | `{}` |  | Gaussian copulas |
| `catalog` | null or mapping, see `batch.population.catalog` | `null` |  | catalog |
| `bind` | mapping with at least 1 entries of non-empty text to text matching [A-Za-z_][A-Za-z0-9_]*(?:\[[0-9]+\])? | required |  | existing effective configuration paths |
| `max_attempts` | integer >= 1 | `1` |  | rejection limit |
| `seed` | integer >= 0 | required |  | population stream seed |
| `count` | integer >= 1 | required |  | member count |
| `start` | integer >= 0 | `0` |  | first absolute population index |
| `name_prefix` | text matching [A-Za-z0-9_.-]+ | `system` |  | population member name prefix |

## batch.population.variables.<key> (kind: constant)

Selected by `kind: constant`.

| key | value | default | unit | meaning |
|---|---|---|---|---|
| `value` | any value | required |  | constant JSON-shaped value |

## batch.population.variables.<key> (kind: choice)

Selected by `kind: choice`.

| key | value | default | unit | meaning |
|---|---|---|---|---|
| `values` | list of at least 1 items, each any value | required |  | choice values |
| `weights` | null or list, each number >= 0 | `null` |  | choice weights |

## batch.population.variables.<key> (kind: uniform)

Selected by `kind: uniform`.

| key | value | default | unit | meaning |
|---|---|---|---|---|
| `low` | number or mapping, see `batch.population.variables.<key>.low` | required |  | finite numeric parameter or earlier reference |
| `high` | number or mapping, see `batch.population.variables.<key>.high` | required |  | finite numeric parameter or earlier reference |

## batch.population.variables.<key>.low

| key | value | default | unit | meaning |
|---|---|---|---|---|
| `var` | text matching [A-Za-z_][A-Za-z0-9_]*(?:\[[0-9]+\])? | required |  | earlier variable, optionally indexed |

## batch.population.variables.<key>.high

| key | value | default | unit | meaning |
|---|---|---|---|---|
| `var` | text matching [A-Za-z_][A-Za-z0-9_]*(?:\[[0-9]+\])? | required |  | earlier variable, optionally indexed |

## batch.population.variables.<key> (kind: log_uniform)

Selected by `kind: log_uniform`.

| key | value | default | unit | meaning |
|---|---|---|---|---|
| `low` | number or mapping, see `batch.population.variables.<key>.low` | required |  | finite numeric parameter or earlier reference |
| `high` | number or mapping, see `batch.population.variables.<key>.high` | required |  | finite numeric parameter or earlier reference |

## batch.population.variables.<key> (kind: normal)

Selected by `kind: normal`.

| key | value | default | unit | meaning |
|---|---|---|---|---|
| `mean` | number or mapping, see `batch.population.variables.<key>.mean` | required |  | finite numeric parameter or earlier reference |
| `std` | number or mapping, see `batch.population.variables.<key>.std` | required |  | finite numeric parameter or earlier reference |

## batch.population.variables.<key>.mean

| key | value | default | unit | meaning |
|---|---|---|---|---|
| `var` | text matching [A-Za-z_][A-Za-z0-9_]*(?:\[[0-9]+\])? | required |  | earlier variable, optionally indexed |

## batch.population.variables.<key>.std

| key | value | default | unit | meaning |
|---|---|---|---|---|
| `var` | text matching [A-Za-z_][A-Za-z0-9_]*(?:\[[0-9]+\])? | required |  | earlier variable, optionally indexed |

## batch.population.variables.<key> (kind: truncated_normal)

Selected by `kind: truncated_normal`.

| key | value | default | unit | meaning |
|---|---|---|---|---|
| `mean` | number or mapping, see `batch.population.variables.<key>.mean` | required |  | finite numeric parameter or earlier reference |
| `std` | number or mapping, see `batch.population.variables.<key>.std` | required |  | finite numeric parameter or earlier reference |
| `low` | number or mapping, see `batch.population.variables.<key>.low` | required |  | finite numeric parameter or earlier reference |
| `high` | number or mapping, see `batch.population.variables.<key>.high` | required |  | finite numeric parameter or earlier reference |

## batch.population.variables.<key> (kind: lognormal)

Selected by `kind: lognormal`.

| key | value | default | unit | meaning |
|---|---|---|---|---|
| `median` | number or mapping, see `batch.population.variables.<key>.median` | required |  | finite numeric parameter or earlier reference |
| `sigma_ln` | number or mapping, see `batch.population.variables.<key>.sigma_ln` | required |  | finite numeric parameter or earlier reference |

## batch.population.variables.<key>.median

| key | value | default | unit | meaning |
|---|---|---|---|---|
| `var` | text matching [A-Za-z_][A-Za-z0-9_]*(?:\[[0-9]+\])? | required |  | earlier variable, optionally indexed |

## batch.population.variables.<key>.sigma_ln

| key | value | default | unit | meaning |
|---|---|---|---|---|
| `var` | text matching [A-Za-z_][A-Za-z0-9_]*(?:\[[0-9]+\])? | required |  | earlier variable, optionally indexed |

## batch.population.variables.<key> (kind: truncated_lognormal)

Selected by `kind: truncated_lognormal`.

| key | value | default | unit | meaning |
|---|---|---|---|---|
| `median` | number or mapping, see `batch.population.variables.<key>.median` | required |  | finite numeric parameter or earlier reference |
| `sigma_ln` | number or mapping, see `batch.population.variables.<key>.sigma_ln` | required |  | finite numeric parameter or earlier reference |
| `low` | number or mapping, see `batch.population.variables.<key>.low` | required |  | finite numeric parameter or earlier reference |
| `high` | number or mapping, see `batch.population.variables.<key>.high` | required |  | finite numeric parameter or earlier reference |

## batch.population.variables.<key> (kind: vector)

Selected by `kind: vector`.

| key | value | default | unit | meaning |
|---|---|---|---|---|
| `of` | list of at least 2 items, each number or mapping | required |  | vector components |

## batch.population.variables.<key>.of[i]

| key | value | default | unit | meaning |
|---|---|---|---|---|
| `var` | text matching [A-Za-z_][A-Za-z0-9_]*(?:\[[0-9]+\])? | required |  | earlier variable, optionally indexed |

## batch.population.variables.<key> (kind: polar_offset)

Selected by `kind: polar_offset`.

| key | value | default | unit | meaning |
|---|---|---|---|---|
| `radius` | number or mapping, see `batch.population.variables.<key>.radius` | required |  | radius |
| `angle_deg` | number or mapping, see `batch.population.variables.<key>.angle_deg` | required |  | angle in degrees |
| `centre_y` | number or mapping, see `batch.population.variables.<key>.centre_y` | `0.0` |  | centre y |
| `centre_x` | number or mapping, see `batch.population.variables.<key>.centre_x` | `0.0` |  | centre x |

## batch.population.variables.<key>.radius

| key | value | default | unit | meaning |
|---|---|---|---|---|
| `var` | text matching [A-Za-z_][A-Za-z0-9_]*(?:\[[0-9]+\])? | required |  | earlier variable, optionally indexed |

## batch.population.variables.<key>.angle_deg

| key | value | default | unit | meaning |
|---|---|---|---|---|
| `var` | text matching [A-Za-z_][A-Za-z0-9_]*(?:\[[0-9]+\])? | required |  | earlier variable, optionally indexed |

## batch.population.variables.<key>.centre_y

| key | value | default | unit | meaning |
|---|---|---|---|---|
| `var` | text matching [A-Za-z_][A-Za-z0-9_]*(?:\[[0-9]+\])? | required |  | earlier variable, optionally indexed |

## batch.population.variables.<key>.centre_x

| key | value | default | unit | meaning |
|---|---|---|---|---|
| `var` | text matching [A-Za-z_][A-Za-z0-9_]*(?:\[[0-9]+\])? | required |  | earlier variable, optionally indexed |

## batch.population.variables.<key> (kind: ell_comps)

Selected by `kind: ell_comps`.

| key | value | default | unit | meaning |
|---|---|---|---|---|
| `axis_ratio` | number or mapping, see `batch.population.variables.<key>.axis_ratio` | required |  | minor/major axis ratio |
| `angle_deg` | number or mapping, see `batch.population.variables.<key>.angle_deg` | required |  | major axis angle |

## batch.population.variables.<key>.axis_ratio

| key | value | default | unit | meaning |
|---|---|---|---|---|
| `var` | text matching [A-Za-z_][A-Za-z0-9_]*(?:\[[0-9]+\])? | required |  | earlier variable, optionally indexed |

## batch.population.variables.<key> (kind: shear_components)

Selected by `kind: shear_components`.

| key | value | default | unit | meaning |
|---|---|---|---|---|
| `magnitude` | number or mapping, see `batch.population.variables.<key>.magnitude` | required |  | shear magnitude |
| `angle_deg` | number or mapping, see `batch.population.variables.<key>.angle_deg` | required |  | shear angle |

## batch.population.variables.<key>.magnitude

| key | value | default | unit | meaning |
|---|---|---|---|---|
| `var` | text matching [A-Za-z_][A-Za-z0-9_]*(?:\[[0-9]+\])? | required |  | earlier variable, optionally indexed |

## batch.population.variables.<key> (kind: multipole_components)

Selected by `kind: multipole_components`.

| key | value | default | unit | meaning |
|---|---|---|---|---|
| `strength` | number or mapping, see `batch.population.variables.<key>.strength` | required |  | multipole strength |
| `angle_deg` | number or mapping, see `batch.population.variables.<key>.angle_deg` | required |  | multipole angle |
| `order` | integer >= 1 | required |  | multipole order |

## batch.population.variables.<key>.strength

| key | value | default | unit | meaning |
|---|---|---|---|---|
| `var` | text matching [A-Za-z_][A-Za-z0-9_]*(?:\[[0-9]+\])? | required |  | earlier variable, optionally indexed |

## batch.population.variables.<key> (kind: function)

Selected by `kind: function`.

| key | value | default | unit | meaning |
|---|---|---|---|---|
| `function` | text matching [A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*:[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)* | required |  | module:function |
| `inputs` | mapping of non-empty text to number or mapping or list, each number or mapping | `{}` |  | keyword inputs |

## batch.population.variables.<key>.inputs.<key>

| key | value | default | unit | meaning |
|---|---|---|---|---|
| `var` | text matching [A-Za-z_][A-Za-z0-9_]*(?:\[[0-9]+\])? | required |  | earlier variable, optionally indexed |

## batch.population.variables.<key>.inputs.<key>[i]

| key | value | default | unit | meaning |
|---|---|---|---|---|
| `var` | text matching [A-Za-z_][A-Za-z0-9_]*(?:\[[0-9]+\])? | required |  | earlier variable, optionally indexed |

## batch.population.copulas.<key>

| key | value | default | unit | meaning |
|---|---|---|---|---|
| `variables` | list of at least 2 items, each text matching [A-Za-z_][A-Za-z0-9_]*, no repeats | required |  | coupled distributions |
| `correlation` | list of at least 2 items, each list of at least 2 items, each number | required |  | normal-score correlation matrix |

## batch.population.catalog

| key | value | default | unit | meaning |
|---|---|---|---|---|
| `path` | non-empty text | required |  | CSV/NPZ path |
| `columns` | mapping of non-empty text to non-empty text | `{}` |  | numeric columns |
| `text_columns` | mapping of non-empty text to non-empty text | `{}` |  | text columns |

## batch.arms[i]

| key | value | default | unit | meaning |
|---|---|---|---|---|
| `name` | text matching [A-Za-z0-9_.-]+ | required |  | arm name |
| `overrides` | mapping | `{}` |  | configuration overrides |
| `directions` | null or integer >= 1 | `null` |  | paired knowledge-error directions |

## batch.simulate

| key | value | default | unit | meaning |
|---|---|---|---|---|
| `inject` | true or false | required |  | inject the configured scene hypothesis |
| `noise` | true or false | required |  | draw detector noise |
| `replicates` | integer >= 1 | `1` |  | independent noise replicates |
| `arms` | null or list of at least 1 items, each text matching [A-Za-z0-9_.-]+, no repeats | `null` |  | participating arm names |

## batch.forecast

| key | value | default | unit | meaning |
|---|---|---|---|---|
| `masses_msun` | list of at least 1 items, each number > 0, no repeats | required |  | forecast masses |
| `arms` | null or list of at least 1 items, each text matching [A-Za-z0-9_.-]+, no repeats | `null` |  | participating arm names |

## batch.nonlinear.<key>

| key | value | default | unit | meaning |
|---|---|---|---|---|
| `arms` | null or list of at least 1 items, each text matching [A-Za-z0-9_.-]+, no repeats | `null` |  | participating arm names |
| `forecast_arm` | null or text matching [A-Za-z0-9_.-]+ | `null` |  | arm supplying trials and signed forecast references |
| `trials` | mapping by `kind` (explicit, configured, forecast_positions, forecast_argmax), see `batch.nonlinear.<key>.trials` | required |  | trial selection |
| `inject` | true or false | required |  | inject the trial |
| `noise` | true or false | required |  | draw detector noise |
| `replicates` | integer >= 1 | `1` |  | noise replicates |
| `fit` | mapping | required |  | nonlinear fit |
| `sampler` | mapping | `{n_live_smooth: 100, n_live_subhalo_fixed: 100, n_live_subhalo_search: 200, n_eff: null, n_shell: null, f_live: null, discard_exploration: null, n_like_max: null, number_of_cores: 1, use_jax: false, jax_n_batch: 100, retain_search_internal: false}` |  | sampler settings |
| `refine` | null or mapping | `null` |  | bounded refinement |
| `retry` | null or mapping, see `batch.nonlinear.<key>.retry` | `null` |  | one follow-up attempt |

## batch.nonlinear.<key>.trials (kind: explicit)

Selected by `kind: explicit`.

| key | value | default | unit | meaning |
|---|---|---|---|---|
| `explicit` | list of at least 1 items, each mapping, see `batch.nonlinear.<key>.trials.explicit[i]` | required |  | trials |

## batch.nonlinear.<key>.trials.explicit[i]

| key | value | default | unit | meaning |
|---|---|---|---|---|
| `members` | one of: all, integer >= 0, or list of at least 1 items, each integer >= 0, no repeats | `all` |  | member indices |
| `mass_msun` | number > 0 | required |  | trial halo mass |
| `position_yx` | pair, each number | required |  | trial position in arcseconds |

## batch.nonlinear.<key>.trials (kind: configured)

Selected by `kind: configured`.

No keys.

## batch.nonlinear.<key>.trials (kind: forecast_positions)

Selected by `kind: forecast_positions`.

| key | value | default | unit | meaning |
|---|---|---|---|---|
| `masses_msun` | list of at least 1 items, each number > 0, no repeats | required |  | trial masses present in forecast |

## batch.nonlinear.<key>.trials (kind: forecast_argmax)

Selected by `kind: forecast_argmax`.

| key | value | default | unit | meaning |
|---|---|---|---|---|
| `masses_msun` | list of at least 1 items, each number > 0, no repeats | required |  | trial masses present in forecast |
| `aperture_radius_arcsec` | null or number > 0 | `null` |  | closed selection disc |

## batch.nonlinear.<key>.retry

| key | value | default | unit | meaning |
|---|---|---|---|---|
| `acceptance` | mapping, see `batch.nonlinear.<key>.retry.acceptance` | required |  | accepted outcomes |
| `require_retained_state` | true or false | required |  | require searched roles to retain backend state |
| `stationarity_tolerance` | null or number > 0 | required |  | None is the paper rule; positive bounds projected gradient |
| `sampler` | mapping | `{}` |  | sampler overrides for attempt1 |
| `refine` | mapping | `{}` |  | refinement overrides for attempt1 |

## batch.nonlinear.<key>.retry.acceptance

| key | value | default | unit | meaning |
|---|---|---|---|---|
| `smooth` | list of at least 1 items, each one of: accepted_repeatable_profile, unresolved_optimization, sampler_only, verified_zero_residual_anchor, failed, no repeats | required |  | accepted role outcomes |
| `subhalo` | list of at least 1 items, each one of: accepted_repeatable_profile, unresolved_optimization, sampler_only, verified_zero_residual_anchor, failed, no repeats | required |  | accepted role outcomes |

## batch.execution

| key | value | default | unit | meaning |
|---|---|---|---|---|
| `engine` | one of: reference, jax | `reference` |  | forecast engine |
| `reference_workers` | integer >= 1 | `1` |  | reference workers inside each batch worker |
| `batch_size` | integer >= 1 | `16` |  | JAX position batch size |
| `devices` | one of: cpu or list of at least 1 items, each integer >= 0, no repeats | `cpu` |  | CPU or indices of the parent visible devices |
| `workers_per_device` | integer >= 1 | `1` |  | workers sharing each assigned device |
| `threads_per_worker` | integer >= 1 | `1` |  | BLAS threads per batch worker |
| `preparation_cache_size` | integer >= 1 | `2` |  | bounded preparations retained per worker |
| `memory_fraction` | number in (0, 1] | `0.75` |  | total preallocated fraction per device |
| `training_workers` | integer >= 1 | `1` |  | backend emulator-training workers per batch worker |

## fit

| key | value | default | unit | meaning |
|---|---|---|---|---|
| `mode` | one of: fixed_template, local_search, freed | required |  | subhalo model of H1: fixed at the hypothesis, centre free, or centre and mass free |
| `mask` | one of: all_pixels_minus_psf_border, forecast_mask_minus_psf_border | `all_pixels_minus_psf_border` |  | fitted pixels before the PSF border is removed: all pixels, or the forecast mask |
| `prior_widths` | mapping, see `fit.prior_widths` | `{}` |  | prior boxes of the fitted parameters |
| `mass_support` | null or mapping, see `fit.mass_support` | `null` |  | freed mass prior; required with mode freed |
| `h1` | one of: search, truth_anchor | `search` |  | H1 by a sampler search, or by the truth vector on expected data |
| `anchor_chi2_tolerance` | number > 0 | `1e-08` |  | largest chi-square at which the truth vector is the H1 maximum |

- mass_support is required with mode freed and refused otherwise

## fit.prior_widths

| key | value | default | unit | meaning |
|---|---|---|---|---|
| `rules` | mapping, see `fit.prior_widths.rules` | `{}` |  | box rules keyed `<galaxy>.<parameter kind>` |
| `subhalo_local_window_arcsec` | number > 0 | `0.03` | arcsec | half width of the subhalo centre box in local_search |
| `subhalo_freed_window_arcsec` | number > 0 | `0.15` | arcsec | half width of the subhalo centre box in freed |

## fit.prior_widths.rules

| key | value | default | unit | meaning |
|---|---|---|---|---|
| `lens.amplitude` | mapping, see `fit.prior_widths.rules.lens.amplitude` | `{}` |  | box rule of `lens.amplitude` parameters |
| `lens.einstein_radius` | mapping, see `fit.prior_widths.rules.lens.einstein_radius` | `{}` |  | box rule of `lens.einstein_radius` parameters |
| `lens.ellipticity` | mapping, see `fit.prior_widths.rules.lens.ellipticity` | `{}` |  | box rule of `lens.ellipticity` parameters |
| `lens.multipole` | mapping, see `fit.prior_widths.rules.lens.multipole` | `{}` |  | box rule of `lens.multipole` parameters |
| `lens.orientation` | mapping, see `fit.prior_widths.rules.lens.orientation` | `{}` |  | box rule of `lens.orientation` parameters |
| `lens.position` | mapping, see `fit.prior_widths.rules.lens.position` | `{}` |  | box rule of `lens.position` parameters |
| `lens.sersic_index` | mapping, see `fit.prior_widths.rules.lens.sersic_index` | `{}` |  | box rule of `lens.sersic_index` parameters |
| `lens.shear` | mapping, see `fit.prior_widths.rules.lens.shear` | `{}` |  | box rule of `lens.shear` parameters |
| `lens.size` | mapping, see `fit.prior_widths.rules.lens.size` | `{}` |  | box rule of `lens.size` parameters |
| `lens.slope` | mapping, see `fit.prior_widths.rules.lens.slope` | `{}` |  | box rule of `lens.slope` parameters |
| `source.amplitude` | mapping, see `fit.prior_widths.rules.source.amplitude` | `{}` |  | box rule of `source.amplitude` parameters |
| `source.ellipticity` | mapping, see `fit.prior_widths.rules.source.ellipticity` | `{}` |  | box rule of `source.ellipticity` parameters |
| `source.orientation` | mapping, see `fit.prior_widths.rules.source.orientation` | `{}` |  | box rule of `source.orientation` parameters |
| `source.position` | mapping, see `fit.prior_widths.rules.source.position` | `{}` |  | box rule of `source.position` parameters |
| `source.sersic_index` | mapping, see `fit.prior_widths.rules.source.sersic_index` | `{}` |  | box rule of `source.sersic_index` parameters |
| `source.size` | mapping, see `fit.prior_widths.rules.source.size` | `{}` |  | box rule of `source.size` parameters |

## fit.prior_widths.rules.lens.amplitude

| key | value | default | unit | meaning |
|---|---|---|---|---|
| `half_width` | number > 0 | `0.5` |  | half width of the box: absolute, or a fraction of \|truth\| when fractional |
| `fractional` | true or false | `true` |  | the half width is a fraction of \|truth\| |
| `clip` | null or pair, each number | `null` |  | open interval (lower, upper) the box is clipped into |

- clip lower < clip upper

## fit.prior_widths.rules.lens.einstein_radius

| key | value | default | unit | meaning |
|---|---|---|---|---|
| `half_width` | number > 0 | `0.01` |  | half width of the box: absolute, or a fraction of \|truth\| when fractional |
| `fractional` | true or false | `false` |  | the half width is a fraction of \|truth\| |
| `clip` | null or pair, each number | `null` |  | open interval (lower, upper) the box is clipped into |

- clip lower < clip upper

## fit.prior_widths.rules.lens.ellipticity

| key | value | default | unit | meaning |
|---|---|---|---|---|
| `half_width` | number > 0 | `0.02` |  | half width of the box: absolute, or a fraction of \|truth\| when fractional |
| `fractional` | true or false | `false` |  | the half width is a fraction of \|truth\| |
| `clip` | null or pair, each number | `[-0.9, 0.9]` |  | open interval (lower, upper) the box is clipped into |

- clip lower < clip upper

## fit.prior_widths.rules.lens.multipole

| key | value | default | unit | meaning |
|---|---|---|---|---|
| `half_width` | number > 0 | `0.01` |  | half width of the box: absolute, or a fraction of \|truth\| when fractional |
| `fractional` | true or false | `false` |  | the half width is a fraction of \|truth\| |
| `clip` | null or pair, each number | `null` |  | open interval (lower, upper) the box is clipped into |

- clip lower < clip upper

## fit.prior_widths.rules.lens.orientation

| key | value | default | unit | meaning |
|---|---|---|---|---|
| `half_width` | number > 0 | `5.0` |  | half width of the box: absolute, or a fraction of \|truth\| when fractional |
| `fractional` | true or false | `false` |  | the half width is a fraction of \|truth\| |
| `clip` | null or pair, each number | `null` |  | open interval (lower, upper) the box is clipped into |

- clip lower < clip upper

## fit.prior_widths.rules.lens.position

| key | value | default | unit | meaning |
|---|---|---|---|---|
| `half_width` | number > 0 | `0.005` |  | half width of the box: absolute, or a fraction of \|truth\| when fractional |
| `fractional` | true or false | `false` |  | the half width is a fraction of \|truth\| |
| `clip` | null or pair, each number | `null` |  | open interval (lower, upper) the box is clipped into |

- clip lower < clip upper

## fit.prior_widths.rules.lens.sersic_index

| key | value | default | unit | meaning |
|---|---|---|---|---|
| `half_width` | number > 0 | `0.3` |  | half width of the box: absolute, or a fraction of \|truth\| when fractional |
| `fractional` | true or false | `true` |  | the half width is a fraction of \|truth\| |
| `clip` | null or pair, each number | `null` |  | open interval (lower, upper) the box is clipped into |

- clip lower < clip upper

## fit.prior_widths.rules.lens.shear

| key | value | default | unit | meaning |
|---|---|---|---|---|
| `half_width` | number > 0 | `0.01` |  | half width of the box: absolute, or a fraction of \|truth\| when fractional |
| `fractional` | true or false | `false` |  | the half width is a fraction of \|truth\| |
| `clip` | null or pair, each number | `null` |  | open interval (lower, upper) the box is clipped into |

- clip lower < clip upper

## fit.prior_widths.rules.lens.size

| key | value | default | unit | meaning |
|---|---|---|---|---|
| `half_width` | number > 0 | `0.3` |  | half width of the box: absolute, or a fraction of \|truth\| when fractional |
| `fractional` | true or false | `true` |  | the half width is a fraction of \|truth\| |
| `clip` | null or pair, each number | `null` |  | open interval (lower, upper) the box is clipped into |

- clip lower < clip upper

## fit.prior_widths.rules.lens.slope

| key | value | default | unit | meaning |
|---|---|---|---|---|
| `half_width` | number > 0 | `0.05` |  | half width of the box: absolute, or a fraction of \|truth\| when fractional |
| `fractional` | true or false | `false` |  | the half width is a fraction of \|truth\| |
| `clip` | null or pair, each number | `null` |  | open interval (lower, upper) the box is clipped into |

- clip lower < clip upper

## fit.prior_widths.rules.source.amplitude

| key | value | default | unit | meaning |
|---|---|---|---|---|
| `half_width` | number > 0 | `0.5` |  | half width of the box: absolute, or a fraction of \|truth\| when fractional |
| `fractional` | true or false | `true` |  | the half width is a fraction of \|truth\| |
| `clip` | null or pair, each number | `null` |  | open interval (lower, upper) the box is clipped into |

- clip lower < clip upper

## fit.prior_widths.rules.source.ellipticity

| key | value | default | unit | meaning |
|---|---|---|---|---|
| `half_width` | number > 0 | `0.05` |  | half width of the box: absolute, or a fraction of \|truth\| when fractional |
| `fractional` | true or false | `false` |  | the half width is a fraction of \|truth\| |
| `clip` | null or pair, each number | `[-0.9, 0.9]` |  | open interval (lower, upper) the box is clipped into |

- clip lower < clip upper

## fit.prior_widths.rules.source.orientation

| key | value | default | unit | meaning |
|---|---|---|---|---|
| `half_width` | number > 0 | `5.0` |  | half width of the box: absolute, or a fraction of \|truth\| when fractional |
| `fractional` | true or false | `false` |  | the half width is a fraction of \|truth\| |
| `clip` | null or pair, each number | `null` |  | open interval (lower, upper) the box is clipped into |

- clip lower < clip upper

## fit.prior_widths.rules.source.position

| key | value | default | unit | meaning |
|---|---|---|---|---|
| `half_width` | number > 0 | `0.01` |  | half width of the box: absolute, or a fraction of \|truth\| when fractional |
| `fractional` | true or false | `false` |  | the half width is a fraction of \|truth\| |
| `clip` | null or pair, each number | `null` |  | open interval (lower, upper) the box is clipped into |

- clip lower < clip upper

## fit.prior_widths.rules.source.sersic_index

| key | value | default | unit | meaning |
|---|---|---|---|---|
| `half_width` | number > 0 | `0.3` |  | half width of the box: absolute, or a fraction of \|truth\| when fractional |
| `fractional` | true or false | `true` |  | the half width is a fraction of \|truth\| |
| `clip` | null or pair, each number | `null` |  | open interval (lower, upper) the box is clipped into |

- clip lower < clip upper

## fit.prior_widths.rules.source.size

| key | value | default | unit | meaning |
|---|---|---|---|---|
| `half_width` | number > 0 | `0.3` |  | half width of the box: absolute, or a fraction of \|truth\| when fractional |
| `fractional` | true or false | `true` |  | the half width is a fraction of \|truth\| |
| `clip` | null or pair, each number | `null` |  | open interval (lower, upper) the box is clipped into |

- clip lower < clip upper

## fit.mass_support

| key | value | default | unit | meaning |
|---|---|---|---|---|
| `log10_mass_min` | number | required | dex | lower end of the freed mass prior, log10(M200 / Msun) (point mass: log10(M / Msun)) |
| `log10_mass_max` | number | required | dex | upper end of the freed mass prior; must exceed log10_mass_min |

- log10_mass_min < log10_mass_max

## sampler

| key | value | default | unit | meaning |
|---|---|---|---|---|
| `n_live_smooth` | integer >= 1 | `100` |  | live points of the H0 search |
| `n_live_subhalo_fixed` | integer >= 1 | `100` |  | live points of the H1 search in mode fixed_template |
| `n_live_subhalo_search` | integer >= 1 | `200` |  | live points of the H1 search in modes local_search and freed |
| `n_eff` | null or number > 0 | `null` |  | effective sample size at which Nautilus stops; null: the backend default |
| `n_shell` | null or integer >= 1 | `null` |  | minimum points per shell before Nautilus stops; null: the backend default |
| `f_live` | null or number in (0, 1] | `null` |  | live-set evidence fraction at which exploration ends; null: the backend default |
| `discard_exploration` | null or true or false | `null` |  | drop exploration-phase points from the posterior; null: the backend default |
| `n_like_max` | null or integer >= 1 | `null` |  | largest number of likelihood calls; null: no limit |
| `number_of_cores` | integer >= 1 | `1` |  | sampler processes; 1 with use_jax |
| `use_jax` | true or false | `false` |  | JAX likelihood, vectorized over batches of vectors |
| `jax_n_batch` | integer >= 1 | `100` |  | vectors per JAX likelihood batch |
| `retain_search_internal` | true or false | `false` |  | keep the raw Nautilus state on disk after the fit |

- use_jax requires number_of_cores 1

## refine

| key | value | default | unit | meaning |
|---|---|---|---|---|
| `original_start_count` | integer >= 1 | `8` |  | posterior samples started from, besides the sampler maximum |
| `start_separation_normalized_l2` | number > 0 | `0.05` |  | a sample is a new start when this far (unit-box L2) from every start |
| `start_separation_posterior_sigma` | number > 0 | `1.0` |  | or when this many posterior sigmas from every start |
| `maxiter` | integer >= 1 | `500` |  | L-BFGS-B iterations per start |
| `ftol` | number >= 0 | `0.0` |  | L-BFGS-B relative reduction tolerance per start |
| `gtol` | number >= 0 | `1e-10` |  | L-BFGS-B projected-gradient tolerance per start |
| `maxls` | integer >= 1 | `50` |  | line-search steps per iteration |
| `repeat_maxiter` | integer >= 1 | `1000` |  | iterations of the tighter repeat from the best point |
| `repeat_ftol` | number >= 0 | `0.0` |  | relative reduction tolerance of the tighter repeat |
| `repeat_gtol` | number >= 0 | `1e-12` |  | projected-gradient tolerance of the tighter repeat |
| `support_log_likelihood_tolerance` | number >= 0 | `0.1` |  | a start supports the best when it ends within this log L |
| `repeat_log_likelihood_tolerance` | number >= 0 | `0.1` |  | the tighter repeat may move the best by at most this log L |
| `minimum_distinct_original_start_support` | integer >= 1 | `2` |  | supporting original starts needed for acceptance |
| `scalar_residual_tolerance` | number > 0 | `0.0001` |  | largest residual and direct log L inconsistency at the best |

## classification

| key | value | default | unit | meaning |
|---|---|---|---|---|
| `q_threshold` | number > 0 | required |  | required detection threshold |
| `marginal_half_width` | number >= 0 | required |  | open half width about the threshold |
| `acceptance` | mapping, see `classification.acceptance` | required |  | role statuses accepted by this rule |
| `require_retained_state` | true or false | required |  | require the raw state of searched roles |
| `retry_log_likelihood_tolerance` | number >= 0 | required |  | allowed decrease in each role on retry |
| `stationarity_tolerance` | null or number > 0 | required |  | None preserves the paper rule; positive bounds the projected gradient |

## classification.acceptance

| key | value | default | unit | meaning |
|---|---|---|---|---|
| `smooth` | list, each one of: accepted_repeatable_profile, unresolved_optimization, sampler_only, verified_zero_residual_anchor, failed, no repeats | required |  | accepted smooth role statuses |
| `subhalo` | list, each one of: accepted_repeatable_profile, unresolved_optimization, sampler_only, verified_zero_residual_anchor, failed, no repeats | required |  | accepted subhalo role statuses |

