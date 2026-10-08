# Configuration files

A configuration is a YAML mapping with these top-level sections:

| Section | Describes |
|---|---|
| `run_name` | A label for the run. It is not part of the scientific identity. |
| `seed` | The seed for random choices in the scene, such as random subhalo placement. |
| `cosmology` | A named flat ΛCDM cosmology, such as `Planck15`, or custom parameters. |
| `scene` | The pixel grid, lens, source, subhalo model and optional injected subhalo. |
| `psf` | The truth PSF and, optionally, a different model PSF. |
| `instrument` | Detector gain, read noise, dark current, and optionally a bandpass and collecting area. |
| `observation` | Exposure time, number of exposures and sky level. |
| `forecast` | Trial positions, the pixel mask and the nuisance parameters. Needed only for forecasts. |

The [configuration reference](../configuration.md) lists every key with its type,
default and unit. You can also print it, or one section of it, from the command line:

```bash
hwoslaps reference
hwoslaps reference scene.subhalo
```

## Components and types

The lens and source are built from named components. The name is yours; it labels the
component in results and parameter names. The `type` selects the profile:

```yaml
scene:
  lens:
    redshift: 0.2
    mass:
      main: {type: Isothermal, centre: [0.0, 0.0], einstein_radius: 1.0, ell_comps: [0.1, 0.0]}
      shear: {type: ExternalShear, gamma_1: 0.02, gamma_2: 0.0}
  source:
    redshift: 0.6
    light:
      disk: {type: Exponential, centre: [0.0, 0.05], ell_comps: [0.1, 0.0],
             effective_radius: 0.1, intensity: 1.0}
```

| Kind of component | Types |
|---|---|
| Lens mass | `Isothermal`, `PowerLaw`, `ExternalShear`; `Isothermal` and `PowerLaw` accept m = 3 and m = 4 multipoles |
| Lens or source light | `Exponential`, `Sersic`, `Image` (a pixelized image from a file) |
| Subhalo | `PointMass`, `SIS`, `NFW`, `TNFW` (truncated NFW) |

A component's parameters are named by their path, for example
`lens.mass.main.einstein_radius` or `source.light.disk.centre_x`. These names are used
to fix parameters, set priors and read results.

Some sections choose between alternatives with a `kind` key instead of `type`, such as
`psf.truth.kind: kernel` or `forecast.positions.kind: grid`.

## Light in physical units

A light component can be given as a surface-brightness `intensity` in detected
electrons per second per pixel, or as an AB magnitude with a spectrum. The second
form needs a bandpass and collecting area in the `instrument` section:

```yaml
scene:
  source:
    light:
      disk: {type: Exponential, centre: [-0.03, 0.08], ell_comps: [0.145, 0.251],
             effective_radius: 0.11, flux: {ab_mag: 24.845}, sed: {kind: flat_fnu}}
instrument:
  collecting_area_m2: null       # null: derive the area from the telescope pupil
  bandpass: {kind: top_hat, min_nm: 450.0, max_nm: 550.0, throughput: 0.21}
observation:
  sky: {ab_mag_per_arcsec2: 23.0}
```

hwoslaps converts the magnitude to a detected rate using the bandpass, the collecting
area and the spectrum, then sets the profile's intensity so that its total flux
matches. The [HWO reference](../examples/hwo.md) uses this form. The conversion is
recorded with each result.

## Combining files

Configurations are usually split into reusable files: one for the scene, one for
the instrument, one for the forecast. Pass them in order; later files override earlier
ones.

```bash
hwoslaps validate examples/hwo_reference/scene_smooth_ring.yaml \
                  examples/hwo_reference/instrument.yaml \
                  examples/hwo_reference/forecast.yaml
```

```python
from hwoslaps import load_config

config = load_config([
    "examples/hwo_reference/scene_smooth_ring.yaml",
    "examples/hwo_reference/instrument.yaml",
    "examples/hwo_reference/forecast.yaml",
])
```

When files are combined:

- mappings merge key by key, so a later file (an *overlay*) only needs the keys it changes;
- lists and single values replace the earlier value;
- changing a `kind` or `type` replaces that whole block, so keys from the old kind do
  not leak into the new one;
- `null` removes an optional value. For example, `flux: null` removes an AB magnitude
  so that a literal `intensity` can be used instead.

File paths inside a configuration, such as a PSF kernel or an image source, are
relative to the file that contains them. A file can be reused from any working
directory.

## Changing values

From the command line, `--set` changes one value. The value is read as YAML:

```bash
hwoslaps validate configs/minimal.yaml --set observation.exposure_time_s=4000
hwoslaps forecast configs/minimal.yaml --set scene.lens.mass.main.einstein_radius=1.2 \
    --masses 1e7 1e8 -o out/larger_ring
```

In Python, `replace` returns a new configuration with the overrides applied:

```python
longer = config.replace({"observation": {"exposure_time_s": 8000.0}})
```

Configurations are immutable; `replace` never changes the original. Every value is
checked when the configuration is built. A misspelled key fails with its full path:

```text
ConfigError: observation.exposure_tim_s: unknown key; allowed: exposure_count, exposure_time_s, sky
```

`hwoslaps validate --print` prints the complete configuration with every default
filled in. `config.to_mapping()` returns the same thing as a Python dictionary.

## YAML numbers

hwoslaps reads YAML 1.2, so `1e8`, `1.0e8` and `100000000` are all numbers. (Many YAML
readers treat `1e8` as a string.) Duplicate keys are an error.

## Digests

Every configuration has a **digest**: a SHA-256 hash of its scientific content,
including the bytes of every file it references. Two configurations with the same
digest produce the same forecast. The digest ignores `run_name`.

```python
print(config.digest())
print(config.comparison_digest())
```

The **comparison digest** ignores the model PSF. Two forecasts that differ only in
their model PSF share a comparison digest, which is how hwoslaps checks that a PSF
comparison is like for like. See [PSF knowledge error](psfs.md#psf-knowledge-error).

Results, observations and nonlinear fits all record the digest of the configuration
that made them.
