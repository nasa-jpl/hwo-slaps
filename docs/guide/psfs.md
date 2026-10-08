# PSFs and PSF errors

Every configuration has a **truth** PSF, which makes the data, and a **model** PSF,
which the analysis assumes. By default the model equals the truth. Studies of PSF
quality vary both together; studies of PSF knowledge error keep the truth fixed and
change only the model.

## Kernel PSFs

The simplest PSF is a detector kernel stored in a `.npy` (or `.npz`) file:

```yaml
psf:
  truth: {kind: kernel, path: minimal_kernel.npy, pixel_scale_arcsec: 0.1}
```

The kernel must be sampled at the detector pixel scale and have odd dimensions. It is
normalized to unit sum. With `normalize: false` it is used as it is, and must already sum
to one within 10⁻¹⁰. Add `file_sha256` to make
hwoslaps check the file's hash before reading it, so a changed file is caught.

## Optical PSFs

An optical PSF is computed from a telescope pupil with HCIPy. This is the HWO
reference telescope:

```yaml
psf:
  truth:
    kind: optical
    pupil: {kind: hex_segmented, diameter_m: 7.225765, pixels: 512, supersampling: 4,
            rings: 2, segment_point_to_point_m: 1.65, gap_m: 0.006}
    focal_length_m: 144.0
    wavelength_nm: 500.0
    detector_oversampling: 3
    kernel_shape: [999, 999]
```

`pupil`
: A `hex_segmented` mirror (rings of hexagonal segments, with gaps) or a `circular`
  aperture. Both accept a central obscuration and spiders.

`focal_length_m`, `wavelength_nm`
: Together with the detector pixel scale, these set how finely the PSF is sampled.
  hwoslaps raises an error if the PSF would be under-sampled or aliased.

`detector_oversampling`
: Sub-samples per detector pixel used to integrate the PSF over each pixel.

`kernel_shape`
: The size of the kernel in detector pixels. Larger kernels capture more of the PSF
  wings and take longer to convolve.

The kernel records the fraction of the PSF's power that falls inside it (its
*captured fraction*). For the HWO reference at 500 nm this is 0.99995.

### Wavefront errors

Wavefront errors are given in nanometres of optical path difference, as global
Zernike modes or as hexike modes on individual segments. Modes use Noll indices.

```yaml
psf:
  truth:
    wavefront:
      zernikes: {4: 10.0, 5: -3.0}                 # 10 nm of defocus, -3 nm of astigmatism
      segment_hexikes: {0: {2: 5.0}, 3: {4: 2.0}}  # segment index: {Noll index: nm}
```

Instead of listing coefficients, you can **draw** a wavefront from a prior at an exact
RMS amplitude:

```yaml
psf:
  truth:
    draw: {prior: {packaged: jwst_wss_static_v1}, amplitude_rms_nm: 35.0, seed: 7,
           family: combined}
```

`amplitude_rms_nm` is the piston-removed RMS over the illuminated pupil. `family`
chooses global modes, segment modes or both (`combined`). Two priors ship with
hwoslaps:

| Prior | Shape of the mode weights |
|---|---|
| `jwst_wss_static_v1` | A static telescope wavefront |
| `jwst_wss_drift_v1` | A wavefront drift between sensing updates |

Both are study inputs; they are not measured HWO wavefront errors. You can also supply
your own prior table as a YAML file (`prior: {path: ...}`) or a power law in radial
order (`prior: {power_law: ...}`). See the
[configuration reference](../configuration.md) for their keys.

## Model PSFs

`psf.model.kind` chooses the PSF the analysis assumes:

| `kind` | Model PSF |
|---|---|
| `matched` | The truth PSF (the default) |
| `kernel` | A different kernel file |
| `optical` | A separately described optical system |
| `wavefront` | The truth optics with replaced (`wavefront`) or added (`offset`) wavefront coefficients |
| `knowledge_error` | The truth optics plus a wavefront error drawn from a prior at a given RMS |
| `monochromatic` | A single-wavelength version of a chromatic truth PSF |

The `optical`, `wavefront`, `knowledge_error` and `monochromatic` models need an
optical truth.

When the model differs from the truth, forecasts report `q_mismatch` and
`q_spurious` in addition to `q_asimov`, and `result.detection_metric` becomes
`q_mismatch`. See [How hwoslaps works](../concepts.md#when-the-model-psf-is-wrong).

(psf-knowledge-error)=
## PSF knowledge error

A PSF knowledge-error study asks how large an error in the model PSF can be before it
changes which subhalos are detected. It compares a **matched** forecast, whose
model PSF is correct, against **mismatched** forecasts whose model PSF is wrong by a known
amount.

The HWO reference includes one such overlay. It adds a 10 nm RMS drift-shaped error to
the model PSF:

```{literalinclude} ../../examples/hwo_reference/knowledge_error.yaml
:language: yaml
```

Each seed gives a different error pattern, or **direction**, at the same RMS. A study
usually runs several directions at each of several amplitudes.

### Comparing two forecasts

`knowledge_error_areas` compares a matched and a mismatched forecast of the same
configuration on the same grid:

```python
from hwoslaps import load_forecast
from hwoslaps.analysis import aperture_selection, knowledge_error_areas

reference = load_forecast("out/matched/forecast.npz")      # the matched forecast
mismatched = load_forecast("out/ke_10nm_seed1/forecast.npz")

aperture = aperture_selection(reference, centre_yx=(0.0, 0.0), radius_arcsec=1.5)
areas = knowledge_error_areas(reference, mismatched, q_threshold=10.0,
                              min_reference_count=33, selection=aperture)
```

The two forecasts must agree in everything except the model PSF: the same comparison
digest, positions, pixel mask and nuisance parameters. hwoslaps checks this and raises an
error otherwise.

For each mass, `areas` holds counts and areas of detections, and three ratios to the area
detected by the matched forecast inside the selection:

| Field | Ratio |
|---|---|
| `detected_area_ratio` | Area detected with the wrong model PSF, divided by the matched area. This is *R* in the RASTI paper. |
| `spurious_ratio` | Area of false detections caused by the PSF error alone, divided by the matched area. This is *F* in the RASTI paper (not the information *F* of [How hwoslaps works](../concepts.md)). |
| `retention` | Area detected by both the matched and the mismatched forecast, divided by the matched area. Always at most 1. |

The ratios are `NaN` at masses where the matched forecast detects fewer than
`min_reference_count` positions, because a ratio of a few cells is too noisy to use.
`spurious_area_arcsec2` covers every position, inside the selection or not.

### Tolerances

The **tolerance** is the largest error amplitude that still passes two gates across the
directions: the 10th percentile of *R* stays at or above a minimum, and the 90th
percentile of *F* stays at or below a maximum. The RASTI paper used *Q*₁₀[*R*] ≥ 0.9 and
*Q*₉₀[*F*] ≤ 0.1, with a floor of 33 reference cells. It computed the tolerance for each
lens and mass separately over that lens's eight directions, then reported the median
across lenses.

`knowledge_error_tolerance` applies the gates to one set of directions. Pass *R*
(`detected_area_ratio`) as its first argument. Despite their names, that argument and the
`retention_*` fields of `ToleranceCriterion` take *R*, not the `retention` field of the
areas:

```python
import numpy as np
from hwoslaps.analysis import ToleranceCriterion, knowledge_error_tolerance

criterion = ToleranceCriterion(retention_quantile=0.1, retention_min=0.9,
                               spurious_quantile=0.9, spurious_max=0.1)

def tolerance_nm(reference, mismatched, mass_index, aperture):
    """Largest passing amplitude for one lens and mass.

    mismatched maps amplitude in nm to {direction: ForecastResult}.
    """
    ratio, spurious = {}, {}
    for amplitude, by_direction in mismatched.items():
        ratio[amplitude], spurious[amplitude] = {}, {}
        for direction, result in by_direction.items():
            areas = knowledge_error_areas(reference, result, q_threshold=10.0,
                                          min_reference_count=33, selection=aperture)
            if areas.reference_count[mass_index] < 33:
                return None               # too few reference detections at this mass
            ratio[amplitude][direction] = areas.detected_area_ratio[mass_index]
            spurious[amplitude][direction] = areas.spurious_ratio[mass_index]
    directions = set(ratio[next(iter(ratio))])
    found = knowledge_error_tolerance(ratio, spurious, eligible=directions, criterion=criterion)
    return found.amplitude                # None if no amplitude passes

# One value per lens and mass, then the median across lenses:
# requirement_nm = np.median([value for value in values if value is not None])
```

The tolerance is the largest passing amplitude, even if a smaller one failed.
`found.passing` lists every passing amplitude and `found.first_failing` the smallest
failing one; check both. Leave out of the maps any amplitude that is not part of the test.
The paper, for example, also ran a 35 nm amplitude as a fixed end point and excluded it
from the tolerance.

`plot_knowledge_error(areas)` draws the three ratios against mass.

## Wavefront nuisances

For an optical model PSF, wavefront modes can be profiled as nuisance parameters, in the
same way as lens and source parameters. This asks how much subhalo signal could be
absorbed by fitting the PSF at the same time:

```yaml
forecast:
  nuisances:
    wavefront:
      modes:
        zernikes: {nolls: [4, 5, 6]}
        segment_hexikes: {segments: all, nolls: [2, 3]}
      step_nm: 1.0
      prior_sigma_nm: 5.0
```

Global Zernike Noll 1 (piston) is not allowed. `step_nm` and `prior_sigma_nm` can be
single numbers or set per family.

## Chromatic PSFs

A broadband optical PSF is the photon-weighted sum of monochromatic PSFs across the
bandpass. Set `wavelength_samples` instead of `wavelength_nm`, and give each light
component a spectrum:

```yaml
psf:
  truth: {wavelength_nm: null, wavelength_samples: 11, kernel_shape: [901, 901]}
scene:
  source:
    light:
      disk: {type: Exponential, ..., flux: {ab_mag: 24.9}, sed: {kind: flat_fnu}}
      clump: {type: Exponential, ..., flux: {ab_mag: 26.5}, sed: {kind: power_law, index: 2.0}}
```

Each light component with its own spectrum is a **light group** and gets its own
effective PSF, weighted by its photon spectrum across the band. Spectra can be
`flat_fnu`, `flat_flambda`, a `power_law` with $f_\nu \propto \nu^{\mathrm{index}}$, or a
`table` read from a file.

Each wavelength's kernel is normalized on its finite support, so power that falls
outside the kernel is redistributed inside it, by a different amount at each
wavelength. Check that results do not change when you increase `wavelength_samples`
and `kernel_shape`. The [chromatic example](../examples/chromatic.md) runs exactly
this comparison.

`psf.model: {kind: monochromatic}` fits a single-wavelength model to chromatic data,
at a given `wavelength_nm` or, with `null`, at each light group's photon-weighted mean
wavelength.

Nonlinear fits need a single model kernel. For a chromatic truth, use a model kernel
file or a monochromatic model with an explicit `wavelength_nm`.

## Inspecting a PSF

The prepared forecast holds the bound kernels. `plot_kernel` draws one:

```python
import matplotlib.pyplot as plt
from hwoslaps.plotting import plot_kernel

with prepare_forecast(config) as prepared:
    kernel = prepared.psfs.truth_kernels.kernels[0]
    ax = plot_kernel(kernel, log=True)
    ax.figure.savefig("psf.png")
```
