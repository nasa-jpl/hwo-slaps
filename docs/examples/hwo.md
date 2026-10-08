# HWO reference

This example forecasts subhalo detection for a Habitable Worlds Observatory imaging
observation. It uses the telescope of exploratory analytic case EAC1 and the ultraviolet and
visible (UVIS) channel of the High Resolution Imager (HRI), taken from the HWO
Science-Engineering Interface (SEI) v0.1.9. The observing band, lens and source are those of
the RASTI paper (Vassilakis et al., *Point spread function requirements for dark matter
subhalo detection with the Habitable Worlds Observatory*, submitted to RAS Techniques and
Instruments). SEI v0.1.9 describes pre-formulation concepts, so these are study inputs, not
final instrument values.

```{figure} ../_static/hwo-reference.png
:alt: Left, the simulated HWO ring in ADU. Right, the forecast q map for a 10^8 solar-mass NFW subhalo.
:width: 100%

The expected smooth observation (left) and `q_asimov` for a 10⁸ M☉ NFW subhalo
(right), from the full GPU run.
```

## The configuration

The configuration is split into three files, which the driver combines in order.

### Telescope and detector

```{literalinclude} ../../examples/hwo_reference/instrument.yaml
:language: yaml
:caption: examples/hwo_reference/instrument.yaml
```

| Quantity | Value | Source |
|---|---|---|
| Primary mirror | 19 hexagonal segments in two rings, 1.65 m point to point, 6 mm gaps; 7.23 m circumscribed | SEI `EAC1.yaml` |
| Focal length | 144 m | SEI `EAC1.yaml` |
| Detector pixel scale | 0.00716 arcsec (7.16 mas) | SEI `HRI.yaml` |
| PSF wavelength | 500 nm | SEI `HRI.yaml`, UVIS diffraction-limited wavelength |
| Read noise | 0.283 e⁻ per exposure (two reads of 0.2 e⁻) | SEI HRI detector |
| Dark current | 0.002 e⁻ s⁻¹ pixel⁻¹ | SEI `HRI.yaml` |
| Band | 450 to 550 nm, top-hat throughput 0.21 | Study choice; SEI has no UVIS filter curves |
| Exposure | One 2000 s exposure | Study choice |
| Sky | 23 AB mag arcsec⁻² | Study choice |

`collecting_area_m2: null` makes hwoslaps compute the collecting area from the sampled
pupil: 33.61 m².

### Lens and source

```{literalinclude} ../../examples/hwo_reference/scene_smooth_ring.yaml
:language: yaml
:caption: examples/hwo_reference/scene_smooth_ring.yaml
```

An isothermal lens at redshift 0.2 with Einstein radius 1 arcsec, and an exponential
source at redshift 0.6. The source's 24.845 AB magnitude is its intrinsic, unlensed
brightness in the observing band. With the flat-$f_\nu$ spectrum, bandpass and collecting
area, it gives a detected rate of 8.95 e⁻ s⁻¹. The sky gives 0.00251 e⁻ s⁻¹ per pixel.

### Forecast

```{literalinclude} ../../examples/hwo_reference/forecast.yaml
:language: yaml
:caption: examples/hwo_reference/forecast.yaml
```

A 61 × 61 grid of trial positions, 0.05 arcsec apart, using every pixel and profiling
every lens and source parameter.

## Running it

The full run uses a 999 × 999 pixel PSF kernel and all 3721 positions, at 10⁷, 10⁸ and
10⁹ M☉. It needs one GPU and takes about a minute:

```bash
CUDA_VISIBLE_DEVICES=0 JAX_ENABLE_X64=1 \
python examples/hwo_reference/run.py --q-threshold 10 --output out/hwo_reference
```

A reduced run for a CPU keeps the same scene, detector and telescope but uses a
101 × 101 kernel and 0.3 arcsec position spacing. With eight CPU workers it takes
a little over a minute:

```bash
python examples/hwo_reference/run.py --quick --q-threshold 10 --output out/hwo_quick
```

| Option | Effect |
|---|---|
| `--q-threshold` | Detection threshold for the printed summary (required) |
| `--quick` | The reduced CPU run |
| `--reference-workers` | CPU workers for `--quick` (default 8) |
| `--seed` | Noise seed of the noisy smooth observation (default 11) |
| `--overlay FILE` | Combine another configuration file after the three above; repeatable |
| `--plot` | Also save `observation.png` and `forecast.png` |

Before running, the driver checks the SEI files in `examples/hwo_reference/sei_v0.1.9/`
against their recorded hashes.

## Outputs

| File | Contents |
|---|---|
| `forecast.npz` | The forecast at every mass and position |
| `expected.npz` | The expected smooth observation, with its noise map and PSF |
| `noisy.npz` | A noisy smooth observation, for use as a control |
| `run.json` | The command, inputs, photometry, pixel sampling, threshold and timing |

Read them back in Python:

```python
from hwoslaps import load_forecast, load_observation, summarize

result = load_forecast("out/hwo_reference/forecast.npz")
expected = load_observation("out/hwo_reference/expected.npz")

summary = summarize(result, q_threshold=10.0)
print(summary.q_max, summary.detectable_area_arcsec2)
print(result.provenance["photometry"]["components"])
```

## Variations

Overlay files in `examples/hwo_reference/` change one aspect of the set-up:

| File | Change |
|---|---|
| `paper_values.yaml` | Use the literal source intensity and sky rate from the paper instead of the AB magnitudes |
| `psf_state_hwo35.yaml` | Add a 35 nm RMS static wavefront error to the telescope |
| `knowledge_error.yaml` | Make the model PSF wrong by a 10 nm RMS drift-shaped wavefront error |
| `fit_kernel_51.yaml` | Use a 51 × 51 kernel, the size used for the paper's nonlinear fits |

For example, to see how a 10 nm PSF knowledge error affects detections:

```bash
CUDA_VISIBLE_DEVICES=0 JAX_ENABLE_X64=1 \
python examples/hwo_reference/run.py --q-threshold 10 \
    --overlay examples/hwo_reference/knowledge_error.yaml --output out/hwo_knowledge_error
```

This forecast reports `q_mismatch` and `q_spurious`. Compare it with the matched run
using `knowledge_error_areas`, as described in
[PSFs and PSF errors](../guide/psfs.md#psf-knowledge-error).

The same files work with the command line:

```bash
hwoslaps forecast examples/hwo_reference/scene_smooth_ring.yaml \
    examples/hwo_reference/instrument.yaml examples/hwo_reference/forecast.yaml \
    examples/hwo_reference/paper_values.yaml \
    --engine jax --masses 1e7 1e8 1e9 -o out/hwo_paper_values
```

## The same forecast in Python

```python
from hwoslaps import Execution, forecast, load_config, prepare_forecast


def main():
    config = load_config([
        "examples/hwo_reference/scene_smooth_ring.yaml",
        "examples/hwo_reference/instrument.yaml",
        "examples/hwo_reference/forecast.yaml",
    ])
    # The reduced CPU set-up of --quick:
    config = config.replace({
        "psf": {"truth": {"kernel_shape": [101, 101]}},
        "forecast": {"positions": {"spacing_arcsec": 0.3}},
    })
    with prepare_forecast(config, execution=Execution(reference_workers=8)) as prepared:
        result = forecast(prepared, masses_msun=[1e7, 1e8, 1e9])
        print(prepared.observation.sampling)
    print(result.detections(q_threshold=10.0).sum(axis=1))


if __name__ == "__main__":
    main()
```

The `if __name__ == "__main__":` guard is needed because the eight reference workers
start as separate processes.

## Pixel sampling

With 7.16 mas pixels, the lensed source varies by about 2.4% across a pixel
(`sampling` of 0.024), well below the calibrated level of 0.063.
See [Observations](../guide/observations.md#pixel-sampling).

## References

- Liu et al., [HWO exploratory analytic cases](https://arxiv.org/abs/2602.11046)
- Stark et al., [exposure time calculator comparison](https://arxiv.org/abs/2502.18556)
- The LUVOIR team, [LUVOIR final report](https://arxiv.org/abs/1912.06219)
- Stark et al., [bandpasses](https://arxiv.org/abs/2404.05654)
- Newton et al., [SLACS source properties](https://arxiv.org/abs/1104.2608)
- The [HWO Science-Engineering Interface repository](https://github.com/HWO-GOMAP-Working-Groups/Sci-Eng-Interface)
