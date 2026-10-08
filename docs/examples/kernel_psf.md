# Kernel PSFs

This example compares a forecast made with the correct PSF against one made with a PSF
that is 2% too wide. It shows how to supply PSFs as kernel files and how to measure the
effect of a PSF error on detections.

## The kernels

Both PSFs are circular Moffat profiles with β = 3, integrated over 0.1 arcsec pixels and
stored as 15 × 15 kernels. The truth has a full width at half maximum of 2.0 pixels;
the model has 2.04 pixels. They replace the PSF of the minimal configuration:

```{literalinclude} ../../examples/kernel_psf/truth.yaml
:language: yaml
:caption: examples/kernel_psf/truth.yaml
```

```{literalinclude} ../../examples/kernel_psf/knowledge_error.yaml
:language: yaml
:caption: examples/kernel_psf/knowledge_error.yaml
```

The `file_sha256` entries make hwoslaps check each file before using it. To regenerate
the kernels:

```bash
python examples/kernel_psf/make_kernels.py --output-dir out/generated_moffat
```

## Running it

```bash
python examples/kernel_psf/run.py --q-threshold 10 --min-reference-count 1 --output out/kernel_psf
```

The driver forecasts 10⁷, 10⁸ and 10⁹ M☉ twice, once with each model PSF, and compares
the two with `knowledge_error_areas`. `--min-reference-count` is the number of matched
detections below which the area ratios are left out (`NaN`); 1 keeps every mass in this
small example. Both forecasts together take about ten seconds on
a CPU. It writes `matched/` and `mismatched/` product directories and a
`knowledge_error.json` summary.

## Reading the comparison

```python
from hwoslaps import load_forecast
from hwoslaps.analysis import knowledge_error_areas

reference = load_forecast("out/kernel_psf/matched/forecast.npz")
mismatched = load_forecast("out/kernel_psf/mismatched/forecast.npz")
areas = knowledge_error_areas(reference, mismatched, q_threshold=10.0, min_reference_count=1)

print(areas.detected_area_ratio)   # R: area detected with the wrong PSF / correct area
print(areas.spurious_ratio)        # F: false-detection area / correct area
```

[PSFs and PSF errors](../guide/psfs.md#psf-knowledge-error) explains each ratio and how
to turn many such comparisons into a PSF tolerance.
