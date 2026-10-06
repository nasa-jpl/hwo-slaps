# Kernel PSF example

These illustrative detector kernels describe no instrument. They are circular Moffat
profiles with beta 3, integrated over 0.1 arcsec pixels using 11 x 11 sub-samples per pixel.
Truth has FWHM 2.0 pixels; the model has FWHM 2.04 pixels, a 2-percent width error.
Both 15 x 15 kernels are normalized to unit sum. Scene and detector come from the minimal
config.

Generate the arrays first. The generator publishes complete files, refuses overwrites,
and prints actual file SHA256 values:

```bash
python examples/kernel_psf/make_kernels.py
python examples/kernel_psf/run.py --q-threshold 10 --min-reference-count 1 --output out/kernel_psf
```

The CPU driver prepares matched and mismatched models, forecasts 1e7, 1e8 and 1e9
solar masses, and calls `knowledge_error_areas` on the real products. Threshold and
reference-count floor are reader choices. It records retained area, detected/reference
area ratio, and spurious area. A mass below the reference-count floor has null ratios
and `eligible: false`.

Apply the reduction to saved products:

```python
from hwoslaps.artifacts import load_forecast
from hwoslaps.analysis.knowledge_error import knowledge_error_areas

reference = load_forecast("out/kernel_psf/matched/forecast.npz")
mismatched = load_forecast("out/kernel_psf/mismatched/forecast.npz")
areas = knowledge_error_areas(reference, mismatched, q_threshold=10, min_reference_count=1)
```

Generated arrays and pinned hashes await supported-environment generation. Budget:
120 s for the paired CPU forecasts and reduction. Measured runtime and sampling:
pending; the run and area records retain actual per-group mappings.
