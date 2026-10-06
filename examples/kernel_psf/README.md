# Kernel PSF example

These illustrative detector kernels describe no instrument. They are circular Moffat
profiles with beta 3, integrated over 0.1 arcsec pixels using 11 x 11 sub-samples per pixel.
Truth has FWHM 2.0 pixels; the model has FWHM 2.04 pixels, a 2-percent width error.
Both 15 x 15 kernels are normalized to unit sum. Scene and detector come from the minimal
config.

The shipped arrays were generated with the supplied generator and their actual file hashes
are pinned in the overlays. The generator publishes complete files, refuses overwrites,
and prints actual file SHA256 values. To reproduce them, choose a new directory:

```bash
python examples/kernel_psf/make_kernels.py --output-dir out/generated_moffat
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

The actual XTX-generated truth and model file SHA256 values are
`8da22fd72c5ee449317090f483b9b228cba0d01c0e13e2d10c5344c68c6bf62e` and
`923100e01d1a9f918727f3f459cdc144a5cbe2451843eba4db258c84b787b64f`.
Budget: 120 s for the paired CPU forecasts and reduction. On XTX, Python 3.11, BLAS threads 1,
the paired run took 8.8843 s on 2026-10-06. Both arms recorded sampling
`source: 0.32274029790875214`; this is the measured discretization diagnostic, with no
accuracy claim inferred from it. The run and area records retain the actual mappings.
