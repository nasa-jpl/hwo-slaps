# Kernel PSFs

Compares a forecast made with the correct PSF kernel against one made with a kernel 2%
too wide. Both are pixel-integrated Moffat profiles (beta = 3) on 0.1 arcsec pixels.

```bash
python examples/kernel_psf/run.py --q-threshold 10 --min-reference-count 1 --output out/kernel_psf
python examples/kernel_psf/make_kernels.py --output-dir out/generated_moffat   # regenerate the kernels
```

The driver forecasts both PSFs in about ten seconds on a CPU and compares them with
`knowledge_error_areas`. See `docs/examples/kernel_psf.md` in the handbook.
