# Chromatic PSF

Forecasts a lens whose source has two components of different colour, each with its own
broadband PSF, using the HWO reference geometry and SEI coating and detector curves with an
assumed 0.832 filter transmission. The values are illustrative.

```bash
CUDA_VISIBLE_DEVICES=0 JAX_ENABLE_X64=1 python examples/chromatic/run.py --q-threshold 10 --output out/chromatic
```

`convergence.py` compares forecasts with 11 and 22 wavelengths and with 901 and 601 pixel
kernels; the commands are in `docs/examples/chromatic.md` in the handbook.
`CAPTURED_FRACTIONS.md` lists the captured fraction of each wavelength's kernel from the
reference runs.
