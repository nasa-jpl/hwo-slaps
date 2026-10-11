# HWO reference

Forecasts for a Habitable Worlds Observatory imaging observation: the EAC1 telescope and
HRI UVIS detector from SEI v0.1.9 (vendored in `sei_v0.1.9/`), with the RASTI paper's
450 to 550 nm observing band, isothermal lens and exponential source.

```bash
# Full run: 999 x 999 PSF kernel, 3721 positions, one GPU, about a minute
CUDA_VISIBLE_DEVICES=0 JAX_ENABLE_X64=1 python examples/hwo_reference/run.py --q-threshold 10 --output out/hwo_reference

# Reduced run on eight CPU workers: 101 x 101 kernel, 0.3 arcsec spacing, about a minute
python examples/hwo_reference/run.py --quick --q-threshold 10 --output out/hwo_quick
```

The driver checks the SEI files against `sei_v0.1.9/SHA256SUMS`, then writes
`forecast.npz`, `expected.npz`, a noisy smooth control `noisy.npz` and `run.json`.

| File | Contents |
|---|---|
| `scene_smooth_ring.yaml`, `instrument.yaml`, `forecast.yaml` | The base configuration |
| `paper_values.yaml` | Overlay: the paper's literal source intensity and sky rate |
| `psf_state_hwo35.yaml` | Overlay: a 35 nm RMS static wavefront error |
| `knowledge_error.yaml` | Overlay: a 10 nm RMS model PSF error |
| `fit_kernel_51.yaml` | Overlay: the 51 x 51 kernel of the paper's nonlinear fits |

Add overlays with `--overlay FILE`. The walkthrough, with the source of every input value,
is `docs/examples/hwo.md` in the handbook.
