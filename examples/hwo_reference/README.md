# HWO reference

This reference uses SEI v0.1.9 EAC1 telescope and HRI UVIS detector inputs with the
450-550 nm observing band used for RASTI-26-183 forecasts. SEI v0.1.9 contains
pre-formulation values and exploratory architectures. The top-hat throughput 0.21 is a
study choice; UVIS filter curves are absent from the SEI wheel.

The isothermal lens has redshift 0.2; the exponential source has redshift 0.6. Source flux
is intrinsic, before lensing, and given by its in-band AB magnitude. The engine derives
the surface-brightness amplitude from a continuous integral. The paper-value overlay
supplies the original discrete-normalization amplitude.

| Quantity | Input or target | Source |
|---|---|---|
| Pupil diameter | 7.225765 m, circumscribed | `EAC1.yaml:PM.circumscribing_diameter` |
| Segments | 19 in two rings, 1.65 m point-to-point, 0.006 m optical gaps | `EAC1.yaml:PM.segmentation_parameters` |
| Focal length | 144 m | `EAC1.yaml:OTA_full.focal_length` |
| Detector pixel scale | 0.00716 arcsec | `HRI.yaml:UVIS.plate_scale`, 7.16 mas |
| PSF wavelength | 500 nm | `HRI.yaml:UVIS.DL_wavelength`, 0.5 micrometre |
| Read noise | 0.28284271247461906 e- per exposure | HRI detector: 0.2 e- per read, combined for two reads |
| Dark current | 0.002 e-/s/pixel | `HRI.yaml:UVIS.detector.detector_DC` |
| Gain | 1 e-/ADU | Chosen |
| Exposure | 2000 s, one exposure | Chosen |
| Band | 450-550 nm, top-hat throughput 0.21 | Study band; LUVOIR HDI system-QE reference |
| Collecting area target | 33.606448937520405 m² | Pinned pupil integral, 512 pupil pixels, supersampling 4 |
| Sky | 23 AB mag/arcsec²; target 0.002510279845963486 e-/s/pixel | Cited ETC convention and observing preset |
| Source | 24.845 AB mag; target 8.951505744562876 e-/s | Adopted 24.3 F814W mag plus 0.545 mag colour |
| Source amplitude target | 0.003174147284617635 e-/s per pixel sample at the effective radius | Pinned discrete paper normalization |
| Blank variance target | 9.100559691926973 e-² | Sky and dark counts plus read variance in engine order |

The reference test `tests/observation/test_observation_hwo_reference.py` checks collecting
area at relative 1e-12, AB source and sky rates at 1e-9, continuous amplitude at 2e-7, and
blank variance at 1e-15 through the real expected-observation boundary. The source-rate target applies to
the AB input. The literal-amplitude overlay retains the small continuous/discrete difference.
These photometric checks are separate from forecast convergence.

The CPU smoke command preserves the scene and pupil, uses a 101 x 101 kernel, and changes
position spacing to 0.3 arcsec:

```bash
python examples/hwo_reference/run.py --quick --seed 11 --q-threshold 10 --output out/hwo_quick
```

The full command uses one visible GPU, 999 x 999 kernel support, 3721 positions and masses
1e7, 1e8 and 1e9 solar masses:

```bash
CUDA_VISIBLE_DEVICES=0 JAX_ENABLE_X64=1 python examples/hwo_reference/run.py --seed 11 --q-threshold 10 --output out/hwo_reference
```

The driver verifies vendored hashes, prints photometry and sampling, and writes
`forecast.npz`, `expected.npz`, `noisy.npz` and `run.json`. The noisy product is a
smooth control with the supplied seed. Optional `--plot` saves maps. Existing outputs
are refused.

The fragments also compose through the CLI:

```bash
hwoslaps validate examples/hwo_reference/instrument.yaml examples/hwo_reference/scene_smooth_ring.yaml examples/hwo_reference/forecast.yaml
hwoslaps forecast examples/hwo_reference/instrument.yaml examples/hwo_reference/scene_smooth_ring.yaml examples/hwo_reference/forecast.yaml examples/hwo_reference/paper_values.yaml examples/hwo_reference/psf_state_hwo35.yaml --engine jax --masses 1e7 1e8 1e9 -o out/hwo_paper_values
hwoslaps forecast examples/hwo_reference/instrument.yaml examples/hwo_reference/scene_smooth_ring.yaml examples/hwo_reference/forecast.yaml examples/hwo_reference/knowledge_error.yaml --engine jax --masses 1e7 1e8 1e9 -o out/hwo_knowledge_error
```

`paper_values.yaml` supplies literal paper source amplitude and sky rate.
`psf_state_hwo35.yaml` draws 35 nm aperture RMS from the packaged static prior,
combined family, seed 20260835. `fit_kernel_51.yaml` supplies the paper nonlinear
fit support. A batch pairs this fit arm with its larger forecast arm through `forecast_arm`.

Budget: CPU smoke 120 s; full GPU 600 s. Measured runtime: pending. Sampling per group:
pending; the run record stores actual observation values. A budget overrun is reported
without changing physical inputs.

References: [Liu et al., EAC concepts](https://arxiv.org/abs/2602.11046),
[Stark et al., ETC comparison](https://arxiv.org/abs/2502.18556),
[LUVOIR Final Report](https://arxiv.org/abs/1912.06219),
[Stark et al., bandpasses](https://arxiv.org/abs/2404.05654),
[Newton et al., SLACS sources](https://arxiv.org/abs/1104.2608), and the
[SEI repository](https://github.com/HWO-GOMAP-Working-Groups/Sci-Eng-Interface).
Targets preserve the pinned RASTI-26-183 observing preset.
