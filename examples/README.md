# Examples

Run the commands from the repository root in the scientific environment. Each run needs a
new output directory, and every driver takes a required `--q-threshold`.

| Example | What it shows | Runs on | Handbook page |
|---|---|---|---|
| [HWO reference](hwo_reference/README.md) | The RASTI paper's HWO set-up: segmented telescope, AB photometry and saved products | CPU (reduced) or one GPU | `docs/examples/hwo.md` |
| [Monolithic telescope](monolithic_illustrative/README.md) | A circular telescope with obscuration and spiders, lens light, shear and a signal-to-noise mask | CPU | `docs/examples/monolithic.md` |
| [Chromatic PSF](chromatic/README.md) | Broadband PSFs for sources of different colours, with a convergence check | One GPU | `docs/examples/chromatic.md` |
| [Kernel PSFs](kernel_psf/README.md) | PSF kernel files and PSF knowledge-error areas | CPU | `docs/examples/kernel_psf.md` |
| [Population batch](population/README.md) | A population of lenses run as a resumable batch, with a nonlinear fit | CPU or GPUs | `docs/examples/population.md` |

Only the HWO reference reproduces a published set-up. The other examples use illustrative
values chosen to exercise features of HWO-SLAPS.
