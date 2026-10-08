# Chromatic PSF

The PSF of a diffraction-limited telescope grows with wavelength, so sources of
different colours see different broadband PSFs. This example forecasts a lens whose
source has two components of different colour, using a broadband PSF for each, and
then checks that the result has converged in the number of wavelengths and the kernel
size.

The lens geometry and telescope are those of the [HWO reference](hwo.md). The
bandpass uses SEI coating and detector curves with an assumed flat filter transmission
of 0.832. The values are illustrative.

## The configuration

```{literalinclude} ../../examples/chromatic/scene_two_colour.yaml
:language: yaml
:caption: examples/chromatic/scene_two_colour.yaml
```

The lens has Sérsic light with a red power-law spectrum. The source has a disk with a
flat-*f*ν spectrum and a small, bluer clump. Each component with its own spectrum is a
separate light group with its own broadband PSF.

```{literalinclude} ../../examples/chromatic/instrument_sei_chain.yaml
:language: yaml
:caption: examples/chromatic/instrument_sei_chain.yaml
```

`wavelength_samples: 11` computes the PSF at 11 wavelengths across the 450 to 550 nm
band. The bandpass is the product of thirteen mirror reflections, the detector quantum
efficiency and the filter. The kernel is 901 × 901 pixels (6.45 arcsec). With a 512-pixel
pupil, the computed PSF repeats every 6.58 arcsec at the short end of the band, so a
999-pixel kernel (7.15 arcsec) would include a repeated copy.

## Running it

The default run forecasts on the full position grid with one GPU, in a little over two
minutes:

```bash
CUDA_VISIBLE_DEVICES=0 JAX_ENABLE_X64=1 \
python examples/chromatic/run.py --q-threshold 10 --output out/chromatic
```

The driver prints and saves each wavelength's captured fraction, the pixel sampling of
each light group, the photometry and the spectral weights.

## Checking convergence

Normalizing each wavelength's kernel on its finite support moves a little power from
the wings into the kernel, by a different amount at each wavelength. The convergence
check runs six forecasts on a ring of 36 positions at the Einstein radius:

| Variant | Wavelengths | Kernel |
|---|---|---|
| `11_901` | 11 | 901 × 901 |
| `22_901` | 22 | 901 × 901 |
| `11_601` | 11 | 601 × 601 |

each with the matched PSF and with a monochromatic model PSF at each light group's mean
wavelength:

```bash
for variant in 11_901 22_901 11_601; do
  for arm in matched monochromatic; do
    CUDA_VISIBLE_DEVICES=0 JAX_ENABLE_X64=1 python examples/chromatic/run.py --ring \
        --variant $variant --arm $arm --q-threshold 10 \
        --output out/chromatic_convergence/${variant}_${arm}
  done
done
python examples/chromatic/convergence.py --directory out/chromatic_convergence \
    --q-threshold 10 --output out/chromatic_convergence/gates.json
```

`convergence.py` passes when, at each mass, the largest *q* changes by at most 1%
between variants, for both the matched and the monochromatic model. In the reference
runs the largest change was 0.13%. Each product takes two to three minutes on one GPU.

## Fitting chromatic data

A nonlinear fit needs a single model PSF kernel. The monochromatic model at each group's
mean wavelength still has one kernel per group, so for fits use a model kernel file or
set one explicit `wavelength_nm` for the monochromatic model.
