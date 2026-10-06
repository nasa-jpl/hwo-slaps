# Chromatic example

This illustrative example combines HWO reference geometry with SEI coating and detector
curves and an assumed flat filter transmission of 0.832. The UVIS filter curve is absent
from the SEI wheel. The band is 450-550 nm. Lens light has power-law SED index -2;
source light contains a flat-fnu disk and a clump of index 2, with distinct effective PSFs.
Lens light contributes image-plane light and detector noise.

The initial model uses 11 wavelength bins and 901 x 901 support. At the short band edge,
512 pupil pixels span an aliasing period of about 6.58 arcsec. A 999-pixel support would
span 7.15 arcsec; 901 pixels span 6.45 arcsec. Every propagated wavelength undergoes
engine sampling checks. Normalized finite-support kernels remain an approximation,
so wavelength and support checks must pass before claiming chromatic accuracy.

The pinned flat-fnu derivation targets band mean 0.206800227875, source rate
8.815111560954138 e-/s at 24.845 AB mag, and sky 0.0024720306865494263 e-/s/pixel at
23 AB mag/arcsec². The two-colour scene has different component magnitudes in its config;
these target rates do not describe its summed source light.

```bash
CUDA_VISIBLE_DEVICES=0 JAX_ENABLE_X64=1 python examples/chromatic/run.py --q-threshold 10 --output out/chromatic
```

The driver prints and saves every node's captured fraction, actual light-group sampling,
photometry and spectral weights. Forecast and expected observation products accompany
`run.json`. Optional `--plot` saves maps for the grid run.

Run six products on a 36-position ring at the Einstein radius. Each call has a 600 s
one-GPU budget; run sequentially with one GPU visible:

```bash
CUDA_VISIBLE_DEVICES=0 JAX_ENABLE_X64=1 python examples/chromatic/run.py --ring --variant 11_901 --arm matched --q-threshold 10 --output out/chromatic_convergence/11_901_matched
CUDA_VISIBLE_DEVICES=0 JAX_ENABLE_X64=1 python examples/chromatic/run.py --ring --variant 11_901 --arm monochromatic --q-threshold 10 --output out/chromatic_convergence/11_901_monochromatic
CUDA_VISIBLE_DEVICES=0 JAX_ENABLE_X64=1 python examples/chromatic/run.py --ring --variant 22_901 --arm matched --q-threshold 10 --output out/chromatic_convergence/22_901_matched
CUDA_VISIBLE_DEVICES=0 JAX_ENABLE_X64=1 python examples/chromatic/run.py --ring --variant 22_901 --arm monochromatic --q-threshold 10 --output out/chromatic_convergence/22_901_monochromatic
CUDA_VISIBLE_DEVICES=0 JAX_ENABLE_X64=1 python examples/chromatic/run.py --ring --variant 11_601 --arm matched --q-threshold 10 --output out/chromatic_convergence/11_601_matched
CUDA_VISIBLE_DEVICES=0 JAX_ENABLE_X64=1 python examples/chromatic/run.py --ring --variant 11_601 --arm monochromatic --q-threshold 10 --output out/chromatic_convergence/11_601_monochromatic
python examples/chromatic/convergence.py --directory out/chromatic_convergence --q-threshold 10 --output out/chromatic_convergence/gates.json
```

The reader requires identical scene, band, mass and position bytes, allowing only the stated
PSF changes. It uses signed-amplitude maxima from the public summary. Per mass, relative
changes of matched and monochromatic-model maximum q must be at most 1e-2. Absolute
change of the monochromatic maximum spurious q must be at most 1e-2 times the matched
reference maximum q. A zero model-arm reference maximum leaves its relative check
unresolved and fails the gate.

The reader binds each companion record to the forecast file hash, actual engine/device and
repeated scientific inputs. The report's `passed` field covers numerical convergence;
`all_products_within_budget` separately records the six runtime-budget results. A numerical
pass does not waive an exceeded runtime budget.

Budgets: 600 s per GPU product. On XTX on 2026-10-06, one NVIDIA B200 visible,
Python 3.11 and BLAS threads 1, all six ring products completed within budget:

| Nodes / support | Matched runtime (s) | Monochromatic runtime (s) |
|---|---:|---:|
| 11 / 901 | 134.9195 | 165.4700 |
| 22 / 901 | 139.1775 | 170.1850 |
| 11 / 601 | 97.7286 | 109.2301 |

Both masses passed the unchanged 1e-2 wavelength and support gates. The largest
relative maximum-q change was 0.00131434 (support comparison, monochromatic arm,
1e8 solar masses). The largest spurious-q change divided by matched maximum q was
1.51542e-8. These results cover the prescribed comparisons and these inputs.
Actual sampling was `lens:bulge = 0.14503789176869344`,
`source:clump = 0.11777160550304527` and `source:disk = 0.024480394447703186`.
Per-node captured fractions and wavelengths are listed in [CAPTURED_FRACTIONS.md](CAPTURED_FRACTIONS.md)
and retained in each run record. These sampling diagnostics do not establish general
discretization accuracy. The default grid command completed in 137.4071 s with
3721 positions and the same sampling diagnostics, within its 600 s budget.

A nonlinear fit requires one shared model kernel.
The group-mean monochromatic arms here can still have multiple distinct model kernels,
as can matched chromatic arms. For nonlinear fitting, use one common external kernel or
set an explicit common model wavelength; keep the convergence inputs above unchanged.
