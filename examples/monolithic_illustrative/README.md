# Monolithic instrument

Illustrative instrument. The numbers are round values chosen to exercise the engine; do not
quote results as a forecast for any mission.

The 2 m circular aperture has obscuration ratio 0.25 and three 0.02 m spiders. Its PSF is
monochromatic at 700 nm with four sub-samples per detector-pixel side. The photometric
band is 600-800 nm with top-hat throughput 0.4. Gain is 2 e-/ADU, read noise 3 e- per
exposure, and dark current 0.005 e-/s/pixel. Three exposures sum to 2400 s.

The lens at redshift 0.4 has Einstein radius 1.2 arcsec, ellipticity, external shear and
Sersic lens light of index 4. The source at redshift 1.5 has 23.5 AB mag and effective radius
0.15 arcsec. Centres and ellipticities are illustrative choices in `scene.yaml`. The
120 x 120 grid has 0.04 arcsec pixels and oversampling 4. A `source_snr` mask uses
source-plane signal and threshold 3; lens light contributes detector variance.

```bash
hwoslaps validate examples/monolithic_illustrative/scene.yaml examples/monolithic_illustrative/instrument.yaml examples/monolithic_illustrative/forecast.yaml
python examples/monolithic_illustrative/run.py --q-threshold 10 --output out/monolithic
```

The CPU reference driver uses eight workers, forecasts a 1e8 solar-mass NFW halo and writes expected
observation, forecast and run record. Add `--plot` to save maps. Resolved rates,
collecting area and actual sampling of each light group appear in the run record.
Set `--reference-workers` to a positive count to select another CPU allocation.

Budget: 120 s CPU. On XTX on 2026-10-06, Python 3.11, BLAS threads 1 and eight CPU
workers, the actual driver completed in 24.7548 s. Its complete scientific forecast
and expected-observation members matched the serial baseline bitwise. The original
serial command reached its 120 s limit; a separate completion diagnostic took
120.6165 s and retained its budget failure. Actual sampling was
`lens = 0.3100470893969617` and `source = 0.09437939132284434`.
Sampling describes the configured discretization; the example supplies no mission
performance prediction or detection-accuracy claim based on that diagnostic alone.
