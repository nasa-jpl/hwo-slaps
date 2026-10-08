# Monolithic telescope

This example models a 2 m telescope with a circular primary mirror, a central
obscuration and three spiders, observing a lens with bright lens light and external
shear. Its values are round numbers chosen to exercise these features. They describe
no real observatory, so do not quote its results as a forecast.

## The configuration

```{literalinclude} ../../examples/monolithic_illustrative/instrument.yaml
:language: yaml
:caption: examples/monolithic_illustrative/instrument.yaml
```

The telescope has a 25% central obscuration and three 2 cm spiders. Its PSF is computed
at 700 nm. Three exposures add up to 2400 s, so the read noise enters three times.

```{literalinclude} ../../examples/monolithic_illustrative/scene.yaml
:language: yaml
:caption: examples/monolithic_illustrative/scene.yaml
```

The lens at redshift 0.4 has an Einstein radius of 1.2 arcsec, external shear, and a
de Vaucouleurs (Sérsic index 4) light profile of 19.5 AB mag. The source at redshift 1.5
has 23.5 AB mag. Lens light adds photon noise everywhere in the image, and its
parameters are profiled along with the lens mass and source.

```{literalinclude} ../../examples/monolithic_illustrative/forecast.yaml
:language: yaml
:caption: examples/monolithic_illustrative/forecast.yaml
```

The `source_snr` mask keeps only pixels where the lensed source is detected at a
signal-to-noise of at least 3. The noise in that ratio includes the lens light.

## Running it

```bash
python examples/monolithic_illustrative/run.py --q-threshold 10 --output out/monolithic
```

The driver forecasts a 10⁸ M☉ NFW subhalo at every grid position on eight CPU workers,
which takes under half a minute. It writes `forecast.npz`, `expected.npz` and
`run.json`; `--plot` also saves maps, and `--reference-workers` changes the number of
workers.

To check the configuration without running it:

```bash
hwoslaps validate examples/monolithic_illustrative/scene.yaml \
    examples/monolithic_illustrative/instrument.yaml \
    examples/monolithic_illustrative/forecast.yaml
```

## Pixel sampling

The 0.04 arcsec pixels are coarse for this lens. `run.json` records a sampling
diagnostic of 0.31 for the lens light and 0.094 for the source, both above the 0.06
level where the binned rendering was calibrated. Before using a configuration like this
for science, compare it against one with finer pixels. See
[Observations](../guide/observations.md#pixel-sampling).
