# Monolithic telescope

An illustrative 2 m telescope with a 25% central obscuration and three spiders, observing
a lens with Sérsic lens light and external shear. The values are round numbers chosen to
exercise these features; they describe no real observatory.

```bash
python examples/monolithic_illustrative/run.py --q-threshold 10 --output out/monolithic
```

The driver forecasts a 10^8 solar-mass NFW subhalo on eight CPU workers in under half a
minute and writes `forecast.npz`, `expected.npz` and `run.json`. Its 0.04 arcsec pixels
are coarse for this lens; see `docs/examples/monolithic.md` in the handbook.
