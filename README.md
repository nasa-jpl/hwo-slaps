# HWO-SLAPS

HWO-SLAPS is an end-to-end strong-lensing simulation and Fisher-forecast pipeline for connecting Habitable Worlds Observatory PSF stability to low-mass dark-matter subhalo detectability.

The active study is the RASTI HWO Special Issue paper on subhalo mass reach and PSF knowledge requirements. The submitted SPIE study is archived separately. Start here:

- [Current study, results status, and code map](docs/study/study_roadmap.md)
- [Publication scope and manuscript plan](docs/study/venue_plan.md)

## Pipeline

The package has four active modules:

1. `lensing`: galaxy-galaxy strong-lensing scenes with optional subhalos.
2. `psf`: segmented-aperture HWO-style PSFs with controlled aberrations.
3. `observation`: PSF convolution and detector-noise simulation.
4. `modeling`: profiled Fisher / Asimov forecasts and PyAutoLens/Nautilus nonlinear validation.

`campaign` provides frozen designs, manifests, and adaptive mass ladders;
`analysis` provides selection and ranking statistics. Campaign generation,
execution, and harvest entry points live in `scripts/`. The study roadmap
maps these components so a task can load only the relevant code.

## Quick Start

Create the conda environment and install the developer dependency checkouts:

```bash
bash install.sh
```

This clones or updates PyAutoLens and HCIPy as editable GitHub installs in the
parent directory of this repo by default. Override the checkout location with
`--checkout-root` or `HWOSLAPS_DEV_ROOT`.

For an NVIDIA GPU machine such as a B200 node, install the same environment with
CUDA-enabled JAX:

```bash
bash install.sh --gpu
```

Run a config:

```bash
python runner.py --config configs/master_config.yaml
```

Run the core tests:

```bash
python -m pytest -q tests/
```

## Repository Notes

- `configs/master_config.yaml` is the current runnable example config.
- `outputs/` is ignored and used for run artifacts.
- `scratch/` is predominantly ignored and holds local study records. Its
  `README.md` distinguishes current entry points from historical material.
- Planning docs that should persist live under `docs/`.

## Copyright

Copyright 2025, by the California Institute of Technology. ALL RIGHTS RESERVED. United States Government Sponsorship acknowledged. Any commercial use must be negotiated with the Office of Technology Transfer at the California Institute of Technology.

This software may be subject to U.S. export control laws. By accepting this software, the user agrees to comply with all applicable U.S. export laws and regulations. User has the responsibility to obtain export licenses, or other export authority as may be required before exporting such information to foreign countries or providing access to foreign persons.

## Authors

Georgios Vassilakis (JPL)
