# HWO-SLAPS

HWO-SLAPS is an end-to-end strong-lensing simulation and Fisher-forecast pipeline for connecting Habitable Worlds Observatory PSF stability to low-mass dark-matter subhalo detectability.

The immediate study target is a controlled SPIE 2026 proceedings/poster analysis, followed by an expanded RASTI HWO Special Issue paper. See:

- [Venue plan](docs/study/venue_plan.md)
- [Study roadmap](docs/study/study_roadmap.md)

## Pipeline

The package has four active modules:

1. `lensing`: galaxy-galaxy strong-lensing scenes with optional subhalos.
2. `psf`: segmented-aperture HWO-style PSFs with controlled aberrations.
3. `observation`: PSF convolution and detector-noise simulation.
4. `modeling`: publication-grade Fisher/Asimov subhalo detectability.

The missing study layer is intentional next work: canonical study configs, sweep manifests, aggregation, and publication figures.

## Quick Start

Create the conda environment:

```bash
bash install.sh
```

Install HCIPy in that environment:

```bash
conda activate hwo-slaps
git clone https://github.com/ehpor/hcipy.git
pip install -e ./hcipy
```

Run a config:

```bash
python runner.py --config configs/master_config.yaml
```

Run the core tests:

```bash
python -m pytest -q
```

## Repository Notes

- `configs/master_config.yaml` is the current runnable example config.
- `outputs/` is ignored and used for run artifacts.
- `scratch/` is ignored and used for prototypes, archived runs, and local notes.
- Planning docs that should persist live under `docs/`.

## Copyright

Copyright 2025, by the California Institute of Technology. ALL RIGHTS RESERVED. United States Government Sponsorship acknowledged. Any commercial use must be negotiated with the Office of Technology Transfer at the California Institute of Technology.

This software may be subject to U.S. export control laws. By accepting this software, the user agrees to comply with all applicable U.S. export laws and regulations. User has the responsibility to obtain export licenses, or other export authority as may be required before exporting such information to foreign countries or providing access to foreign persons.

## Authors

Georgios Vassilakis (JPL)
