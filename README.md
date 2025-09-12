# hwo-slaps
HWO-SLAPS: HWO Strong Lensing Analysis Pipeline for Subhalos

## Overview

This pipeline determines the PSF quality and stability requirements for the Habitable Worlds Observatory (HWO) to detect 10^7 M☉ dark matter subhalos through strong gravitational lensing observations.

The pipeline simulates the complete observation process: generates realistic galaxy lensing systems with embedded subhalos, creates time-varying PSFs with telescope aberrations, simulates observations with noise, attempts to recover subhalos through lens modeling, and analyzes detection performance to derive specific telescope requirements.

## Objectives

- Quantify static PSF quality needed for 10^7 M☉ subhalo detection
- Determine PSF stability requirements over various timescales  
- Identify which optical aberrations are most harmful to subhalo detection
- Provide actionable requirements for HWO telescope design

## Pipeline Modules

1. **Lensing System Generation** - Creates galaxy-galaxy strong lensing systems with known subhalo populations
2. **PSF Generation** - Simulates telescope PSFs with realistic aberrations and temporal variations
3. **Observation Simulation** - Applies PSF convolution and adds instrument noise
4. **Lens Modeling & Detection** - Attempts to recover injected subhalos from simulated data

## Implementation Progress

| Module      | Component                   | Prototyping | API Integration | Static PSF |
|-------------|----------------------------|-------------|-----------------|------------|
| **Module 1** | Lensing System Generation   | ✅          | ✅              | ✅         |
| **Module 2** | PSF Generation              | ✅          | ✅              | ✅         |
| **Module 3** | Observation Simulation      | ✅          | ✅              | ✅         |
| **Module 4** | Lens Modeling & Detection   | ✅          | ✅              | ✅         |

**Key:** ✅ Complete | 🟡 In Progress | 🔴 Not Started

## Installation
This will create a conda environment called hwo-slaps:
```bash
bash install.sh
```
Next, test that the installation imports with:
```bash
python tests/test_installation.py
```


## Copyright
Copyright 2025, by the California Institute of Technology. ALL RIGHTS RESERVED. United States Government Sponsorship acknowledged. Any commercial use must be negotiated with the Office of Technology Transfer at the California Institute of Technology.
 
This software may be subject to U.S. export control laws. By accepting this software, the user agrees to comply with all applicable U.S. export laws and regulations. User has the responsibility to obtain export licenses, or other export authority as may be required before exporting such information to foreign countries or providing access to foreign persons.

## Authors
Georgios Vassilakis (JPL)
  
