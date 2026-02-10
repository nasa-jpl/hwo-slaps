# Lensing Physics Test Tiers

Use these commands to run the dedicated lensing-physics suite in two tiers.

## Core Tier (no `autolens` required)

```bash
pytest -q \
  tests/test_lensing_physics.py \
  tests/test_lensing_concentration.py \
  tests/test_config_validation_redshift_order.py \
  tests/test_config_validation_nfw_concentration.py \
  tests/test_config_validation_subhalo_angle_offset.py
```

## Integration Tier (`autolens` required)

```bash
pytest -q \
  tests/test_lensing_physics_integration.py \
  tests/test_lensing_nfw_provenance.py \
  tests/test_lensing_rng_isolation.py \
  tests/test_detection_coordinate_convention.py
```
