# Test Notes

Run the full suite:

```bash
python -m pytest -q
```

Run the fast physics/config tier:

```bash
python -m pytest -q \
  tests/test_lensing_physics.py \
  tests/test_lensing_concentration.py \
  tests/test_config_validation_redshift_order.py \
  tests/test_config_validation_nfw_concentration.py \
  tests/test_config_validation_subhalo_angle_offset.py \
  tests/test_fisher_core.py \
  tests/test_fisher_adapter.py
```

Run the AutoLens integration tier:

```bash
python -m pytest -q \
  tests/test_lensing_physics_integration.py \
  tests/test_lensing_nfw_provenance.py \
  tests/test_lensing_rng_isolation.py \
  tests/test_observation_module.py \
  tests/test_pipeline_fisher_routing.py \
  tests/test_fisher_detector_runtime.py
```

The Fisher / Asimov path is the maintained modeling path for study work.
