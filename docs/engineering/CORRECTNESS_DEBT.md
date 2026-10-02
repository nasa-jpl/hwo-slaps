# Correctness repairs and remaining scientific boundaries

Three independently reproduced defects were repaired separately from numerical
model/optimizer changes. XTX baseline controls use the untouched bf73e2c source
and the same keeper harness as the candidate.

- Supervised input exhaustion: a completed pending batch returned four of ten
  inputs. The owner now distinguishes input exhaustion from an empty batch;
  the old strict xfail is a passing ten-input regression.
- Profile settings round trip: an explicit posterior start-separation value of
  2.0 reconstructed as 1.0. Serialization now preserves that field; defaults and
  optimizer mathematics are unchanged.
- Trial construction from a real smooth scene: subhalo_mass=None caused
  float(None) for NFW, SIS and PointMass. Missing truth mass now triggers the
  existing physical recomputation, with three real-backend keeper cases.

New-interface bugs found during validation were repaired at their owners:
external kernels no longer access unused segment metadata; declared fit kernels
survive preparation; expected observations are distinguished from injections;
actual kernel metadata reaches result/CLI persistence; invalid PSF payload
errors name the offending fields; test collection uses real package namespaces.

These repairs do not establish a measured change in the submitted paper's
results. Unsupported physical families, flexible source reconstruction,
foreground light and chromatic image formation remain explicit future work.
Current source/optimizer equations and supported sampling are covered by
independent physical oracles and reference/JAX parity; validation receipts
record the actual proof and its limits.
