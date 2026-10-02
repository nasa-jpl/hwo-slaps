"""Packing arithmetic for measured cases; config evidence is checked upstream."""

import pytest

from studies.rasti.campaign.profile_execution import (
    STAGE3_MEMORY_PROFILE_REGISTRY,
    memory_admissible,
    validate_stage3_job,
)


@pytest.mark.parametrize("label,count", [("790", 3), ("900", 2), ("1284", 1)])
def test_reserved_memory_homogeneous_packing(label, count):
    profile = STAGE3_MEMORY_PROFILE_REGISTRY[label]
    peak = profile["peak_mib"]
    assert memory_admissible(0, [peak] * (count - 1), peak, 183359, 0.85)
    assert not memory_admissible(0, [peak] * count, peak, 183359, 0.85)
    job = {**profile, "memory_class": label, "memory_profile_id": profile["registry_id"]}
    policy = {"version": "stage3_v7", "memory_profiles": STAGE3_MEMORY_PROFILE_REGISTRY}
    validate_stage3_job(job, policy)
    job["kernel_shape"] = [999, 999]
    with pytest.raises(ValueError, match="memory profile registry"):
        validate_stage3_job(job, policy)
