"""Contracts for the Stage 0 observation runner's pure helpers.

The rendering path itself is exercised by the campaign smoke rather than
by a unit test. What is pinned here is the numerical guard that stands
between the PSF convolution and the pre-registered expected-variance
map, because that guard is the only place the runner is allowed to
change a pixel value.
"""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_ROOT = PROJECT_ROOT/"scripts"
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

from run_stage0_observation import (  # noqa: E402
    ARTIFACT_NAME,
    CONVOLUTION_ROUNDOFF_TOLERANCE,
    _clip_convolution_roundoff,
)


def test_artifact_name_matches_the_frozen_declaration():
    """The runner writes the artifact the design freeze declares."""
    from hwoslaps.campaign.design_freeze import load_design_freeze

    freeze = load_design_freeze(
        PROJECT_ROOT/"configs"/"design"/"design_freeze_v1.yaml"
    )
    assert freeze["stage0"]["artifact"] == ARTIFACT_NAME
    assert freeze["stage0"]["runner"].endswith("run_stage0_observation.py")


def test_a_non_negative_map_passes_through_unchanged():
    """Nothing is touched when the convolution left no negatives."""
    values = np.array([[0.0, 1.0], [2.5, 4.0]])
    clipped, minimum = _clip_convolution_roundoff(values)
    assert np.array_equal(clipped, values)
    assert minimum == 0.0


def test_round_off_negatives_are_clipped_and_reported():
    """A round-off excursion is clipped to zero and its size recorded."""
    values = np.array([[-2.8e-15, 1.0], [100.0, 0.0]])
    clipped, minimum = _clip_convolution_roundoff(values)
    assert minimum == pytest.approx(-2.8e-15)
    assert clipped.min() == 0.0
    assert clipped[0, 1] == 1.0
    assert clipped[1, 0] == 100.0


def test_a_real_negative_excursion_fails_closed():
    """A negative far beyond round-off is a fault, not a rounding artifact."""
    peak = 100.0
    values = np.array([[-0.5*peak, 1.0], [peak, 0.0]])
    with pytest.raises(ValueError, match="not round-off"):
        _clip_convolution_roundoff(values)


def test_the_tolerance_is_a_declared_fraction_of_the_peak():
    """The guard scales with the map, and sits at the round-off level."""
    assert CONVOLUTION_ROUNDOFF_TOLERANCE == 1.0e-9
    peak = 1.0e6
    just_inside = np.array([[-0.5*CONVOLUTION_ROUNDOFF_TOLERANCE*peak, peak]])
    just_outside = np.array([[-2.0*CONVOLUTION_ROUNDOFF_TOLERANCE*peak, peak]])
    assert _clip_convolution_roundoff(just_inside)[0].min() == 0.0
    with pytest.raises(ValueError):
        _clip_convolution_roundoff(just_outside)
