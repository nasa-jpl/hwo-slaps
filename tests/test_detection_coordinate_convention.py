"""Coordinate-convention regression tests for detection paths."""

from __future__ import annotations

import numpy as np
import pytest


pytest.importorskip("autolens")

from hwoslaps.modeling.chi_square_detector import DetectionResult
from hwoslaps.modeling.generator import perform_subhalo_detection


class _DummyArray:
    """Minimal Array2D-like object for tests."""

    def __init__(self, native: np.ndarray):
        self.native = native
        self.shape_native = native.shape


class _DummyObservation:
    """Minimal observation container required by perform_subhalo_detection."""

    def __init__(self, shape=(8, 8), pixel_scale=0.1):
        self.noiseless_source_eps = np.zeros(shape, dtype=float)
        self.exposure_time = 1000.0
        self.data = _DummyArray(np.zeros(shape, dtype=float))
        self.pixel_scale = pixel_scale
        self.detector_config = {
            "gain": 1.0,
            "read_noise": 1.0,
            "dark_current": 0.0,
            "sky_background": 0.0,
        }


class _DummyLensing:
    """Minimal lensing container with subhalo truth."""

    def __init__(self, has_subhalo=True, position_yx=(0.3, -0.2)):
        self.has_subhalo = has_subhalo
        self.subhalo_position = position_yx
        self.subhalo_mass = 1.0e8
        self.subhalo_model = "PointMass"


class _MockDetector:
    """Mock detector returning deterministic results."""

    def __init__(
        self,
        observation_data_no_subhalo,
        source_counts_ground_truth,
        snr_threshold,
        significance_levels,
    ):
        self.snr_threshold = snr_threshold
        self.significance_levels = significance_levels
        self.pixels_unmasked = 4
        self.num_regions = 1
        self.max_region_snr = 5.0
        self.snr_mask = np.array([True, True, False, False])
        self.snr_array = np.array([2.0, 2.5, 0.1, 0.2])
        self.labeled_regions = np.array([1, 1, 0, 0])
        self.variance_2d = np.ones((2, 2), dtype=float)

    def detect_at_position(self, observation_with_subhalo, subhalo_position):
        residual = np.zeros(4, dtype=float)
        out = {}
        for p in self.significance_levels:
            out[p] = DetectionResult(
                chi2_value=1.0,
                threshold=0.5,
                detected=True,
                significance_level=p,
                dof=4,
                position=subhalo_position,
                snr_mask=self.snr_mask,
                residual=residual,
                num_regions=self.num_regions,
                max_region_snr=self.max_region_snr,
            )
        return out


def test_gof_preserves_canonical_yx_position(monkeypatch):
    import hwoslaps.modeling.generator as modeling_generator

    monkeypatch.setattr(modeling_generator, "ChiSquareSubhaloDetector", _MockDetector)

    obs_baseline = _DummyObservation()
    obs_test = _DummyObservation()
    lensing_baseline = _DummyLensing(has_subhalo=False, position_yx=(0.0, 0.0))
    lensing_test = _DummyLensing(has_subhalo=True, position_yx=(0.3, -0.2))

    detection_data = perform_subhalo_detection(
        observation_baseline=obs_baseline,
        observation_test=obs_test,
        lensing_baseline=lensing_baseline,
        lensing_test=lensing_test,
        detection_config={"snr_threshold": 1.0, "significance_levels": [1.0e-3]},
        full_config={"run_name": "unit"},
    )

    assert detection_data.true_subhalo_position == (0.3, -0.2)
    assert detection_data.detection_results[1.0e-3].position == (0.3, -0.2)
