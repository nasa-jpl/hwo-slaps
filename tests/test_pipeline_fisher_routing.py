"""Pipeline routing test for the Fisher detection path."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip("autolens")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from hwoslaps.modeling.utils_fisher import FisherDetectionData, FisherLocalData
from hwoslaps.pipeline import Pipeline


def _make_fisher_config_with_required_fields() -> dict:
    return {
        "mode": "local",
        "snr_threshold": 3.0,
        "include_background_offset": True,
        "finite_diff": {
            "centre_arcsec": 1.0e-3,
            "einstein_radius_arcsec": 1.0e-3,
            "ell_comp": 1.0e-3,
            "source_intensity_frac": 1.0e-2,
            "source_reff_frac": 1.0e-2,
        },
        "map": {
            "type": "ring",
            "ring": {
                "num_angles": 24,
                "offset_pixels": 0.0,
            },
            "explicit_positions_yx": None,
        },
    }


def _make_detector_stub(label: str):
    class _DetectorStub:
        def __init__(self, *args, **kwargs):
            self.label = label
            self.pixels_unmasked = 25
            self.n_nuisance = 2
            self.gram_condition_number = 1.5
            self.nuisance_names = ["n0", "n1"]
            self.prior_precision_diagonal = [0.0, 0.0]
            self.n_psf_modes = 0
            self.psf_mode_names = []
            self.n_psf_fit_modes = 0
            self.n_psf_scan_modes = 0
            self.psf_fit_mode_names = []
            self.psf_scan_mode_names = []
            self.psf_mismatch_enabled = False

        def compute_local(self, observation_test, lensing_test):
            return FisherLocalData(
                snr_asimov=1.0,
                delta_chi2_raw=1.0,
                delta_chi2_profiled=1.0,
                degradation=1.0,
                pixels_unmasked=self.pixels_unmasked,
                n_nuisance=self.n_nuisance,
                gram_condition_number=self.gram_condition_number,
            )

        def compute_map(self):
            raise AssertionError("compute_map should not be called in local mode")

    return _DetectorStub


def test_pipeline_routes_detection_to_fisher(monkeypatch):
    """Route a detection-mode run to the Fisher generator exactly once."""
    import hwoslaps.modeling.generator_fisher as fisher_generator
    import hwoslaps.pipeline as pipeline_module

    call_counts = {"fisher": 0}

    dummy_psf = SimpleNamespace()
    dummy_lensing = SimpleNamespace(has_subhalo=True)
    dummy_obs = SimpleNamespace()
    fisher_result = FisherDetectionData(
        mode="local",
        local=FisherLocalData(
            snr_asimov=2.0,
            delta_chi2_raw=5.0,
            delta_chi2_profiled=4.0,
            degradation=0.8,
            pixels_unmasked=100,
            n_nuisance=12,
            gram_condition_number=10.0,
            true_subhalo_position=(0.1, -0.2),
            true_subhalo_mass=1.0e8,
            true_subhalo_model="PointMass",
        ),
        map=None,
        snr_threshold=3.0,
        include_background_offset=True,
        finite_diff={},
        map_config={},
        pixels_unmasked=100,
        n_nuisance=12,
        gram_condition_number=10.0,
        pixel_scale=0.1,
        config={"run_name": "unit"},
    )

    monkeypatch.setattr(pipeline_module, "generate_psf_system", lambda *args, **kwargs: dummy_psf)
    monkeypatch.setattr(pipeline_module, "generate_lensing_system", lambda *args, **kwargs: dummy_lensing)
    monkeypatch.setattr(pipeline_module, "generate_observation", lambda *args, **kwargs: dummy_obs)

    def _fake_fisher(*args, **kwargs):
        call_counts["fisher"] += 1
        return fisher_result

    monkeypatch.setattr(fisher_generator, "perform_fisher_detection", _fake_fisher)

    config = {
        "run_name": "unit",
        "lensing": {"subhalo": {"enabled": True}},
        "psf": {},
        "observation": {},
        "modeling": {
            "enabled": True,
            "detection": "fisher",
            "fisher": {"mode": "local"},
        },
        "plotting": {"enabled": False, "output_dir": "/tmp"},
    }

    result = Pipeline(verbose=False)._run_detection_pipeline(config)

    assert result is fisher_result
    assert call_counts["fisher"] == 1


def test_generator_uses_fisher_detector(monkeypatch):
    """Construct FisherDetector inside perform_fisher_detection."""
    import hwoslaps.modeling.generator_fisher as fisher_generator

    calls = []

    class _Detector(_make_detector_stub("fisher")):
        def __init__(self, *args, **kwargs):
            calls.append("fisher")
            super().__init__(*args, **kwargs)

    monkeypatch.setattr(fisher_generator, "FisherDetector", _Detector)

    result = fisher_generator.perform_fisher_detection(
        observation_baseline=SimpleNamespace(pixel_scale=0.1),
        observation_test=SimpleNamespace(),
        lensing_baseline=SimpleNamespace(),
        lensing_test=SimpleNamespace(),
        psf_data=SimpleNamespace(),
        detection_config={"fisher": _make_fisher_config_with_required_fields()},
        full_config={"run_name": "unit"},
    )

    assert calls == ["fisher"]
    assert result.mode == "local"
    assert result.psf_mismatch_enabled is False
