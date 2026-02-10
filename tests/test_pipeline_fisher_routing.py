"""Pipeline routing test for the Fisher detection path."""

from __future__ import annotations

from types import SimpleNamespace
from pathlib import Path
import sys

import pytest

pytest.importorskip("autolens")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from hwoslaps.pipeline import Pipeline
from hwoslaps.modeling.utils_fisher import FisherDetectionData, FisherLocalData


def test_pipeline_routes_detection_to_fisher(monkeypatch):
    import hwoslaps.pipeline as pipeline_module
    import hwoslaps.modeling.generator_fisher as fisher_generator
    import hwoslaps.modeling as modeling_module

    call_counts = {"fisher": 0, "legacy": 0}

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

    def _fake_legacy(*args, **kwargs):
        call_counts["legacy"] += 1
        raise AssertionError("Legacy chi-square detector should not run for modeling.detection='fisher'.")

    monkeypatch.setattr(fisher_generator, "perform_fisher_detection", _fake_fisher)
    monkeypatch.setattr(modeling_module, "perform_subhalo_detection", _fake_legacy)

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
    assert call_counts["legacy"] == 0
