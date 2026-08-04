"""Integration tests for lensing physics paths requiring `autolens`."""

from __future__ import annotations

import copy
import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("autolens")


TESTS_ROOT = Path(__file__).resolve().parent
if str(TESTS_ROOT) not in sys.path:
    sys.path.insert(0, str(TESTS_ROOT))

from _lensing_physics_helpers import (
    bootstrap_hwoslaps_namespace,
    load_lensing_anchor_fixture,
    load_master_config,
    load_module,
)

INTEGRATION_ANCHOR_MASSES = {
    "PointMass": 1.0e8,
    "SIS": 1.0e8,
    "NFW": 1.0e9,
}


def _load_lensing_generator_module():
    bootstrap_hwoslaps_namespace()
    load_module("constants.py", "hwoslaps.constants")
    load_module("lensing/mass_models.py", "hwoslaps.lensing.mass_models")
    load_module("lensing/utils.py", "hwoslaps.lensing.utils")
    return load_module("lensing/generator.py", "hwoslaps.lensing.generator")


def _build_lensing_config_for_model(model_name: str):
    config = load_master_config()
    cfg = copy.deepcopy(config)
    cfg["run_name"] = f"physics-{model_name.lower()}"
    cfg["global_seed"] = 11
    cfg["lensing"]["grid"]["shape"] = [64, 64]
    cfg["lensing"]["subhalo"]["enabled"] = True
    cfg["lensing"]["subhalo"]["mass"] = INTEGRATION_ANCHOR_MASSES[model_name]
    cfg["lensing"]["subhalo"]["model"] = model_name
    cfg["lensing"]["subhalo"]["position"] = {
        "type": "direct",
        "centre": [0.08, -0.05],
    }
    if model_name != "NFW":
        cfg["lensing"]["subhalo"].pop("concentration", None)
    return cfg


@pytest.mark.parametrize("model_name", ["PointMass", "SIS", "NFW"])
def test_env_02_smoke_generate_lensing_system_by_model(model_name: str):
    """Generate a finite scene for each supported subhalo model."""
    generator_module = _load_lensing_generator_module()
    cfg = _build_lensing_config_for_model(model_name)

    lensing_data = generator_module.generate_lensing_system(cfg["lensing"], full_config=cfg)

    assert lensing_data.image.shape == tuple(cfg["lensing"]["grid"]["shape"])
    assert lensing_data.tracer is not None
    assert lensing_data.image.size > 0
    assert np.isfinite(lensing_data.total_flux)
    if model_name == "NFW":
        assert lensing_data.subhalo_einstein_radius is None
    else:
        assert lensing_data.subhalo_einstein_radius is not None
        assert lensing_data.subhalo_einstein_radius > 0.0


@pytest.mark.parametrize("model_name", ["PointMass", "SIS", "NFW"])
def test_reg_04_optional_image_summary_anchors(model_name: str):
    """Reproduce the stored image shape, flux, and peak anchors."""
    anchors = load_lensing_anchor_fixture()
    summary_anchors = anchors["integration_image_summary"]
    model_key = model_name.lower()
    expected_summary = summary_anchors.get(model_key)
    if expected_summary is None:
        pytest.skip("No integration image summary anchor set for this model.")

    generator_module = _load_lensing_generator_module()
    cfg = _build_lensing_config_for_model(model_name)
    lensing_data = generator_module.generate_lensing_system(cfg["lensing"], full_config=cfg)
    observed = {
        "shape": list(lensing_data.image.shape),
        "total_flux": float(np.sum(lensing_data.image)),
        "peak": float(np.max(lensing_data.image)),
    }

    assert observed["shape"] == expected_summary["shape"]
    assert observed["total_flux"] == pytest.approx(expected_summary["total_flux"], rel=1.0e-10)
    assert observed["peak"] == pytest.approx(expected_summary["peak"], rel=1.0e-10)
