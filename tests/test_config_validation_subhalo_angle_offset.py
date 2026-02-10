"""Validation tests for signed angle-mode subhalo radial offsets."""

from __future__ import annotations

import copy
import importlib.util
from pathlib import Path

import pytest
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src" / "hwoslaps"


def _load_module(relative_path: str, module_name: str):
    module_path = SRC_ROOT / relative_path
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


validation = _load_module("config/validation.py", "hwoslaps_config_validation_subhalo_angle_offset")


def _load_master_config():
    config_path = PROJECT_ROOT / "configs" / "master_config.yaml"
    with config_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def test_angle_mode_accepts_negative_offset_pixels():
    config = _load_master_config()
    good_config = copy.deepcopy(config)
    good_config["lensing"]["subhalo"]["enabled"] = True
    good_config["lensing"]["subhalo"]["position"] = {
        "type": "angle",
        "angle": 45.0,
        "offset_pixels": -5.0,
    }

    validation.validate_or_raise(good_config)


@pytest.mark.parametrize("bad_offset", [float("inf"), float("-inf"), float("nan"), True, "bad"])
def test_angle_mode_rejects_non_finite_or_non_numeric_offset_pixels(bad_offset):
    config = _load_master_config()
    bad_config = copy.deepcopy(config)
    bad_config["lensing"]["subhalo"]["enabled"] = True
    bad_config["lensing"]["subhalo"]["position"] = {
        "type": "angle",
        "angle": 45.0,
        "offset_pixels": bad_offset,
    }

    with pytest.raises(ValueError, match="offset_pixels must be a finite number"):
        validation.validate_or_raise(bad_config)
