"""Validation tests for NFW concentration configuration schema."""

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


validation = _load_module("config/validation.py", "hwoslaps_config_validation_nfw_concentration")


def _load_master_config():
    config_path = PROJECT_ROOT / "configs" / "master_config.yaml"
    with config_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def test_nfw_requires_concentration_block():
    config = _load_master_config()
    bad_config = copy.deepcopy(config)
    bad_config["lensing"]["subhalo"]["enabled"] = True
    bad_config["lensing"]["subhalo"]["model"] = "NFW"
    bad_config["lensing"]["subhalo"].pop("concentration", None)

    with pytest.raises(ValueError, match="Missing required key 'concentration'"):
        validation.validate_or_raise(bad_config)


def test_moline_requires_x_sub():
    config = _load_master_config()
    bad_config = copy.deepcopy(config)
    bad_config["lensing"]["subhalo"]["enabled"] = True
    bad_config["lensing"]["subhalo"]["model"] = "NFW"
    bad_config["lensing"]["subhalo"]["concentration"] = {
        "model": "moline2017_eq7",
        "h": 0.6774,
    }

    with pytest.raises(ValueError, match="Missing required key 'x_sub'"):
        validation.validate_or_raise(bad_config)


def test_power_law_concentration_mode_is_accepted():
    config = _load_master_config()
    good_config = copy.deepcopy(config)
    good_config["lensing"]["subhalo"]["enabled"] = True
    good_config["lensing"]["subhalo"]["model"] = "NFW"
    good_config["lensing"]["subhalo"]["concentration"] = {
        "model": "power_law",
    }

    validation.validate_or_raise(good_config)
