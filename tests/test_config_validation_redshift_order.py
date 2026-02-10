"""Validation tests for lens/source redshift physical-domain ordering."""

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


validation = _load_module("config/validation.py", "hwoslaps_config_validation_redshift_order")


def _load_master_config():
    config_path = PROJECT_ROOT / "configs" / "master_config.yaml"
    with config_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


@pytest.mark.parametrize("lens_z,source_z", [(0.5, 0.5), (1.0, 0.8)])
def test_rejects_non_physical_redshift_order(lens_z: float, source_z: float):
    config = _load_master_config()
    bad_config = copy.deepcopy(config)
    bad_config["lensing"]["lens_galaxy"]["redshift"] = lens_z
    bad_config["lensing"]["source_galaxy"]["redshift"] = source_z

    with pytest.raises(ValueError, match="source_galaxy.redshift must be greater than"):
        validation.validate_or_raise(bad_config)


def test_accepts_physical_redshift_order():
    config = _load_master_config()
    good_config = copy.deepcopy(config)
    good_config["lensing"]["lens_galaxy"]["redshift"] = 0.2
    good_config["lensing"]["source_galaxy"]["redshift"] = 2.0

    validation.validate_or_raise(good_config)
