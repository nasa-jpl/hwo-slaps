"""Validation tests for lens/source redshift and scalar physical domains."""

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


def _set_nested_value(config, path, value):
    current = config
    for key in path[:-1]:
        current = current[key]
    current[path[-1]] = value


@pytest.mark.parametrize("lens_z,source_z", [(0.5, 0.5), (1.0, 0.8)])
def test_rejects_non_physical_redshift_order(lens_z: float, source_z: float):
    config = _load_master_config()
    bad_config = copy.deepcopy(config)
    bad_config["lensing"]["lens_galaxy"]["redshift"] = lens_z
    bad_config["lensing"]["source_galaxy"]["redshift"] = source_z

    with pytest.raises(ValueError, match="source_galaxy.redshift must be greater than"):
        validation.validate_or_raise(bad_config)


@pytest.mark.parametrize(
    "lens_z,source_z,bad_key",
    [
        (0.0, 2.0, "lensing.lens_galaxy.redshift"),
        (-0.1, 2.0, "lensing.lens_galaxy.redshift"),
        (0.2, 0.0, "lensing.source_galaxy.redshift"),
        (0.2, -1.0, "lensing.source_galaxy.redshift"),
    ],
)
def test_rejects_non_positive_redshifts(lens_z: float, source_z: float, bad_key: str):
    config = _load_master_config()
    bad_config = copy.deepcopy(config)
    bad_config["lensing"]["lens_galaxy"]["redshift"] = lens_z
    bad_config["lensing"]["source_galaxy"]["redshift"] = source_z

    with pytest.raises(ValueError, match=f"{bad_key} must be positive"):
        validation.validate_or_raise(bad_config)


@pytest.mark.parametrize(
    "path,value,expected_error",
    [
        (("lensing", "lens_galaxy", "redshift"), float("nan"), "lensing.lens_galaxy.redshift"),
        (("lensing", "source_galaxy", "redshift"), float("inf"), "lensing.source_galaxy.redshift"),
        (
            ("lensing", "lens_galaxy", "mass", "einstein_radius"),
            float("nan"),
            "lensing.lens_galaxy.mass.einstein_radius",
        ),
        (
            ("lensing", "source_galaxy", "light", "intensity"),
            float("inf"),
            "lensing.source_galaxy.light.intensity",
        ),
        (
            ("lensing", "source_galaxy", "light", "effective_radius"),
            float("nan"),
            "lensing.source_galaxy.light.effective_radius",
        ),
    ],
)
def test_rejects_non_finite_lensing_scalars(path, value, expected_error: str):
    config = _load_master_config()
    bad_config = copy.deepcopy(config)
    _set_nested_value(bad_config, path, value)

    with pytest.raises(ValueError, match=f"{expected_error} must be finite"):
        validation.validate_or_raise(bad_config)


@pytest.mark.parametrize(
    "path,value,expected_error",
    [
        (("lensing", "lens_galaxy", "redshift"), True, "lensing.lens_galaxy.redshift"),
        (("lensing", "source_galaxy", "redshift"), True, "lensing.source_galaxy.redshift"),
        (("lensing", "lens_galaxy", "redshift"), "0.2", "lensing.lens_galaxy.redshift"),
        (("lensing", "source_galaxy", "redshift"), "2.0", "lensing.source_galaxy.redshift"),
    ],
)
def test_rejects_non_numeric_redshift_types(path, value, expected_error: str):
    config = _load_master_config()
    bad_config = copy.deepcopy(config)
    _set_nested_value(bad_config, path, value)

    with pytest.raises(ValueError, match=f"{expected_error} must be numeric"):
        validation.validate_or_raise(bad_config)


def test_accepts_physical_redshift_order():
    config = _load_master_config()
    good_config = copy.deepcopy(config)
    good_config["lensing"]["lens_galaxy"]["redshift"] = 0.2
    good_config["lensing"]["source_galaxy"]["redshift"] = 2.0

    validation.validate_or_raise(good_config)
