"""Validation tests for observation detector physical domains."""

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


validation = _load_module("config/validation.py", "hwoslaps_config_validation_observation")


def _load_master_config():
    config_path = PROJECT_ROOT / "configs" / "master_config.yaml"
    with config_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _set_observation_value(config: dict, path: tuple[str, ...], value) -> None:
    current = config["observation"]
    for key in path[:-1]:
        current = current[key]
    current[path[-1]] = value


@pytest.mark.parametrize("bad_exposure_time", [True, float("nan"), float("inf")])
def test_observation_rejects_non_finite_or_bool_exposure_time(bad_exposure_time):
    config = _load_master_config()
    bad_config = copy.deepcopy(config)
    bad_config["observation"]["exposure_time"] = bad_exposure_time

    with pytest.raises(ValueError, match="observation.exposure_time"):
        validation.validate_or_raise(bad_config)


@pytest.mark.parametrize("bad_exposure_time", [0.0, -1.0])
def test_observation_rejects_non_positive_exposure_time(bad_exposure_time):
    config = _load_master_config()
    bad_config = copy.deepcopy(config)
    bad_config["observation"]["exposure_time"] = bad_exposure_time

    with pytest.raises(ValueError, match="observation.exposure_time"):
        validation.validate_or_raise(bad_config)


def test_observation_requires_throughput():
    config = _load_master_config()
    bad_config = copy.deepcopy(config)
    del bad_config["observation"]["throughput"]

    with pytest.raises(ValueError, match="throughput"):
        validation.validate_or_raise(bad_config)


@pytest.mark.parametrize(
    "bad_throughput", [True, float("nan"), float("inf"), 0.0, -0.5]
)
def test_observation_rejects_nonphysical_throughput(bad_throughput):
    config = _load_master_config()
    bad_config = copy.deepcopy(config)
    bad_config["observation"]["throughput"] = bad_throughput

    with pytest.raises(ValueError, match="observation.throughput"):
        validation.validate_or_raise(bad_config)


@pytest.mark.parametrize(
    "path,bad_value",
    [
        (("detector", "gain"), True),
        (("detector", "gain"), 0.0),
        (("detector", "gain"), -1.0),
        (("detector", "gain"), float("nan")),
        (("detector", "gain"), float("inf")),
        (("detector", "read_noise"), True),
        (("detector", "read_noise"), -0.1),
        (("detector", "read_noise"), float("nan")),
        (("detector", "read_noise"), float("inf")),
        (("detector", "dark_current"), True),
        (("detector", "dark_current"), -0.001),
        (("detector", "dark_current"), float("nan")),
        (("detector", "dark_current"), float("inf")),
        (("detector", "sky_background"), True),
        (("detector", "sky_background"), -0.1),
        (("detector", "sky_background"), float("nan")),
        (("detector", "sky_background"), float("inf")),
    ],
)
def test_observation_rejects_nonphysical_detector_values(path, bad_value):
    config = _load_master_config()
    bad_config = copy.deepcopy(config)
    _set_observation_value(bad_config, path, bad_value)

    key_path = "observation." + ".".join(path)
    with pytest.raises(ValueError, match=key_path):
        validation.validate_or_raise(bad_config)


@pytest.mark.parametrize(
    "path,good_value",
    [
        (("detector", "gain"), 1.0),
        (("detector", "read_noise"), 0.0),
        (("detector", "dark_current"), 0.0),
        (("detector", "sky_background"), 0.0),
    ],
)
def test_observation_accepts_detector_boundary_values_that_are_physical(path, good_value):
    config = _load_master_config()
    good_config = copy.deepcopy(config)
    _set_observation_value(good_config, path, good_value)

    validation.validate_or_raise(good_config)


@pytest.mark.parametrize("bad_output_format", [None, "fits", ""])
def test_observation_rejects_unsupported_output_format_when_declared(bad_output_format):
    config = _load_master_config()
    bad_config = copy.deepcopy(config)
    bad_config["observation"]["output_format"] = bad_output_format

    with pytest.raises(ValueError, match="observation.output_format"):
        validation.validate_or_raise(bad_config)
