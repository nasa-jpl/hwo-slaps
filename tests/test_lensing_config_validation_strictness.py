"""Strict validation tests for lensing configs before study sweeps."""

from __future__ import annotations

import copy
import sys
from pathlib import Path

import numpy as np
import pytest

TESTS_ROOT = Path(__file__).resolve().parent
if str(TESTS_ROOT) not in sys.path:
    sys.path.insert(0, str(TESTS_ROOT))

from _lensing_physics_helpers import load_master_config, load_validation_module  # noqa: E402


VALIDATION = load_validation_module()


def _base_config():
    return copy.deepcopy(load_master_config())


def _set_nested(config, path, value):
    current = config
    for key in path[:-1]:
        current = current[key]
    current[path[-1]] = value


def _assert_rejected(config):
    with pytest.raises(ValueError):
        VALIDATION.validate_or_raise(config)


def test_top_level_global_seed_rejects_bool():
    config = _base_config()
    config["global_seed"] = True
    _assert_rejected(config)


@pytest.mark.parametrize(
    "shape",
    [
        [64.5, 64],
        [64, "64"],
        [0, 64],
        [-1, 64],
        [np.nan, 64],
    ],
)
def test_lensing_grid_shape_requires_positive_integer_pixels(shape):
    config = _base_config()
    config["lensing"]["grid"]["shape"] = shape
    _assert_rejected(config)


@pytest.mark.parametrize("pixel_scale", [True, np.nan, np.inf, -0.01, 0.0])
def test_lensing_grid_pixel_scale_requires_positive_finite_non_bool_number(pixel_scale):
    config = _base_config()
    config["lensing"]["grid"]["pixel_scale"] = pixel_scale
    _assert_rejected(config)


@pytest.mark.parametrize(
    "path,value",
    [
        (("lensing", "lens_galaxy", "mass", "centre"), [np.nan, 0.0]),
        (("lensing", "lens_galaxy", "mass", "centre"), [True, 0.0]),
        (("lensing", "source_galaxy", "light", "centre"), ["bad", 0.0]),
        (("lensing", "source_galaxy", "light", "centre"), [0.0, np.inf]),
        (("lensing", "subhalo", "position", "centre"), [np.nan, 0.0]),
        (("lensing", "subhalo", "position", "centre"), [0.0, "bad"]),
    ],
)
def test_lensing_coordinate_pairs_require_finite_non_bool_numbers(path, value):
    config = _base_config()
    config["lensing"]["subhalo"]["position"] = {
        "type": "direct",
        "centre": [0.1, -0.2],
    }
    _set_nested(config, path, value)
    _assert_rejected(config)


@pytest.mark.parametrize(
    "path,value",
    [
        (("lensing", "lens_galaxy", "mass", "ell_comps"), [np.nan, 0.0]),
        (("lensing", "lens_galaxy", "mass", "ell_comps"), [True, 0.0]),
        (("lensing", "lens_galaxy", "mass", "ell_comps"), [1.0, 0.0]),
        (("lensing", "lens_galaxy", "mass", "ell_comps"), [2.0, 0.0]),
        (("lensing", "source_galaxy", "light", "ell_comps"), [0.0, np.inf]),
        (("lensing", "source_galaxy", "light", "ell_comps"), [0.8, 0.8]),
    ],
)
def test_lensing_ellipticity_components_require_finite_physical_values(path, value):
    config = _base_config()
    _set_nested(config, path, value)
    _assert_rejected(config)


@pytest.mark.parametrize("bad_angle", [True, np.nan, np.inf, "90"])
def test_angle_position_requires_finite_non_bool_numeric_angle(bad_angle):
    config = _base_config()
    config["lensing"]["subhalo"]["position"] = {
        "type": "angle",
        "angle": bad_angle,
        "offset_pixels": 0.0,
    }
    _assert_rejected(config)


@pytest.mark.parametrize("bad_scatter", [True, np.nan, np.inf, -1.0, "20"])
def test_random_position_requires_finite_non_bool_nonnegative_scatter(bad_scatter):
    config = _base_config()
    config["lensing"]["subhalo"]["position"] = {
        "type": "random",
        "scatter_pixels": bad_scatter,
    }
    _assert_rejected(config)


@pytest.mark.parametrize(
    "concentration",
    [
        {"model": "moline2017_eq7", "x_sub": True, "h": 0.6774},
        {"model": "moline2017_eq7", "x_sub": np.nan, "h": 0.6774},
        {"model": "moline2017_eq7", "x_sub": 1.0, "h": True},
        {"model": "moline2017_eq7", "x_sub": 1.0, "h": np.inf},
    ],
)
def test_nfw_concentration_inputs_reject_bool_or_non_finite_values(concentration):
    config = _base_config()
    config["lensing"]["subhalo"]["enabled"] = True
    config["lensing"]["subhalo"]["model"] = "NFW"
    config["lensing"]["subhalo"]["concentration"] = concentration
    _assert_rejected(config)
