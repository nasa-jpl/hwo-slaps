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

from _lensing_physics_helpers import (  # noqa: E402
    load_master_config,
    load_validation_module,
)

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
    """Reject a boolean global_seed, which int accepts by subclassing."""
    config = _base_config()
    config["global_seed"] = True
    _assert_rejected(config)


@pytest.mark.parametrize(
    "bad_mass",
    [np.nan, np.inf, -np.inf, 0.0, -1.0, -1.0e7],
)
def test_lensing_subhalo_mass_requires_positive_finite_value(bad_mass):
    """Reject a subhalo mass that is not positive and finite."""
    config = _base_config()
    config["lensing"]["subhalo"]["enabled"] = True
    config["lensing"]["subhalo"]["mass"] = bad_mass

    with pytest.raises(ValueError, match="lensing.subhalo.mass must be positive"):
        VALIDATION.validate_or_raise(config)


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
    """Reject a grid shape that is not a pair of positive integers."""
    config = _base_config()
    config["lensing"]["grid"]["shape"] = shape
    _assert_rejected(config)


@pytest.mark.parametrize("pixel_scale", [True, np.nan, np.inf, -0.01, 0.0])
def test_lensing_grid_pixel_scale_requires_positive_finite_non_bool_number(pixel_scale):
    """Reject a pixel scale that is boolean, non-finite, or <= 0."""
    config = _base_config()
    config["lensing"]["grid"]["pixel_scale"] = pixel_scale
    _assert_rejected(config)


@pytest.mark.parametrize("lens_z,source_z", [(0.5, 0.5), (1.0, 0.8)])
def test_lensing_redshift_order_requires_source_behind_lens(lens_z, source_z):
    """Reject a source redshift at or in front of the lens redshift."""
    config = _base_config()
    config["lensing"]["lens_galaxy"]["redshift"] = lens_z
    config["lensing"]["source_galaxy"]["redshift"] = source_z

    with pytest.raises(ValueError, match="source_galaxy.redshift must be greater than"):
        VALIDATION.validate_or_raise(config)


@pytest.mark.parametrize(
    "lens_z,source_z,bad_key",
    [
        (0.0, 2.0, "lensing.lens_galaxy.redshift"),
        (-0.1, 2.0, "lensing.lens_galaxy.redshift"),
        (0.2, 0.0, "lensing.source_galaxy.redshift"),
        (0.2, -1.0, "lensing.source_galaxy.redshift"),
    ],
)
def test_lensing_redshifts_must_be_positive(lens_z, source_z, bad_key):
    """Reject a lens or source redshift that is not positive."""
    config = _base_config()
    config["lensing"]["lens_galaxy"]["redshift"] = lens_z
    config["lensing"]["source_galaxy"]["redshift"] = source_z

    with pytest.raises(ValueError, match=f"{bad_key} must be positive"):
        VALIDATION.validate_or_raise(config)


@pytest.mark.parametrize(
    "path,value,expected_error",
    [
        (("lensing", "lens_galaxy", "redshift"), np.nan, "lensing.lens_galaxy.redshift"),
        (("lensing", "source_galaxy", "redshift"), np.inf, "lensing.source_galaxy.redshift"),
        (
            ("lensing", "lens_galaxy", "mass", "einstein_radius"),
            np.nan,
            "lensing.lens_galaxy.mass.einstein_radius",
        ),
        (
            ("lensing", "source_galaxy", "light", "intensity"),
            np.inf,
            "lensing.source_galaxy.light.intensity",
        ),
        (
            ("lensing", "source_galaxy", "light", "effective_radius"),
            np.nan,
            "lensing.source_galaxy.light.effective_radius",
        ),
    ],
)
def test_lensing_scalar_domains_reject_non_finite_values(path, value, expected_error):
    """Reject NaN or infinite values in scalar lensing parameters."""
    config = _base_config()
    _set_nested(config, path, value)

    with pytest.raises(ValueError, match=f"{expected_error} must be finite"):
        VALIDATION.validate_or_raise(config)


@pytest.mark.parametrize(
    "path,value,expected_error",
    [
        (("lensing", "lens_galaxy", "redshift"), True, "lensing.lens_galaxy.redshift"),
        (("lensing", "source_galaxy", "redshift"), True, "lensing.source_galaxy.redshift"),
        (("lensing", "lens_galaxy", "redshift"), "0.2", "lensing.lens_galaxy.redshift"),
        (("lensing", "source_galaxy", "redshift"), "2.0", "lensing.source_galaxy.redshift"),
    ],
)
def test_lensing_redshift_types_must_be_numeric(path, value, expected_error):
    """Reject boolean or string redshifts that are not numeric."""
    config = _base_config()
    _set_nested(config, path, value)

    with pytest.raises(ValueError, match=f"{expected_error} must be numeric"):
        VALIDATION.validate_or_raise(config)


def test_lensing_accepts_physical_redshift_order():
    """Accept a lens redshift strictly in front of the source."""
    config = _base_config()
    config["lensing"]["lens_galaxy"]["redshift"] = 0.2
    config["lensing"]["source_galaxy"]["redshift"] = 2.0

    VALIDATION.validate_or_raise(config)


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
    """Reject centre pairs holding booleans, strings, or non-finite values."""
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
    """Reject ellipticity components that are non-finite or |e| >= 1."""
    config = _base_config()
    _set_nested(config, path, value)
    _assert_rejected(config)


@pytest.mark.parametrize("bad_angle", [True, np.nan, np.inf, "90"])
def test_angle_position_requires_finite_non_bool_numeric_angle(bad_angle):
    """Reject a subhalo placement angle that is not a finite number."""
    config = _base_config()
    config["lensing"]["subhalo"]["position"] = {
        "type": "angle",
        "angle": bad_angle,
        "offset_pixels": 0.0,
    }
    _assert_rejected(config)


@pytest.mark.parametrize("bad_scatter", [True, np.nan, np.inf, -1.0, "20"])
def test_random_position_requires_finite_non_bool_nonnegative_scatter(bad_scatter):
    """Reject random placement scatter that is negative or non-finite."""
    config = _base_config()
    config["lensing"]["subhalo"]["position"] = {
        "type": "random",
        "scatter_pixels": bad_scatter,
    }
    _assert_rejected(config)


def test_angle_position_accepts_negative_offset_pixels():
    """Accept a negative offset placing the subhalo inside the ring."""
    config = _base_config()
    config["lensing"]["subhalo"]["enabled"] = True
    config["lensing"]["subhalo"]["position"] = {
        "type": "angle",
        "angle": 45.0,
        "offset_pixels": -5.0,
    }

    VALIDATION.validate_or_raise(config)


@pytest.mark.parametrize("bad_offset", [np.inf, -np.inf, np.nan, True, "bad"])
def test_angle_position_rejects_non_finite_or_non_numeric_offset_pixels(bad_offset):
    """Reject an offset_pixels value that is not a finite number."""
    config = _base_config()
    config["lensing"]["subhalo"]["enabled"] = True
    config["lensing"]["subhalo"]["position"] = {
        "type": "angle",
        "angle": 45.0,
        "offset_pixels": bad_offset,
    }

    with pytest.raises(ValueError, match="offset_pixels must be a finite number"):
        VALIDATION.validate_or_raise(config)


def test_nfw_subhalo_requires_concentration_block():
    """Reject an NFW subhalo with no concentration block."""
    config = _base_config()
    config["lensing"]["subhalo"]["enabled"] = True
    config["lensing"]["subhalo"]["model"] = "NFW"
    config["lensing"]["subhalo"].pop("concentration", None)

    with pytest.raises(ValueError, match="Missing required key 'concentration'"):
        VALIDATION.validate_or_raise(config)


def test_moline_concentration_requires_x_sub():
    """Reject the Moline concentration model with no x_sub value."""
    config = _base_config()
    config["lensing"]["subhalo"]["enabled"] = True
    config["lensing"]["subhalo"]["model"] = "NFW"
    config["lensing"]["subhalo"]["concentration"] = {
        "model": "moline2017_eq7",
        "h": 0.6774,
    }

    with pytest.raises(ValueError, match="Missing required key 'x_sub'"):
        VALIDATION.validate_or_raise(config)


def test_power_law_concentration_mode_is_accepted():
    """Accept the power-law concentration model with no extra keys."""
    config = _base_config()
    config["lensing"]["subhalo"]["enabled"] = True
    config["lensing"]["subhalo"]["model"] = "NFW"
    config["lensing"]["subhalo"]["concentration"] = {
        "model": "power_law",
    }

    VALIDATION.validate_or_raise(config)


@pytest.mark.parametrize(
    "concentration",
    [
        {"model": "moline2017_eq7", "x_sub": True, "h": 0.6774},
        {"model": "moline2017_eq7", "x_sub": np.nan, "h": 0.6774},
        {"model": "moline2017_eq7", "x_sub": 1.6, "h": 0.6774},
        {"model": "moline2017_eq7", "x_sub": 1.0, "h": True},
        {"model": "moline2017_eq7", "x_sub": 1.0, "h": np.inf},
    ],
)
def test_nfw_concentration_inputs_reject_bool_or_non_finite_values(concentration):
    """Reject boolean, non-finite, or out-of-domain concentration inputs."""
    config = _base_config()
    config["lensing"]["subhalo"]["enabled"] = True
    config["lensing"]["subhalo"]["model"] = "NFW"
    config["lensing"]["subhalo"]["concentration"] = concentration
    _assert_rejected(config)


@pytest.mark.parametrize("mass_msun", [1.0e5, 1.0e13])
def test_moline_config_rejects_mass_outside_study_domain(mass_msun):
    """Reject subhalo masses outside the Moline calibration domain."""
    config = _base_config()
    config["lensing"]["subhalo"]["enabled"] = True
    config["lensing"]["subhalo"]["model"] = "NFW"
    config["lensing"]["subhalo"]["mass"] = mass_msun
    config["lensing"]["subhalo"]["concentration"] = {
        "model": "moline2017_eq7",
        "x_sub": 1.0,
        "h": 0.6774,
    }
    _assert_rejected(config)


def test_concentration_block_rejects_unknown_keys():
    """Reject a misspelled or unknown concentration key."""
    config = _base_config()
    config["lensing"]["subhalo"]["enabled"] = True
    config["lensing"]["subhalo"]["model"] = "NFW"
    config["lensing"]["subhalo"]["concentration"] = {
        "model": "moline2017_eq7",
        "x_sub": 1.0,
        "h": 0.6774,
        "offset": 0.5,
    }

    with pytest.raises(ValueError, match="unsupported keys"):
        VALIDATION.validate_or_raise(config)


@pytest.mark.parametrize("bad_model", ["NFWTrunc", "nfwtruncated", "TruncatedNFW"])
def test_subhalo_model_rejects_unknown_names(bad_model):
    """Reject a subhalo model outside the supported set."""
    config = _base_config()
    config["lensing"]["subhalo"]["enabled"] = True
    config["lensing"]["subhalo"]["model"] = bad_model

    with pytest.raises(ValueError, match="lensing.subhalo.model must be one of"):
        VALIDATION.validate_or_raise(config)
