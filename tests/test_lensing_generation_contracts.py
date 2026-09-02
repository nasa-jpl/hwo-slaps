"""Integration-level contracts for generated lensing data."""

from __future__ import annotations

import copy

import numpy as np
import pytest

pytest.importorskip("autolens")

from hwoslaps.lensing.generator import generate_lensing_system  # noqa: E402
from hwoslaps.lensing.mass_models import concentration_mass_relation  # noqa: E402


def _small_lensing_config():
    return {
        "run_name": "lensing-contract",
        "global_seed": 123,
        "lensing": {
            "grid": {"shape": [48, 48], "pixel_scale": 0.02},
            "lens_galaxy": {
                "redshift": 0.2,
                "mass": {
                    "type": "Isothermal",
                    "einstein_radius": 1.0,
                    "centre": [0.0, 0.0],
                    "ell_comps": [0.1, 0.0],
                },
            },
            "source_galaxy": {
                "redshift": 2.5,
                "light": {
                    "type": "Exponential",
                    "centre": [-0.03, 0.08],
                    "ell_comps": [0.14516129, 0.25142673],
                    "intensity": 2.0,
                    "effective_radius": 0.11,
                },
            },
            "subhalo": {
                "enabled": True,
                "mass": 1.0e8,
                "model": "PointMass",
                "position": {
                    "type": "direct",
                    "centre": [0.08, -0.05],
                },
            },
            "cosmology": "Planck15",
        },
    }


def test_lensing_data_config_is_immutable_snapshot_of_generation_config():
    """Snapshot the generation config so later edits cannot reach it."""
    config = _small_lensing_config()

    lensing_data = generate_lensing_system(
        copy.deepcopy(config["lensing"]),
        full_config=config,
    )

    config["run_name"] = "mutated-after-generation"
    config["lensing"]["subhalo"]["mass"] = 9.9e12
    config["lensing"]["subhalo"]["position"]["centre"] = [9.0, 9.0]

    assert lensing_data.config["run_name"] == "lensing-contract"
    assert lensing_data.config["lensing"]["subhalo"]["mass"] == pytest.approx(1.0e8)
    assert lensing_data.config["lensing"]["subhalo"]["position"]["centre"] == [0.08, -0.05]


def test_repeated_generation_isolates_the_uniform_sub_grid():
    """Reuse the cached construction without sharing mutable grid state.

    Ray tracing rebuilds the uniform sub-pixel grid from the mask, which
    dominates per-node cost in a grid map. Repeated generations at one
    geometry must reuse that sub-grid and still hand each caller its own
    grid with bit-identical coordinates.
    """
    config = _small_lensing_config()

    from hwoslaps.lensing.generator import (
        _UNIFORM_GRID_TEMPLATES,
        clear_uniform_grid_cache,
    )

    clear_uniform_grid_cache()
    first = generate_lensing_system(config["lensing"], full_config=config)
    second = generate_lensing_system(config["lensing"], full_config=config)

    assert first.grid is not second.grid
    assert first.grid.over_sampled is not second.grid.over_sampled
    assert first.grid.over_sampler is not second.grid.over_sampler
    np.testing.assert_array_equal(first.grid.array, second.grid.array)
    np.testing.assert_array_equal(first.image, second.image)

    expected = np.array(second.grid.over_sampled.array, copy=True)
    try:
        first.grid.over_sampled.array[0, 0] += 1.0
    except ValueError:
        pass
    third = generate_lensing_system(config["lensing"], full_config=config)
    np.testing.assert_array_equal(third.grid.over_sampled.array, expected)

    wider = copy.deepcopy(config)
    wider["lensing"]["grid"]["shape"] = [64, 64]
    other = generate_lensing_system(wider["lensing"], full_config=wider)

    assert other.grid.over_sampled is not first.grid.over_sampled
    for size in (8, 16, 32, 48, 80):
        geometry = copy.deepcopy(config)
        geometry["lensing"]["grid"]["shape"] = [size, size]
        generate_lensing_system(geometry["lensing"], full_config=geometry)
    assert len(_UNIFORM_GRID_TEMPLATES) <= 4
    clear_uniform_grid_cache()
    assert not _UNIFORM_GRID_TEMPLATES


def test_nfw_concentration_provenance_is_recorded():
    """Record the concentration model, inputs, and derived NFW terms."""
    config = _small_lensing_config()
    config["lensing"]["grid"]["shape"] = [120, 120]
    config["lensing"]["subhalo"]["mass"] = 1.0e9
    config["lensing"]["subhalo"]["model"] = "NFW"
    config["lensing"]["subhalo"]["concentration"] = {
        "model": "moline2017_eq7",
        "x_sub": 1.2,
        "h": None,
    }

    lensing_data = generate_lensing_system(
        copy.deepcopy(config["lensing"]),
        full_config=config,
    )

    assert lensing_data.subhalo_model == "NFW"
    assert lensing_data.subhalo_concentration is not None
    assert lensing_data.subhalo_concentration_model == "moline2017_eq7"
    assert lensing_data.subhalo_concentration_source == "Moline2017 Eq7 Table2"
    assert lensing_data.subhalo_concentration_x_sub == pytest.approx(1.2)
    assert lensing_data.subhalo_concentration_h is not None
    assert lensing_data.subhalo_concentration_h > 0
    assert lensing_data.subhalo_einstein_radius is None
    assert lensing_data.subhalo_kappa_s is not None
    assert lensing_data.subhalo_kappa_s > 0
    assert lensing_data.subhalo_scale_radius_arcsec is not None
    assert lensing_data.subhalo_scale_radius_arcsec > 0
    assert lensing_data.subhalo_profile_parameters["kappa_s"] == pytest.approx(
        lensing_data.subhalo_kappa_s
    )
    assert lensing_data.subhalo_profile_parameters["scale_radius"] == pytest.approx(
        lensing_data.subhalo_scale_radius_arcsec
    )

    expected = concentration_mass_relation(
        config["lensing"]["subhalo"]["mass"],
        model="moline2017_eq7",
        x_sub=lensing_data.subhalo_concentration_x_sub,
        h=lensing_data.subhalo_concentration_h,
    )
    assert lensing_data.subhalo_concentration == pytest.approx(expected, rel=1e-12)


def test_pointmass_subhalo_einstein_radius_is_populated():
    """Populate the Einstein radius for a point-mass subhalo."""
    config = _small_lensing_config()

    lensing_data = generate_lensing_system(
        copy.deepcopy(config["lensing"]),
        full_config=config,
    )

    assert lensing_data.subhalo_model == "PointMass"
    assert lensing_data.subhalo_einstein_radius is not None
    assert lensing_data.subhalo_einstein_radius > 0


def _nfw_config(model="NFW", concentration=None):
    """Return a small NFW-family configuration for provenance contracts."""
    config = _small_lensing_config()
    config["lensing"]["grid"]["shape"] = [120, 120]
    config["lensing"]["subhalo"]["mass"] = 1.0e9
    config["lensing"]["subhalo"]["model"] = model
    config["lensing"]["subhalo"]["concentration"] = concentration or {
        "model": "moline2017_eq7",
        "x_sub": 1.2,
        "h": None,
    }
    return config


def _generated(config):
    """Generate a lensing system from a full configuration."""
    return generate_lensing_system(
        copy.deepcopy(config["lensing"]),
        full_config=config,
    )
