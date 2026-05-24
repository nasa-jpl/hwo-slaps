"""Integration-level contracts for generated lensing data."""

from __future__ import annotations

import copy

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


def test_nfw_concentration_provenance_is_recorded():
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
    config = _small_lensing_config()

    lensing_data = generate_lensing_system(
        copy.deepcopy(config["lensing"]),
        full_config=config,
    )

    assert lensing_data.subhalo_model == "PointMass"
    assert lensing_data.subhalo_einstein_radius is not None
    assert lensing_data.subhalo_einstein_radius > 0
