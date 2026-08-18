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


def _nfw_config(model="NFW", concentration=None, truncation=None):
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
    if truncation is not None:
        config["lensing"]["subhalo"]["truncation"] = truncation
    return config


def _generated(config):
    """Generate a lensing system from a full configuration."""
    return generate_lensing_system(
        copy.deepcopy(config["lensing"]),
        full_config=config,
    )


def test_untruncated_nfw_records_no_truncation_or_offset_provenance():
    """Leave the new truncation and offset fields absent by default."""
    lensing_data = _generated(_nfw_config())

    assert lensing_data.subhalo_truncation_mode is None
    assert lensing_data.subhalo_truncation_tau is None
    assert lensing_data.subhalo_truncation_radius_arcsec is None
    assert lensing_data.subhalo_concentration_offset_dex is None
    assert (
        lensing_data.subhalo_concentration_pre_offset
        == lensing_data.subhalo_concentration
    )


def test_truncated_nfw_truncation_provenance_is_recorded():
    """Record truncation mode, ratio, and radius in the lensing metadata."""
    lensing_data = _generated(
        _nfw_config(
            model="NFWTruncated",
            truncation={"mode": "scale_ratio", "tau": 10.0},
        )
    )

    assert lensing_data.subhalo_model == "NFWTruncated"
    assert lensing_data.subhalo_truncation_mode == "scale_ratio"
    assert lensing_data.subhalo_truncation_tau == 10.0
    assert (
        lensing_data.subhalo_truncation_radius_arcsec
        == 10.0 * lensing_data.subhalo_scale_radius_arcsec
    )
    assert (
        lensing_data.subhalo_profile_parameters["truncation_radius"]
        == lensing_data.subhalo_truncation_radius_arcsec
    )
    assert lensing_data.subhalo_einstein_radius is None


def test_explicit_arcsec_truncation_provenance_is_recorded():
    """Record a directly declared truncation radius with no ratio."""
    lensing_data = _generated(
        _nfw_config(
            model="NFWTruncated",
            truncation={"mode": "explicit_arcsec", "radius_arcsec": 0.05},
        )
    )

    assert lensing_data.subhalo_truncation_mode == "explicit_arcsec"
    assert lensing_data.subhalo_truncation_tau is None
    assert lensing_data.subhalo_truncation_radius_arcsec == 0.05


def test_truncated_nfw_matches_untruncated_scale_parameters():
    """Share kappa_s and the scale radius with the untruncated NFW run."""
    untruncated = _generated(_nfw_config())
    truncated = _generated(
        _nfw_config(
            model="NFWTruncated",
            truncation={"mode": "scale_ratio", "tau": 10.0},
        )
    )

    assert truncated.subhalo_concentration == untruncated.subhalo_concentration
    assert truncated.subhalo_kappa_s == untruncated.subhalo_kappa_s
    assert (
        truncated.subhalo_scale_radius_arcsec
        == untruncated.subhalo_scale_radius_arcsec
    )
    assert not np.allclose(truncated.image, untruncated.image)


def test_explicit_concentration_provenance_is_recorded():
    """Record a directly declared concentration and its provenance."""
    lensing_data = _generated(
        _nfw_config(concentration={"model": "explicit", "c200": 15.0})
    )

    assert lensing_data.subhalo_concentration == 15.0
    assert lensing_data.subhalo_concentration_model == "explicit"
    assert lensing_data.subhalo_concentration_source == "explicit c200"
    assert lensing_data.subhalo_concentration_x_sub is None
    assert lensing_data.subhalo_concentration_h is None
    assert lensing_data.subhalo_concentration_pre_offset == 15.0
    assert lensing_data.subhalo_concentration_offset_dex is None


def test_concentration_offset_provenance_records_both_values():
    """Record the offset and the pre-offset concentration together."""
    baseline = _generated(_nfw_config())
    offset = _generated(
        _nfw_config(
            concentration={
                "model": "moline2017_eq7",
                "x_sub": 1.2,
                "h": None,
                "offset_dex": 0.5,
            }
        )
    )

    assert offset.subhalo_concentration_offset_dex == 0.5
    assert (
        offset.subhalo_concentration_pre_offset
        == baseline.subhalo_concentration
    )
    assert (
        offset.subhalo_concentration
        == baseline.subhalo_concentration * 10.0**0.5
    )
    assert offset.subhalo_concentration_model == "moline2017_eq7"


def test_zero_concentration_offset_reproduces_the_default_run():
    """Reproduce the un-offset run bit-identically at offset_dex zero."""
    baseline = _generated(_nfw_config())
    zero_offset = _generated(
        _nfw_config(
            concentration={
                "model": "moline2017_eq7",
                "x_sub": 1.2,
                "h": None,
                "offset_dex": 0.0,
            }
        )
    )

    assert zero_offset.subhalo_concentration == baseline.subhalo_concentration
    assert zero_offset.subhalo_kappa_s == baseline.subhalo_kappa_s
    assert (
        zero_offset.subhalo_scale_radius_arcsec
        == baseline.subhalo_scale_radius_arcsec
    )
    np.testing.assert_array_equal(zero_offset.image, baseline.image)
