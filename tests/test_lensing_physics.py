"""Dedicated lensing physics unit tests.

This module provides core equation and validation checks that do not require
`autolens`.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest
from astropy import constants as const

TESTS_ROOT = Path(__file__).resolve().parent
if str(TESTS_ROOT) not in sys.path:
    sys.path.insert(0, str(TESTS_ROOT))

from _lensing_physics_helpers import (
    Planck15CosmologyAdapter,
    load_constants_module,
    load_lensing_anchor_fixture,
    load_lensing_utils_module,
    load_mass_models_module,
)


COSMOLOGY = Planck15CosmologyAdapter()
CONSTANTS = load_constants_module()
MASS_MODELS = load_mass_models_module()
LENSING_UTILS = load_lensing_utils_module()


def _nfw_lensing_terms(mass_msun: float, concentration: float, z_lens: float, z_source: float):
    """Compute NFW lensing terms from mass-model outputs."""
    rs_kpc, rho_s = MASS_MODELS.nfw_scale_parameters(
        mass_msun,
        concentration,
        z_lens,
        COSMOLOGY,
    )

    D_l_m = float(COSMOLOGY.angular_diameter_distance(z_lens).value) * CONSTANTS.MPC_TO_M
    D_s_m = float(COSMOLOGY.angular_diameter_distance(z_source).value) * CONSTANTS.MPC_TO_M
    D_ls_m = (
        float(COSMOLOGY.angular_diameter_distance_z1z2(z_lens, z_source).value)
        * CONSTANTS.MPC_TO_M
    )

    sigma_crit = (const.c.value**2 / (4.0 * np.pi * const.G.value)) * (D_s_m / (D_l_m * D_ls_m))
    rs_m = rs_kpc * CONSTANTS.KPC_TO_M
    kappa_s = (rho_s * rs_m) / sigma_crit
    scale_radius_arcsec = (rs_m / D_l_m) * CONSTANTS.ARCSEC_PER_RAD
    return {
        "rs_kpc": float(rs_kpc),
        "rho_s_kg_m3": float(rho_s),
        "kappa_s": float(kappa_s),
        "scale_radius_arcsec": float(scale_radius_arcsec),
    }


def test_pm_01_positive_einstein_radius():
    masses = [1.0e7, 1.0e8, 1.0e9]
    radii = [
        MASS_MODELS.einstein_radius_point_mass(mass, 0.2, 2.5, COSMOLOGY)
        for mass in masses
    ]
    assert all(radius > 0.0 for radius in radii)


def test_pm_02_mass_scaling_sqrt10():
    masses = [1.0e7, 1.0e8, 1.0e9]
    radii = [
        MASS_MODELS.einstein_radius_point_mass(mass, 0.2, 2.5, COSMOLOGY)
        for mass in masses
    ]
    expected = math.sqrt(10.0)
    assert radii[1] / radii[0] == pytest.approx(expected, rel=1.0e-6)
    assert radii[2] / radii[1] == pytest.approx(expected, rel=1.0e-6)


def test_pm_03_redshift_sensitivity_positive():
    radius_a = MASS_MODELS.einstein_radius_point_mass(1.0e8, 0.2, 2.5, COSMOLOGY)
    radius_b = MASS_MODELS.einstein_radius_point_mass(1.0e8, 0.5, 2.0, COSMOLOGY)
    assert radius_a > 0.0
    assert radius_b > 0.0
    assert radius_a != pytest.approx(radius_b, rel=0.0, abs=0.0)


def test_sis_01_positive_sigma_v_and_einstein_radius():
    masses = [1.0e7, 1.0e8, 1.0e9]
    for mass in masses:
        sigma_v = MASS_MODELS.sigma_v_from_m200_sis(mass, 0.2, COSMOLOGY)
        theta_e = MASS_MODELS.einstein_radius_sis_m200(mass, 0.2, 2.5, COSMOLOGY)
        assert sigma_v > 0.0
        assert theta_e > 0.0


def test_sis_02_mass_scaling_ten_power_two_thirds():
    masses = [1.0e7, 1.0e8, 1.0e9]
    radii = [
        MASS_MODELS.einstein_radius_sis_m200(mass, 0.2, 2.5, COSMOLOGY)
        for mass in masses
    ]
    expected = 10.0 ** (2.0 / 3.0)
    assert radii[1] / radii[0] == pytest.approx(expected, rel=1.0e-6)
    assert radii[2] / radii[1] == pytest.approx(expected, rel=1.0e-6)


def test_nfw_01_positive_physical_parameters_and_kappa_s():
    for mass in np.logspace(7, 10, 4):
        concentration = MASS_MODELS.concentration_mass_relation(
            mass,
            model="moline2017_eq7",
            x_sub=1.0,
            h=COSMOLOGY.reduced_h,
        )
        terms = _nfw_lensing_terms(
            mass_msun=float(mass),
            concentration=concentration,
            z_lens=0.2,
            z_source=2.5,
        )
        assert concentration > 0.0
        assert terms["rs_kpc"] > 0.0
        assert terms["rho_s_kg_m3"] > 0.0
        assert terms["kappa_s"] > 0.0
        assert terms["scale_radius_arcsec"] > 0.0


def test_nfw_02_scale_radius_monotonic_in_mass():
    masses = np.logspace(7, 10, 4)
    radii = []
    for mass in masses:
        concentration = MASS_MODELS.concentration_mass_relation(
            mass,
            model="moline2017_eq7",
            x_sub=1.0,
            h=COSMOLOGY.reduced_h,
        )
        terms = _nfw_lensing_terms(
            mass_msun=float(mass),
            concentration=concentration,
            z_lens=0.2,
            z_source=2.5,
        )
        radii.append(terms["rs_kpc"])
    assert all(later > earlier for earlier, later in zip(radii[:-1], radii[1:]))


def test_if_01_position_contract_is_canonical_yx():
    outward = LENSING_UTILS.get_einstein_ring_position(
        angle_deg=90.0,
        einstein_radius=1.0,
        offset_pixels=5.0,
        pixel_scale=0.1,
    )
    inward = LENSING_UTILS.get_einstein_ring_position(
        angle_deg=90.0,
        einstein_radius=1.0,
        offset_pixels=-5.0,
        pixel_scale=0.1,
    )
    assert outward[0] == pytest.approx(1.5, rel=0.0, abs=1.0e-12)
    assert outward[1] == pytest.approx(0.0, rel=0.0, abs=1.0e-12)
    assert inward[0] == pytest.approx(0.5, rel=0.0, abs=1.0e-12)
    assert inward[1] == pytest.approx(0.0, rel=0.0, abs=1.0e-12)


def test_reg_01_point_mass_anchor():
    anchors = load_lensing_anchor_fixture()
    inputs = anchors["inputs"]["point_mass"]
    expected = anchors["scalars"]["point_mass"]["theta_e_arcsec"]
    theta_e = MASS_MODELS.einstein_radius_point_mass(
        inputs["mass_msun"],
        inputs["z_lens"],
        inputs["z_source"],
        COSMOLOGY,
    )
    assert theta_e == pytest.approx(expected, rel=1.0e-10)


def test_reg_02_sis_anchor():
    anchors = load_lensing_anchor_fixture()
    inputs = anchors["inputs"]["sis"]
    expected = anchors["scalars"]["sis"]
    sigma_v = MASS_MODELS.sigma_v_from_m200_sis(
        inputs["mass_msun"],
        inputs["z_lens"],
        COSMOLOGY,
    )
    theta_e = MASS_MODELS.einstein_radius_sis_m200(
        inputs["mass_msun"],
        inputs["z_lens"],
        inputs["z_source"],
        COSMOLOGY,
    )
    assert sigma_v == pytest.approx(expected["sigma_v_km_s"], rel=1.0e-10)
    assert theta_e == pytest.approx(expected["theta_e_arcsec"], rel=1.0e-10)


def test_reg_03_nfw_anchor():
    anchors = load_lensing_anchor_fixture()
    inputs = anchors["inputs"]["nfw"]
    expected = anchors["scalars"]["nfw"]
    concentration = MASS_MODELS.concentration_mass_relation(
        inputs["mass_msun"],
        model=inputs["concentration_model"],
        x_sub=inputs["x_sub"],
        h=inputs["h"],
    )
    terms = _nfw_lensing_terms(
        mass_msun=inputs["mass_msun"],
        concentration=concentration,
        z_lens=inputs["z_lens"],
        z_source=inputs["z_source"],
    )
    assert concentration == pytest.approx(expected["c200"], rel=1.0e-10)
    assert terms["rs_kpc"] == pytest.approx(expected["rs_kpc"], rel=1.0e-10)
    assert terms["rho_s_kg_m3"] == pytest.approx(expected["rho_s_kg_m3"], rel=1.0e-10)
    assert terms["kappa_s"] == pytest.approx(expected["kappa_s"], rel=1.0e-10)
    assert terms["scale_radius_arcsec"] == pytest.approx(
        expected["scale_radius_arcsec"],
        rel=1.0e-10,
    )
