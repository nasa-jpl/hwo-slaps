"""Halo physics: paper anchors, the 8fa6209 operation orders over a mass sweep, identities, records."""

import dataclasses
import json
import math
import pickle
from pathlib import Path

import numpy as np
import pytest

from hwoslaps.constants import ARCSEC_PER_RAD, C_M_S, G_SI, KM_TO_M, KPC_TO_M, MPC_TO_M, MSUN_KG
from hwoslaps.scene.cosmology import Cosmology, CosmologySpec, LensingGeometry
from hwoslaps.scene.halos import (FixedConcentration, Halo, HaloModel, Moline2017, PowerLawConcentration, concentration,
                                  halo_lensing, halo_lensing_traced, make_halo)

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "scene"
POINT_MASS = HaloModel("PointMass", None, None)
SIS = HaloModel("SIS", None, None)
NFW = HaloModel("NFW", Moline2017(x_sub=1.0, h=None), None)


def _astropy_planck15_geometry(z_deflector, z_source):
    """The geometry the 8fa6209 anchors were made with: astropy Planck15 distances and H(z)."""
    from astropy import units as u
    from astropy.cosmology import Planck15

    d_d = float(Planck15.angular_diameter_distance(z_deflector).to(u.Mpc).value)
    d_s = float(Planck15.angular_diameter_distance(z_source).to(u.Mpc).value)
    d_ds = float(Planck15.angular_diameter_distance_z1z2(z_deflector, z_source).to(u.Mpc).value)
    hubble = float(Planck15.H(z_deflector).value)
    rho_crit = 3 * (hubble * KM_TO_M / MPC_TO_M) ** 2 / (8 * np.pi * G_SI)
    sigma_crit = (C_M_S**2 / (4 * np.pi * G_SI)) * ((d_s * MPC_TO_M) / ((d_d * MPC_TO_M) * (d_ds * MPC_TO_M)))
    return LensingGeometry(z_deflector, z_source, d_d, d_s, d_ds, hubble, rho_crit, sigma_crit)


def test_halo_scales_reproduce_physics_anchors():
    anchors = json.loads((FIXTURES / "halo_anchors.json").read_text())
    inputs, scalars = anchors["inputs"], anchors["scalars"]
    rows = [(POINT_MASS, "point_mass", {"theta_e_arcsec": ("parameters", "einstein_radius")}),
            (SIS, "sis", {"theta_e_arcsec": ("parameters", "einstein_radius"),
                          "sigma_v_km_s": ("derived", "velocity_dispersion_km_s")}),
            (NFW, "nfw", {"c200": ("derived", "concentration"), "rs_kpc": ("derived", "scale_radius_kpc"),
                          "rho_s_kg_m3": ("derived", "rho_s_kg_m3"), "kappa_s": ("parameters", "kappa_s"),
                          "scale_radius_arcsec": ("parameters", "scale_radius")})]
    for model, key, fields in rows:
        geometry = _astropy_planck15_geometry(inputs[key]["z_lens"], inputs[key]["z_source"])
        lensing = halo_lensing(model, inputs[key]["mass_msun"], geometry, reduced_h=inputs["nfw"]["h"])
        for anchor, (group, name) in fields.items():
            assert getattr(lensing, group)[name] == pytest.approx(scalars[key][anchor], rel=1.0e-14), (key, anchor)


@pytest.mark.backend
def test_halo_scales_match_the_base_tree_over_a_mass_sweep(planck15):
    sweep = json.loads((FIXTURES / "halo_scales_8fa6209.json").read_text())
    reduced_h = float.fromhex(sweep["reduced_h"])
    assert planck15.reduced_h == reduced_h
    masses = [float.fromhex(value) for value in sweep["masses_msun"]]
    for pair in sweep["pairs"]:
        geometry = planck15.geometry(pair["z_deflector"], pair["z_source"])
        for field, value in pair["geometry"].items():
            assert getattr(geometry, field) == float.fromhex(value), field
        scalar = {key: [float.fromhex(v) for v in values] for key, values in pair["scalar"].items()}
        traced = {key: [float.fromhex(v) for v in values] for key, values in pair["traced"].items()}
        concrete = {"point_mass": [halo_lensing(POINT_MASS, m, geometry, reduced_h=reduced_h) for m in masses],
                    "sis": [halo_lensing(SIS, m, geometry, reduced_h=reduced_h) for m in masses],
                    "nfw": [halo_lensing(NFW, m, geometry, reduced_h=reduced_h) for m in masses]}
        assert [h.parameters["einstein_radius"] for h in concrete["point_mass"]] == scalar["point_mass_einstein_radius"]
        assert [h.parameters["einstein_radius"] for h in concrete["sis"]] == scalar["sis_einstein_radius"]
        assert [h.derived["velocity_dispersion_km_s"] for h in concrete["sis"]] == scalar["sis_velocity_dispersion_km_s"]
        assert [h.derived["concentration"] for h in concrete["nfw"]] == scalar["nfw_concentration"]
        assert [h.derived["scale_radius_kpc"] for h in concrete["nfw"]] == scalar["nfw_scale_radius_kpc"]
        assert [h.derived["rho_s_kg_m3"] for h in concrete["nfw"]] == scalar["nfw_rho_s_kg_m3"]
        assert [h.parameters["kappa_s"] for h in concrete["nfw"]] == scalar["nfw_kappa_s"]
        assert [h.parameters["scale_radius"] for h in concrete["nfw"]] == scalar["nfw_scale_radius"]

        def twin(model, name):
            return [float(halo_lensing_traced(model, m, geometry, reduced_h=reduced_h, xp=np)[name]) for m in masses]

        assert twin(POINT_MASS, "einstein_radius") == traced["point_mass_einstein_radius"]
        assert twin(SIS, "einstein_radius") == traced["sis_einstein_radius"]
        assert twin(NFW, "kappa_s") == traced["nfw_kappa_s"]
        assert twin(NFW, "scale_radius") == traced["nfw_scale_radius"]
        assert [float(concentration(NFW.concentration, m, pair["z_deflector"], reduced_h)) for m in masses] == \
            traced["nfw_concentration"]


@pytest.mark.backend
def test_the_two_operation_orders_agree_to_one_part_in_1e15(planck15):
    masses = np.logspace(6.0, 11.0, 401)
    worst = 0.0
    for z_deflector, z_source in ((0.2, 0.6), (0.2, 2.5), (0.5, 1.0)):
        geometry = planck15.geometry(z_deflector, z_source)
        for model in (POINT_MASS, SIS, NFW):
            for mass in masses:
                concrete = halo_lensing(model, float(mass), geometry, reduced_h=planck15.reduced_h).parameters
                traced = halo_lensing_traced(model, mass, geometry, reduced_h=planck15.reduced_h, xp=np)
                assert concrete.keys() == traced.keys()
                for name, value in concrete.items():
                    worst = max(worst, abs(float(traced[name]) / value - 1.0))
    assert worst <= 1.0e-15


def test_halo_scales_satisfy_the_lensing_mass_identities():
    geometry = LensingGeometry(z_deflector=0.3, z_source=1.4, d_deflector_mpc=950.0, d_source_mpc=1750.0,
                               d_deflector_source_mpc=1100.0, hubble_km_s_mpc=80.0, rho_crit_kg_m3=1.2e-26,
                               sigma_crit_kg_m2=2.9)
    mass = 3.0e8
    nfw = halo_lensing(NFW, mass, geometry, reduced_h=0.7)
    c = nfw.derived["concentration"]
    theta_s_rad = nfw.parameters["scale_radius"] / ARCSEC_PER_RAD
    enclosed_kg = (4 * math.pi * nfw.parameters["kappa_s"] * geometry.sigma_crit_kg_m2
                   * (theta_s_rad * geometry.d_deflector_mpc * MPC_TO_M) ** 2 * (math.log(1 + c) - c / (1 + c)))
    assert enclosed_kg == pytest.approx(mass * MSUN_KG, rel=1.0e-12)

    r200_m = (3 * mass * MSUN_KG / (4 * math.pi * 200 * geometry.rho_crit_kg_m3)) ** (1.0 / 3.0)
    assert nfw.derived["r200_kpc"] == pytest.approx(r200_m / KPC_TO_M, rel=1.0e-12)
    assert nfw.derived["scale_radius_kpc"] == pytest.approx(r200_m / c / KPC_TO_M, rel=1.0e-12)
    theta_point_rad = halo_lensing(POINT_MASS, mass, geometry, reduced_h=0.7).parameters["einstein_radius"] / ARCSEC_PER_RAD
    theta_sis_rad = halo_lensing(SIS, mass, geometry, reduced_h=0.7).parameters["einstein_radius"] / ARCSEC_PER_RAD
    assert theta_sis_rad == pytest.approx(theta_point_rad**2 * math.pi * geometry.d_deflector_mpc * MPC_TO_M
                                          / (2 * r200_m), rel=1.0e-12)
    assert concentration(Moline2017(x_sub=1.0, h=0.7), 1.0e8 / 0.7, 0.3, 0.6774) == pytest.approx(19.9, rel=1.0e-13)


@pytest.mark.parametrize("relation, mass, redshift, expected", [
    (Moline2017(1.0, 0.6774), 1.0e6, 0.2, 28.915897079765447),
    (Moline2017(1.0, 0.6774), 1.0e8, 0.2, 20.560847592361515),
    (Moline2017(1.0, 0.6774), 1.0e9, 0.2, 16.792762422153395),
    (Moline2017(0.5, 0.6774), 5.0e9, 0.2, 16.720675479468554),
    (Moline2017(0.3, 0.70), 1.0e10, 0.2, 17.138469176898678),
    (Moline2017(1.0, None), 1.0e12, 0.2, 8.136344788704443),
    (PowerLawConcentration(5.71, 2.0e12 / 0.6774, -0.084, -0.47), 1.0e9, 0.5,
     5.71 * (1.0e9 / (2.0e12 / 0.6774)) ** -0.084 * 1.5 ** -0.47),
    (FixedConcentration(12.5), 1.0e7, 0.9, 12.5),
], ids=["moline-1e6", "moline-1e8", "moline-1e9", "moline-x0.5", "moline-h0.70", "moline-h-from-cosmology",
        "duffy-2008-power-law", "fixed"])
def test_concentration_relations_follow_their_published_forms(relation, mass, redshift, expected):
    assert concentration(relation, mass, redshift, 0.6774) == pytest.approx(expected, rel=1.0e-12)


def test_halos_outside_their_domains_are_refused(planck15):
    def halo(model=NFW, mass=1.0e8, position=(0.1, 0.2), redshift=0.2, source_redshift=0.6):
        return make_halo(model, mass, position, redshift=redshift, source_redshift=source_redshift, cosmology=planck15)

    for mass in (1.0e6, 1.0e12):
        assert halo(mass=mass).mass_msun == mass
    rows = [dict(mass=9.99e5), dict(mass=1.01e12), dict(model=POINT_MASS, mass=0.0), dict(model=SIS, mass=math.nan),
            dict(position=(0.1, math.inf)), dict(position=(0.1,)), dict(redshift=0.6), dict(redshift=0.0)]
    for row in rows:
        with pytest.raises(ValueError):
            halo(**row)
    with pytest.raises(ValueError, match="calibrated for M200"):
        halo_lensing(NFW, 5.0e5, _astropy_planck15_geometry(0.2, 0.6), reduced_h=0.6774)
    with pytest.raises(ValueError):
        HaloModel("NFW", None, None)
    with pytest.raises(ValueError):
        HaloModel("SIS", FixedConcentration(10.0), None)
    with pytest.raises(ValueError):
        Moline2017(x_sub=1.6, h=None)
    for relation in (lambda: FixedConcentration(0.0), lambda: PowerLawConcentration(5.0, 1.0e12, math.nan, 0.0)):
        with pytest.raises(ValueError):
            relation()


def test_cosmology_is_an_immutable_value_recorded_with_its_convention(planck15):
    from astropy.cosmology import Planck15

    record = planck15.to_mapping()
    assert record == {"name": "Planck15",
                      "parameters": {"H0": 67.74, "Om0": 0.3075, "Ob0": 0.0486, "Tcmb0": 2.7255, "Neff": 3.046,
                                     "m_nu_eV": [0.0, 0.0, 0.06]},
                      "rho_crit_convention": "matter_lambda", "autogalaxy_vs_astropy_d_a_rel_z0p5": None}
    assert record["parameters"]["H0"] == Planck15.H0.value and planck15.reduced_h == 0.6774
    assert Cosmology.from_mapping(json.loads(json.dumps(record))) == planck15
    assert hash(Cosmology(CosmologySpec("Planck15", None))) == hash(planck15)
    with pytest.raises(ValueError, match="differs"):
        Cosmology.from_mapping(dict(record, rho_crit_convention="full"))
    with pytest.raises(AttributeError):
        planck15.spec = CosmologySpec("Planck15", None)
    with pytest.raises(ValueError, match="0 < z_deflector < z_source"):
        planck15.geometry(0.6, 0.2)


@pytest.mark.backend
@pytest.mark.parametrize("model", [POINT_MASS, SIS, NFW, HaloModel("NFW", PowerLawConcentration(5.71, 3.0e12, -0.084, -0.47), None),
                                   HaloModel("NFW", FixedConcentration(15.0), None)],
                         ids=["point-mass", "sis", "nfw-moline", "nfw-power-law", "nfw-fixed"])
def test_halo_records_round_trip_and_pickle(model, planck15):
    halo = make_halo(model, 2.0e8, (0.4, -0.8), redshift=0.2, source_redshift=0.6, cosmology=planck15)
    record = json.loads(json.dumps(halo.to_mapping()))
    assert Halo.from_mapping(record) == halo
    assert record["lensing"]["parameters"] == dict(halo.lensing().parameters)
    restored = pickle.loads(pickle.dumps(halo))
    assert restored == halo and restored.lensing() == halo.lensing() and restored.geometry == halo.geometry
    heavier = dataclasses.replace(halo, mass_msun=4.0e8)
    assert heavier.lensing() == make_halo(model, 4.0e8, (0.4, -0.8), redshift=0.2, source_redshift=0.6,
                                          cosmology=planck15).lensing()
    tampered = json.loads(json.dumps(record))
    tampered["lensing"]["parameters"] = {name: value * 1.01 for name, value in tampered["lensing"]["parameters"].items()}
    with pytest.raises(ValueError, match="lensing"):
        Halo.from_mapping(tampered)


@pytest.mark.backend
def test_halo_profiles_are_the_autolens_classes_of_their_types(planck15):
    import autolens as al

    for model, profile_class in ((POINT_MASS, al.mp.PointMass), (SIS, al.mp.IsothermalSph), (NFW, al.mp.NFWSph)):
        halo = make_halo(model, 1.0e9, (0.4, -0.8), redshift=0.2, source_redshift=0.6, cosmology=planck15)
        profile = halo.autolens_profile()
        assert type(profile) is profile_class and profile.centre == (0.4, -0.8)
        for name, value in halo.lensing().parameters.items():
            assert getattr(profile, name) == value
        assert halo.autolens_profile(centre=(0.0, 0.0)).centre == (0.0, 0.0)
