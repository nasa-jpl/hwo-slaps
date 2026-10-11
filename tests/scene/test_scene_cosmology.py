"""Chosen flat cosmologies: independent geometry, parameter identity and halo H0 scaling."""

from dataclasses import asdict

import numpy as np
import pytest

from hwoslaps.config.checks import ConfigError
from hwoslaps.scene.cosmology import Cosmology, parse_cosmology
from hwoslaps.scene.halos import FixedConcentration, HaloModel, TauTruncation, make_halo


@pytest.mark.backend
@pytest.mark.parametrize("redshift", [0.2, 0.5, 1.0, 2.5])
def test_custom_flat_lcdm_matches_astropy(redshift):
    from astropy import constants, units
    from astropy.cosmology import FlatLambdaCDM

    chosen = Cosmology(parse_cosmology({"flat_lcdm": {"H0": 70., "Om0": 0.3, "Ob0": 0.045}}))
    reference = FlatLambdaCDM(H0=70., Om0=0.3, Ob0=0.045, Tcmb0=0.)
    geometry = chosen.geometry(redshift, redshift + 0.5)
    distance = reference.angular_diameter_distance(redshift).to_value(units.Mpc)
    source = reference.angular_diameter_distance(redshift + 0.5).to_value(units.Mpc)
    between = reference.angular_diameter_distance_z1z2(redshift, redshift + 0.5).to_value(units.Mpc)
    np.testing.assert_allclose([geometry.d_deflector_mpc, geometry.d_source_mpc, geometry.d_deflector_source_mpc],
                               [distance, source, between], rtol=1e-12, atol=0.)
    assert geometry.rho_crit_kg_m3 == pytest.approx(reference.critical_density(redshift).to_value(units.kg / units.m**3),
                                                  rel=1e-12)
    expected_sigma = (constants.c**2 / (4 * np.pi * constants.G) *
                      (source / (distance * between) / units.Mpc)).to_value(units.kg / units.m**2)
    assert geometry.sigma_crit_kg_m2 == pytest.approx(expected_sigma, rel=1e-12)
    if redshift > 0.2:
        assert chosen.geometry(0.2, redshift).d_deflector_source_mpc == pytest.approx(
            reference.angular_diameter_distance_z1z2(0.2, redshift).to_value(units.Mpc), rel=1e-12)
    assert chosen.to_mapping()["rho_crit_convention"] == "matter_lambda"


@pytest.mark.backend
@pytest.mark.parametrize("name", ["Planck18", "WMAP9"])
def test_named_realizations_copy_astropy_parameters(name):
    from astropy.cosmology import realizations

    reference = getattr(realizations, name)
    chosen = Cosmology(parse_cosmology({"name": name}))
    parameters = asdict(chosen.parameters)
    assert parameters == {"H0": float(reference.H0.value), "Om0": float(reference.Om0),
                          "Ob0": float(reference.Ob0 or 0.), "Tcmb0": float(reference.Tcmb0.value),
                          "Neff": float(reference.Neff),
                          "m_nu_eV": tuple(float(value) for value in reference.m_nu.to_value("eV"))}
    record = chosen.to_mapping()
    expected_offset = (chosen.geometry(0.5, 1.).d_deflector_mpc /
                       reference.angular_diameter_distance(0.5).value - 1.)
    assert record["autogalaxy_vs_astropy_d_a_rel_z0p5"] == expected_offset
    assert abs(expected_offset) < 1e-5
    restored = Cosmology.from_mapping(record)
    assert restored == chosen and hash(restored) == hash(chosen)
    assert restored.geometry(0.5, 1.) == chosen.geometry(0.5, 1.)


@pytest.mark.backend
@pytest.mark.parametrize("kind, exponent", [("PointMass", 0.5), ("SIS", 2 / 3), ("NFW", 1 / 3), ("TNFW", 1 / 3)])
def test_halo_scales_follow_the_hubble_constant(kind, exponent):
    relation = FixedConcentration(12.) if kind in ("NFW", "TNFW") else None
    model = HaloModel(kind, relation, TauTruncation(10.) if kind == "TNFW" else None)
    scales = []
    for hubble in (60., 80.):
        cosmology = Cosmology(parse_cosmology({"flat_lcdm": {"H0": hubble, "Om0": 0.3}}))
        halo = make_halo(model, 1e8, (0.2, -0.3), redshift=0.2, source_redshift=0.6, cosmology=cosmology)
        scales.append(halo.lensing().parameters)
    for key in scales[0]:
        assert scales[1][key] / scales[0][key] == pytest.approx((80. / 60.)**exponent, rel=1e-12), key


@pytest.mark.parametrize("mapping, path", [
    ({}, "cosmology"),
    ({"name": "Planck15", "flat_lcdm": {"H0": 70., "Om0": 0.3}}, "cosmology"),
    ({"name": "not_a_realization"}, "cosmology.name"),
    ({"flat_lcdm": {"H0": 0., "Om0": 0.3}}, "cosmology.flat_lcdm.H0"),
    ({"flat_lcdm": {"H0": 70., "Om0": 1.}}, "cosmology.flat_lcdm.Om0"),
    ({"flat_lcdm": {"H0": 70., "Om0": 0.3, "Ob0": 0.4}}, "cosmology.flat_lcdm.Ob0"),
    ({"flat_lcdm": {"H0": 70., "Om0": 0.3, "m_nu_eV": [0., 0.]}}, "cosmology.flat_lcdm.m_nu_eV"),
])
def test_cosmology_inputs_refuse_invalid_names_and_physical_domains(mapping, path):
    with pytest.raises(ConfigError) as error:
        parse_cosmology(mapping)
    assert error.value.path == path
