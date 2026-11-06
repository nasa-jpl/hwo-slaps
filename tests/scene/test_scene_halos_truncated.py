"""TNFW parent-mass convention, BMO far field and independent enclosed-density equations."""

import math
import json
from pathlib import Path

import numpy as np
import pytest

from hwoslaps.constants import ARCSEC_PER_RAD, MPC_TO_M, MSUN_KG
from hwoslaps.scene.cosmology import LensingGeometry
from hwoslaps.scene.halos import (FixedConcentration, Halo, HaloModel, Moline2017, OverdensityTruncation,
                                  TauTruncation, halo_lensing)

GEOMETRY = LensingGeometry(0.3, 1.4, 950., 1750., 1100., 80., 1.2e-26, 2.9)


@pytest.mark.backend
@pytest.mark.parametrize("tau", [3., 10., 40., 1e3])
def test_truncated_nfw_total_mass_and_nfw_limit(tau, planck15):
    import autolens as al

    model = HaloModel("TNFW", Moline2017(1., None), TauTruncation(tau))
    halo = Halo(model, 1e8, (0., 0.), 0.2, 0.6, planck15)
    lensing = halo.lensing()
    theta = lensing.parameters["scale_radius"]
    kappa = lensing.parameters["kappa_s"]
    fraction = tau**2 / (tau**2 + 1.)**2 * ((tau**2 - 1.) * math.log(tau) + tau * math.pi - (tau**2 + 1.))
    # The originalAPI scalar fixture supplies parent scales independently of the new TNFW branch.
    fixture = Path(__file__).resolve().parents[1] / "fixtures/scene/halo_scales_8fa6209.json"
    sweep = json.loads(fixture.read_text())
    index = [float.fromhex(value) for value in sweep["masses_msun"]].index(1e8)
    base = next(pair for pair in sweep["pairs"] if pair["z_deflector"] == 0.2 and pair["z_source"] == 0.6)["scalar"]
    assert theta == float.fromhex(base["nfw_scale_radius"][index])
    assert kappa == float.fromhex(base["nfw_kappa_s"][index])
    concentration = float.fromhex(base["nfw_concentration"][index])
    assert lensing.derived["concentration"] == concentration
    expected_mass = 1e8 * fraction / (math.log(1 + concentration) - concentration / (1 + concentration))
    assert model.mass_definition == "M200c_parent"
    assert lensing.derived["total_mass_msun"] == pytest.approx(expected_mass, rel=1e-14)
    assert lensing.parameters["truncation_radius"] == tau * theta
    profile = halo.autolens_profile()
    if tau < 1e3:
        radius = 1e4 * theta
        alpha = np.asarray(profile.deflections_yx_2d_from(grid=al.Grid2DIrregular(values=[[radius, 0.]])))[0, 0]
        assert alpha * radius == pytest.approx(4 * kappa * theta**2 * fraction, rel=1e-5)
    else:
        radii = np.array([0.2, 0.5, 0.8])
        f = 2 * np.arctanh(np.sqrt((1 - radii) / (1 + radii))) / np.sqrt(1 - radii**2)
        expected = 4 * kappa * theta * (np.log(radii / 2) + f) / radii
        grid = al.Grid2DIrregular(values=np.column_stack((radii * theta, np.zeros(3))))
        np.testing.assert_allclose(np.asarray(profile.deflections_yx_2d_from(grid=grid))[:, 0], expected,
                                   rtol=1e-5, atol=0.)
    record = halo.to_mapping()
    assert record["mass_msun"] == 1e8 and record["mass_definition"] == "M200c_parent"
    assert Halo.from_mapping(record) == halo


@pytest.mark.parametrize("overdensity", [50., 100., 200.])
@pytest.mark.parametrize("concentration", [5., 20.])
def test_overdensity_truncation_encloses_the_requested_density(overdensity, concentration):
    model = HaloModel("TNFW", FixedConcentration(concentration), OverdensityTruncation(overdensity))
    scales = halo_lensing(model, 1e8, GEOMETRY, reduced_h=0.7)
    tau = scales.parameters["truncation_radius"] / scales.parameters["scale_radius"]
    f = lambda value: math.log(1 + value) - value / (1 + value)
    enclosed_kg = 1e8 * MSUN_KG * f(tau) / f(concentration)
    scale_m = scales.parameters["scale_radius"] / ARCSEC_PER_RAD * GEOMETRY.d_deflector_mpc * MPC_TO_M
    density = enclosed_kg / (4 / 3 * math.pi * (tau * scale_m)**3)
    assert density == pytest.approx(overdensity * GEOMETRY.rho_crit_kg_m3, rel=1e-12)
    if overdensity == 200.:
        assert tau == pytest.approx(concentration, rel=1e-14)
