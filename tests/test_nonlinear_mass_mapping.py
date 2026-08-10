"""Tests for freed-fit mass conversion tables and adapter profiles."""

from __future__ import annotations

from copy import deepcopy
import multiprocessing
import pickle

import autofit as af
import autolens as al
import numpy as np
import pytest
import yaml
from astropy import constants as const

from hwoslaps.constants import ARCSEC_PER_RAD, KPC_TO_M, MPC_TO_M
from hwoslaps.lensing.generator import generate_lensing_system
from hwoslaps.lensing.mass_models import (
    angular_diameter_distance_mpc,
    angular_diameter_distance_z1z2_mpc,
    concentration_mass_relation,
    einstein_radius_point_mass,
    einstein_radius_sis_m200,
    nfw_lensing_parameters,
    nfw_scale_parameters,
)
from hwoslaps.modeling.nonlinear.mass_mapping import (
    NFWMCRSubhaloSph,
    PointMassMCRSubhalo,
    SISMCRSubhalo,
    build_mass_mapping_context,
    build_mass_mapping_context_explicit,
    evaluate_mass_mapping,
)


def _scene_config(name="scene1_smooth_ring.yaml"):
    """Load one canonical scene configuration."""
    with open(f"configs/scenes/{name}", encoding="utf-8") as stream:
        return yaml.safe_load(stream)


def _old_nfw_formula(mass, concentration, z_lens, z_source, cosmology):
    """Reproduce the pre-refactor generator NFW conversion exactly."""
    rs_kpc, rho_s = nfw_scale_parameters(
        mass,
        concentration,
        z_lens,
        cosmology,
    )
    D_l_m = angular_diameter_distance_mpc(cosmology, z_lens) * MPC_TO_M
    D_s_m = angular_diameter_distance_mpc(cosmology, z_source) * MPC_TO_M
    D_ls_m = (
        angular_diameter_distance_z1z2_mpc(cosmology, z_lens, z_source)
        * MPC_TO_M
    )
    c_SI = float(const.c.value)
    G_SI = float(const.G.value)
    Sigma_crit = (c_SI**2 / (4 * np.pi * G_SI)) * (
        D_s_m / (D_l_m * D_ls_m)
    )
    rs_m = rs_kpc * KPC_TO_M
    kappa_s = (rho_s * rs_m) / Sigma_crit
    scale_radius = (rs_m / D_l_m) * ARCSEC_PER_RAD
    return kappa_s, scale_radius


@pytest.mark.parametrize("mass", [1.0e6, 10.0**6.75, 1.0e7, 10.0**7.5, 1.0e8])
@pytest.mark.parametrize("relation", ["moline2017_eq7", "power_law"])
def test_nfw_lensing_parameters_reproduce_generator_formula(mass, relation):
    """Match the exact pre-refactor NFW arithmetic for both relations."""
    cosmology = al.cosmo.Planck15()
    if relation == "moline2017_eq7":
        concentration = concentration_mass_relation(
            mass,
            model=relation,
            x_sub=1.0,
            h=0.6774,
        )
    else:
        concentration = concentration_mass_relation(
            mass,
            model=relation,
            z=0.2,
        )
    expected = _old_nfw_formula(mass, concentration, 0.2, 0.6, cosmology)
    actual = nfw_lensing_parameters(
        mass,
        concentration,
        0.2,
        0.6,
        cosmology,
    )
    assert actual == pytest.approx(expected, rel=1.0e-15, abs=0.0)


def test_generator_nfw_refactor_preserves_direct_formula():
    """Keep generated NFW profile scales identical to the old formula."""
    config = deepcopy(_scene_config())
    config["lensing"]["grid"]["shape"] = [21, 21]
    data = generate_lensing_system(config["lensing"], full_config=config)
    cosmology = al.cosmo.Planck15()
    concentration = concentration_mass_relation(
        data.subhalo_mass,
        model="moline2017_eq7",
        x_sub=1.0,
        h=0.6774,
    )
    expected = _old_nfw_formula(
        data.subhalo_mass,
        concentration,
        data.lens_redshift,
        data.source_redshift,
        cosmology,
    )
    assert data.subhalo_kappa_s == expected[0]
    assert data.subhalo_scale_radius_arcsec == expected[1]


def _explicit_context(model, relation="moline2017_eq7", h=0.6774):
    """Build a canonical explicit mass context."""
    return build_mass_mapping_context_explicit(
        subhalo_model=model,
        concentration_model=relation if model == "NFW" else None,
        x_sub=1.0 if model == "NFW" and relation.startswith("moline") else None,
        h=h if model == "NFW" and relation.startswith("moline") else None,
        z_lens=0.2,
        z_source=0.6,
        cosmology_name="Planck15",
    )


@pytest.mark.parametrize("relation", ["moline2017_eq7", "power_law"])
@pytest.mark.parametrize("model", ["NFW", "SIS", "PointMass"])
def test_closed_form_mapping_meets_direct_accuracy_contract(model, relation):
    """Keep closed-form profile scales within 1e-11 of legacy calculations."""
    context = _explicit_context(model, relation)
    cosmology = al.cosmo.Planck15()
    random = np.random.default_rng(17)
    for log_mass in random.uniform(6.0, 8.5, 512):
        mapped = evaluate_mass_mapping(context, log_mass)
        mass = 10.0**log_mass
        if model == "NFW":
            if relation == "moline2017_eq7":
                concentration = concentration_mass_relation(
                    mass,
                    model=relation,
                    x_sub=1.0,
                    h=0.6774,
                )
            else:
                concentration = concentration_mass_relation(
                    mass,
                    model=relation,
                    z=0.2,
                )
            direct = nfw_lensing_parameters(
                mass,
                concentration,
                0.2,
                0.6,
                cosmology,
            )
            assert mapped["kappa_s"] == pytest.approx(
                direct[0], rel=1.0e-11, abs=0.0
            )
            assert mapped["scale_radius_arcsec"] == pytest.approx(
                direct[1], rel=1.0e-11, abs=0.0
            )
        elif model == "SIS":
            direct = einstein_radius_sis_m200(
                mass, 0.2, 0.6, cosmology
            )
            assert mapped["einstein_radius_arcsec"] == pytest.approx(
                direct, rel=1.0e-11, abs=0.0
            )
        else:
            direct = einstein_radius_point_mass(
                mass, 0.2, 0.6, cosmology
            )
            assert mapped["einstein_radius_arcsec"] == pytest.approx(
                direct, rel=1.0e-11, abs=0.0
            )


def test_mapping_range_endpoints_and_extrapolation_guards():
    """Evaluate closed endpoints and reject either extrapolation side."""
    context = _explicit_context("NFW")
    evaluate_mass_mapping(context, context.log10_m200_lower)
    evaluate_mass_mapping(context, context.log10_m200_upper)
    with pytest.raises(ValueError, match="outside"):
        evaluate_mass_mapping(context, context.log10_m200_lower - 1.0e-12)
    with pytest.raises(ValueError, match="outside"):
        evaluate_mass_mapping(context, context.log10_m200_upper + 1.0e-12)
    with pytest.raises(ValueError, match="outside"):
        evaluate_mass_mapping(context, np.nan)
    with pytest.raises(ValueError):
        build_mass_mapping_context_explicit(
            subhalo_model="NFW",
            concentration_model="moline2017_eq7",
            x_sub=1.0,
            h=0.6774,
            z_lens=0.2,
            z_source=0.6,
            cosmology_name="Planck15",
            log10_m200_range=(5.9, 8.5),
        )


@pytest.mark.parametrize("subhalo_model", ["SIS", "PointMass"])
def test_build_mass_mapping_context_rejects_concentration_block_for_non_nfw(
    subhalo_model: str,
) -> None:
    """Reject concentration settings for non-NFW context construction."""
    config = deepcopy(_scene_config())
    config["lensing"]["subhalo"]["model"] = subhalo_model
    with pytest.raises(
        ValueError, match="SIS and PointMass contexts do not accept concentration"
    ):
        build_mass_mapping_context(config)


def test_build_mass_mapping_context_rejects_power_law_x_sub_or_h():
    """Reject x_sub/h for power-law concentration in NFW context."""
    config = deepcopy(_scene_config())
    config["lensing"]["subhalo"]["concentration"] = {
        "model": "power_law",
        "x_sub": 1.0,
        "h": 0.6774,
    }
    with pytest.raises(ValueError, match="power_law contexts do not accept x_sub or h"):
        build_mass_mapping_context(config)


def test_context_h_resolution_and_explicit_hash_equivalence():
    """Resolve null h and match config and explicit context identities."""
    config = _scene_config()
    config_context = build_mass_mapping_context(config)
    inferred = _explicit_context("NFW", h=None)
    explicit = _explicit_context("NFW", h=0.6774)
    alternate = _explicit_context("NFW", h=0.7)
    assert inferred.h == pytest.approx(0.6774)
    assert explicit.h == pytest.approx(0.6774)
    assert alternate.h == pytest.approx(0.7)
    inferred_kappa_s = evaluate_mass_mapping(inferred, 7.0)["kappa_s"]
    alternate_kappa_s = evaluate_mass_mapping(alternate, 7.0)["kappa_s"]
    relative_difference = abs(alternate_kappa_s / inferred_kappa_s - 1.0)
    assert relative_difference > 1.0e-6
    assert config_context.context_hash == inferred.context_hash

    disabled = deepcopy(config)
    disabled["lensing"]["subhalo"]["enabled"] = False
    with pytest.raises(ValueError, match="explicit"):
        build_mass_mapping_context(disabled)


def _spawn_mapping_worker(context, queue):
    """Evaluate a pickled mapping context in a spawn worker."""
    value = evaluate_mass_mapping(context, 7.0)
    queue.put((NFWMCRSubhaloSph.__module__, value["kappa_s"]))


def test_mass_adapter_pickle_and_spawn_round_trips():
    """Round-trip adapter models, instances, and spawn reconstruction."""
    context = _explicit_context("NFW")
    model = af.Model(NFWMCRSubhaloSph, mapping_context=context)
    model.log10_m200 = af.UniformPrior(lower_limit=6.0, upper_limit=8.5)
    instance = NFWMCRSubhaloSph(
        centre=(0.1, -0.2),
        log10_m200=7.0,
        mapping_context=context,
    )
    assert pickle.loads(pickle.dumps(model)).cls is NFWMCRSubhaloSph
    restored = pickle.loads(pickle.dumps(instance))
    assert restored.kappa_s == pytest.approx(instance.kappa_s)

    spawn = multiprocessing.get_context("spawn")
    queue = spawn.Queue()
    process = spawn.Process(target=_spawn_mapping_worker, args=(context, queue))
    process.start()
    module_name, kappa_s = queue.get(timeout=30)
    process.join(timeout=30)
    assert process.exitcode == 0
    assert module_name.endswith("mass_mapping")
    assert kappa_s == pytest.approx(instance.kappa_s)


@pytest.mark.parametrize(
    "model,adapter",
    [
        ("NFW", NFWMCRSubhaloSph),
        ("SIS", SISMCRSubhalo),
        ("PointMass", PointMassMCRSubhalo),
    ],
)
def test_mass_adapters_match_direct_profile_parameters(model, adapter):
    """Derive the direct profile scales and reject wrong-model contexts."""
    context = _explicit_context(model)
    profile = adapter(log10_m200=7.0, mapping_context=context)
    direct = evaluate_mass_mapping(context, 7.0)
    if model == "NFW":
        assert profile.kappa_s == pytest.approx(
            direct["kappa_s"], rel=1.0e-10, abs=0.0
        )
        assert profile.scale_radius == pytest.approx(
            direct["scale_radius_arcsec"], rel=1.0e-10, abs=0.0
        )
    else:
        assert profile.einstein_radius == pytest.approx(
            direct["einstein_radius_arcsec"], rel=1.0e-10, abs=0.0
        )
    wrong = _explicit_context("SIS" if model != "SIS" else "PointMass")
    with pytest.raises(ValueError, match="requires"):
        adapter(log10_m200=7.0, mapping_context=wrong)
