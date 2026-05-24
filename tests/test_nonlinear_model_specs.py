"""Tests for source-neutral nonlinear model specifications."""

from __future__ import annotations

import pytest

from hwoslaps.modeling.nonlinear.autolens_model_builder import (
    smooth_model_spec_from_config,
    subhalo_model_spec_from_trial,
)
from hwoslaps.modeling.nonlinear.trial import SubhaloTrial


def _config() -> dict:
    return {
        "lensing": {
            "lens_galaxy": {
                "redshift": 0.2,
                "mass": {
                    "type": "Isothermal",
                    "centre": [0.0, 0.0],
                    "einstein_radius": 1.0,
                    "ell_comps": [0.1, 0.0],
                },
            },
            "source_galaxy": {
                "redshift": 0.6,
                "light": {
                    "type": "Exponential",
                    "centre": [-0.03, 0.08],
                    "ell_comps": [0.1, 0.2],
                    "intensity": 2.0,
                    "effective_radius": 0.11,
                },
            },
        }
    }


def test_smooth_model_spec_uses_expected_profiles_and_priors():
    spec = smooth_model_spec_from_config(_config())

    assert spec.model_type == "smooth"
    assert spec.galaxies["lens"].components["mass"].class_name == "Isothermal"
    assert spec.galaxies["source"].components["light"].class_name == "Exponential"
    assert spec.galaxies["lens"].components["mass"].parameters["centre_0"].kind == "uniform"


def test_nfw_fixed_template_spec_preserves_forward_model_parameters():
    trial = SubhaloTrial(
        case_id="nfw",
        mass_msun=1.0e7,
        position_yx_arcsec=(0.2, -0.1),
        model="NFW",
        profile_class="NFWSph",
        lens_redshift=0.2,
        source_redshift=0.6,
        kappa_s=0.01,
        scale_radius_arcsec=0.2,
    )

    spec = subhalo_model_spec_from_trial(_config(), trial, fit_mode="fixed_template")
    mass = spec.galaxies["subhalo"].components["mass"]

    assert mass.class_name == "NFWSph"
    assert mass.parameters["centre_0"].kind == "fixed"
    assert mass.parameters["centre_0"].value == pytest.approx(0.2)
    assert mass.parameters["kappa_s"].value == pytest.approx(0.01)
    assert mass.parameters["scale_radius"].value == pytest.approx(0.2)


def test_local_search_frees_only_the_subhalo_center_by_default():
    trial = SubhaloTrial(
        case_id="sis",
        mass_msun=1.0e7,
        position_yx_arcsec=(0.2, -0.1),
        model="SIS",
        profile_class="IsothermalSph",
        lens_redshift=0.2,
        source_redshift=0.6,
        einstein_radius_arcsec=0.003,
    )

    spec = subhalo_model_spec_from_trial(_config(), trial, fit_mode="local_search")
    mass = spec.galaxies["subhalo"].components["mass"]

    assert mass.parameters["centre_0"].kind == "uniform"
    assert mass.parameters["centre_0"].lower == pytest.approx(0.17)
    assert mass.parameters["centre_0"].upper == pytest.approx(0.23)
    assert mass.parameters["einstein_radius"].kind == "fixed"


def test_nfw_spec_requires_hwo_slaps_scale_parameters():
    trial = SubhaloTrial(
        case_id="bad",
        mass_msun=1.0e7,
        position_yx_arcsec=(0.2, -0.1),
        model="NFW",
        profile_class="NFWSph",
        lens_redshift=0.2,
        source_redshift=0.6,
    )

    with pytest.raises(ValueError, match="kappa_s"):
        subhalo_model_spec_from_trial(_config(), trial)
