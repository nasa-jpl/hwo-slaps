"""Tests for flexible macro-lens validation and generation."""

from __future__ import annotations

import copy
from pathlib import Path

import autogalaxy as ag
import autolens as al
import numpy as np
import pytest
import yaml

from hwoslaps.config.validation import validate_or_raise
from hwoslaps.lensing.generator import (
    _create_lens_galaxy,
    generate_lensing_system,
)
from hwoslaps.lensing.utils import print_lensing_data_summary
from hwoslaps.plotting.lensing_plots import (
    _no_subhalo_tracer,
    plot_lensing_baseline_scene,
    plot_lensing_comparison,
    plot_lensing_fractional_comparison,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _config() -> dict:
    """Return a small complete configuration with Item 5 Fisher steps."""
    with (PROJECT_ROOT / "configs" / "master_config.yaml").open(
        "r", encoding="utf-8"
    ) as stream:
        config = yaml.safe_load(stream)
    config["plotting"]["enabled"] = False
    config["lensing"]["grid"] = {"shape": [21, 21], "pixel_scale": 0.08}
    config["lensing"]["subhalo"]["enabled"] = False
    config["modeling"]["fisher"]["finite_diff"].update(
        {
            "slope": 1.0e-3,
            "multipole_comp": 1.0e-3,
            "shear_comp": 1.0e-3,
        }
    )
    return config


def _power_law_mass() -> dict:
    """Return a representative flexible PowerLaw mass block."""
    return {
        "type": "PowerLaw",
        "centre": [0.0, 0.0],
        "ell_comps": [0.1, 0.0],
        "einstein_radius": 1.0,
        "slope": 2.0,
        "multipoles": {"m3": [0.0, 0.01], "m4": [0.02, 0.0]},
    }


def _generated(config: dict):
    """Generate one scene from a complete configuration."""
    return generate_lensing_system(config["lensing"], full_config=config)


@pytest.mark.parametrize("unsupported", ["slope", "multipoles", "extra"])
def test_isothermal_rejects_unsupported_mass_keys(unsupported):
    """Reject slope, multipoles, and unknown keys for Isothermal mass."""
    config = _config()
    values = {"slope": 2.0, "multipoles": {"m4": [0.0, 0.01]}, "extra": 1}
    config["lensing"]["lens_galaxy"]["mass"][unsupported] = values[unsupported]

    with pytest.raises(ValueError, match="unsupported keys"):
        validate_or_raise(config)


def test_power_law_requires_slope_and_rejects_unknown_keys():
    """Require slope and reject keys outside the PowerLaw schema."""
    config = _config()
    mass = _power_law_mass()
    mass.pop("slope")
    config["lensing"]["lens_galaxy"]["mass"] = mass
    with pytest.raises(ValueError, match="slope"):
        validate_or_raise(config)

    mass["slope"] = 2.0
    mass["extra"] = 1
    with pytest.raises(ValueError, match="unsupported keys"):
        validate_or_raise(config)


@pytest.mark.parametrize(
    "slope, accepted",
    [(1.0, False), (3.0, False), (1.01, True), (2.99, True)],
)
def test_power_law_slope_domain(slope, accepted):
    """Accept only finite PowerLaw slopes strictly between one and three."""
    config = _config()
    mass = _power_law_mass()
    mass["slope"] = slope
    config["lensing"]["lens_galaxy"]["mass"] = mass

    if accepted:
        validate_or_raise(config)
    else:
        with pytest.raises(ValueError, match="slope"):
            validate_or_raise(config)


@pytest.mark.parametrize(
    "multipoles, accepted",
    [
        ({}, False),
        ({"m2": [0.0, 0.01]}, False),
        ({"m3": [0.01]}, False),
        ({"m4": [0.0, 0.0]}, True),
    ],
)
def test_power_law_multipole_validation(multipoles, accepted):
    """Validate non-empty m3/m4 finite pairs while accepting a zero pair."""
    config = _config()
    mass = _power_law_mass()
    mass["multipoles"] = multipoles
    config["lensing"]["lens_galaxy"]["mass"] = mass

    if accepted:
        validate_or_raise(config)
    else:
        with pytest.raises(ValueError, match="multipoles"):
            validate_or_raise(config)


@pytest.mark.parametrize("shear", [[0.02, -0.01], None])
def test_shear_is_optional_for_isothermal(shear):
    """Allow an optional finite external-shear pair with Isothermal mass."""
    config = _config()
    if shear is not None:
        config["lensing"]["lens_galaxy"]["shear"] = shear
    validate_or_raise(config)


@pytest.mark.parametrize("shear", [[0.1], [0.0, float("nan")], "bad"])
def test_shear_requires_finite_pair(shear):
    """Reject malformed or non-finite external shear values."""
    config = _config()
    config["lensing"]["lens_galaxy"]["shear"] = shear
    with pytest.raises(ValueError, match="shear"):
        validate_or_raise(config)


@pytest.mark.parametrize("unknown_key", ["sheer", "light"])
def test_truth_lens_galaxy_rejects_unknown_keys(unknown_key):
    """Reject misspelled or unsupported truth-side galaxy keys."""
    config = _config()
    config["lensing"]["lens_galaxy"][unknown_key] = [0.02, -0.01]

    with pytest.raises(ValueError) as error:
        validate_or_raise(config)

    assert "lensing.lens_galaxy" in str(error.value)
    assert unknown_key in str(error.value)


def test_truth_lens_galaxy_accepts_all_legal_keys():
    """Accept redshift, flexible mass, and shear as the complete schema."""
    config = _config()
    config["lensing"]["lens_galaxy"]["mass"] = _power_law_mass()
    config["lensing"]["lens_galaxy"]["shear"] = [0.02, -0.01]

    validate_or_raise(config)


def test_fit_lens_matched_and_explicit_structure():
    """Enforce matched and explicit fit-lens block structure."""
    config = _config()
    validate_or_raise(config)

    config["modeling"]["fit_lens"] = {
        "mode": "matched",
        "lens_galaxy": {"mass": copy.deepcopy(_power_law_mass())},
    }
    with pytest.raises(ValueError, match="lens_galaxy"):
        validate_or_raise(config)

    config["modeling"]["fit_lens"] = {"mode": "explicit"}
    with pytest.raises(ValueError, match="lens_galaxy"):
        validate_or_raise(config)


@pytest.mark.parametrize(
    "mutation, match",
    [
        (("redshift", 0.3), "unsupported keys"),
        (("extra", 1), "unsupported keys"),
    ],
)
def test_explicit_fit_lens_rejects_unknown_lens_keys(mutation, match):
    """Reject fit-lens redshift and all other unsupported galaxy keys."""
    config = _config()
    key, value = mutation
    config["modeling"]["fit_lens"] = {
        "mode": "explicit",
        "lens_galaxy": {"mass": copy.deepcopy(_power_law_mass()), key: value},
    }
    with pytest.raises(ValueError, match=match):
        validate_or_raise(config)


def test_explicit_fit_lens_reuses_mass_validation():
    """Apply the same PowerLaw slope domain to the explicit fit model."""
    config = _config()
    mass = _power_law_mass()
    mass["slope"] = 3.0
    config["modeling"]["fit_lens"] = {
        "mode": "explicit",
        "lens_galaxy": {"mass": mass},
    }
    with pytest.raises(ValueError, match="modeling.fit_lens.*slope"):
        validate_or_raise(config)


@pytest.mark.parametrize("key", ["slope", "multipole_comp", "shear_comp"])
def test_item5_finite_difference_steps_are_required_and_positive(key):
    """Require every positive finite Item 5 Fisher step."""
    config = _config()
    config["modeling"]["fisher"]["finite_diff"].pop(key)
    with pytest.raises(ValueError, match=key):
        validate_or_raise(config)

    config = _config()
    config["modeling"]["fisher"]["finite_diff"][key] = 0.0
    with pytest.raises(ValueError, match=key):
        validate_or_raise(config)


@pytest.mark.parametrize("slope", [float("nan"), float("inf")])
def test_power_law_rejects_nonfinite_slope(slope):
    """Reject non-finite truth-side PowerLaw slopes."""
    config = _config()
    mass = _power_law_mass()
    mass["slope"] = slope
    config["lensing"]["lens_galaxy"]["mass"] = mass

    with pytest.raises(ValueError, match="slope"):
        validate_or_raise(config)


@pytest.mark.parametrize("component", [float("nan"), float("inf")])
@pytest.mark.parametrize("side", ["truth", "fit"])
def test_power_law_rejects_nonfinite_multipole_components(component, side):
    """Reject non-finite multipoles on both truth and fit mass blocks."""
    config = _config()
    mass = _power_law_mass()
    mass["multipoles"]["m4"] = [0.02, component]
    if side == "truth":
        config["lensing"]["lens_galaxy"]["mass"] = mass
    else:
        config["modeling"]["fit_lens"] = {
            "mode": "explicit",
            "lens_galaxy": {"mass": mass},
        }

    with pytest.raises(ValueError, match="multipoles.*m4"):
        validate_or_raise(config)


def test_shear_rejects_boolean_component():
    """Reject booleans masquerading as truth-side shear numbers."""
    config = _config()
    config["lensing"]["lens_galaxy"]["shear"] = [True, 0.01]

    with pytest.raises(ValueError, match="shear"):
        validate_or_raise(config)


@pytest.mark.parametrize("key", ["slope", "multipole_comp", "shear_comp"])
@pytest.mark.parametrize(
    "value",
    [float("nan"), float("inf"), -1.0e-3, True],
)
def test_item5_finite_difference_steps_reject_invalid_values(key, value):
    """Reject every invalid scalar category for every new Fisher step."""
    config = _config()
    config["modeling"]["fisher"]["finite_diff"][key] = value

    with pytest.raises(ValueError, match=key):
        validate_or_raise(config)


def test_master_config_declares_item5_finite_difference_steps():
    """Keep all three required Item 5 steps in the canonical config."""
    with (PROJECT_ROOT / "configs" / "master_config.yaml").open(
        "r", encoding="utf-8"
    ) as stream:
        config = yaml.safe_load(stream)
    finite_diff = config["modeling"]["fisher"]["finite_diff"]
    assert finite_diff["slope"] > 0.0
    assert finite_diff["multipole_comp"] > 0.0
    assert finite_diff["shear_comp"] > 0.0


def test_isothermal_galaxy_matches_direct_construction_exactly():
    """Keep the Isothermal image exactly equal to direct construction."""
    config = _config()
    generated = _generated(config)
    lens_config = config["lensing"]["lens_galaxy"]
    mass = lens_config["mass"]
    direct_lens = al.Galaxy(
        redshift=lens_config["redshift"],
        mass=al.mp.Isothermal(
            centre=tuple(mass["centre"]),
            ell_comps=tuple(mass["ell_comps"]),
            einstein_radius=mass["einstein_radius"],
        ),
    )
    direct = al.Tracer(
        galaxies=[direct_lens, generated.tracer.galaxies[1]],
        cosmology=generated.tracer.cosmology,
    ).image_2d_from(grid=generated.grid)
    np.testing.assert_array_equal(generated.image, direct.native)


def test_power_law_slope_two_matches_non_circular_isothermal():
    """Match the noncircular SIE limit without the Isothermal q clamp.

    The circular case is excluded because Isothermal clamps its axis ratio at
    0.99999 and consequently differs from PowerLaw by about 6e-5 relatively.
    """
    isothermal_config = _config()
    power_law_config = copy.deepcopy(isothermal_config)
    mass = _power_law_mass()
    mass.pop("multipoles")
    power_law_config["lensing"]["lens_galaxy"]["mass"] = mass

    isothermal = _generated(isothermal_config).image
    power_law = _generated(power_law_config).image
    # The observed baseline agreement is 7.8e-13 relative; this retains more
    # than two orders of margin while still detecting numerical drift.
    np.testing.assert_allclose(power_law, isothermal, rtol=1.0e-10, atol=0.0)


def test_asymmetric_flexible_lens_matches_independent_convention_oracles():
    """Pin profile components, angles, and off-axis macro deflections."""
    config = _config()
    mass = {
        "type": "PowerLaw",
        "centre": [0.03, -0.02],
        "ell_comps": [0.07, -0.04],
        "einstein_radius": 0.9,
        "slope": 2.08,
        "multipoles": {
            "m3": [0.011, -0.017],
            "m4": [0.019, 0.013],
        },
    }
    shear = [0.023, -0.014]
    config["lensing"]["lens_galaxy"]["mass"] = mass
    config["lensing"]["lens_galaxy"]["shear"] = shear
    lens = _generated(config).tracer.galaxies[0]

    assert tuple(lens.mass.centre) == tuple(mass["centre"])
    assert tuple(lens.mass.ell_comps) == tuple(mass["ell_comps"])
    for order_name, expected_m in (("m3", 3), ("m4", 4)):
        profile = getattr(lens, f"multipole_{order_name}")
        components = tuple(mass["multipoles"][order_name])
        assert profile.m == expected_m
        assert tuple(profile.multipole_comps) == components
        _, realized_angle_deg = ag.convert.multipole_k_m_and_phi_m_from(
            components,
            expected_m,
        )
        expected_angle_deg = np.degrees(
            np.arctan2(components[0], components[1])
        ) / expected_m
        assert realized_angle_deg == pytest.approx(expected_angle_deg)
    assert lens.shear.gamma_1 == shear[0]
    assert lens.shear.gamma_2 == shear[1]

    assert ag.convert.multipole_k_m_and_phi_m_from(
        (0.02, 0.0), 4
    )[1] == pytest.approx(22.5)
    assert ag.convert.multipole_k_m_and_phi_m_from(
        (0.0, 0.02), 4
    )[1] == pytest.approx(0.0)

    direct_lens = al.Galaxy(
        redshift=config["lensing"]["lens_galaxy"]["redshift"],
        mass=al.mp.PowerLaw(
            centre=tuple(mass["centre"]),
            ell_comps=tuple(mass["ell_comps"]),
            einstein_radius=mass["einstein_radius"],
            slope=mass["slope"],
        ),
        multipole_m3=al.mp.PowerLawMultipole(
            m=3,
            centre=tuple(mass["centre"]),
            einstein_radius=mass["einstein_radius"],
            slope=mass["slope"],
            multipole_comps=tuple(mass["multipoles"]["m3"]),
        ),
        multipole_m4=al.mp.PowerLawMultipole(
            m=4,
            centre=tuple(mass["centre"]),
            einstein_radius=mass["einstein_radius"],
            slope=mass["slope"],
            multipole_comps=tuple(mass["multipoles"]["m4"]),
        ),
        shear=al.mp.ExternalShear(gamma_1=shear[0], gamma_2=shear[1]),
    )
    points = al.Grid2DIrregular(
        values=[
            (-0.31, 0.27),
            (0.19, 0.42),
            (0.44, -0.36),
            (-0.52, -0.11),
            (0.08, -0.47),
        ]
    )
    np.testing.assert_allclose(
        lens.deflections_yx_2d_from(grid=points),
        direct_lens.deflections_yx_2d_from(grid=points),
        rtol=1.0e-12,
        atol=1.0e-12,
    )


def test_multipole_and_shear_each_change_the_lensed_image():
    """Make both an m4 term and external shear observably affect the image."""
    base_config = _config()
    base_mass = _power_law_mass()
    base_mass.pop("multipoles")
    base_config["lensing"]["lens_galaxy"]["mass"] = base_mass
    base_image = _generated(base_config).image

    multipole_config = copy.deepcopy(base_config)
    multipole_config["lensing"]["lens_galaxy"]["mass"]["multipoles"] = {
        "m4": [0.02, 0.0]
    }
    shear_config = copy.deepcopy(base_config)
    shear_config["lensing"]["lens_galaxy"]["shear"] = [0.02, -0.01]

    assert np.max(np.abs(_generated(multipole_config).image - base_image)) > 1.0e-8
    assert np.max(np.abs(_generated(shear_config).image - base_image)) > 1.0e-8


def test_subhalo_injection_preserves_all_macro_mass_profiles():
    """Retain PowerLaw, m3, m4, and shear when injecting a subhalo."""
    config = _config()
    config["lensing"]["lens_galaxy"]["mass"] = _power_law_mass()
    config["lensing"]["lens_galaxy"]["shear"] = [0.02, -0.01]
    config["lensing"]["subhalo"] = {
        "enabled": True,
        "mass": 1.0e7,
        "model": "PointMass",
        "position": {"type": "direct", "centre": [0.1, 0.9]},
    }
    with_subhalo = _generated(config)
    lens_with_subhalo = with_subhalo.tracer.galaxies[0]
    profile_types = (
        type(lens_with_subhalo.mass),
        type(lens_with_subhalo.multipole_m3),
        type(lens_with_subhalo.multipole_m4),
        type(lens_with_subhalo.shear),
        type(lens_with_subhalo.subhalo),
    )
    assert profile_types == (
        al.mp.PowerLaw,
        al.mp.PowerLawMultipole,
        al.mp.PowerLawMultipole,
        al.mp.ExternalShear,
        al.mp.PointMass,
    )

    no_subhalo_config = copy.deepcopy(config)
    no_subhalo_config["lensing"]["subhalo"]["enabled"] = False
    without_subhalo = _generated(no_subhalo_config)
    direct_macro = _create_lens_galaxy(config["lensing"]["lens_galaxy"])
    direct_tracer = al.Tracer(
        galaxies=[direct_macro, without_subhalo.tracer.galaxies[1]],
        cosmology=without_subhalo.tracer.cosmology,
    )
    direct_image = direct_tracer.image_2d_from(grid=without_subhalo.grid).native
    np.testing.assert_array_equal(without_subhalo.image, direct_image)
    assert np.max(np.abs(with_subhalo.image - without_subhalo.image)) > 1.0e-8


def test_lensing_data_records_flexible_lens_truth_and_summary(capsys):
    """Populate flexible truth metadata and print it in the scene summary."""
    config = _config()
    config["lensing"]["lens_galaxy"]["mass"] = _power_law_mass()
    config["lensing"]["lens_galaxy"]["shear"] = [0.02, -0.01]
    data = _generated(config)

    assert data.lens_mass_type == "PowerLaw"
    assert data.lens_slope == 2.0
    assert data.lens_multipoles == {"m3": (0.0, 0.01), "m4": (0.02, 0.0)}
    assert data.lens_shear == (0.02, -0.01)
    lens = data.tracer.galaxies[0]
    for multipole in (lens.multipole_m3, lens.multipole_m4):
        assert tuple(multipole.centre) == tuple(lens.mass.centre)
        assert multipole.einstein_radius == lens.mass.einstein_radius
        assert multipole.slope == lens.mass.slope

    print_lensing_data_summary(data)
    summary = capsys.readouterr().out
    assert "Mass type: PowerLaw" in summary
    assert "Slope: 2" in summary
    assert "m3" in summary
    assert "Shear" in summary

    isothermal = _generated(_config())
    assert isothermal.lens_mass_type == "Isothermal"
    assert isothermal.lens_slope is None
    assert isothermal.lens_multipoles is None
    assert isothermal.lens_shear is None


def test_plotting_baseline_rebuilds_exact_flexible_macro_lens():
    """Exclude only the subhalo from a flexible plotting baseline."""
    config = _config()
    config["lensing"]["lens_galaxy"]["mass"] = _power_law_mass()
    config["lensing"]["lens_galaxy"]["shear"] = [0.02, -0.01]
    config["lensing"]["subhalo"] = {
        "enabled": True,
        "mass": 1.0e7,
        "model": "PointMass",
        "position": {"type": "direct", "centre": [0.1, 0.9]},
    }
    with_subhalo = _generated(config)
    rebuilt = _no_subhalo_tracer(with_subhalo).image_2d_from(
        grid=with_subhalo.grid
    ).native
    expected_config = copy.deepcopy(config)
    expected_config["lensing"]["subhalo"]["enabled"] = False
    expected = _generated(expected_config).image
    np.testing.assert_array_equal(rebuilt, expected)

    legacy_lens = al.Galaxy(
        redshift=with_subhalo.lens_redshift,
        mass=al.mp.Isothermal(
            centre=with_subhalo.lens_centre,
            ell_comps=with_subhalo.lens_ellipticity,
            einstein_radius=with_subhalo.lens_einstein_radius,
        ),
    )
    legacy = al.Tracer(
        galaxies=[legacy_lens, with_subhalo.tracer.galaxies[1]],
        cosmology=with_subhalo.tracer.cosmology,
    ).image_2d_from(grid=with_subhalo.grid).native
    assert not np.allclose(rebuilt, legacy)


def test_plotting_baseline_keeps_isothermal_reconstruction_identical():
    """Keep the Isothermal plotting baseline identical to direct assembly."""
    config = _config()
    config["lensing"]["subhalo"] = {
        "enabled": True,
        "mass": 1.0e7,
        "model": "PointMass",
        "position": {"type": "direct", "centre": [0.1, 0.9]},
    }
    with_subhalo = _generated(config)
    rebuilt = _no_subhalo_tracer(with_subhalo).image_2d_from(
        grid=with_subhalo.grid
    ).native
    mass = config["lensing"]["lens_galaxy"]["mass"]
    direct_lens = al.Galaxy(
        redshift=with_subhalo.lens_redshift,
        mass=al.mp.Isothermal(
            centre=tuple(mass["centre"]),
            ell_comps=tuple(mass["ell_comps"]),
            einstein_radius=mass["einstein_radius"],
        ),
    )
    direct = al.Tracer(
        galaxies=[direct_lens, with_subhalo.tracer.galaxies[1]],
        cosmology=with_subhalo.tracer.cosmology,
    ).image_2d_from(grid=with_subhalo.grid).native
    np.testing.assert_array_equal(rebuilt, direct)


def test_registered_flexible_lensing_plots_write_nonempty_files(tmp_path):
    """Exercise every registered lens consumer on a flexible macro scene."""
    config = _config()
    config["run_name"] = "item5-lensing-plots"
    config["lensing"]["lens_galaxy"]["mass"] = _power_law_mass()
    config["lensing"]["lens_galaxy"]["shear"] = [0.02, -0.01]
    config["lensing"]["subhalo"] = {
        "enabled": True,
        "mass": 1.0e7,
        "model": "PointMass",
        "position": {"type": "direct", "centre": [0.1, 0.9]},
    }
    data = _generated(config)
    plot_config = {"output_dir": str(tmp_path)}

    plot_lensing_comparison(data, plot_config)
    plot_lensing_fractional_comparison(data, plot_config)
    plot_lensing_baseline_scene(data, plot_config)

    output_dir = tmp_path / config["run_name"] / "lensing"
    paths = [
        output_dir / "lensing_comparison.png",
        output_dir / "lensing_fractional_comparison.png",
        *output_dir.glob("lensing_baseline_scene_*Msun.png"),
    ]
    assert len(paths) == 3
    assert all(path.is_file() and path.stat().st_size > 0 for path in paths)
