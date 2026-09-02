"""Tests for Isothermal macro-lens validation and generation."""

from __future__ import annotations

from pathlib import Path

import autolens as al
import numpy as np
import pytest
import yaml

from hwoslaps.config.validation import validate_or_raise
from hwoslaps.lensing.generator import (
    generate_lensing_system,
)
from hwoslaps.plotting.lensing_plots import (
    _no_subhalo_tracer,
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


@pytest.mark.parametrize("unknown_key", ["sheer", "light"])
def test_truth_lens_galaxy_rejects_unknown_keys(unknown_key):
    """Reject misspelled or unsupported truth-side galaxy keys."""
    config = _config()
    config["lensing"]["lens_galaxy"][unknown_key] = [0.02, -0.01]

    with pytest.raises(ValueError) as error:
        validate_or_raise(config)

    assert "lensing.lens_galaxy" in str(error.value)
    assert unknown_key in str(error.value)

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
