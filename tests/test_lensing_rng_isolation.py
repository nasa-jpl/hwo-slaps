"""Tests for seed handling and RNG isolation in lensing generation."""

import copy

import numpy as np
import pytest

pytest.importorskip("autolens")

from hwoslaps.lensing.generator import generate_lensing_system


def _make_lensing_config():
    return {
        "grid": {"shape": [120, 120], "pixel_scale": 0.02},
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
            "mass": 1.0e9,
            "model": "PointMass",
            "position": {
                "type": "random",
                "scatter_pixels": 20,
            },
        },
        "cosmology": "Planck15",
    }


def test_subhalo_random_position_reproducible_for_same_seed():
    """Draw the same random subhalo position for one seed."""
    config = _make_lensing_config()
    full_config = {"global_seed": 123, "run_name": "rng-seed-a"}

    a = generate_lensing_system(copy.deepcopy(config), full_config=full_config)
    b = generate_lensing_system(copy.deepcopy(config), full_config=full_config)

    np.testing.assert_allclose(a.subhalo_position, b.subhalo_position, rtol=0.0, atol=0.0)


def test_generate_lensing_system_requires_an_explicit_seed_source():
    """Require deterministic randomness from a standalone seed or full config."""
    config = _make_lensing_config()
    with pytest.raises(ValueError, match="global_seed"):
        generate_lensing_system(copy.deepcopy(config))


def test_standalone_lensing_seed_preserves_scene_and_random_placement():
    """The standalone contract produces the exact legacy seeded image."""
    config = _make_lensing_config()
    config['grid']['shape'] = [48, 48]
    full_config = {'global_seed': 123, 'run_name': 'standalone'}
    legacy = generate_lensing_system(config, full_config=full_config)
    standalone = generate_lensing_system(config, seed=123, run_name='standalone')
    np.testing.assert_array_equal(legacy.image, standalone.image)
    np.testing.assert_array_equal(legacy.subhalo_position, standalone.subhalo_position)
    assert standalone.config['lensing'] == config
    assert standalone.config['global_seed'] == 123
    assert standalone.config['run_name'] == 'standalone'


def test_explicit_lensing_seed_records_override_in_provenance():
    config = _make_lensing_config()
    config['grid']['shape'] = [48, 48]
    full_config = {'global_seed': 7, 'run_name': 'original'}
    result = generate_lensing_system(config, full_config=full_config, seed=123)
    expected = generate_lensing_system(config, seed=123)
    np.testing.assert_array_equal(result.image, expected.image)
    assert result.config['global_seed'] == 123
    assert full_config['global_seed'] == 7


def test_generate_lensing_system_requires_global_seed_key():
    """Require a global_seed key inside full_config."""
    config = _make_lensing_config()
    with pytest.raises(ValueError, match="Missing required key 'global_seed'"):
        generate_lensing_system(
            copy.deepcopy(config),
            full_config={"run_name": "missing-seed"},
        )


def test_generate_lensing_system_rejects_non_int_global_seed():
    """Reject a boolean global_seed, which int accepts by subclassing."""
    config = _make_lensing_config()
    with pytest.raises(ValueError, match="full_config.global_seed must be an int"):
        generate_lensing_system(
            copy.deepcopy(config),
            full_config={"global_seed": True, "run_name": "bad-seed-type"},
        )


def test_subhalo_random_position_changes_with_seed():
    """Draw a different subhalo position for a different seed."""
    config = _make_lensing_config()

    a = generate_lensing_system(
        copy.deepcopy(config),
        full_config={"global_seed": 123, "run_name": "rng-a"},
    )
    b = generate_lensing_system(
        copy.deepcopy(config),
        full_config={"global_seed": 456, "run_name": "rng-b"},
    )

    assert not np.allclose(a.subhalo_position, b.subhalo_position, rtol=0.0, atol=0.0)


def test_lensing_generation_does_not_mutate_numpy_global_rng():
    """Leave the NumPy global RNG untouched during generation."""
    config = _make_lensing_config()
    full_config = {"global_seed": 42, "run_name": "rng-isolation"}

    np.random.seed(777)
    expected = np.random.random(8)

    np.random.seed(777)
    _ = generate_lensing_system(copy.deepcopy(config), full_config=full_config)
    after = np.random.random(8)

    np.testing.assert_allclose(after, expected, rtol=0.0, atol=0.0)
