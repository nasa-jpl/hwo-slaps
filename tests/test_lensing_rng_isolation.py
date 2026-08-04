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
    config = _make_lensing_config()
    full_config = {"global_seed": 123, "run_name": "rng-seed-a"}

    a = generate_lensing_system(copy.deepcopy(config), full_config=full_config)
    b = generate_lensing_system(copy.deepcopy(config), full_config=full_config)

    np.testing.assert_allclose(a.subhalo_position, b.subhalo_position, rtol=0.0, atol=0.0)


def test_generate_lensing_system_requires_full_config_argument():
    config = _make_lensing_config()
    with pytest.raises(TypeError):
        generate_lensing_system(copy.deepcopy(config))


def test_generate_lensing_system_requires_global_seed_key():
    config = _make_lensing_config()
    with pytest.raises(ValueError, match="Missing required key 'global_seed'"):
        generate_lensing_system(
            copy.deepcopy(config),
            full_config={"run_name": "missing-seed"},
        )


def test_generate_lensing_system_rejects_non_int_global_seed():
    config = _make_lensing_config()
    with pytest.raises(ValueError, match="full_config.global_seed must be an int"):
        generate_lensing_system(
            copy.deepcopy(config),
            full_config={"global_seed": True, "run_name": "bad-seed-type"},
        )


def test_subhalo_random_position_changes_with_seed():
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
    config = _make_lensing_config()
    full_config = {"global_seed": 42, "run_name": "rng-isolation"}

    np.random.seed(777)
    expected = np.random.random(8)

    np.random.seed(777)
    _ = generate_lensing_system(copy.deepcopy(config), full_config=full_config)
    after = np.random.random(8)

    np.testing.assert_allclose(after, expected, rtol=0.0, atol=0.0)
