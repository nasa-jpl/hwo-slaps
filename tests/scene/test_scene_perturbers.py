"""Halo population draws against analytic distributions and public scene-domain guards."""

import copy
from dataclasses import replace

import numpy as np
import pytest
from scipy.stats import kstest, spearmanr

from hwoslaps.config.checks import ConfigError
from hwoslaps.scene.cosmology import Cosmology, parse_cosmology
from hwoslaps.scene.perturbers import draw_population, realize_perturbers
from hwoslaps.scene.spec import parse_scene
from hwoslaps.scene.subhalo import configured_injection


@pytest.fixture
def population_block():
    return {"type": "PointMass", "mass_function": {"kind": "power_law", "slope": -1.9,
            "mass_min_msun": 1e6, "mass_max_msun": 1e9, "count": 20000},
            "spatial": {"kind": "uniform_disk", "radius_arcsec": 2.0}}


def _scene(mapping, blocks, *, halos=()):
    value = copy.deepcopy(mapping)
    value["perturbers"] = {"halos": list(halos), "populations": blocks}
    return parse_scene(value)


def _mass_cdf(values, low, high, slope):
    exponent = slope + 1
    logarithm = np.log(np.asarray(values) / low)
    width = np.log(high / low)
    return logarithm / width if exponent == 0 else np.expm1(exponent * logarithm) / np.expm1(exponent * width)


@pytest.mark.parametrize("slope", [-1.9, -1.0, -2.5, -1.0 - 1e-6, -1.0 + 1e-6])
@pytest.mark.parametrize("spatial", [{"kind": "uniform_disk", "radius_arcsec": 2.0},
                                    {"kind": "uniform_annulus", "inner_arcsec": 0.4, "outer_arcsec": 2.0}],
                         ids=["disc", "annulus"])
def test_population_draws_follow_the_analytic_distributions(scene_mapping, population_block, slope, spatial):
    block = copy.deepcopy(population_block)
    block["mass_function"]["slope"] = slope
    block["spatial"] = spatial
    spec = _scene(scene_mapping, [block]).perturbers.populations[0]
    centre = (0.25, -0.4)
    masses, positions = draw_population(spec, seed=20261005, index=0, lens_centre_yx=centre)
    assert masses.shape == (20000,) and positions.shape == (20000, 2)
    cdf = lambda values: _mass_cdf(values, 1e6, 1e9, slope)
    assert kstest(masses, cdf).pvalue > 1e-3
    offset = positions - centre
    squared_radius = np.sum(offset**2, axis=1)
    inner = spatial.get("inner_arcsec", 0.0)
    radial_cdf = (squared_radius - inner**2) / (2.0**2 - inner**2)
    angle_cdf = np.mod(np.arctan2(offset[:, 0], offset[:, 1]), 2 * np.pi) / (2 * np.pi)
    assert kstest(radial_cdf, "uniform").pvalue > 1e-3
    assert kstest(angle_cdf, "uniform").pvalue > 1e-3


def test_population_count_is_poisson(scene_mapping, population_block):
    block = copy.deepcopy(population_block)
    block["mass_function"].pop("count")
    block["mass_function"]["expected_count"] = 3.0
    spec = _scene(scene_mapping, [block]).perturbers.populations[0]
    counts = np.array([len(draw_population(spec, seed=seed, index=0, lens_centre_yx=(0., 0.))[0])
                       for seed in range(4000)])
    assert counts.mean() == pytest.approx(3., abs=5 * np.sqrt(3. / 4000))
    assert counts.var(ddof=1) == pytest.approx(3., abs=5 * np.sqrt((3. + 2 * 3.**2) / 3999))


def test_population_streams_are_isolated(scene_mapping, population_block):
    scene = _scene(scene_mapping, [population_block])
    spec = scene.perturbers.populations[0]
    args = {"seed": 11, "index": 0, "lens_centre_yx": (0.25, -0.4)}
    masses, positions = draw_population(spec, **args)
    repeated = draw_population(spec, **args)
    np.testing.assert_array_equal(repeated[0], masses)
    np.testing.assert_array_equal(repeated[1], positions)
    changed = copy.deepcopy(population_block)
    changed["spatial"] = {"kind": "uniform_annulus", "inner_arcsec": 0.4, "outer_arcsec": 2.0}
    spatial_spec = _scene(scene_mapping, [changed]).perturbers.populations[0]
    np.testing.assert_array_equal(draw_population(spatial_spec, **args)[0], masses)
    longer = replace(spec, mass_function=replace(spec.mass_function, count=20001))
    extended = draw_population(longer, **args)
    np.testing.assert_array_equal(extended[0][:-1], masses)
    np.testing.assert_array_equal(extended[1][:-1], positions)
    first = _scene(scene_mapping, [population_block, population_block]).perturbers.populations
    np.testing.assert_array_equal(draw_population(first[0], **args)[0], masses)
    other = draw_population(first[1], **{**args, "index": 1})
    assert not np.array_equal(other[0], masses) and not np.array_equal(other[1], positions)
    cosmology = Cosmology(parse_cosmology({"name": "Planck15"}))
    injected = copy.deepcopy(scene_mapping)
    injected["injection"] = {"mass_msun": 1e8,
                             "position": {"kind": "random", "radius": 0.8, "scatter_arcsec": 0.1}}
    with_injection = _scene(injected, [population_block])
    configured_injection(with_injection, cosmology, seed=11)
    np.testing.assert_array_equal(draw_population(with_injection.perturbers.populations[0], **args)[1], positions)
    offset = positions - args["lens_centre_yx"]
    radius = np.hypot(offset[:, 0], offset[:, 1])
    angle = np.mod(np.arctan2(offset[:, 0], offset[:, 1]), 2 * np.pi)
    # Marginal KS tests alone cannot detect two producer quantities using one child stream.
    assert abs(spearmanr(masses, radius).statistic) < 0.05
    assert abs(spearmanr(radius, angle).statistic) < 0.05


@pytest.mark.parametrize("edit, path", [
    ({"mass_function.expected_count": 3.}, "mass_function"),
    ({"mass_function.count": None}, "mass_function"),
    ({"mass_function.count": -1}, "mass_function.count"),
    ({"mass_function.count": True}, "mass_function.count"),
    ({"mass_function.mass_max_msun": 1e6}, "mass_function.mass_max_msun"),
    ({"spatial": {"kind": "uniform_annulus", "inner_arcsec": 2., "outer_arcsec": 1.}}, "spatial.outer_arcsec"),
    ({"redshift": 0.6}, "redshift"),
    ({"redshift": 0.0}, "redshift"),
    ({"type": "NFW", "concentration": {"kind": "moline2017_eq7", "x_sub": 1.},
      "mass_function.mass_min_msun": 1e5}, "mass_function.mass_min_msun"),
    ({"type": "NFW", "concentration": {"kind": "moline2017_eq7", "x_sub": 1.},
      "mass_function.mass_max_msun": 1e13}, "mass_function.mass_max_msun"),
    ({"type": "NFW", "concentration": {"kind": "moline2017_eq7", "x_sub": 1.},
      "redshift": 0.4}, "redshift"),
])
def test_population_block_rejects_invalid_values(scene_mapping, population_block, edit, path):
    block = copy.deepcopy(population_block)
    for key, value in edit.items():
        *parents, last = key.split(".")
        node = block
        for parent in parents:
            node = node[parent]
        node[last] = value
    with pytest.raises(ConfigError) as caught:
        _scene(scene_mapping, [block])
    assert caught.value.path == "scene.perturbers.populations[0]." + path


def test_zero_draw_and_listed_then_population_realization(scene_mapping, population_block):
    zero = copy.deepcopy(population_block)
    zero["mass_function"]["count"] = 0
    spec = _scene(scene_mapping, [zero])
    masses, positions = draw_population(spec.perturbers.populations[0], seed=11, index=0, lens_centre_yx=spec.lens_centre)
    assert masses.shape == (0,) and positions.shape == (0, 2)
    cosmology = Cosmology(parse_cosmology({"name": "Planck15"}))
    listed = {"type": "PointMass", "mass_msun": 1e8, "centre": [0.1, 0.2], "redshift": 0.4}
    first, second = copy.deepcopy(population_block), copy.deepcopy(population_block)
    first["mass_function"]["count"] = 2
    second["mass_function"]["count"] = 1
    second["redshift"] = 0.3
    mixed = _scene(scene_mapping, [first, second], halos=[listed])
    actual = realize_perturbers(mixed, cosmology, seed=11)
    baseline = realize_perturbers(_scene(scene_mapping, [], halos=[listed]), cosmology, seed=99)
    assert actual[:1] == baseline
    assert len(actual) == 4
    for index, start in ((0, 1), (1, 3)):
        expected_mass, expected_position = draw_population(mixed.perturbers.populations[index], seed=11, index=index,
                                                         lens_centre_yx=mixed.lens_centre)
        count = len(expected_mass)
        np.testing.assert_array_equal([halo.mass_msun for halo in actual[start:start + count]], expected_mass)
        np.testing.assert_array_equal([halo.position_yx_arcsec for halo in actual[start:start + count]], expected_position)
    assert [halo.redshift for halo in actual] == [0.4, mixed.lens.redshift, mixed.lens.redshift, 0.3]


@pytest.mark.parametrize("slope", [-1e308, 1e308])
def test_unrepresentable_population_powers_refuse_invalid_output(scene_mapping, population_block, slope):
    block = copy.deepcopy(population_block)
    block["mass_function"].update(count=3, slope=slope)
    spec = _scene(scene_mapping, [block]).perturbers.populations[0]
    with pytest.raises(ValueError, match="inverse CDF"):
        draw_population(spec, seed=11, index=0, lens_centre_yx=(0., 0.))
