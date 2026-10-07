"""The configured injection: the injected halo is the hypothesis family at its redshift, placed as configured."""

import math

import numpy as np
import pytest

from hwoslaps.config.checks import ConfigError
from hwoslaps.scene.spec import parse_scene
from hwoslaps.scene.subhalo import configured_injection
from hwoslaps.seeding import stream_rng

LENS_CENTRE = (0.3, -0.2)


def _inject(scene_mapping, position):
    scene_mapping["lens"]["mass"]["main"]["centre"] = list(LENS_CENTRE)
    scene_mapping["injection"] = {"mass_msun": 2.0e8, "position": position}
    return parse_scene(scene_mapping)


def test_injection_is_the_hypothesis_halo_at_the_configured_position(scene_mapping, planck15):
    assert configured_injection(parse_scene(scene_mapping), planck15, seed=3) is None
    scene_mapping["injection"] = {"mass_msun": 2.0e8, "position": {"kind": "direct", "centre": [0.25, -0.75]}}
    spec = parse_scene(scene_mapping)
    halo = configured_injection(spec, planck15, seed=3)
    assert (halo.model, halo.mass_msun, halo.position_yx_arcsec) == (spec.subhalo, 2.0e8, (0.25, -0.75))
    assert (halo.redshift, halo.source_redshift, halo.cosmology) == (0.2, 0.6, planck15)

    scene_mapping["subhalo"] = {"type": "PointMass", "redshift": 0.45}
    off_plane = configured_injection(parse_scene(scene_mapping), planck15, seed=3)
    assert (off_plane.model.type, off_plane.redshift) == ("PointMass", 0.45)


def _stream_draws(seed, scatter):
    """Angle (deg) and radial offset (arcsec) of a random placement, drawn by the documented stream."""
    stream = stream_rng(seed, "scene.injection_position")
    return stream.uniform(0.0, 360.0), stream.uniform(-scatter, scatter)


@pytest.mark.parametrize("position, seed", [
    ({"kind": "angle", "angle_deg": 0.0}, 3),
    ({"kind": "angle", "angle_deg": 90.0, "offset_arcsec": 0.1}, 3),
    ({"kind": "angle", "angle_deg": 210.0, "offset_arcsec": -0.25}, 3),
    ({"kind": "random", "scatter_arcsec": 0.05}, 3),
    ({"kind": "random", "scatter_arcsec": 0.05}, 20261005),
], ids=["angle-0", "angle-90-outward", "angle-210-inward", "random-seed-3", "random-seed-20261005"])
def test_placement_is_about_the_lens_centre(scene_mapping, planck15, position, seed):
    spec = _inject(scene_mapping, position)
    y, x = configured_injection(spec, planck15, seed=seed).position_yx_arcsec
    if position["kind"] == "angle":
        angle_deg, offset = position["angle_deg"], position.get("offset_arcsec", 0.0)
    else:
        angle_deg, offset = _stream_draws(seed, position["scatter_arcsec"])
    radius = 0.8 + offset
    assert y == pytest.approx(LENS_CENTRE[0] + radius * math.sin(math.radians(angle_deg)), abs=1.0e-15)
    assert x == pytest.approx(LENS_CENTRE[1] + radius * math.cos(math.radians(angle_deg)), abs=1.0e-15)


def test_random_placement_draws_from_its_named_stream(scene_mapping, planck15):
    original_global_state = np.random.get_state()
    try:
        np.random.seed(777)
        expected_global_draws = np.random.random(8)
        np.random.seed(777)
        spec = _inject(scene_mapping, {"kind": "random", "scatter_arcsec": 0.05})
        placed = {seed: configured_injection(spec, planck15, seed=seed).position_yx_arcsec for seed in (5, 6, 70)}
        assert configured_injection(spec, planck15, seed=5).position_yx_arcsec == placed[5]
        assert len(set(placed.values())) == 3
        for seed, (y, x) in placed.items():
            drawn_angle = math.degrees(math.atan2(y - LENS_CENTRE[0], x - LENS_CENTRE[1])) % 360.0
            assert drawn_angle == pytest.approx(_stream_draws(seed, 0.05)[0], abs=1.0e-9)
            for aliased in (np.random.default_rng(seed + 1), np.random.default_rng(seed)):
                assert drawn_angle != pytest.approx(aliased.uniform(0.0, 360.0), abs=1.0e-6)
        np.testing.assert_array_equal(np.random.random(8), expected_global_draws)
    finally:
        np.random.set_state(original_global_state)


def test_placement_radius_is_a_number_or_the_lens_einstein_radius(scene_mapping, planck15):
    numeric = _inject(scene_mapping, {"kind": "angle", "angle_deg": 90.0, "radius": 1.5, "offset_arcsec": -0.5})
    assert configured_injection(numeric, planck15, seed=3).position_yx_arcsec == pytest.approx((1.3, -0.2), abs=1.0e-15)
    for offset in (-0.8, -0.81):
        inside_out = _inject(scene_mapping, {"kind": "angle", "angle_deg": 0.0, "offset_arcsec": offset})
        with pytest.raises(ValueError, match="not positive"):
            configured_injection(inside_out, planck15, seed=3)
    scene_mapping["lens"]["mass"]["second"] = dict(scene_mapping["lens"]["mass"]["main"], centre=[1.0, 1.0])
    with pytest.raises(ConfigError) as error:
        _inject(scene_mapping, {"kind": "angle", "angle_deg": 0.0})
    assert error.value.path == "scene.injection.position.radius"


@pytest.mark.backend
def test_placement_on_the_critical_curve_uses_the_effective_einstein_radius(scene_mapping, planck15):
    from hwoslaps.scene.critical_curve import effective_einstein_radius

    spec = _inject(scene_mapping, {"kind": "angle", "angle_deg": 45.0, "radius": "critical_curve"})
    y, x = configured_injection(spec, planck15, seed=3).position_yx_arcsec
    assert math.hypot(y - LENS_CENTRE[0], x - LENS_CENTRE[1]) == pytest.approx(
        effective_einstein_radius(spec, planck15), rel=1.0e-14)
