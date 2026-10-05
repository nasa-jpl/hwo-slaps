"""The configured injection: the injected halo is the hypothesis family at its redshift, placed as configured."""

import math

import pytest

from hwoslaps.config.checks import ConfigError
from hwoslaps.scene.spec import parse_scene
from hwoslaps.scene.subhalo import configured_injection

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


@pytest.mark.parametrize("angle_deg, offset_arcsec", [(0.0, 0.0), (90.0, 0.1), (210.0, -0.25)])
def test_placement_is_about_the_lens_centre(scene_mapping, planck15, angle_deg, offset_arcsec):
    spec = _inject(scene_mapping, {"kind": "angle", "angle_deg": angle_deg, "offset_arcsec": offset_arcsec})
    y, x = configured_injection(spec, planck15, seed=3).position_yx_arcsec
    radius = 0.8 + offset_arcsec
    assert y == pytest.approx(LENS_CENTRE[0] + radius * math.sin(math.radians(angle_deg)), abs=1.0e-15)
    assert x == pytest.approx(LENS_CENTRE[1] + radius * math.cos(math.radians(angle_deg)), abs=1.0e-15)


def test_placement_radius_is_a_number_or_the_lens_einstein_radius(scene_mapping, planck15):
    numeric = _inject(scene_mapping, {"kind": "angle", "angle_deg": 90.0, "radius": 1.5, "offset_arcsec": -0.5})
    assert configured_injection(numeric, planck15, seed=3).position_yx_arcsec == pytest.approx((1.3, -0.2), abs=1.0e-15)
    inside_out = _inject(scene_mapping, {"kind": "angle", "angle_deg": 0.0, "offset_arcsec": -0.8})
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
