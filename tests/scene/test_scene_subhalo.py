"""The configured injection: the injected halo is the hypothesis family at its redshift, placed as configured."""

from hwoslaps.scene.spec import parse_scene
from hwoslaps.scene.subhalo import configured_injection


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
