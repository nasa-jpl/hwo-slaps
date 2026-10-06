"""Scene specification: strict reading, domains and scene rules, typed values, light groups."""

import copy

import pytest

from hwoslaps.config.checks import ConfigError
from hwoslaps.scene.halos import HaloModel, Moline2017
from hwoslaps.scene.spec import LightGroup, parse_scene

PERTURBER = {"type": "PointMass", "mass_msun": 1.0e8, "centre": [0.3, 0.4]}
NFW_PERTURBER = {"type": "NFW", "concentration": {"kind": "moline2017_eq7", "x_sub": 1.0}, "mass_msun": 1.0e8,
                 "centre": [0.3, 0.4]}


def _set(mapping, dotted, value):
    keys = dotted.split(".")
    for key in keys[:-1]:
        mapping = mapping[key]
    if value is _DELETE:
        del mapping[keys[-1]]
    else:
        mapping[keys[-1]] = value


_DELETE = object()


# (row id, edits applied to the minimal scene, dotted ConfigError path)
DOMAIN_ROWS = [
    ("mass-ellipticity-at-the-autogalaxy-clamp", {"lens.mass.main.ell_comps": [0.999, 0.0]}, "scene.lens.mass.main.ell_comps"),
    ("light-ellipticity-at-the-autogalaxy-clamp", {"source.light.disk.ell_comps": [0.6, 0.7993]},
     "scene.source.light.disk.ell_comps"),
    ("non-positive-einstein-radius", {"lens.mass.main.einstein_radius": 0.0}, "scene.lens.mass.main.einstein_radius"),
    ("source-in-front-of-the-lens", {"source.redshift": 0.2}, "scene.source.redshift"),
    ("missing-light-amplitude", {"source.light.disk.intensity": _DELETE}, "scene.source.light.disk"),
    ("unknown-component-key", {"lens.mass.main.slope": 2.1}, "scene.lens.mass.main.slope"),
    ("no-lens-mass", {"lens.mass": {}}, "scene.lens.mass"),
    ("zero-over-sampling", {"grid.over_sample_size": 0}, "scene.grid.over_sample_size"),
    ("nfw-without-concentration", {"subhalo.concentration": _DELETE}, "scene.subhalo.concentration"),
    ("point-mass-with-concentration", {"subhalo.type": "PointMass"}, "scene.subhalo.concentration"),
    ("moline-host-radius-beyond-calibration", {"subhalo.concentration.x_sub": 1.6}, "scene.subhalo.concentration.x_sub"),
    ("unknown-concentration-key", {"subhalo.concentration.c0": 9.0}, "scene.subhalo.concentration.c0"),
    ("hypothesis-behind-the-source", {"subhalo.redshift": 0.6}, "scene.subhalo.redshift"),
    ("moline-off-the-lens-plane", {"subhalo.redshift": 0.4}, "scene.subhalo.redshift"),
    ("moline-injection-mass-below-calibration",
     {"injection": {"mass_msun": 1.0e5, "position": {"kind": "direct", "centre": [0.1, 0.2]}}},
     "scene.injection.mass_msun"),
    ("perturber-behind-the-source", {"perturbers": {"halos": [dict(PERTURBER, redshift=0.7)]}},
     "scene.perturbers.halos[0].redshift"),
    ("moline-perturber-off-the-lens-plane", {"perturbers": {"halos": [dict(NFW_PERTURBER, redshift=0.4)]}},
     "scene.perturbers.halos[0].redshift"),
    ("moline-perturber-mass-above-calibration", {"perturbers": {"halos": [dict(NFW_PERTURBER, mass_msun=2.0e12)]}},
     "scene.perturbers.halos[0].mass_msun"),
    ("component-named-subhalo", {"lens.mass.subhalo": {"type": "Isothermal", "centre": [0.0, 0.0],
                                                       "einstein_radius": 0.1, "ell_comps": [0.0, 0.0]}},
     "scene.lens.mass.subhalo"),
    ("component-named-redshift", {"source.light.redshift": {"type": "Exponential", "centre": [0.0, 0.0],
                                                            "ell_comps": [0.0, 0.0], "effective_radius": 0.1,
                                                            "intensity": 1.0}},
     "scene.source.light.redshift"),
    ("component-named-id", {"lens.mass.id": {"type": "Isothermal", "centre": [0.0, 0.0], "einstein_radius": 0.1,
                                             "ell_comps": [0.0, 0.0]}},
     "scene.lens.mass.id"),
    ("component-named-cls", {"source.light.cls": {"type": "Exponential", "centre": [0.0, 0.0], "ell_comps": [0.0, 0.0],
                                                  "effective_radius": 0.1, "intensity": 1.0}},
     "scene.source.light.cls"),
    ("component-with-the-perturber-prefix", {"lens.mass.perturber_main": {"type": "Isothermal", "centre": [0.0, 0.0],
                                                                          "einstein_radius": 0.1,
                                                                          "ell_comps": [0.0, 0.0]}},
     "scene.lens.mass.perturber_main"),
    ("component-with-a-layout-suffix", {"lens.mass.main_multipole_m3": {"type": "Isothermal", "centre": [0.0, 0.0],
                                                                        "einstein_radius": 0.1,
                                                                        "ell_comps": [0.0, 0.0]}},
     "scene.lens.mass.main_multipole_m3"),
    ("mass-and-light-sharing-a-name", {"lens.light": {"main": {"type": "Exponential", "centre": [0.0, 0.0],
                                                               "ell_comps": [0.0, 0.0], "effective_radius": 0.5,
                                                               "intensity": 1.0}}},
     "scene.lens.light.main"),
]


@pytest.mark.parametrize("edits, path", [row[1:] for row in DOMAIN_ROWS], ids=[row[0] for row in DOMAIN_ROWS])
def test_scene_parse_refuses_values_outside_the_scene_domains(scene_mapping, edits, path):
    for dotted, value in edits.items():
        _set(scene_mapping, dotted, value)
    with pytest.raises(ConfigError) as error:
        parse_scene(scene_mapping)
    assert error.value.path == path


def test_parsed_scene_holds_the_read_values_and_defaults(scene_mapping):
    original = copy.deepcopy(scene_mapping)
    scene_mapping["lens"]["light"] = {"bulge": {"type": "Exponential", "centre": [0.0, 0.0], "ell_comps": [0.0, 0.1],
                                                "effective_radius": 0.5, "intensity": 3.0}}
    scene_mapping["perturbers"] = {"halos": [PERTURBER, dict(PERTURBER, redshift=0.4)]}
    spec = parse_scene(scene_mapping)

    assert spec.grid.shape == (40, 40) and spec.grid.pixel_scale_arcsec == 0.05 and spec.grid.over_sample_size == 2
    (main,) = spec.lens.mass
    assert (main.name, main.plane, main.role, main.type) == ("main", "lens", "mass", "Isothermal")
    assert dict(main.values) == {"centre": (0.0, 0.0), "einstein_radius": 0.8, "ell_comps": (0.05, 0.0),
                                "multipoles": None}
    assert spec.lens.light[0].values["intensity"] == 3.0 and spec.source.mass == ()
    assert spec.subhalo == HaloModel("NFW", Moline2017(x_sub=1.0, h=None), None)
    assert spec.subhalo_redshift is None and spec.injection is None
    assert [(halo.mass_msun, halo.centre_yx, halo.redshift) for halo in spec.perturbers.halos] == [
        (1.0e8, (0.3, 0.4), None), (1.0e8, (0.3, 0.4), 0.4)]
    assert spec.lens_centre == (0.0, 0.0) and spec.einstein_radius() == 0.8
    with pytest.raises(TypeError):
        main.values["einstein_radius"] = 1.0
    assert scene_mapping["subhalo"] == original["subhalo"]


def test_light_groups_follow_the_planes_lens_first(scene_mapping):
    scene_mapping["source"]["light"]["knot"] = dict(scene_mapping["source"]["light"]["disk"], centre=[0.1, 0.1])
    assert dict(parse_scene(scene_mapping).light_groups()) == {
        "source": LightGroup("source", None, ("disk", "knot"))}
    scene_mapping["lens"]["light"] = {"bulge": {"type": "Exponential", "centre": [0.0, 0.0], "ell_comps": [0.0, 0.0],
                                                "effective_radius": 0.5, "intensity": 3.0}}
    groups = parse_scene(scene_mapping).light_groups()
    assert list(groups) == ["lens", "source"]
    assert groups["lens"] == LightGroup("lens", None, ("bulge",))


def test_einstein_radius_needs_exactly_one_lens_component_with_one(scene_mapping):
    scene_mapping["lens"]["mass"]["second"] = dict(scene_mapping["lens"]["mass"]["main"], centre=[0.5, 0.5],
                                                   einstein_radius=0.3)
    spec = parse_scene(scene_mapping)
    with pytest.raises(ValueError, match="2 mass components with an Einstein radius"):
        spec.einstein_radius()
    assert spec.einstein_radii() == (0.8, 0.3) and spec.lens_centre == (0.0, 0.0)
