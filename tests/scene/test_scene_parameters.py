"""Scene parameters: names and order (the nuisance columns), replacement by name, pattern matching."""

import dataclasses

import pytest

from hwoslaps.config.checks import ConfigError
from hwoslaps.scene.parameters import match_parameters, scene_parameters, with_parameter
from hwoslaps.scene.spec import parse_scene


@pytest.fixture
def rich_scene(scene_mapping, tmp_path):
    asset = tmp_path / "clumps.npz"
    asset.write_bytes(b"parsing checks only that the file exists")
    scene_mapping["lens"]["light"] = {"bulge": {"type": "Exponential", "centre": [0.0, 0.01], "ell_comps": [0.0, 0.1],
                                                "effective_radius": 0.5, "intensity": 3.0}}
    scene_mapping["source"]["light"]["clumps"] = {"type": "Image", "asset_path": str(asset), "centre": [0.05, 0.0],
                                                  "rotation_deg": 15.0, "total_flux": 0.3}
    return parse_scene(scene_mapping)


def test_parameter_names_follow_registry_order(rich_scene):
    parameters = scene_parameters(rich_scene)
    assert [(p.name, p.definition.kind, p.definition.step_mode, p.value) for p in parameters] == [
        ("lens.mass.main.centre_y", "position", "additive", 0.0),
        ("lens.mass.main.centre_x", "position", "additive", 0.0),
        ("lens.mass.main.einstein_radius", "einstein_radius", "additive", 0.8),
        ("lens.mass.main.ell_comp_1", "ellipticity", "additive", 0.05),
        ("lens.mass.main.ell_comp_2", "ellipticity", "additive", 0.0),
        ("lens.light.bulge.centre_y", "position", "additive", 0.0),
        ("lens.light.bulge.centre_x", "position", "additive", 0.01),
        ("lens.light.bulge.ell_comp_1", "ellipticity", "additive", 0.0),
        ("lens.light.bulge.ell_comp_2", "ellipticity", "additive", 0.1),
        ("lens.light.bulge.intensity", "amplitude", "multiplicative", 3.0),
        ("lens.light.bulge.effective_radius", "size", "multiplicative", 0.5),
        ("source.light.disk.centre_y", "position", "additive", 0.02),
        ("source.light.disk.centre_x", "position", "additive", -0.03),
        ("source.light.disk.ell_comp_1", "ellipticity", "additive", 0.1),
        ("source.light.disk.ell_comp_2", "ellipticity", "additive", 0.05),
        ("source.light.disk.intensity", "amplitude", "multiplicative", 1.0),
        ("source.light.disk.effective_radius", "size", "multiplicative", 0.12),
        ("source.light.clumps.centre_y", "position", "additive", 0.05),
        ("source.light.clumps.centre_x", "position", "additive", 0.0),
        ("source.light.clumps.flux_scale", "amplitude", "multiplicative", 1.0),
        ("source.light.clumps.size_scale", "size", "multiplicative", 1.0),
        ("source.light.clumps.rotation_deg", "orientation", "additive", 15.0),
    ]


def test_with_parameter_replaces_one_scalar(rich_scene):
    moved = with_parameter(rich_scene, "source.light.disk.centre_x", -0.031)
    before = {p.name: p.value for p in scene_parameters(rich_scene)}
    after = {p.name: p.value for p in scene_parameters(moved)}
    assert after.pop("source.light.disk.centre_x") == -0.031
    assert before.pop("source.light.disk.centre_x") == -0.03
    assert after == before
    assert moved.lens is rich_scene.lens and moved.source.light[1] is rich_scene.source.light[1]
    assert dataclasses.replace(moved, source=rich_scene.source) == rich_scene
    assert with_parameter(rich_scene, "lens.light.bulge.intensity", 3.03).lens.light[0].values["intensity"] == 3.03


@pytest.mark.parametrize("name, value, path", [
    ("lens.mass.main.einstein_radius", -0.001, "scene.lens.mass.main.einstein_radius"),
    ("source.light.disk.ell_comp_2", 0.9945, "scene.source.light.disk.ell_comps"),
    ("source.light.clumps.size_scale", 0.0, "scene.source.light.clumps.size_scale"),
], ids=["einstein-radius-below-zero", "ellipticity-pair-at-the-clamp", "zero-size-scale"])
def test_with_parameter_refuses_values_outside_the_domain(rich_scene, name, value, path):
    with pytest.raises(ConfigError) as error:
        with_parameter(rich_scene, name, value)
    assert error.value.path == path
    assert name in error.value.message and repr(value) in error.value.message


def test_match_parameters_expands_patterns_in_parameter_order(rich_scene):
    names = [p.name for p in scene_parameters(rich_scene)]
    assert match_parameters(names, ["source.light.clumps.rotation_deg", "lens.light.*"], path="fixed") == (
        "lens.light.bulge.centre_y", "lens.light.bulge.centre_x", "lens.light.bulge.ell_comp_1",
        "lens.light.bulge.ell_comp_2", "lens.light.bulge.intensity", "lens.light.bulge.effective_radius",
        "source.light.clumps.rotation_deg")
    with pytest.raises(ConfigError) as error:
        match_parameters(names, ["lens.mass.*", "source.light.Disk.*"], path="forecast.nuisances.fixed")
    assert error.value.path == "forecast.nuisances.fixed[1]"
    assert "source.light.disk.intensity" in error.value.message
