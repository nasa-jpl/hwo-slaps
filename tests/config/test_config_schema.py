"""Configuration replay, composition, scientific identity and section wiring."""

from copy import deepcopy
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from hwoslaps.config.checks import ConfigError
from hwoslaps.config.loading import dump_yaml, read_yaml, set_path
from hwoslaps.config.schema import compose_config, load_config, parse_config, resolve_config
from hwoslaps.identity import file_digest

PARITY = Path(__file__).resolve().parents[1] / "fixtures" / "paper_parity" / "engine"
PARITY_INPUTS = ("p1_optical_matched", "p2_delta_knowledge_error", "p3_image_source_kernel",
                 "p4_subhalo_sis", "p4_subhalo_pointmass")


def _base(name, minimal_mapping, image_asset):
    if name in PARITY_INPUTS:
        return load_config(PARITY / f"{name}.yaml")
    mapping = deepcopy(minimal_mapping)
    if name == "image_and_lens_light":
        mapping["scene"]["lens"]["light"] = {
            "bulge": {"type": "Exponential", "centre": [0.0, 0.0], "ell_comps": [0.0, 0.0],
                      "intensity": 0.3, "effective_radius": 0.2}}
        mapping["scene"]["source"]["light"]["light"] = {
            "type": "Image", "asset_path": str(image_asset), "centre": [-0.03, 0.08], "total_flux": 0.29}
    return parse_config(mapping)


@pytest.mark.parametrize("name", ("minimal", "image_and_lens_light") + PARITY_INPUTS)
def test_effective_configuration_round_trips_through_yaml(name, minimal_mapping, image_asset, write_config):
    config = _base(name, minimal_mapping, image_asset)
    effective = config.to_mapping()
    replay = load_config(write_config(effective))
    assert replay == config
    assert replay.digest() == config.digest()
    assert effective["observation"]["exposure_count"] == 1
    assert effective["forecast"]["nuisances"]["background_offset"] is True
    if name != "p2_delta_knowledge_error":
        assert effective["psf"]["model"] == {"kind": "matched"}
    digest = config.digest()
    effective["scene"]["lens"]["mass"]["mass"]["centre"][0] = 13.0
    assert config.to_mapping()["scene"]["lens"]["mass"]["mass"]["centre"] == [0.0, 0.0]
    assert config.scene.lens.mass[0].values["centre"] == (0.0, 0.0)
    assert config.digest() == digest


def _mapping_paths(value, path=()):
    if isinstance(value, dict):
        yield path
        for key, item in value.items():
            yield from _mapping_paths(item, path + (key,))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            yield from _mapping_paths(item, path + (index,))


def _dotted(path, root):
    text = ""
    node = root
    for part in path:
        text = f"{text}[{part}]" if isinstance(node, list) else f"{text}.{part}" if text else str(part)
        node = node[part]
    return text


@pytest.mark.parametrize("name", ("minimal", "image_and_lens_light") + PARITY_INPUTS)
def test_unknown_keys_are_rejected_at_every_mapping_level(name, minimal_mapping, image_asset):
    effective = _base(name, minimal_mapping, image_asset).to_mapping()
    for path in _mapping_paths(effective):
        candidate = deepcopy(effective)
        node = candidate
        for part in path:
            node = node[part]
        node["zz_unknown"] = 1
        with pytest.raises(ConfigError) as error:
            parse_config(candidate)
        assert error.value.path.startswith(_dotted(path, effective)), (path, error.value.path)


def test_paths_resolve_against_their_declaring_file(minimal_mapping, tmp_path, monkeypatch):
    first, second, override = (tmp_path / name for name in ("first", "second", "override"))
    for directory in (first, second, override):
        directory.mkdir()
        np.save(directory / "kernel.npy", np.ones((1, 1)))
    mapping = deepcopy(minimal_mapping)
    mapping["psf"]["truth"]["path"] = "kernel.npy"
    a = first / "base.yaml"
    a.write_text(dump_yaml(mapping))
    b = second / "model.yaml"
    b.write_text(dump_yaml({"psf": {"model": {"kind": "kernel", "path": "kernel.npy",
                                             "pixel_scale_arcsec": 0.05}}}))
    composed = compose_config([a, b], base_dir=override)
    assert composed["psf"]["truth"]["path"] == str(first / "kernel.npy")
    assert composed["psf"]["model"]["path"] == str(second / "kernel.npy")
    config = load_config([a, b], overrides={"psf": {"model": {"path": "kernel.npy"}}}, base_dir=override)
    assert config.psf.truth.path == first / "kernel.npy"
    assert config.psf.model.path == override / "kernel.npy"
    monkeypatch.chdir(override)
    current = load_config(a, overrides={"psf": {"truth": {"path": "kernel.npy"}}})
    assert current.psf.truth.path == override / "kernel.npy"
    replay = parse_config(config.to_mapping(), base_dir=first)
    assert replay == config
    optical = load_config(PARITY / "p2_delta_knowledge_error.yaml", base_dir=override)
    assert optical.to_mapping()["psf"]["model"]["draw"]["prior"]["packaged"] == "jwst_wss_drift_v1"


@pytest.mark.parametrize("switch", ("knowledge_to_matched", "optical_to_kernel", "wavefront_to_offset"))
def test_overlays_switch_alternatives(switch, minimal_mapping, tmp_path):
    original = load_config(PARITY / "p2_delta_knowledge_error.yaml")
    if switch == "knowledge_to_matched":
        overlay = {"psf": {"model": {"kind": "matched"}}}
        expected = original.to_mapping()
        expected["psf"]["model"] = {"kind": "matched"}
    elif switch == "optical_to_kernel":
        overlay = {"psf": {"truth": {"kind": "kernel", "path": minimal_mapping["psf"]["truth"]["path"],
                                     "pixel_scale_arcsec": 0.03}, "model": {"kind": "matched"}},
                   "forecast": {"nuisances": {"wavefront": None}}}
        expected = original.to_mapping()
        expected["psf"] = overlay["psf"]
        expected["forecast"]["nuisances"]["wavefront"] = None
    else:
        original = original.replace({"psf": {"model": {"kind": "wavefront", "wavefront": {"zernikes": {4: 2}}}}})
        overlay = {"psf": {"model": {"wavefront": None, "offset": {"zernikes": {5: 3}}}}}
        expected = original.to_mapping()
        expected["psf"]["model"] = {"kind": "wavefront", "wavefront": None, "offset": {"zernikes": {5: 3}}}
    base_path, overlay_path = tmp_path / "base.yaml", tmp_path / "overlay.yaml"
    base_path.write_text(dump_yaml(original.to_mapping()))
    overlay_path.write_text(dump_yaml(overlay))
    final = parse_config(expected)
    assert load_config([base_path, overlay_path]) == final
    assert original.replace(overlay) == final


@pytest.mark.parametrize(("row", "path"), (
    ("truth_scale", "psf.truth.pixel_scale_arcsec"),
    ("model_scale", "psf.model.pixel_scale_arcsec"),
    ("wavefront_needs_optics", "psf.model.kind"),
    ("knowledge_needs_optics", "psf.model.kind"),
    ("coeff_segment_exists", "psf.truth.wavefront.segment_hexikes.100"),
    ("optical_sampling", "psf.truth"),
    ("fixed_name", "forecast.nuisances.fixed[0]"),
    ("step_name", "forecast.nuisances.steps.unknown"),
    ("prior_name", "forecast.nuisances.priors.unknown"),
    ("wavefront_basis", "forecast.nuisances.wavefront"),
    ("nuisance_segment_exists", "forecast.nuisances.wavefront.modes.segment_hexikes.segments"),
    ("ring_radius", "forecast.positions.radius"),
))
def test_cross_section_rules(row, path, minimal_mapping):
    valid = (load_config(PARITY / "p1_optical_matched.yaml").to_mapping()
             if row in ("coeff_segment_exists", "optical_sampling", "nuisance_segment_exists")
             else parse_config(minimal_mapping).to_mapping())
    if row == "truth_scale":
        bad = set_path(valid, "psf.truth.pixel_scale_arcsec", 0.06, create=False)
    elif row == "model_scale":
        valid["psf"]["model"] = dict(valid["psf"]["truth"])
        bad = set_path(valid, "psf.model.pixel_scale_arcsec", 0.06, create=False)
    elif row == "wavefront_needs_optics":
        optical = load_config(PARITY / "p1_optical_matched.yaml").to_mapping()
        optical["psf"]["model"] = {"kind": "wavefront", "offset": {"zernikes": {4: 1}}}
        valid = optical
        bad = deepcopy(valid)
        bad["psf"]["truth"] = minimal_mapping["psf"]["truth"]
    elif row == "knowledge_needs_optics":
        valid = load_config(PARITY / "p2_delta_knowledge_error.yaml").to_mapping()
        bad = deepcopy(valid)
        bad["psf"]["truth"] = minimal_mapping["psf"]["truth"]
    elif row == "coeff_segment_exists":
        bad = set_path(valid, "psf.truth.wavefront.segment_hexikes", {100: {4: 1}}, create=False)
    elif row == "optical_sampling":
        bad = set_path(valid, "psf.truth.detector_oversampling", 1, create=False)
    elif row == "fixed_name":
        valid["forecast"]["nuisances"]["fixed"] = ["source.light.*.intensity"]
        bad = set_path(valid, "forecast.nuisances.fixed", ["unknown.*"], create=False)
    elif row == "step_name":
        valid["forecast"]["nuisances"]["steps"] = {"position": 0.001, "lens.mass.mass.einstein_radius": 0.002}
        bad = set_path(valid, "forecast.nuisances.steps", {"unknown": 0.001}, create=False)
    elif row == "prior_name":
        valid["forecast"]["nuisances"]["priors"] = {"source.light.light.intensity": 0.1}
        bad = set_path(valid, "forecast.nuisances.priors", {"unknown": 0.1}, create=False)
    elif row == "wavefront_basis":
        valid = load_config(PARITY / "p1_optical_matched.yaml").to_mapping()
        bad = deepcopy(valid)
        bad["psf"]["model"] = {"kind": "kernel", "path": minimal_mapping["psf"]["truth"]["path"],
                                 "pixel_scale_arcsec": 0.03}
    elif row == "nuisance_segment_exists":
        bad = set_path(valid, "forecast.nuisances.wavefront.modes.segment_hexikes.segments", [100], create=False)
    else:
        valid["forecast"]["positions"] = {"kind": "ring", "count": 5}
        bad = deepcopy(valid)
        bad["scene"]["lens"]["mass"]["companion"] = {
            "type": "Isothermal", "centre": [0.5, 0.0], "einstein_radius": 0.1, "ell_comps": [0.0, 0.0]}
    parse_config(valid)
    with pytest.raises(ConfigError) as error:
        parse_config(bad)
    assert error.value.path == path


@pytest.mark.parametrize(("section", "value", "path"), (
    ("positions", {"kind": "grid", "spacing_arcsec": 0.2, "half_width_arcsec": 0.1}, "forecast.positions.half_width_arcsec"),
    ("positions", {"kind": "grid", "spacing_arcsec": 0.1, "half_width_arcsec": 0.2,
                   "annulus": {"inner_arcsec": 0.2, "outer_arcsec": 0.1}}, "forecast.positions.annulus"),
    ("positions", {"kind": "explicit", "positions_yx": [[1, 2], [1, 2]]}, "forecast.positions.positions_yx[1]"),
    ("positions", {"kind": "ring", "count": 0}, "forecast.positions.count"),
    ("positions", {"kind": "ring", "count": 2, "radius": 0.1, "offset_arcsec": -0.2}, "forecast.positions.offset_arcsec"),
    ("mask", {"kind": "source_snr", "snr_min": 0}, "forecast.mask.snr_min"),
    ("mask", {"kind": "annulus", "inner_arcsec": 0.2, "outer_arcsec": 0.1}, "forecast.mask"),
    ("mask", {"kind": "all_pixels", "snr_min": 1}, "forecast.mask.snr_min"),
    ("nuisances", {"fixed": ["lens.*", "lens.*"]}, "forecast.nuisances.fixed[1]"),
    ("nuisances", {"steps": {"position": 0}}, "forecast.nuisances.steps.position"),
    ("nuisances", {"priors": {"lens.mass.mass.centre_y": -1}}, "forecast.nuisances.priors.lens.mass.mass.centre_y"),
    ("nuisances", {"wavefront": {"modes": {"zernikes": {"nolls": [1]}}}}, "forecast.nuisances.wavefront.modes.zernikes.nolls"),
    ("nuisances", {"wavefront": {"modes": {"zernikes": {"nolls": [4]}}, "step_nm": {"segment_hexikes": 1}}},
     "forecast.nuisances.wavefront.step_nm"),
))
def test_forecast_inputs_reject_invalid_layout_mask_and_nuisance_values(section, value, path, minimal_mapping):
    candidate = deepcopy(minimal_mapping)
    candidate["forecast"][section] = value
    with pytest.raises(ConfigError) as error:
        parse_config(candidate)
    assert error.value.path == path


@pytest.mark.parametrize(("positions", "mask"), (
    ({"kind": "grid", "spacing_arcsec": 0.1, "half_width_arcsec": 0.2,
      "annulus": {"inner_arcsec": 0.0, "outer_arcsec": 0.3}}, {"kind": "source_snr", "snr_min": 1}),
    ({"kind": "ring", "count": 3, "radius": 0.8, "offset_arcsec": 0.1},
     {"kind": "annulus", "inner_arcsec": 0.0, "outer_arcsec": 0.3, "about": "grid"}),
    ({"kind": "explicit", "positions_yx": [[0, 1]]}, {"kind": "psf_border"}),
))
def test_forecast_variants_preserve_typed_scientific_fields(positions, mask, minimal_mapping):
    candidate = deepcopy(minimal_mapping)
    candidate["forecast"].update(positions=positions, mask=mask)
    config = parse_config(candidate)
    assert config.forecast.positions.kind == positions["kind"]
    assert config.forecast.mask.kind == mask["kind"]
    if positions["kind"] == "grid":
        assert config.forecast.positions.annulus == (0.0, 0.3)
    elif positions["kind"] == "ring":
        assert config.forecast.positions.radius == 0.8
        assert config.forecast.positions.offset_arcsec == 0.1
        assert config.forecast.mask.about == "grid"
    else:
        assert config.forecast.positions.positions_yx == ((0.0, 1.0),)


def test_digest_depends_on_file_content_not_location(minimal_mapping, tmp_path):
    first = parse_config(minimal_mapping)
    other_path = tmp_path / "copy.npy"
    original_path = first.psf.truth.path
    other_path.write_bytes(original_path.read_bytes())
    second = first.replace({"run_name": "elsewhere", "psf": {"truth": {"path": str(other_path)}}})
    assert first.digest() == second.digest()
    assert first.comparison_digest() == second.comparison_digest()
    assert first != second
    assert first.file_digests() == {str(original_path): file_digest(original_path)}
    before = second.digest()
    with other_path.open("ab") as stream:
        stream.write(b"changed")
    assert second.digest() != before
    assert first.file_digests()[str(original_path)] != second.file_digests()[str(other_path)]
    renamed = first.replace({"run_name": "new_label"})
    assert renamed == first and renamed.digest() == first.digest()
    with pytest.raises(TypeError):
        hash(first)


def test_digest_and_equality_respect_component_order(minimal_mapping, write_config):
    mapping = deepcopy(minimal_mapping)
    mapping["scene"]["lens"]["mass"]["companion"] = {
        "type": "Isothermal", "centre": [0.5, 0.0], "einstein_radius": 0.1, "ell_comps": [0.0, 0.0]}
    first = parse_config(mapping)
    mapping["scene"]["lens"]["mass"] = dict(reversed(tuple(mapping["scene"]["lens"]["mass"].items())))
    second = parse_config(mapping)
    assert first != second
    assert first.digest() != second.digest()
    assert first.comparison_digest() != second.comparison_digest()
    for config in (first, second):
        replay = load_config(write_config(config.to_mapping()))
        assert replay == config and replay.digest() == config.digest()


def test_comparison_digest_holds_every_scientific_input_except_model_psf(minimal_mapping):
    base = parse_config(minimal_mapping)
    model = base.replace({"psf": {"model": dict(base.to_mapping()["psf"]["truth"])}})
    assert model.digest() != base.digest()
    assert model.comparison_digest() == base.comparison_digest()
    exposure = model.replace({"observation": {"exposure_time_s": 901}})
    assert exposure.comparison_digest() != base.comparison_digest()


def test_copy_isolates_nested_typed_inputs_without_reopening_assets(minimal_mapping):
    base = parse_config(minimal_mapping)
    caller_values = {"centre": [0.0, 0.0], "ell_comps": [0.05, 0.0], "einstein_radius": 0.8}
    component = replace(base.scene.lens.mass[0], values=caller_values)
    caller = replace(base, scene=replace(base.scene, lens=replace(base.scene.lens, mass=(component,))))
    copy = deepcopy(caller)
    caller_values["centre"][0] = 4.0
    assert copy.scene.lens.mass[0].values["centre"] == [0.0, 0.0]
    copy.scene.lens.mass[0].values["ell_comps"][0] = 0.2
    assert caller.scene.lens.mass[0].values["ell_comps"] == [0.05, 0.0]
    base.psf.truth.path.unlink()
    isolated = deepcopy(base)
    assert isolated.scene.lens.mass[0].values["centre"] == (0.0, 0.0)
    with pytest.raises(TypeError):
        isolated.scene.lens.mass[0].values["centre"][0] = 3


def test_config_source_resolution_and_optional_forecast(minimal_mapping, write_config):
    config = parse_config(minimal_mapping)
    assert resolve_config(config) is config
    assert resolve_config(minimal_mapping) == config
    assert resolve_config(write_config(minimal_mapping)) == config
    assert resolve_config([write_config(minimal_mapping)]) == config
    no_forecast = deepcopy(minimal_mapping)
    del no_forecast["forecast"]
    assert parse_config(no_forecast).forecast is None
    with pytest.raises(ConfigError, match="at least one"):
        compose_config([])
    with pytest.raises(ConfigError, match="mapping"):
        parse_config([])
