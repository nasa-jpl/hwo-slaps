"""Spectral configuration choices and the AB cross-section boundary."""

from copy import deepcopy

import pytest

from hwoslaps.config.checks import ConfigError
from hwoslaps.config.schema import parse_config
from hwoslaps.scene.parameters import scene_parameter_names, with_parameter
from hwoslaps.scene.spec import parse_scene


@pytest.fixture
def flux_mapping(minimal_mapping):
    component = minimal_mapping["scene"]["source"]["light"]["light"]
    component.pop("intensity")
    component["flux"] = {"ab_mag": 24.845}
    minimal_mapping["instrument"]["collecting_area_m2"] = 33.606448937520405
    minimal_mapping["instrument"]["bandpass"] = {"kind": "top_hat", "min_nm": 450.0, "max_nm": 550.0, "throughput": 0.21}
    return minimal_mapping


def test_unresolved_flux_names_parse_without_placeholder_amplitude(flux_mapping):
    config = parse_config(flux_mapping)
    component = config.scene.source.light[0]
    assert component.values["intensity"] is None
    assert component.flux.ab_mag == 24.845
    assert "source.light.light.intensity" in scene_parameter_names(config.scene)


@pytest.mark.parametrize("change,path", [
    ("missing_band", "instrument.bandpass"), ("missing_area", "instrument.collecting_area_m2"),
    ("both_amplitudes", "scene.source.light.light"), ("reference_without_sed", "scene.source.light.light.sed"),
    ("sky_reference_without_sed", "observation.sky.sed"), ("unknown_flux", "scene.source.light.light.flux.typo"),
    ("unknown_sed", "scene.source.light.light.sed.typo"), ("unknown_reference_band", "scene.source.light.light.flux.reference_band.typo"),
    ("unknown_bandpass", "instrument.bandpass.typo"), ("unknown_sky", "observation.sky.typo"),
])
def test_spectral_keys_fail_at_their_own_boundary(flux_mapping, change, path):
    mapping = flux_mapping
    component = mapping["scene"]["source"]["light"]["light"]
    if change == "missing_band":
        del mapping["instrument"]["bandpass"]
    elif change == "missing_area":
        del mapping["instrument"]["collecting_area_m2"]
    elif change == "both_amplitudes":
        component["intensity"] = 1.0
    elif change == "reference_without_sed":
        component["flux"]["reference_band"] = deepcopy(mapping["instrument"]["bandpass"])
    elif change == "sky_reference_without_sed":
        mapping["observation"]["sky"] = {"ab_mag_per_arcsec2": 23.0, "reference_band": deepcopy(mapping["instrument"]["bandpass"])}
    elif change == "unknown_flux":
        component["flux"]["typo"] = 1
    elif change == "unknown_sed":
        component["sed"] = {"kind": "flat_fnu", "typo": 1}
    elif change == "unknown_reference_band":
        component["sed"] = {"kind": "flat_fnu"}
        component["flux"]["reference_band"] = deepcopy(mapping["instrument"]["bandpass"])
        component["flux"]["reference_band"]["typo"] = 1
    elif change == "unknown_bandpass":
        mapping["instrument"]["bandpass"]["typo"] = 1
    else:
        mapping["observation"]["sky"]["typo"] = 1
    with pytest.raises(ConfigError) as error:
        parse_config(mapping)
    assert error.value.path == path


def test_top_hat_to_product_overlay_replaces_variant_keys(flux_mapping):
    config = parse_config(flux_mapping)
    changed = config.replace({"instrument": {"bandpass": {"kind": "product", "support_nm": [450.0, 550.0],
        "factors": [{"kind": "constant", "value": 0.21}]}}})
    effective = changed.to_mapping()["instrument"]["bandpass"]
    assert effective["kind"] == "product"
    assert "min_nm" not in effective
    assert "throughput" not in effective
    assert parse_config(changed.to_mapping()) == changed


def test_parameter_copy_keeps_fixed_sed_and_flux_metadata(flux_mapping):
    component = flux_mapping["scene"]["source"]["light"]["light"]
    component["sed"] = {"kind": "power_law", "index": 0.7}
    scene = parse_scene(flux_mapping["scene"])
    changed = with_parameter(scene, "source.light.light.effective_radius", 0.2)
    assert changed.source.light[0].sed == scene.source.light[0].sed
    assert changed.source.light[0].flux == scene.source.light[0].flux
    assert changed.source.light[0].values["effective_radius"] == 0.2
    with pytest.raises(ConfigError, match="resolve"):
        with_parameter(scene, "source.light.light.intensity", 2.0)
