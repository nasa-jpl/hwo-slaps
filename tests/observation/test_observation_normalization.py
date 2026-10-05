"""Flux-to-amplitude normalization, real grouped monochromatic renders and spectral ownership."""

from copy import deepcopy
from pathlib import Path
import json
import pickle

import numpy as np
import pytest

pytestmark = pytest.mark.backend


def test_flux_normalization_matches_independent_unlensed_exponential_sum(minimal_mapping):
    import autolens as al
    from hwoslaps.config.schema import parse_config
    from hwoslaps.observation.normalization import resolve_observing
    from hwoslaps.fisher.psf_pair import truth_provider

    mapping = minimal_mapping
    mapping["scene"]["grid"].update(shape=[300, 300], pixel_scale_arcsec=0.01, over_sample_size=8)
    mapping["psf"]["truth"]["pixel_scale_arcsec"] = 0.01
    light = mapping["scene"]["source"]["light"]["light"]
    light.pop("intensity")
    light.update(centre=[0.02, -0.03], ell_comps=[0.1, 0.05], effective_radius=0.1, flux={"rate_e_per_s": 8.0})
    config = parse_config(mapping)
    setup = resolve_observing(config.scene, config.instrument, config.observation, truth=truth_provider(config))
    resolved = setup.scene.source.light[0]
    profile = al.lp.Exponential(**dict(resolved.values))
    grid = al.Grid2D.uniform(shape_native=(300, 300), pixel_scales=0.01, over_sample_size=8)
    rendered = np.asarray(profile.image_2d_from(grid=grid).native)
    assert rendered.sum() == pytest.approx(8.0, rel=1.0e-6)
    record = setup.photometry.components["source.light"]
    assert record["mapping_ratio"] == pytest.approx(rendered.sum()/8.0, rel=1.0e-12)
    assert record["rate_e_per_s"] == 8.0
    assert config.scene.source.light[0].values["intensity"] is None
    assert resolved.flux is None


def test_paper_amplitude_is_preserved_and_ab_inputs_reproduce_rates(minimal_mapping):
    from hwoslaps.config.schema import parse_config
    from hwoslaps.observation.normalization import resolve_observing
    from hwoslaps.fisher.psf_pair import truth_provider

    mapping = minimal_mapping
    config = parse_config(mapping)
    setup = resolve_observing(config.scene, config.instrument, config.observation, truth=truth_provider(config))
    assert setup.scene is config.scene
    assert setup.scene.source.light[0] is config.scene.source.light[0]
    assert setup.photometry is None
    mapping["instrument"].update(collecting_area_m2=33.606448937520405,
        bandpass={"kind": "top_hat", "min_nm": 450.0, "max_nm": 550.0, "throughput": 0.21})
    light = mapping["scene"]["source"]["light"]["light"]
    light.pop("intensity")
    light["flux"] = {"ab_mag": 24.845}
    mapping["observation"]["sky"] = {"ab_mag_per_arcsec2": 23.0}
    mapping["scene"]["grid"]["pixel_scale_arcsec"] = 0.00716
    mapping["psf"]["truth"]["pixel_scale_arcsec"] = 0.00716
    config = parse_config(mapping)
    setup = resolve_observing(config.scene, config.instrument, config.observation, truth=truth_provider(config))
    assert setup.photometry.components["source.light"]["rate_e_per_s"] == pytest.approx(8.951505744562876, rel=1.0e-12)
    assert setup.exposure.sky_rate_e_per_s == pytest.approx(0.002510279845963486, rel=1.0e-12)


def test_image_flux_normalization_includes_flux_and_size_scale(minimal_mapping, image_asset):
    import autolens as al
    from hwoslaps.config.schema import parse_config
    from hwoslaps.fisher.psf_pair import truth_provider
    from hwoslaps.observation.normalization import resolve_observing
    from hwoslaps.scene.image_profile import ImageLightProfile
    from hwoslaps.scene.image_source import load_image_asset

    mapping = minimal_mapping
    mapping["scene"]["grid"].update(shape=[250, 250], pixel_scale_arcsec=0.01, over_sample_size=8)
    mapping["psf"]["truth"]["pixel_scale_arcsec"] = 0.01
    mapping["scene"]["source"]["light"] = {"light": {"type": "Image", "asset_path": str(image_asset),
        "centre": [0.0, 0.0], "rotation_deg": 30.0, "flux_scale": 0.8, "size_scale": 1.3, "flux": {"rate_e_per_s": 7.0}}}
    config = parse_config(mapping)
    asset = load_image_asset(image_asset)
    setup = resolve_observing(config.scene, config.instrument, config.observation, truth=truth_provider(config), assets={str(image_asset): asset})
    values = dict(setup.scene.source.light[0].values)
    values.pop("asset_path")
    profile = ImageLightProfile.from_asset(asset, **values)
    grid = al.Grid2D.uniform(shape_native=(250, 250), pixel_scales=0.01, over_sample_size=8)
    rendered = np.asarray(profile.image_2d_from(grid=grid).native)
    assert rendered.sum() == pytest.approx(7.0, rel=5.0e-5)
    assert setup.photometry.components["source.light"]["mapping_ratio"] == pytest.approx(rendered.sum()/7.0, rel=1.0e-12)
    assert setup.file_digests[str(image_asset)] == asset.digest


def test_grouped_monochromatic_scene_keeps_real_flux_and_nuisance_paths(minimal_mapping):
    from hwoslaps.fisher.api import Execution, forecast, prepare_forecast

    mapping = minimal_mapping
    first = mapping["scene"]["source"]["light"]["light"]
    second = deepcopy(first)
    second.update(centre=[0.04, -0.06], intensity=0.3, effective_radius=0.18)
    mapping["scene"]["source"]["light"]["second"] = second
    with prepare_forecast(mapping) as plain:
        expected_mean = plain.mean_truth_adu.copy()
        expected = forecast(plain, masses_msun=[1.0e8])
    first["sed"] = {"kind": "flat_fnu"}
    second["sed"] = {"kind": "power_law", "index": -2.0}
    with prepare_forecast(mapping) as grouped:
        assert tuple(grouped.scene.light_groups) == ("source:light", "source:second")
        np.testing.assert_allclose(grouped.mean_truth_adu, expected_mean, rtol=1.0e-13, atol=1.0e-11)
        actual = forecast(grouped, masses_msun=[1.0e8])
        np.testing.assert_allclose(actual.fisher_profiled, expected.fisher_profiled, rtol=1.0e-10)
        assert "source.light.second.intensity" in grouped.nuisances.names
    with prepare_forecast(mapping, execution=Execution(engine="jax")) as grouped_jax:
        jax = forecast(grouped_jax, masses_msun=[1.0e8])
    np.testing.assert_allclose(jax.fisher_profiled, actual.fisher_profiled, rtol=5.0e-6)


def test_captured_table_seds_survive_file_removal_in_real_renderer_transport(minimal_mapping, tmp_path):
    from hwoslaps.fisher.api import prepare_forecast

    path = tmp_path / "sed.npz"
    np.savez(path, wave=[400.0, 500.0, 600.0], shape=[0.5, 1.0, 0.8])
    component = minimal_mapping["scene"]["source"]["light"]["light"]
    component["sed"] = {"kind": "table", "path": str(path), "wavelength_key": "wave", "value_key": "shape",
                        "wavelength_unit": "nm", "quantity": "fnu"}
    with prepare_forecast(minimal_mapping) as prepared:
        renderer = pickle.loads(pickle.dumps(prepared.renderer))
        expected = renderer.mean_adu(renderer.scene(), prepared.psfs.truth_kernels)
        path.unlink()
        actual = renderer.mean_adu(renderer.scene(), prepared.psfs.truth_kernels)
        np.testing.assert_array_equal(actual, expected)
        assert tuple(renderer.loaded_seds) == ("source.light",)
        with pytest.raises(TypeError):
            renderer.loaded_seds["source.light"] = None
        with pytest.raises(FileNotFoundError):
            prepared.validate_identity()


def test_loaded_spectral_publication_is_refused_before_resolution_render(minimal_mapping, tmp_path):
    import sys
    from hwoslaps.fisher.api import prepare_forecast
    from hwoslaps.identity import read_file_snapshot

    path = tmp_path / "sed.npz"
    np.savez(path, wave=[400.0, 500.0, 600.0], shape=[0.5, 1.0, 0.8])
    light = minimal_mapping["scene"]["source"]["light"]["light"]
    light.pop("intensity")
    light["flux"] = {"rate_e_per_s": 8.0}
    light["sed"] = {"kind": "table", "path": str(path), "wavelength_key": "wave", "value_key": "shape",
                    "wavelength_unit": "nm", "quantity": "fnu"}
    original = path.read_bytes()
    previous = sys.getprofile()
    published = []
    def publish(frame, event, returned):
        if event == "return" and frame.f_code is read_file_snapshot.__code__ and Path(frame.f_locals["path"]) == path:
            sys.setprofile(previous)
            np.savez(path, wave=[400.0, 500.0, 600.0], shape=[0.2, 1.0, 1.0])
            published.append(True)
    sys.setprofile(publish)
    try:
        # The already decoded old snapshot stays truthful; final preparation must reject the new disk epoch.
        with pytest.raises(ValueError, match=str(path)):
            prepare_forecast(minimal_mapping)
        assert published == [True]
    finally:
        sys.setprofile(previous)
        path.write_bytes(original)
