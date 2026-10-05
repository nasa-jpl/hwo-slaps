"""Flux-to-amplitude normalization, real grouped monochromatic renders and spectral ownership."""

from copy import deepcopy
from pathlib import Path
import math
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
    from hwoslaps.fisher.api import Execution, prepare_forecast

    path = tmp_path / "sed.npz"
    np.savez(path, wave=[400.0, 500.0, 600.0], shape=[0.5, 1.0, 0.8])
    component = minimal_mapping["scene"]["source"]["light"]["light"]
    component["sed"] = {"kind": "table", "path": str(path), "wavelength_key": "wave", "value_key": "shape",
                        "wavelength_unit": "nm", "quantity": "fnu"}
    second = deepcopy(component)
    second.update(centre=[0.04, -0.06], intensity=0.3)
    second["sed"]["quantity"] = "flambda"
    minimal_mapping["scene"]["source"]["light"]["second"] = second
    with prepare_forecast(minimal_mapping) as serial, prepare_forecast(
            minimal_mapping, execution=Execution(reference_workers=2)) as prepared:
        renderer = pickle.loads(pickle.dumps(prepared.renderer))
        assert tuple(serial.scene.light_groups) == ("source:light", "source:second")
        expected = serial.renderer.mean_adu(serial.scene, serial.psfs.truth_kernels)
        expected_bank = serial.engine.evaluate(serial.positions.positions_yx, [1.0e8])[0]
        pooled_bank = prepared.engine.evaluate(prepared.positions.positions_yx, [1.0e8])[0]
        np.testing.assert_array_equal(pooled_bank.fisher_raw, expected_bank.fisher_raw)
        np.testing.assert_array_equal(pooled_bank.fisher_profiled, expected_bank.fisher_profiled)
        path.unlink()
        actual = renderer.mean_adu(renderer.scene(), prepared.psfs.truth_kernels)
        actual_bank = prepared.engine.evaluate(prepared.positions.positions_yx, [1.0e8])[0]
        np.testing.assert_array_equal(actual, expected)
        np.testing.assert_array_equal(actual_bank.fisher_raw, expected_bank.fisher_raw)
        np.testing.assert_array_equal(actual_bank.fisher_profiled, expected_bank.fisher_profiled)
        assert tuple(renderer.loaded_seds) == ("source.light", "source.second")
        with pytest.raises(TypeError):
            renderer.loaded_seds["source.light"] = None
        with pytest.raises(FileNotFoundError):
            prepared.validate_identity()


def test_new_spectral_epoch_is_refused_before_diagnostic_render(minimal_mapping, tmp_path):
    import sys
    from hwoslaps.fisher.api import prepare_forecast
    from hwoslaps.identity import file_digest
    from hwoslaps.scene.builder import render_component_unlensed

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
    renders = []
    def publish(frame, event, returned):
        if event == "call" and frame.f_code is render_component_unlensed.__code__:
            renders.append(True)
        if not published and event == "return" and frame.f_code is file_digest.__code__ and Path(frame.f_locals["path"]) == path:
            np.savez(path, wave=[400.0, 500.0, 600.0], shape=[0.2, 1.0, 1.0])
            published.append(True)
    sys.setprofile(publish)
    try:
        with pytest.raises(ValueError, match=str(path)):
            prepare_forecast(minimal_mapping)
        assert published == [True]
        assert not renders
    finally:
        sys.setprofile(previous)
        path.write_bytes(original)


def test_reference_band_flux_resolves_the_sed_colour_at_the_scene_boundary(minimal_mapping):
    from hwoslaps.config.schema import parse_config
    from hwoslaps.fisher.psf_pair import truth_provider
    from hwoslaps.observation.normalization import resolve_observing

    minimal_mapping["instrument"].update(collecting_area_m2=33.6,
        bandpass={"kind": "top_hat", "min_nm": 450.0, "max_nm": 550.0, "throughput": 0.21})
    component = minimal_mapping["scene"]["source"]["light"]["light"]
    component.pop("intensity")
    component["sed"] = {"kind": "power_law", "index": -1.5}
    component["flux"] = {"ab_mag": 24.3,
        "reference_band": {"kind": "top_hat", "min_nm": 700.0, "max_nm": 900.0, "throughput": 0.8}}
    config = parse_config(minimal_mapping)
    setup = resolve_observing(config.scene, config.instrument, config.observation, truth=truth_provider(config))
    beta = 1.5
    reference_mean = (900.0**beta-700.0**beta)/(beta*math.log(900.0/700.0))
    instrument_integral = 0.21*(550.0**beta-450.0**beta)/beta
    expected = 33.6*3631.0*10.0**(-0.4*24.3)*1.0e-26/6.62607015e-34*instrument_integral/reference_mean
    assert setup.photometry.components["source.light"]["rate_e_per_s"] == pytest.approx(expected, rel=1.0e-9)
    assert tuple(setup.loaded_seds) == ("source.light",)


def test_optical_truth_area_resolves_pinned_paper_photometry(minimal_mapping):
    from hwoslaps.config.schema import parse_config
    from hwoslaps.fisher.psf_pair import truth_provider
    from hwoslaps.observation.normalization import resolve_observing

    minimal_mapping["scene"]["grid"]["pixel_scale_arcsec"] = 0.00716
    minimal_mapping["psf"] = {"truth": {"kind": "optical",
        "pupil": {"kind": "hex_segmented", "diameter_m": 7.225765, "pixels": 512, "supersampling": 4,
                  "rings": 2, "segment_point_to_point_m": 1.65, "gap_m": 0.006},
        "focal_length_m": 144.0, "wavelength_nm": 500.0, "detector_oversampling": 3, "kernel_shape": [17, 17]}}
    minimal_mapping["instrument"]["bandpass"] = {"kind": "top_hat", "min_nm": 450.0, "max_nm": 550.0, "throughput": 0.21}
    component = minimal_mapping["scene"]["source"]["light"]["light"]
    component.pop("intensity")
    component["flux"] = {"ab_mag": 24.845}
    minimal_mapping["observation"]["sky"] = {"ab_mag_per_arcsec2": 23.0}
    config = parse_config(minimal_mapping)
    setup = resolve_observing(config.scene, config.instrument, config.observation, truth=truth_provider(config))
    assert setup.instrument.collecting_area_source == "optical_pupil"
    assert setup.instrument.collecting_area_m2 == pytest.approx(33.606448937520405, rel=1.0e-12)
    assert setup.photometry.components["source.light"]["rate_e_per_s"] == pytest.approx(8.951505744562876, rel=1.0e-12)
    assert setup.exposure.sky_rate_e_per_s == pytest.approx(0.002510279845963486, rel=1.0e-12)


@pytest.mark.parametrize("product", ["forecast", "simulation"])
def test_single_node_cube_loaded_b_after_captured_a_is_refused_when_disk_returns_to_a(
        minimal_mapping, tiny_gaussian_kernel, tmp_path, product):
    import sys
    from hwoslaps.fisher.api import prepare_forecast
    from hwoslaps.identity import file_digest, read_file_snapshot
    from hwoslaps.simulation import simulate

    path = tmp_path / "cube.npz"
    np.savez(path, kernels=tiny_gaussian_kernel[None, ...], wavelengths_m=np.array([5.0e-7]))
    original = path.read_bytes()
    different = tiny_gaussian_kernel.copy()
    different[3, 3] *= 1.4
    different /= different.sum()
    minimal_mapping["psf"] = {"truth": {"kind": "kernel_cube", "path": str(path),
        "pixel_scale_arcsec": 0.05, "normalize": False}}
    previous = sys.getprofile()
    transitions = []
    def publish_and_restore(frame, event, returned):
        if event != "return":
            return
        if not transitions and frame.f_code is file_digest.__code__ and Path(frame.f_locals["path"]) == path:
            np.savez(path, kernels=different[None, ...], wavelengths_m=np.array([5.0e-7]))
            transitions.append("published_b")
        elif transitions == ["published_b"] and frame.f_code is read_file_snapshot.__code__ and Path(frame.f_locals["path"]) == path:
            sys.setprofile(previous)
            path.write_bytes(original)
            transitions.append("restored_a")
    sys.setprofile(publish_and_restore)
    try:
        with pytest.raises(ValueError, match=str(path)):
            if product == "forecast":
                prepare_forecast(minimal_mapping)
            else:
                simulate(minimal_mapping, subhalo=None, noise_seed=None)
        assert transitions == ["published_b", "restored_a"]
    finally:
        sys.setprofile(previous)
        path.write_bytes(original)
