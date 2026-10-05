"""Reference and JAX rendering conventions at the prepared forecast boundary."""

from copy import deepcopy

import numpy as np
import pytest

pytestmark = pytest.mark.backend
SCENES = ("matched", "kernel_mismatch", "image", "lens_light", "two_source_components")


def scene_mapping(mapping, scenario, image_asset, tmp_path, tiny_gaussian_kernel):
    if scenario == "kernel_mismatch":
        kernel = tiny_gaussian_kernel.copy()
        kernel[3, 3] *= 1.1
        path = tmp_path / "model.npy"
        np.save(path, kernel / kernel.sum())
        mapping["psf"]["model"] = {"kind": "kernel", "path": str(path), "pixel_scale_arcsec": 0.05}
    elif scenario == "image":
        mapping["scene"]["source"]["light"] = {"light": {"type": "Image", "asset_path": str(image_asset),
            "centre": [-0.03, 0.08], "flux_scale": 1.0, "size_scale": 1.2, "rotation_deg": 10.0, "total_flux": 1.0}}
    elif scenario == "lens_light":
        mapping["scene"]["lens"]["light"] = {"bulge": {"type": "Exponential", "centre": [0.0, 0.0],
            "ell_comps": [0.07, 0.03], "intensity": 0.8, "effective_radius": 0.4}}
    elif scenario == "two_source_components":
        second = deepcopy(mapping["scene"]["source"]["light"]["light"])
        second.update(centre=[0.04, -0.06], effective_radius=0.18, intensity=0.3, ell_comps=[0.08, 0.03])
        mapping["scene"]["source"]["light"]["second"] = second
    return mapping


def assert_engines(mapping):
    from hwoslaps.fisher.api import Execution, forecast, prepare_forecast

    with prepare_forecast(mapping) as reference, prepare_forecast(mapping, execution=Execution(engine="jax")) as jax:
        expected = forecast(reference, masses_msun=[1.0e8, 3.0e8])
        actual = forecast(jax, masses_msun=[1.0e8, 3.0e8])
    for name in ("fisher_raw", "fisher_profiled", "amplitude_hat", "amplitude_spurious"):
        if getattr(expected, name) is not None:
            np.testing.assert_allclose(getattr(actual, name), getattr(expected, name), rtol=5.0e-6, atol=0.0)


@pytest.mark.parametrize("scenario", SCENES)
def test_jax_engine_matches_reference(minimal_mapping, scenario, image_asset, tmp_path, tiny_gaussian_kernel):
    assert_engines(scene_mapping(minimal_mapping, scenario, image_asset, tmp_path, tiny_gaussian_kernel))


@pytest.mark.xtx_gpu
@pytest.mark.parametrize("scenario", SCENES)
def test_jax_gpu_engine_matches_reference(minimal_mapping, scenario, image_asset, tmp_path, tiny_gaussian_kernel):
    assert_engines(scene_mapping(minimal_mapping, scenario, image_asset, tmp_path, tiny_gaussian_kernel))


@pytest.mark.parametrize("engine", ["reference", "jax"])
def test_mass_sequence_rows_equal_fresh_evaluations(minimal_mapping, engine):
    from hwoslaps.fisher.api import Execution, forecast, prepare_forecast

    execution = Execution(engine=engine)
    masses = [3.0e8, 1.0e8, 3.0e8]
    with prepare_forecast(minimal_mapping, execution=execution) as prepared:
        rows = forecast(prepared, masses_msun=masses)
    for index, mass in enumerate(masses):
        with prepare_forecast(minimal_mapping, execution=execution) as fresh:
            expected = forecast(fresh, masses_msun=[mass])
        np.testing.assert_array_equal(rows.fisher_profiled[index], expected.fisher_profiled[0])


def test_light_evaluator_refuses_unsupported_profile():
    import autogalaxy as ag
    from hwoslaps.fisher.engines.jax_profiles import build_light_evaluator

    with pytest.raises(ValueError, match="Gaussian"):
        build_light_evaluator(ag.lp.Gaussian(), np.array([[0.1, 0.1], [0.2, 0.3]]))


def test_fast_fft_length_is_smallest_seven_smooth_bound():
    from hwoslaps.fisher.engines.fft import next_fast_length

    def smooth(value):
        for factor in (2, 3, 5, 7):
            while value % factor == 0:
                value //= factor
        return value == 1
    for target in [1, 11, 17, 31, 73, 107, 257, 499]:
        expected = next(value for value in range(target, 2 * target + 1) if smooth(value))
        assert next_fast_length(target) == expected


def test_padded_fft_convolution_matches_direct_real_space():
    from scipy.signal import convolve2d
    from hwoslaps.fisher.engines.fft import convolution_fft_shape

    image = np.random.default_rng(31).normal(size=(11, 13))
    kernel = np.array([[0.0, 0.1, 0.0], [0.2, 0.4, 0.2], [0.0, 0.1, 0.0]])
    shape = convolution_fft_shape(image.shape, kernel.shape)
    full = np.fft.irfft2(np.fft.rfft2(image, s=shape) * np.fft.rfft2(kernel, s=shape), s=shape)
    actual = full[1:12, 1:14]
    np.testing.assert_allclose(actual, convolve2d(image, kernel, mode="same"), rtol=1.0e-12, atol=1.0e-12)


def test_image_evaluator_refuses_reference_convention_drift(image_asset):
    from hwoslaps.fisher.engines.jax_profiles import build_light_evaluator
    from hwoslaps.scene.image_source import load_image_asset
    from hwoslaps.scene.image_profile import ImageLightProfile

    class ShiftedConvention(ImageLightProfile):
        def image_2d_from(self, grid, **kwargs):
            return 1.01 * super().image_2d_from(grid=grid, **kwargs)

    profile = ShiftedConvention.from_asset(load_image_asset(image_asset), centre=(0.0, 0.0), rotation_deg=0.0,
                                          total_flux=1.0, flux_scale=1.0, size_scale=1.0)
    with pytest.raises(ValueError, match="convention drift"):
        build_light_evaluator(profile, np.array([[0.0, 0.0], [0.01, 0.02], [-0.01, 0.0]]))
