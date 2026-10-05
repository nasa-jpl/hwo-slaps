"""FitData owners: detector units, copied kernels, include masks and actual grid sampling."""

import numpy as np
import pytest

from hwoslaps.inference.data import build_fit_data
from hwoslaps.identity import json_ready

pytestmark = pytest.mark.backend


@pytest.mark.parametrize("kind", ["expected", "noisy"])
def test_fit_data_units_and_background_follow_the_observation_kind(kind, prepared_forecast_factory):
    prepared = prepared_forecast_factory({"instrument": {"detector": {"gain_e_per_adu": 2.5}}})
    observation = prepared.observation if kind == "expected" else prepared.observation.draw(11)
    data = build_fit_data(observation, prepared.psfs.model_kernels.single, mask_name="all_pixels_minus_psf_border",
                          base_mask=np.ones((40, 40), dtype=bool), over_sample_size=2)
    exposure = observation.exposure
    gain, time = 2.5, exposure.exposure_time_s
    background = ((exposure.sky_rate_e_per_s * time + exposure.detector.dark_current_e_per_s * time) / gain)
    expected = observation.light_rate_e_per_s if kind == "expected" else (
        observation.data_adu * gain / time - background * gain / time)
    np.testing.assert_array_equal(np.asarray(data.imaging.data), expected[data.mask])
    np.testing.assert_array_equal(np.asarray(data.imaging.noise_map), (observation.noise_map_adu * gain / time)[data.mask])
    assert data.kind == kind and data.record["background_adu"] == (0.0 if kind == "expected" else background)
    assert json_ready(data.record["truth_kernels"]) == observation.psfs.to_mapping()


@pytest.mark.parametrize("base", ["all", "forecast", "custom"])
def test_fitted_pixels_remove_the_kernel_border_from_each_base_mask(base, prepared_forecast):
    include = np.ones((40, 40), dtype=bool)
    if base == "forecast":
        include[10:13, 15:20] = False
    elif base == "custom":
        include[:, 20:] = False
    expected = include.copy()
    expected[:3] = expected[-3:] = False
    expected[:, :3] = expected[:, -3:] = False
    data = build_fit_data(prepared_forecast.observation, prepared_forecast.psfs.model_kernels.single,
                          mask_name=base, base_mask=include, over_sample_size=2)
    np.testing.assert_array_equal(data.mask, expected)
    assert data.pixel_count == int(expected.sum()) and data.record["mask"]["name"] == base
    assert not data.mask.flags.writeable


def test_building_fit_data_leaves_the_prepared_kernel_unchanged(prepared_forecast_factory, tmp_path):
    import autolens as al

    # Public external-kernel inputs may be unit-sum within tolerance rather than exactly one.
    values = np.zeros((7, 7), dtype=np.float64)
    values[3, 3] = 1.0 + 2.0 ** -45
    path = tmp_path / "near_unit_kernel.npy"
    np.save(path, values)
    prepared = prepared_forecast_factory({"psf": {"model": {"kind": "kernel", "path": str(path),
                                                          "pixel_scale_arcsec": 0.05, "normalize": False}}})
    kernel = prepared.psfs.model_kernels.single
    assert float(kernel.kernel.sum()) == 1.0 + 2.0 ** -45
    before = kernel.kernel.tobytes()
    convolver_before = np.asarray(kernel.convolver().kernel.native).tobytes()
    fitted = build_fit_data(prepared.observation, kernel, mask_name="all_pixels_minus_psf_border",
                            base_mask=np.ones((40, 40), dtype=bool), over_sample_size=2)
    assert kernel.kernel.tobytes() == before
    assert fitted.record["fitted_kernel"] != fitted.record["model_kernel"]
    prepared.validate_identity()
    # The pre-copy control uses the real dataset constructor and the actual cached convolver.
    al.Imaging(data=fitted.imaging.data, noise_map=fitted.imaging.noise_map, psf=kernel.convolver(),
               over_sample_size_lp=2)
    assert np.asarray(kernel.convolver().kernel.native).tobytes() != convolver_before
    with pytest.raises(ValueError, match="model convolver kernel"):
        prepared.validate_identity()

@pytest.mark.parametrize("size", [1, 2, 4])
def test_dataset_samples_light_and_blurring_grids_at_generation_size(size, prepared_forecast_factory):
    prepared = prepared_forecast_factory({"scene": {"grid": {"over_sample_size": size}}})
    fitted = build_fit_data(prepared.observation, prepared.psfs.model_kernels.single, mask_name="all",
                            base_mask=np.ones((40, 40), dtype=bool), over_sample_size=size)
    assert fitted.record["over_sample_size"] == {"generation": size, "light_profile": size, "blurring": size}
    for grid in (fitted.imaging.grids.lp, fitted.imaging.grids.blurring):
        np.testing.assert_array_equal(np.unique(np.asarray(grid.over_sample_size)), [size])
