"""Physics and correctness tests for observation generation."""

from __future__ import annotations

import copy

import autolens as al
import numpy as np
import pytest

from hwoslaps.lensing.utils import LensingData
from hwoslaps.observation import generate_observation
from hwoslaps.observation.noise_models import apply_detector_noise, create_noise_map
from hwoslaps.psf.utils import PSFData


def _make_lensing_data(
    image: np.ndarray | None = None,
    *,
    shape: tuple[int, int] = (9, 9),
    pixel_scale: float = 0.1,
) -> LensingData:
    if image is None:
        y, x = np.indices(shape)
        image = ((y + 1) * (x + 1)).astype(float) / 100.0
    else:
        image = np.asarray(image, dtype=float)
        shape = image.shape

    grid = al.Grid2D.uniform(
        shape_native=shape,
        pixel_scales=pixel_scale,
        over_sample_size=1,
    )

    return LensingData(
        image=image,
        grid=grid,
        tracer=al.Tracer(galaxies=[]),
        pixel_scale=pixel_scale,
        lens_redshift=0.5,
        source_redshift=1.0,
        lens_einstein_radius=1.0,
        cosmology_name="Planck15",
        config={},
    )


def _make_psf_data(kernel_values: np.ndarray, *, pixel_scale: float = 0.1) -> PSFData:
    kernel = al.Kernel2D.no_mask(
        values=np.asarray(kernel_values, dtype=float),
        pixel_scales=pixel_scale,
        normalize=False,
    )
    return PSFData(
        psf=None,
        wavefront=None,
        telescope_data={},
        kernel=kernel,
        kernel_pixel_scale=pixel_scale,
        wavelength_nm=550.0,
        pupil_diameter_m=6.0,
        focal_length_m=120.0,
        pixel_scale_arcsec=pixel_scale,
        sampling_factor=2.0,
        requested_sampling_factor=2.0,
        used_sampling_factor=2.0,
        integer_subsampling_factor=1,
        num_segments=1,
        segment_flat_to_flat_m=1.0,
        segment_point_to_point_m=1.0,
        gap_size_m=0.0,
        num_rings=0,
        config={},
    )


def _observation_config(
    *,
    exposure_time: float = 100.0,
    gain: float = 1.0,
    read_noise: float = 0.2,
    dark_current: float = 0.002,
    sky_background: float = 0.5,
) -> dict:
    return {
        "exposure_time": exposure_time,
        "detector": {
            "gain": gain,
            "read_noise": read_noise,
            "dark_current": dark_current,
            "sky_background": sky_background,
        },
    }


def test_create_noise_map_matches_ccd_variance_formula():
    source_eps = np.array([[0.0, 1.0], [2.0, 3.0]])
    exposure_time = 7.0
    detector = {
        "gain": 2.0,
        "read_noise": 3.0,
        "dark_current": 0.1,
        "sky_background": 0.5,
    }

    expected = (
        np.sqrt(
            source_eps * exposure_time
            + detector["dark_current"] * exposure_time
            + detector["sky_background"] * exposure_time
            + detector["read_noise"] ** 2
        )
        / detector["gain"]
    )

    np.testing.assert_allclose(
        create_noise_map(source_eps, exposure_time, detector),
        expected,
        rtol=0.0,
        atol=0.0,
    )


@pytest.mark.parametrize(
    "detector_update",
    [
        {"gain": 0.0},
        {"gain": -1.0},
        {"read_noise": -0.1},
        {"dark_current": -0.01},
        {"sky_background": -1.0},
    ],
)
@pytest.mark.parametrize("noise_function", [create_noise_map, apply_detector_noise])
def test_noise_functions_reject_nonphysical_detector_domain(detector_update, noise_function):
    source_eps = np.ones((2, 2), dtype=float)
    detector = {
        "gain": 1.0,
        "read_noise": 0.2,
        "dark_current": 0.002,
        "sky_background": 0.5,
    }
    detector.update(detector_update)

    with pytest.raises(ValueError):
        noise_function(source_eps, 100.0, detector)


def test_apply_detector_noise_uses_local_rng_without_mutating_global_state():
    source_eps = np.ones((4, 4), dtype=float)
    detector = {
        "gain": 1.0,
        "read_noise": 0.2,
        "dark_current": 0.002,
        "sky_background": 0.5,
    }

    np.random.seed(12345)
    expected_after = np.random.random(5)

    np.random.seed(12345)
    apply_detector_noise(source_eps, 100.0, detector, seed=11)
    after = np.random.random(5)

    np.testing.assert_allclose(after, expected_after, rtol=0.0, atol=0.0)


def test_generate_observation_source_free_scene_has_zero_source_snr():
    lensing = _make_lensing_data(image=np.zeros((5, 5)), pixel_scale=0.1)
    psf_data = _make_psf_data(np.array([[1.0]]), pixel_scale=0.1)

    obs = generate_observation(
        lensing_data=lensing,
        psf_data=psf_data,
        observation_config=_observation_config(
            exposure_time=100.0,
            gain=1.0,
            read_noise=0.0,
            dark_current=0.0,
            sky_background=1.0,
        ),
        full_config={"global_seed": 0, "run_name": "source_free_snr"},
    )

    assert np.max(obs.source_electrons) == pytest.approx(0.0)
    assert np.max(obs.signal_to_noise_map.native) == pytest.approx(0.0)
    assert obs.peak_snr == pytest.approx(0.0)


def test_generate_observation_rejects_even_psf_kernel_instead_of_trimming():
    lensing = _make_lensing_data(shape=(9, 9), pixel_scale=0.1)
    psf_data = _make_psf_data(np.ones((4, 4)), pixel_scale=0.1)

    with pytest.raises(ValueError, match="odd"):
        generate_observation(
            lensing_data=lensing,
            psf_data=psf_data,
            observation_config=_observation_config(),
            full_config={"global_seed": 1, "run_name": "even_kernel"},
        )


def test_generate_observation_rejects_unnormalized_odd_psf_kernel():
    lensing = _make_lensing_data(shape=(9, 9), pixel_scale=0.1)
    psf_data = _make_psf_data(np.array([[2.0]]), pixel_scale=0.1)

    with pytest.raises(ValueError, match="normal"):
        generate_observation(
            lensing_data=lensing,
            psf_data=psf_data,
            observation_config=_observation_config(),
            full_config={"global_seed": 1, "run_name": "unnormalized_kernel"},
        )


def test_generate_observation_deep_copies_observation_config_provenance():
    lensing = _make_lensing_data(shape=(9, 9), pixel_scale=0.1)
    psf_data = _make_psf_data(np.array([[1.0]]), pixel_scale=0.1)
    observation_config = _observation_config(gain=2.0)

    obs = generate_observation(
        lensing_data=lensing,
        psf_data=psf_data,
        observation_config=observation_config,
        full_config={"global_seed": 2, "run_name": "config_copy"},
    )
    observation_config["detector"]["gain"] = 99.0

    assert obs.config["detector"]["gain"] == pytest.approx(2.0)
    assert obs.metadata["detector"]["gain"] == pytest.approx(2.0)


@pytest.mark.parametrize(
    "full_config,match",
    [
        (None, "full_config"),
        ({"run_name": "missing_seed"}, "global_seed"),
        ({"global_seed": 1}, "run_name"),
    ],
)
def test_generate_observation_reports_missing_full_config_requirements(full_config, match):
    lensing = _make_lensing_data(shape=(9, 9), pixel_scale=0.1)
    psf_data = _make_psf_data(np.array([[1.0]]), pixel_scale=0.1)

    with pytest.raises(ValueError, match=match):
        generate_observation(
            lensing_data=lensing,
            psf_data=psf_data,
            observation_config=_observation_config(),
            full_config=full_config,
        )
