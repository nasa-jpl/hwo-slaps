"""Physics and correctness tests for observation generation."""

from __future__ import annotations

import copy

import autolens as al
import numpy as np
import pytest

from hwoslaps.lensing.utils import LensingData
from hwoslaps.observation import generate_observation
from hwoslaps.observation.noise_models import apply_detector_noise, create_noise_map
from hwoslaps.psf.utils import PSFData, make_pyauto_kernel


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
    kernel = make_pyauto_kernel(
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


def _source_snr_from_formula(
    source_eps: np.ndarray,
    exposure_time: float,
    detector: dict,
) -> np.ndarray:
    source_e = source_eps * exposure_time
    dark_e = detector["dark_current"] * exposure_time
    sky_e = detector["sky_background"] * exposure_time
    variance_e2 = source_e + dark_e + sky_e + detector["read_noise"]**2
    return source_e / np.sqrt(variance_e2)


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


def test_observation_source_snr_scales_with_exposure_depth():
    source_eps = np.array(
        [
            [0.2, 0.5, 1.0],
            [1.5, 2.0, 3.0],
            [0.4, 0.8, 1.2],
        ],
        dtype=float,
    )
    lensing = _make_lensing_data(image=source_eps, pixel_scale=0.1)
    psf_data = _make_psf_data(np.array([[1.0]]), pixel_scale=0.1)
    detector = {
        "gain": 2.0,
        "read_noise": 3.0,
        "dark_current": 0.01,
        "sky_background": 0.4,
    }

    snr_maps = []
    for exposure_time in (50.0, 200.0, 800.0):
        obs = generate_observation(
            lensing_data=lensing,
            psf_data=psf_data,
            observation_config=_observation_config(
                exposure_time=exposure_time,
                **detector,
            ),
            full_config={
                "global_seed": int(exposure_time),
                "run_name": f"exposure_{exposure_time:g}",
            },
        )
        expected_snr = _source_snr_from_formula(source_eps, exposure_time, detector)

        np.testing.assert_allclose(
            obs.signal_to_noise_map.native,
            expected_snr,
            rtol=1e-12,
            atol=1e-12,
        )
        assert obs.peak_snr == pytest.approx(float(np.max(expected_snr)))
        snr_maps.append(obs.signal_to_noise_map.native)

    assert np.all(snr_maps[1] > snr_maps[0])
    assert np.all(snr_maps[2] > snr_maps[1])


def test_detector_noise_monte_carlo_matches_expected_moments():
    source_eps = np.full((128, 128), 2.0, dtype=float)
    exposure_time = 50.0
    detector = {
        "gain": 2.0,
        "read_noise": 4.0,
        "dark_current": 0.1,
        "sky_background": 1.0,
    }
    expected_e = (
        source_eps[0, 0]
        + detector["dark_current"]
        + detector["sky_background"]
    ) * exposure_time
    expected_mean_adu = expected_e / detector["gain"]
    expected_variance_adu2 = (
        expected_e
        + detector["read_noise"]**2
    ) / detector["gain"]**2

    final_image_adu, components = apply_detector_noise(
        source_eps,
        exposure_time,
        detector,
        seed=123,
    )

    samples = final_image_adu.ravel()
    assert float(np.mean(samples)) == pytest.approx(expected_mean_adu, abs=0.35)
    assert float(np.var(samples, ddof=1)) == pytest.approx(
        expected_variance_adu2,
        rel=0.06,
    )
    np.testing.assert_allclose(
        components["expected_e"],
        np.full_like(source_eps, expected_e),
        rtol=0.0,
        atol=0.0,
    )


def test_generate_observation_seed_controls_noise_only():
    lensing = _make_lensing_data(shape=(9, 9), pixel_scale=0.1)
    psf_data = _make_psf_data(np.array([[1.0]]), pixel_scale=0.1)
    observation_config = _observation_config(
        exposure_time=125.0,
        gain=1.5,
        read_noise=0.7,
        dark_current=0.01,
        sky_background=0.8,
    )

    obs_a = generate_observation(
        lensing_data=lensing,
        psf_data=psf_data,
        observation_config=observation_config,
        full_config={"global_seed": 55, "run_name": "seed_a"},
    )
    obs_b = generate_observation(
        lensing_data=lensing,
        psf_data=psf_data,
        observation_config=observation_config,
        full_config={"global_seed": 55, "run_name": "seed_b"},
    )
    obs_c = generate_observation(
        lensing_data=lensing,
        psf_data=psf_data,
        observation_config=observation_config,
        full_config={"global_seed": 56, "run_name": "seed_c"},
    )

    np.testing.assert_allclose(obs_a.data.native, obs_b.data.native, rtol=0.0, atol=0.0)
    assert not np.array_equal(obs_a.data.native, obs_c.data.native)
    np.testing.assert_allclose(
        obs_a.noiseless_source_eps,
        obs_c.noiseless_source_eps,
        rtol=0.0,
        atol=0.0,
    )
    np.testing.assert_allclose(obs_a.noise_map.native, obs_c.noise_map.native, rtol=0.0, atol=0.0)


def test_generate_observation_nontrivial_psf_convolution_centers_kernel():
    source_eps = np.zeros((7, 7), dtype=float)
    source_eps[3, 3] = 1.0
    kernel = np.array(
        [
            [0.00, 0.05, 0.00],
            [0.10, 0.50, 0.20],
            [0.00, 0.15, 0.00],
        ],
        dtype=float,
    )
    lensing = _make_lensing_data(image=source_eps, pixel_scale=0.1)
    psf_data = _make_psf_data(kernel, pixel_scale=0.1)

    obs = generate_observation(
        lensing_data=lensing,
        psf_data=psf_data,
        observation_config=_observation_config(
            exposure_time=100.0,
            gain=1.0,
            read_noise=0.1,
            dark_current=0.0,
            sky_background=0.0,
        ),
        full_config={"global_seed": 9, "run_name": "nontrivial_psf"},
    )

    expected = np.zeros_like(source_eps)
    expected[2:5, 2:5] = kernel
    np.testing.assert_allclose(obs.noiseless_source_eps, expected, rtol=0.0, atol=1e-12)
    assert float(np.sum(obs.noiseless_source_eps)) == pytest.approx(1.0)


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
