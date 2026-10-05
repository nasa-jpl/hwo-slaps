"""The exposure's detector response and the detector noise draw, against hand equations."""

import math

import numpy as np
import pytest

from hwoslaps.instrument import Detector
from hwoslaps.observation.expected import Exposure
from hwoslaps.observation.noise import draw_noisy_adu


@pytest.mark.parametrize("count, read_variance", [(1, 9.0), (3, 27.0)])
def test_detector_response_matches_hand_equations(count, read_variance):
    # Binary-fraction inputs, so every expected value is an exact literal.
    rate = np.array([0.0, 0.25, 2.0, 7.0, -1e-12])
    exposure = Exposure(Detector(gain_e_per_adu=2.0, read_noise_e=3.0, dark_current_e_per_s=0.125),
                        exposure_time_s=8.0, sky_rate_e_per_s=0.5, exposure_count=count)

    assert (exposure.sky_e, exposure.dark_e) == (4.0, 1.0)
    assert exposure.read_variance_e2 == read_variance
    assert exposure.read_sigma_e == pytest.approx(math.sqrt(read_variance), rel=1e-15)
    assert exposure.background_adu == 2.5
    assert exposure.blank_variance_e2 == 5.0 + read_variance
    counts = np.array([5.0, 7.0, 21.0, 61.0, 5.0])
    np.testing.assert_array_equal(exposure.counts_e(rate), counts)
    np.testing.assert_array_equal(exposure.variance_e2(rate), counts + read_variance)
    np.testing.assert_array_equal(exposure.noise_map_adu(rate), np.sqrt(counts + read_variance) / 2.0)
    mean = exposure.mean_adu(rate)
    np.testing.assert_array_equal(mean[:4], [2.5, 3.5, 10.5, 30.5])
    assert mean[4] < 2.5  # the mean keeps a negative rate; only the Poisson mean clamps it
    np.testing.assert_array_equal(exposure.signal_e(rate)[:4], [0.0, 2.0, 16.0, 56.0])
    assert exposure.signal_adu(7.0) == 28.0
    assert exposure.rate_from_adu(28.0) == 7.0


@pytest.mark.parametrize("count, read_sigma", [(1, 3.0), (4, 6.0)])
def test_noise_draw_is_numpy_poisson_then_normal(count, read_sigma):
    exposure = Exposure(Detector(2.0, 3.0, 0.1), exposure_time_s=7.0, sky_rate_e_per_s=0.5,
                        exposure_count=count)
    counts = np.arange(12, dtype=float).reshape(3, 4) + 5.0
    rng = np.random.default_rng(812)
    detected = rng.poisson(counts).astype(float)
    expected = (detected + rng.normal(0.0, read_sigma, size=counts.shape)) / 2.0

    np.random.seed(12345)
    reference_global_draws = np.random.random(5)
    np.random.seed(12345)
    drawn = draw_noisy_adu(counts, exposure, 812)

    np.testing.assert_array_equal(drawn, expected)
    np.testing.assert_array_equal(np.random.random(5), reference_global_draws)


def test_noise_moments_match_poisson_plus_read_variance():
    exposure = Exposure(Detector(2.0, 4.0, 0.1), exposure_time_s=50.0, sky_rate_e_per_s=1.0,
                        exposure_count=3)
    counts = exposure.counts_e(np.full((128, 128), 2.0))
    assert np.all(counts == 155.0)

    samples = draw_noisy_adu(counts, exposure, 123).ravel()

    assert float(np.mean(samples)) == pytest.approx(155.0 / 2.0, abs=0.35)
    assert float(np.var(samples, ddof=1)) == pytest.approx((155.0 + 3 * 16.0) / 2.0**2, rel=0.06)


def test_zero_variance_pixels_are_refused():
    with pytest.raises(ValueError, match="zero variance"):
        Exposure(Detector(1.0, 0.0, 0.0), exposure_time_s=10.0, sky_rate_e_per_s=0.0)
    for read_noise, dark, sky in [(0.1, 0.0, 0.0), (0.0, 0.1, 0.0), (0.0, 0.0, 0.1)]:
        exposure = Exposure(Detector(1.0, read_noise, dark), exposure_time_s=10.0, sky_rate_e_per_s=sky)
        assert exposure.blank_variance_e2 > 0.0
