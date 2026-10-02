"""Dependency-light detector contracts shared by forecasts and simulations."""

from types import MappingProxyType

import numpy as np
import pytest

from hwoslaps.observation import (
    apply_detector_noise,
    create_noise_map,
    detector_moments,
)


DETECTOR = {'gain': 2.0, 'read_noise': 3.0, 'dark_current': 0.1, 'sky_background': 0.5}


def test_detector_moments_match_independent_ccd_equations():
    source = np.array([[0.0, 0.2], [2.0, 7.0]])
    exposure = 7.0
    moments = detector_moments(source, exposure, MappingProxyType(DETECTOR))
    expected = source*exposure + 0.1*exposure + 0.5*exposure
    variance = expected + 3.0**2
    np.testing.assert_array_equal(moments.source_e, source*exposure)
    np.testing.assert_array_equal(moments.expected_e, expected)
    np.testing.assert_array_equal(moments.variance_e2, variance)
    np.testing.assert_array_equal(create_noise_map(source, exposure, DETECTOR), np.sqrt(variance)/2.0)


def test_noise_draw_preserves_exact_legacy_random_sequence():
    source = np.arange(12, dtype=float).reshape(3, 4)/7
    exposure = 7.0
    expected = source*exposure + 0.1*exposure + 0.5*exposure
    reference_rng = np.random.default_rng(812)
    detected = reference_rng.poisson(expected).astype(float)
    final_e = detected + reference_rng.normal(0.0, 3.0, size=source.shape)
    result, components = apply_detector_noise(source, exposure, DETECTOR, seed=812)
    np.testing.assert_array_equal(result, final_e/2.0)
    np.testing.assert_array_equal(components['detected_e'], detected)
    np.testing.assert_array_equal(components['expected_e'], expected)


def test_caller_owned_rng_matches_seed_path_and_advances_its_stream():
    source = np.ones((3, 4))
    rng = np.random.default_rng(812)
    result, _ = apply_detector_noise(source, 7.0, DETECTOR, rng=rng)
    seeded, _ = apply_detector_noise(source, 7.0, DETECTOR, seed=812)
    np.testing.assert_array_equal(result, seeded)
    next_result, _ = apply_detector_noise(source, 7.0, DETECTOR, rng=rng)
    assert not np.array_equal(result, next_result)


def test_seed_and_generator_are_mutually_exclusive():
    with pytest.raises(ValueError, match='either seed or rng'):
        apply_detector_noise(np.ones((3, 4)), 7.0, DETECTOR, seed=1, rng=np.random.default_rng(1))


@pytest.mark.parametrize('rng', [12, np.random.RandomState(12), object()])
def test_rng_contract_rejects_unsupported_random_streams(rng):
    with pytest.raises(ValueError, match='numpy.random.Generator'):
        apply_detector_noise(np.ones((3, 4)), 7.0, DETECTOR, rng=rng)
