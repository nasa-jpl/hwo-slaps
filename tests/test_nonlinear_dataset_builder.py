"""Tests for nonlinear validation dataset helpers."""

from __future__ import annotations

import numpy as np

from hwoslaps.modeling.nonlinear.dataset_builder import (
    _exclude_psf_edge_pixels,
    data_array_from_observation,
    noise_rate_from_observation,
    source_only_data_adu,
    source_only_data_electron_rate,
)


class _Observation:
    gain = 2.0
    exposure_time = 10.0
    noiseless_source_eps = np.array([[1.5, 2.5]])
    source_electrons = noiseless_source_eps * exposure_time
    data = type("Data", (), {"native": np.array([[25.0, 35.0]])})()
    noise_map = type("Noise", (), {"native": np.array([[4.0, 6.0]])})()
    sky_electrons_per_pixel = 10.0
    dark_electrons_per_pixel = 2.0


def test_exclude_psf_edge_pixels_removes_kernel_half_width_border():
    use_mask = np.ones((7, 9), dtype=bool)

    safe_mask = _exclude_psf_edge_pixels(use_mask, psf_shape=(5, 3))

    assert np.count_nonzero(safe_mask) == 21
    assert not np.any(safe_mask[:2, :])
    assert not np.any(safe_mask[-2:, :])
    assert not np.any(safe_mask[:, :1])
    assert not np.any(safe_mask[:, -1:])
    assert np.all(safe_mask[2:-2, 1:-1])


def test_validation_dataset_uses_rate_units_matching_pyautolens_model():
    observation = _Observation()

    source_rate = source_only_data_electron_rate(observation, dataset_kind="noisy")
    source_rate_alias = source_only_data_adu(observation, dataset_kind="noisy")
    asimov = data_array_from_observation(observation, dataset_kind="asimov")
    noisy = data_array_from_observation(observation, dataset_kind="noisy")
    noise = noise_rate_from_observation(observation)

    assert np.allclose(source_rate, [[5.0, 7.0]])
    assert np.allclose(source_rate_alias, source_rate)
    assert np.allclose(asimov, [[1.5, 2.5]])
    # ADU -> electrons / second, then subtract known sky+dark in those units.
    assert np.allclose(noisy, [[3.8, 5.8]])
    assert np.allclose(noise, [[0.8, 1.2]])
