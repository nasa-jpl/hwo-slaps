"""Tests for nonlinear validation dataset helpers."""

from __future__ import annotations

import numpy as np

from hwoslaps.modeling.nonlinear.dataset_builder import _exclude_psf_edge_pixels


def test_exclude_psf_edge_pixels_removes_kernel_half_width_border():
    use_mask = np.ones((7, 9), dtype=bool)

    safe_mask = _exclude_psf_edge_pixels(use_mask, psf_shape=(5, 3))

    assert np.count_nonzero(safe_mask) == 21
    assert not np.any(safe_mask[:2, :])
    assert not np.any(safe_mask[-2:, :])
    assert not np.any(safe_mask[:, :1])
    assert not np.any(safe_mask[:, -1:])
    assert np.all(safe_mask[2:-2, 1:-1])
