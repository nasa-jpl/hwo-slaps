"""Tests for local nonlinear profile-likelihood helper utilities."""

from __future__ import annotations

import numpy as np
import pytest

from hwoslaps.modeling.nonlinear.local_profile import (
    fit_local_least_squares_profile,
    profile_likelihood_q,
)


def test_multistart_least_squares_selects_best_profile_attempt():
    def residual_fn(x):
        return np.array([x[0] - 2.0, 2.0 * (x[1] + 1.0)])

    result = fit_local_least_squares_profile(
        model_name="toy",
        residual_fn=residual_fn,
        initial_points=[[-10.0, 5.0], [10.0, -5.0]],
        labels=["left", "right"],
        max_nfev=20,
    )

    assert result.model_name == "toy"
    assert len(result.attempts) == 2
    assert result.chi2_min == pytest.approx(0.0, abs=1.0e-16)
    assert result.best.x == pytest.approx([2.0, -1.0], abs=1.0e-8)
    assert result.convergence_abs_spread == pytest.approx(0.0, abs=1.0e-16)


def test_profile_likelihood_q_is_non_negative():
    assert profile_likelihood_q(smooth_chi2_min=12.0, subhalo_chi2_min=2.0) == 10.0
    assert profile_likelihood_q(smooth_chi2_min=2.0, subhalo_chi2_min=12.0) == 0.0
