"""Contracts for nonlinear likelihood-ratio metric conventions."""

from __future__ import annotations

import numpy as np
import pytest

from hwoslaps.modeling.nonlinear.likelihood_metrics import (
    SCDD_DELTA_LOG_L_THRESHOLD,
    SCDD_Q_THRESHOLD,
    delta_log_l_from_q,
    profile_likelihood_ratio,
    q_from_delta_log_l,
    z_from_q,
)


def test_scdd_delta_log_l_threshold_maps_to_q_and_z():
    """Convert the SCDD delta log-likelihood threshold to q and Z."""
    q_value = q_from_delta_log_l(SCDD_DELTA_LOG_L_THRESHOLD)

    assert q_value == pytest.approx(SCDD_Q_THRESHOLD)
    assert delta_log_l_from_q(q_value) == pytest.approx(SCDD_DELTA_LOG_L_THRESHOLD)
    assert z_from_q(q_value) == pytest.approx(np.sqrt(SCDD_Q_THRESHOLD))


def test_profile_likelihood_ratio_stores_signed_and_clipped_values():
    """Keep the signed gap while clipping q and Z at zero."""
    metric = profile_likelihood_ratio(
        log_l_smooth=-100.0,
        log_l_subhalo=-102.0,
    )

    assert metric.signed_delta_log_l == pytest.approx(-2.0)
    assert metric.delta_log_l == pytest.approx(0.0)
    assert metric.q == pytest.approx(0.0)
    assert metric.z_local == pytest.approx(0.0)
    assert metric.detected_scdd_local is False


def test_profile_likelihood_ratio_detects_from_signed_delta_log_l():
    """Flag a detection once the signed gap clears the threshold."""
    metric = profile_likelihood_ratio(
        log_l_smooth=-105.0,
        log_l_subhalo=-100.0,
    )

    assert metric.signed_delta_log_l == pytest.approx(SCDD_DELTA_LOG_L_THRESHOLD)
    assert metric.q == pytest.approx(SCDD_Q_THRESHOLD)
    assert metric.detected_scdd_local is True


def test_metric_rejects_non_finite_likelihoods():
    """Reject a non-finite log-likelihood input by name."""
    with pytest.raises(ValueError, match="log_l_smooth"):
        profile_likelihood_ratio(log_l_smooth=np.nan, log_l_subhalo=-1.0)
