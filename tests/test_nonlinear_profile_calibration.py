"""Independent mathematical contracts for nonlinear tangent diagnostics."""
import numpy as np
import pytest
from hwoslaps.modeling.nonlinear.profile_calibration import linearized_comparator

def test_matched_comparator_profiles_common_nuisance_and_reports_bounds():
    def residual(x):
        return np.array([3.0 - x[0], 4.0])

    comparison = linearized_comparator(
        residual, np.array([0.0]), np.zeros(2), np.array([-1.0]), np.array([1.0])
    )
    assert comparison["q"] == pytest.approx(16.0)
    assert comparison["q_with_finite_prior_box"] == pytest.approx(20.0)
    assert comparison["bounds_active"] == [1]


def test_background_convention_changes_comparator():
    def residual(x):
        return np.array([1.0 - x[0], 1.0 + x[0], 1.0])

    comparison = linearized_comparator(
        residual,
        np.array([0.0]),
        np.zeros(3),
        np.array([-2.0]),
        np.array([2.0]),
        background_column=np.ones(3),
    )
    assert comparison["q"] == pytest.approx(3.0)
    assert comparison["q_with_free_background_only"] < 1.0e-20
