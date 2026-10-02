"""Independent spatial estimands and signed detection contracts."""

import numpy as np
import pytest

from hwoslaps.modeling.forecast_results import ForecastResult, summarize_forecast


def _result(q, **extra):
    q = np.asarray(q, dtype=float)
    return ForecastResult(
        masses_msun=np.array([1e7, 1e8]),
        positions_yx=np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0]]),
        q_asimov=q,
        fisher_raw=q + 2,
        fisher_profiled=q,
        sigma_amplitude=np.ones_like(q),
        degradation=np.ones_like(q),
        **extra,
    )


def test_spatial_summary_uses_explicit_selection_weights_and_separate_boundary():
    result = _result([[4, 12, 100], [10, 20, 1]])
    summary = summarize_forecast(
        result,
        10,
        selection=np.array([True, True, False]),
        cell_areas_arcsec2=[2, 3, 99],
        boundary=np.array([False, False, True]),
    )
    np.testing.assert_array_equal(summary.q_max, [12, 20])
    np.testing.assert_array_equal(summary.detectable_fraction, [0.5, 1])
    np.testing.assert_array_equal(summary.detectable_area_arcsec2, [3, 5])
    np.testing.assert_array_equal(summary.boundary_detectable, [True, False])
    summary = summarize_forecast(result, 10)
    assert summary.detectable_area_arcsec2 is None
    assert summary.boundary_detectable is None


@pytest.mark.parametrize(
    "metric,amplitude_name", [("q_mismatch", "amplitude_hat"), ("q_spurious", "amplitude_spurious")]
)
def test_mismatch_detection_requires_positive_amplitude(metric, amplitude_name):
    result = _result(
        [[10, 10, 10], [10, 10, 10]],
        **{
            metric: np.full((2, 3), 20.0),
            amplitude_name: np.array([[-1, 0, 1], [1, -1, 1]]),
        },
    )
    summary = summarize_forecast(result, 10, metric=metric)
    np.testing.assert_array_equal(summary.detectable_fraction, [1 / 3, 2 / 3])


def test_nonfinite_consumed_statistics_are_reported_without_fabricating_zero_area():
    result = _result([[4, 12, np.nan], [10, 20, np.nan]])
    selected = np.array([True, True, False])
    summarize_forecast(result, 10, selection=selected)
    with pytest.raises(ValueError, match="non-finite"):
        summarize_forecast(result, 10)
    with pytest.raises(ValueError, match="non-finite"):
        summarize_forecast(result, 10, selection=selected, boundary=np.array([False, False, True]))


def test_result_rejects_wrong_axes_and_missing_required_statistics():
    with pytest.raises(ValueError, match="shape"):
        _result(np.ones((3, 2)))
    with pytest.raises(ValueError, match="required"):
        ForecastResult([1e8], [[0, 0]], None, [[1]], [[1]], [[1]], [[1]])


def test_default_detection_uses_actual_mismatched_data_and_explicit_null_control():
    result = _result(
        [[20, 20, 20], [20, 20, 20]],
        q_mismatch=np.array([[20, 20, 20], [5, 20, 20]]),
        amplitude_hat=np.array([[-1, 0, 1], [1, -1, 1]]),
        q_spurious=np.full((2, 3), 20.0),
        amplitude_spurious=np.array([[-1, 0, 1], [-1, -1, -1]]),
    )
    assert result.detection_metric == "q_mismatch"
    np.testing.assert_array_equal(result.detections(10), [[False, False, True], [False, False, True]])
    np.testing.assert_array_equal(summarize_forecast(result, 10).detectable_fraction, [1 / 3, 1 / 3])
    np.testing.assert_array_equal(
        summarize_forecast(result, 10, metric="q_asimov").detectable_fraction, [1, 1]
    )
    null = summarize_forecast(result, 10, metric="q_spurious")
    np.testing.assert_array_equal(null.detectable_fraction, [1 / 3, 0])
    np.testing.assert_array_equal(null.q_max, [20, 0])


@pytest.mark.parametrize("threshold", [True, np.bool_(True), False])
def test_boolean_detection_threshold_is_rejected(threshold):
    result = _result([[20, 20, 20], [20, 20, 20]])
    with pytest.raises(ValueError, match="boolean"):
        result.detections(threshold)
    with pytest.raises(ValueError, match="boolean"):
        summarize_forecast(result, threshold)


@pytest.mark.parametrize("q,fraction,area", [(1.0, 0.0, 0.0), (10.0, 1.0, 6.0)])
def test_uniform_detection_limits_have_literal_area_and_fraction(q, fraction, area):
    result = _result(np.full((2, 3), q))
    summary = summarize_forecast(result, 10, cell_areas_arcsec2=2.0)
    np.testing.assert_array_equal(summary.detectable_fraction, [fraction, fraction])
    np.testing.assert_array_equal(summary.detectable_area_arcsec2, [area, area])
