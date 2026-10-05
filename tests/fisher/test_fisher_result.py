"""ForecastResult rules: detections, validation, immutability and the derived statistics.

Oracles: hand arrays, and the statistics the 974cee9 ``modeling.fisher_core``
bank finisher printed for the same primaries (run-directory script
``probes/sta_literals.py``, W1-STA ledger).
"""

import numpy as np
import pytest

from hwoslaps.fisher.positions import explicit_positions
from hwoslaps.fisher.result import ForecastResult

nan, inf = np.nan, np.inf


def result(fisher_profiled, *, amplitude_hat=None, amplitude_spurious=None, fisher_raw=None, masses=None,
           relation=None, provenance=None):
    profiled = np.asarray(fisher_profiled, dtype=float)
    count = profiled.shape[1]
    mismatched = amplitude_hat is not None
    return ForecastResult(
        masses_msun=np.geomspace(1e7, 1e9, profiled.shape[0]) if masses is None else masses,
        positions=explicit_positions(np.column_stack((np.arange(count) * 0.1, np.zeros(count))), (0.0, 0.0)),
        fisher_raw=profiled + 1.0 if fisher_raw is None else fisher_raw,
        fisher_profiled=profiled,
        amplitude_hat=amplitude_hat,
        amplitude_spurious=amplitude_spurious,
        psf_relation=relation or ("knowledge_error" if mismatched else "matched"),
        config={"forecast": {"positions": {"kind": "explicit"}}},
        provenance={"statistic": "profiled_linear_gaussian_q"} if provenance is None else provenance,
    )


def test_mismatch_detections_require_positive_amplitude():
    information = np.full((2, 4), 25.0)
    amplitude_hat = np.array([[1.0, -1.0, 0.0, nan], [0.5, 0.7, -2.0, 1.0]])
    amplitude_spurious = np.array([[-1.0, 1.0, 0.8, 0.0], [nan, -0.9, 0.9, 0.1]])
    mismatched = result(information, amplitude_hat=amplitude_hat, amplitude_spurious=amplitude_spurious)
    assert mismatched.detection_metric == "q_mismatch"
    np.testing.assert_array_equal(mismatched.detections(q_threshold=10.0),
                                  [[True, False, False, False], [False, True, False, True]])
    np.testing.assert_array_equal(mismatched.detections(q_threshold=10.0, metric="q_spurious"),
                                  [[False, True, True, False], [False, False, True, False]])
    np.testing.assert_array_equal(mismatched.detections(q_threshold=10.0, metric="q_asimov"), np.ones((2, 4), bool))
    matched = result(information)
    assert matched.detection_metric == "q_asimov"
    with pytest.raises(ValueError, match="q_mismatch was not evaluated"):
        matched.detections(q_threshold=10.0, metric="q_mismatch")


def test_properties_reproduce_974cee9_bank_statistics_bitwise():
    raw = np.array([[4.0, 9.0, 2.5], [1.0, 0.0, 7.3]])
    profiled = np.array([[3.0, 9.0, 0.25], [0.0, 0.0, 6.9399999999999995]])
    amplitude_hat = np.array([[1.0666666666666667, -0.23333333333333334, 1.0], [nan, nan, 0.03746397694524496]])
    amplitude_spurious = np.array([[-0.013333333333333334, 0.1, -0.019999999999999997],
                                   [nan, nan, -0.2890489913544668]])
    forecast = result(profiled, amplitude_hat=amplitude_hat, amplitude_spurious=amplitude_spurious, fisher_raw=raw)
    expected = {
        "sigma_amplitude": [0.5773502691896258, 0.3333333333333333, 2.0, inf, inf, 0.37959480900056175],
        "z_asimov": [1.7320508075688772, 3.0, 0.5, 0.0, 0.0, 2.634387974463898],
        "degradation": [0.75, 1.0, 0.1, 0.0, 0.0, 0.9506849315068493],
        "z_mismatch": [1.8475208614068024, -0.7, 0.5, nan, nan, 0.09869465034014605],
        "q_mismatch": [3.413333333333333, 0.48999999999999994, 0.25, nan, nan, 0.00974063400576369],
        "z_spurious": [0.023094010767585032, 0.30000000000000004, 0.009999999999999998, nan, nan,
                       0.7614671868551266],
        "q_spurious": [0.0005333333333333334, 0.09000000000000002, 9.999999999999996e-05, nan, nan,
                       0.5798322766570603],
        "q_asimov": [3.0, 9.0, 0.25, 0.0, 0.0, 6.9399999999999995],
    }
    for name, values in expected.items():
        statistic = getattr(forecast, name)
        assert statistic.shape == (2, 3), name
        assert statistic.tobytes() == np.array(values).reshape(2, 3).tobytes(), name


def test_result_validates_axes_metadata_and_provenance():
    good = np.ones((2, 3))
    with pytest.raises(ValueError, match=r"shape \(2, 3\)"):
        result(good, fisher_raw=np.ones((3, 2)))
    with pytest.raises(ValueError, match="masses_msun"):
        result(good, masses=np.array([1e8, -1e8]))
    with pytest.raises(ValueError, match="finite and non-negative"):
        result(np.array([[1.0, -1.0, 1.0], [1.0, 1.0, 1.0]]))
    with pytest.raises(ValueError, match="together or not at all"):
        result(good, amplitude_hat=good)
    with pytest.raises(ValueError, match="'matched' result has no fitted amplitudes"):
        result(good, amplitude_hat=good, amplitude_spurious=good, relation="matched")
    with pytest.raises(ValueError, match="'kernel' result needs fitted amplitudes"):
        result(good, relation="kernel")
    with pytest.raises(ValueError, match="non-finite"):
        result(good, provenance={"gram_condition_number": inf})
    with pytest.raises(TypeError, match="array_digest"):
        result(good, provenance={"mask": np.ones(3)})
    forecast = result(good, provenance={"engine": ("reference", 1), "rank": np.int64(3)})
    assert forecast.provenance == {"engine": ["reference", 1], "rank": 3}
    assert forecast.config == {"forecast": {"positions": {"kind": "explicit"}}}


@pytest.mark.parametrize("threshold", [True, np.bool_(True), False, 0.0, -1.0, inf, nan, "10"])
def test_threshold_must_be_a_positive_finite_number(threshold):
    with pytest.raises(ValueError, match="q_threshold must be a positive finite number"):
        result(np.full((1, 2), 20.0)).detections(q_threshold=threshold)


def test_result_arrays_are_read_only():
    profiled = np.array([[2.0, 3.0]])
    hat = np.array([[0.5, -0.5]])
    forecast = result(profiled, amplitude_hat=hat, amplitude_spurious=hat.copy())
    profiled[0, 0] = 99.0
    hat[0, 0] = 99.0
    assert forecast.fisher_profiled[0, 0] == 2.0 and forecast.amplitude_hat[0, 0] == 0.5
    for name in ("masses_msun", "fisher_raw", "fisher_profiled", "amplitude_hat", "amplitude_spurious",
                 "positions_yx"):
        with pytest.raises(ValueError, match="read-only"):
            getattr(forecast, name)[0] = 0.0
