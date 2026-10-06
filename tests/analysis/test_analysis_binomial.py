"""Independent binomial tail oracles and actual-count separation tables."""

import numpy as np
import pytest
from scipy.stats import binom, binomtest

from hwoslaps.analysis.binomial import BinomialCount, clopper_pearson
from hwoslaps.analysis.knowledge_error import first_separating_amplitude


@pytest.mark.parametrize("trials", [1, 590])
def test_clopper_pearson_closed_forms_at_zero_and_full_counts(trials):
    assert clopper_pearson(0, trials, confidence=0.95) == pytest.approx(
        (0, 1 - 0.025 ** (1 / trials)), abs=1e-12)
    assert clopper_pearson(trials, trials, confidence=0.95) == pytest.approx(
        (0.025 ** (1 / trials), 1), abs=1e-12)


@pytest.mark.parametrize("count,trials", [(0, 590), (1, 590), (3, 493), (50, 100), (590, 590), (3, 20), (17, 40)])
@pytest.mark.parametrize("confidence", [0.8, 0.95])
def test_clopper_pearson_bounds_solve_the_binomial_tail_equations(count, trials, confidence):
    lower, upper = clopper_pearson(count, trials, confidence=confidence)
    independent = binomtest(count, trials).proportion_ci(confidence, method="exact")
    assert (lower, upper) == pytest.approx((independent.low, independent.high), abs=1e-12)
    if count:
        assert binom.sf(count - 1, trials, lower) == pytest.approx((1 - confidence) / 2, abs=1e-10)
    if count != trials:
        assert binom.cdf(count, trials, upper) == pytest.approx((1 - confidence) / 2, abs=1e-10)


@pytest.mark.parametrize("count,trials", [(True, 5), (np.bool_(False), 5), (1.5, 5), (1, True),
                                         (1, 5.0), (-1, 5), (6, 5), (0, 0)])
def test_clopper_pearson_rejects_invalid_counts(count, trials):
    with pytest.raises(ValueError):
        BinomialCount(count, trials)
    with pytest.raises(ValueError):
        clopper_pearson(count, trials, confidence=0.95)


@pytest.mark.parametrize("confidence", [True, np.bool_(False), 0, 1, -0.1, np.nan, np.inf, "0.95"])
def test_clopper_pearson_rejects_invalid_confidence(confidence):
    with pytest.raises(ValueError, match="confidence"):
        clopper_pearson(1, 10, confidence=confidence)


def test_first_separating_amplitude():
    # A9's submitted-paper null and A1's low-null table exercise distinct count regimes.
    for null, counts, expected in ((BinomialCount(3, 493), [1, 2, 20], 10.0),
                                   (BinomialCount(0, 590), [1, 20, 30], 5.0)):
        trials = 100 if null.count else 36
        controls = {a: BinomialCount(k, trials) for a, k in zip((2.0, 5.0, 10.0), counts)}
        result = first_separating_amplitude(controls, null, confidence=0.95)
        assert result.amplitude == expected and result.separates_all_larger is True
        oracle = binomtest(null.count, null.trials).proportion_ci(0.95, method="exact")
        assert result.null_interval == pytest.approx((oracle.low, oracle.high), abs=1e-12)
        for a, count in controls.items():
            bounds = binomtest(count.count, count.trials).proportion_ci(0.95, method="exact")
            assert result.control_intervals[a] == pytest.approx((bounds.low, bounds.high), abs=1e-12)
        record = result.to_mapping()
        assert record["interval"] == "clopper_pearson_nominal"
        assert record["confidence"] == 0.95 and record["amplitude"] == expected
        assert record["separates_all_larger"] is True
        assert record["null_interval"] == list(result.null_interval)
        assert record["control_intervals"] == {a: list(bounds) for a, bounds in result.control_intervals.items()}
        none = first_separating_amplitude({2.0: controls[2.0]}, null, confidence=0.95)
        assert none.amplitude is None and none.separates_all_larger is None

    # At confidence .5, full/zero counts out of two have touching bounds exactly .5.
    equality = first_separating_amplitude({1.0: BinomialCount(2, 2)}, BinomialCount(0, 2), confidence=0.5)
    assert equality.control_intervals[1.0][0] == equality.null_interval[1] == 0.5
    assert equality.amplitude is None
    nonmonotonic = first_separating_amplitude(
        {2.0: BinomialCount(20, 36), 5.0: BinomialCount(1, 36)}, BinomialCount(0, 590), confidence=0.95)
    assert nonmonotonic.amplitude == 2.0 and nonmonotonic.separates_all_larger is False
    empty = first_separating_amplitude({}, BinomialCount(0, 590), confidence=0.95)
    assert empty.amplitude is None and empty.control_intervals == {} and empty.separates_all_larger is None


@pytest.mark.parametrize("amplitude", [True, -1.0, np.inf, np.nan, "2"])
def test_separation_refuses_invalid_amplitudes(amplitude):
    with pytest.raises(ValueError, match="amplitude"):
        first_separating_amplitude({amplitude: BinomialCount(1, 10)}, BinomialCount(0, 10), confidence=0.95)
