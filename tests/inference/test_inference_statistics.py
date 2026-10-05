"""The nonlinear statistic: q from the two role maxima and its Gaussian equivalent."""

from __future__ import annotations

import math

import pytest

from hwoslaps.inference.statistics import likelihood_ratio, z_from_q


@pytest.mark.parametrize(("smooth", "subhalo", "q_signed", "q_clipped"), [
    (15816.188918883025, 16008.188527701757, 383.99921763746534, 383.99921763746534),
    (12361.15145778017, 12536.946336097151, 351.5897566339627, 351.5897566339627),
    (-10.0, -12.5, -5.0, 0.0),
], ids=["b4a-refined-maxima", "b4b-refined-maxima", "negative"])
def test_likelihood_ratio_is_twice_the_signed_difference(smooth, subhalo, q_signed, q_clipped):
    """The B4 rows are the refined role maxima and q of the 8fa6209 B4a and B4b base runs."""
    assert likelihood_ratio(smooth, subhalo) == (q_signed, q_clipped)


@pytest.mark.parametrize(("q", "z"), [(383.99921763746534, math.sqrt(383.99921763746534)), (0.0, 0.0),
                                      (-5.0, math.nan)], ids=["positive", "zero", "negative"])
def test_z_is_the_root_of_a_non_negative_q(q, z):
    assert z_from_q(q) == z or (math.isnan(z) and math.isnan(z_from_q(q)))


@pytest.mark.parametrize(("call", "arguments"), [
    (likelihood_ratio, (math.nan, 1.0)), (likelihood_ratio, (1.0, math.inf)), (z_from_q, (math.nan,)),
], ids=["nan-smooth", "inf-subhalo", "nan-q"])
def test_non_finite_inputs_raise(call, arguments):
    with pytest.raises(ValueError, match="finite"):
        call(*arguments)
