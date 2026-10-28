"""A filled source pixel needs an area average absent from a native point kernel."""

import math

import pytest
from scipy.integrate import quad
from scipy.special import erf


def test_piecewise_constant_light_does_not_make_native_convolution_exact():
    # A unit Gaussian PSF and unit source/detector squares. One dimension of the
    # native kernel integrates a point at the source-cell centre over the detector.
    point_response = erf(0.5 / math.sqrt(2.0))
    # The continuous response also averages over every emitting location u.
    filled_response, error = quad(lambda u: 0.5 * (erf((0.5 - u) / math.sqrt(2.0))
                                                  - erf((-0.5 - u) / math.sqrt(2.0))), -0.5, 0.5,
                                  epsabs=1e-13, epsrel=1e-13)
    analytic = erf(1 / math.sqrt(2.0)) + math.sqrt(2 / math.pi) * (math.exp(-0.5) - 1)
    assert error < 1e-13
    assert filled_response == pytest.approx(analytic, rel=2e-14)
    assert point_response**2 == pytest.approx(0.14663149630841185, rel=2e-14)
    assert filled_response**2 == pytest.approx(0.1359738930, rel=2e-9)
    assert point_response**2 / filled_response**2 - 1 > 0.078
