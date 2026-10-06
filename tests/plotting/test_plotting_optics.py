"""Actual detector-kernel and pupil artist values and physical axes."""

import numpy as np
import pytest

from hwoslaps.plotting import plot_kernel, plot_pupil

pytestmark = pytest.mark.backend


def test_optics_plots_preserve_values_and_requested_log_scale(plt):
    from matplotlib.colors import LogNorm
    from hwoslaps.optics.kernels import DetectorPSF
    from hwoslaps.optics.pupils import build_pupil, parse_pupil

    kernel = DetectorPSF.from_array(np.arange(9.).reshape(3, 3), .2)
    _, provided = plt.subplots()
    assert plot_kernel(kernel, log=False, ax=provided) is provided
    np.testing.assert_array_equal(provided.images[0].get_array(), kernel.kernel)
    assert provided.images[0].get_extent() == pytest.approx((-.3, .3, -.3, .3))
    logarithmic = plot_kernel(kernel, log=True)
    assert isinstance(logarithmic.images[0].norm, LogNorm)
    np.testing.assert_array_equal(logarithmic.images[0].get_array().data, kernel.kernel)
    assert np.ma.getmaskarray(logarithmic.images[0].get_array())[0, 0]
    pupil = build_pupil(parse_pupil({"kind": "circular", "diameter_m": 2., "pixels": 32,
                                    "supersampling": 2}, "pupil"))
    pupil_ax = plot_pupil(pupil)
    np.testing.assert_array_equal(pupil_ax.images[0].get_array(), pupil.transmission.shaped)
    assert pupil_ax.images[0].get_extent() == (-1, 1, -1, 1)
    assert pupil_ax.get_xlabel() == "x (m)" and pupil_ax.get_ylabel() == "y (m)"
