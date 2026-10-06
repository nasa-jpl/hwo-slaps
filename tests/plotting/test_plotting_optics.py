"""Actual detector-kernel and pupil artist values and physical axes."""

import numpy as np
import pytest

from hwoslaps.plotting import plot_kernel, plot_pupil

pytestmark = pytest.mark.backend


def test_optics_plots_preserve_values_and_requested_log_scale(plt):
    from matplotlib.colors import LogNorm
    from matplotlib.backend_bases import MouseEvent
    from hwoslaps.optics.kernels import DetectorPSF, convolve_real_space
    from hwoslaps.scene.spec import pixel_centres_yx
    from hwoslaps.optics.pupils import build_pupil, parse_pupil

    kernel = DetectorPSF.from_array(np.arange(9.).reshape(3, 3), .2, normalize=True)
    _, provided = plt.subplots()
    assert plot_kernel(kernel, log=False, ax=provided) is provided
    np.testing.assert_array_equal(provided.images[0].get_array(), kernel.kernel)
    assert provided.images[0].get_extent() == pytest.approx((-.3, .3, -.3, .3))
    assert provided.images[0].origin == "upper"

    # An asymmetric native delta moves a point up/right by one detector pixel.
    displaced = np.zeros((3, 3))
    displaced[0, 2] = 1
    shifted_kernel = DetectorPSF.from_array(displaced, .2, normalize=False)
    point = np.zeros((5, 5))
    point[2, 2] = 1
    convolved = convolve_real_space(point, shifted_kernel.kernel, .2)
    expected = np.zeros((5, 5))
    expected[1, 3] = 1
    np.testing.assert_allclose(convolved, expected, rtol=0, atol=1e-14)
    y, x = pixel_centres_yx((5, 5), .2)
    peak = np.unravel_index(np.argmax(convolved), convolved.shape)
    sky_y, sky_x = y[peak], x[peak]
    assert (sky_y, sky_x) == pytest.approx((.2, .2))
    kernel_ax = plot_kernel(shifted_kernel, log=False)
    point_ax = plt.subplots()[1]
    point_artist = point_ax.imshow(convolved, origin="upper", extent=(-.5, .5, -.5, .5))
    for axes, artist in ((kernel_ax, kernel_ax.images[0]), (point_ax, point_artist)):
        axes.figure.canvas.draw()
        display_x, display_y = axes.transData.transform((sky_x, sky_y))
        event = MouseEvent("motion_notify_event", axes.figure.canvas, display_x, display_y)
        assert float(artist.get_cursor_data(event)) == pytest.approx(1., rel=0, abs=1e-14)
    np.testing.assert_array_equal(kernel_ax.images[0].get_array(), displaced)
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
