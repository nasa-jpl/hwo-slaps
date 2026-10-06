"""Supported backend patch behavior at the real convolver and blurring-mask boundary."""

import warnings

import numpy as np
import pytest
from scipy.ndimage import binary_dilation

from hwoslaps.optics.kernels import make_convolver

pytestmark = pytest.mark.backend
SCALE = 0.05


@pytest.mark.parametrize("changed", ["equal_contents", "kernel_contents", "mask_contents"])
def test_backend_convolver_state_follows_kernel_and_mask_contents(changed):
    import autoarray as aa

    values = np.random.default_rng(1).random((21, 21))
    convolver = make_convolver(values, SCALE)
    mask = aa.Mask2D.all_false(shape_native=(40, 40), pixel_scales=SCALE)
    first = convolver.state_from(mask)
    second_mask = np.array(mask, dtype=bool, copy=True)
    if changed == "kernel_contents":
        convolver.kernel.array[...] *= 2.0
        np.testing.assert_array_equal(np.asarray(convolver.kernel.native), 2.0 * values)
    elif changed == "mask_contents":
        second_mask[7, 11] = True
    second = convolver.state_from(aa.Mask2D(mask=second_mask, pixel_scales=SCALE))
    assert (second is first) is (changed == "equal_contents")


def test_backend_convolution_is_bitwise_stable_across_a_state_cache_hit():
    import autoarray as aa

    mask = aa.Mask2D.all_false(shape_native=(40, 40), pixel_scales=SCALE)
    image = aa.Array2D(values=np.random.default_rng(3).random((40, 40)), mask=mask)
    convolver = make_convolver(np.random.default_rng(1).random((21, 21)), SCALE)
    state = convolver.state_from(mask)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        first = convolver.convolved_image_from(image=image, blurring_image=None)
        second = convolver.convolved_image_from(image=image, blurring_image=None)
    assert convolver.state_from(mask) is state
    np.testing.assert_array_equal(np.asarray(second.native), np.asarray(first.native))


@pytest.mark.parametrize("kernel_shape", [(3, 3), (1, 5), (5, 1), (4, 4), (2, 6), (7, 3), (9, 9)])
@pytest.mark.parametrize("array_shape", [(1, 1), (5, 8), (31, 17)])
def test_patched_blurring_mask_matches_independent_dense_geometry(kernel_shape, array_shape):
    from autoarray.mask.mask_2d_util import blurring_mask_2d_from

    ky, kx = kernel_shape
    included = np.random.default_rng(ky * 100 + kx + array_shape[0]).random(array_shape) > 0.7
    included[0, 0] = True
    # A guard border contains the full footprint; the original corner remains an edge case.
    included = np.pad(included, ((ky // 2, ky // 2), (kx // 2, kx // 2)), constant_values=False)
    mask = ~included
    dense_neighborhood = binary_dilation(included, structure=np.ones(kernel_shape, dtype=bool))
    expected = ~(mask & dense_neighborhood)
    actual = blurring_mask_2d_from(mask, kernel_shape_native=kernel_shape, allow_padding=False)
    np.testing.assert_array_equal(actual, expected)


def test_patched_blurring_mask_has_the_single_pixel_hand_footprint():
    from autoarray.mask.mask_2d_util import blurring_mask_2d_from

    mask = np.ones((7, 7), dtype=bool)
    mask[3, 3] = False
    expected = np.ones((7, 7), dtype=bool)
    expected[2:5, 2:5] = False
    expected[3, 3] = True
    np.testing.assert_array_equal(
        blurring_mask_2d_from(mask, kernel_shape_native=(3, 3), allow_padding=True), expected)
