"""Canary tests for convolver-state reuse and separable blurring dilation.

These pin two performance-critical semantics the Fisher setup path depends
on after the 2026-08-18 convolver fixes:

1. ``autoarray.Convolver.state_from`` memoizes its ``ConvolverState`` across
   calls with identical kernel and mask contents. Without the memo, every
   convolution call rebuilds the state, whose blurring-mask derivation
   costs O(mask_pixels * kernel_area) time and scratch memory (measured
   ~13 GB and ~22 s per call at a 201-kernel, ~390 GB and ~20 min at 499).
   An environment that loses the local autoarray patch fails these tests
   loudly instead of silently reverting to that behavior.
2. The two-pass 1D blurring dilation is exactly equivalent to the dense
   kernel-box dilation it replaced, for all structure parities.
3. ``make_pyauto_convolver`` returns one cached convolver per kernel object
   while the kernel values and pixel scales are unchanged, so the memoized
   state is actually shared across the detector's repeated convolution calls
   and never shared across detector grids.
"""

import warnings

import numpy as np
import pytest
from scipy.ndimage import binary_dilation

import autolens as al

from hwoslaps.psf.utils import make_pyauto_convolver, make_pyauto_kernel

PIXEL_SCALE = 0.05


def _kernel(size=21, seed=1):
    rng = np.random.default_rng(seed)
    return make_pyauto_kernel(
        values=rng.random((size, size)), pixel_scales=PIXEL_SCALE
    )


def _mask(shape=(40, 40)):
    return al.Mask2D.all_false(shape_native=shape, pixel_scales=PIXEL_SCALE)


class TestStateFromMemoization:
    def test_identical_inputs_reuse_one_state(self):
        convolver = al.Convolver(kernel=_kernel())
        state_first = convolver.state_from(mask=_mask())
        state_second = convolver.state_from(mask=_mask())
        assert state_second is state_first

    def test_kernel_mutation_rebuilds_state(self):
        kernel = _kernel()
        convolver = al.Convolver(kernel=kernel)
        state_first = convolver.state_from(mask=_mask())
        kernel._array = kernel._array * 2.0
        state_second = convolver.state_from(mask=_mask())
        assert state_second is not state_first

    def test_mask_content_change_rebuilds_state(self):
        convolver = al.Convolver(kernel=_kernel())
        state_first = convolver.state_from(mask=_mask())
        changed = np.zeros((40, 40), dtype=bool)
        changed[7, 11] = True
        state_second = convolver.state_from(
            mask=al.Mask2D(mask=changed, pixel_scales=PIXEL_SCALE)
        )
        assert state_second is not state_first

    def test_convolved_image_bitwise_stable_across_cache_hit(self):
        rng = np.random.default_rng(3)
        mask = _mask()
        image = al.Array2D(values=rng.random((40, 40)), mask=mask)
        convolver = al.Convolver(kernel=_kernel())
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            out_first = convolver.convolved_image_from(
                image=image, blurring_image=None
            )
            out_second = convolver.convolved_image_from(
                image=image, blurring_image=None
            )
        assert np.array_equal(
            np.asarray(out_first.native), np.asarray(out_second.native)
        )


class TestSeparableBlurringDilation:
    @pytest.mark.parametrize(
        "structure_shape",
        [(3, 3), (1, 5), (5, 1), (4, 4), (2, 6), (7, 3), (9, 9)],
    )
    @pytest.mark.parametrize("array_shape", [(1, 1), (5, 8), (31, 17)])
    def test_two_pass_equals_dense_box(self, structure_shape, array_shape):
        ky, kx = structure_shape
        rng = np.random.default_rng(ky * 100 + kx + array_shape[0])
        array = rng.random(array_shape) > 0.7
        dense = binary_dilation(
            array, structure=np.ones((ky, kx), dtype=bool)
        )
        two_pass = binary_dilation(
            binary_dilation(array, structure=np.ones((ky, 1), dtype=bool)),
            structure=np.ones((1, kx), dtype=bool),
        )
        assert np.array_equal(two_pass, dense)

    def test_blurring_mask_single_pixel_hand_case(self):
        from autoarray.mask.mask_2d_util import blurring_mask_2d_from

        mask = np.ones((7, 7), dtype=bool)
        mask[3, 3] = False
        blurring = blurring_mask_2d_from(
            mask, kernel_shape_native=(3, 3), allow_padding=True
        )
        expected = np.ones((7, 7), dtype=bool)
        expected[2:5, 2:5] = False
        expected[3, 3] = True
        assert np.array_equal(blurring, expected)


class TestMakePyautoConvolverCache:
    def test_same_kernel_returns_cached_convolver(self):
        kernel = _kernel()
        first = make_pyauto_convolver(kernel)
        second = make_pyauto_convolver(kernel)
        assert second is first

    def test_inplace_value_change_invalidates_cache(self):
        kernel = _kernel()
        first = make_pyauto_convolver(kernel)
        kernel._array = kernel._array * 2.0
        second = make_pyauto_convolver(kernel)
        assert second is not first

    def test_convolver_input_passes_through(self):
        convolver = make_pyauto_convolver(_kernel())
        assert make_pyauto_convolver(convolver) is convolver

    def test_distinct_pixel_scales_do_not_share_convolver(self):
        values = np.random.default_rng(5).random((21, 21))
        coarse = make_pyauto_kernel(values=values, pixel_scales=PIXEL_SCALE)
        fine = make_pyauto_kernel(
            values=values, pixel_scales=PIXEL_SCALE / 2.0
        )
        first = make_pyauto_convolver(coarse)
        second = make_pyauto_convolver(fine)
        assert second is not first
        assert first.kernel is coarse
        assert second.kernel is fine

    def test_copied_kernel_at_new_pixel_scale_invalidates_cache(self):
        kernel = _kernel()
        first = make_pyauto_convolver(kernel)
        rebound = kernel.copy()
        assert hasattr(rebound, "_hwoslaps_convolver_cache")
        rebound.mask = al.Mask2D.all_false(
            shape_native=kernel.shape_native, pixel_scales=PIXEL_SCALE / 2.0
        )
        second = make_pyauto_convolver(rebound)
        assert second is not first
        assert second.kernel is rebound
        assert second.kernel.pixel_scales != first.kernel.pixel_scales
