"""In-repo ConvolverState reuse and upstream separable blurring-mask geometry."""

import numpy as np
import pytest
from scipy.ndimage import binary_dilation

from hwoslaps.observation.expected import convolve_light
from hwoslaps.optics.kernels import DetectorPSF, KernelBinding, make_convolver
from hwoslaps.scene.spec import LightGroup

pytestmark = pytest.mark.backend
SCALE = 0.05


def test_convolver_for_shape_reuses_state_by_shape_and_pixel_scale():
    import autoarray as aa

    values = np.random.default_rng(1).random((21, 21))
    psf = DetectorPSF.from_array(values, SCALE, normalize=True)
    kernel_before = psf.kernel.tobytes()

    first = psf.convolver_for_shape((40, 40), SCALE)
    again = psf.convolver_for_shape((40, 40), SCALE)
    assert again is first
    assert first._state is not None
    mask_40 = aa.Mask2D.all_false(shape_native=(40, 40), pixel_scales=SCALE)
    assert first.state_from(mask_40) is first._state

    different_shape = psf.convolver_for_shape((42, 40), SCALE)
    assert different_shape is not first and different_shape._state is not first._state
    mask_42 = aa.Mask2D.all_false(shape_native=(42, 40), pixel_scales=SCALE)
    assert different_shape.state_from(mask_42) is different_shape._state

    different_scale = psf.convolver_for_shape((40, 40), 0.06)
    assert different_scale is not first and different_scale._state is not first._state
    mask_scale = aa.Mask2D.all_false(shape_native=(40, 40), pixel_scales=0.06)
    assert different_scale.state_from(mask_scale) is different_scale._state

    assert psf.kernel.tobytes() == kernel_before
    assert psf.convolver() is not first
    for bad_shape in ((0, 40), (40,), (True, 40), "40x40"):
        with pytest.raises(ValueError, match="shape_native"):
            psf.convolver_for_shape(bad_shape)


def test_convolve_light_reuses_state_and_matches_uncached_convolver():
    rng = np.random.default_rng(3)
    psf = DetectorPSF.from_array(rng.random((21, 21)), SCALE, normalize=True)
    image = rng.random((40, 40))
    groups = {"source": LightGroup(plane="source", sed=None, components=("light",))}
    binding = KernelBinding.uniform(psf, ["source"])

    first = convolve_light({"source": image}, groups, binding, SCALE)
    state_after_first = psf.convolver_for_shape((40, 40), SCALE)._state
    second = convolve_light({"source": image}, groups, binding, SCALE)
    state_after_second = psf.convolver_for_shape((40, 40), SCALE)._state

    assert state_after_first is not None and state_after_second is state_after_first
    assert second["source"].tobytes() == first["source"].tobytes()

    import autoarray as aa

    mask = aa.Mask2D.all_false(shape_native=(40, 40), pixel_scales=SCALE)
    uncached = make_convolver(psf.kernel, SCALE).convolved_image_from(
        image=aa.Array2D(values=image.copy(), mask=mask), blurring_image=None)
    assert first["source"].tobytes() == np.asarray(uncached.native, dtype=float).tobytes()


@pytest.mark.parametrize("kernel_shape", [(3, 3), (1, 5), (5, 1), (4, 4), (2, 6), (7, 3), (9, 9)])
@pytest.mark.parametrize("array_shape", [(1, 1), (5, 8), (31, 17)])
def test_blurring_mask_matches_independent_dense_geometry(kernel_shape, array_shape):
    from autoarray.mask.mask_2d_util import blurring_mask_2d_from

    ky, kx = kernel_shape
    included = np.random.default_rng(ky * 100 + kx + array_shape[0]).random(array_shape) > 0.7
    included[0, 0] = True
    # The guard border keeps each footprint in the valid domain without backend padding.
    included = np.pad(included, ((ky // 2, ky // 2), (kx // 2, kx // 2)), constant_values=False)
    mask = ~included
    dense_neighborhood = binary_dilation(included, structure=np.ones(kernel_shape, dtype=bool))
    expected = ~(mask & dense_neighborhood)
    actual = blurring_mask_2d_from(mask, kernel_shape_native=kernel_shape, allow_padding=False)
    np.testing.assert_array_equal(actual, expected)


def test_blurring_mask_has_the_single_pixel_hand_footprint():
    from autoarray.mask.mask_2d_util import blurring_mask_2d_from

    mask = np.ones((7, 7), dtype=bool)
    mask[3, 3] = False
    expected = np.ones((7, 7), dtype=bool)
    expected[2:5, 2:5] = False
    expected[3, 3] = True
    np.testing.assert_array_equal(
        blurring_mask_2d_from(mask, kernel_shape_native=(3, 3), allow_padding=True), expected)
