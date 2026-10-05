"""Masks, flattening and noise whitening of fisher.data_space.

Oracles: hand pixel counts and hand radii at unit pixel scale, hand arrays for
the S/N rule, and explicit inverse-covariance algebra on a hand covariance.
"""

import numpy as np
import pytest

from hwoslaps.fisher.data_space import (
    DataSpace, all_pixels_mask, annulus_mask, build_data_space, grid_centre_yx, load_noise_covariance,
    psf_border_mask, source_snr_mask,
)
from hwoslaps.fisher.statistics import Whitener


def unit_grid(shape, shift=(0.0, 0.0)):
    """Native (y, x) pixel centres at unit pixel scale about the image centre, shifted by ``shift``."""
    y, x = np.mgrid[0:shape[0], 0:shape[1]].astype(float)
    return y - (shape[0] - 1) / 2 + shift[0], x - (shape[1] - 1) / 2 + shift[1]


@pytest.mark.parametrize("shape, kernel, rows, columns", [
    ((9, 11), (3, 5), slice(1, -1), slice(2, -2)),
    ((11, 11), (5, 3), slice(2, -2), slice(1, -1)),
    ((11, 11), (1, 1), slice(None), slice(None)),
    ((6, 8), (4, 2), slice(2, -2), slice(1, -1)),
], ids=["3x5", "5x3", "1x1", "even"])
def test_psf_border_mask_removes_half_kernel_borders(shape, kernel, rows, columns):
    expected = np.zeros(shape, dtype=bool)
    expected[rows, columns] = True
    mask = psf_border_mask(shape, kernel)
    np.testing.assert_array_equal(mask, expected)
    if kernel == (3, 5):
        assert np.count_nonzero(mask) == (9 - 2) * (11 - 4)
    np.testing.assert_array_equal(all_pixels_mask(shape), np.ones(shape, dtype=bool))


@pytest.mark.parametrize("about, inner, outer, expected", [
    ("grid", 1.0, 2.0, [(r, c) for r in range(5) for c in range(5)
                        if 1.0 <= np.hypot(r - 2, c - 2) <= 2.0]),
    ("lens", 0.0, 1.0, [(2, 1), (3, 0), (3, 1), (3, 2), (4, 1)]),
], ids=["grid-centre", "lens-centre"])
def test_annulus_mask_is_closed_and_centred(about, inner, outer, expected):
    y, x = unit_grid((5, 5), shift=(0.25, -0.5))
    centre = grid_centre_yx(y, x) if about == "grid" else (1.25, -1.5)
    assert grid_centre_yx(y, x) == (0.25, -0.5)
    mask = annulus_mask(y, x, centre_yx=centre, inner_arcsec=inner, outer_arcsec=outer)
    assert [tuple(index) for index in np.argwhere(mask)] == expected
    if about == "grid":
        assert len(expected) == 12
    for radii in ((-0.1, 1.0), (1.0, 1.0), (1.5, 1.0), (np.nan, 1.0)):
        with pytest.raises(ValueError, match="0 <= inner < outer"):
            annulus_mask(y, x, centre_yx=centre, inner_arcsec=radii[0], outer_arcsec=radii[1])


def test_source_snr_mask_uses_strict_threshold_and_sigma_floor():
    source = np.array([[10.0, 5.0, 0.0], [1.0e-11, 4.0e-12, 1.0]])
    sigma = np.array([[1.0, 1.0, 1.0], [0.0, 1.0e-13, 0.1]])
    np.testing.assert_array_equal(source_snr_mask(source, sigma, 5.0), [[True, False, False], [True, False, True]])


def test_masked_covariance_keeps_rows_and_columns_of_unmasked_pixels():
    mask = np.array([[True, False], [False, True]])
    sigma = np.array([[2.0, 1.0], [1.0, 3.0]])
    covariance = np.array([[4.0, 0.1, 0.2, 1.5],
                           [0.1, 1.0, 0.3, 0.4],
                           [0.2, 0.3, 1.0, 0.5],
                           [1.5, 0.4, 0.5, 9.0]])
    space = build_data_space(mask, sigma, covariance)
    assert space.pixel_count == 2 and space.whitener.mode == "dense"
    block = np.array([[4.0, 1.5], [1.5, 9.0]])
    values = np.array([[1.0, -0.5], [2.0, 0.25]])
    whitened = space.whiten(values)
    np.testing.assert_allclose(whitened.T @ whitened, values.T @ np.linalg.solve(block, values), rtol=1e-12)
    diagonal = build_data_space(mask, sigma, None)
    np.testing.assert_array_equal(diagonal.whiten(np.array([2.0, 6.0])), [1.0, 2.0])


def test_dense_covariance_must_match_the_noise_map(tmp_path):
    mask = np.array([[True, True], [False, True]])
    sigma = np.array([[1.0, 2.0], [3.0, 0.5]])
    covariance = np.diag(sigma.reshape(-1) ** 2)
    covariance[0, 1] = covariance[1, 0] = 0.3
    np.save(tmp_path / "covariance.npy", covariance)
    loaded = load_noise_covariance(tmp_path / "covariance.npy", 4)
    np.testing.assert_array_equal(loaded, covariance)
    build_data_space(mask, sigma, loaded)
    unmasked_differs = covariance.copy()
    unmasked_differs[2, 2] = 100.0
    build_data_space(mask, sigma, unmasked_differs)
    other = covariance.copy()
    other[3, 3] *= 1.0 + 2.0e-6
    with pytest.raises(ValueError, match="built for another observation"):
        build_data_space(mask, sigma, other)
    with pytest.raises(ValueError, match=r"shape \(4, 4\)"):
        build_data_space(mask, sigma, covariance[:3, :3])
    with pytest.raises(ValueError, match="expected"):
        load_noise_covariance(tmp_path / "covariance.npy", 9)
    covariance[1, 1] = np.nan
    np.save(tmp_path / "broken.npy", covariance)
    with pytest.raises(ValueError, match="non-finite"):
        load_noise_covariance(tmp_path / "broken.npy", 4)


def test_design_stacks_masked_images_as_row_major_columns():
    mask = np.array([[True, False, True], [False, True, True]])
    space = DataSpace(mask, Whitener.from_sigma(np.array([1.0, 2.0, 4.0, 0.5])))
    first = np.arange(6.0).reshape(2, 3)
    second = 10.0 * first + 1.0
    np.testing.assert_array_equal(space.flatten(first), [0.0, 2.0, 4.0, 5.0])
    np.testing.assert_array_equal(space.design([first, second]), [[0.0, 1.0], [2.0, 21.0], [4.0, 41.0],
                                                                  [5.0, 51.0]])
    for whitener in (space.whitener, Whitener.from_covariance(np.diag([1.0, 4.0, 16.0, 0.25]))):
        empty = DataSpace(mask, whitener)
        assert empty.design([]).shape == (4, 0)
        assert empty.whiten(empty.design([])).shape == (4, 0)
    with pytest.raises(ValueError, match="differs from the mask shape"):
        space.flatten(np.ones((3, 2)))
    with pytest.raises(ValueError, match="sigma_adu shape"):
        build_data_space(mask, np.ones((3, 2)), None)
    with pytest.raises(ValueError, match="covers 3 pixels"):
        DataSpace(mask, Whitener.from_sigma(np.ones(3)))


@pytest.mark.parametrize("build, kind", [
    (lambda: source_snr_mask(np.ones((3, 3)), np.ones((3, 3)), 2.0), "source_snr"),
    (lambda: annulus_mask(*unit_grid((3, 3)), centre_yx=(0.0, 0.0), inner_arcsec=5.0, outer_arcsec=6.0),
     "annulus"),
    (lambda: psf_border_mask((4, 6), (5, 3)), "psf_border"),
    (lambda: build_data_space(np.zeros((2, 2), dtype=bool), np.ones((2, 2)), None), "data space"),
], ids=["source_snr", "annulus", "psf_border", "data-space"])
def test_empty_masks_raise(build, kind):
    with pytest.raises(ValueError, match=f"the {kind} mask selects no pixels"):
        build()
