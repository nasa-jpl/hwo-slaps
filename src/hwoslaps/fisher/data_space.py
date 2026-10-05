"""The pixels a forecast uses and the noise metric on them.

Masks are boolean images, ``True`` where a pixel enters the statistic:

- ``all_pixels``: every pixel.
- ``source_snr``: ``source_adu / max(sigma_adu, 1e-12) > snr_min`` (strict), with
  ``source_adu`` the source-plane light alone, ``(rate * t) / gain``.
- ``annulus``: pixel centres with ``hypot(y - cy, x - cx)`` in the closed
  interval ``[inner, outer]``, about the lens centre or the grid centre.
- ``psf_border``: every pixel minus a border of ``k // 2`` rows and columns on
  each edge for a model kernel of shape ``k``; the one rule shared by forecasts
  and nonlinear fits.

An empty mask raises naming its kind. The data space flattens images over the
mask in row-major order and whitens with the diagonal noise map or, when a
dense covariance over the full image is given, with its masked block.
"""

from __future__ import annotations

import os
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike

from ..identity import array_digest
from .statistics import Whitener

__all__ = [
    "DataSpace", "all_pixels_mask", "annulus_mask", "build_data_space", "grid_centre_yx",
    "load_noise_covariance", "psf_border_mask", "source_snr_mask",
]


def _shape(shape: Sequence[int], what: str) -> tuple[int, int]:
    values = tuple(shape)
    if len(values) != 2 or not all(isinstance(n, (int, np.integer)) and not isinstance(n, bool) and n > 0
                                   for n in values):
        raise ValueError(f"{what} must be two positive integers, got {shape!r}")
    return int(values[0]), int(values[1])


def _image(values: ArrayLike, what: str) -> np.ndarray:
    image = np.asarray(values, dtype=float)
    if image.ndim != 2:
        raise ValueError(f"{what} must be a 2-D image, got shape {image.shape}")
    if not np.all(np.isfinite(image)):
        raise ValueError(f"{what} contains non-finite values")
    return image


def _non_empty(mask: np.ndarray, kind: str) -> np.ndarray:
    if not np.any(mask):
        raise ValueError(f"the {kind} mask selects no pixels")
    return mask


def all_pixels_mask(shape: tuple[int, int]) -> np.ndarray:
    """Every pixel of an image of ``shape``."""
    return np.ones(_shape(shape, "shape"), dtype=bool)


def source_snr_mask(source_adu: np.ndarray, sigma_adu: np.ndarray, snr_min: float) -> np.ndarray:
    """Pixels where the source-plane light exceeds ``snr_min`` times the noise (strict)."""
    source = _image(source_adu, "source_adu")
    sigma = _image(sigma_adu, "sigma_adu")
    if source.shape != sigma.shape:
        raise ValueError(f"source_adu {source.shape} and sigma_adu {sigma.shape} differ in shape")
    if isinstance(snr_min, bool) or not np.isfinite(snr_min) or snr_min <= 0.0:
        raise ValueError(f"snr_min must be a positive finite number, got {snr_min!r}")
    return _non_empty(source / np.maximum(sigma, 1.0e-12) > snr_min, "source_snr")


def annulus_mask(y_arcsec: np.ndarray, x_arcsec: np.ndarray, *, centre_yx: tuple[float, float],
                 inner_arcsec: float, outer_arcsec: float) -> np.ndarray:
    """Pixel centres in the closed annulus ``[inner, outer]`` about ``centre_yx``."""
    y = _image(y_arcsec, "y_arcsec")
    x = _image(x_arcsec, "x_arcsec")
    if y.shape != x.shape:
        raise ValueError(f"y_arcsec {y.shape} and x_arcsec {x.shape} differ in shape")
    centre = np.asarray(centre_yx, dtype=float)
    if centre.shape != (2,) or not np.all(np.isfinite(centre)):
        raise ValueError(f"centre_yx must be two finite coordinates, got {centre_yx!r}")
    inner, outer = float(inner_arcsec), float(outer_arcsec)
    if not (np.isfinite(inner) and np.isfinite(outer) and 0.0 <= inner < outer):
        raise ValueError(f"the annulus needs finite radii with 0 <= inner < outer, got ({inner_arcsec!r}, "
                         f"{outer_arcsec!r})")
    radius = np.hypot(y - float(centre[0]), x - float(centre[1]))
    return _non_empty((radius >= inner) & (radius <= outer), "annulus")


def psf_border_mask(shape: tuple[int, int], kernel_shape: tuple[int, int]) -> np.ndarray:
    """Every pixel minus a border of ``k // 2`` rows and columns per edge for a kernel of shape ``k``."""
    rows, columns = _shape(shape, "shape")
    y_half, x_half = (n // 2 for n in _shape(kernel_shape, "kernel_shape"))
    mask = np.ones((rows, columns), dtype=bool)
    if y_half > 0:
        mask[:y_half, :] = False
        mask[-y_half:, :] = False
    if x_half > 0:
        mask[:, :x_half] = False
        mask[:, -x_half:] = False
    return _non_empty(mask, "psf_border")


def grid_centre_yx(y_arcsec: np.ndarray, x_arcsec: np.ndarray) -> tuple[float, float]:
    """Geometric centre of a pixel grid: the midpoint of its extreme pixel centres per axis."""
    y = _image(y_arcsec, "y_arcsec")
    x = _image(x_arcsec, "x_arcsec")
    return 0.5 * (float(np.max(y)) + float(np.min(y))), 0.5 * (float(np.max(x)) + float(np.min(x)))


def load_noise_covariance(path: str | os.PathLike[str], n_pixels: int) -> np.ndarray:
    """A finite ``(n_pixels, n_pixels)`` float covariance over the full image from a ``.npy`` file."""
    if isinstance(n_pixels, bool) or not isinstance(n_pixels, (int, np.integer)) or n_pixels <= 0:
        raise ValueError(f"n_pixels must be a positive integer, got {n_pixels!r}")
    covariance = np.load(path, allow_pickle=False)
    if not isinstance(covariance, np.ndarray) or covariance.dtype.kind not in "iuf":
        raise ValueError(f"{os.fspath(path)} must hold one numeric .npy array")
    if covariance.shape != (n_pixels, n_pixels):
        raise ValueError(f"{os.fspath(path)} holds shape {covariance.shape}, expected ({n_pixels}, {n_pixels})")
    covariance = covariance.astype(float)
    if not np.all(np.isfinite(covariance)):
        raise ValueError(f"{os.fspath(path)} contains non-finite values")
    return covariance


@dataclass(frozen=True, eq=False)
class DataSpace:
    """A boolean pixel mask and the whitener of the masked noise; the mask is read-only."""

    mask: np.ndarray
    whitener: Whitener

    def __post_init__(self) -> None:
        mask = np.asarray(self.mask)
        if mask.dtype != bool or mask.ndim != 2:
            raise ValueError(f"mask must be a 2-D boolean image, got dtype {mask.dtype} and shape {mask.shape}")
        if not isinstance(self.whitener, Whitener):
            raise ValueError("whitener must be a Whitener")
        if self.whitener.size != np.count_nonzero(mask):
            raise ValueError(f"the whitener covers {self.whitener.size} pixels, the mask "
                             f"{np.count_nonzero(mask)}")
        frozen = mask.copy()
        frozen.setflags(write=False)
        object.__setattr__(self, "mask", frozen)

    def __setstate__(self, state: dict[str, object]) -> None:
        """Unpickle through the constructor checks; numpy restores arrays writeable."""
        self.__dict__.update(state)
        self.__post_init__()

    @property
    def pixel_count(self) -> int:
        """Number of masked pixels."""
        return int(self.whitener.size)

    def flatten(self, image: ArrayLike) -> np.ndarray:
        """The masked pixels of an image, in row-major order."""
        values = np.asarray(image, dtype=float)
        if values.shape != self.mask.shape:
            raise ValueError(f"image shape {values.shape} differs from the mask shape {self.mask.shape}")
        return values[self.mask]

    def design(self, images: Sequence[ArrayLike]) -> np.ndarray:
        """One column per image, ``(pixel_count, len(images))``; zero images give zero columns."""
        if len(images) == 0:
            return np.empty((self.pixel_count, 0), dtype=float)
        return np.column_stack([self.flatten(image) for image in images])

    def whiten(self, values: ArrayLike) -> np.ndarray:
        """Whiten a masked vector or the columns of a masked matrix."""
        return self.whitener.apply(values)

    def digest(self) -> str:
        """Content digest of the mask."""
        return array_digest(self.mask)


def build_data_space(mask: np.ndarray, sigma_adu: np.ndarray, covariance: np.ndarray | None) -> DataSpace:
    """The data space of a mask, the noise map and an optional dense covariance over the full image.

    A covariance must agree with the noise map on the masked pixels,
    ``diag(C) = sigma_adu**2`` within 1e-6 relative, so a file built for another
    observation is refused.
    """
    mask_array = np.asarray(mask)
    if mask_array.dtype != bool or mask_array.ndim != 2:
        raise ValueError(f"mask must be a 2-D boolean image, got dtype {mask_array.dtype} and shape "
                         f"{mask_array.shape}")
    _non_empty(mask_array, "data space")
    sigma = _image(sigma_adu, "sigma_adu")
    if sigma.shape != mask_array.shape:
        raise ValueError(f"sigma_adu shape {sigma.shape} differs from the mask shape {mask_array.shape}")
    if covariance is None:
        return DataSpace(mask_array, Whitener.from_sigma(sigma[mask_array]))
    n_pixels = mask_array.size
    full = np.asarray(covariance, dtype=float)
    if full.shape != (n_pixels, n_pixels):
        raise ValueError(f"covariance must have shape ({n_pixels}, {n_pixels}) for an image of "
                         f"{mask_array.shape}, got {full.shape}")
    selected = mask_array.reshape(-1)
    variance = sigma[mask_array] ** 2
    if not np.allclose(np.diag(full)[selected], variance, rtol=1.0e-6, atol=0.0):
        raise ValueError("the covariance diagonal differs from sigma_adu**2 on the masked pixels by more than "
                         "1e-6 relative: it was built for another observation")
    return DataSpace(mask_array, Whitener.from_covariance(full[np.ix_(selected, selected)]))
