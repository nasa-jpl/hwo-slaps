"""Image-space adapters for the Fisher / Asimov core.

These helpers convert 2D mean images, noise maps, and derivative images into the
1D vectors required by :mod:`fisher_core`.  They are intentionally kept
free of AutoLens-specific object types so they can be reused by multiple forward
models.
"""

from __future__ import annotations

from typing import Callable, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from .fisher_core import (
    AsimovAmplitudeResult,
    SpuriousAmplitudeResult,
    SystematicModeScanResult,
    SignalBankResult,
    compute_asimov_detectability,
    compute_spurious_amplitude,
    evaluate_signal_bank,
    scan_systematic_modes,
)


Array2DLike = np.ndarray


def _as_2d_image(
    image: Array2DLike,
    *,
    name: str,
    expected_shape: Optional[Tuple[int, int]] = None,
) -> np.ndarray:
    """Coerce and validate one finite 2D image."""
    arr = np.asarray(image, dtype=float)
    if arr.ndim != 2:
        raise ValueError(f"{name} must be a 2D array.")
    if expected_shape is not None and arr.shape != expected_shape:
        raise ValueError(
            f"{name} shape {arr.shape} does not match expected image shape {expected_shape}."
        )
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} contains non-finite values.")
    return arr


def _as_2d_image_sequence(
    images: Sequence[Array2DLike],
    *,
    name: str,
    expected_shape: Optional[Tuple[int, int]] = None,
) -> List[np.ndarray]:
    """Validate a non-empty sequence of finite 2D images."""
    if len(images) == 0:
        raise ValueError(f"{name} must contain at least one element.")
    return [
        _as_2d_image(image, name=f"{name}[{i}]", expected_shape=expected_shape)
        for i, image in enumerate(images)
    ]


def validate_mask(mask: Optional[np.ndarray], shape: Tuple[int, int]) -> np.ndarray:
    """Return a boolean mask with ``True`` meaning "use this pixel"."""
    if mask is None:
        return np.ones(shape, dtype=bool)
    mask_arr = np.asarray(mask)
    if mask_arr.shape != shape:
        raise ValueError(f"mask shape {mask_arr.shape} does not match image shape {shape}.")
    if mask_arr.dtype != bool:
        mask_arr = mask_arr.astype(bool)
    return mask_arr


def flatten_masked_image(image: Array2DLike, mask: Optional[np.ndarray] = None) -> np.ndarray:
    """Flatten a 2D image over the pixels selected by ``mask``."""
    img = _as_2d_image(image, name="image")
    use = validate_mask(mask, img.shape)
    flat = img[use]
    if flat.size == 0:
        raise ValueError("mask selects zero pixels.")
    return flat


def stack_masked_images(
    images: Sequence[Array2DLike],
    mask: Optional[np.ndarray] = None,
    expected_shape: Optional[Tuple[int, int]] = None,
) -> np.ndarray:
    """Stack many 2D derivative images into a masked design matrix.

    The returned array has shape ``(n_selected_pixels, n_images)``.
    """
    validated = _as_2d_image_sequence(images, name="images", expected_shape=expected_shape)
    first = validated[0]
    use = validate_mask(mask, first.shape)
    cols: List[np.ndarray] = []
    for arr in validated:
        cols.append(arr[use])
    return np.column_stack(cols)


def extract_masked_covariance(covariance: np.ndarray, mask: Optional[np.ndarray], image_shape: Tuple[int, int]) -> np.ndarray:
    """Extract the sub-covariance corresponding to the selected pixels."""
    cov = np.asarray(covariance, dtype=float)
    n_pix = int(np.prod(image_shape))
    if cov.shape != (n_pix, n_pix):
        raise ValueError(
            f"covariance must have shape ({n_pix}, {n_pix}) for image shape {image_shape}."
        )
    use = validate_mask(mask, image_shape).reshape(-1)
    return cov[np.ix_(use, use)]


def compute_asimov_from_images(
    smooth_mean_image: Array2DLike,
    subhalo_mean_image: Array2DLike,
    sigma_image: Optional[Array2DLike] = None,
    nuisance_images: Optional[Sequence[Array2DLike]] = None,
    prior_precision=None,
    mask: Optional[np.ndarray] = None,
    amplitude_true: float = 1.0,
    nuisance_names: Optional[Sequence[str]] = None,
    covariance: Optional[np.ndarray] = None,
) -> AsimovAmplitudeResult:
    """Compute Fisher / Asimov Asimov detectability directly from 2D images."""
    smooth = _as_2d_image(smooth_mean_image, name="smooth_mean_image")
    subhalo = _as_2d_image(
        subhalo_mean_image,
        name="subhalo_mean_image",
        expected_shape=smooth.shape,
    )
    signal = flatten_masked_image(subhalo - smooth, mask=mask)

    sigma = None
    if sigma_image is not None:
        sigma_arr = _as_2d_image(
            sigma_image,
            name="sigma_image",
            expected_shape=smooth.shape,
        )
        sigma = flatten_masked_image(sigma_arr, mask=mask)

    nuisance = None
    if nuisance_images is not None and len(nuisance_images) > 0:
        nuisance = stack_masked_images(
            nuisance_images,
            mask=mask,
            expected_shape=smooth.shape,
        )

    cov = None
    if covariance is not None:
        cov = extract_masked_covariance(covariance, mask=mask, image_shape=smooth.shape)

    return compute_asimov_detectability(
        signal=signal,
        nuisance_jacobian=nuisance,
        sigma=sigma,
        covariance=cov,
        prior_precision=prior_precision,
        amplitude_true=amplitude_true,
        nuisance_names=nuisance_names,
    )


def evaluate_signal_bank_from_images(
    smooth_mean_image: Array2DLike,
    subhalo_mean_images: Sequence[Array2DLike],
    sigma_image: Optional[Array2DLike] = None,
    nuisance_images: Optional[Sequence[Array2DLike]] = None,
    prior_precision=None,
    mask: Optional[np.ndarray] = None,
    amplitude_true=1.0,
    nuisance_names: Optional[Sequence[str]] = None,
    covariance: Optional[np.ndarray] = None,
) -> SignalBankResult:
    """Vectorized map/mass-sweep helper operating on 2D image templates."""
    smooth = _as_2d_image(smooth_mean_image, name="smooth_mean_image")
    if len(subhalo_mean_images) == 0:
        raise ValueError("subhalo_mean_images must contain at least one template.")
    signal_vectors = []
    for i, img in enumerate(subhalo_mean_images):
        subhalo = _as_2d_image(
            img,
            name=f"subhalo_mean_images[{i}]",
            expected_shape=smooth.shape,
        )
        signal_vectors.append(flatten_masked_image(subhalo - smooth, mask=mask))
    signals = np.vstack(signal_vectors)

    sigma = None
    if sigma_image is not None:
        sigma_arr = _as_2d_image(
            sigma_image,
            name="sigma_image",
            expected_shape=smooth.shape,
        )
        sigma = flatten_masked_image(sigma_arr, mask=mask)

    nuisance = None
    if nuisance_images is not None and len(nuisance_images) > 0:
        nuisance = stack_masked_images(
            nuisance_images,
            mask=mask,
            expected_shape=smooth.shape,
        )

    cov = None
    if covariance is not None:
        cov = extract_masked_covariance(covariance, mask=mask, image_shape=smooth.shape)

    return evaluate_signal_bank(
        signal_bank=signals,
        nuisance_jacobian=nuisance,
        sigma=sigma,
        covariance=cov,
        prior_precision=prior_precision,
        amplitude_true=amplitude_true,
        nuisance_names=nuisance_names,
    )


def compute_spurious_from_images(
    smooth_mean_image: Array2DLike,
    subhalo_mean_image: Array2DLike,
    bias_image: Array2DLike,
    sigma_image: Optional[Array2DLike] = None,
    nuisance_images: Optional[Sequence[Array2DLike]] = None,
    prior_precision=None,
    mask: Optional[np.ndarray] = None,
    nuisance_names: Optional[Sequence[str]] = None,
    covariance: Optional[np.ndarray] = None,
) -> SpuriousAmplitudeResult:
    """Compute spurious subhalo amplitude from a 2D systematic bias image."""
    smooth = _as_2d_image(smooth_mean_image, name="smooth_mean_image")
    subhalo = _as_2d_image(
        subhalo_mean_image,
        name="subhalo_mean_image",
        expected_shape=smooth.shape,
    )
    bias = _as_2d_image(bias_image, name="bias_image", expected_shape=smooth.shape)

    signal = flatten_masked_image(subhalo - smooth, mask=mask)
    bias_flat = flatten_masked_image(bias, mask=mask)

    sigma = None
    if sigma_image is not None:
        sigma_arr = _as_2d_image(
            sigma_image,
            name="sigma_image",
            expected_shape=smooth.shape,
        )
        sigma = flatten_masked_image(sigma_arr, mask=mask)

    nuisance = None
    if nuisance_images is not None and len(nuisance_images) > 0:
        nuisance = stack_masked_images(
            nuisance_images,
            mask=mask,
            expected_shape=smooth.shape,
        )

    cov = None
    if covariance is not None:
        cov = extract_masked_covariance(covariance, mask=mask, image_shape=smooth.shape)

    return compute_spurious_amplitude(
        signal=signal,
        bias=bias_flat,
        nuisance_jacobian=nuisance,
        sigma=sigma,
        covariance=cov,
        prior_precision=prior_precision,
        nuisance_names=nuisance_names,
    )


def scan_systematic_modes_from_images(
    smooth_mean_image: Array2DLike,
    subhalo_mean_image: Array2DLike,
    systematic_mode_images: Sequence[Array2DLike],
    sigma_image: Optional[Array2DLike] = None,
    nuisance_images: Optional[Sequence[Array2DLike]] = None,
    prior_precision=None,
    mask: Optional[np.ndarray] = None,
    nuisance_names: Optional[Sequence[str]] = None,
    mode_names: Optional[Sequence[str]] = None,
    mode_sigmas=None,
    z_tolerance: Optional[float] = 1.0,
    systematic_covariance: Optional[np.ndarray] = None,
    covariance: Optional[np.ndarray] = None,
    progress: Optional[Callable[[Iterable[int]], Iterable[int]]] = None,
) -> SystematicModeScanResult:
    """Mode-by-mode PSF/systematics scan working directly on 2D images."""
    smooth = _as_2d_image(smooth_mean_image, name="smooth_mean_image")
    subhalo = _as_2d_image(
        subhalo_mean_image,
        name="subhalo_mean_image",
        expected_shape=smooth.shape,
    )

    signal = flatten_masked_image(subhalo - smooth, mask=mask)
    modes = stack_masked_images(
        systematic_mode_images,
        mask=mask,
        expected_shape=smooth.shape,
    )

    sigma = None
    if sigma_image is not None:
        sigma_arr = _as_2d_image(
            sigma_image,
            name="sigma_image",
            expected_shape=smooth.shape,
        )
        sigma = flatten_masked_image(sigma_arr, mask=mask)

    nuisance = None
    if nuisance_images is not None and len(nuisance_images) > 0:
        nuisance = stack_masked_images(
            nuisance_images,
            mask=mask,
            expected_shape=smooth.shape,
        )

    cov = None
    if covariance is not None:
        cov = extract_masked_covariance(covariance, mask=mask, image_shape=smooth.shape)

    return scan_systematic_modes(
        signal=signal,
        systematic_modes=modes,
        nuisance_jacobian=nuisance,
        sigma=sigma,
        covariance=cov,
        prior_precision=prior_precision,
        nuisance_names=nuisance_names,
        mode_names=mode_names,
        mode_sigmas=mode_sigmas,
        z_tolerance=z_tolerance,
        systematic_covariance=systematic_covariance,
        progress=progress,
    )


__all__ = [
    "validate_mask",
    "flatten_masked_image",
    "stack_masked_images",
    "extract_masked_covariance",
    "compute_asimov_from_images",
    "evaluate_signal_bank_from_images",
    "compute_spurious_from_images",
    "scan_systematic_modes_from_images",
]
