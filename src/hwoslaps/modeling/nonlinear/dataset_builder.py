"""Build PyAutoLens datasets for nonlinear metric validation."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np


@dataclass(frozen=True)
class NonlinearDatasetMetadata:
    """Metadata describing a nonlinear validation dataset.

    Parameters
    ----------
    dataset_kind : `str`
        Dataset type, either ``"asimov"`` or ``"noisy"``.
    data_units : `str`
        Unit label for the image data.
    background_treatment : `str`
        Background handling mode.
    sky_dark_background_adu : `float`
        Known sky-plus-dark pedestal in ADU per pixel.
    mask_name : `str`
        Name of the mask source.
    n_unmasked_pixels : `int`
        Number of pixels available to the fit.
    psf_truth_label : `str`
        Label for the PSF used to generate the data.
    psf_fit_label : `str`
        Label for the PSF supplied to the fit.
    """

    dataset_kind: str
    data_units: str
    background_treatment: str
    sky_dark_background_adu: float
    mask_name: str
    n_unmasked_pixels: int
    psf_truth_label: str
    psf_fit_label: str

    def to_dict(self) -> Dict[str, object]:
        """Convert metadata to a JSON-compatible dictionary."""
        return asdict(self)


def _validate_choice(value: str, allowed: Tuple[str, ...], name: str) -> str:
    """Validate a string against a fixed set of choices."""
    if value not in allowed:
        allowed_text = ", ".join(allowed)
        raise ValueError(f"{name} must be one of: {allowed_text}")
    return value


def known_sky_dark_background_adu(observation: Any) -> float:
    """Return the known sky-plus-dark pedestal in ADU per pixel.

    Parameters
    ----------
    observation : `object`
        HWO-SLAPS observation data.

    Returns
    -------
    background_adu : `float`
        Sky plus dark current contribution in ADU.
    """
    return float(
        (
            observation.sky_electrons_per_pixel
            + observation.dark_electrons_per_pixel
        )
        / observation.gain
    )


def source_only_data_adu(observation: Any, dataset_kind: str) -> np.ndarray:
    """Return source-only validation data in ADU.

    Parameters
    ----------
    observation : `object`
        HWO-SLAPS observation data.
    dataset_kind : `str`
        Dataset type, either ``"asimov"`` or ``"noisy"``.

    Returns
    -------
    data : `numpy.ndarray`
        Source-only data in ADU.
    """
    dataset_kind = _validate_choice(dataset_kind, ("asimov", "noisy"), "dataset_kind")
    if dataset_kind == "asimov":
        return np.asarray(observation.source_electrons, dtype=float) / float(observation.gain)
    return np.asarray(observation.data.native, dtype=float)


def data_array_from_observation(
    observation: Any,
    dataset_kind: str,
    background_treatment: str = "subtract_known",
) -> np.ndarray:
    """Build the image data used by PyAutoLens validation.

    Parameters
    ----------
    observation : `object`
        HWO-SLAPS observation data.
    dataset_kind : `str`
        Dataset type, either ``"asimov"`` or ``"noisy"``.
    background_treatment : `str`, optional
        Background handling mode. Supported values are ``"subtract_known"``
        and ``"none"``.

    Returns
    -------
    data : `numpy.ndarray`
        Validation data in ADU.
    """
    background_treatment = _validate_choice(
        background_treatment,
        ("subtract_known", "none"),
        "background_treatment",
    )
    data = source_only_data_adu(observation, dataset_kind)
    if dataset_kind == "noisy" and background_treatment == "subtract_known":
        data = data - known_sky_dark_background_adu(observation)
    return np.asarray(data, dtype=float)


def mask_from_fisher_use_mask(fisher_use_mask: np.ndarray, pixel_scale: float) -> Any:
    """Convert a Fisher include-mask to a PyAutoLens mask.

    Parameters
    ----------
    fisher_use_mask : `numpy.ndarray`
        Boolean Fisher mask where True means the pixel is used.
    pixel_scale : `float`
        Pixel scale in arcseconds.

    Returns
    -------
    mask : `autolens.Mask2D`
        PyAutoLens mask, where True means masked.
    """
    import autolens as al

    use_mask = np.asarray(fisher_use_mask, dtype=bool)
    if use_mask.ndim != 2:
        raise ValueError("fisher_use_mask must be a 2D boolean array")
    autolens_mask = np.logical_not(use_mask)
    try:
        return al.Mask2D(mask=autolens_mask, pixel_scales=float(pixel_scale))
    except TypeError:
        return al.Mask2D(values=autolens_mask, pixel_scales=float(pixel_scale))


def _all_false_mask(shape_native: Tuple[int, int], pixel_scale: float) -> Any:
    """Create an all-unmasked PyAutoLens mask."""
    import autolens as al

    return al.Mask2D.all_false(
        shape_native=tuple(shape_native),
        pixel_scales=float(pixel_scale),
    )


def _kernel_from_any(psf_for_fit: Any, pixel_scale: float) -> Any:
    """Return a PyAutoLens kernel from an existing kernel or array."""
    import autolens as al

    if hasattr(psf_for_fit, "native"):
        return psf_for_fit
    kernel_array = np.asarray(psf_for_fit, dtype=float)
    if kernel_array.ndim != 2:
        raise ValueError("psf_for_fit must be a 2D kernel or PyAutoLens Kernel2D")
    try:
        return al.Kernel2D.no_mask(values=kernel_array, pixel_scales=float(pixel_scale))
    except AttributeError:
        mask = _all_false_mask(kernel_array.shape, pixel_scale)
        return al.Kernel2D(values=kernel_array, mask=mask)


def imaging_from_observation(
    observation: Any,
    psf_for_fit: Optional[Any] = None,
    dataset_kind: str = "asimov",
    background_treatment: str = "subtract_known",
    mask_bool_use: Optional[np.ndarray] = None,
    psf_truth_label: str = "observation",
    psf_fit_label: str = "fit",
) -> Tuple[Any, NonlinearDatasetMetadata]:
    """Convert an HWO-SLAPS observation into a PyAutoLens dataset.

    Parameters
    ----------
    observation : `object`
        HWO-SLAPS observation data.
    psf_for_fit : `object`, optional
        PSF kernel supplied to the nonlinear fit. If None, use the
        observation PSF.
    dataset_kind : `str`, optional
        Dataset type, either ``"asimov"`` or ``"noisy"``.
    background_treatment : `str`, optional
        Background handling mode.
    mask_bool_use : `numpy.ndarray`, optional
        Boolean mask where True means the pixel is included.
    psf_truth_label : `str`, optional
        Label describing the PSF used to generate the data.
    psf_fit_label : `str`, optional
        Label describing the PSF used for fitting.

    Returns
    -------
    dataset : `autolens.Imaging`
        PyAutoLens imaging dataset.
    metadata : `NonlinearDatasetMetadata`
        Dataset provenance metadata.
    """
    import autolens as al

    data = data_array_from_observation(
        observation,
        dataset_kind=dataset_kind,
        background_treatment=background_treatment,
    )
    if mask_bool_use is None:
        mask = _all_false_mask(data.shape, observation.pixel_scale)
        mask_name = "all_pixels"
        n_unmasked_pixels = int(data.size)
    else:
        mask = mask_from_fisher_use_mask(mask_bool_use, observation.pixel_scale)
        mask_name = "fisher"
        n_unmasked_pixels = int(np.count_nonzero(mask_bool_use))

    psf = _kernel_from_any(
        observation.psf if psf_for_fit is None else psf_for_fit,
        observation.pixel_scale,
    )
    data_array = al.Array2D(values=data, mask=mask)
    noise_array = al.Array2D(values=np.asarray(observation.noise_map.native, dtype=float), mask=mask)
    dataset = al.Imaging(data=data_array, noise_map=noise_array, psf=psf)

    metadata = NonlinearDatasetMetadata(
        dataset_kind=dataset_kind,
        data_units="adu",
        background_treatment=background_treatment,
        sky_dark_background_adu=known_sky_dark_background_adu(observation),
        mask_name=mask_name,
        n_unmasked_pixels=n_unmasked_pixels,
        psf_truth_label=psf_truth_label,
        psf_fit_label=psf_fit_label,
    )
    return dataset, metadata
