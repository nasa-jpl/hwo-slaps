"""Build PyAutoLens datasets for nonlinear metric validation."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np

from ...psf.utils import (
    make_pyauto_convolver,
    make_pyauto_kernel,
    pyauto_kernel_native,
)
from ...psf.mismatch import _kernel_sha256


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
    psf_fit_supplied : `bool`, optional
        Whether the caller supplied a fit-side PSF.
    psf_fit_sha256 : `str`, optional
        Shape-aware digest of the actual dataset fit kernel.
    """

    dataset_kind: str
    data_units: str
    background_treatment: str
    sky_dark_background_adu: float
    mask_name: str
    n_unmasked_pixels: int
    psf_truth_label: str
    psf_fit_label: str
    psf_fit_supplied: bool = False
    psf_fit_sha256: str = ""

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


def source_only_data_electron_rate(observation: Any, dataset_kind: str) -> np.ndarray:
    """Return source-only validation data in electron-rate units.

    Parameters
    ----------
    observation : `object`
        HWO-SLAPS observation data.
    dataset_kind : `str`
        Dataset type, either ``"asimov"`` or ``"noisy"``.

    Returns
    -------
    data : `numpy.ndarray`
        Source-only data in electrons per second, matching the PyAutoLens
        light-profile intensity units used by the forward model.
    """
    dataset_kind = _validate_choice(dataset_kind, ("asimov", "noisy"), "dataset_kind")
    if dataset_kind == "asimov":
        # Copy: al.Array2D zeroes masked pixels in place, and a no-copy
        # view here would corrupt the caller's observation object.
        return np.array(observation.noiseless_source_eps, dtype=float)
    return (
        np.asarray(observation.data.native, dtype=float)
        * float(observation.gain)
        / float(observation.exposure_time)
    )


def source_only_data_adu(observation: Any, dataset_kind: str) -> np.ndarray:
    """Return the source-only electron rate (deprecated alias).

    Notes
    -----
    Deprecated alias for `source_only_data_electron_rate`; the name refers to
    ADU for historical reasons but the returned values are electron rates.
    """
    return source_only_data_electron_rate(observation, dataset_kind)


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
        Validation data in electrons per second.
    """
    background_treatment = _validate_choice(
        background_treatment,
        ("subtract_known", "none"),
        "background_treatment",
    )
    data = source_only_data_electron_rate(observation, dataset_kind)
    if dataset_kind == "noisy" and background_treatment == "subtract_known":
        data = data - (
            known_sky_dark_background_adu(observation)
            * float(observation.gain)
            / float(observation.exposure_time)
        )
    return np.asarray(data, dtype=float)


def noise_rate_from_observation(observation: Any) -> np.ndarray:
    """Return the observation noise map in electron-rate units."""
    return (
        np.asarray(observation.noise_map.native, dtype=float)
        * float(observation.gain)
        / float(observation.exposure_time)
    )


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


def _exclude_psf_edge_pixels(use_mask: np.ndarray, psf_shape: Tuple[int, int]) -> np.ndarray:
    """Exclude pixels whose PSF stencil would extend beyond the image.

    Parameters
    ----------
    use_mask : `numpy.ndarray`
        Boolean include-mask where True means use the pixel.
    psf_shape : `tuple` [`int`, `int`]
        Native PSF kernel shape.

    Returns
    -------
    use_mask : `numpy.ndarray`
        Include-mask with unsafe edge pixels removed.
    """
    use_mask = np.asarray(use_mask, dtype=bool).copy()
    if use_mask.ndim != 2:
        raise ValueError("use_mask must be a 2D boolean array")

    y_half = int(psf_shape[0]) // 2
    x_half = int(psf_shape[1]) // 2
    if y_half > 0:
        use_mask[:y_half, :] = False
        use_mask[-y_half:, :] = False
    if x_half > 0:
        use_mask[:, :x_half] = False
        use_mask[:, -x_half:] = False
    return use_mask


def _kernel_from_any(psf_for_fit: Any, pixel_scale: float) -> Any:
    """Return a PyAuto convolver from an existing PSF object or array."""
    if hasattr(psf_for_fit, "convolved_image_via_real_space_from"):
        return psf_for_fit
    if hasattr(psf_for_fit, "native"):
        return make_pyauto_convolver(psf_for_fit)
    kernel_array = np.asarray(psf_for_fit, dtype=float)
    if kernel_array.ndim != 2:
        raise ValueError("psf_for_fit must be a 2D kernel or PyAuto PSF object")
    return make_pyauto_convolver(
        make_pyauto_kernel(
            values=kernel_array,
            pixel_scales=float(pixel_scale),
            normalize=True,
        )
    )


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
    psf_fit_supplied = psf_for_fit is not None
    psf = _kernel_from_any(
        observation.psf if psf_for_fit is None else psf_for_fit,
        observation.pixel_scale,
    )
    psf_native = pyauto_kernel_native(psf)
    psf_shape = tuple(psf_native.shape)

    if mask_bool_use is None:
        use_mask = _exclude_psf_edge_pixels(
            np.ones(data.shape, dtype=bool),
            psf_shape=psf_shape,
        )
        mask = mask_from_fisher_use_mask(use_mask, observation.pixel_scale)
        mask_name = "all_pixels_minus_psf_border"
        n_unmasked_pixels = int(np.count_nonzero(use_mask))
    else:
        use_mask = _exclude_psf_edge_pixels(mask_bool_use, psf_shape=psf_shape)
        mask = mask_from_fisher_use_mask(use_mask, observation.pixel_scale)
        mask_name = "fisher_minus_psf_border"
        n_unmasked_pixels = int(np.count_nonzero(use_mask))

    data_array = al.Array2D(values=data, mask=mask)
    noise_array = al.Array2D(values=noise_rate_from_observation(observation), mask=mask)
    dataset = al.Imaging(data=data_array, noise_map=noise_array, psf=psf)

    # al.Imaging sum-normalizes the PSF at construction, so the recorded
    # digest must describe the kernel the fit actually consumes.
    fitted_psf_native = pyauto_kernel_native(dataset.psf)

    metadata = NonlinearDatasetMetadata(
        dataset_kind=dataset_kind,
        data_units="e_per_s",
        background_treatment=background_treatment,
        sky_dark_background_adu=known_sky_dark_background_adu(observation),
        mask_name=mask_name,
        n_unmasked_pixels=n_unmasked_pixels,
        psf_truth_label=psf_truth_label,
        psf_fit_label=psf_fit_label,
        psf_fit_supplied=psf_fit_supplied,
        psf_fit_sha256=_kernel_sha256(fitted_psf_native),
    )
    return dataset, metadata


def fitted_kernel_sha256(dataset, wrapped_kernel, kernel_pixel_scale):
    """Digest the as-fitted dataset PSF, bound to the executor kernel.

    ``al.Imaging`` sum-normalizes the PSF at construction, so the digest
    the executors bind through the validator guard must describe
    ``dataset.psf``, the kernel the fit actually consumes.

    Parameters
    ----------
    dataset : `autolens.Imaging`
        Dataset returned by `imaging_from_observation`.
    wrapped_kernel : `object`
        The PyAuto kernel or convolver the executor handed to the
        dataset builder.
    kernel_pixel_scale : `float`
        Kernel pixel scale in arcseconds per pixel.

    Returns
    -------
    digest : `str`
        Canonical SHA-256 of the as-fitted native kernel.

    Raises
    ------
    ValueError
        Raised if the dataset kernel is byte-identical to neither the
        wrapped kernel nor its sum-normalization.
    """
    fitted = np.ascontiguousarray(
        pyauto_kernel_native(dataset.psf),
        dtype=np.float64,
    )
    supplied = np.ascontiguousarray(
        pyauto_kernel_native(wrapped_kernel),
        dtype=np.float64,
    )
    if fitted.tobytes() != supplied.tobytes():
        normalized = np.ascontiguousarray(
            pyauto_kernel_native(
                make_pyauto_kernel(
                    values=supplied,
                    pixel_scales=float(kernel_pixel_scale),
                    normalize=True,
                )
            ),
            dtype=np.float64,
        )
        if fitted.tobytes() != normalized.tobytes():
            raise ValueError(
                "dataset PSF is byte-identical to neither the wrapped "
                "fit kernel nor its sum-normalization; the dataset does "
                "not carry the executor kernel"
            )
    return _kernel_sha256(fitted)
