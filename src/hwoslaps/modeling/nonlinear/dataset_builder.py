"""Build PyAutoLens datasets for nonlinear metric validation."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from copy import deepcopy
import hashlib
import json
from typing import Any, Dict, Optional, Tuple

import numpy as np

from ...psf.utils import (
    make_pyauto_convolver,
    make_pyauto_kernel,
    pyauto_kernel_native,
)
from ...psf.mismatch import _kernel_sha256

RENDERING_CONTRACT_REVISION = "uniform-core-ring-v2.1"


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
    objective_version: str = "legacy_ring1_v1"
    generation_sub_size: Optional[int] = None
    light_profile_sub_size: Optional[int] = None
    blurring_sub_size: Optional[int] = None

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
    objective_version: str = "legacy_ring1_v1",
    generation_sub_size: Optional[int] = None,
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
    if objective_version not in ("legacy_ring1_v1", "consistent_sampling_v2"):
        raise ValueError("Unsupported nonlinear objective_version")
    if objective_version == "consistent_sampling_v2":
        from ...lensing.sampling import positive_sub_size

        generation_sub_size = positive_sub_size(generation_sub_size)
        recorded_size = getattr(observation, "metadata", {}).get("generation_sub_size")
        if recorded_size is None or positive_sub_size(recorded_size) != generation_sub_size:
            raise ValueError("Declared sampling differs from actual generation sampling")
        dataset = al.Imaging(
            data=data_array, noise_map=noise_array, psf=psf,
            over_sample_size_lp=generation_sub_size,
        )
        _set_consistent_blurring(dataset, generation_sub_size)
    else:
        if generation_sub_size is not None:
            raise ValueError("Legacy reconstruction does not accept a sampling override")
        recorded_size = getattr(observation, "metadata", {}).get("generation_sub_size")
        if recorded_size is not None:
            from ...lensing.sampling import positive_sub_size

            if positive_sub_size(recorded_size) != 4:
                raise ValueError("Legacy objective requires historical generation sampling of 4")
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
        objective_version=objective_version,
        generation_sub_size=(generation_sub_size if objective_version == "consistent_sampling_v2"
                             else recorded_size),
    )
    if objective_version == "consistent_sampling_v2":
        metadata = _sampling_metadata(dataset, metadata, generation_sub_size)
    else:
        from ...lensing.sampling import actual_sub_size

        metadata = replace(
            metadata, light_profile_sub_size=actual_sub_size(dataset.grids.lp),
            blurring_sub_size=actual_sub_size(dataset.grids.blurring),
        )
    return dataset, metadata


def _set_consistent_blurring(dataset, generation_sub_size):
    """Set the pinned AutoArray cached ring on a fresh dataset only."""
    import autolens as al
    from ...lensing.sampling import actual_sub_size, positive_sub_size

    size = positive_sub_size(generation_sub_size)
    if actual_sub_size(dataset.grids.lp) != size:
        raise ValueError("Light-profile sampling differs from generation")
    ring = dataset.grids.blurring
    if ring is not None:
        # AutoArray has no public ring-size argument in the pinned runtime.
        # Assert the actual grid afterwards so a cache API change fails closed.
        dataset.grids._blurring = al.Grid2D.from_mask(
            mask=ring.mask, over_sample_size=size,
        )
        if actual_sub_size(dataset.grids.blurring) not in (None, size):
            raise RuntimeError("AutoArray ring sampling override did not take effect")
    dataset._hwoslaps_objective_version = "consistent_sampling_v2"


def _sampling_metadata(dataset, metadata, generation_sub_size):
    from ...lensing.sampling import actual_sub_size

    return replace(
        metadata, objective_version="consistent_sampling_v2",
        generation_sub_size=int(generation_sub_size),
        light_profile_sub_size=actual_sub_size(dataset.grids.lp),
        blurring_sub_size=actual_sub_size(dataset.grids.blurring),
    )


def consistent_sampling_copy(dataset, metadata, generation_sub_size=4):
    """Clone an archived dataset and apply an explicitly versioned ring rule.

    Caller must first verify the historical scalar/model identity and bind the
    actual generation sampling from its execution provenance. No old fit result
    or convergence flag is returned as accepted output for the new objective.
    """
    import autolens as al
    from ...lensing.sampling import actual_sub_size, positive_sub_size

    size = positive_sub_size(generation_sub_size)
    if actual_sub_size(dataset.grids.lp) != size:
        raise ValueError("Cannot migrate a core/generation sampling mismatch")
    if isinstance(metadata, dict):
        try:
            metadata = NonlinearDatasetMetadata(**metadata)
        except (TypeError, ValueError) as exc:
            raise ValueError("Historical dataset metadata cannot be reconstructed") from exc
    if not isinstance(metadata, NonlinearDatasetMetadata):
        raise ValueError("Historical metadata must be NonlinearDatasetMetadata or its dictionary")
    if metadata.objective_version != "legacy_ring1_v1":
        raise ValueError("Migration requires an explicitly historical baseline")
    if metadata.generation_sub_size is None:
        raise ValueError("Migration requires verified generation sampling metadata")
    if positive_sub_size(metadata.generation_sub_size) != size:
        raise ValueError("Historical generation metadata differs from migration sampling")
    if actual_sub_size(dataset.grids.blurring) not in (None, 1):
        raise ValueError("Historical migration requires ring1 or no external blurring pixels")
    before = {
        "data": np.array(dataset.data.native, copy=True),
        "noise": np.array(dataset.noise_map.native, copy=True),
        "psf": np.array(pyauto_kernel_native(dataset.psf), copy=True),
    }
    mask_copy = deepcopy(dataset.mask)
    psf_kernel = dataset.psf.kernel if hasattr(dataset.psf, "kernel") else dataset.psf
    psf = make_pyauto_convolver(make_pyauto_kernel(
        values=before["psf"].copy(), pixel_scales=psf_kernel.pixel_scales, normalize=False,
    ))
    candidate = al.Imaging(
        data=al.Array2D(values=before["data"].copy(), mask=mask_copy),
        noise_map=al.Array2D(values=before["noise"].copy(), mask=mask_copy),
        psf=psf, use_normalized_psf=False, over_sample_size_lp=size,
    )
    _set_consistent_blurring(candidate, size)
    for name, actual in [("data", candidate.data.native), ("noise", candidate.noise_map.native),
                         ("psf", pyauto_kernel_native(candidate.psf))]:
        if np.asarray(actual).tobytes() != before[name].tobytes():
            raise ValueError("Objective migration changed input bytes: " + name)
    for old, new in [(dataset.grids.lp, candidate.grids.lp),
                     (dataset.grids.blurring, candidate.grids.blurring)]:
        if (old is None) != (new is None):
            raise ValueError("Objective migration changed grid presence")
        if old is not None:
            if not np.array_equal(old.mask, new.mask) or not np.array_equal(old.array, new.array):
                raise ValueError("Objective migration changed grid mask/coordinates")
    return candidate, _sampling_metadata(candidate, metadata, size)


def _grid_geometry_identity(grid):
    """Bind actual ray coordinates, mask and geometry without byte copies."""
    if grid is None:
        return None

    def digest(value):
        if value is None:
            return None
        arr = np.ascontiguousarray(getattr(value, "array", value))
        h = hashlib.sha256()
        h.update(str(arr.dtype).encode())
        h.update(json.dumps(list(arr.shape)).encode())
        if arr.size:
            h.update(memoryview(arr).cast("B"))
        return h.hexdigest()

    return {
        "mask_sha256": digest(grid.mask), "coordinates_sha256": digest(grid),
        "subpixel_coordinates_sha256": digest(grid.over_sampled),
        "pixel_scales": list(grid.pixel_scales), "origin": list(grid.mask.origin),
    }


def rendering_identity(dataset, metadata):
    """Read the actual corrected-grid contract before hashing an analysis."""
    from ...lensing.sampling import actual_sub_size, positive_sub_size

    get = (metadata.get if isinstance(metadata, dict)
           else lambda key, default=None: getattr(metadata, key, default))
    version = get("objective_version", "legacy_ring1_v1")
    if version == "legacy_ring1_v1":
        if getattr(dataset, "_hwoslaps_objective_version", None) is not None:
            raise ValueError("Corrected dataset cannot use historical identity metadata")
        return None
    if version != "consistent_sampling_v2":
        raise ValueError("Unsupported rendering identity version")
    size = positive_sub_size(get("generation_sub_size"))
    core, ring = actual_sub_size(dataset.grids.lp), actual_sub_size(dataset.grids.blurring)
    if core != size or ring not in (None, size):
        raise ValueError("Generation/core/ring sampling contract is violated")
    if get("light_profile_sub_size") != core or get("blurring_sub_size") != ring:
        raise ValueError("Sampling metadata differs from actual grids")
    return {"objective_version": version, "renderer_contract_revision": RENDERING_CONTRACT_REVISION,
            "generation_sub_size": size, "light_profile_sub_size": core, "blurring_sub_size": ring,
            "external_blurring_pixels_present": ring is not None,
            "light_profile_geometry": _grid_geometry_identity(dataset.grids.lp),
            "blurring_geometry": _grid_geometry_identity(dataset.grids.blurring)}


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
