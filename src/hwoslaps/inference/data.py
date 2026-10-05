"""The nonlinear imaging dataset: observation-kind units, fitted pixels, copied model kernel and sampling."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np

from ..fisher.data_space import psf_border_mask
from ..identity import KernelIdentity, array_digest
from ..observation.observation import Observation
from ..optics.kernels import DetectorPSF, make_convolver
from .settings import OBJECTIVE_VERSION

__all__ = ["FitData", "build_fit_data", "support_half_widths"]


@dataclass(frozen=True, eq=False)
class FitData:
    imaging: Any
    kind: Literal["expected", "noisy"]
    mask: np.ndarray
    record: Mapping[str, Any]

    @property
    def pixel_count(self) -> int:
        return int(np.count_nonzero(self.mask))


def support_half_widths(shape: tuple[int, int], pixel_scale_arcsec: float,
                        kernel_shape: tuple[int, int]) -> tuple[float, float]:
    """Border-valid hypothesis support in its own plane, in arcsec about the image centre."""
    pixels = tuple(int(n) // 2 - int(k) // 2 - 1 for n, k in zip(shape, kernel_shape, strict=True))
    if len(pixels) != 2 or any(value <= 0 for value in pixels):
        raise ValueError(f"image shape {shape} and kernel shape {kernel_shape} leave no valid fit support; "
                         "the fit support is set by the truth configuration; use a smaller kernel_shape arm")
    return tuple(value * float(pixel_scale_arcsec) for value in pixels)


def _sampling_size(grid: Any) -> int | None:
    if grid is None:
        return None
    sizes = np.asarray(grid.over_sample_size, dtype=int)
    unique = np.unique(sizes)
    if unique.size != 1:
        raise ValueError(f"the dataset grid has nonuniform sampling {unique.tolist()}")
    return int(unique[0])


def _consistent_sampling(imaging: Any, size: int) -> dict[str, int | None]:
    import autolens as al

    if _sampling_size(imaging.grids.lp) != size:
        raise ValueError("light-profile sampling differs from generation")
    ring = imaging.grids.blurring
    if ring is not None:
        # The pinned AutoArray runtime has no public ring-size argument. Verify the resulting grid.
        imaging.grids._blurring = al.Grid2D.from_mask(mask=ring.mask, over_sample_size=size)
        if _sampling_size(imaging.grids.blurring) != size:
            raise RuntimeError("AutoArray blurring-grid sampling override did not take effect")
    return {"generation": size, "light_profile": _sampling_size(imaging.grids.lp),
            "blurring": _sampling_size(imaging.grids.blurring)}


def build_fit_data(observation: Observation, model_kernel: DetectorPSF, *, mask_name: str,
                   base_mask: np.ndarray, over_sample_size: int) -> FitData:
    """Build a fresh AutoLens dataset in e-/s; normalization touches only a model-kernel copy."""
    if model_kernel.shape == (1, 1):
        raise ValueError("a 1x1 model kernel leaves AutoArray an empty blurring grid; use a larger kernel")
    if isinstance(over_sample_size, bool) or not isinstance(over_sample_size, (int, np.integer)) \
            or over_sample_size < 1:
        raise ValueError("over_sample_size must be an integer >= 1")
    if over_sample_size != observation.grid.over_sample_size:
        raise ValueError("fit sampling differs from the observation's generation sampling")
    if model_kernel.pixel_scale_arcsec != observation.pixel_scale_arcsec:
        raise ValueError("model kernel pixel scale differs from the observation")
    shape = tuple(observation.grid.shape)
    include = np.array(base_mask, dtype=bool, copy=True)
    if include.shape != shape:
        raise ValueError(f"base mask shape {include.shape} differs from observation shape {shape}")
    include &= psf_border_mask(shape, model_kernel.shape)
    if not np.any(include):
        raise ValueError("the fitted mask has no pixels after removing the model-kernel border")
    exposure = observation.exposure
    gain, time = float(exposure.detector.gain_e_per_adu), float(exposure.exposure_time_s)
    if observation.kind == "expected":
        data = np.array(observation.light_rate_e_per_s, dtype=float)
        background_adu = 0.0
    else:
        data = np.asarray(observation.data_adu, dtype=float) * gain / time
        background_adu = float((exposure.sky_rate_e_per_s * time
                                + exposure.detector.dark_current_e_per_s * time) / gain)
        data = data - (background_adu * gain / time)
    noise = np.asarray(observation.noise_map_adu, dtype=float) * gain / time
    import autolens as al

    mask = al.Mask2D(mask=~include, pixel_scales=observation.pixel_scale_arcsec)
    convolver = make_convolver(np.array(model_kernel.kernel, dtype=float, copy=True),
                               model_kernel.pixel_scale_arcsec)
    imaging = al.Imaging(data=al.Array2D(values=data, mask=mask), noise_map=al.Array2D(values=noise, mask=mask),
                         psf=convolver, over_sample_size_lp=over_sample_size)
    sampling = _consistent_sampling(imaging, int(over_sample_size))
    fitted = np.asarray(imaging.psf.kernel.native, dtype=float)
    include.setflags(write=False)
    record = {"kind": observation.kind, "units": "e_per_s", "background": "subtract_known",
              "background_adu": background_adu, "shape": list(shape),
              "pixel_scale_arcsec": observation.pixel_scale_arcsec,
              "mask": {"name": mask_name, "digest": array_digest(include), "pixel_count": int(include.sum())},
              "data_digest": array_digest(data), "noise_digest": array_digest(noise),
              "truth_kernels": observation.psfs.to_mapping(), "model_kernel": model_kernel.kernel_identity().to_mapping(),
              "fitted_kernel": KernelIdentity(array_digest(fitted), tuple(fitted.shape),
                                               model_kernel.pixel_scale_arcsec).to_mapping(),
              "objective_version": OBJECTIVE_VERSION, "over_sample_size": sampling,
              "analysis_class": "hwoslaps.inference.backend:CompatAnalysisImaging"}
    return FitData(imaging=imaging, kind=observation.kind, mask=include, record=record)
