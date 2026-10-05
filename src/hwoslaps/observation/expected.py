"""The expected detector response: light convolution and the exposure's linear response.

This module is the single implementation of the forward model from rendered light to the
detector mean and variance, shared by the observation, the forecast and inference. Its
arithmetic follows a fixed operation order that the paper-parity anchors pin to the last
bit; see ``Exposure``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from numbers import Integral
from numbers import Real as _Number
from typing import TYPE_CHECKING, Any, Literal, Mapping

import numpy as np
from numpy.typing import ArrayLike

from ..instrument import Detector

if TYPE_CHECKING:
    from ..optics.kernels import DetectorPSF, KernelBinding
    from ..scene.spec import LightGroup

__all__ = ["Exposure", "Plane", "convolve_light"]

Plane = Literal["lens", "source"]
_PLANE_ORDER: tuple[Plane, ...] = ("lens", "source")
_ROUND_OFF_FRACTION = 1.0e-10


def _finite_number(name: str, value: Any, *, positive: bool) -> float:
    valid = (isinstance(value, _Number) and not isinstance(value, (bool, np.bool_))
             and math.isfinite(value) and (value > 0 if positive else value >= 0))
    if not valid:
        raise ValueError(f"{name} must be a finite number {'> 0' if positive else '>= 0'}, got {value!r}")
    return float(value)


@dataclass(frozen=True)
class Exposure:
    """Linear response of one co-added exposure: detector, total time, count and sky rate.

    ``exposure_time_s`` is the total time of ``exposure_count`` equal exposures summed into
    one image, so the mean does not depend on the count and read noise enters once per
    exposure. ``sky_rate_e_per_s`` is the detected sky rate per pixel. Rates passed to the
    methods are detected e-/s per pixel; images are in ADU.

    The mean adds sky before dark and the Poisson mean adds dark before sky. Both orders are
    the paper code's and are pinned by the parity anchors to the last bit: do not reorder.
    """

    detector: Detector
    exposure_time_s: float
    sky_rate_e_per_s: float
    exposure_count: int = 1

    def __post_init__(self) -> None:
        if not isinstance(self.detector, Detector):
            raise ValueError(f"detector must be a Detector, got {self.detector!r}")
        object.__setattr__(self, "exposure_time_s",
                           _finite_number("exposure_time_s", self.exposure_time_s, positive=True))
        object.__setattr__(self, "sky_rate_e_per_s",
                           _finite_number("sky_rate_e_per_s", self.sky_rate_e_per_s, positive=False))
        count = self.exposure_count
        if not isinstance(count, Integral) or isinstance(count, (bool, np.bool_)) or count < 1:
            raise ValueError(f"exposure_count must be an integer >= 1, got {count!r}")
        object.__setattr__(self, "exposure_count", int(count))
        if not self.blank_variance_e2 > 0.0:
            raise ValueError("a pixel without signal would have zero variance: read noise, dark "
                             "current and sky rate are all zero")

    @property
    def sky_e(self) -> float:
        return self.sky_rate_e_per_s * self.exposure_time_s

    @property
    def dark_e(self) -> float:
        return self.detector.dark_current_e_per_s * self.exposure_time_s

    @property
    def read_variance_e2(self) -> float:
        return self.exposure_count * self.detector.read_noise_e ** 2

    @property
    def read_sigma_e(self) -> float:
        return self.detector.read_noise_e * math.sqrt(self.exposure_count)

    @property
    def background_adu(self) -> float:
        """Mean of a pixel without signal, ``(sky_e + dark_e) / gain``."""
        return (self.sky_e + self.dark_e) / self.detector.gain_e_per_adu

    @property
    def blank_variance_e2(self) -> float:
        """Variance of a pixel without signal, equal to ``variance_e2(0)``."""
        return (self.dark_e + self.sky_e) + self.read_variance_e2

    def signal_e(self, rate_e_per_s: ArrayLike) -> np.ndarray:
        return np.asarray(rate_e_per_s, dtype=float) * self.exposure_time_s

    def signal_adu(self, rate_e_per_s: ArrayLike) -> np.ndarray:
        return self.signal_e(rate_e_per_s) / self.detector.gain_e_per_adu

    def mean_adu(self, rate_e_per_s: ArrayLike) -> np.ndarray:
        """Detector mean ``((rate * t + sky_e) + dark_e) / gain``; negative rates are kept."""
        return ((self.signal_e(rate_e_per_s) + self.sky_e) + self.dark_e) / self.detector.gain_e_per_adu

    def counts_e(self, rate_e_per_s: ArrayLike) -> np.ndarray:
        """Poisson mean ``(maximum(rate, 0) * t + dark_e) + sky_e``."""
        rate = np.maximum(np.asarray(rate_e_per_s, dtype=float), 0.0)
        return (rate * self.exposure_time_s + self.dark_e) + self.sky_e

    def variance_e2(self, rate_e_per_s: ArrayLike) -> np.ndarray:
        return self.counts_e(rate_e_per_s) + self.read_variance_e2

    def noise_map_adu(self, rate_e_per_s: ArrayLike) -> np.ndarray:
        return np.sqrt(self.variance_e2(rate_e_per_s)) / self.detector.gain_e_per_adu

    def rate_from_adu(self, values_adu: ArrayLike) -> np.ndarray:
        return (np.asarray(values_adu, dtype=float) * self.detector.gain_e_per_adu) / self.exposure_time_s

    def to_mapping(self) -> dict[str, Any]:
        return {
            "detector": self.detector.to_mapping(),
            "exposure_time_s": self.exposure_time_s,
            "exposure_count": self.exposure_count,
            "sky_rate_e_per_s": self.sky_rate_e_per_s,
        }


def convolve_light(light_images: Mapping[str, np.ndarray], groups: Mapping[str, LightGroup],
                   kernels: KernelBinding, pixel_scale_arcsec: float) -> dict[Plane, np.ndarray]:
    """Convolved light rate of each plane, e-/s per pixel.

    The light groups are partitioned by (plane, kernel index), in the order of their first
    group in ``groups``. A partition's images are summed in group order (a one-group
    partition uses its image unchanged) and convolved once with the kernel's cached
    convolver; the partitions of a plane are summed in the same order. One plane lit through
    one kernel is therefore a single convolution. Planes without light are absent.

    Round-off negatives of the FFT convolution are kept; a plane with a value below
    ``-1e-10`` times its largest magnitude, or a non-finite value, raises ValueError.
    """
    from ..optics.kernels import PIXEL_SCALE_ATOL_ARCSEC

    keys, group_keys, bound_keys = set(light_images), set(groups), set(kernels.group_index)
    if not keys == group_keys == bound_keys:
        raise ValueError(f"light images {sorted(keys)}, light groups {sorted(group_keys)} and kernel "
                         f"binding groups {sorted(bound_keys)} must name the same groups")
    for index, kernel in enumerate(kernels.kernels):
        if abs(kernel.pixel_scale_arcsec - pixel_scale_arcsec) > PIXEL_SCALE_ATOL_ARCSEC:
            raise ValueError(f"kernel {index} (groups {list(kernels.groups_of(index))}) is sampled at "
                             f"{kernel.pixel_scale_arcsec} arcsec per pixel; the light grid at "
                             f"{pixel_scale_arcsec}")

    partitions: dict[tuple[Plane, int], list[str]] = {}
    for key, group in groups.items():
        partitions.setdefault((group.plane, kernels.group_index[key]), []).append(key)

    by_plane: dict[Plane, list[np.ndarray]] = {}
    for (plane, index), members in partitions.items():
        image = light_images[members[0]]
        for key in members[1:]:
            image = image + light_images[key]
        by_plane.setdefault(plane, []).append(_convolve(image, kernels.kernels[index], pixel_scale_arcsec))

    rates: dict[Plane, np.ndarray] = {}
    for plane in _PLANE_ORDER:
        if plane not in by_plane:
            continue
        rate = by_plane[plane][0]
        for term in by_plane[plane][1:]:
            rate = rate + term
        _check_round_off(rate, plane)
        rates[plane] = rate
    return rates


def _convolve(image: np.ndarray, kernel: DetectorPSF, pixel_scale_arcsec: float) -> np.ndarray:
    import autoarray as aa

    mask = aa.Mask2D.all_false(shape_native=image.shape, pixel_scales=pixel_scale_arcsec)
    convolved = kernel.convolver().convolved_image_from(image=aa.Array2D(values=image, mask=mask),
                                                        blurring_image=None)
    return np.array(convolved.native, dtype=float)


def _check_round_off(rate: np.ndarray, plane: Plane) -> None:
    if not np.all(np.isfinite(rate)):
        raise ValueError(f"the convolved {plane}-plane light is not finite")
    tolerance = _ROUND_OFF_FRACTION * float(np.max(np.abs(rate), initial=0.0))
    lowest = float(np.min(rate, initial=0.0))
    if lowest < -tolerance:
        raise ValueError(f"the convolved {plane}-plane light reaches {lowest!r}, below the FFT round-off "
                         f"bound {-tolerance!r}: the rendered light is negative")
