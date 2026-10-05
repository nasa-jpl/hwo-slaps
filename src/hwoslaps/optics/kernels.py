"""Detector kernels: the kernel value, kernel files, kernel sharing, the autoarray bridge, BLAS determinism.

A detector kernel is the PSF integrated over detector pixels at one angular sampling,
normalized to unit sum on its support. ``DetectorPSF`` holds a private read-only
copy, so a recorded ``KernelIdentity`` stays valid for the life of the object. Every
handoff to autoarray goes through a fresh copy (``make_convolver``,
``convolve_real_space``): ``al.Imaging`` divides its kernel by its sum and rebinds
the array, so a dataset constructor never receives the cached convolver.

Kernels are never resampled, cropped or padded: an even or malformed kernel is an
error.
"""

from __future__ import annotations

import functools
import math
import types
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from numbers import Real
from os import PathLike
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import ArrayLike

from ..identity import KernelIdentity, array_digest, file_digest, json_ready

__all__ = [
    "PIXEL_SCALE_ATOL_ARCSEC", "DetectorPSF", "KernelBinding", "convolve_real_space",
    "deterministic_blas", "make_convolver",
]

PIXEL_SCALE_ATOL_ARCSEC: float = 1.0e-12
"""The one tolerance, in arcsec, for two pixel scales to count as the same sampling."""

_UNIT_SUM_ATOL = 1.0e-10


@contextmanager
def deterministic_blas() -> Iterator[None]:
    """Hold every BLAS and OpenMP pool at one thread for the duration of the block.

    Kernel bytes depend on the thread count of the matrix products inside the
    propagation (16 threads move the paper 999 x 999 kernel by 5e-18). Every other optics
    computation that feeds kernel bytes (basis builds, QR factorizations, draw arithmetic)
    runs inside this block as well, although no byte has yet been seen to depend on their
    thread count. The caller's limits are restored on exit.

    threadpoolctl limits only libraries that are already loaded, and HCIPy's matrix Fourier
    transform calls SciPy's own BLAS (``scipy.linalg.blas.zgemm``), so SciPy's BLAS is loaded
    first.
    """
    import scipy.linalg  # noqa: F401
    from threadpoolctl import threadpool_limits

    with threadpool_limits(limits=1):
        yield


def _pixel_scale(value: Any, what: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError(f"{what} must be a real number of arcsec, got {value!r}")
    scale = float(value)
    if not (math.isfinite(scale) and scale > 0.0):
        raise ValueError(f"{what} must be finite and positive, got {value!r}")
    return scale


def _kernel_array(values: ArrayLike, *, signed: bool) -> np.ndarray:
    """A float64 C-contiguous copy of a 2-D kernel with odd sides and finite values."""
    array = np.array(values, dtype=np.float64, copy=True, order="C")
    if array.ndim != 2:
        raise ValueError(f"a detector kernel must be two-dimensional, got shape {array.shape}")
    if array.shape[0] % 2 == 0 or array.shape[1] % 2 == 0:
        raise ValueError(f"a detector kernel needs odd sides (a centre pixel), got shape {array.shape}")
    if not np.all(np.isfinite(array)):
        raise ValueError("detector kernel values must be finite")
    if not signed and np.any(array < 0.0):
        raise ValueError("detector kernel values must be non-negative")
    return array


@dataclass(frozen=True, eq=False)
class DetectorPSF:
    """A pixel-integrated detector kernel at one angular sampling.

    ``kernel`` is a private float64 C-contiguous read-only copy: odd sides, finite,
    non-negative, unit sum within 1e-10. ``source`` is a JSON-safe record
    of where the kernel came from, with a text ``kind`` (``array``, ``file``,
    ``optical``, ``cube`` or ``effective``) and ``captured_power_fraction``: the share
    of the propagated light inside the support for an optical kernel, ``None`` when
    the normalized array no longer holds it. Two kernels are the same kernel when
    their ``kernel_identity()`` values are equal; ``==`` compares objects.
    """

    kernel: np.ndarray
    pixel_scale_arcsec: float
    source: Mapping[str, Any]

    def __post_init__(self) -> None:
        array = _kernel_array(self.kernel, signed=False)
        total = float(np.sum(array))
        if not abs(total - 1.0) <= _UNIT_SUM_ATOL:
            raise ValueError(f"a detector kernel must sum to one within {_UNIT_SUM_ATOL}, got {total!r}")
        array.flags.writeable = False
        if not isinstance(self.source, Mapping) or not isinstance(self.source.get("kind"), str):
            raise ValueError(f"source must be a mapping with a text 'kind', got {self.source!r}")
        if "captured_power_fraction" not in self.source:
            raise ValueError("source must record 'captured_power_fraction' (None when the kernel does not hold it), "
                             f"got {dict(self.source)!r}")
        fraction = self.source["captured_power_fraction"]
        if fraction is not None and (isinstance(fraction, (bool, np.bool_)) or not isinstance(fraction, Real)
                                     or not 0.0 < float(fraction) <= 1.0):
            raise ValueError(f"source captured_power_fraction must be None or a number in (0, 1], got {fraction!r}")
        object.__setattr__(self, "kernel", array)
        object.__setattr__(self, "pixel_scale_arcsec", _pixel_scale(self.pixel_scale_arcsec, "pixel_scale_arcsec"))
        object.__setattr__(self, "source", types.MappingProxyType(json_ready(dict(self.source))))

    @classmethod
    def from_array(cls, values: ArrayLike, pixel_scale_arcsec: float, *, normalize: bool,
                   source: Mapping[str, Any] | None = None) -> DetectorPSF:
        """A kernel from caller values; ``normalize=True`` divides once by the sum.

        Without normalization the values must already sum to one. The default source
        record is ``{"kind": "array", "captured_power_fraction": None}``.
        """
        if not isinstance(normalize, bool):
            raise ValueError(f"normalize must be true or false, got {normalize!r}")
        array = _kernel_array(values, signed=False)
        if normalize:
            total = float(np.sum(array))
            if not (math.isfinite(total) and total > 0.0):
                raise ValueError(f"a detector kernel needs positive finite flux to normalize, got sum {total!r}")
            array = array / total
        record = {"kind": "array", "captured_power_fraction": None} if source is None else source
        return cls(array, pixel_scale_arcsec, record)

    @classmethod
    def from_file(cls, path: str | PathLike[str], *, pixel_scale_arcsec: float, array_key: str | None,
                  normalize: bool, file_sha256: str | None) -> DetectorPSF:
        """A kernel read from ``.npy`` (``array_key`` None) or ``.npz`` (the named member).

        When ``file_sha256`` is given the file bytes must have that SHA-256 before the
        file is parsed. Pickled object arrays are refused.
        """
        location = Path(path)
        digest = file_digest(location)
        if file_sha256 is not None and digest != file_sha256:
            raise ValueError(f"{location}: file SHA-256 is {digest}, the configuration states {file_sha256}")
        if location.suffix == ".npy":
            if array_key is not None:
                raise ValueError(f"{location}: a .npy kernel file has no members; array_key must be None")
            values = np.load(location, allow_pickle=False)
        elif location.suffix == ".npz":
            if not isinstance(array_key, str) or not array_key:
                raise ValueError(f"{location}: a .npz kernel file needs the member name in array_key")
            with np.load(location, allow_pickle=False) as members:
                if array_key not in members.files:
                    raise ValueError(f"{location}: no member {array_key!r}; members are {sorted(members.files)}")
                values = members[array_key]
        else:
            raise ValueError(f"{location}: kernel files are .npy or .npz")
        record = {"kind": "file", "path": str(location), "array_key": array_key, "file_sha256": digest,
                  "normalized": normalize, "captured_power_fraction": None}
        return cls.from_array(values, pixel_scale_arcsec, normalize=normalize, source=record)

    @property
    def shape(self) -> tuple[int, int]:
        return (int(self.kernel.shape[0]), int(self.kernel.shape[1]))

    def kernel_identity(self) -> KernelIdentity:
        return KernelIdentity(array_digest(self.kernel), self.shape, self.pixel_scale_arcsec)

    def convolver(self) -> Any:
        """The autoarray Convolver of this kernel, built once over a private copy.

        For convolutions only: never hand it to a dataset constructor, which would
        normalize it in place (use ``make_convolver`` for a dataset).
        """
        return self._convolver

    @functools.cached_property
    def _convolver(self) -> Any:
        return make_convolver(self.kernel, self.pixel_scale_arcsec)


def make_convolver(values: ArrayLike, pixel_scale_arcsec: float) -> Any:
    """An autoarray Convolver over a private copy of ``values`` (no normalization)."""
    import autoarray as aa

    array = _kernel_array(values, signed=True)
    scale = _pixel_scale(pixel_scale_arcsec, "pixel_scale_arcsec")
    return aa.Convolver(kernel=aa.Array2D.no_mask(values=array, pixel_scales=scale))


def convolve_real_space(image: ArrayLike, kernel_values: ArrayLike, pixel_scale_arcsec: float) -> np.ndarray:
    """``image`` convolved with ``kernel_values`` in real space, zero outside the image.

    The kernel may be signed (a wavefront-derivative kernel) and is never normalized;
    its sides must be odd. Returns a new float64 array of the image's shape.
    """
    import autoarray as aa

    scale = _pixel_scale(pixel_scale_arcsec, "pixel_scale_arcsec")
    pixels = np.array(image, dtype=np.float64, copy=True)
    if pixels.ndim != 2 or not np.all(np.isfinite(pixels)):
        raise ValueError(f"the image must be a finite two-dimensional array, got shape {pixels.shape}")
    mask = aa.Mask2D.all_false(shape_native=pixels.shape, pixel_scales=scale)
    convolved = make_convolver(kernel_values, scale).convolved_image_via_real_space_from(
        image=aa.Array2D(values=pixels, mask=mask), blurring_image=None)
    return np.array(convolved.native, dtype=np.float64)


@dataclass(frozen=True)
class KernelBinding:
    """Which light groups share which kernel.

    ``kernels`` holds the distinct kernels, compared by ``KernelIdentity.sha256``, in
    order of first appearance; ``group_index`` maps each light group key to its
    kernel. Every kernel serves at least one group, and all kernels share one pixel
    scale (a binding serves one scene grid).
    """

    kernels: tuple[DetectorPSF, ...]
    group_index: Mapping[str, int]

    def __post_init__(self) -> None:
        kernels = tuple(self.kernels)
        if not kernels or not all(isinstance(kernel, DetectorPSF) for kernel in kernels):
            raise ValueError("a kernel binding needs at least one DetectorPSF")
        digests = [kernel.kernel_identity().sha256 for kernel in kernels]
        if len(set(digests)) != len(digests):
            raise ValueError("kernels of a binding must be distinct; equal kernels share one entry")
        scale = kernels[0].pixel_scale_arcsec
        if any(abs(kernel.pixel_scale_arcsec - scale) > PIXEL_SCALE_ATOL_ARCSEC for kernel in kernels):
            raise ValueError("kernels of a binding must share one pixel scale")
        index = dict(self.group_index)
        if not index or not all(isinstance(group, str) and group for group in index):
            raise ValueError("a kernel binding maps at least one light group key to a kernel")
        if sorted(set(index.values())) != list(range(len(kernels))):
            raise ValueError(f"group indices must use every kernel index 0..{len(kernels) - 1}, got {index}")
        object.__setattr__(self, "kernels", kernels)
        object.__setattr__(self, "group_index", types.MappingProxyType(index))

    @classmethod
    def from_groups(cls, kernels: Mapping[str, DetectorPSF]) -> KernelBinding:
        """Bind each group to its kernel; groups whose kernels are bitwise equal share one entry."""
        distinct: list[DetectorPSF] = []
        position: dict[str, int] = {}
        index: dict[str, int] = {}
        for group, kernel in kernels.items():
            digest = kernel.kernel_identity().sha256
            if digest not in position:
                position[digest] = len(distinct)
                distinct.append(kernel)
            index[group] = position[digest]
        return cls(tuple(distinct), index)

    @classmethod
    def uniform(cls, kernel: DetectorPSF, groups: Sequence[str]) -> KernelBinding:
        """One kernel for every group."""
        return cls((kernel,), {group: 0 for group in groups})

    def for_group(self, group: str) -> DetectorPSF:
        if group not in self.group_index:
            raise KeyError(f"no kernel bound to light group {group!r}; groups are {list(self.group_index)}")
        return self.kernels[self.group_index[group]]

    def groups_of(self, index: int) -> tuple[str, ...]:
        if not 0 <= index < len(self.kernels):
            raise IndexError(f"kernel index {index} outside 0..{len(self.kernels) - 1}")
        return tuple(group for group, position in self.group_index.items() if position == index)

    @property
    def single(self) -> DetectorPSF:
        """The kernel of every group; raises unless exactly one distinct kernel is bound."""
        if len(self.kernels) != 1:
            raise ValueError(f"the light groups use {len(self.kernels)} distinct kernels, not one")
        return self.kernels[0]

    def to_mapping(self) -> dict[str, Any]:
        return {
            "kernels": [{"identity": kernel.kernel_identity().to_mapping(), "source": dict(kernel.source)}
                        for kernel in self.kernels],
            "groups": dict(self.group_index),
        }
