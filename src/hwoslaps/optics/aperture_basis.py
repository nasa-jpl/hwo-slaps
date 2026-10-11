"""The coefficient convention of the mode-weight priors: orthonormal modes on the illuminated pupil.

Prior tables weight modes of sequentially QR-orthonormalized aperture bases: the raw
global Zernikes on the illuminated pupil, and on each active segment the raw hexikes on
the segment's illuminated pixels, orthonormalized in Noll order with unit mean square.
``ApertureBasisTransform`` converts coefficients in those bases to raw HCIPy coefficients
(nm of OPD) of the same pupil.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from numbers import Integral

import numpy as np

from .kernels import deterministic_blas
from .wavefront import WavefrontBasis, WavefrontCoefficients, WavefrontMode

__all__ = ["ApertureBasisTransform", "positive_diagonal_qr"]


def positive_diagonal_qr(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Reduced QR of ``values`` (pixels x modes) with the signs fixed so that ``diag(R) > 0``.

    Raises when the columns are rank deficient: ``|diag R| <= eps max(shape) ||R||_inf``.
    """
    array = np.asarray(values, dtype=float)
    if array.ndim != 2 or array.shape[1] == 0:
        raise ValueError(f"mode values must have shape (n_pixels, n_modes), got {array.shape}")
    if array.shape[0] < array.shape[1]:
        raise ValueError(f"{array.shape[0]} pixels cannot carry {array.shape[1]} independent modes")
    if not np.all(np.isfinite(array)):
        raise ValueError("mode values must be finite")
    q_matrix, r_matrix = np.linalg.qr(array, mode="reduced")
    tolerance = np.finfo(float).eps * max(array.shape) * np.linalg.norm(r_matrix, ord=np.inf)
    if np.any(np.abs(np.diag(r_matrix)) <= tolerance):
        raise ValueError("the modes are rank deficient on these pixels")
    signs = np.where(np.diag(r_matrix) < 0.0, -1.0, 1.0)
    return q_matrix * signs[np.newaxis, :], r_matrix * signs[:, np.newaxis]


def _nolls(values: Sequence[int], name: str, minimum: int) -> tuple[int, ...]:
    nolls = tuple(values)
    if not all(isinstance(n, Integral) and not isinstance(n, bool) and n >= minimum for n in nolls):
        raise ValueError(f"{name} must be integer Noll indices >= {minimum}, got {nolls}")
    nolls = tuple(int(n) for n in nolls)
    if list(nolls) != sorted(set(nolls)):
        raise ValueError(f"{name} must be strictly increasing, got {nolls}")
    return nolls


def _raw(coefficients: Mapping[int, float], nolls: tuple[int, ...], factor: np.ndarray, pixel_count: int,
         what: str) -> list[tuple[int, float]]:
    if set(coefficients) != set(nolls):
        raise ValueError(f"{what} must have exactly the transform modes {list(nolls)}, got {sorted(coefficients)}")
    values = np.array([coefficients[noll] for noll in nolls], dtype=float)
    if not np.all(np.isfinite(values)):
        raise ValueError(f"{what} must be finite")
    raw = np.sqrt(pixel_count) * np.linalg.solve(factor, values)
    return [(noll, float(raw[index])) for index, noll in enumerate(nolls)]


class ApertureBasisTransform:
    """Orthonormal aperture coefficients to raw coefficients on one pupil.

    The global side (Noll >= 4) is defined on ``pupil.illuminated_mask``; the segment side
    (Noll >= 1) on each active segment's illuminated pixels. Only the sides with modes are
    built. A segment with fewer illuminated pixels than segment modes raises, before any
    hexike surface is built.
    """

    def __init__(self, basis: WavefrontBasis, *, global_nolls: Sequence[int] = (),
                 segment_nolls: Sequence[int] = ()) -> None:
        self._global_nolls = _nolls(global_nolls, "global_nolls", 4)
        self._segment_nolls = _nolls(segment_nolls, "segment_nolls", 1)
        if not self._global_nolls and not self._segment_nolls:
            raise ValueError("an aperture basis transform needs global or segment modes")
        pupil = basis.pupil
        mask = pupil.illuminated_mask
        segment_masks: dict[int, np.ndarray] = {}
        if self._segment_nolls:
            if not pupil.active_segments:
                raise ValueError(f"a {pupil.spec.kind} pupil has no active segment for segment modes")
            for segment in pupil.active_segments:
                segment_mask = mask & (np.asarray(pupil.segments[segment]) > 0.5)
                pixels = int(np.count_nonzero(segment_mask))
                if pixels < len(self._segment_nolls):
                    raise ValueError(f"segment {segment} has {pixels} illuminated pixels, fewer than its "
                                     f"{len(self._segment_nolls)} segment modes")
                segment_masks[segment] = segment_mask
        with deterministic_blas():
            self._global = None
            if self._global_nolls:
                values = basis.zernike_samples(self._global_nolls, mask)
                self._global = (positive_diagonal_qr(values)[1], values.shape[0])
            self._segments = {}
            for segment, segment_mask in segment_masks.items():
                values = basis.hexike_samples(segment, self._segment_nolls, segment_mask)
                self._segments[segment] = (positive_diagonal_qr(values)[1], values.shape[0])

    @property
    def global_nolls(self) -> tuple[int, ...]:
        return self._global_nolls

    @property
    def segment_nolls(self) -> tuple[int, ...]:
        return self._segment_nolls

    def to_raw(self, *, segment: Mapping[int, Mapping[int, float]] | None = None,
               global_: Mapping[int, float] | None = None) -> WavefrontCoefficients:
        """Raw coefficients (nm of OPD) of orthonormal ones: ``raw = sqrt(n_pixels) R^-1 c`` per side.

        Each given side has exactly the transform's modes; segment keys are active segments.
        """
        entries: list[tuple[WavefrontMode, float]] = []
        with deterministic_blas():
            for segment_index, modes in (segment or {}).items():
                if segment_index not in self._segments:
                    raise ValueError(f"segment {segment_index} has no orthonormal basis; the transform covers "
                                     f"segments {sorted(self._segments)}")
                factor, pixels = self._segments[segment_index]
                entries += [(WavefrontMode("segment_hexikes", noll, segment_index), value) for noll, value in
                            _raw(modes, self._segment_nolls, factor, pixels, f"segment {segment_index} coefficients")]
            if global_:
                if self._global is None:
                    raise ValueError("this transform has no global modes")
                factor, pixels = self._global
                entries += [(WavefrontMode("zernikes", noll), value) for noll, value in
                            _raw(global_, self._global_nolls, factor, pixels, "global coefficients")]
        return WavefrontCoefficients(tuple(entries))
