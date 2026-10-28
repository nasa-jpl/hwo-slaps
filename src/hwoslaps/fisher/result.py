"""Forecast results by subhalo mass and position.

A :class:`ForecastResult` stores the bank finisher's primaries with axes
``(mass, position)``: the raw and profiled information on the subhalo
amplitude and, for a mismatched model PSF, the fitted amplitudes of the data
residual and of the truth-minus-model bias. The other statistics are
properties computed with the finisher's own functions
(:mod:`hwoslaps.fisher.statistics`), so they carry the bank's bits.

A detection is ``q >= T`` with a finite ``q``; for ``q_mismatch`` and
``q_spurious`` the matching fitted amplitude must also be finite and positive,
because ``q = a**2 F`` is large for a fit of either sign. The threshold ``T`` is
always the caller's. :func:`is_detection` is the one implementation of this
rule; :meth:`ForecastResult.detections` and every other reader of forecast
detections call it.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from numbers import Real
from typing import Any, Literal

import numpy as np
from numpy.typing import ArrayLike

from ..identity import json_ready
from . import statistics as bank
from .positions import PositionSet

__all__ = ["ForecastResult", "Metric", "is_detection"]

Metric = Literal["q_asimov", "q_mismatch", "q_spurious"]
"""A detection statistic of a result."""


def _read_only(values: np.ndarray) -> np.ndarray:
    copy = np.array(values, order="C")
    copy.setflags(write=False)
    return copy


def is_detection(q: ArrayLike, amplitude: ArrayLike | None, *, q_threshold: float) -> np.ndarray:
    """Detections, elementwise: finite ``q >= q_threshold`` and, given a fit, a finite positive amplitude.

    ``amplitude`` is the fitted amplitude behind ``q_mismatch`` or
    ``q_spurious`` (same shape as ``q``) and None for ``q_asimov``. Scalars give
    a numpy boolean. The threshold must be a positive finite number, not boolean.
    """
    if isinstance(q_threshold, (bool, np.bool_)) or not isinstance(q_threshold, Real) \
            or not np.isfinite(q_threshold) or q_threshold <= 0:
        raise ValueError(f"q_threshold must be a positive finite number, not boolean, got {q_threshold!r}")
    values = np.asarray(q, dtype=float)
    detected = np.isfinite(values) & (values >= float(q_threshold))
    if amplitude is None:
        return detected
    fitted = np.asarray(amplitude, dtype=float)
    if fitted.shape != values.shape:
        raise ValueError(f"amplitude shape {fitted.shape} differs from the statistic shape {values.shape}")
    return detected & np.isfinite(fitted) & (fitted > 0.0)


@dataclass(frozen=True, eq=False)
class ForecastResult:
    """Profiled linear-Gaussian forecast statistics by mass and position; arrays are read-only.

    ``amplitude_hat`` and ``amplitude_spurious`` exist exactly when the model PSF
    differs from the truth (``psf_relation`` other than ``"matched"``).
    ``config`` and ``provenance`` are kept in their JSON form.
    ``template_status`` records valid, zero-information and numerical-failure
    rows; explicit failures can retain provisional finite statistics.
    """

    masses_msun: np.ndarray
    positions: PositionSet
    fisher_raw: np.ndarray
    fisher_profiled: np.ndarray
    amplitude_hat: np.ndarray | None
    amplitude_spurious: np.ndarray | None
    psf_relation: str
    config: Mapping[str, Any]
    provenance: Mapping[str, Any]
    template_status: np.ndarray | None = None

    def __post_init__(self) -> None:
        masses = np.asarray(self.masses_msun, dtype=float)
        if masses.ndim != 1 or masses.size == 0 or not np.all(np.isfinite(masses)) or np.any(masses <= 0.0):
            raise ValueError("masses_msun must be a non-empty vector of positive finite masses")
        if not isinstance(self.positions, PositionSet):
            raise ValueError("positions must be a PositionSet")
        shape = (masses.size, len(self.positions))
        for name in ("fisher_raw", "fisher_profiled"):
            values = np.asarray(getattr(self, name), dtype=float)
            if values.shape != shape:
                raise ValueError(f"{name} must have shape {shape} (masses, positions), got {values.shape}")
            if np.any(values < 0.0):
                raise ValueError(f"{name} must be non-negative; non-finite values require failed_numeric status")
            object.__setattr__(self, name, _read_only(values))
        if not isinstance(self.psf_relation, str) or not self.psf_relation:
            raise ValueError(f"psf_relation must be a non-empty string, got {self.psf_relation!r}")
        amplitudes = (self.amplitude_hat, self.amplitude_spurious)
        mismatched = self.psf_relation != "matched"
        if any(value is None for value in amplitudes) and not all(value is None for value in amplitudes):
            raise ValueError("amplitude_hat and amplitude_spurious are given together or not at all")
        if (amplitudes[0] is not None) != mismatched:
            raise ValueError(f"a {self.psf_relation!r} result {'needs' if mismatched else 'has no'} "
                             "fitted amplitudes")
        for name in ("amplitude_hat", "amplitude_spurious"):
            if getattr(self, name) is not None:
                values = np.asarray(getattr(self, name), dtype=float)
                if values.shape != shape:
                    raise ValueError(f"{name} must have shape {shape} (masses, positions), got {values.shape}")
                object.__setattr__(self, name, _read_only(values))
        failed = ~np.isfinite(self.fisher_raw) | ~np.isfinite(self.fisher_profiled)
        zero = self.fisher_profiled == 0.0
        for name in ("amplitude_hat", "amplitude_spurious"):
            values = getattr(self, name)
            if values is not None:
                with np.errstate(over="ignore", invalid="ignore"):
                    fitted_q = (values * np.sqrt(self.fisher_profiled))**2
                failed |= (~zero & (~np.isfinite(values) | ~np.isfinite(fitted_q))) | (zero & ~np.isnan(values))
        inferred = np.where(failed, "failed_numeric", np.where(zero, "zero_information", "valid"))
        status = inferred if self.template_status is None else np.asarray(self.template_status)
        if status.shape != shape or status.dtype.kind != "U" or not np.all(np.isin(
                status, ("valid", "zero_information", "failed_numeric"))):
            raise ValueError("template_status must have the statistic shape and valid, zero_information or failed_numeric values")
        # Explicit failure may retain provisional finite values; a valid/zero
        # record can never disguise failed or inconsistently identified values.
        if np.any((status != "failed_numeric") & (status != inferred)):
            raise ValueError("template_status is inconsistent with information or fitted amplitudes")
        object.__setattr__(self, "template_status", _read_only(status))
        for name in ("config", "provenance"):
            mapping = getattr(self, name)
            if not isinstance(mapping, Mapping):
                raise ValueError(f"{name} must be a mapping")
            object.__setattr__(self, name, json_ready(mapping))
        object.__setattr__(self, "masses_msun", _read_only(masses))

    def __setstate__(self, state: dict[str, object]) -> None:
        """Unpickle through the constructor checks; numpy restores arrays writeable."""
        self.__dict__.update(state)
        self.__post_init__()

    @property
    def positions_yx(self) -> np.ndarray:
        """``(n_positions, 2)`` evaluated positions."""
        return self.positions.positions_yx

    @property
    def cell_areas_arcsec2(self) -> np.ndarray | None:
        """Cell area of each position (grid layouts)."""
        return self.positions.cell_areas_arcsec2

    @property
    def boundary(self) -> np.ndarray | None:
        """Boundary flag of each position (grid layouts holding every boundary node)."""
        return self.positions.boundary

    @property
    def q_asimov(self) -> np.ndarray:
        """Asimov statistic of the unit-amplitude template, equal to the profiled information."""
        return self.fisher_profiled

    @property
    def z_asimov(self) -> np.ndarray:
        """Local Asimov significance ``sqrt(q_asimov)``."""
        return bank.z_asimov(self.fisher_profiled)

    @property
    def sigma_amplitude(self) -> np.ndarray:
        """Amplitude uncertainty ``1 / sqrt(F)``, infinite where ``F = 0``."""
        return bank.sigma_amplitude(self.fisher_profiled)

    @property
    def degradation(self) -> np.ndarray:
        """Fraction of the raw information kept by profiling."""
        return bank.degradation(self.fisher_raw, self.fisher_profiled)

    @property
    def z_mismatch(self) -> np.ndarray | None:
        """Signed significance of the fitted data amplitude, ``a_hat sqrt(F)``."""
        if self.amplitude_hat is None:
            return None
        z, _ = bank.amplitude_significance(self.amplitude_hat, self.fisher_profiled, signed=True)
        return z

    @property
    def q_mismatch(self) -> np.ndarray | None:
        """``z_mismatch**2``."""
        if self.amplitude_hat is None:
            return None
        _, q = bank.amplitude_significance(self.amplitude_hat, self.fisher_profiled, signed=True)
        return q

    @property
    def z_spurious(self) -> np.ndarray | None:
        """Significance of the fitted bias amplitude, ``|a_spurious| sqrt(F)``."""
        if self.amplitude_spurious is None:
            return None
        z, _ = bank.amplitude_significance(self.amplitude_spurious, self.fisher_profiled, signed=False)
        return z

    @property
    def q_spurious(self) -> np.ndarray | None:
        """``z_spurious**2``."""
        if self.amplitude_spurious is None:
            return None
        _, q = bank.amplitude_significance(self.amplitude_spurious, self.fisher_profiled, signed=False)
        return q

    @property
    def detection_metric(self) -> Literal["q_asimov", "q_mismatch"]:
        """The statistic of the data and model pair: ``q_mismatch`` with a mismatched model PSF."""
        return "q_asimov" if self.amplitude_hat is None else "q_mismatch"

    def detections(self, *, q_threshold: float, metric: Metric | None = None,
                   invalid_policy: Literal["raise", "exclude"] = "raise") -> np.ndarray:
        """Boolean ``(masses, positions)`` detections at the caller's threshold, by :func:`is_detection`.

        ``metric`` defaults to :attr:`detection_metric`; ``q_asimov`` stays
        available as the matched diagnostic and ``q_spurious`` as the null control.
        Failed numerical templates raise by default. Explicit exclusion leaves
        their mask in :attr:`template_status`; zero information remains distinct.
        """
        if invalid_policy not in ("raise", "exclude"):
            raise ValueError("invalid_policy must be 'raise' or 'exclude'")
        failed = self.template_status == "failed_numeric"
        if invalid_policy == "raise" and np.any(failed):
            raise ValueError(f"forecast contains {np.count_nonzero(failed)} failed_numeric templates; "
                             "use an explicit exclusion policy and report the excluded cohort")
        chosen = self.detection_metric if metric is None else metric
        if chosen not in ("q_asimov", "q_mismatch", "q_spurious"):
            raise ValueError(f"metric must be q_asimov, q_mismatch or q_spurious, got {chosen!r}")
        values = getattr(self, chosen)
        if values is None:
            raise ValueError(f"{chosen} was not evaluated: the model PSF is matched to the truth")
        amplitude = None
        if chosen != "q_asimov":
            amplitude = self.amplitude_hat if chosen == "q_mismatch" else self.amplitude_spurious
        return is_detection(values, amplitude, q_threshold=q_threshold) & ~failed
