"""Profiled linear-Gaussian algebra for banks of subhalo signal templates.

Near the smooth model the whitened mean image is ``mu0 + A s + J eta`` for a
subhalo amplitude ``A``, signal template ``s`` and nuisance design ``J`` (one
column per nuisance parameter). Profiling ``eta`` under Gaussian priors of
precision ``P`` leaves the information on ``A``

    F = s.s - (s.J) N+ (J.s),    N = 0.5 (J^T J + (J^T J)^T) + diag(P),

where the pseudo-inverse ``N+`` keeps the eigenvalues of ``N`` above
``max(rcond, p eps) max|lambda|``. The cutoff is relative to the largest
eigenvalue, so one common rescaling of every column (priors scaled alike)
leaves ``F`` unchanged, but it depends on the units of individual columns:
with flat priors, ``J = diag(1, 1e-7)`` and ``s = (0, 1)`` give ``F = 1`` at
rank 1, while the same span with ``J = I`` gives ``F = 0``. ``nuisance_rank``
and ``condition_number`` show when the cutoff acts.

``q_asimov = F`` is the Asimov statistic of a unit-amplitude template. For data
residuals ``d`` made with another PSF and a fixed truth-minus-model bias ``b``,
the profiled fitted amplitudes are

    a_hat = (s.d - (s.J) N+ (J.d)) / F,    a_spurious = (s.b - (s.J) N+ (J.b)) / F,

with ``z_mismatch = a_hat sqrt(F)``, ``z_spurious = |a_spurious| sqrt(F)`` and
``q = z**2``; amplitudes and their statistics are NaN where ``F = 0``.

The derived statistics are module functions shared by the bank finisher and
:class:`hwoslaps.fisher.result.ForecastResult`, so a forecast's properties are
the bank's bits. Host algebra runs with one BLAS thread, so results do not
depend on the caller's thread count.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import MISSING, dataclass, fields
from numbers import Real
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike
from threadpoolctl import threadpool_limits

__all__ = [
    "BankReductions", "ProfileLikelihoodWorkspace", "SignalBankResult", "Whitener",
    "amplitude_significance", "degradation", "sigma_amplitude", "z_asimov",
]


def sigma_amplitude(fisher_profiled: np.ndarray) -> np.ndarray:
    """Amplitude uncertainty ``1 / sqrt(F)``, infinite where ``F <= 0``."""
    sigma = np.full_like(fisher_profiled, np.inf, dtype=float)
    positive = fisher_profiled > 0.0
    sigma[positive] = 1.0 / np.sqrt(fisher_profiled[positive])
    return sigma


def z_asimov(q_asimov: np.ndarray) -> np.ndarray:
    """Local Asimov significance ``sqrt(max(q, 0))``."""
    return np.sqrt(np.maximum(q_asimov, 0.0))


def degradation(fisher_raw: np.ndarray, fisher_profiled: np.ndarray) -> np.ndarray:
    """Fraction of the raw information kept by profiling, in [0, 1]; 0 where the raw information is 0."""
    retained = np.divide(fisher_profiled, fisher_raw, out=np.zeros_like(fisher_profiled), where=fisher_raw > 0.0)
    return np.clip(retained, 0.0, 1.0)


def amplitude_significance(amplitude: np.ndarray, fisher_profiled: np.ndarray, *,
                           signed: bool) -> tuple[np.ndarray, np.ndarray]:
    """``(z, q = z**2)`` of a fitted amplitude: ``z = a sqrt(F)`` (``|a| sqrt(F)`` unsigned), NaN where ``F <= 0``."""
    positive = fisher_profiled > 0.0
    z = np.full_like(fisher_profiled, np.nan, dtype=float)
    scale = amplitude[positive] if signed else np.abs(amplitude[positive])
    z[positive] = scale * np.sqrt(fisher_profiled[positive])
    return z, z**2


def _array(values: ArrayLike, what: str, *, ndim: int) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim != ndim:
        raise ValueError(f"{what} must be a {ndim}-D array, got shape {array.shape}")
    return array


def _read_only(values: np.ndarray) -> np.ndarray:
    copy = np.array(values, order="C")
    copy.setflags(write=False)
    return copy


def _require_finite(values: np.ndarray, what: str) -> None:
    if not np.all(np.isfinite(values)):
        raise ValueError(f"{what} contains non-finite values")


@dataclass(frozen=True, eq=False)
class Whitener:
    """Apply ``C^(-1/2)`` along the first (data) axis.

    ``diagonal`` divides by the per-pixel standard deviation ``sigma``;
    ``dense`` solves with the lower Cholesky factor of the covariance.
    """

    mode: Literal["diagonal", "dense"]
    sigma: np.ndarray | None
    cholesky_factor: np.ndarray | None

    def __post_init__(self) -> None:
        if self.mode == "diagonal":
            if self.sigma is None or self.cholesky_factor is not None:
                raise ValueError("a diagonal whitener holds sigma and no Cholesky factor")
            sigma = _array(self.sigma, "sigma", ndim=1)
            if sigma.size == 0:
                raise ValueError("sigma must hold at least one value")
            _require_finite(sigma, "sigma")
            if np.any(sigma <= 0.0):
                raise ValueError("sigma must be strictly positive")
            object.__setattr__(self, "sigma", _read_only(sigma))
        elif self.mode == "dense":
            if self.cholesky_factor is None or self.sigma is not None:
                raise ValueError("a dense whitener holds a Cholesky factor and no sigma")
            factor = _array(self.cholesky_factor, "cholesky_factor", ndim=2)
            if factor.shape[0] != factor.shape[1] or factor.size == 0:
                raise ValueError(f"cholesky_factor must be a non-empty square matrix, got shape {factor.shape}")
            _require_finite(factor, "cholesky_factor")
            object.__setattr__(self, "cholesky_factor", _read_only(factor))
        else:
            raise ValueError(f"whitener mode must be 'diagonal' or 'dense', got {self.mode!r}")

    @classmethod
    def from_sigma(cls, sigma: ArrayLike) -> Whitener:
        """Diagonal whitener from 1-D, finite, strictly positive standard deviations."""
        return cls("diagonal", np.asarray(sigma, dtype=float), None)

    @classmethod
    def from_covariance(cls, covariance: ArrayLike) -> Whitener:
        """Dense whitener from a symmetric positive-definite covariance (symmetrized first)."""
        matrix = _array(covariance, "covariance", ndim=2)
        if matrix.shape[0] != matrix.shape[1] or matrix.size == 0:
            raise ValueError(f"covariance must be a non-empty square matrix, got shape {matrix.shape}")
        _require_finite(matrix, "covariance")
        matrix = 0.5 * (matrix + matrix.T)
        try:
            factor = np.linalg.cholesky(matrix)
        except np.linalg.LinAlgError as error:
            raise ValueError("covariance must be symmetric positive definite") from error
        return cls("dense", None, factor)

    @property
    def size(self) -> int:
        """Length of the data axis."""
        return int(self.sigma.size if self.mode == "diagonal" else self.cholesky_factor.shape[0])

    def apply(self, values: ArrayLike) -> np.ndarray:
        """Whiten a vector ``(n,)`` or the columns of a matrix ``(n, k)``."""
        array = np.asarray(values, dtype=float)
        if array.ndim not in (1, 2) or array.shape[0] != self.size:
            raise ValueError(f"values must have shape ({self.size},) or ({self.size}, k), got {array.shape}")
        if self.mode == "diagonal":
            return array / self.sigma if array.ndim == 1 else array / self.sigma[:, None]
        return np.linalg.solve(self.cholesky_factor, array)


@dataclass(frozen=True, eq=False)
class BankReductions:
    """Reductions over the whitened data axis, one row per signal.

    ``raw = s.s``, ``cross = s @ J``; with data residuals ``signal_data_inner =
    s.d`` and ``data_cross = d @ J``; with a bias ``signal_bias_inner = s.b``.
    ``finite`` and ``data_finite`` carry the all-finite verdict of each whitened
    vector, taken where the vectors existed (an accelerator never sends them to
    the host).
    """

    raw: np.ndarray
    cross: np.ndarray
    finite: np.ndarray
    signal_data_inner: np.ndarray | None = None
    data_cross: np.ndarray | None = None
    data_finite: np.ndarray | None = None
    signal_bias_inner: np.ndarray | None = None

    def __post_init__(self) -> None:
        raw = _array(self.raw, "raw", ndim=1)
        size = raw.shape[0]
        cross = _array(self.cross, "cross", ndim=2)
        if cross.shape[0] != size:
            raise ValueError(f"cross must have {size} rows, got shape {cross.shape}")
        values = {"raw": raw, "cross": cross, "finite": self._verdict(self.finite, "finite", size)}
        data = (self.signal_data_inner, self.data_cross, self.data_finite)
        if any(value is None for value in data) and not all(value is None for value in data):
            raise ValueError("signal_data_inner, data_cross and data_finite are given together or not at all")
        if self.signal_data_inner is not None:
            values["signal_data_inner"] = self._row(self.signal_data_inner, "signal_data_inner", size)
            data_cross = _array(self.data_cross, "data_cross", ndim=2)
            if data_cross.shape != cross.shape:
                raise ValueError(f"data_cross must have the shape of cross {cross.shape}, got {data_cross.shape}")
            values["data_cross"] = data_cross
            values["data_finite"] = self._verdict(self.data_finite, "data_finite", size)
        if self.signal_bias_inner is not None:
            values["signal_bias_inner"] = self._row(self.signal_bias_inner, "signal_bias_inner", size)
        for name, value in values.items():
            object.__setattr__(self, name, value)

    @staticmethod
    def _row(values: ArrayLike, what: str, size: int) -> np.ndarray:
        array = _array(values, what, ndim=1)
        if array.shape != (size,):
            raise ValueError(f"{what} must have shape ({size},), got {array.shape}")
        return array

    @staticmethod
    def _verdict(values: ArrayLike, what: str, size: int) -> np.ndarray:
        array = np.asarray(values)
        if array.dtype != bool or array.shape != (size,):
            raise ValueError(f"{what} must be a boolean vector of length {size}")
        return array

    @property
    def size(self) -> int:
        """Number of signals."""
        return int(self.raw.shape[0])

    @staticmethod
    def concatenate(parts: Sequence[BankReductions]) -> BankReductions:
        """Join batches in order; every part must carry the same optional reductions."""
        return BankReductions(**_concatenated(parts, BankReductions))


@dataclass(frozen=True, eq=False)
class SignalBankResult:
    """Profiled statistics of a signal bank, one row per signal; arrays are read-only.

    The mismatch fields (``amplitude_hat``, ``q_mismatch``, ``z_mismatch``) exist
    when data residuals were supplied, the spurious fields when a bias was.
    """

    fisher_raw: np.ndarray
    fisher_profiled: np.ndarray
    sigma_amplitude: np.ndarray
    q_asimov: np.ndarray
    degradation: np.ndarray
    amplitude_hat: np.ndarray | None = None
    q_mismatch: np.ndarray | None = None
    z_mismatch: np.ndarray | None = None
    amplitude_spurious: np.ndarray | None = None
    q_spurious: np.ndarray | None = None
    z_spurious: np.ndarray | None = None

    def __post_init__(self) -> None:
        size = None
        for field in fields(self):
            value = getattr(self, field.name)
            if value is None:
                if field.default is MISSING:
                    raise ValueError(f"{field.name} is a required statistic")
                continue
            array = _array(value, field.name, ndim=1)
            if size is None:
                size = array.shape[0]
            elif array.shape != (size,):
                raise ValueError(f"{field.name} must have shape ({size},), got {array.shape}")
            object.__setattr__(self, field.name, _read_only(array))
        for group in (("amplitude_hat", "q_mismatch", "z_mismatch"),
                      ("amplitude_spurious", "q_spurious", "z_spurious")):
            present = [getattr(self, name) is not None for name in group]
            if any(present) and not all(present):
                raise ValueError(f"{', '.join(group)} are given together or not at all")

    @property
    def size(self) -> int:
        """Number of signals."""
        return int(self.fisher_raw.shape[0])

    @staticmethod
    def concatenate(parts: Sequence[SignalBankResult]) -> SignalBankResult:
        """Join banks in order; every part must carry the same optional fields."""
        return SignalBankResult(**_concatenated(parts, SignalBankResult))


def _concatenated(parts: Sequence[BankReductions | SignalBankResult], kind: type) -> dict[str, np.ndarray | None]:
    if not parts or not all(isinstance(part, kind) for part in parts):
        raise ValueError(f"concatenate needs at least one {kind.__name__}")
    joined: dict[str, np.ndarray | None] = {}
    for field in fields(kind):
        values = [getattr(part, field.name) for part in parts]
        if all(value is None for value in values):
            joined[field.name] = None
        elif any(value is None for value in values):
            raise ValueError(f"every part must carry {field.name} or none may")
        else:
            joined[field.name] = np.concatenate(values)
    return joined


def _symmetric_pseudo_inverse(matrix: np.ndarray, rcond: float) -> tuple[np.ndarray, int, float]:
    if matrix.size == 0:
        return np.zeros_like(matrix), 0, 1.0
    symmetric = 0.5 * (matrix + matrix.T)
    eigenvalues, eigenvectors = np.linalg.eigh(symmetric)
    max_abs = float(np.max(np.abs(eigenvalues)))
    tolerance = 0.0 if max_abs == 0.0 else max(rcond, symmetric.shape[0] * np.finfo(float).eps) * max_abs
    keep = eigenvalues > tolerance
    if not np.any(keep):
        return np.zeros_like(symmetric), 0, np.inf
    kept_vectors = eigenvectors[:, keep]
    pseudo_inverse = (kept_vectors * (1.0 / eigenvalues[keep])) @ kept_vectors.T
    condition = float(np.max(eigenvalues[keep]) / np.min(eigenvalues[keep]))
    return pseudo_inverse, int(np.count_nonzero(keep)), condition


class ProfileLikelihoodWorkspace:
    """Profiled amplitude statistics for one whitened nuisance design.

    The design ``(n_data, p)`` (``p`` may be 0) fixes the data size; the
    normal matrix and its pseudo-inverse are built once. Attributes are fixed at
    construction and the stored arrays are read-only.
    """

    def __init__(self, nuisance_whitened: ArrayLike, prior_precision: ArrayLike,
                 nuisance_names: Sequence[str], *, rcond: float = 1.0e-12) -> None:
        design = _array(nuisance_whitened, "nuisance_whitened", ndim=2)
        if design.shape[0] == 0:
            raise ValueError("nuisance_whitened must have at least one data row")
        _require_finite(design, "nuisance_whitened")
        n_nuisance = design.shape[1]
        precision = _array(prior_precision, "prior_precision", ndim=1)
        if precision.shape != (n_nuisance,):
            raise ValueError(f"prior_precision must have shape ({n_nuisance},), got {precision.shape}")
        _require_finite(precision, "prior_precision")
        if np.any(precision < 0.0):
            raise ValueError("prior_precision must be non-negative")
        names = tuple(nuisance_names)
        if len(names) != n_nuisance or not all(isinstance(name, str) and name for name in names):
            raise ValueError(f"nuisance_names must be {n_nuisance} non-empty strings, got {names!r}")
        if len(set(names)) != len(names):
            raise ValueError(f"nuisance_names must be unique, got {names!r}")
        if isinstance(rcond, bool) or not isinstance(rcond, Real) or not np.isfinite(rcond) or rcond <= 0.0:
            raise ValueError(f"rcond must be a positive finite number, got {rcond!r}")
        self.n_data: int = int(design.shape[0])
        self.n_nuisance: int = int(n_nuisance)
        self.nuisance_names: tuple[str, ...] = names
        self.nuisance_whitened: np.ndarray = _read_only(design)
        with threadpool_limits(limits=1):
            gram = self.nuisance_whitened.T @ self.nuisance_whitened
            normal = 0.5 * (gram + gram.T) + np.diag(precision)
            pseudo_inverse, rank, condition = _symmetric_pseudo_inverse(normal, float(rcond))
        pseudo_inverse.setflags(write=False)
        self.normal_pinv: np.ndarray = pseudo_inverse
        self.nuisance_rank: int = rank
        self.condition_number: float = condition

    def evaluate_bank(self, signals_whitened: ArrayLike, *, data_whitened: ArrayLike | None = None,
                      bias_whitened: ArrayLike | None = None) -> SignalBankResult:
        """Statistics of whitened signals ``(n_signals, n_data)``.

        ``data_whitened`` (same shape) pairs each signal with the data residual
        it is fitted to; ``bias_whitened`` ``(n_data,)`` is the fixed bias.
        """
        signals = self._bank(signals_whitened, "signals_whitened")
        data = None
        if data_whitened is not None:
            data = self._bank(data_whitened, "data_whitened")
            if data.shape != signals.shape:
                raise ValueError(f"data_whitened must have the signals' shape {signals.shape}, got {data.shape}")
        bias = None if bias_whitened is None else self._bias(bias_whitened)
        with threadpool_limits(limits=1):
            design = self.nuisance_whitened
            raw = np.einsum("ij,ij->i", signals, signals)
            cross = signals @ design
            signal_data_inner = data_cross = None
            if data is not None:
                signal_data_inner = np.einsum("ij,ij->i", signals, data)
                data_cross = data @ design
            signal_bias_inner = nuisance_bias = None
            if bias is not None:
                signal_bias_inner = signals @ bias
                nuisance_bias = design.T @ bias
            return self._finish(raw, cross, signal_data_inner, data_cross, signal_bias_inner, nuisance_bias)

    def evaluate_reductions(self, reductions: BankReductions, *,
                            bias_whitened: ArrayLike | None = None) -> SignalBankResult:
        """Statistics from reductions taken off the host; equal to :meth:`evaluate_bank` on the same vectors."""
        if reductions.cross.shape[1] != self.n_nuisance:
            raise ValueError(f"cross must have {self.n_nuisance} columns, got {reductions.cross.shape[1]}")
        if not np.all(reductions.finite):
            raise ValueError("signals_whitened contains non-finite values")
        _require_finite(reductions.raw, "raw")
        _require_finite(reductions.cross, "cross")
        if reductions.signal_data_inner is not None:
            if not np.all(reductions.data_finite):
                raise ValueError("data_whitened contains non-finite values")
            _require_finite(reductions.signal_data_inner, "signal_data_inner")
            _require_finite(reductions.data_cross, "data_cross")
        if (reductions.signal_bias_inner is None) != (bias_whitened is None):
            raise ValueError("signal_bias_inner and bias_whitened are given together or not at all")
        bias = None
        if bias_whitened is not None:
            _require_finite(reductions.signal_bias_inner, "signal_bias_inner")
            bias = self._bias(bias_whitened)
        with threadpool_limits(limits=1):
            nuisance_bias = None if bias is None else self.nuisance_whitened.T @ bias
            return self._finish(reductions.raw, reductions.cross, reductions.signal_data_inner,
                                reductions.data_cross, reductions.signal_bias_inner, nuisance_bias)

    def _bank(self, values: ArrayLike, what: str) -> np.ndarray:
        bank = _array(values, what, ndim=2)
        if bank.shape[0] == 0 or bank.shape[1] != self.n_data:
            raise ValueError(f"{what} must have shape (n_signals >= 1, {self.n_data}), got {bank.shape}")
        _require_finite(bank, what)
        return bank

    def _bias(self, values: ArrayLike) -> np.ndarray:
        bias = _array(values, "bias_whitened", ndim=1)
        if bias.shape != (self.n_data,):
            raise ValueError(f"bias_whitened must have shape ({self.n_data},), got {bias.shape}")
        _require_finite(bias, "bias_whitened")
        return bias

    def _finish(self, raw: np.ndarray, cross: np.ndarray, signal_data_inner: np.ndarray | None,
                data_cross: np.ndarray | None, signal_bias_inner: np.ndarray | None,
                nuisance_bias: np.ndarray | None) -> SignalBankResult:
        if self.n_nuisance == 0:
            profiled = raw.copy()
        else:
            profiled = raw - np.einsum("ij,jk,ik->i", cross, self.normal_pinv, cross)
            if np.any(profiled < -1.0e-10 * np.maximum(raw, 1.0)):
                raise ValueError("profiled information is significantly negative for at least one signal: "
                                 "the nuisance design or priors are numerically inconsistent")
            profiled = np.where(profiled < 0.0, 0.0, profiled)
        positive = profiled > 0.0
        statistics = {
            "fisher_raw": raw,
            "fisher_profiled": profiled,
            "sigma_amplitude": sigma_amplitude(profiled),
            "q_asimov": profiled,
            "degradation": degradation(raw, profiled),
        }
        if signal_data_inner is not None:
            numerator = signal_data_inner
            if self.n_nuisance > 0:
                numerator = numerator - np.einsum("ij,jk,ik->i", cross, self.normal_pinv, data_cross)
            amplitude_hat = np.full_like(profiled, np.nan, dtype=float)
            amplitude_hat[positive] = numerator[positive] / profiled[positive]
            z, q = amplitude_significance(amplitude_hat, profiled, signed=True)
            statistics.update(amplitude_hat=amplitude_hat, z_mismatch=z, q_mismatch=q)
        if signal_bias_inner is not None:
            numerator = signal_bias_inner
            if self.n_nuisance > 0:
                numerator = numerator - cross @ (self.normal_pinv @ nuisance_bias)
            amplitude_spurious = np.full_like(profiled, np.nan, dtype=float)
            amplitude_spurious[positive] = numerator[positive] / profiled[positive]
            z, q = amplitude_significance(amplitude_spurious, profiled, signed=False)
            statistics.update(amplitude_spurious=amplitude_spurious, z_spurious=z, q_spurious=q)
        return SignalBankResult(**statistics)
