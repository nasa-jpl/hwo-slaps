"""Stable profiled linear-Gaussian algebra in augmented data/prior space.

With whitened nuisance design J and diagonal Gaussian precision P, form
A = [J; diag(sqrt(P))] and normalize each nonzero augmented column. An SVD
retains singular values greater than max(rcond, p*eps) times the largest.
The default rcond=1e-12 acts on singular values in normalized coordinates,
not eigenvalues of a normal matrix in arbitrary physical parameter units.

Signals, data and bias are augmented with zero prior rows. Information is the
squared norm of the orthogonal signal residual, including its prior block;
fitted and spurious numerators are residual inner products. A second projection
pass removes floating point leakage along the retained basis. No information
or numerator is obtained by subtracting two large scalar reductions.

Zero-information templates have q=0 and unidentified (NaN) fitted amplitudes.
Non-finite inputs or reductions raise explicitly. Dense host algebra uses one
BLAS thread and accelerators return only residual-space scalar reductions.
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
    "amplitude_significance", "degradation", "residual_blocks", "sigma_amplitude", "z_asimov",
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
    """``(z, q = z**2)`` of a fitted amplitude: ``z = a sqrt(F)`` (``|a| sqrt(F)`` unsigned), zero where ``F = 0``; amplitudes remain unidentified."""
    positive = fisher_profiled > 0.0
    z = np.where(fisher_profiled == 0.0, 0.0, np.nan)
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

    def __setstate__(self, state: dict[str, object]) -> None:
        """Unpickle through the constructor checks; numpy restores arrays writeable."""
        self.__dict__.update(state)
        self.__post_init__()

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
            with threadpool_limits(limits=1):
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
        with threadpool_limits(limits=1):
            return np.linalg.solve(self.cholesky_factor, array)


def residual_blocks(values, basis_data, basis_prior):
    """Project row vectors (or one vector) without materializing augmented rows.

    This array-only primitive is shared by NumPy and JAX. The prior residual
    is essential: discarding it changes a Gaussian-constrained profile fit.
    Storage is O(batch*n_data + batch*n_prior), independent of bank size.
    """
    coefficients = values @ basis_data
    data = values - coefficients @ basis_data.T
    prior = -(coefficients @ basis_prior.T)
    correction = data @ basis_data + prior @ basis_prior
    return data - correction @ basis_data.T, prior - correction @ basis_prior.T


@dataclass(frozen=True, eq=False)
class BankReductions:
    """Device reductions of orthogonal residuals, one row per signal.

    ``profiled`` includes squared data AND prior residual blocks. Optional
    numerators are residual inner products, never differences of raw products.
    Finiteness verdicts are taken on device while the source vectors exist.
    """

    raw: np.ndarray
    profiled: np.ndarray
    finite: np.ndarray
    signal_data_residual_inner: np.ndarray | None = None
    data_finite: np.ndarray | None = None
    signal_bias_residual_inner: np.ndarray | None = None

    def __post_init__(self) -> None:
        raw = _array(self.raw, "raw", ndim=1)
        size = raw.shape[0]
        values = {"raw": raw, "profiled": self._row(self.profiled, "profiled", size),
                  "finite": self._verdict(self.finite, "finite", size)}
        if (self.signal_data_residual_inner is None) != (self.data_finite is None):
            raise ValueError("signal_data_residual_inner and data_finite are given together or not at all")
        for name in ("signal_data_residual_inner", "signal_bias_residual_inner"):
            if getattr(self, name) is not None:
                values[name] = self._row(getattr(self, name), name, size)
        if self.data_finite is not None:
            values["data_finite"] = self._verdict(self.data_finite, "data_finite", size)
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
        """Join batches in order; every part carries the same optional reductions."""
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

    def __setstate__(self, state: dict[str, object]) -> None:
        """Unpickle through the constructor checks; numpy restores arrays writeable."""
        self.__dict__.update(state)
        self.__post_init__()

    @property
    def template_status(self) -> np.ndarray:
        """Identifiability per signal; failed numerical evaluations raise earlier."""
        return _read_only(np.where(self.fisher_profiled == 0.0, "zero_information", "valid"))

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


class ProfileLikelihoodWorkspace:
    """Profiled amplitude statistics for one whitened nuisance design.

    The design ``(n_data, p)`` (``p`` may be 0) fixes the data size; the
    normalized augmented SVD and residual basis are built once. Attributes are fixed at
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
        self.prior_precision = _read_only(precision)
        self.rank_relative_tolerance = max(float(rcond), max(1, n_nuisance) * np.finfo(float).eps)
        with threadpool_limits(limits=1):
            augmented = np.vstack((design, np.diag(np.sqrt(precision))))
            # Scale via maxima first to avoid overflow/underflow in column norms.
            maxima = np.max(np.abs(augmented), axis=0)
            nonzero = maxima > 0.0
            scaled = np.zeros_like(augmented)
            scaled[:, nonzero] = augmented[:, nonzero] / maxima[nonzero]
            norms = np.sqrt(np.sum(scaled * scaled, axis=0))
            scaled[:, nonzero] /= norms[nonzero]
            try:
                basis, singular, _ = np.linalg.svd(scaled, full_matrices=False)
            except np.linalg.LinAlgError as error:
                raise ValueError("normalized augmented nuisance SVD failed") from error
        _require_finite(singular, "normalized nuisance singular values")
        largest = float(singular[0]) if singular.size else 0.0
        keep = singular > self.rank_relative_tolerance * largest
        refined = singular > self.rank_relative_tolerance * 0.1 * largest
        self.nuisance_rank = int(np.count_nonzero(keep))
        self.rank_at_refined_tolerance = int(np.count_nonzero(refined))
        self.rank_tolerance_sensitive = self.nuisance_rank != self.rank_at_refined_tolerance
        self.condition_number = (float(singular[keep][0] / singular[keep][-1]) if self.nuisance_rank else
                                 (1.0 if n_nuisance == 0 else np.inf))
        self.normalized_singular_values = _read_only(singular)
        self.basis_data = _read_only(basis[:self.n_data, keep])
        self.basis_prior = _read_only(basis[self.n_data:, keep])
        # A norm at the roundoff floor is unidentified. This squared threshold
        # does not erase genuinely small residuals at sqrt(eps)*signal scale.
        self.zero_information_relative_norm = 8.0 * np.finfo(float).eps * max(1.0, np.sqrt(self.nuisance_rank))

    def __setstate__(self, state: dict[str, object]) -> None:
        """Restore read-only arrays after unpickling."""
        self.__dict__.update(state)
        for name in ("nuisance_whitened", "prior_precision", "normalized_singular_values", "basis_data", "basis_prior"):
            getattr(self, name).setflags(write=False)

    def evaluate_bank(self, signals_whitened: ArrayLike, *, data_whitened: ArrayLike | None = None,
                      bias_whitened: ArrayLike | None = None) -> SignalBankResult:
        """Statistics of whitened signals ``(n_signals, n_data)``.

        Optional data pairs with each signal; bias is a fixed ``(n_data,)``
        vector. Only the current bank's data and prior residuals are retained.
        """
        signals = self._bank(signals_whitened, "signals_whitened")
        data = None
        if data_whitened is not None:
            data = self._bank(data_whitened, "data_whitened")
            if data.shape != signals.shape:
                raise ValueError(f"data_whitened must have the signals' shape {signals.shape}, got {data.shape}")
        bias = None if bias_whitened is None else self._bias(bias_whitened)
        with threadpool_limits(limits=1):
            sr, sp = residual_blocks(signals, self.basis_data, self.basis_prior)
            raw = np.einsum("ij,ij->i", signals, signals)
            profiled = np.einsum("ij,ij->i", sr, sr) + np.einsum("ij,ij->i", sp, sp)
            data_inner = bias_inner = None
            if data is not None:
                dr, dp = residual_blocks(data, self.basis_data, self.basis_prior)
                data_inner = np.einsum("ij,ij->i", sr, dr) + np.einsum("ij,ij->i", sp, dp)
            if bias is not None:
                br, bp = residual_blocks(bias, self.basis_data, self.basis_prior)
                bias_inner = sr @ br + sp @ bp
            return self.evaluate_reductions(BankReductions(raw, profiled, np.ones(len(signals), dtype=bool),
                data_inner, None if data is None else np.ones(len(signals), dtype=bool), bias_inner),
                bias_whitened=bias)

    def evaluate_reductions(self, reductions: BankReductions, *,
                            bias_whitened: ArrayLike | None = None) -> SignalBankResult:
        """Finish residual-space reductions without transferring image banks."""
        if not np.all(reductions.finite):
            raise ValueError("signals_whitened contains non-finite values")
        if reductions.data_finite is not None and not np.all(reductions.data_finite):
            raise ValueError("data_whitened contains non-finite values")
        for name in ("raw", "profiled", "signal_data_residual_inner", "signal_bias_residual_inner"):
            value = getattr(reductions, name)
            if value is not None:
                _require_finite(value, name)
        if (reductions.signal_bias_residual_inner is None) != (bias_whitened is None):
            raise ValueError("signal_bias_residual_inner and bias_whitened are given together or not at all")
        if bias_whitened is not None:
            self._bias(bias_whitened)
        raw, profiled = reductions.raw, reductions.profiled
        if np.any(raw < 0.0) or np.any(profiled < 0.0):
            raise ValueError("residual information must be non-negative")
        if np.any(profiled > raw * (1.0 + 1.e-10) + np.finfo(float).tiny):
            raise ValueError("profiled residual information exceeds raw information")
        profiled = np.where(profiled <= raw * self.zero_information_relative_norm**2, 0.0, profiled)
        return self._finish(raw, profiled, reductions.signal_data_residual_inner,
                            reductions.signal_bias_residual_inner)

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

    def _finish(self, raw: np.ndarray, profiled: np.ndarray, data_inner: np.ndarray | None,
                bias_inner: np.ndarray | None) -> SignalBankResult:
        positive = profiled > 0.0
        statistics = {
            "fisher_raw": raw, "fisher_profiled": profiled,
            "sigma_amplitude": sigma_amplitude(profiled), "q_asimov": profiled,
            "degradation": degradation(raw, profiled),
        }
        for numerator, amplitude_name, z_name, q_name, signed in (
            (data_inner, "amplitude_hat", "z_mismatch", "q_mismatch", True),
            (bias_inner, "amplitude_spurious", "z_spurious", "q_spurious", False),
        ):
            if numerator is None:
                continue
            amplitude = np.full_like(profiled, np.nan)
            amplitude[positive] = numerator[positive] / profiled[positive]
            z, q = amplitude_significance(amplitude, profiled, signed=signed)
            if not np.all(np.isfinite(amplitude[positive])) or not np.all(np.isfinite(q)):
                raise ValueError("fitted amplitude or statistic is non-finite for an identifiable template")
            statistics.update({amplitude_name: amplitude, z_name: z, q_name: q})
        return SignalBankResult(**statistics)
