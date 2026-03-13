"""Publication-grade Fisher/Asimov detectability tools.

This module implements the *statistical core* for a robust, sweepable
subhalo-detectability metric based on the linear-Gaussian profile likelihood.
It is intentionally independent of AutoLens / HCIPy so that the likelihood
machinery can be unit-tested and reused across different forward models.

Problem set-up
--------------
Assume a pixelized data vector ``d`` with mean model

    d ~ N(mu(A, eta), C)

where ``A`` is a scalar amplitude multiplying a fixed subhalo template and
``eta`` are nuisance parameters (macro lens, source, background, PSF modes,
regularization hyper-parameters approximated locally, etc.).  Linearizing the
mean image around a smooth null model gives

    mu(A, eta) ~= mu0 + A s + J eta,

with signal template ``s`` and nuisance Jacobian ``J`` in data space.
After whitening by ``C^{-1/2}``, the profiled Fisher information on ``A`` is

    F_A|eta = s^T s - s^T J (J^T J + P)^+ J^T s,

where ``P`` is the nuisance prior precision matrix and ``+`` denotes a
numerically stable pseudo-inverse.  The corresponding Asimov / expected
local discovery metric is

    q_A = A_true^2 F_A|eta,   Z_A ~= sqrt(q_A).

This is the local linear-Gaussian limit in which the score, Wald, and profile
likelihood-ratio tests coincide asymptotically.  It is therefore the correct
fast surrogate for likelihood-ratio sensitivity sweeps, provided it is validated
against a sparse set of full nonlinear fits.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Callable, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
from scipy.stats import norm

ArrayLike = Union[np.ndarray, Sequence[float]]
MatrixLike = Union[np.ndarray, Sequence[Sequence[float]]]


# -----------------------------------------------------------------------------
# Dataclasses
# -----------------------------------------------------------------------------


@dataclass(frozen=True)
class AsimovAmplitudeResult:
    """Result for one signal template after profiling nuisance parameters.

    Attributes
    ----------
    amplitude_true
        The assumed true amplitude of the template.
    fisher_raw
        Information on amplitude if nuisance parameters are fixed.
    fisher_profiled
        Information on amplitude after profiling / constraining nuisances.
    sigma_amplitude_raw
        Raw amplitude uncertainty, equal to ``1 / sqrt(fisher_raw)``.
    sigma_amplitude_profiled
        Profiled amplitude uncertainty, equal to ``1 / sqrt(fisher_profiled)``.
    q_asimov_local
        Local Asimov discovery statistic.
    z_asimov_local
        Local median significance, approximated by ``sqrt(q_asimov_local)``.
    local_p_one_sided
        One-sided local tail probability implied by ``z_asimov_local``.
    degradation
        Fractional information retained after nuisance profiling.
    absorbed_fraction
        Fractional information lost to nuisance degeneracies.
    residual_norm_whitened
        Squared norm of the whitened residual ``||s - J eta_hat||^2``.
        With nuisance priors this is *not* equal to ``fisher_profiled`` because
        prior penalties contribute additional profiled information.
    nuisance_prior_penalty
        Penalty term ``eta_hat^T P eta_hat`` from Gaussian nuisance priors.
    nuisance_rank
        Effective rank of the nuisance normal matrix.
    nuisance_condition_number
        Condition number of the retained nuisance eigen-spectrum.
    whitened_size
        Number of whitened data coordinates used by the computation.
    """

    amplitude_true: float
    fisher_raw: float
    fisher_profiled: float
    sigma_amplitude_raw: float
    sigma_amplitude_profiled: float
    q_asimov_local: float
    z_asimov_local: float
    local_p_one_sided: float
    degradation: float
    absorbed_fraction: float
    residual_norm_whitened: float
    nuisance_prior_penalty: float
    nuisance_rank: int
    nuisance_condition_number: float
    whitened_size: int


@dataclass(frozen=True)
class SignalBankResult:
    """Vectorized Asimov results for a bank of signal templates."""

    fisher_raw: np.ndarray
    fisher_profiled: np.ndarray
    sigma_amplitude_profiled: np.ndarray
    q_asimov_local: np.ndarray
    z_asimov_local: np.ndarray
    degradation: np.ndarray
    absorbed_fraction: np.ndarray
    amplitude_true: np.ndarray
    nuisance_rank: int
    nuisance_condition_number: float
    whitened_size: int


@dataclass(frozen=True)
class SpuriousAmplitudeResult:
    """Spurious best-fit amplitude induced by a systematic residual field."""

    amplitude_spurious: float
    z_spurious: float
    numerator: float
    sigma_amplitude_profiled: float
    fisher_profiled: float
    nuisance_rank: int
    nuisance_condition_number: float


@dataclass(frozen=True)
class SystematicModeCoupling:
    """Coupling of one systematic mode to the profiled subhalo amplitude."""

    mode_name: str
    amplitude_per_unit: float
    z_per_unit: float
    one_sigma_z: Optional[float]
    tolerance_for_zmax: Optional[float]


@dataclass(frozen=True)
class SystematicModeScanResult:
    """Mode-by-mode spurious-detection summary for systematic basis vectors."""

    couplings: Tuple[SystematicModeCoupling, ...]
    rms_spurious_amplitude: Optional[float]
    rms_spurious_z: Optional[float]
    sigma_amplitude_profiled: float
    fisher_profiled: float


# -----------------------------------------------------------------------------
# Whitening utilities
# -----------------------------------------------------------------------------


@dataclass(frozen=True)
class Whitener:
    """Apply ``C^{-1/2}`` to vectors or matrices.

    Notes
    -----
    The dense-covariance path uses a Cholesky factorization.  This is intended
    for moderate-sized problems.  For large correlated-noise problems, whiten
    externally and pass the already-whitened arrays directly to
    :class:`ProfileLikelihoodWorkspace`.
    """

    mode: str
    sigma_diag: Optional[np.ndarray] = None
    cholesky_factor: Optional[np.ndarray] = None

    @classmethod
    def identity(cls, size: int) -> "Whitener":
        size = int(size)
        if size <= 0:
            raise ValueError("Whitener.identity requires a positive size.")
        return cls(mode="identity")

    @classmethod
    def from_sigma(cls, sigma: ArrayLike) -> "Whitener":
        sigma_arr = np.asarray(sigma, dtype=float)
        if sigma_arr.ndim != 1:
            raise ValueError("sigma must be a 1D array of per-pixel standard deviations.")
        if sigma_arr.size == 0:
            raise ValueError("sigma must contain at least one element.")
        if not np.all(np.isfinite(sigma_arr)):
            raise ValueError("sigma contains non-finite values.")
        if np.any(sigma_arr <= 0.0):
            raise ValueError("sigma must be strictly positive.")
        return cls(mode="diagonal", sigma_diag=sigma_arr.copy())

    @classmethod
    def from_covariance(cls, covariance: MatrixLike) -> "Whitener":
        cov = np.asarray(covariance, dtype=float)
        if cov.ndim != 2 or cov.shape[0] != cov.shape[1]:
            raise ValueError("covariance must be a square 2D array.")
        if cov.size == 0:
            raise ValueError("covariance must be non-empty.")
        if not np.all(np.isfinite(cov)):
            raise ValueError("covariance contains non-finite values.")
        cov = 0.5 * (cov + cov.T)
        try:
            chol = np.linalg.cholesky(cov)
        except np.linalg.LinAlgError as exc:
            raise ValueError(
                "covariance must be symmetric positive definite for Cholesky whitening. "
                "If your noise model is singular or implicit, whiten externally and pass "
                "already-whitened arrays to ProfileLikelihoodWorkspace."
            ) from exc
        return cls(mode="dense", cholesky_factor=chol)

    def apply(self, arr: np.ndarray) -> np.ndarray:
        """Apply the whitener to a vector or a matrix.

        The first axis is always interpreted as the data axis.  Therefore
        vectors must have shape ``(n_data,)`` and matrices must have shape
        ``(n_data, n_columns)``.
        """
        x = np.asarray(arr, dtype=float)
        if x.ndim not in (1, 2):
            raise ValueError("Whitener.apply accepts only 1D or 2D arrays.")

        if self.mode == "identity":
            return x.copy()

        if self.mode == "diagonal":
            sigma = self.sigma_diag
            assert sigma is not None
            if x.shape[0] != sigma.size:
                raise ValueError(
                    f"Array leading dimension {x.shape[0]} does not match sigma length {sigma.size}."
                )
            if x.ndim == 1:
                return x / sigma
            return x / sigma[:, None]

        if self.mode == "dense":
            chol = self.cholesky_factor
            assert chol is not None
            if x.shape[0] != chol.shape[0]:
                raise ValueError(
                    f"Array leading dimension {x.shape[0]} does not match covariance size {chol.shape[0]}."
                )
            return np.linalg.solve(chol, x)

        raise RuntimeError(f"Unknown whitener mode: {self.mode}")


# -----------------------------------------------------------------------------
# Linear-algebra core
# -----------------------------------------------------------------------------


class ProfileLikelihoodWorkspace:
    """Reusable workspace for profiled linear-Gaussian amplitude tests.

    Parameters
    ----------
    nuisance_whitened
        Nuisance design matrix in whitened coordinates, with shape
        ``(n_data, n_nuisance)``.  May be ``None`` or empty.
    prior_precision
        Gaussian prior precision on nuisance parameters.  Accepted forms are:
        ``None`` (no prior), scalar ``lambda`` (``lambda I``), 1D diagonal, or
        full square matrix.
    nuisance_names
        Optional names for nuisance parameters.
    rcond
        Relative eigenvalue tolerance used when pseudo-inverting the nuisance
        normal matrix.
    """

    def __init__(
        self,
        nuisance_whitened: Optional[np.ndarray] = None,
        prior_precision: Optional[Union[float, ArrayLike, MatrixLike]] = None,
        nuisance_names: Optional[Sequence[str]] = None,
        rcond: float = 1.0e-12,
    ):
        if not isinstance(rcond, (int, float)) or not isfinite(float(rcond)) or float(rcond) <= 0.0:
            raise ValueError("rcond must be a positive finite scalar.")
        self.rcond = float(rcond)

        self.nuisance_whitened = self._coerce_design(nuisance_whitened)
        self.n_data = self.nuisance_whitened.shape[0]
        self.n_nuisance = self.nuisance_whitened.shape[1]

        if nuisance_names is None:
            self.nuisance_names = tuple(f"nuisance_{i}" for i in range(self.n_nuisance))
        else:
            if len(nuisance_names) != self.n_nuisance:
                raise ValueError(
                    "nuisance_names length must match the number of nuisance columns."
                )
            self.nuisance_names = tuple(str(name) for name in nuisance_names)

        self.prior_precision = self._coerce_prior_precision(prior_precision, self.n_nuisance)
        self.normal_matrix = self._build_normal_matrix()
        self.normal_pinv, self.nuisance_rank, self.nuisance_condition_number = self._stable_symmetric_pinv(
            self.normal_matrix,
            rcond=self.rcond,
        )

    @staticmethod
    def _coerce_design(design: Optional[np.ndarray]) -> np.ndarray:
        if design is None:
            return np.zeros((0, 0), dtype=float)
        arr = np.asarray(design, dtype=float)
        if arr.ndim == 1:
            arr = arr[:, None]
        if arr.ndim != 2:
            raise ValueError("nuisance_whitened must be a 2D array.")
        if arr.shape[0] == 0:
            return np.zeros((0, arr.shape[1]), dtype=float)
        if not np.all(np.isfinite(arr)):
            raise ValueError("nuisance_whitened contains non-finite values.")
        return arr.copy()

    @staticmethod
    def _coerce_prior_precision(
        prior_precision: Optional[Union[float, ArrayLike, MatrixLike]],
        n_nuisance: int,
    ) -> np.ndarray:
        if n_nuisance == 0:
            return np.zeros((0, 0), dtype=float)

        if prior_precision is None:
            return np.zeros((n_nuisance, n_nuisance), dtype=float)

        if isinstance(prior_precision, (int, float)) and not isinstance(prior_precision, bool):
            lam = float(prior_precision)
            if not isfinite(lam) or lam < 0.0:
                raise ValueError("Scalar prior_precision must be finite and non-negative.")
            return np.eye(n_nuisance, dtype=float) * lam

        arr = np.asarray(prior_precision, dtype=float)
        if arr.ndim == 1:
            if arr.size != n_nuisance:
                raise ValueError(
                    "1D prior_precision must have length equal to the number of nuisance parameters."
                )
            if not np.all(np.isfinite(arr)) or np.any(arr < 0.0):
                raise ValueError("1D prior_precision must be finite and non-negative.")
            return np.diag(arr)

        if arr.ndim != 2 or arr.shape != (n_nuisance, n_nuisance):
            raise ValueError(
                "prior_precision must be scalar, a length-p diagonal, or a square (p, p) matrix."
            )
        if not np.all(np.isfinite(arr)):
            raise ValueError("prior_precision contains non-finite values.")
        arr = 0.5 * (arr + arr.T)
        eigvals = np.linalg.eigvalsh(arr)
        if np.min(eigvals) < -1.0e-12:
            raise ValueError("prior_precision must be positive semi-definite.")
        return arr

    def _build_normal_matrix(self) -> np.ndarray:
        if self.n_nuisance == 0:
            return np.zeros((0, 0), dtype=float)
        jtj = self.nuisance_whitened.T @ self.nuisance_whitened
        return 0.5 * (jtj + jtj.T) + self.prior_precision

    @staticmethod
    def _stable_symmetric_pinv(matrix: np.ndarray, rcond: float) -> Tuple[np.ndarray, int, float]:
        if matrix.size == 0:
            return np.zeros_like(matrix), 0, 1.0

        sym = 0.5 * (matrix + matrix.T)
        eigvals, eigvecs = np.linalg.eigh(sym)
        max_abs = float(np.max(np.abs(eigvals))) if eigvals.size else 0.0
        tol = max(rcond * max(max_abs, 1.0), 1.0e-15)
        keep = eigvals > tol

        pinv = np.zeros_like(sym)
        if np.any(keep):
            v_keep = eigvecs[:, keep]
            inv_keep = 1.0 / eigvals[keep]
            pinv = (v_keep * inv_keep) @ v_keep.T
            cond = float(np.max(eigvals[keep]) / np.min(eigvals[keep]))
            rank = int(np.count_nonzero(keep))
        else:
            cond = np.inf
            rank = 0
        return pinv, rank, cond

    def _validate_signal(self, signal_whitened: ArrayLike) -> np.ndarray:
        s = np.asarray(signal_whitened, dtype=float)
        if s.ndim != 1:
            raise ValueError("signal_whitened must be a 1D array.")
        if self.n_data == 0:
            # Establish n_data from the first signal if no nuisances were supplied.
            self.n_data = s.size  # type: ignore[misc]
        elif s.size != self.n_data:
            raise ValueError(
                f"signal length {s.size} does not match workspace data size {self.n_data}."
            )
        if not np.all(np.isfinite(s)):
            raise ValueError("signal_whitened contains non-finite values.")
        return s

    def _validate_signal_bank(self, signal_bank_whitened: MatrixLike) -> np.ndarray:
        s_bank = np.asarray(signal_bank_whitened, dtype=float)
        if s_bank.ndim != 2:
            raise ValueError("signal_bank_whitened must be a 2D array with shape (n_signals, n_data).")
        if self.n_data == 0:
            self.n_data = s_bank.shape[1]  # type: ignore[misc]
        elif s_bank.shape[1] != self.n_data:
            raise ValueError(
                f"signal bank width {s_bank.shape[1]} does not match workspace data size {self.n_data}."
            )
        if not np.all(np.isfinite(s_bank)):
            raise ValueError("signal_bank_whitened contains non-finite values.")
        return s_bank

    def _profiled_information(self, signal_whitened: np.ndarray) -> Tuple[float, float, np.ndarray, float, float]:
        raw = float(signal_whitened @ signal_whitened)

        if self.n_nuisance == 0:
            coeffs = np.zeros(0, dtype=float)
            residual_norm = raw
            prior_penalty = 0.0
            return raw, raw, coeffs, residual_norm, prior_penalty

        jts = self.nuisance_whitened.T @ signal_whitened
        coeffs = self.normal_pinv @ jts
        residual = signal_whitened - self.nuisance_whitened @ coeffs
        residual_norm = float(residual @ residual)
        prior_penalty = float(coeffs @ (self.prior_precision @ coeffs))
        profiled = raw - float(jts @ coeffs)

        # Numerical guard: the Schur complement should be non-negative.
        if profiled < 0.0:
            if profiled > -1.0e-10 * max(raw, 1.0):
                profiled = 0.0
            else:
                raise ValueError(
                    "Profiled information is significantly negative, indicating numerical instability "
                    "or an inconsistent nuisance prior / design matrix."
                )
        return raw, float(profiled), coeffs, residual_norm, prior_penalty

    def evaluate_signal(
        self,
        signal_whitened: ArrayLike,
        amplitude_true: float = 1.0,
    ) -> AsimovAmplitudeResult:
        """Evaluate one whitened signal template."""
        if not isinstance(amplitude_true, (int, float)) or not isfinite(float(amplitude_true)):
            raise ValueError("amplitude_true must be a finite scalar.")
        amplitude_true = float(amplitude_true)

        signal = self._validate_signal(signal_whitened)
        raw, profiled, _, residual_norm, prior_penalty = self._profiled_information(signal)

        sigma_raw = np.inf if raw <= 0.0 else 1.0 / np.sqrt(raw)
        sigma_profiled = np.inf if profiled <= 0.0 else 1.0 / np.sqrt(profiled)
        q_asimov = amplitude_true * amplitude_true * profiled
        z_asimov = float(np.sqrt(max(q_asimov, 0.0)))
        local_p = float(norm.sf(z_asimov))
        degradation = float(profiled / raw) if raw > 0.0 else 0.0
        degradation = min(max(degradation, 0.0), 1.0)

        return AsimovAmplitudeResult(
            amplitude_true=amplitude_true,
            fisher_raw=float(raw),
            fisher_profiled=float(profiled),
            sigma_amplitude_raw=float(sigma_raw),
            sigma_amplitude_profiled=float(sigma_profiled),
            q_asimov_local=float(q_asimov),
            z_asimov_local=z_asimov,
            local_p_one_sided=local_p,
            degradation=degradation,
            absorbed_fraction=float(1.0 - degradation),
            residual_norm_whitened=float(residual_norm),
            nuisance_prior_penalty=float(prior_penalty),
            nuisance_rank=int(self.nuisance_rank),
            nuisance_condition_number=float(self.nuisance_condition_number),
            whitened_size=int(signal.size),
        )

    def evaluate_signal_bank(
        self,
        signal_bank_whitened: MatrixLike,
        amplitude_true: Union[float, ArrayLike] = 1.0,
    ) -> SignalBankResult:
        """Vectorized evaluation for many whitened signal templates."""
        signals = self._validate_signal_bank(signal_bank_whitened)

        if np.isscalar(amplitude_true):
            amp = np.full(signals.shape[0], float(amplitude_true), dtype=float)
        else:
            amp = np.asarray(amplitude_true, dtype=float)
            if amp.ndim != 1 or amp.size != signals.shape[0]:
                raise ValueError(
                    "amplitude_true must be a scalar or a 1D array with length equal to n_signals."
                )
        if not np.all(np.isfinite(amp)):
            raise ValueError("amplitude_true contains non-finite values.")

        raw = np.einsum("ij,ij->i", signals, signals)
        if self.n_nuisance == 0:
            profiled = raw.copy()
        else:
            cross = signals @ self.nuisance_whitened
            profiled = raw - np.einsum("ij,jk,ik->i", cross, self.normal_pinv, cross)
            tol = 1.0e-10 * np.maximum(raw, 1.0)
            bad = profiled < -tol
            if np.any(bad):
                raise ValueError(
                    "Profiled information became significantly negative for at least one signal in the bank."
                )
            profiled = np.where(profiled < 0.0, 0.0, profiled)

        sigma_profiled = np.full_like(profiled, np.inf, dtype=float)
        positive = profiled > 0.0
        sigma_profiled[positive] = 1.0 / np.sqrt(profiled[positive])
        q_asimov = amp * amp * profiled
        z_asimov = np.sqrt(np.maximum(q_asimov, 0.0))
        degradation = np.divide(profiled, raw, out=np.zeros_like(profiled), where=raw > 0.0)
        degradation = np.clip(degradation, 0.0, 1.0)

        return SignalBankResult(
            fisher_raw=raw,
            fisher_profiled=profiled,
            sigma_amplitude_profiled=sigma_profiled,
            q_asimov_local=q_asimov,
            z_asimov_local=z_asimov,
            degradation=degradation,
            absorbed_fraction=1.0 - degradation,
            amplitude_true=amp,
            nuisance_rank=int(self.nuisance_rank),
            nuisance_condition_number=float(self.nuisance_condition_number),
            whitened_size=int(signals.shape[1]),
        )

    def spurious_from_bias(
        self,
        signal_whitened: ArrayLike,
        bias_whitened: ArrayLike,
    ) -> SpuriousAmplitudeResult:
        """Compute the spurious best-fit amplitude induced by a bias field."""
        signal = self._validate_signal(signal_whitened)
        bias = np.asarray(bias_whitened, dtype=float)
        if bias.ndim != 1 or bias.size != signal.size:
            raise ValueError("bias_whitened must be a 1D array with the same size as signal_whitened.")
        if not np.all(np.isfinite(bias)):
            raise ValueError("bias_whitened contains non-finite values.")

        signal_result = self.evaluate_signal(signal)
        fisher_profiled = signal_result.fisher_profiled
        if fisher_profiled <= 0.0:
            amp_spur = np.inf
            numerator = np.nan
            z_spur = np.inf
        elif self.n_nuisance == 0:
            numerator = float(signal @ bias)
            amp_spur = numerator / fisher_profiled
            z_spur = abs(amp_spur) / signal_result.sigma_amplitude_profiled
        else:
            jts = self.nuisance_whitened.T @ signal
            jtb = self.nuisance_whitened.T @ bias
            numerator = float((signal @ bias) - (jts @ (self.normal_pinv @ jtb)))
            amp_spur = numerator / fisher_profiled
            z_spur = abs(amp_spur) / signal_result.sigma_amplitude_profiled

        return SpuriousAmplitudeResult(
            amplitude_spurious=float(amp_spur),
            z_spurious=float(z_spur),
            numerator=float(numerator),
            sigma_amplitude_profiled=float(signal_result.sigma_amplitude_profiled),
            fisher_profiled=float(fisher_profiled),
            nuisance_rank=int(self.nuisance_rank),
            nuisance_condition_number=float(self.nuisance_condition_number),
        )

    def scan_systematic_modes(
        self,
        signal_whitened: ArrayLike,
        systematic_modes_whitened: MatrixLike,
        mode_names: Optional[Sequence[str]] = None,
        mode_sigmas: Optional[ArrayLike] = None,
        z_tolerance: Optional[float] = 1.0,
        systematic_covariance: Optional[MatrixLike] = None,
        progress: Optional[Callable[[Iterable[int]], Iterable[int]]] = None,
    ) -> SystematicModeScanResult:
        """Scan a basis of systematic modes for spurious subhalo coupling.

        Parameters
        ----------
        signal_whitened
            Whitened subhalo template.
        systematic_modes_whitened
            Matrix of systematic modes (e.g. PSF derivative images) in whitened
            coordinates, with shape ``(n_data, n_modes)``.
        mode_names
            Optional names for systematic modes.
        mode_sigmas
            Optional 1-sigma amplitudes for each systematic mode.  When provided,
            the result reports the corresponding one-sigma spurious significance.
        z_tolerance
            Optional significance budget used to convert each mode coupling into a
            tolerance ``|delta a_k| < z_tolerance / |z_per_unit|``.
        systematic_covariance
            Optional covariance of the systematic mode amplitudes.  When provided,
            the result also reports the RMS spurious amplitude and significance
            from the full covariance.
        """
        signal = self._validate_signal(signal_whitened)
        modes = np.asarray(systematic_modes_whitened, dtype=float)
        if modes.ndim == 1:
            modes = modes[:, None]
        if modes.ndim != 2 or modes.shape[0] != signal.size:
            raise ValueError(
                "systematic_modes_whitened must have shape (n_data, n_modes) and match signal size."
            )
        if not np.all(np.isfinite(modes)):
            raise ValueError("systematic_modes_whitened contains non-finite values.")

        n_modes = modes.shape[1]
        if mode_names is None:
            names = [f"mode_{i}" for i in range(n_modes)]
        else:
            if len(mode_names) != n_modes:
                raise ValueError("mode_names length must equal the number of systematic modes.")
            names = [str(name) for name in mode_names]

        sigmas = None
        if mode_sigmas is not None:
            sigmas = np.asarray(mode_sigmas, dtype=float)
            if sigmas.ndim != 1 or sigmas.size != n_modes:
                raise ValueError("mode_sigmas must be a 1D array with length equal to n_modes.")
            if not np.all(np.isfinite(sigmas)) or np.any(sigmas < 0.0):
                raise ValueError("mode_sigmas must be finite and non-negative.")

        signal_result = self.evaluate_signal(signal)
        fisher_profiled = signal_result.fisher_profiled
        sigma_amp = signal_result.sigma_amplitude_profiled
        if fisher_profiled <= 0.0:
            couplings = tuple(
                SystematicModeCoupling(
                    mode_name=name,
                    amplitude_per_unit=np.inf,
                    z_per_unit=np.inf,
                    one_sigma_z=np.inf if sigmas is not None else None,
                    tolerance_for_zmax=0.0 if z_tolerance is not None else None,
                )
                for name in names
            )
            return SystematicModeScanResult(
                couplings=couplings,
                rms_spurious_amplitude=np.inf,
                rms_spurious_z=np.inf,
                sigma_amplitude_profiled=float(sigma_amp),
                fisher_profiled=float(fisher_profiled),
            )

        amp_per_unit = np.empty(n_modes, dtype=float)
        mode_indices: Iterable[int] = range(n_modes)
        if progress is not None:
            mode_indices = progress(mode_indices)
        for idx in mode_indices:
            amp_per_unit[idx] = self.spurious_from_bias(signal, modes[:, idx]).amplitude_spurious
        z_per_unit = np.abs(amp_per_unit) / sigma_amp

        couplings_list: List[SystematicModeCoupling] = []
        for idx, name in enumerate(names):
            one_sigma_z = None if sigmas is None else float(z_per_unit[idx] * sigmas[idx])
            if z_tolerance is None:
                tol = None
            elif z_per_unit[idx] == 0.0:
                tol = np.inf
            else:
                tol = float(z_tolerance / z_per_unit[idx])
            couplings_list.append(
                SystematicModeCoupling(
                    mode_name=name,
                    amplitude_per_unit=float(amp_per_unit[idx]),
                    z_per_unit=float(z_per_unit[idx]),
                    one_sigma_z=one_sigma_z,
                    tolerance_for_zmax=tol,
                )
            )

        rms_amp = None
        rms_z = None
        if systematic_covariance is not None:
            cov = np.asarray(systematic_covariance, dtype=float)
            if cov.ndim != 2 or cov.shape != (n_modes, n_modes):
                raise ValueError(
                    "systematic_covariance must be a square (n_modes, n_modes) matrix."
                )
            if not np.all(np.isfinite(cov)):
                raise ValueError("systematic_covariance contains non-finite values.")
            cov = 0.5 * (cov + cov.T)
            eigvals = np.linalg.eigvalsh(cov)
            if np.min(eigvals) < -1.0e-12:
                raise ValueError("systematic_covariance must be positive semi-definite.")
            var_amp = float(amp_per_unit @ cov @ amp_per_unit)
            var_amp = max(var_amp, 0.0)
            rms_amp = float(np.sqrt(var_amp))
            rms_z = float(rms_amp / sigma_amp)

        return SystematicModeScanResult(
            couplings=tuple(couplings_list),
            rms_spurious_amplitude=rms_amp,
            rms_spurious_z=rms_z,
            sigma_amplitude_profiled=float(sigma_amp),
            fisher_profiled=float(fisher_profiled),
        )


# -----------------------------------------------------------------------------
# Convenience wrappers operating on unwhitened arrays
# -----------------------------------------------------------------------------


def _build_whitener(
    n_data: int,
    sigma: Optional[ArrayLike] = None,
    covariance: Optional[MatrixLike] = None,
) -> Whitener:
    if sigma is not None and covariance is not None:
        raise ValueError("Provide either sigma or covariance, not both.")
    if sigma is not None:
        return Whitener.from_sigma(sigma)
    if covariance is not None:
        return Whitener.from_covariance(covariance)
    return Whitener.identity(n_data)


def _coerce_data_vector(name: str, vec: ArrayLike) -> np.ndarray:
    arr = np.asarray(vec, dtype=float)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be a 1D array.")
    if arr.size == 0:
        raise ValueError(f"{name} must be non-empty.")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} contains non-finite values.")
    return arr


def _coerce_data_matrix(name: str, mat: Optional[MatrixLike], n_data: int) -> Optional[np.ndarray]:
    if mat is None:
        return None
    arr = np.asarray(mat, dtype=float)
    if arr.ndim == 1:
        arr = arr[:, None]
    if arr.ndim != 2:
        raise ValueError(f"{name} must be a 2D array.")
    if arr.shape[0] != n_data:
        raise ValueError(f"{name} must have shape (n_data, n_columns) with n_data={n_data}.")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} contains non-finite values.")
    return arr


def compute_asimov_detectability(
    signal: ArrayLike,
    nuisance_jacobian: Optional[MatrixLike] = None,
    sigma: Optional[ArrayLike] = None,
    covariance: Optional[MatrixLike] = None,
    prior_precision: Optional[Union[float, ArrayLike, MatrixLike]] = None,
    amplitude_true: float = 1.0,
    nuisance_names: Optional[Sequence[str]] = None,
    rcond: float = 1.0e-12,
) -> AsimovAmplitudeResult:
    """Convenience wrapper for one unwhitened signal template."""
    signal_arr = _coerce_data_vector("signal", signal)
    nuisance_arr = _coerce_data_matrix("nuisance_jacobian", nuisance_jacobian, signal_arr.size)
    whitener = _build_whitener(signal_arr.size, sigma=sigma, covariance=covariance)
    signal_w = whitener.apply(signal_arr)
    nuisance_w = None if nuisance_arr is None else whitener.apply(nuisance_arr)
    workspace = ProfileLikelihoodWorkspace(
        nuisance_whitened=nuisance_w,
        prior_precision=prior_precision,
        nuisance_names=nuisance_names,
        rcond=rcond,
    )
    return workspace.evaluate_signal(signal_w, amplitude_true=amplitude_true)


def evaluate_signal_bank(
    signal_bank: MatrixLike,
    nuisance_jacobian: Optional[MatrixLike] = None,
    sigma: Optional[ArrayLike] = None,
    covariance: Optional[MatrixLike] = None,
    prior_precision: Optional[Union[float, ArrayLike, MatrixLike]] = None,
    amplitude_true: Union[float, ArrayLike] = 1.0,
    nuisance_names: Optional[Sequence[str]] = None,
    rcond: float = 1.0e-12,
) -> SignalBankResult:
    """Convenience wrapper for a bank of unwhitened signal templates.

    ``signal_bank`` must have shape ``(n_signals, n_data)``.
    """
    signals = np.asarray(signal_bank, dtype=float)
    if signals.ndim != 2:
        raise ValueError("signal_bank must be a 2D array with shape (n_signals, n_data).")
    if signals.shape[0] == 0 or signals.shape[1] == 0:
        raise ValueError("signal_bank must be non-empty.")
    if not np.all(np.isfinite(signals)):
        raise ValueError("signal_bank contains non-finite values.")

    nuisance_arr = _coerce_data_matrix("nuisance_jacobian", nuisance_jacobian, signals.shape[1])
    whitener = _build_whitener(signals.shape[1], sigma=sigma, covariance=covariance)
    signal_w = whitener.apply(signals.T).T
    nuisance_w = None if nuisance_arr is None else whitener.apply(nuisance_arr)
    workspace = ProfileLikelihoodWorkspace(
        nuisance_whitened=nuisance_w,
        prior_precision=prior_precision,
        nuisance_names=nuisance_names,
        rcond=rcond,
    )
    return workspace.evaluate_signal_bank(signal_w, amplitude_true=amplitude_true)


def compute_spurious_amplitude(
    signal: ArrayLike,
    bias: ArrayLike,
    nuisance_jacobian: Optional[MatrixLike] = None,
    sigma: Optional[ArrayLike] = None,
    covariance: Optional[MatrixLike] = None,
    prior_precision: Optional[Union[float, ArrayLike, MatrixLike]] = None,
    nuisance_names: Optional[Sequence[str]] = None,
    rcond: float = 1.0e-12,
) -> SpuriousAmplitudeResult:
    """Convenience wrapper for one unwhitened systematic bias field."""
    signal_arr = _coerce_data_vector("signal", signal)
    bias_arr = _coerce_data_vector("bias", bias)
    if bias_arr.size != signal_arr.size:
        raise ValueError("bias must have the same size as signal.")
    nuisance_arr = _coerce_data_matrix("nuisance_jacobian", nuisance_jacobian, signal_arr.size)
    whitener = _build_whitener(signal_arr.size, sigma=sigma, covariance=covariance)
    signal_w = whitener.apply(signal_arr)
    bias_w = whitener.apply(bias_arr)
    nuisance_w = None if nuisance_arr is None else whitener.apply(nuisance_arr)
    workspace = ProfileLikelihoodWorkspace(
        nuisance_whitened=nuisance_w,
        prior_precision=prior_precision,
        nuisance_names=nuisance_names,
        rcond=rcond,
    )
    return workspace.spurious_from_bias(signal_w, bias_w)


def scan_systematic_modes(
    signal: ArrayLike,
    systematic_modes: MatrixLike,
    nuisance_jacobian: Optional[MatrixLike] = None,
    sigma: Optional[ArrayLike] = None,
    covariance: Optional[MatrixLike] = None,
    prior_precision: Optional[Union[float, ArrayLike, MatrixLike]] = None,
    nuisance_names: Optional[Sequence[str]] = None,
    mode_names: Optional[Sequence[str]] = None,
    mode_sigmas: Optional[ArrayLike] = None,
    z_tolerance: Optional[float] = 1.0,
    systematic_covariance: Optional[MatrixLike] = None,
    progress: Optional[Callable[[Iterable[int]], Iterable[int]]] = None,
    rcond: float = 1.0e-12,
) -> SystematicModeScanResult:
    """Convenience wrapper for systematic mode scans on unwhitened arrays."""
    signal_arr = _coerce_data_vector("signal", signal)
    syst_arr = _coerce_data_matrix("systematic_modes", systematic_modes, signal_arr.size)
    nuisance_arr = _coerce_data_matrix("nuisance_jacobian", nuisance_jacobian, signal_arr.size)
    whitener = _build_whitener(signal_arr.size, sigma=sigma, covariance=covariance)
    signal_w = whitener.apply(signal_arr)
    syst_w = whitener.apply(syst_arr)
    nuisance_w = None if nuisance_arr is None else whitener.apply(nuisance_arr)
    workspace = ProfileLikelihoodWorkspace(
        nuisance_whitened=nuisance_w,
        prior_precision=prior_precision,
        nuisance_names=nuisance_names,
        rcond=rcond,
    )
    return workspace.scan_systematic_modes(
        signal_whitened=signal_w,
        systematic_modes_whitened=syst_w,
        mode_names=mode_names,
        mode_sigmas=mode_sigmas,
        z_tolerance=z_tolerance,
        systematic_covariance=systematic_covariance,
        progress=progress,
    )


# -----------------------------------------------------------------------------
# Simple helpers for thresholds / look-elsewhere bookkeeping / sensitivity maps
# -----------------------------------------------------------------------------


def sidak_local_p(global_p: float, n_eff: int) -> float:
    """Convert a desired global false-alarm rate into a local Sidak p-value."""
    if not isinstance(global_p, (int, float)) or not isfinite(float(global_p)):
        raise ValueError("global_p must be a finite scalar.")
    global_p = float(global_p)
    if not (0.0 < global_p < 1.0):
        raise ValueError("global_p must lie strictly between 0 and 1.")
    if isinstance(n_eff, bool) or not isinstance(n_eff, int) or n_eff <= 0:
        raise ValueError("n_eff must be a positive integer.")
    return float(1.0 - (1.0 - global_p) ** (1.0 / n_eff))


def sidak_local_z(global_p: float, n_eff: int) -> float:
    """Local one-sided Gaussian threshold corresponding to Sidak correction."""
    return float(norm.isf(sidak_local_p(global_p, n_eff)))


def bonferroni_local_p(global_p: float, n_eff: int) -> float:
    """Conservative Bonferroni local p-value for ``n_eff`` effective trials."""
    if not isinstance(global_p, (int, float)) or not isfinite(float(global_p)):
        raise ValueError("global_p must be a finite scalar.")
    global_p = float(global_p)
    if not (0.0 < global_p < 1.0):
        raise ValueError("global_p must lie strictly between 0 and 1.")
    if isinstance(n_eff, bool) or not isinstance(n_eff, int) or n_eff <= 0:
        raise ValueError("n_eff must be a positive integer.")
    return float(global_p / n_eff)


def global_p_from_local(local_p: float, n_eff: int) -> float:
    """Approximate global false-alarm rate from a local p-value and ``n_eff`` trials."""
    if not isinstance(local_p, (int, float)) or not isfinite(float(local_p)):
        raise ValueError("local_p must be a finite scalar.")
    local_p = float(local_p)
    if not (0.0 <= local_p <= 1.0):
        raise ValueError("local_p must lie in [0, 1].")
    if isinstance(n_eff, bool) or not isinstance(n_eff, int) or n_eff <= 0:
        raise ValueError("n_eff must be a positive integer.")
    return float(1.0 - (1.0 - local_p) ** n_eff)


def detectable_area(metric_values: ArrayLike, cell_area: float, threshold: float) -> float:
    """Compute detectable area for a sensitivity map.

    Parameters
    ----------
    metric_values
        Array of local detectability values (e.g. ``Z_A`` or ``q_A``) per map cell.
    cell_area
        Physical or angular area represented by one map cell.
    threshold
        Detection threshold applied to ``metric_values``.
    """
    values = np.asarray(metric_values, dtype=float)
    if values.size == 0:
        raise ValueError("metric_values must be non-empty.")
    if not np.all(np.isfinite(values)):
        raise ValueError("metric_values contains non-finite values.")
    if not isinstance(cell_area, (int, float)) or not isfinite(float(cell_area)) or float(cell_area) <= 0.0:
        raise ValueError("cell_area must be a positive finite scalar.")
    if not isinstance(threshold, (int, float)) or not isfinite(float(threshold)):
        raise ValueError("threshold must be a finite scalar.")
    return float(np.count_nonzero(values >= float(threshold)) * float(cell_area))


__all__ = [
    "AsimovAmplitudeResult",
    "SignalBankResult",
    "SpuriousAmplitudeResult",
    "SystematicModeCoupling",
    "SystematicModeScanResult",
    "Whitener",
    "ProfileLikelihoodWorkspace",
    "compute_asimov_detectability",
    "evaluate_signal_bank",
    "compute_spurious_amplitude",
    "scan_systematic_modes",
    "sidak_local_p",
    "sidak_local_z",
    "bonferroni_local_p",
    "global_p_from_local",
    "detectable_area",
]
