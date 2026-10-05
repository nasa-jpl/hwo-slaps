"""Template-engine inputs and the host bank schedule shared by every engine."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Protocol, Sequence

import numpy as np

from ...observation.expected import Exposure
from ...optics.kernels import KernelBinding
from ...scene.builder import Scene
from ...scene.cosmology import Cosmology
from ...scene.halos import Halo, HaloModel, make_halo
from ..data_space import DataSpace
from ..renderer import SceneRenderer
from ..statistics import BankReductions, ProfileLikelihoodWorkspace, SignalBankResult

HOST_FLUSH_POSITIONS = 256


@dataclass(frozen=True)
class EngineContext:
    """Mass-independent preparation inputs; a halo is constructed only for an evaluated mass."""

    renderer: SceneRenderer
    smooth_scene: Scene
    hypothesis_model: HaloModel
    hypothesis_redshift: float
    source_redshift: float
    cosmology: Cosmology
    model_kernels: KernelBinding
    truth_kernels: KernelBinding | None
    exposure: Exposure
    mean_model_adu: np.ndarray
    data_space: DataSpace
    workspace: ProfileLikelihoodWorkspace
    bias_whitened: np.ndarray | None
    lens_centre_yx: tuple[float, float]
    domain_radius_arcsec: float
    model_constant_adu: float | np.ndarray
    truth_constant_adu: float | np.ndarray | None

    @property
    def mismatched(self) -> bool:
        return self.truth_kernels is not None

    def hypothesis(self, mass_msun: float, position_yx: tuple[float, float]) -> Halo:
        return make_halo(self.hypothesis_model, mass_msun, position_yx, redshift=self.hypothesis_redshift,
                         source_redshift=self.source_redshift, cosmology=self.cosmology)


class TemplateEngine(Protocol):
    kind: str

    def evaluate(self, positions_yx: np.ndarray, masses_msun: Sequence[float]) -> list[SignalBankResult]: ...
    def describe(self) -> Mapping[str, Any]: ...
    def close(self) -> None: ...


def checked_positions(positions_yx: np.ndarray) -> np.ndarray:
    """Finite, non-empty positions in (y, x) order."""
    positions = np.asarray(positions_yx, dtype=float)
    if positions.ndim != 2 or positions.shape[1] != 2 or not len(positions) or not np.all(np.isfinite(positions)):
        raise ValueError("positions_yx must be a non-empty finite array of shape (n, 2)")
    return positions


class BankAccumulator:
    """Finish dense rows every 256 positions, or reductions at the first batch reaching 256."""

    def __init__(self, context: EngineContext) -> None:
        self.context = context
        self._signals: list[np.ndarray] = []
        self._reductions: list[BankReductions] = []
        self._pending = 0
        self._results: list[SignalBankResult] = []

    def add_signal(self, signal: np.ndarray) -> None:
        if self._reductions:
            raise ValueError("one bank cannot mix signal rows and device reductions")
        values = np.asarray(signal, dtype=float)
        shape = (2, self.context.data_space.pixel_count) if self.context.mismatched else (self.context.data_space.pixel_count,)
        if values.shape != shape:
            raise ValueError(f"signal must have shape {shape}, got {values.shape}")
        self._signals.append(values)
        if len(self._signals) >= HOST_FLUSH_POSITIONS:
            self._flush_signals()

    def add_reductions(self, reductions: BankReductions) -> None:
        if self._signals:
            raise ValueError("one bank cannot mix signal rows and device reductions")
        self._reductions.append(reductions)
        self._pending += reductions.size
        if self._pending >= HOST_FLUSH_POSITIONS:
            self._flush_reductions()

    def _flush_signals(self) -> None:
        if not self._signals:
            return
        data = self.context.data_space
        if self.context.mismatched:
            signal = data.whiten(np.column_stack([pair[0] for pair in self._signals]))
            residual = data.whiten(np.column_stack([pair[1] for pair in self._signals]))
            result = self.context.workspace.evaluate_bank(signal.T, data_whitened=residual.T,
                                                           bias_whitened=self.context.bias_whitened)
        else:
            signal = data.whiten(np.column_stack(self._signals))
            result = self.context.workspace.evaluate_bank(signal.T)
        self._results.append(result)
        self._signals.clear()

    def _flush_reductions(self) -> None:
        if not self._reductions:
            return
        reductions = BankReductions.concatenate(self._reductions)
        self._results.append(self.context.workspace.evaluate_reductions(
            reductions, bias_whitened=self.context.bias_whitened))
        self._reductions.clear()
        self._pending = 0

    def finish(self) -> SignalBankResult:
        self._flush_signals()
        self._flush_reductions()
        if not self._results:
            raise ValueError("a signal bank must contain at least one position")
        return SignalBankResult.concatenate(self._results)


def make_engine(kind: str, context: EngineContext, execution: Any) -> TemplateEngine:
    if kind == "reference":
        from .reference import ReferenceEngine
        return ReferenceEngine(context, workers=execution.reference_workers, progress=execution.progress)
    if kind == "jax":
        from .jax_templates import JaxTemplateEngine
        return JaxTemplateEngine(context, batch_size=execution.batch_size, progress=execution.progress)
    raise ValueError(f"unknown template engine {kind!r}")
