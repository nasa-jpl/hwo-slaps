"""AutoLens node rendering in serial or a supervised, ordered spawned pool."""

from __future__ import annotations

import multiprocessing
import os
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from dataclasses import dataclass
from functools import cached_property
from typing import Any, Callable, Iterable, Iterator, Mapping, Sequence, TypeVar

import numpy as np
from threadpoolctl import threadpool_limits

from ...optics.kernels import DetectorPSF, KernelBinding
from ...scene.cosmology import Cosmology
from ...scene.halos import HaloModel, make_halo
from ..renderer import SceneRenderer
from ..statistics import SignalBankResult
from .base import BankAccumulator, EngineContext, checked_positions

T = TypeVar("T")
_WORKER_PAYLOAD = None
_WORKER_LIMITER = None


@dataclass(frozen=True)
class KernelPayload:
    """Kernel bytes and group indices, without cached backend objects or mapping proxies."""

    values: tuple[tuple[np.ndarray, float], ...]
    group_index: Mapping[str, int]

    @classmethod
    def from_binding(cls, binding: KernelBinding) -> KernelPayload:
        return cls(tuple((kernel.kernel, kernel.pixel_scale_arcsec) for kernel in binding.kernels),
                   dict(binding.group_index))

    def binding(self) -> KernelBinding:
        kernels = tuple(DetectorPSF.from_array(values, scale, normalize=False) for values, scale in self.values)
        return KernelBinding(kernels, self.group_index)


@dataclass(frozen=True)
class NodePayload:
    """The picklable, mass-independent inputs needed by a reference node."""

    renderer: SceneRenderer
    hypothesis_model: HaloModel
    hypothesis_redshift: float
    source_redshift: float
    cosmology: Cosmology
    model_kernels: KernelPayload
    truth_kernels: KernelPayload | None
    mean_model_adu: np.ndarray
    mask: np.ndarray

    @cached_property
    def model_binding(self) -> KernelBinding:
        return self.model_kernels.binding()

    @cached_property
    def truth_binding(self) -> KernelBinding | None:
        return None if self.truth_kernels is None else self.truth_kernels.binding()


def node_signal(payload: NodePayload, mass_msun: float, position_yx: tuple[float, float]) -> np.ndarray:
    halo = make_halo(payload.hypothesis_model, mass_msun, position_yx, redshift=payload.hypothesis_redshift,
                     source_redshift=payload.source_redshift, cosmology=payload.cosmology)
    scene = payload.renderer.scene(subhalo=halo)
    model = payload.renderer.mean_adu(scene, payload.model_binding)
    signal = (model - payload.mean_model_adu)[payload.mask]
    if payload.truth_binding is None:
        return signal
    truth = payload.renderer.mean_adu(scene, payload.truth_binding)
    return np.stack((signal, (truth - payload.mean_model_adu)[payload.mask]))


def ordered_process_map(function: Callable[[Any], T], items: Iterable[Any], *, workers: int,
                        initializer: Callable[..., None], initargs: tuple) -> Iterator[T]:
    """Yield every input in order, keeping at most twice the worker count pending.

    On a failed task or a closed iterator, terminate and join every worker before returning.
    """
    executor = ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn"),
                                   initializer=initializer, initargs=initargs)
    pending, buffered = {}, {}
    items_iter = iter(items)
    next_submit = next_yield = 0
    exhausted = False
    completed = False
    try:
        while pending or not exhausted:
            while len(pending) + len(buffered) < 2 * workers and not exhausted:
                try:
                    item = next(items_iter)
                except StopIteration:
                    exhausted = True
                    break
                pending[executor.submit(function, item)] = next_submit
                next_submit += 1
            if not pending:
                break
            done, _ = wait(pending, return_when=FIRST_COMPLETED)
            for future in done:
                buffered[pending.pop(future)] = future.result()
            while next_yield in buffered:
                yield buffered.pop(next_yield)
                next_yield += 1
        completed = True
    finally:
        if not completed:
            for future in pending:
                future.cancel()
            processes = tuple((executor._processes or {}).values())
            for process in processes:
                if process.is_alive():
                    process.terminate()
            for process in processes:
                process.join(timeout=5)
                if process.is_alive():
                    process.kill()
                    process.join()
        executor.shutdown(wait=True, cancel_futures=not completed)


def _initialize_worker(payload: NodePayload) -> None:
    global _WORKER_PAYLOAD, _WORKER_LIMITER
    os.environ["JAX_PLATFORMS"] = "cpu"
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[name] = "1"
    _WORKER_LIMITER = threadpool_limits(limits=1)
    _WORKER_PAYLOAD = payload


def _worker_node(task):
    mass_index, mass, position = task
    return mass_index, node_signal(_WORKER_PAYLOAD, mass, position)


class ReferenceEngine:
    kind = "reference"

    def __init__(self, context: EngineContext, *, workers: int, progress: bool) -> None:
        if isinstance(workers, bool) or not isinstance(workers, (int, np.integer)) or workers < 1:
            raise ValueError("workers must be a positive integer")
        self.context, self.workers, self.progress = context, int(workers), progress
        self._closed = False
        self.payload = NodePayload(context.renderer, context.hypothesis_model, context.hypothesis_redshift,
                                   context.source_redshift, context.cosmology,
                                   KernelPayload.from_binding(context.model_kernels),
                                   None if context.truth_kernels is None else KernelPayload.from_binding(context.truth_kernels),
                                   context.mean_model_adu, context.data_space.mask)

    def evaluate(self, positions_yx: np.ndarray, masses_msun: Sequence[float]) -> list[SignalBankResult]:
        if self._closed:
            raise RuntimeError("reference engine is closed")
        positions = checked_positions(positions_yx)
        masses = tuple(masses_msun)
        if not masses:
            raise ValueError("masses_msun must be non-empty")
        for mass in masses:
            self.context.hypothesis(mass, tuple(positions[0]))
        accumulators = [BankAccumulator(self.context) for _ in masses]
        tasks = ((index, mass, tuple(position)) for index, mass in enumerate(masses) for position in positions)
        if self.workers == 1:
            rows = ((index, node_signal(self.payload, mass, position)) for index, mass, position in tasks)
        else:
            rows = ordered_process_map(_worker_node, tasks, workers=self.workers,
                                       initializer=_initialize_worker, initargs=(self.payload,))
        ordered_rows = rows
        if self.progress:
            from tqdm.auto import tqdm
            rows = tqdm(rows, total=len(masses) * len(positions), desc="Fisher templates")
        try:
            for index, signal in rows:
                accumulators[index].add_signal(signal)
        finally:
            rows.close()
            if ordered_rows is not rows:
                ordered_rows.close()
        return [bank.finish() for bank in accumulators]

    def describe(self) -> Mapping[str, Any]:
        return {"kind": self.kind, "workers": self.workers,
                "start_method": "spawn" if self.workers > 1 else "serial"}

    def close(self) -> None:
        self._closed = True
