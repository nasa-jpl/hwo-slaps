"""Execution helpers for Fisher template banks.

Worker policy is independent of the detector and accepts the process-level
worker override explicitly. Numerical configuration is never rewritten to
select a runtime worker count. The process map retains spawn and failure
supervision semantics.
"""

from __future__ import annotations

from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
import multiprocessing
from typing import Any, Dict


def supervised_ordered_map(
    func,
    items,
    num_workers,
    initializer=None,
    initargs=(),
):
    """Yield ordered task results while supervising worker health."""
    context = multiprocessing.get_context("spawn")
    executor = ProcessPoolExecutor(
        max_workers=num_workers,
        mp_context=context,
        initializer=initializer,
        initargs=initargs,
    )
    pending = {}
    buffered = {}
    items_iter = iter(items)
    next_submit = 0
    next_yield = 0
    max_pending = max(1, num_workers * 2)
    try:
        while pending or next_submit == 0:
            while len(pending) < max_pending:
                try:
                    item = next(items_iter)
                except StopIteration:
                    break
                pending[executor.submit(func, item)] = next_submit
                next_submit += 1
            if not pending:
                break
            done, _ = wait(pending, return_when=FIRST_COMPLETED)
            for future in done:
                index = pending.pop(future)
                buffered[index] = future.result()
            while next_yield in buffered:
                yield buffered.pop(next_yield)
                next_yield += 1
        executor.shutdown(wait=True)
    except BaseException:
        executor.shutdown(wait=False, cancel_futures=True)
        raise


def grid_num_workers(map_config: Dict[str, Any], worker_override: str = "") -> int:
    """Resolve reference workers without changing hashed map configuration.

    JAX batching is controlled by its engine rather than the reference process
    pool, so the process-level override does not apply to that backend.
    """
    engine = str(map_config.get("engine", "reference")).lower()
    configured = int(map_config.get("num_workers", 1))
    if engine == "jax":
        return configured
    raw = worker_override.strip()
    if not raw:
        return configured
    workers = int(raw)
    if workers <= 0:
        raise ValueError("HWOSLAPS_FISHER_GRID_WORKERS must be a positive integer")
    return workers


def grid_runtime_provenance(
    map_config: Dict[str, Any], worker_override: str = ""
) -> Dict[str, Any]:
    """Describe requested and effective execution for result provenance."""
    requested = int(map_config.get("num_workers", 1))
    engine = str(map_config.get("engine", "reference")).lower()
    if engine == "jax":
        effective = 1
        start_method = "jax"
    else:
        effective = grid_num_workers(map_config, worker_override)
        start_method = "spawn" if effective > 1 else "serial"
    return {
        "fisher_grid_workers_requested": requested,
        "fisher_grid_workers_effective": effective,
        "fisher_grid_start_method": start_method,
    }
