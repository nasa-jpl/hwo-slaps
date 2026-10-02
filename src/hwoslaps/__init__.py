"""Configurable strong-lensing simulations and sensitivity forecasts.

Public entry points load lazily, so importing the package does not initialize
optics, plotting, inference backends, or accelerator runtimes.
"""
from __future__ import annotations

from importlib import import_module
from typing import Any, TYPE_CHECKING

_PUBLIC_API = {
    "run_pipeline": ("pipeline", "run_pipeline"),
    "run_enhanced_pipeline": ("pipeline", "run_enhanced_pipeline"),
    "run_with_artifacts": ("cli", "run_with_artifacts"),
    "sample_population": ("population", "sample_population"),
    "iter_population_configs": ("population", "iter_population_configs"),
}
__all__ = list(_PUBLIC_API)

if TYPE_CHECKING:
    from .cli import run_with_artifacts
    from .pipeline import run_enhanced_pipeline, run_pipeline
    from .population import iter_population_configs, sample_population


def __getattr__(name: str) -> Any:
    if name not in _PUBLIC_API:
        raise AttributeError(name)
    module, member = _PUBLIC_API[name]
    value = getattr(import_module(f".{module}", __name__), member)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(__all__)
