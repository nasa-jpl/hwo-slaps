"""Habitable Worlds Observatory Strong Lensing Analysis Pipeline System."""

from __future__ import annotations

from typing import Any, TYPE_CHECKING

__all__ = ["run_enhanced_pipeline"]

if TYPE_CHECKING:
    from .pipeline import run_enhanced_pipeline


def __getattr__(name: str) -> Any:
    """Resolve the pipeline entry point without eager PyAutoLens imports."""
    if name == "run_enhanced_pipeline":
        from .pipeline import run_enhanced_pipeline

        return run_enhanced_pipeline
    raise AttributeError(name)


def __dir__() -> list[str]:
    """Return package public names for IDE and star-import compatibility."""
    return sorted(__all__)
