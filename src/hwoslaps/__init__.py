"""Habitable Worlds Observatory Strong Lensing Analysis Pipeline System."""

from typing import Any

__all__ = ["run_enhanced_pipeline"]


def __getattr__(name: str) -> Any:
    """Resolve the pipeline entry point without eager PyAutoLens imports."""
    if name == "run_enhanced_pipeline":
        from .pipeline import run_enhanced_pipeline

        return run_enhanced_pipeline
    raise AttributeError(name)
