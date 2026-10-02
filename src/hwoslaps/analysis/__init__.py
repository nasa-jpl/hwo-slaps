"""Analysis-side science forecasts for HWO-SLAPS."""

from __future__ import annotations

from typing import Any, TYPE_CHECKING

_SELECTION_NAMES = (
    "RADIAN_TO_ARCSEC",
    "SelectionResult",
    "aperture_mask",
    "apply_floor_cuts",
    "arc_snr",
    "blank_variance_e2",
    "complexity",
    "diffraction_scale_arcsec",
    "expected_variance_e2",
    "gradient_power",
    "oracle_recovered_fraction",
    "rank_by_score",
    "rank_by_sensitivity",
    "rank_pool",
    "ranking_positions",
    "selection_scores",
    "spearman_rank_correlation",
    "standardize",
    "top_k_jaccard",
)

__all__ = sorted(_SELECTION_NAMES)

if TYPE_CHECKING:
    from .selection_score import (
        RADIAN_TO_ARCSEC,
        SelectionResult,
        aperture_mask,
        apply_floor_cuts,
        arc_snr,
        blank_variance_e2,
        complexity,
        diffraction_scale_arcsec,
        expected_variance_e2,
        gradient_power,
        oracle_recovered_fraction,
        rank_by_score,
        rank_by_sensitivity,
        rank_pool,
        ranking_positions,
        selection_scores,
        spearman_rank_correlation,
        standardize,
        top_k_jaccard,
    )


def __getattr__(name: str) -> Any:
    """Resolve analysis APIs without eager analysis-module imports."""
    if name in _SELECTION_NAMES:
        from . import selection_score

        return getattr(selection_score, name)
    raise AttributeError(name)


def __dir__() -> list[str]:
    """Return package public names for IDE and star-import compatibility."""
    return sorted(__all__)
