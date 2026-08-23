"""Analysis-side science forecasts for HWO-SLAPS."""

from __future__ import annotations

from typing import Any, TYPE_CHECKING

_FORECAST_NAMES = (
    "SubhaloForecastData",
    "forecast_ratio",
    "half_mode_mass_from_thermal_kev",
    "load_subhalo_forecast_npz",
    "run_subhalo_forecast",
    "save_subhalo_forecast_npz",
    "sigma_sub_from_f_sub",
    "thermal_kev_from_half_mode_mass",
    "validate_subhalo_forecast_config",
    "wdm_suppression",
)

_RANK_STABILITY_NAMES = (
    "MEMBER_SCALARS",
    "NOISE_STREAM",
    "compare_rankings",
    "curve_comparison",
    "definitions_block",
    "estimator_ratios",
    "load_member",
    "load_pool",
    "member_geometry",
    "noiseless_observables",
    "noisy_observables",
    "rank_measured_pool",
    "replicate_indices",
    "replicate_noise_seed",
    "replicate_stability",
    "run_rank_stability",
    "seed_binding",
    "stability_tier_size",
    "system_index",
)

_SELECTION_NAMES = (
    "APERTURE_THETA_E_MULTIPLE",
    "FLOOR_ARC_SNR",
    "FLOOR_THETA_E_ARCSEC",
    "GOLDEN_TIER_SIZE",
    "RADIAN_TO_ARCSEC",
    "SCORE_VARIANTS",
    "SELECTED_TIER_SIZE",
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

__all__ = sorted(_FORECAST_NAMES + _RANK_STABILITY_NAMES + _SELECTION_NAMES)

if TYPE_CHECKING:
    from .rank_stability import (
        MEMBER_SCALARS,
        NOISE_STREAM,
        compare_rankings,
        curve_comparison,
        definitions_block,
        estimator_ratios,
        load_member,
        load_pool,
        member_geometry,
        noiseless_observables,
        noisy_observables,
        rank_measured_pool,
        replicate_indices,
        replicate_noise_seed,
        replicate_stability,
        run_rank_stability,
        seed_binding,
        stability_tier_size,
        system_index,
    )
    from .selection_score import (
        APERTURE_THETA_E_MULTIPLE,
        FLOOR_ARC_SNR,
        FLOOR_THETA_E_ARCSEC,
        GOLDEN_TIER_SIZE,
        RADIAN_TO_ARCSEC,
        SCORE_VARIANTS,
        SELECTED_TIER_SIZE,
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
    from .subhalo_forecast import (
        SubhaloForecastData,
        forecast_ratio,
        half_mode_mass_from_thermal_kev,
        load_subhalo_forecast_npz,
        run_subhalo_forecast,
        save_subhalo_forecast_npz,
        sigma_sub_from_f_sub,
        thermal_kev_from_half_mode_mass,
        validate_subhalo_forecast_config,
        wdm_suppression,
    )


def __getattr__(name: str) -> Any:
    """Resolve analysis APIs without eager analysis-module imports."""
    if name in _FORECAST_NAMES:
        from . import subhalo_forecast

        return getattr(subhalo_forecast, name)
    if name in _RANK_STABILITY_NAMES:
        from . import rank_stability

        return getattr(rank_stability, name)
    if name in _SELECTION_NAMES:
        from . import selection_score

        return getattr(selection_score, name)
    raise AttributeError(name)


def __dir__() -> list[str]:
    """Return package public names for IDE and star-import compatibility."""
    return sorted(__all__)
