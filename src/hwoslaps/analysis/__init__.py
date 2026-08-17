"""Analysis-side science forecasts for HWO-SLAPS."""

from __future__ import annotations

from typing import Any, TYPE_CHECKING

__all__ = [
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
]

if TYPE_CHECKING:
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
    """Resolve forecast APIs without eager analysis-module imports."""
    if name in __all__:
        from . import subhalo_forecast

        return getattr(subhalo_forecast, name)
    raise AttributeError(name)


def __dir__() -> list[str]:
    """Return package public names for IDE and star-import compatibility."""
    return sorted(__all__)
