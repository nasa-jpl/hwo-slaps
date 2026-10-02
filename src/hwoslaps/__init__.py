"""Reusable strong-lensing observations and subhalo sensitivity forecasts."""
from __future__ import annotations
from importlib import import_module
from typing import Any

_PUBLIC_API = {
    "PreparedForecast": ("forecasting", "PreparedForecast"),
    "prepare_forecast": ("forecasting", "prepare_forecast"),
    "forecast": ("forecasting", "forecast"),
    "simulate": ("forecasting", "simulate"),
    "ForecastResult": ("modeling.forecast_results", "ForecastResult"),
    "summarize_forecast": ("modeling.forecast_results", "summarize_forecast"),
    "mass_reach": ("modeling.mass_reach", "mass_reach"),
    "adaptive_mass_reach": ("modeling.mass_reach", "adaptive_mass_reach"),
    "validate_nonlinear": ("modeling.nonlinear.api", "validate_nonlinear"),
    "sample_population": ("population", "sample_population"),
    "iter_population_configs": ("population", "iter_population_configs"),
    "save_forecast_result": ("forecast_artifacts", "save_forecast_result"),
    "load_forecast_result": ("forecast_artifacts", "load_forecast_result"),
}
__all__ = list(_PUBLIC_API)


def __getattr__(name: str) -> Any:
    if name not in _PUBLIC_API:
        raise AttributeError(name)
    module, member = _PUBLIC_API[name]
    value = getattr(import_module(f".{module}", __name__), member)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(__all__)
