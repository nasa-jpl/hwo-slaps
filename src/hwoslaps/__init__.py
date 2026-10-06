"""Public engine values and operations, imported when requested."""
from importlib import import_module

from ._version import __version__

_PUBLIC_API = {
    "ConfigError": ("config.checks", "ConfigError"),
    "EngineConfig": ("config.schema", "EngineConfig"),
    "load_config": ("config.schema", "load_config"),
    "simulate": ("simulation", "simulate"),
    "Observation": ("observation.observation", "Observation"),
    "Halo": ("scene.halos", "Halo"),
    "prepare_forecast": ("fisher.api", "prepare_forecast"),
    "forecast": ("fisher.api", "forecast"),
    "PreparedForecast": ("fisher.api", "PreparedForecast"),
    "Execution": ("fisher.api", "Execution"),
    "ForecastResult": ("fisher.result", "ForecastResult"),
    "summarize": ("analysis.reductions", "summarize"),
    "mass_reach": ("analysis.reach", "mass_reach"),
    "prepare_case": ("inference.api", "prepare_case"),
    "validate_nonlinear": ("inference.api", "validate_nonlinear"),
    "FitSpec": ("inference.settings", "FitSpec"),
    "SamplerSettings": ("inference.settings", "SamplerSettings"),
    "RefineSettings": ("inference.settings", "RefineSettings"),
    "CaseResult": ("inference.result", "CaseResult"),
    "load_batch_spec": ("batch.spec", "load_batch_spec"),
    "run_batch": ("batch.runner", "run_batch"),
    "save_forecast": ("artifacts", "save_forecast"),
    "load_forecast": ("artifacts", "load_forecast"),
    "save_observation": ("artifacts", "save_observation"),
    "load_observation": ("artifacts", "load_observation"),
    "save_case": ("artifacts", "save_case"),
    "load_case": ("artifacts", "load_case"),
}
__all__ = ["__version__", *list(_PUBLIC_API)]


def __getattr__(name):
    if name not in _PUBLIC_API:
        raise AttributeError(name)
    module, member = _PUBLIC_API[name]
    value = getattr(import_module(f".{module}", __name__), member)
    globals()[name] = value
    return value


def __dir__():
    return sorted(__all__)
