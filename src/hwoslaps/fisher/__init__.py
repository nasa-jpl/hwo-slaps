"""Public engine values and operations, imported when requested."""
from importlib import import_module

_PUBLIC_API = {
    "Execution": ("api", "Execution"),
    "PreparedForecast": ("api", "PreparedForecast"),
    "prepare_forecast": ("api", "prepare_forecast"),
    "forecast": ("api", "forecast"),
    "ForecastResult": ("result", "ForecastResult"),
    "PositionSet": ("positions", "PositionSet"),
    "PsfPair": ("psf_pair", "PsfPair"),
    "NuisanceDesign": ("nuisances", "NuisanceDesign"),
    "ProfileLikelihoodWorkspace": ("statistics", "ProfileLikelihoodWorkspace"),
}
__all__ = list(_PUBLIC_API)


def __getattr__(name):
    if name not in _PUBLIC_API:
        raise AttributeError(name)
    module, member = _PUBLIC_API[name]
    value = getattr(import_module(f".{module}", __name__), member)
    globals()[name] = value
    return value


def __dir__():
    return sorted(__all__)
