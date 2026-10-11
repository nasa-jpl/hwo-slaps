"""Public engine values and operations, imported when requested."""
from importlib import import_module

_PUBLIC_API = {
    "prepare_case": ("api", "prepare_case"),
    "PreparedCase": ("api", "PreparedCase"),
    "validate_nonlinear": ("api", "validate_nonlinear"),
    "BackendSession": ("backend", "BackendSession"),
    "FitSpec": ("settings", "FitSpec"),
    "PixelMask": ("settings", "PixelMask"),
    "PriorWidths": ("settings", "PriorWidths"),
    "BoxRule": ("settings", "BoxRule"),
    "MassSupport": ("settings", "MassSupport"),
    "SamplerSettings": ("settings", "SamplerSettings"),
    "RefineSettings": ("settings", "RefineSettings"),
    "CaseResult": ("result", "CaseResult"),
    "RoleFit": ("result", "RoleFit"),
    "RoleStatus": ("result", "RoleStatus"),
    "RefineOutcome": ("result", "RefineOutcome"),
    "SamplerRecord": ("result", "SamplerRecord"),
    "RetentionInventory": ("result", "RetentionInventory"),
    "SubhaloRecovery": ("recovery", "SubhaloRecovery"),
    "SubhaloEstimate": ("recovery", "SubhaloEstimate"),
    "ForecastReference": ("result", "ForecastReference"),
    "ObservationRecord": ("result", "ObservationRecord"),
    "OBJECTIVE_VERSION": ("settings", "OBJECTIVE_VERSION"),
    "PROCEDURE_VERSION": ("settings", "PROCEDURE_VERSION"),
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
