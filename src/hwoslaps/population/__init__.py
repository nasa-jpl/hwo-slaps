"""Public engine values and operations, imported when requested."""
from importlib import import_module

_PUBLIC_API = {
    "PopulationSpec": ("sampling", "PopulationSpec"),
    "PopulationMember": ("sampling", "PopulationMember"),
    "PopulationError": ("distributions", "PopulationError"),
    "sample_population": ("sampling", "sample_population"),
    "iter_population_members": ("sampling", "iter_population_members"),
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
