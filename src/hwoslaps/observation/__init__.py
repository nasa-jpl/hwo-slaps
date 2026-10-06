"""Public engine values and operations, imported when requested."""
from importlib import import_module

_PUBLIC_API = {
    "Observation": ("observation", "Observation"),
    "observe": ("observation", "observe"),
    "Exposure": ("expected", "Exposure"),
    "convolve_light": ("expected", "convolve_light"),
    "resolve_observing": ("normalization", "resolve_observing"),
    "PhotometryRecord": ("normalization", "PhotometryRecord"),
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
