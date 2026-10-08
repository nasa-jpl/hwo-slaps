"""Public engine values and operations, imported when requested."""
from importlib import import_module

_PUBLIC_API = {
    "BatchSpec": ("spec", "BatchSpec"),
    "BatchExecution": ("spec", "BatchExecution"),
    "BatchReport": ("runner", "BatchReport"),
    "BatchResults": ("results", "BatchResults"),
    "load_batch_spec": ("spec", "load_batch_spec"),
    "parse_batch": ("spec", "parse_batch"),
    "plan_batch": ("jobs", "plan_batch"),
    "run_batch": ("runner", "run_batch"),
    "open_batch": ("results", "open_batch"),
    "BatchError": ("state", "BatchError"),
    "BatchConflict": ("state", "BatchConflict"),
    "BatchLocked": ("state", "BatchLocked"),
    "BatchIncomplete": ("state", "BatchIncomplete"),
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
