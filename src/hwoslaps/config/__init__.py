"""Public engine values and operations, imported when requested."""
from importlib import import_module

_PUBLIC_API = {
    "ConfigError": ("checks", "ConfigError"),
    "EngineConfig": ("schema", "EngineConfig"),
    "compose_config": ("schema", "compose_config"),
    "parse_config": ("schema", "parse_config"),
    "load_config": ("schema", "load_config"),
    "resolve_config": ("schema", "resolve_config"),
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
