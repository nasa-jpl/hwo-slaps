"""Optional plots of current engine products; graphics imports occur on calls."""

from importlib import import_module

_EXPORTS = {
    "plot_statistic_map": ".forecast",
    "plot_detection_map": ".forecast",
    "plot_mass_curve": ".forecast",
    "plot_knowledge_error": ".forecast",
    "plot_kernel": ".optics",
    "plot_pupil": ".optics",
    "plot_observation": ".observation",
}
__all__ = list(_EXPORTS)


def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(name)
    return getattr(import_module(_EXPORTS[name], __name__), name)


def __dir__():
    return sorted(__all__)
