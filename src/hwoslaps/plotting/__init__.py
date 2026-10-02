"""Optional visualization consumers; graphics dependencies load on demand."""

from importlib import import_module

_EXPORTS = {
    'plot_lensing_comparison': '.lensing_plots',
    'plot_lensing_baseline_scene': '.lensing_plots',
    'plot_psf_comparison': '.psf_plots',
    'plot_psf_zoom': '.psf_plots',
    'plot_psf_system_overview': '.psf_plots',
    'plot_psf_complete_analysis': '.psf_plots',
    'plot_observation_comparison': '.observation_plots',
    'plot_fisher_local_summary': '.detection_plots',
    'plot_fisher_psf_mode_scan': '.detection_plots',
    'plot_fisher_detection_map_summary': '.detection_plots',
    'plot_fisher_map_degradation': '.detection_plots',
    'generate_all_plots': '.registry',
    'get_plot_registry': '.registry',
}
__all__ = list(_EXPORTS)


def __getattr__(name):
    module_name = _EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(name)
    return getattr(import_module(module_name, __name__), name)


def __dir__():
    return sorted(__all__)
