"""Public engine values and operations, imported when requested."""
from importlib import import_module

_PUBLIC_API = {
    "DetectorPSF": ("kernels", "DetectorPSF"),
    "KernelBinding": ("kernels", "KernelBinding"),
    "PSFProvider": ("providers", "PSFProvider"),
    "OpticalPSF": ("optical_psf", "OpticalPSF"),
    "KernelPSF": ("providers", "KernelPSF"),
    "ModelPSF": ("providers", "ModelPSF"),
    "build_psf_provider": ("providers", "build_psf_provider"),
    "build_model_psf": ("providers", "build_model_psf"),
    "Pupil": ("pupils", "Pupil"),
    "build_pupil": ("pupils", "build_pupil"),
    "WavefrontMode": ("wavefront", "WavefrontMode"),
    "WavefrontCoefficients": ("wavefront", "WavefrontCoefficients"),
    "WavefrontBasis": ("wavefront", "WavefrontBasis"),
    "draw_wavefront": ("knowledge_error", "draw_wavefront"),
    "draw_knowledge_error": ("knowledge_error", "draw_knowledge_error"),
    "strehl_ratio": ("metrics", "strehl_ratio"),
    "encircled_energy": ("metrics", "encircled_energy"),
    "fwhm_arcsec": ("metrics", "fwhm_arcsec"),
    "captured_power_fraction": ("metrics", "captured_power_fraction"),
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
