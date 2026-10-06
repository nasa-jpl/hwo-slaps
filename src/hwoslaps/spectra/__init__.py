"""Public engine values and operations, imported when requested."""
from importlib import import_module

_PUBLIC_API = {
    "Bandpass": ("bandpass", "Bandpass"),
    "build_bandpass": ("bandpass", "build_bandpass"),
    "SED": ("sed", "SED"),
    "build_sed": ("sed", "build_sed"),
    "integrate_dlnlambda": ("bandpass", "integrate_dlnlambda"),
    "ab_to_fnu_jy": ("photometry", "ab_to_fnu_jy"),
    "fnu_jy_to_ab": ("photometry", "fnu_jy_to_ab"),
    "rate_from_ab": ("photometry", "rate_from_ab"),
    "sky_rate_e_per_s_per_pixel": ("photometry", "sky_rate_e_per_s_per_pixel"),
    "synthetic_ab_mag": ("photometry", "synthetic_ab_mag"),
    "effective_wavelength_m": ("photometry", "effective_wavelength_m"),
    "band_mean_throughput": ("photometry", "band_mean_throughput"),
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
