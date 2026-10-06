"""Public engine values and operations, imported when requested."""
from importlib import import_module

_PUBLIC_API = {
    "SceneSpec": ("spec", "SceneSpec"),
    "GridSpec": ("spec", "GridSpec"),
    "LightGroup": ("spec", "LightGroup"),
    "Scene": ("builder", "Scene"),
    "build_scene": ("builder", "build_scene"),
    "Halo": ("halos", "Halo"),
    "HaloModel": ("halos", "HaloModel"),
    "make_halo": ("halos", "make_halo"),
    "Cosmology": ("cosmology", "Cosmology"),
    "CosmologySpec": ("cosmology", "CosmologySpec"),
    "LensingGeometry": ("cosmology", "LensingGeometry"),
    "configured_injection": ("subhalo", "configured_injection"),
    "realize_perturbers": ("perturbers", "realize_perturbers"),
    "PROFILE_TYPES": ("profiles", "PROFILE_TYPES"),
    "ParameterKind": ("profiles", "ParameterKind"),
    "scene_parameters": ("parameters", "scene_parameters"),
    "with_parameter": ("parameters", "with_parameter"),
    "ImageAsset": ("image_source", "ImageAsset"),
    "load_image_asset": ("image_source", "load_image_asset"),
    "prepare_image_asset": ("image_source", "prepare_image_asset"),
    "effective_einstein_radius": ("critical_curve", "effective_einstein_radius"),
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
