"""H0/H1 fit models from the scene registry, preserving the truth scene's plane and prior order."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from ..config.checks import ConfigError
from ..scene.halos import Halo
from ..scene.parameters import scene_parameters
from ..scene.profiles import ELLIPTICITY_LIMIT, PROFILE_TYPES, Fixed, Link, ParameterRef
from .fit_model import FitArgument, FitComponent, FitGalaxy, FitModel, fixed, linked, uniform
from .settings import FitSpec

if TYPE_CHECKING:
    from ..scene.builder import Scene
    from ..scene.spec import ComponentSpec
    from .subhalo_classes import SubhaloMassMapping

__all__ = ["RoleModels", "build_role_models"]


@dataclass(frozen=True)
class RoleModels:
    smooth: FitModel
    subhalo: FitModel
    mass_mapping: SubhaloMassMapping | None


def _joint_ellipticity(component: ComponentSpec, parameters: dict[str, Any], free: set[str], fit: FitSpec) -> None:
    definitions = [parameter for parameter in parameters.values() if parameter.definition.kind == "ellipticity"]
    varied = [parameter.name in free for parameter in definitions]
    if not definitions or not any(varied):
        return
    corner = []
    for parameter, varies in zip(definitions, varied, strict=True):
        if varies:
            bounds = fit.prior_widths.rule(component.plane, "ellipticity").box(parameter.value,
                                                                              parameter.definition.domain)
            corner.append(max(abs(value) for value in bounds))
        else:
            corner.append(abs(parameter.value))
    if math.hypot(*corner) >= ELLIPTICITY_LIMIT:
        count = sum(varied)
        linear = sum(abs(parameter.value) for parameter, varies in zip(definitions, varied, strict=True) if varies)
        constant = sum(parameter.value ** 2 for parameter in definitions) - ELLIPTICITY_LIMIT ** 2
        maximum = (-linear + math.sqrt(linear ** 2 - count * constant)) / count
        path = f"scene.{component.plane}.{component.role}.{component.name}.ell_comps"
        raise ConfigError(path, f"fit box corner {corner} has hypot >= {ELLIPTICITY_LIMIT}; largest symmetric "
                                f"ellipticity half width is {maximum:.3g} (strictly below)")


def _ellipticity_box_reaches_origin(component: ComponentSpec, parameters: dict[str, Any], free: set[str],
                                    fit: FitSpec) -> bool:
    pair = [parameter for parameter in parameters.values() if parameter.definition.kind == "ellipticity"]
    if not any(parameter.name in free for parameter in pair):
        return False
    for parameter in pair:
        if parameter.name in free:
            lower, upper = fit.prior_widths.rule(component.plane, "ellipticity").box(
                parameter.value, parameter.definition.domain)
            if not lower <= 0.0 <= upper:
                return False
        elif parameter.value != 0.0:
            return False
    return len(pair) == 2


def _component_models(scene: Scene, component: ComponentSpec, free: set[str], fit: FitSpec,
                      loaded: dict[tuple[str, str], Any], use_jax: bool) -> list[tuple[str, FitComponent]]:
    prefix = f"{component.plane}.{component.role}.{component.name}."
    parameters = {parameter.definition.name: parameter for parameter in scene_parameters(scene.spec)
                  if parameter.name.startswith(prefix)}
    _joint_ellipticity(component, parameters, free, fit)
    profile_type = PROFILE_TYPES[component.type]
    profiles = []
    for layout in profile_type.layout(component.values):
        ranks = {argument.name: min((list(parameters).index(element.parameter)
                                    for element in argument.elements if isinstance(element, ParameterRef)),
                                   default=len(parameters)) for argument in layout.arguments}
        arguments = []
        for argument in layout.arguments:
            elements = []
            for element in argument.elements:
                if isinstance(element, ParameterRef):
                    parameter = parameters[element.parameter]
                    if parameter.name in free:
                        bounds = fit.prior_widths.rule(component.plane, parameter.definition.kind).box(
                            parameter.value, parameter.definition.domain)
                        elements.append(uniform(*bounds, truth=parameter.value))
                    else:
                        elements.append(fixed(parameter.value))
                elif isinstance(element, Fixed):
                    elements.append(fixed(element.value))
                elif isinstance(element, Link):
                    elements.append(linked(component.name + element.suffix, element.argument, element.element))
                else:
                    raise TypeError(f"unsupported registry element {element!r}")
            arguments.append(FitArgument(argument.name, tuple(elements), pair=len(elements) == 2))
        if layout.profile_class == "ImageLightProfile":
            profile = loaded[(component.plane, component.name + layout.suffix)]
            arguments = [argument for argument in arguments if argument.name != "asset_path"]
            arguments.extend((FitArgument("pixel_scale_arcsec", (fixed(profile.pixel_scale_arcsec),), pair=False),
                              FitArgument("sb", (fixed(profile.sb),), pair=False)))
            path = "hwoslaps.scene.image_profile:ImageLightProfile"
        elif use_jax and component.type == "Exponential" and _ellipticity_box_reaches_origin(component, parameters, free, fit):
            path = "hwoslaps.inference.light_profiles:Exponential"
        else:
            path = f"autolens:{'mp' if component.role == 'mass' else 'lp'}.{layout.profile_class}"
        arguments.sort(key=lambda argument: ranks.get(argument.name, len(parameters)))
        profiles.append((component.name + layout.suffix, FitComponent(path, tuple(arguments))))
    return profiles


def _fixed_halo(halo: Halo) -> FitComponent:
    arguments = [FitArgument("centre", tuple(fixed(value) for value in halo.position_yx_arcsec), pair=True)]
    arguments.extend(FitArgument(name, (fixed(value),), pair=False) for name, value in halo.lensing().parameters.items())
    return FitComponent(f"autolens:mp.{halo.model.profile_class}", tuple(arguments))


def _subhalo(hypothesis: Halo, fit: FitSpec, mapping: SubhaloMassMapping | None) -> FitComponent:
    if fit.mode == "fixed_template":
        return _fixed_halo(hypothesis)
    width = (fit.prior_widths.subhalo_local_window_arcsec if fit.mode == "local_search"
             else fit.prior_widths.subhalo_freed_window_arcsec)
    centre = FitArgument("centre", tuple(uniform(value - width, value + width, truth=value)
                                         for value in hypothesis.position_yx_arcsec), pair=True)
    if fit.mode == "local_search":
        fixed_profile = _fixed_halo(hypothesis)
        return FitComponent(fixed_profile.profile_class, (centre, *fixed_profile.arguments[1:]))
    from .subhalo_classes import freed_profile_class

    profile_class = freed_profile_class(hypothesis.model.type)
    return FitComponent(f"{profile_class.__module__}:{profile_class.__qualname__}", (
        centre, FitArgument("log10_m200", (uniform(fit.mass_support.log10_mass_min,
                                                   fit.mass_support.log10_mass_max,
                                                   truth=math.log10(hypothesis.mass_msun)),), pair=False),
        FitArgument("mass_mapping", (fixed(mapping),), pair=False)))


def build_role_models(scene: Scene, hypothesis: Halo, free_parameters: Sequence[str], fit: FitSpec,
                      *, use_jax: bool) -> RoleModels:
    """Build both roles with exactly the forecast's scene nuisance parameters free."""
    free = set(free_parameters)
    available = {parameter.name for parameter in scene_parameters(scene.spec)}
    if not free <= available:
        raise ValueError(f"free parameters absent from the scene: {sorted(free - available)}")
    if hypothesis.cosmology != scene.cosmology or hypothesis.source_redshift != scene.spec.source.redshift:
        raise ValueError("the hypothesis cosmology or source redshift differs from the scene")
    mapping = None
    if fit.mode == "freed":
        if not fit.mass_support.contains(math.log10(hypothesis.mass_msun)):
            raise ValueError(f"trial mass {hypothesis.mass_msun} lies outside the freed mass support")
        from .subhalo_classes import mass_mapping

        mapping = mass_mapping(hypothesis, scene.cosmology, fit.mass_support)
    loaded = {}
    for galaxy in (scene.spec.lens, scene.spec.source):
        profiles = iter(scene.light_profiles.get(galaxy.plane, ()))
        for component in galaxy.light:
            for layout in PROFILE_TYPES[component.type].layout(component.values):
                loaded[(galaxy.plane, component.name + layout.suffix)] = next(profiles)
    lens = [profile for component in scene.spec.lens.mass + scene.spec.lens.light
            for profile in _component_models(scene, component, free, fit, loaded, use_jax)]
    source = [profile for component in scene.spec.source.light
              for profile in _component_models(scene, component, free, fit, loaded, use_jax)]
    lens_redshift = scene.spec.lens.redshift
    off_plane = sorted({halo.redshift for halo in scene.perturbers} - {lens_redshift})
    perturbers = {redshift: [] for redshift in off_plane}
    for index, halo in enumerate(scene.perturbers):
        holder = lens if halo.redshift == lens_redshift else perturbers[halo.redshift]
        holder.append((f"perturber_{index}", _fixed_halo(halo)))
    halo_profile = _subhalo(hypothesis, fit, mapping)
    middle = tuple(FitGalaxy(f"perturbers_{index}", redshift, tuple(perturbers[redshift]))
                   for index, redshift in enumerate(off_plane))
    tail = FitGalaxy("source", scene.spec.source.redshift, tuple(source))
    smooth = FitModel("smooth", (FitGalaxy("lens", lens_redshift, tuple(lens)), *middle, tail))
    if hypothesis.redshift == lens_redshift:
        galaxies = (FitGalaxy("lens", lens_redshift, (*lens, ("subhalo", halo_profile))), *middle, tail)
    else:
        galaxies = (FitGalaxy("lens", lens_redshift, tuple(lens)), *middle,
                    FitGalaxy("subhalo_plane", hypothesis.redshift, (("subhalo", halo_profile),)), tail)
    return RoleModels(smooth, FitModel("subhalo", galaxies), mapping)
