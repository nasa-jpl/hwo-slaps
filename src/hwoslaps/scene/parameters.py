"""Scene parameters: the named scalar parameters of the lens and source components.

Names are ``<galaxy>.<mass|light>.<component>.<parameter>``, for example
``lens.mass.main.einstein_radius`` or ``source.light.disk.intensity``. The order is lens
mass, lens light, then source light, components in declaration order and parameters in
registry order: the forecast's nuisance column order and the fit's free-parameter order.
Perturbers and the subhalo are not scene parameters.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Sequence
from dataclasses import dataclass
from fnmatch import fnmatchcase

from ..config.checks import ConfigError
from .profiles import PROFILE_TYPES, ParameterDef
from .spec import LIGHT_COMPONENT_TABLE, MASS_COMPONENT_TABLE, ComponentSpec, GalaxySpec, SceneSpec, component_from_values

__all__ = ["SceneParameter", "match_parameters", "scene_parameters", "with_parameter"]


@dataclass(frozen=True)
class SceneParameter:
    name: str
    value: float
    definition: ParameterDef


def _components(spec: SceneSpec) -> list[tuple[GalaxySpec, ComponentSpec]]:
    return ([(spec.lens, component) for component in spec.lens.mass + spec.lens.light]
            + [(spec.source, component) for component in spec.source.light])


def _name(component: ComponentSpec, definition: ParameterDef) -> str:
    return f"{component.plane}.{component.role}.{component.name}.{definition.name}"


def scene_parameters(spec: SceneSpec) -> tuple[SceneParameter, ...]:
    """Every scene parameter with its value, in nuisance order."""
    parameters = []
    for _, component in _components(spec):
        for definition in PROFILE_TYPES[component.type].parameters(component.values):
            value = component.values[definition.key]
            parameters.append(SceneParameter(_name(component, definition),
                                             float(value if definition.index is None else value[definition.index]),
                                             definition))
    return tuple(parameters)


def with_parameter(spec: SceneSpec, name: str, value: float) -> SceneSpec:
    """A copy of ``spec`` with one parameter replaced.

    The value is checked against the parameter's own domain, then the joint rules of its
    component's table run on the replaced values. Nothing else is read again, so no file is
    touched: a finite-difference step inside a preparation works on its loaded assets. A
    refused value raises ``ConfigError`` naming the parameter and the value.
    """
    for galaxy, component in _components(spec):
        for definition in PROFILE_TYPES[component.type].parameters(component.values):
            if _name(component, definition) == name:
                return _replaced(spec, galaxy, component, definition, value)
    raise KeyError(f"no scene parameter {name!r}; parameters: {', '.join(p.name for p in scene_parameters(spec))}")


def _replaced(spec: SceneSpec, galaxy: GalaxySpec, component: ComponentSpec, definition: ParameterDef,
              value: float) -> SceneSpec:
    label = f"{_name(component, definition)} = {value!r}"
    path = f"scene.{component.plane}.{component.role}.{component.name}"
    if not definition.domain.contains(value):
        element = "" if definition.index is None else f"[{definition.index}]"
        raise ConfigError(f"{path}.{definition.key}{element}", f"{label}: outside {definition.domain.describe()}")
    values = dict(component.values)
    if definition.index is None:
        values[definition.key] = float(value)
    else:
        pair = list(values[definition.key])
        pair[definition.index] = float(value)
        values[definition.key] = pair
    table = (MASS_COMPONENT_TABLE if component.role == "mass" else LIGHT_COMPONENT_TABLE).tables[component.type]
    for rule in table.rules:
        try:
            rule.check(values, path)
        except ConfigError as error:
            raise ConfigError(error.path, f"{label}: {error.message}") from None
    replacement = component_from_values(component.name, component.plane, component.role,
                                        {"type": component.type, **values})
    role_components = tuple(replacement if item is component else item for item in getattr(galaxy, component.role))
    new_galaxy = dataclasses.replace(galaxy, **{component.role: role_components})
    return dataclasses.replace(spec, **{galaxy.plane: new_galaxy})


def match_parameters(names: Sequence[str], patterns: Sequence[str], *, path: str) -> tuple[str, ...]:
    """The ``names`` matched by any fnmatch pattern, in ``names`` order.

    A pattern that matches nothing raises ``ConfigError`` at ``<path>[<index>]`` listing the names.
    """
    matched = set()
    for index, pattern in enumerate(patterns):
        hits = [name for name in names if fnmatchcase(name, pattern)]
        if not hits:
            raise ConfigError(f"{path}[{index}]", f"{pattern!r} matches no parameter; parameters: {', '.join(names)}")
        matched.update(hits)
    return tuple(name for name in names if name in matched)
