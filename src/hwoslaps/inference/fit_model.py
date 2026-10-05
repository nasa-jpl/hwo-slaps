"""Backend-neutral fit models and their conversion to AutoFit.

A ``FitModel`` lists galaxies, each galaxy's components (an AutoLens profile class and its
constructor arguments) and the prior of every argument element: fixed, uniform over a box
built around its truth value, or linked to an element of an earlier component of the same
galaxy. The free parameters are the uniform elements in declaration order. AutoFit orders the
unit-box vector by prior creation, so ``autofit_model`` creates the uniform priors in this
order before any model object and checks the resulting prior paths against
``FitModel.parameter_names``.
"""

from __future__ import annotations

import importlib
import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np

from ..identity import array_digest, mapping_digest

__all__ = ["FitArgument", "FitComponent", "FitGalaxy", "FitModel", "FitPrior", "autofit_model", "fixed",
           "linked", "uniform"]


@dataclass(frozen=True)
class FitPrior:
    """The prior of one argument element: a fixed value, a uniform box, or a link."""

    kind: Literal["fixed", "uniform", "linked"]
    value: Any = None
    lower: float | None = None
    upper: float | None = None
    truth: float | None = None
    link: tuple[str, str, int | None] | None = None

    def __post_init__(self) -> None:
        unused = {"fixed": ("lower", "upper", "truth", "link"), "uniform": ("value", "link"),
                  "linked": ("value", "lower", "upper", "truth")}
        if self.kind not in unused:
            raise ValueError(f"prior kind must be fixed, uniform or linked, got {self.kind!r}")
        set_fields = [name for name in unused[self.kind] if getattr(self, name) is not None]
        if set_fields:
            raise ValueError(f"a {self.kind} prior takes no {', '.join(set_fields)}")
        if self.kind == "fixed" and self.value is None:
            raise ValueError("a fixed prior needs a value")
        if self.kind == "uniform":
            bounds = [float(getattr(self, name)) for name in ("lower", "upper", "truth")]
            lower, upper, truth = bounds
            if not (all(math.isfinite(bound) for bound in bounds) and lower < upper and lower <= truth <= upper):
                raise ValueError(f"a uniform prior needs finite lower < upper with the truth inside, "
                                 f"got [{lower!r}, {upper!r}] around {truth!r}")
            object.__setattr__(self, "lower", lower)
            object.__setattr__(self, "upper", upper)
            object.__setattr__(self, "truth", truth)
        if self.kind == "linked":
            if not (isinstance(self.link, tuple) and len(self.link) == 3 and all(isinstance(part, str) and part
                                                                                for part in self.link[:2])
                    and (self.link[2] is None or isinstance(self.link[2], int))):
                raise ValueError(f"a linked prior needs (component, argument, element or None), got {self.link!r}")

    def to_mapping(self) -> dict[str, Any]:
        return {"kind": self.kind, "value": None if self.value is None else _value_record(self.value),
                "lower": self.lower, "upper": self.upper, "truth": self.truth,
                "link": None if self.link is None else list(self.link)}


def fixed(value: Any) -> FitPrior:
    return FitPrior(kind="fixed", value=value)


def uniform(lower: float, upper: float, *, truth: float) -> FitPrior:
    return FitPrior(kind="uniform", lower=lower, upper=upper, truth=truth)


def linked(component: str, argument: str, element: int | None = None) -> FitPrior:
    return FitPrior(kind="linked", link=(component, argument, element))


def _value_record(value: Any) -> Any:
    """The identity record of a fixed value: numbers and text as themselves, arrays and
    structural objects by digest."""
    if isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, np.ndarray):
        return {"array_sha256": array_digest(value)}
    if isinstance(value, (tuple, list)):
        return [_value_record(item) for item in value]
    digest = getattr(value, "digest", None)
    if callable(digest):
        return {"type": f"{type(value).__module__}:{type(value).__qualname__}", "digest": digest()}
    raise TypeError(f"a fixed value of type {type(value).__name__} has no identity record; "
                    "use a number, text, an array or an object with digest()")


@dataclass(frozen=True)
class FitArgument:
    """One constructor argument: a scalar (one element) or a pair (two elements)."""

    name: str
    elements: tuple[FitPrior, ...]
    pair: bool

    def __post_init__(self) -> None:
        object.__setattr__(self, "elements", tuple(self.elements))
        if not (isinstance(self.name, str) and self.name):
            raise ValueError(f"argument name must be non-empty text, got {self.name!r}")
        if len(self.elements) != (2 if self.pair else 1) or not all(isinstance(e, FitPrior) for e in self.elements):
            raise ValueError(f"argument {self.name!r} needs {2 if self.pair else 1} FitPrior element(s)")

    def to_mapping(self) -> dict[str, Any]:
        return {"name": self.name, "pair": self.pair, "elements": [element.to_mapping() for element in self.elements]}


@dataclass(frozen=True)
class FitComponent:
    """One profile: its class as ``"module:Class"`` and its arguments in prior creation order."""

    profile_class: str
    arguments: tuple[FitArgument, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "arguments", tuple(self.arguments))
        module, _, name = self.profile_class.partition(":")
        if not (module and name):
            raise ValueError(f"profile_class must be 'module:Class', got {self.profile_class!r}")
        names = [argument.name for argument in self.arguments]
        if len(set(names)) != len(names):
            raise ValueError(f"{self.profile_class} repeats an argument: {names}")

    def argument(self, name: str) -> FitArgument:
        for argument in self.arguments:
            if argument.name == name:
                return argument
        raise KeyError(f"{self.profile_class} has no argument {name!r}")

    def to_mapping(self) -> dict[str, Any]:
        return {"profile_class": self.profile_class, "arguments": [a.to_mapping() for a in self.arguments]}


@dataclass(frozen=True)
class FitGalaxy:
    """One galaxy: name, redshift and its named components in declaration order."""

    name: str
    redshift: float
    components: tuple[tuple[str, FitComponent], ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "components", tuple(tuple(entry) for entry in self.components))
        if not (isinstance(self.name, str) and self.name) or self.name == "subhalo":
            # The pinned AutoLens re-traces galaxies.subhalo.mass.centre from the image plane.
            raise ValueError(f"galaxy name must be non-empty text other than 'subhalo', got {self.name!r}")
        redshift = float(self.redshift)
        if not (math.isfinite(redshift) and redshift >= 0.0):
            raise ValueError(f"galaxy {self.name!r} needs a finite redshift >= 0, got {self.redshift!r}")
        object.__setattr__(self, "redshift", redshift)
        names = [name for name, _ in self.components]
        if len(set(names)) != len(names):
            raise ValueError(f"galaxy {self.name!r} repeats a component: {names}")
        for index, (name, component) in enumerate(self.components):
            for argument in component.arguments:
                for element in argument.elements:
                    if element.kind == "linked":
                        self._check_link(name, element.link, dict(self.components[:index]))

    def _check_link(self, name: str, link: tuple[str, str, int | None], earlier: Mapping[str, FitComponent]) -> None:
        target, argument_name, element = link
        if target not in earlier:
            raise ValueError(f"component {name!r} of galaxy {self.name!r} links to {target!r}, "
                             f"which is not an earlier component of the same galaxy")
        argument = earlier[target].argument(argument_name)
        if (element is None) == argument.pair or (element is not None and element not in (0, 1)):
            raise ValueError(f"link {link} does not name an element of {target}.{argument_name}")
        if argument.elements[0 if element is None else element].kind == "linked":
            raise ValueError(f"link {link} names an element that is itself linked")

    def to_mapping(self) -> dict[str, Any]:
        return {"name": self.name, "redshift": self.redshift,
                "components": [{"name": name, **component.to_mapping()} for name, component in self.components]}


def _element_path(galaxy: str, component: str, argument: FitArgument, index: int) -> str:
    stem = f"galaxies.{galaxy}.{component}.{argument.name}"
    return f"{stem}.{argument.name}_{index}" if argument.pair else stem


def _uniform_elements(model: FitModel):
    """``(key, name, prior)`` of every uniform element in declaration (prior creation) order.

    A linked element shares its target's prior, and AutoFit names a shared prior by the last
    path that holds it, so a linked target takes the path of its last linker.
    """
    for galaxy in model.galaxies:
        names: dict[tuple[str, str, int | None], str] = {}
        uniform: list[tuple[tuple[str, str, int | None], int, FitPrior]] = []
        for component_name, component in galaxy.components:
            for argument in component.arguments:
                for index, element in enumerate(argument.elements):
                    key = (component_name, argument.name, index if argument.pair else None)
                    path = _element_path(galaxy.name, component_name, argument, index)
                    if element.kind == "uniform":
                        names[key] = path
                        uniform.append((key, index, element))
                    elif element.kind == "linked" and element.link in names:
                        names[element.link] = path
        for key, index, element in uniform:
            yield (galaxy.name, key[0], key[1], index), names[key], element


@dataclass(frozen=True)
class FitModel:
    """The galaxies of one role's fit model. The subhalo role has exactly one component named
    ``subhalo``, in galaxy ``lens`` or ``subhalo_plane``; the smooth role has none."""

    role: Literal["smooth", "subhalo"]
    galaxies: tuple[FitGalaxy, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "galaxies", tuple(self.galaxies))
        if self.role not in ("smooth", "subhalo"):
            raise ValueError(f"role must be smooth or subhalo, got {self.role!r}")
        names = [galaxy.name for galaxy in self.galaxies]
        if len(set(names)) != len(names):
            raise ValueError(f"galaxy names repeat: {names}")
        holders = [galaxy.name for galaxy in self.galaxies for name, _ in galaxy.components if name == "subhalo"]
        if self.role == "smooth" and holders:
            raise ValueError(f"the smooth role has no subhalo component, found one in {holders}")
        if self.role == "subhalo" and not (len(holders) == 1 and holders[0] in ("lens", "subhalo_plane")):
            raise ValueError(f"the subhalo role needs one subhalo component in galaxy lens or subhalo_plane, "
                             f"found it in {holders}")

    @property
    def parameter_names(self) -> tuple[str, ...]:
        """AutoFit prior paths of the free parameters, in unit-box vector order."""
        return tuple(name for _, name, _ in _uniform_elements(self))

    @property
    def lower(self) -> np.ndarray:
        return _frozen([element.lower for _, _, element in _uniform_elements(self)])

    @property
    def upper(self) -> np.ndarray:
        return _frozen([element.upper for _, _, element in _uniform_elements(self)])

    @property
    def truth(self) -> np.ndarray:
        return _frozen([element.truth for _, _, element in _uniform_elements(self)])

    @property
    def subhalo_path(self) -> tuple[str, ...] | None:
        """``("galaxies", <galaxy>, "subhalo")`` for the subhalo role; None for the smooth role."""
        for galaxy in self.galaxies:
            if any(name == "subhalo" for name, _ in galaxy.components):
                return ("galaxies", galaxy.name, "subhalo")
        return None

    def to_mapping(self) -> dict[str, Any]:
        return {"role": self.role, "galaxies": [galaxy.to_mapping() for galaxy in self.galaxies]}

    def digest(self) -> str:
        return mapping_digest(self.to_mapping())


def _frozen(values: list[float]) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    array.setflags(write=False)
    return array


def _profile_class(path: str) -> type:
    module, _, qualname = path.partition(":")
    value: Any = importlib.import_module(module)
    for part in qualname.split("."):
        value = getattr(value, part)
    return value


def autofit_model(model: FitModel) -> Any:
    """The ``af.Collection`` of ``model``, with free priors in ``model.parameter_names`` order.

    Fixed arguments whose values are not floats (arrays, integer orders, structural objects)
    are constructor arguments of their ``af.Model``; every other element is assigned after
    construction. A linked element is the same prior object as its target, so AutoFit counts
    it once. Raises RuntimeError when AutoFit reports other free priors or another order,
    which an unassigned constructor argument or a backend ordering change would cause.
    """
    import autofit as af
    import autolens as al

    created = {key: af.UniformPrior(lower_limit=element.lower, upper_limit=element.upper)
               for key, _, element in _uniform_elements(model)}
    galaxies = {}
    for galaxy in model.galaxies:
        assigned: dict[tuple[str, str, int | None], Any] = {}
        components = {}
        for component_name, component in galaxy.components:
            structural = {argument.name: argument.elements[0].value for argument in component.arguments
                          if not argument.pair and argument.elements[0].kind == "fixed"
                          and not isinstance(argument.elements[0].value, float)}
            profile = af.Model(_profile_class(component.profile_class), **structural)
            for argument in component.arguments:
                if argument.name in structural:
                    continue
                for index, element in enumerate(argument.elements):
                    if element.kind == "uniform":
                        value = created[(galaxy.name, component_name, argument.name, index)]
                    elif element.kind == "fixed":
                        value = element.value
                    else:
                        value = assigned[element.link]
                    assigned[(component_name, argument.name, index if argument.pair else None)] = value
                    if argument.pair:
                        setattr(getattr(profile, argument.name), f"{argument.name}_{index}", value)
                    else:
                        setattr(profile, argument.name, value)
            components[component_name] = profile
        galaxies[galaxy.name] = af.Model(al.Galaxy, redshift=galaxy.redshift, **components)
    collection = af.Collection(galaxies=af.Collection(**galaxies))
    paths = [".".join(path) for path in collection.unique_prior_paths]
    if collection.prior_count != len(model.parameter_names) or paths != list(model.parameter_names):
        raise RuntimeError(f"AutoFit built {collection.prior_count} free priors {paths}; the fit model frees "
                           f"{list(model.parameter_names)}")
    return collection
