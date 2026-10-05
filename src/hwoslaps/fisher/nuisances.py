"""Registry-ordered finite differences of the model detector mean."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import numpy as np

from ..config.checks import ConfigError
from ..scene.parameters import match_parameters, scene_parameters, with_parameter
from .spec import check_nuisance_spec

if TYPE_CHECKING:
    from ..scene.builder import Scene
    from ..scene.spec import SceneSpec
    from .psf_pair import PsfPair
    from .renderer import SceneRenderer
    from .spec import NuisanceSpec

__all__ = ["NuisanceDesign", "NuisanceParameter", "build_nuisance_design", "resolve_nuisances"]

_DEFAULT_STEPS = {"position": 1.0e-3, "einstein_radius": 1.0e-3, "ellipticity": 1.0e-3,
                  "slope": 1.0e-3, "multipole": 1.0e-3, "shear": 1.0e-3, "amplitude": 1.0e-2,
                  "size": 1.0e-2, "sersic_index": 1.0e-3, "orientation": 0.1}


@dataclass(frozen=True)
class NuisanceParameter:
    name: str
    kind: Literal["scene", "background", "wavefront"]
    value: float
    step: float
    prior_sigma: float | None


@dataclass(frozen=True, eq=False)
class NuisanceDesign:
    parameters: tuple[NuisanceParameter, ...]
    images: np.ndarray
    prior_precision: np.ndarray

    def __post_init__(self) -> None:
        if self.images.ndim != 3 or self.images.shape[0] != len(self.parameters):
            raise ValueError("nuisance images must have shape (parameters, ny, nx)")
        if self.prior_precision.shape != (len(self.parameters),):
            raise ValueError("prior precision must have one entry per nuisance")
        if not np.all(np.isfinite(self.images)) or not np.all(np.isfinite(self.prior_precision)):
            raise ValueError("nuisance images and prior precision must be finite")
        if np.any(self.prior_precision < 0.0):
            raise ValueError("prior precision must be non-negative")
        self.images.flags.writeable = False
        self.prior_precision.flags.writeable = False

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(parameter.name for parameter in self.parameters)


def _family_value(value: float | Mapping[str, float] | None, family: str) -> float | None:
    return value.get(family) if isinstance(value, Mapping) else value


def resolve_nuisances(scene: SceneSpec, spec: NuisanceSpec, psfs: PsfPair) -> tuple[NuisanceParameter, ...]:
    provider = psfs.model.provider
    check_nuisance_spec(scene, spec, model_has_basis=provider.basis is not None)
    scene_values = scene_parameters(scene)
    fixed = set(match_parameters([parameter.name for parameter in scene_values], spec.fixed,
                                 path="forecast.nuisances.fixed"))
    parameters = []
    for parameter in scene_values:
        if parameter.name in fixed:
            continue
        definition = parameter.definition
        step = spec.steps.get(parameter.name, spec.steps.get(definition.kind, _DEFAULT_STEPS[definition.kind]))
        half_step = abs(parameter.value) * step if definition.step_mode == "multiplicative" else step
        if half_step == 0.0:
            half_step = step
        for sign in (-1.0, 1.0):
            try:
                with_parameter(scene, parameter.name, parameter.value + sign * half_step)
            except ConfigError as error:
                raise ConfigError(f"forecast.nuisances.steps.{parameter.name}",
                                  f"{parameter.name} = {parameter.value!r}, half-step {half_step!r}: {error.message}") from None
        parameters.append(NuisanceParameter(parameter.name, "scene", parameter.value, half_step,
                                            spec.priors.get(parameter.name)))
    if spec.background_offset:
        parameters.append(NuisanceParameter("observation.background_offset_adu", "background", 0.0, 1.0, None))
    if spec.wavefront is not None:
        for mode in provider.basis.select(spec.wavefront.modes):
            if mode.family == "zernikes" and mode.noll == 1:
                raise ConfigError("forecast.nuisances.wavefront.modes", "global Zernike Noll 1 is a piston, not a nuisance")
            parameters.append(NuisanceParameter("psf." + mode.name, "wavefront", provider.coefficients.value(mode),
                                                _family_value(spec.wavefront.step_nm, mode.family),
                                                _family_value(spec.wavefront.prior_sigma_nm, mode.family)))
    return tuple(parameters)


def build_nuisance_design(parameters: Sequence[NuisanceParameter], *, renderer: SceneRenderer,
                          smooth_scene: Scene, psfs: PsfPair) -> NuisanceDesign:
    images = []
    for parameter in parameters:
        if parameter.kind == "scene":
            plus = renderer.scene(spec=with_parameter(smooth_scene.spec, parameter.name, parameter.value + parameter.step))
            minus = renderer.scene(spec=with_parameter(smooth_scene.spec, parameter.name, parameter.value - parameter.step))
            image = (renderer.mean_adu(plus, psfs.model_kernels)
                     - renderer.mean_adu(minus, psfs.model_kernels)) / (2.0 * parameter.step)
        elif parameter.kind == "background":
            image = np.ones(smooth_scene.spec.grid.shape)
        else:
            # A selected zero coefficient need not be present in the provider's coefficient list.
            from ..optics.wavefront import WavefrontMode
            import re
            match = re.fullmatch(r"psf\.zernikes\[(\d+)\]", parameter.name)
            if match:
                mode = WavefrontMode("zernikes", int(match[1]))
            else:
                match = re.fullmatch(r"psf\.segment_hexikes\[(\d+)\]\[(\d+)\]", parameter.name)
                if match is None:
                    raise ValueError(f"unknown wavefront nuisance {parameter.name!r}")
                mode = WavefrontMode("segment_hexikes", int(match[2]), int(match[1]))
            derivatives = psfs.model_kernel_derivatives(mode, parameter.step)
            image = renderer.derivative_adu(smooth_scene, psfs.model_kernels, derivatives)
        images.append(image)
    shape = smooth_scene.spec.grid.shape
    stacked = np.stack(images) if images else np.empty((0, *shape), dtype=float)
    precision = np.array([0.0 if parameter.prior_sigma is None else 1.0 / (parameter.prior_sigma ** 2)
                          for parameter in parameters])
    return NuisanceDesign(tuple(parameters), stacked, precision)
