"""Native light response on emitting sky surrounding a fixed detector footprint."""

from __future__ import annotations

from dataclasses import dataclass, replace
from numbers import Integral
from typing import Any

import numpy as np

from .spec import GridSpec

__all__ = ["NATIVE_RENDER_VERSION", "RenderGeometry", "render_scene_light",
           "rendering_geometry", "response_identity"]

NATIVE_RENDER_VERSION = "native_binned_psf_padded_v1"


def response_identity(scene, rendering):
    """Describe the physical light response independently of execution backend."""
    halos = tuple(getattr(scene, "perturbers", ()))
    if getattr(scene, "subhalo", None) is not None:
        halos += (scene.subhalo,)
    singular = getattr(getattr(scene.spec, "subhalo", None), "type", None) in {"PointMass", "SIS"}
    singular |= any(halo.model.type in {"PointMass", "SIS"} for halo in halos)
    source = {"policy": "uniform_subpixel_mean"}
    if singular:
        from .singular_integration import ATOL, MAX_CELLS, MAX_DEPTH, ORDERS, POLICY, RTOL
        source = {"policy": POLICY, "rtol": RTOL, "atol": ATOL, "orders": list(ORDERS),
                  "max_cells": MAX_CELLS, "max_depth": MAX_DEPTH}
    return dict(rendering) | {"over_sample_size": scene.spec.grid.over_sample_size,
                              "light_integration": {"lens": {"policy": "uniform_subpixel_mean"},
                                                    "source": source}}


@dataclass(frozen=True)
class RenderGeometry:
    """A centred detector and symmetric emitting-sky margins in native pixels.

    Adding twice each margin preserves the detector's original pixel centres and
    subpixel phases, including for even or rectangular detector shapes.
    """

    detector: GridSpec
    padding: tuple[int, int]

    def __post_init__(self):
        if len(self.padding) != 2 or any(isinstance(n, bool) or not isinstance(n, Integral) or n < 0
                                         for n in self.padding):
            raise ValueError("render padding must contain two non-negative integers")
        object.__setattr__(self, "padding", tuple(int(n) for n in self.padding))

    @classmethod
    def from_bindings(cls, detector: GridSpec, *bindings) -> RenderGeometry:
        return cls.from_kernels(detector, *(kernel for binding in bindings if binding is not None
                                           for kernel in binding.kernels))

    @classmethod
    def from_kernels(cls, detector: GridSpec, *kernels) -> RenderGeometry:
        shapes = [kernel.shape for kernel in kernels]
        return cls(detector, tuple(max((shape[axis] // 2 for shape in shapes), default=0) for axis in (0, 1)))

    @property
    def shape(self):
        return tuple(n + 2 * pad for n, pad in zip(self.detector.shape, self.padding, strict=True))

    @property
    def slices(self):
        return tuple(slice(pad, pad + n) for n, pad in zip(self.detector.shape, self.padding, strict=True))

    def grid(self):
        from .builder import over_sampled_grid
        return over_sampled_grid(replace(self.detector, shape=self.shape))

    def crop(self, values):
        if tuple(values.shape[:2]) != self.shape:
            raise ValueError(f"rendered array has shape {values.shape}; expected {self.shape}")
        return values[self.slices]

    def pad(self, values, *, fill=0.0):
        if tuple(values.shape) != tuple(self.detector.shape):
            raise ValueError("array shape differs from the detector footprint")
        return np.pad(values, tuple((n, n) for n in self.padding), constant_values=fill)

    def to_mapping(self):
        return {"operator": NATIVE_RENDER_VERSION, "detector_shape": list(self.detector.shape),
                "render_shape": list(self.shape), "padding_yx": list(self.padding),
                "boundary": "render_exterior_light_then_crop"}


def rendering_geometry(scene, binding, geometry=None):
    required = RenderGeometry.from_bindings(scene.spec.grid, binding)
    if geometry is None:
        return required
    if geometry.detector != scene.spec.grid or any(a < b for a, b in zip(geometry.padding, required.padding)):
        raise ValueError("render geometry does not cover the detector and bound kernel support")
    return geometry


def render_scene_light(scene, geometry: RenderGeometry) -> tuple[Any, dict[str, np.ndarray]]:
    """Evaluate real sky light on the render grid, before any PSF convolution."""
    if geometry.detector != scene.spec.grid:
        raise ValueError("render geometry and scene detector differ")
    if geometry.padding == (0, 0):
        return scene.grid, dict(scene.light_images)
    from .builder import native_light_array

    grid = geometry.grid()
    if list(scene.light_groups) == ["source"]:
        images = {"source": native_light_array(scene.tracer.image_2d_from(grid=grid))}
    else:
        traced = scene.tracer.traced_grid_2d_list_from(grid=grid)
        redshifts = {"lens": scene.spec.lens.redshift, "source": scene.spec.source.redshift}
        images = {}
        for key, group in scene.light_groups.items():
            plane_grid = traced[scene.tracer.plane_redshifts.index(redshifts[group.plane])]
            profiles = scene.light_profiles[key]
            image = native_light_array(profiles[0].image_2d_from(grid=plane_grid))
            for profile in profiles[1:]:
                image = image + native_light_array(profile.image_2d_from(grid=plane_grid))
            images[key] = image
    # Scientific dependency on Batch A: a singular-hypothesis cohort uses the
    # same native brightness integral for its smooth and injected emitting sky.
    # There is deliberately no fallback to point samples if that operator fails.
    declared_singular = getattr(getattr(scene.spec, "subhalo", None), "type", None) in {"PointMass", "SIS"}
    halos = tuple(getattr(scene, "perturbers", ()))
    if getattr(scene, "subhalo", None) is not None:
        halos += (scene.subhalo,)
    if declared_singular or any(halo.model.type in {"PointMass", "SIS"} for halo in halos):
        from .singular_integration import correct_group_images
        redshifts = {"lens": scene.spec.lens.redshift, "source": scene.spec.source.redshift}
        images = correct_group_images(scene.tracer, grid, images, scene.light_profiles, scene.light_groups,
                                      redshifts, force=declared_singular)
    for key, image in images.items():
        if not np.all(np.isfinite(image)):
            raise ValueError(f"the {key} padded light image has non-finite values")
        image.setflags(write=False)
    return grid, images
