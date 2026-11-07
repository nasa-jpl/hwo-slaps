"""NumPy fitting adapter for the shared singular native-brightness integral."""
from __future__ import annotations
import autoarray as aa
import autolens as al
import numpy as np

from ..scene.singular_integration import (SingularIntegrationCapabilityError, correct_native_images,
                                          singular_profiles)


class IntegratedTracer(al.Tracer):
    def __init__(self, *args, force_integration=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.force_integration = force_integration
        self._integrated_image_cache = {}

    def image_2d_list_from(self, grid, xp=np, operated_only=None):
        singular = singular_profiles(self)
        if (not singular and not self.force_integration) or not isinstance(grid, aa.Grid2D):
            return super().image_2d_list_from(grid=grid, xp=xp, operated_only=operated_only)
        if xp is not np:
            raise SingularIntegrationCapabilityError(
                'JAX nonlinear PointMass/SIS pixel integration and its derivatives are unsupported; use NumPy fits')
        if self.total_planes != 2 or any(z != self.plane_redshifts[0] for z, _ in singular):
            raise SingularIntegrationCapabilityError('singular halo integration currently requires a two-plane scene and a lens-plane halo')
        # Both requests must see the same emitting rectangle as truth,
        # independent of whether AutoLens requests fitted pixels or its
        # blurring ring. Rendering separate masked regions changes the operator.
        sizes = np.unique(np.asarray(grid.over_sample_size, dtype=int))
        if len(sizes) != 1:
            raise SingularIntegrationCapabilityError('singular fit integration requires uniform oversampling')
        key = (tuple(grid.mask.shape), tuple(grid.pixel_scales), tuple(grid.mask.origin), int(sizes[0]), operated_only)
        images = self._integrated_image_cache.get(key)
        if images is not None:
            include = ~np.asarray(grid.mask).reshape(-1)
            return [aa.Array2D(values=np.asarray(image).reshape(-1)[include], mask=grid.mask) for image in images]
        full_grid = al.Grid2D.uniform(shape_native=grid.mask.shape, pixel_scales=grid.pixel_scales,
                                      origin=grid.mask.origin, over_sample_size=int(sizes[0]))
        images = super().image_2d_list_from(grid=full_grid, xp=xp, operated_only=operated_only)
        centres = [profile.centre for _, profile in singular]
        def evaluator(points):
            values = super(IntegratedTracer, self).image_2d_list_from(
                grid=al.Grid2DIrregular(values=points), xp=np, operated_only=operated_only)
            return np.asarray(values[-1], dtype=float)
        corrected = correct_native_images(np.asarray(images[-1]), np.asarray(full_grid), grid.pixel_scales,
                                          centres, evaluator)
        images[-1] = aa.Array2D(values=corrected[:, 0], mask=full_grid.mask)
        self._integrated_image_cache[key] = images
        include = ~np.asarray(grid.mask).reshape(-1)
        return [aa.Array2D(values=np.asarray(image).reshape(-1)[include], mask=grid.mask) for image in images]


def integrated_tracer(tracer, *, use_jax, force=False):
    if not singular_profiles(tracer) and not force:
        return tracer
    if use_jax:
        raise SingularIntegrationCapabilityError(
            'JAX nonlinear PointMass/SIS pixel integration and its derivatives are unsupported; use NumPy fits')
    return IntegratedTracer(galaxies=tracer.galaxies, cosmology=tracer.cosmology, force_integration=force)
