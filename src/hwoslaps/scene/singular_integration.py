"""Convergence-checked native-pixel brightness integration for PointMass and SIS.

This corrects the pre-PSF native light image, so detector area and the existing
native convolution convention are applied once. The physical halo and detector
stay fixed during refinement. Only integration nodes change. No deflection is
averaged and no central deflection is invented.
"""
from __future__ import annotations

from functools import lru_cache
import numpy as np

POLICY = 'native-brightness-adaptive-v1'
ORDERS = (8, 16, 32)
MAX_CELLS = 1024
MAX_DEPTH = 20
RTOL = 2.e-6
ATOL = 1.e-10


class SingularIntegrationError(ValueError):
    """A physical pixel integral could not be resolved under the bounded policy."""


class SingularIntegrationCapabilityError(ValueError):
    """The requested model/derivative has no validated integrated implementation."""


def singular_profiles(tracer):
    return tuple((galaxy.redshift, profile) for galaxy in tracer.galaxies
                 for profile in vars(galaxy).values()
                 if getattr(profile, '_halo_kind', None) in ('PointMass', 'SIS'))


@lru_cache(maxsize=len(ORDERS))
def _rule(order):
    return np.polynomial.legendre.leggauss(order)


def _split_interval(lower, upper, centres):
    # Do not construct rectangles narrower than floating-point coordinates can
    # represent. This selects a quadrature partition, never moves a halo.
    margin = 32*np.finfo(float).eps*max(1., abs(lower), abs(upper))
    return np.unique([lower, *(c for c in centres if lower + margin < c < upper - margin), upper])


def integrate_pixel(evaluator, pixel_yx, pixel_scales, centres, *, orders=ORDERS, rtol=RTOL, atol=ATOL,
                    max_cells=MAX_CELLS, max_depth=MAX_DEPTH):
    """Mean brightness on a fixed pixel, with bounded spatial refinement.

    Each leaf rectangle compares 8/16/32 Gauss rules. The sum of the larger of
    the two successive differences must meet the pixel error budget. Refine
    the worst rectangle until it does, or raise at the cell/depth bound. This
    is a numerical convergence check, not a guarantee against arbitrary source
    features hidden between every initial quadrature node.
    """
    centre = np.asarray(pixel_yx, dtype=float)
    scales = np.asarray(pixel_scales, dtype=float)
    centres = np.asarray(centres, dtype=float).reshape(-1, 2)
    ys = _split_interval(centre[0] - scales[0]/2, centre[0] + scales[0]/2, centres[:, 0])
    xs = _split_interval(centre[1] - scales[1]/2, centre[1] + scales[1]/2, centres[:, 1])
    if len(orders) < 3:
        raise ValueError('singular integration needs at least three Gauss orders')
    if (len(ys)-1)*(len(xs)-1) > max_cells:
        raise SingularIntegrationError('unresolved singular pixel: initial partition exceeds the leaf-cell budget')
    def evaluate_rectangle(bounds, depth):
        lo_y, hi_y, lo_x, hi_x = bounds
        estimates = []
        for order in orders:
            nodes, weights = _rule(order)
            y = (lo_y + hi_y)/2 + (hi_y - lo_y)/2*nodes
            x = (lo_x + hi_x)/2 + (hi_x - lo_x)/2*nodes
            yy, xx = np.meshgrid(y, x, indexing='ij')
            values = np.asarray(evaluator(np.column_stack([yy.ravel(), xx.ravel()])), dtype=float)
            if values.ndim == 1:
                values = values[:, None]
            if not np.all(np.isfinite(values)):
                raise SingularIntegrationError('nonfinite lensed brightness during singular pixel integration')
            term = np.einsum('i,j,ijc->c', weights, weights, values.reshape(order, order, -1))
            estimates.append(term*(hi_y - lo_y)*(hi_x - lo_x)/(4*np.prod(scales)))
        error = np.maximum(np.abs(estimates[-1]-estimates[-2]), np.abs(estimates[-2]-estimates[-3]))
        return bounds, depth, estimates[-1], error
    cells = [evaluate_rectangle((a, b, c, d), 0)
             for a, b in zip(ys[:-1], ys[1:]) for c, d in zip(xs[:-1], xs[1:])]
    while True:
        integral = np.sum([cell[2] for cell in cells], axis=0)
        error = np.sum([cell[3] for cell in cells], axis=0)
        tolerance = atol + rtol*np.abs(integral)
        if np.all(error <= tolerance):
            return integral
        worst = int(np.argmax([np.max(cell[3]/tolerance) for cell in cells]))
        bounds, depth, _, _ = cells[worst]
        if len(cells)+3 > max_cells or depth >= max_depth:
            raise SingularIntegrationError(
                f'unresolved singular pixel brightness at {tuple(centre)}; leaf cells={len(cells)}, '
                f'depth={depth}, normalized error={float(np.max(error/tolerance)):.6g}; policy={POLICY}')
        lo_y, hi_y, lo_x, hi_x = bounds
        mid_y, mid_x = (lo_y+hi_y)/2, (lo_x+hi_x)/2
        children = [evaluate_rectangle((a,b,c,d), depth+1)
                    for a,b in ((lo_y,mid_y),(mid_y,hi_y)) for c,d in ((lo_x,mid_x),(mid_x,hi_x))]
        cells[worst:worst+1] = children


def correct_native_images(images, native_yx, pixel_scales, singular_centres, evaluator):
    """Integrate every native cell under one operator, including smooth controls.

    Using the same fully integrated emitting grid for both model roles avoids
    introducing a mass-independent quadrature offset into a halo template.
    """
    native_yx = np.asarray(native_yx, dtype=float)
    images = np.asarray(images)
    columns = 1 if images.ndim == 1 else images.shape[1]
    scales = np.asarray(pixel_scales, dtype=float)
    centres = np.asarray(singular_centres, dtype=float).reshape(-1, 2)
    result = np.empty((len(native_yx), columns))
    # Batch the initial rules for ordinary cells; only unresolved/cusp cells
    # need Python-level adaptive subdivision. This changes no nodes or budget.
    contains = (np.any(np.all(np.abs(native_yx[:, None, :] - centres[None, :, :]) <= scales/2,
                              axis=2), axis=1) if len(centres) else np.zeros(len(native_yx), dtype=bool))
    regular = np.flatnonzero(~contains)
    for start in range(0, len(regular), 64):
        selected = regular[start:start+64]
        estimates = []
        for order in ORDERS:
            nodes, weights = _rule(order)
            yy, xx = np.meshgrid(nodes*scales[0]/2, nodes*scales[1]/2, indexing='ij')
            offsets = np.column_stack([yy.ravel(),xx.ravel()])
            points = (native_yx[selected,None,:] + offsets[None,:,:]).reshape(-1,2)
            values = np.asarray(evaluator(points), dtype=float).reshape(len(selected),order*order,columns)
            if not np.all(np.isfinite(values)):
                raise SingularIntegrationError('nonfinite lensed brightness during singular pixel integration')
            estimates.append(np.einsum('k,nkc->nc', np.outer(weights,weights).ravel()/4, values))
        error = np.maximum(np.abs(estimates[-1]-estimates[-2]), np.abs(estimates[-2]-estimates[-3]))
        accepted = np.all(error <= ATOL + RTOL*np.abs(estimates[-1]), axis=1)
        result[selected[accepted]] = estimates[-1][accepted]
        for index in selected[~accepted]:
            result[index] = integrate_pixel(evaluator, native_yx[index], scales, centres)
    for index in np.flatnonzero(contains):
        result[index] = integrate_pixel(evaluator, native_yx[index], scales, centres)
    return result


def correct_group_images(tracer, grid, images, light_profiles, light_groups, redshifts, *, force=False):
    """Shared truth/reference correction for named light groups."""
    singular = singular_profiles(tracer)
    if not singular and not force:
        return images
    if tracer.total_planes != 2 or any(z != tracer.plane_redshifts[0] for z, _ in singular):
        raise SingularIntegrationCapabilityError('singular halo integration currently requires a two-plane scene and a lens-plane halo')
    import autolens as al
    centres = [profile.centre for _, profile in singular]
    keys = [key for key, group in light_groups.items() if redshifts[group.plane] > tracer.plane_redshifts[0]]
    if not keys:
        return images
    def evaluator(points):
        traced = tracer.traced_grid_2d_list_from(grid=al.Grid2DIrregular(values=points))[-1]
        return np.column_stack([sum(np.asarray(profile.image_2d_from(grid=traced), dtype=float)
                                    for profile in light_profiles[key]) for key in keys])
    values = np.column_stack([np.asarray(images[key]).ravel() for key in keys])
    corrected = correct_native_images(values, np.asarray(grid), grid.pixel_scales, centres, evaluator)
    output = dict(images)
    for column, key in enumerate(keys):
        output[key] = corrected[:, column].reshape(images[key].shape)
        output[key].setflags(write=False)
    return output
