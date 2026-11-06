"""Pixel integrals, continuum limits and explicit unresolved/capability failures."""
import math
import numpy as np
import pytest
from scipy.integrate import quad

from hwoslaps.scene.halo_numerics import spherical_deflections
from hwoslaps.scene.singular_integration import (SingularIntegrationError, correct_native_images,
                                                 integrate_pixel)


def _brightness(kind, centre):
    def evaluate(points):
        delta = points - np.asarray(centre)
        source = points - spherical_deflections(kind, delta, {'einstein_radius': .15})
        return np.exp(-np.sum(source**2, axis=1)/(2*.12**2))
    return evaluate


@pytest.mark.parametrize('kind', ['PointMass', 'SIS'])
def test_physical_singular_pixel_against_independent_polar_integral(kind):
    side = .4
    f = _brightness(kind, (0., 0.))
    def angular(theta):
        def radial(r):
            shifted = r - (.15**2/r if kind == 'PointMass' else .15)
            return r*math.exp(-shifted**2/(2*.12**2))
        return quad(radial, 0., (side/2)/math.cos(theta), epsabs=1e-13, epsrel=1e-11)[0]
    expected = 8*quad(angular, 0., math.pi/4, epsabs=1e-13, epsrel=1e-11)[0]/side**2
    actual = integrate_pixel(f, (0., 0.), (side, side), [(0., 0.)])
    np.testing.assert_allclose(actual, expected, rtol=2e-7, atol=1e-12)


@pytest.mark.parametrize('kind', ['PointMass', 'SIS'])
@pytest.mark.parametrize('position', [(0., 0.), (.2, 0.), (.2, .2)])
def test_continuous_native_edge_corner_and_adjacent_values(kind, position):
    centres = np.array([(y, x) for y in (.4, 0., -.4) for x in (-.4, 0., .4)])
    def image(halo):
        evaluator = _brightness(kind, halo)
        sampled = evaluator(centres)
        return correct_native_images(sampled, centres, (.4, .4), [halo], evaluator)[:, 0]
    exact = image(position)
    assert np.all(np.isfinite(exact))
    for offset in (-1e-10, 1e-10):
        nearby = image(np.asarray(position) + offset)
        np.testing.assert_allclose(nearby, exact, rtol=2e-6, atol=1e-9)


def test_unresolved_integral_raises_without_inventing_a_central_value():
    f = lambda points: np.exp(-1000*np.sum((points - [.113, .049])**2, axis=1))
    with pytest.raises(SingularIntegrationError, match='unresolved singular pixel'):
        integrate_pixel(f, (0., 0.), (1., 1.), [(0., 0.)], orders=(8, 16, 32), rtol=1e-13, max_cells=1)


@pytest.mark.backend
@pytest.mark.parametrize('kind, cls', [('PointMass', 'PointMass'), ('SIS', 'IsothermalSph')])
def test_numpy_fit_tracer_uses_shared_integral_and_jax_rejects(kind, cls):
    import autolens as al
    from hwoslaps.scene import halo_profiles
    from hwoslaps.inference.integrated_tracer import integrated_tracer
    from hwoslaps.scene.singular_integration import SingularIntegrationCapabilityError
    lens = al.Galaxy(redshift=.2, halo=getattr(halo_profiles, cls)(centre=(0., 0.), einstein_radius=.15))
    source = al.Galaxy(redshift=.6, light=al.lp.ExponentialSph(centre=(.03, -.02), intensity=1., effective_radius=.2))
    tracer = al.Tracer(galaxies=[lens, source])
    adapter = integrated_tracer(tracer, use_jax=False)
    grid = al.Grid2D.uniform(shape_native=(3, 3), pixel_scales=.1, over_sample_size=1)
    image = np.asarray(adapter.image_2d_from(grid=grid))
    assert np.all(np.isfinite(image))
    def evaluator(points):
        return np.asarray(tracer.image_2d_from(grid=al.Grid2DIrregular(values=points)))
    expected = integrate_pixel(evaluator, (0., 0.), (.1, .1), [(0., 0.)])
    assert image[4] == pytest.approx(expected[0], rel=1e-12)
    with pytest.raises(SingularIntegrationCapabilityError, match='JAX nonlinear'):
        integrated_tracer(tracer, use_jax=True)


@pytest.mark.parametrize('kind', ['PointMass', 'SIS'])
@pytest.mark.parametrize('position', [(0., 0.), (.2, 0.), (.2, .2)])
def test_complete_region_operator_against_fully_integrated_reference(kind, position):
    centres = np.array([(y, x) for y in (.4, 0., -.4) for x in (-.4, 0., .4)])
    evaluator = _brightness(kind, position)
    actual = correct_native_images(evaluator(centres), centres, (.4, .4), [position], evaluator)[:, 0]
    reference = np.array([integrate_pixel(evaluator, point, (.4, .4), [position], rtol=2e-8)[0]
                          for point in centres])
    # This tests the whole corrected operator, including the finite integration
    # region, separately from the convergence of a single Gauss rule.
    assert np.sum(np.abs(actual-reference))/np.sum(np.abs(reference)) < 4e-6


@pytest.mark.parametrize('kind', ['PointMass', 'SIS'])
@pytest.mark.parametrize('position', [(0., 0.), (.2, 0.), (.2, .2)])
def test_complete_native_image_against_independent_dense_midpoints(kind, position):
    centres = np.array([(y, x) for y in (.4, 0., -.4) for x in (-.4, 0., .4)])
    evaluator = _brightness(kind, position)
    actual = correct_native_images(evaluator(centres), centres, (.4, .4), [position], evaluator)[:, 0]
    def dense(size):
        axis = ((np.arange(size)+.5)/size-.5)*.4
        yy, xx = np.meshgrid(axis,axis,indexing='ij')
        offsets = np.column_stack([yy.ravel(),xx.ravel()])
        return np.array([np.mean(evaluator(point+offsets)) for point in centres])
    coarse, fine = dense(512), dense(1024)
    scale = np.sum(np.abs(fine))
    assert np.sum(np.abs(fine-coarse))/scale < 3e-6
    assert np.sum(np.abs(actual-fine))/scale < 4e-6
