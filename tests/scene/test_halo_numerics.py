"""Independent density oracles and exact/near-origin halo value contracts."""
import math

import numpy as np
import pytest
from scipy.integrate import quad

from hwoslaps.scene.halo_numerics import spherical_deflections


def cylinder_mass_over_x2(x, tau=None):
    def truncation(u):
        return 1. if tau is None else tau*tau / (tau*tau + u*u)
    inner = quad(lambda v: v / (1. + x*v)**2 * truncation(x*v), 0., 1.,
                 epsabs=1e-30, epsrel=1e-11)[0]
    def logarithmic_integrand(t):
        u = math.exp(t)
        ratio = min(1., x/u)
        return truncation(u) / ((1. + u)**2 * (1. + math.sqrt(max(0., 1. - ratio*ratio))))
    outer = quad(logarithmic_integrand, math.log(x), math.log(max(1e8, 1e8*(tau or 1.))),
                 epsabs=1e-30, epsrel=1e-11, limit=1000)[0]
    return inner + outer


@pytest.mark.parametrize("tau", [None, .01, .1, 1., 3., 40., 1e3])
def test_nfw_tnfw_against_independent_density_cylinder(tau):
    radii = np.unique(np.r_[np.logspace(-14., 2., 25), [.099999, .1, .100001, .98, 1., 1.02]])
    parameters = dict(kappa_s=.05, scale_radius=1.)
    if tau is not None:
        parameters['truncation_radius'] = tau
    kind = 'NFW' if tau is None else 'TNFW'
    actual = spherical_deflections(kind, np.column_stack([np.zeros_like(radii), radii]), parameters)[:, 1]
    expected = np.array([.2*x*cylinder_mass_over_x2(x, tau) for x in radii])
    assert np.all(actual > 0.)
    np.testing.assert_allclose(actual, expected, rtol=2e-9, atol=0.)
    centre = spherical_deflections(kind, np.array([[0., 0.]]), parameters)
    np.testing.assert_array_equal(centre, [[0., 0.]])


@pytest.mark.parametrize('kind', ['PointMass', 'SIS'])
def test_singular_profiles_are_analytic_at_every_positive_radius(kind):
    radii = np.r_[1e-16, 1e-12, 1e-8, 1e-6, 1e-4, 1.]
    directions = np.array([[0., 1.], [1., 0.], [-.6, .8]])
    delta = (radii[:, None, None]*directions[None, :, :]).reshape(-1, 2)
    actual = spherical_deflections(kind, delta, {'einstein_radius': .2})
    expected = (.04/radii if kind == 'PointMass' else np.full_like(radii, .2))[:, None, None]*directions
    np.testing.assert_allclose(actual.reshape(expected.shape), expected, rtol=4e-16, atol=0.)
    assert np.all(np.isnan(spherical_deflections(kind, np.array([[0., 0.]]), {'einstein_radius': .2})))


@pytest.mark.backend
@pytest.mark.parametrize('kind', ['NFW', 'TNFW', 'PointMass', 'SIS'])
def test_numpy_jax_values_and_positive_radius_derivatives(kind):
    import jax
    import jax.numpy as jnp
    jax.config.update('jax_enable_x64', True)
    p = {'einstein_radius': .2} if kind in ('PointMass', 'SIS') else dict(kappa_s=.05, scale_radius=1.)
    if kind == 'TNFW':
        p['truncation_radius'] = 3.
    radii = np.r_[np.logspace(-13, 2, 27), .1*(1-1e-7), .1*(1+1e-7), .98, 1., 1.02]
    delta = np.column_stack([np.zeros_like(radii), radii])
    np.testing.assert_allclose(spherical_deflections(kind, jnp.array(delta), p, jnp),
                               spherical_deflections(kind, delta, p), rtol=2e-12, atol=0.)
    f = lambda x: spherical_deflections(kind, jnp.array([[0., x]]), p, jnp)[0, 1]
    gradient = np.array(jax.vmap(jax.grad(f))(jnp.array(radii)))
    assert np.all(np.isfinite(gradient))
    def numpy_value(x):
        return spherical_deflections(kind, np.array([[0., x]]), p)[0, 1]
    h = 1.e-4*radii
    independent = np.array([(numpy_value(x-2*d)-8*numpy_value(x-d)+8*numpy_value(x+d)-numpy_value(x+2*d))/(12*d)
                            for x,d in zip(radii,h)])
    if kind == 'SIS':
        np.testing.assert_allclose(gradient*radii, 0., atol=2e-16)
        resolved = radii >= 1.e-3
        np.testing.assert_allclose(gradient[resolved], independent[resolved], atol=2e-9)
    else:
        np.testing.assert_allclose(gradient, independent, rtol=2e-7, atol=2e-9)
    if kind in ('NFW', 'TNFW'):
        mass_gradient = jax.grad(lambda k: spherical_deflections(kind, jnp.zeros((1, 2)), dict(p, kappa_s=k), jnp).sum())(.05)
        assert mass_gradient == 0.
        # Geometry at a true cusp is deliberately unsupported, not a finite
        # derivative inferred from the removable vector value.
        assert not np.isfinite(jax.grad(f)(0.))


@pytest.mark.backend
@pytest.mark.xtx_gpu
@pytest.mark.parametrize('kind', ['NFW', 'TNFW', 'PointMass', 'SIS'])
def test_cuda_analytic_halo_kernels_and_derivatives(kind):
    import jax
    assert jax.devices()[0].platform == 'gpu'
    test_numpy_jax_values_and_positive_radius_derivatives(kind)
