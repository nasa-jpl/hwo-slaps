"""Inference-only round Exponential derivative and the separate exact-centre cusp policy."""

import numpy as np
import pytest

pytestmark = pytest.mark.backend


def test_exponential_primal_and_nonround_derivative_are_the_backend_bits():
    import jax
    import jax.numpy as jnp
    from jax.experimental import checkify
    import autolens as al
    import autogalaxy as ag
    from hwoslaps.inference.backend import ensure_jax_x64
    from hwoslaps.inference.light_profiles import Exponential

    ensure_jax_x64()
    points = np.array([[0.1, 0.2], [-0.15, 0.08], [0.3, -0.12], [0.05, -0.3]])
    grid = al.Grid2DIrregular(values=points)
    weights = jnp.array([-2.0, 1.0, 0.5, 3.0])

    def images(profile_class, ell, xp):
        profile = profile_class(centre=(-0.03, 0.08), ell_comps=(ell[0], ell[1]), intensity=1.0, effective_radius=0.12)
        return xp.asarray(profile.image_2d_from(grid=grid, xp=xp).array)

    for ell in (np.array([0.0, 0.0]), np.array([0.05, 0.02]), np.array([-0.02, 0.04]), np.array([0.14516129, 0.25142673])):
        np.testing.assert_array_equal(images(Exponential, ell, np), images(ag.lp.Exponential, ell, np))
        np.testing.assert_array_equal(jax.jit(lambda value: images(Exponential, value, jnp))(ell),
                                      jax.jit(lambda value: images(ag.lp.Exponential, value, jnp))(ell))
        error, actual = jax.jit(checkify.checkify(jax.grad(lambda value: weights @ images(Exponential, value, jnp))))(ell)
        error.throw()
        if np.any(ell != 0.0):
            expected = jax.jit(jax.grad(lambda value: weights @ images(ag.lp.Exponential, value, jnp)))(ell)
            np.testing.assert_allclose(actual, expected, rtol=2e-15, atol=1e-15)
        else:
            np.testing.assert_allclose(actual, [-2.437377187323684, -1.180759067759719], rtol=1e-12, atol=0.0)


@pytest.mark.parametrize("gpu", [False, pytest.param(True, marks=pytest.mark.xtx_gpu)], ids=["cpu", "gpu"])
def test_centre_refuses_gradients_and_keeps_value_sampling(gpu, monkeypatch):
    import jax
    import jax.numpy as jnp
    from jax.experimental import checkify
    import autolens as al
    import autogalaxy as ag
    from hwoslaps.inference.backend import ensure_jax_x64
    from hwoslaps.inference.light_profiles import Exponential

    ensure_jax_x64()
    assert jax.default_backend() == ("gpu" if gpu else "cpu")
    executed = []
    original_callback = jax.pure_callback

    def recording_callback(callback, *args, **kwargs):
        def run(*values):
            executed.append(True)
            return callback(*values)
        return original_callback(run, *args, **kwargs)

    monkeypatch.setattr(jax, "pure_callback", recording_callback)
    centre = jnp.array([-0.03, 0.08])
    grid = al.Grid2DIrregular(values=np.array([[-0.03, 0.08]]))

    def value(position, point_grid=grid):
        profile = Exponential(centre=(position[0], position[1]), ell_comps=(0.0, 0.0),
                               intensity=1.0, effective_radius=0.12)
        return jnp.sum(profile.image_2d_from(grid=point_grid, xp=jnp).array)

    primal = float(jax.jit(value)(centre))
    def parent_value(position):
        profile = ag.lp.Exponential(centre=(position[0], position[1]), ell_comps=(0.0, 0.0),
                                    intensity=1.0, effective_radius=0.12)
        return jnp.sum(profile.image_2d_from(grid=grid, xp=jnp).array)
    expected = float(jax.jit(parent_value)(centre))
    assert primal == expected and executed == []
    error, derivative = jax.jit(checkify.checkify(jax.grad(value)))(centre)
    with pytest.raises(Exception, match="Exponential gradient is undefined at an exactly zero-radius"):
        error.throw()
    assert executed == []
    nearby = al.Grid2DIrregular(values=np.array([[-0.03 + 1e-12, 0.08]]))
    error, derivative = jax.jit(checkify.checkify(jax.grad(lambda position: value(position, nearby))))(centre)
    error.throw()
    derivative = np.asarray(derivative)
    assert np.all(np.isfinite(derivative)) and np.max(np.abs(derivative)) > 70.0
    assert executed == []
