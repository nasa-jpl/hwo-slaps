"""Freed-profile scale transport, supported mass domain and real spawned-process reconstruction."""

import multiprocessing
import os
import pickle

import numpy as np
import pytest

from hwoslaps.inference.settings import MassSupport
from hwoslaps.scene.halos import Halo, HaloModel, Moline2017, PowerLawConcentration

pytestmark = pytest.mark.backend


def _profile_scales(kind, mapping, log10_mass):
    from hwoslaps.inference.subhalo_classes import freed_profile_class

    profile = freed_profile_class(kind)(centre=(0.2, -0.3), log10_m200=log10_mass, mass_mapping=mapping)
    names = ("kappa_s", "scale_radius") if kind == "NFW" else ("einstein_radius",)
    return tuple(getattr(profile, name) for name in names)


def _spawned_scales(inputs):
    import autolens as al

    kind, mapping, mass, profile_bytes, model_bytes = inputs
    profile, model = pickle.loads(profile_bytes), pickle.loads(model_bytes)
    assert model.cls is type(profile)
    assert profile.__class__.__qualname__ == profile.__class__.__name__
    points = al.Grid2DIrregular(values=[[0.12, 0.23], [-0.17, 0.09]])
    values = np.asarray(profile.deflections_yx_2d_from(grid=points))
    assert np.all(np.isfinite(values))
    return (os.getpid(), profile.__class__.__module__,
            tuple(float(value) for value in _profile_scales(kind, mapping, mass)), values)


@pytest.mark.parametrize("family", ["NFW-moline", "NFW-moline-h0.7", "NFW-powerlaw", "SIS", "PointMass"])
def test_freed_classes_match_halo_scales(family, prepared_forecast):
    import autolens as al
    import jax
    import jax.numpy as jnp
    from hwoslaps.inference.backend import ensure_jax_x64
    from hwoslaps.inference.subhalo_classes import freed_profile_class, mass_mapping

    ensure_jax_x64()
    kind = family.split("-")[0]
    explicit_h = family == "NFW-moline-h0.7"
    log_mass = 7.0 if explicit_h else 8.0
    relation = (Moline2017(1.0, 0.7 if explicit_h else None) if family.startswith("NFW-moline")
                else PowerLawConcentration(20.0, 1.0e8, -0.1, -1.0) if kind == "NFW" else None)
    halo = Halo(HaloModel(kind, relation, None), 10.0 ** log_mass, (0.2, -0.3), 0.2, 0.6, prepared_forecast.scene.cosmology)
    mapping = mass_mapping(halo, halo.cosmology, MassSupport(6.0, 9.7))
    expected = tuple(halo.lensing().parameters.values())
    plain = _profile_scales(kind, mapping, log_mass)
    compiled = jax.jit(lambda value: _profile_scales(kind, mapping, value))(log_mass)
    np.testing.assert_allclose(plain, expected, rtol=1e-12, atol=0.0)
    np.testing.assert_allclose(compiled, expected, rtol=1e-12, atol=0.0)
    np.testing.assert_array_equal(plain, tuple(mapping.profile_scales(log_mass).values()))
    if explicit_h:
        assert mapping.h == 0.7
        inferred_halo = Halo(HaloModel("NFW", Moline2017(1.0, None), None), 1.0e7,
                             halo.position_yx_arcsec, halo.redshift, halo.source_redshift, halo.cosmology)
        inferred = mass_mapping(inferred_halo, halo.cosmology, mapping.support)
        assert inferred.h == 0.6774
        assert abs(mapping.profile_scales(7.0)["kappa_s"] / inferred.profile_scales(7.0)["kappa_s"] - 1.0) > 1.0e-6

    # Original T7 transport: actual profile deflections, changed mass and an
    # independent NumPy finite difference, beyond the shared scale algebra.
    grid = al.Grid2D.uniform(shape_native=(3, 3), pixel_scales=0.14, origin=(0.04, -0.02))
    profile_class = freed_profile_class(kind)

    def deflections(value, xp):
        profile = profile_class(centre=(0.02, -0.03), log10_m200=value, mass_mapping=mapping)
        return profile.deflections_yx_2d_from(grid=grid, xp=xp).array

    persistent = jax.jit(lambda value: deflections(value, jnp))
    first = np.asarray(jax.block_until_ready(persistent(jnp.asarray(7.0))))
    changed = np.asarray(jax.block_until_ready(persistent(jnp.asarray(7.2))))
    np.testing.assert_allclose(first, np.asarray(deflections(7.0, np)), rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(changed, np.asarray(deflections(7.2, np)), rtol=1e-12, atol=1e-12)
    assert not np.array_equal(first, changed)
    derivative = float(jax.jit(jax.grad(lambda value: jnp.sum(deflections(value, jnp))))(jnp.asarray(7.0)))
    finite_difference = (float(np.asarray(deflections(7.0 + 1e-5, np)).sum())
                         - float(np.asarray(deflections(7.0 - 1e-5, np)).sum())) / (2e-5)
    assert np.isfinite(derivative) and np.isfinite(finite_difference)
    assert abs(derivative - finite_difference) <= 1e-6 * abs(finite_difference) + 1e-8


@pytest.mark.parametrize("kind", ["NFW", "SIS", "PointMass"])
def test_freed_classes_pickle_into_spawned_workers(kind, prepared_forecast):
    import autofit as af
    from hwoslaps.inference.subhalo_classes import freed_profile_class, mass_mapping

    relation = Moline2017(1.0, None) if kind == "NFW" else None
    halo = Halo(HaloModel(kind, relation, None), 1.0e7, (0.2, -0.3), 0.2, 0.6,
                prepared_forecast.scene.cosmology)
    mapping = mass_mapping(halo, halo.cosmology, MassSupport(6.0, 9.7))
    profile_class = freed_profile_class(kind)
    profile = profile_class(centre=halo.position_yx_arcsec, log10_m200=7.0, mass_mapping=mapping)
    inputs = (kind, mapping, 7.0, pickle.dumps(profile), pickle.dumps(af.Model(profile_class)))
    expected = _spawned_scales(inputs)
    with multiprocessing.get_context("spawn").Pool(1) as pool:
        actual = pool.apply_async(_spawned_scales, (inputs,)).get(timeout=60)
    assert actual[0] != os.getpid()
    assert actual[1:3] == expected[1:3]
    np.testing.assert_array_equal(actual[3], expected[3])


@pytest.mark.parametrize("support", [(5.9, 9.7), (6.0, 12.1), (6.0, 9.7)])
def test_mass_mapping_rejects_support_outside_the_relation_domain(support, prepared_forecast):
    from hwoslaps.inference.subhalo_classes import mass_mapping

    halo = prepared_forecast.hypothesis(1.0e8, (0.2, -0.3))
    if support != (6.0, 9.7):
        with pytest.raises(ValueError, match="moline2017_eq7 support"):
            mass_mapping(halo, halo.cosmology, MassSupport(*support))
        return
    mapping = mass_mapping(halo, halo.cosmology, MassSupport(*support))
    for value in support:
        assert all(np.isfinite(scale) and scale > 0 for scale in mapping.profile_scales(value).values())
    for value in (support[0] - 1.0e-12, support[1] + 1.0e-12, np.nan):
        with pytest.raises(ValueError, match="outside support"):
            mapping.profile_scales(value)
