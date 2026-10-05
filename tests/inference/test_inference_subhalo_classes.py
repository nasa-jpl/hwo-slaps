"""Freed-profile scale transport, supported mass domain and real spawned-process reconstruction."""

import multiprocessing

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
    kind, mapping, mass = inputs
    return tuple(float(value) for value in _profile_scales(kind, mapping, mass))


@pytest.mark.parametrize("family", ["NFW-moline", "NFW-powerlaw", "SIS", "PointMass"])
def test_freed_classes_match_halo_scales(family, prepared_forecast):
    import jax
    from hwoslaps.inference.backend import ensure_jax_x64
    from hwoslaps.inference.subhalo_classes import mass_mapping

    ensure_jax_x64()
    kind = family.split("-")[0]
    relation = (Moline2017(1.0, None) if family == "NFW-moline"
                else PowerLawConcentration(20.0, 1.0e8, -0.1, -1.0) if kind == "NFW" else None)
    halo = Halo(HaloModel(kind, relation, None), 1.0e8, (0.2, -0.3), 0.2, 0.6, prepared_forecast.scene.cosmology)
    mapping = mass_mapping(halo, halo.cosmology, MassSupport(6.0, 9.7))
    expected = tuple(halo.lensing().parameters.values())
    plain = _profile_scales(kind, mapping, 8.0)
    compiled = jax.jit(lambda value: _profile_scales(kind, mapping, value))(8.0)
    np.testing.assert_allclose(plain, expected, rtol=1e-12, atol=0.0)
    np.testing.assert_allclose(compiled, expected, rtol=1e-12, atol=0.0)
    np.testing.assert_array_equal(plain, tuple(mapping.profile_scales(8.0).values()))


def test_freed_classes_pickle_into_spawned_workers(prepared_forecast):
    from hwoslaps.inference.subhalo_classes import mass_mapping

    halo = prepared_forecast.hypothesis(1.0e8, (0.2, -0.3))
    mapping = mass_mapping(halo, halo.cosmology, MassSupport(6.0, 9.7))
    inputs = ("NFW", mapping, 8.0)
    with multiprocessing.get_context("spawn").Pool(1) as pool:
        actual = pool.apply_async(_spawned_scales, (inputs,)).get(timeout=60)
    assert actual == _spawned_scales(inputs)


@pytest.mark.parametrize("support", [(5.9, 9.7), (6.0, 12.1)])
def test_mass_mapping_rejects_support_outside_the_relation_domain(support, prepared_forecast):
    from hwoslaps.inference.subhalo_classes import mass_mapping

    halo = prepared_forecast.hypothesis(1.0e8, (0.2, -0.3))
    with pytest.raises(ValueError, match="moline2017_eq7 support"):
        mass_mapping(halo, halo.cosmology, MassSupport(*support))
