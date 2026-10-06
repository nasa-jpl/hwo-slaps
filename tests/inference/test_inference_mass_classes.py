"""Custom/TNFW mass transport, real JAX profile math, spawn and multi-plane truth fits."""

import multiprocessing

import numpy as np
import pytest

from hwoslaps.inference.settings import FitSpec, MassSupport
from hwoslaps.scene.cosmology import Cosmology, parse_cosmology
from hwoslaps.scene.halos import (Halo, HaloModel, OverdensityTruncation, PowerLawConcentration,
                                  TauTruncation, halo_lensing_traced)

pytestmark = pytest.mark.backend
CUSTOM = {"flat_lcdm": {"H0": 60., "Om0": 0.4}}
POINTS = np.array([[0.12, 0.23], [-0.17, 0.09], [0.31, -0.14], [0.07, -0.32]])


def _halo(kind, cosmology_name="custom", truncation=None):
    cosmology = Cosmology(parse_cosmology(CUSTOM if cosmology_name == "custom" else {"name": "Planck15"}))
    relation = PowerLawConcentration(15., 1e8, -0.1, -0.4) if kind in ("NFW", "TNFW") else None
    return Halo(HaloModel(kind, relation, truncation), 1e8, (0.02, -0.03), 0.2, 0.6, cosmology)


def _profile_scales(kind, mapping, log_mass):
    from hwoslaps.inference.subhalo_classes import freed_profile_class

    profile = freed_profile_class(kind)(centre=(0.02, -0.03), log10_m200=log_mass, mass_mapping=mapping)
    names = ("kappa_s", "scale_radius", "truncation_radius") if kind == "TNFW" else (
        ("kappa_s", "scale_radius") if kind == "NFW" else ("einstein_radius",))
    return tuple(getattr(profile, name) for name in names)


@pytest.mark.parametrize("kind, named, truncation", [
    ("PointMass", "custom", None), ("SIS", "custom", None), ("NFW", "custom", None),
    ("TNFW", "custom", TauTruncation(10.)), ("TNFW", "custom", OverdensityTruncation(100.)),
    ("TNFW", "Planck15", TauTruncation(10.)), ("TNFW", "Planck15", OverdensityTruncation(100.)),
])
def test_mass_classes_equal_halo_scales(kind, named, truncation):
    import jax
    from hwoslaps.inference.backend import ensure_jax_x64
    from hwoslaps.inference.subhalo_classes import mass_mapping

    ensure_jax_x64()
    halo = _halo(kind, named, truncation)
    mapping = mass_mapping(halo, halo.cosmology, MassSupport(6., 10.))
    expected_traced = halo_lensing_traced(halo.model, 1e8, halo.geometry, reduced_h=halo.reduced_h, xp=np)
    plain = _profile_scales(kind, mapping, 8.)
    np.testing.assert_array_equal(plain, tuple(expected_traced.values()))
    np.testing.assert_allclose(plain, tuple(halo.lensing().parameters.values()), rtol=1e-14, atol=0.)
    compiled = jax.jit(lambda value: _profile_scales(kind, mapping, value))(8.)
    np.testing.assert_allclose(compiled, plain, rtol=1e-14, atol=0.)


def _spawned_tnfw(inputs):
    halo, mapping = inputs
    return halo.to_mapping(), tuple(float(value) for value in _profile_scales("TNFW", mapping, 8.))


def test_custom_tnfw_halo_and_freed_class_pickle_into_spawned_workers():
    from hwoslaps.inference.subhalo_classes import mass_mapping

    halo = _halo("TNFW", truncation=OverdensityTruncation(100.))
    mapping = mass_mapping(halo, halo.cosmology, MassSupport(6., 10.))
    expected = _spawned_tnfw((halo, mapping))
    with multiprocessing.get_context("spawn").Pool(1) as pool:
        actual = pool.apply_async(_spawned_tnfw, ((halo, mapping),)).get(timeout=60)
    assert actual == expected


def test_truncated_nfw_class_traces_under_jax():
    import autolens as al
    import jax
    import jax.numpy as jnp
    from hwoslaps.inference.backend import ensure_jax_x64
    from hwoslaps.scene.halo_profiles import TruncatedNFWSph
    from hwoslaps.inference.subhalo_classes import TNFWM200SubhaloSph, mass_mapping
    from hwoslaps.scene.halos import FixedConcentration
    from scipy.integrate import quad
    from dataclasses import replace

    ensure_jax_x64()
    grid = al.Grid2DIrregular(values=POINTS)

    def values(profile_class, kappa, xp):
        profile = profile_class(centre=(0.02, -0.03), kappa_s=kappa, scale_radius=0.2, truncation_radius=2.)
        return (profile.deflections_yx_2d_from(grid=grid, xp=xp).array,
                profile.convergence_2d_from(grid=grid, xp=xp).array)

    expected = values(al.mp.NFWTruncatedSph, 0.03, np)
    plain = values(TruncatedNFWSph, 0.03, np)
    compiled = jax.jit(lambda kappa: values(TruncatedNFWSph, kappa, jnp))(0.03)
    for parent, actual, traced in zip(expected, plain, compiled, strict=True):
        np.testing.assert_array_equal(actual, parent)
        np.testing.assert_allclose(traced, parent, rtol=1e-13, atol=0.)
    derivative = jax.jit(jax.jacfwd(lambda kappa: values(TruncatedNFWSph, kappa, jnp)))(0.03)
    for actual, parent in zip(derivative, expected, strict=True):
        np.testing.assert_allclose(actual, parent / 0.03, rtol=1e-13, atol=0.)

    parent_profile = al.mp.NFWTruncatedSph(kappa_s=0.03, scale_radius=1., truncation_radius=10.)
    adapter_profile = TruncatedNFWSph(kappa_s=0.03, scale_radius=1., truncation_radius=10.)
    for name in ("coord_func_f", "coord_func_g"):
        for radius in (0.8, 1., 1.2, np.array([0.8, 1., 1.2], dtype=complex)):
            np.testing.assert_array_equal(getattr(adapter_profile, name)(radius, xp=jnp),
                                          getattr(parent_profile, name)(radius, xp=jnp))

    def density_at_radius(radius):
        return 100. / ((100. + radius**2) * radius * (1. + radius)**2)

    def projected_density(x, kappa):
        def density(z):
            radius = np.hypot(x, z)
            return density_at_radius(radius)

        def radial_derivative(z):
            radius = np.hypot(x, z)
            return -density(z) * (2 * radius / (100. + radius**2) + 1 / radius + 2 / (1 + radius)) * x / radius

        return tuple(2 * kappa * quad(function, 0., np.inf, epsabs=1e-12, epsrel=1e-12)[0]
                     for function in (density, radial_derivative))

    def projected_deflection(x, kappa):
        # Spherical shells contribute their polar-cap fraction inside the projected cylinder.
        interior = quad(lambda radius: density_at_radius(radius) * radius**2, 0., x,
                        epsabs=1e-12, epsrel=1e-12)[0]
        exterior = quad(lambda z: density_at_radius(np.hypot(x, z)) * z / (np.hypot(x, z) + z),
                        0., np.inf, epsabs=1e-12, epsrel=1e-12)[0]
        return 4 * kappa * (interior + x**2 * exterior) / x

    for radius in (0.8, 1., 1.2):
        point_grid = al.Grid2DIrregular(values=[[radius, 0.]])

        def geometry_values(profile_class, arguments, xp):
            scale, centre_y, centre_x = arguments
            profile = profile_class(centre=(centre_y, centre_x), kappa_s=0.03,
                                    scale_radius=scale, truncation_radius=10 * scale)
            return xp.concatenate((profile.deflections_yx_2d_from(grid=point_grid, xp=xp).array.reshape(-1),
                                   profile.convergence_2d_from(grid=point_grid, xp=xp).array.reshape(-1)))

        arguments = np.array([1., 0., 0.])
        parent = geometry_values(al.mp.NFWTruncatedSph, arguments, np)
        np.testing.assert_array_equal(geometry_values(TruncatedNFWSph, arguments, np), parent)
        np.testing.assert_allclose(jax.jit(lambda value: geometry_values(TruncatedNFWSph, value, jnp))(arguments),
                                   parent, rtol=1e-13, atol=0.)
        kappa, radial_derivative = projected_density(radius, 0.03)
        assert parent[2] == pytest.approx(kappa, rel=1e-12)
        alpha = parent[0]
        # Axisymmetric alpha'(R)=2*kappa(R)-alpha(R)/R fixes scale and centre derivatives.
        expected_gradient = np.array([[2 * (alpha - radius * kappa), -(2 * kappa - alpha / radius), 0.],
                                      [0., 0., -alpha / radius],
                                      [-radius * radial_derivative, -radial_derivative, 0.]])
        actual_gradient = jax.jit(jax.jacfwd(lambda value: geometry_values(TruncatedNFWSph, value, jnp)))(arguments)
        np.testing.assert_allclose(actual_gradient, expected_gradient, rtol=1e-10, atol=1e-12,
                                   err_msg=f"TNFW scale/centre derivatives at radius/scale={radius}")
        reverse_gradient = jax.jit(jax.jacrev(lambda value: geometry_values(TruncatedNFWSph, value, jnp)))(arguments)
        np.testing.assert_allclose(reverse_gradient, expected_gradient, rtol=1e-10, atol=1e-12,
                                   err_msg=f"TNFW reverse scale/centre derivatives at radius/scale={radius}")
        step = 1e-3
        # The five-point stencil resolves small off-point derivatives without widening tolerances.
        finite_difference = np.column_stack([
            (-geometry_values(al.mp.NFWTruncatedSph, arguments + 2 * step * direction, np) +
             8 * geometry_values(al.mp.NFWTruncatedSph, arguments + step * direction, np) -
             8 * geometry_values(al.mp.NFWTruncatedSph, arguments - step * direction, np) +
             geometry_values(al.mp.NFWTruncatedSph, arguments - 2 * step * direction, np)) / (12 * step)
            for direction in np.eye(3)])
        np.testing.assert_allclose(actual_gradient, finite_difference, rtol=1e-5, atol=1e-9)

    critical_radii = (np.nextafter(0.98, 0.), np.nextafter(0.98, 1.),
                      np.nextafter(1., 0.), np.nextafter(1., 2.),
                      np.nextafter(1.02, 1.), np.nextafter(1.02, 2.))
    truth_halo = replace(_halo("TNFW", truncation=TauTruncation(10.)), position_yx_arcsec=(0., 0.),
                         model=HaloModel("TNFW", FixedConcentration(12.), TauTruncation(10.)))
    truth_profile = truth_halo.autolens_profile()
    assert truth_profile.__class__.__module__ == "hwoslaps.scene.halo_profiles"
    assert truth_profile.__class__.__name__ == truth_halo.model.profile_class
    for radius in critical_radii:
        point_grid = al.Grid2DIrregular(values=[[radius, 0.]])
        kappa, radial_derivative = projected_density(radius, 0.03)
        alpha = projected_deflection(radius, 0.03)
        expected_values = np.array([alpha, 0., kappa])
        np.testing.assert_allclose(geometry_values(TruncatedNFWSph, arguments, np), expected_values,
                                   rtol=1e-12, atol=1e-14)
        np.testing.assert_allclose(jax.jit(lambda value: geometry_values(TruncatedNFWSph, value, jnp))(arguments),
                                   expected_values, rtol=1e-12, atol=1e-14)
        expected_gradient = np.array([[2 * (alpha - radius * kappa), -(2 * kappa - alpha / radius), 0.],
                                      [0., 0., -alpha / radius],
                                      [-radius * radial_derivative, -radial_derivative, 0.]])
        for derivative_operator in (jax.jacfwd, jax.jacrev):
            gradient = jax.jit(derivative_operator(lambda value: geometry_values(TruncatedNFWSph, value, jnp)))(arguments)
            np.testing.assert_allclose(gradient, expected_gradient, rtol=1e-10, atol=1e-12,
                                       err_msg=f"TNFW critical-neighborhood derivative at radius/scale={radius}")
        scale = truth_profile.scale_radius
        truth_grid = al.Grid2DIrregular(values=[[radius * scale, 0.]])
        truth_kappa, _ = projected_density(radius, truth_profile.kappa_s)
        truth_alpha = projected_deflection(radius, truth_profile.kappa_s) * scale
        np.testing.assert_allclose(truth_profile.deflections_yx_2d_from(grid=truth_grid).array[0],
                                   [truth_alpha, 0.], rtol=1e-12, atol=1e-14)
        np.testing.assert_allclose(truth_profile.convergence_2d_from(grid=truth_grid).array, [truth_kappa],
                                   rtol=1e-12, atol=1e-14)

    halo = replace(_halo("TNFW", truncation=TauTruncation(10.)), position_yx_arcsec=(0., 0.),
                   model=HaloModel("TNFW", FixedConcentration(12.), TauTruncation(10.)))
    mapping = mass_mapping(halo, halo.cosmology, MassSupport(6., 10.))
    scales = jax.jit(lambda mass: jnp.asarray(_profile_scales("TNFW", mapping, mass)))(8.)
    kappa_s, scale, _ = (float(value) for value in scales)
    for ratio in (0.8, 1., 1.2):
        point_grid = al.Grid2DIrregular(values=[[ratio * scale, 0.]])

        def freed_values(mass):
            profile = TNFWM200SubhaloSph(centre=(0., 0.), log10_m200=mass, mass_mapping=mapping)
            return jnp.concatenate((profile.deflections_yx_2d_from(grid=point_grid, xp=jnp).array.reshape(-1),
                                    profile.convergence_2d_from(grid=point_grid, xp=jnp).array.reshape(-1)))

        parent = al.mp.NFWTruncatedSph(centre=(0., 0.), kappa_s=kappa_s, scale_radius=1., truncation_radius=10.)
        dimensionless_grid = al.Grid2DIrregular(values=[[ratio, 0.]])
        alpha = float(parent.deflections_yx_2d_from(grid=dimensionless_grid).array[0, 0]) * scale
        kappa, radial_derivative = projected_density(ratio, kappa_s)
        np.testing.assert_allclose(jax.jit(freed_values)(8.), [alpha, 0., kappa], rtol=1e-12, atol=1e-14)
        # Fixed concentration and tau give kappa_s and theta_s proportional to M**(1/3).
        expected_mass_gradient = np.log(10.) / 3 * np.array([3 * alpha - 2 * ratio * scale * kappa,
                                                           0., kappa - ratio * radial_derivative])
        np.testing.assert_allclose(jax.jit(jax.jacfwd(freed_values))(8.), expected_mass_gradient,
                                   rtol=1e-10, atol=1e-12, err_msg=f"TNFW log-mass derivative at radius/scale={ratio}")
        np.testing.assert_allclose(jax.jit(jax.jacrev(freed_values))(8.), expected_mass_gradient,
                                   rtol=1e-10, atol=1e-12, err_msg=f"TNFW reverse log-mass derivative at radius/scale={ratio}")


@pytest.mark.parametrize("mode", ["fixed_template", "local_search", "freed"])
def test_tnfw_fit_modes_and_fixed_perturbers_trace_the_truth_scene(mode, prepared_forecast_factory):
    import autolens as al
    import jax
    import jax.numpy as jnp
    from hwoslaps.inference.backend import ensure_jax_x64
    from hwoslaps.inference.fit_model import autofit_model
    from hwoslaps.inference.hypotheses import build_role_models

    ensure_jax_x64()
    recipe = {"type": "TNFW", "concentration": {"kind": "fixed", "value": 12.},
              "truncation": {"kind": "tau", "tau": 10.}}
    prepared = prepared_forecast_factory({"cosmology": {"name": None, **CUSTOM}, "scene": {
        "subhalo": recipe, "perturbers": {"halos": [{**recipe, "mass_msun": 5e7,
                                                      "centre": [0.1, -0.2], "redshift": 0.4}]}}})
    trial = prepared.hypothesis(1e8, (0.4, -0.6))
    free = [parameter.name for parameter in prepared.nuisances.parameters if parameter.kind == "scene"]
    fit = FitSpec(mode=mode, mass_support=MassSupport(6., 10.) if mode == "freed" else None)
    models = build_role_models(prepared.scene, trial, free, fit, use_jax=True)
    grid = al.Grid2DIrregular(values=POINTS)
    for role in ("smooth", "subhalo"):
        model = getattr(models, role)
        backend_model = autofit_model(model)
        truth_scene = prepared.renderer.scene(subhalo=None if role == "smooth" else trial)
        expected = truth_scene.tracer.traced_grid_2d_list_from(grid=grid)[-1].array

        def traced(vector):
            instance = backend_model.instance_from_vector(vector=vector, xp=jnp)
            tracer = al.Tracer(galaxies=list(instance.galaxies), cosmology=prepared.scene.cosmology.autogalaxy())
            return tracer.traced_grid_2d_list_from(grid=grid, xp=jnp)[-1].array

        np.testing.assert_allclose(jax.jit(traced)(jnp.asarray(model.truth)), expected, rtol=1e-13, atol=1e-14)


def test_fit_traces_with_the_scene_cosmology(prepared_forecast_factory):
    from hwoslaps.inference.api import prepare_case
    from hwoslaps.inference.backend import ANALYSIS_CLASS

    prepared = prepared_forecast_factory({"cosmology": {"name": None, **CUSTOM}, "scene": {
        "perturbers": {"halos": [{"type": "PointMass", "mass_msun": 1e8,
                                  "centre": [-0.2, 0.5], "redshift": 0.4}]}}})
    trial = prepared.hypothesis(1e8, (0.4, -0.6))
    case = prepare_case(prepared, trial, prepared.observation, fit=FitSpec(mode="fixed_template"), use_jax=False)
    truth = case.truth_vector("smooth")
    assert case.chi_squared("smooth", truth) <= 1e-8
    # The deliberately omitted cosmology must move the actual multi-plane fit image.
    wrong_analysis = ANALYSIS_CLASS(dataset=case.data.imaging, use_jax=False)
    instance = case.autofit_models["smooth"].instance_from_vector(vector=truth.tolist())
    assert float(wrong_analysis.fit_from(instance=instance).chi_squared) > 1e-8
