"""Registry-to-fit model transport, halo freedom, plane assembly and joint fitting domains."""

import math

import numpy as np
import pytest

from hwoslaps.config.checks import ConfigError
from hwoslaps.inference.fit_model import autofit_model
from hwoslaps.inference.hypotheses import build_role_models
from hwoslaps.inference.settings import FitSpec, MassSupport
from hwoslaps.scene.parameters import scene_parameters

pytestmark = pytest.mark.backend

POWER_LAW = {"kind": "power_law", "c0": 20.0, "mass_pivot_msun": 1.0e8,
             "mass_slope": -0.1, "redshift_slope": -1.0}


def _models(prepared, halo, fit):
    free = [parameter.name for parameter in prepared.nuisances.parameters if parameter.kind == "scene"]
    return build_role_models(prepared.scene, halo, free, fit, use_jax=False)


@pytest.mark.parametrize("variant", ["analytic", "image", "lens_light", "perturbers"])
def test_role_models_at_truth_render_the_truth_scene(variant, prepared_forecast_factory, image_asset):
    import autolens as al
    from hwoslaps.scene.builder import build_scene

    overrides = {}
    if variant == "image":
        overrides = {"scene": {"source": {"light": {"light": {"type": "Image", "asset_path": str(image_asset),
                      "total_flux": 1.0, "rotation_deg": 12.0, "centre": [-0.03, 0.08]}}}}}
    elif variant == "lens_light":
        overrides = {"scene": {"lens": {"light": {"bulge": {"type": "Exponential", "centre": [0.0, 0.0],
                      "ell_comps": [0.1, 0.0], "effective_radius": 0.4, "intensity": 0.2}}}}}
    elif variant == "perturbers":
        overrides = {"scene": {"perturbers": {"halos": [
            {"type": "SIS", "mass_msun": 1.0e8, "centre": [0.2, 0.3], "redshift": 0.2},
            {"type": "PointMass", "mass_msun": 1.0e7, "centre": [-0.2, 0.5], "redshift": 0.4}]}}}
    prepared = prepared_forecast_factory(overrides)
    halo = prepared.hypothesis(1.0e8, (0.4, -0.6))
    models = _models(prepared, halo, FitSpec(mode="fixed_template"))
    for role in ("smooth", "subhalo"):
        model = getattr(models, role)
        instance = autofit_model(model).instance_from_vector(vector=model.truth.tolist())
        tracer = al.Tracer(galaxies=list(instance.galaxies), cosmology=prepared.scene.cosmology.autogalaxy())
        truth = build_scene(prepared.scene.spec, prepared.scene.cosmology,
                            subhalo=None if role == "smooth" else halo, perturbers=prepared.scene.perturbers)
        np.testing.assert_array_equal(tracer.image_2d_from(grid=truth.grid).native,
                                      truth.tracer.image_2d_from(grid=truth.grid).native)


@pytest.mark.parametrize("fixed", [[], ["source.light.light.effective_radius"], ["lens.*.ell_comp_*"]])
def test_free_parameters_follow_the_forecast_nuisance_set(fixed, prepared_forecast_factory):
    prepared = prepared_forecast_factory({"forecast": {"nuisances": {"fixed": fixed}}})
    model = _models(prepared, prepared.hypothesis(1.0e8, (0.4, -0.6)), FitSpec(mode="fixed_template")).smooth
    expected = [parameter for parameter in scene_parameters(prepared.scene.spec)
                if parameter.name in prepared.nuisances.names]
    assert len(model.parameter_names) == len(expected)
    np.testing.assert_array_equal(model.truth, [parameter.value for parameter in expected])


@pytest.mark.parametrize("mode", ["fixed_template", "local_search", "freed"])
def test_subhalo_freedom_per_mode(mode, prepared_forecast):
    fit = FitSpec(mode=mode, mass_support=MassSupport(6.0, 9.7) if mode == "freed" else None)
    model = _models(prepared_forecast, prepared_forecast.hypothesis(1.0e8, (0.4, -0.6)), fit).subhalo
    indices = [index for index, name in enumerate(model.parameter_names) if ".subhalo." in name]
    assert len(indices) == {"fixed_template": 0, "local_search": 2, "freed": 3}[mode]
    assert model.subhalo_path == ("galaxies", "lens", "subhalo")
    if indices:
        width = 0.03 if mode == "local_search" else 0.15
        assert [model.parameter_names[index] for index in indices[:2]] == [
            "galaxies.lens.subhalo.centre.centre_0", "galaxies.lens.subhalo.centre.centre_1"]
        np.testing.assert_array_equal(model.lower[indices[:2]], [0.4 - width, -0.6 - width])
        np.testing.assert_array_equal(model.upper[indices[:2]], [0.4 + width, -0.6 + width])
        assert max(indices) < next(index for index, name in enumerate(model.parameter_names) if ".source." in name)
    if mode == "freed":
        assert model.parameter_names[indices[-1]] == "galaxies.lens.subhalo.log10_m200"
        assert (model.lower[indices[-1]], model.upper[indices[-1]]) == (6.0, 9.7)


def test_trial_mass_outside_support_is_refused(prepared_forecast):
    with pytest.raises(ValueError, match="outside the freed mass support"):
        _models(prepared_forecast, prepared_forecast.hypothesis(1.0e10, (0.4, -0.6)),
                FitSpec(mode="freed", mass_support=MassSupport(6.0, 9.7)))


@pytest.mark.parametrize(("redshift", "mode"), [(z, mode) for z in (0.15, 0.4)
                                               for mode in ("fixed_template", "freed")])
def test_off_plane_hypotheses_follow_the_plane_assembly_rule(redshift, mode, prepared_forecast_factory):
    import autolens as al
    from hwoslaps.scene.builder import build_scene

    prepared = prepared_forecast_factory({"scene": {"subhalo": {"redshift": redshift, "concentration": POWER_LAW}}})
    halo = prepared.hypothesis(1.0e8, (0.4, -0.6))
    fit = FitSpec(mode=mode, mass_support=MassSupport(6.0, 9.7) if mode == "freed" else None)
    model = _models(prepared, halo, fit).subhalo
    assert model.subhalo_path == ("galaxies", "subhalo_plane", "subhalo")
    assert all("galaxies.subhalo_plane.subhalo" in name for name in model.parameter_names if ".subhalo." in name)
    instance = autofit_model(model).instance_from_vector(vector=model.truth.tolist())
    tracer = al.Tracer(galaxies=list(instance.galaxies), cosmology=prepared.scene.cosmology.autogalaxy())
    truth = build_scene(prepared.scene.spec, prepared.scene.cosmology, subhalo=halo)
    image = tracer.image_2d_from(grid=truth.grid).native
    expected = truth.tracer.image_2d_from(grid=truth.grid).native
    if mode == "fixed_template":
        np.testing.assert_array_equal(image, expected)
    else:
        np.testing.assert_allclose(image, expected, rtol=1e-12, atol=1e-14)
    # Independent multi-plane recursion with physical angular-diameter distances.
    redshifts = sorted({prepared.scene.spec.lens.redshift, redshift, prepared.scene.spec.source.redshift})
    positions = np.asarray(truth.grid.over_sampled)
    traced = []
    alpha = []
    cosmology = prepared.scene.cosmology
    source_z = redshifts[-1]
    for plane_z in redshifts:
        current = positions.copy()
        for prior_z, deflection in zip(redshifts[:len(alpha)], alpha, strict=True):
            if plane_z == source_z:
                beta = 1.0
            else:
                geometry_lp = cosmology.geometry(prior_z, plane_z)
                geometry_ls = cosmology.geometry(prior_z, source_z)
                beta = (geometry_lp.d_deflector_source_mpc * geometry_ls.d_source_mpc
                        / (geometry_lp.d_source_mpc * geometry_ls.d_deflector_source_mpc))
            current -= beta * deflection
        traced.append(current)
        galaxies = [galaxy for galaxy in instance.galaxies if galaxy.redshift == plane_z]
        alpha.append(sum((np.asarray(galaxy.deflections_yx_2d_from(grid=al.Grid2DIrregular(values=current))) for galaxy in galaxies),
                         np.zeros_like(current)))
    actual = tracer.traced_grid_2d_list_from(grid=al.Grid2DIrregular(values=positions))
    for grid, expected_grid in zip(actual, traced, strict=True):
        np.testing.assert_allclose(np.asarray(grid), expected_grid, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("plane", ["lens", "source"])
def test_boxes_outside_a_joint_domain_are_refused(plane, prepared_forecast_factory):
    role = "mass" if plane == "lens" else "light"
    name = "mass" if plane == "lens" else "light"
    prepared = prepared_forecast_factory({"scene": {plane: {role: {name: {"ell_comps": [0.70, 0.70]}}}}})
    with pytest.raises(ConfigError, match="largest symmetric ellipticity half width is 0.0064"):
        _models(prepared, prepared.hypothesis(1.0e8, (0.4, -0.6)), FitSpec(mode="fixed_template"))


def test_image_rotation_priors_follow_the_parameter_registry(prepared_forecast_factory, image_asset):
    prepared = prepared_forecast_factory({"scene": {"source": {"light": {"light": {
        "type": "Image", "asset_path": str(image_asset), "centre": [-0.03, 0.08],
        "total_flux": 1.0, "rotation_deg": 12.0}}}}})
    models = _models(prepared, prepared.hypothesis(1.0e8, (0.4, -0.6)), FitSpec(mode="fixed_template"))
    model = models.smooth
    indices = [index for index, name in enumerate(model.parameter_names) if ".source." in name]
    assert [model.parameter_names[index] for index in indices] == [
        "galaxies.source.light.centre.centre_0", "galaxies.source.light.centre.centre_1",
        "galaxies.source.light.flux_scale", "galaxies.source.light.size_scale", "galaxies.source.light.rotation_deg"]
    np.testing.assert_array_equal(model.truth[indices], [-0.03, 0.08, 1.0, 1.0, 12.0])
    np.testing.assert_array_equal(model.lower[indices[-1:]], [7.0])
    np.testing.assert_array_equal(model.upper[indices[-1:]], [17.0])
    converted = autofit_model(model)
    assert [".".join(path) for path in converted.unique_prior_paths] == list(model.parameter_names)
