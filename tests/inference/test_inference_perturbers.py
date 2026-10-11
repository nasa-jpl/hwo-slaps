"""Realized populations remain fixed in truth-matched H0/H1 fits.

Requires the reviewed committed INFB a938fd4 dependency in the assembled tree.
Imports remain inside the backend keeper; dependency absence is never skipped.
"""

import copy

import numpy as np
import pytest

pytestmark = pytest.mark.backend


def test_smooth_and_subhalo_models_hold_realized_populations_fixed(minimal_mapping):
    from hwoslaps.fisher.api import prepare_forecast
    from hwoslaps.inference.api import prepare_case
    from hwoslaps.inference.fit_model import autofit_model
    from hwoslaps.inference.settings import FitSpec
    from hwoslaps.scene.builder import build_scene
    import autolens as al

    mapping = copy.deepcopy(minimal_mapping)
    population = {"type": "NFW", "concentration": {"kind": "moline2017_eq7", "x_sub": 1.0},
                  "mass_function": {"kind": "power_law", "slope": -1.9, "mass_min_msun": 5e6,
                                    "mass_max_msun": 5e8, "count": 2},
                  "spatial": {"kind": "uniform_disk", "radius_arcsec": 0.8}}
    off_plane = {"type": "PointMass", "redshift": 0.4,
                 "mass_function": {"kind": "power_law", "slope": -1.0, "mass_min_msun": 1e7,
                                   "mass_max_msun": 1e8, "count": 1},
                 "spatial": {"kind": "uniform_annulus", "inner_arcsec": 0.2, "outer_arcsec": 0.7}}
    mapping["scene"]["perturbers"] = {"populations": [population, off_plane]}
    with prepare_forecast(minimal_mapping) as baseline, prepare_forecast(mapping) as prepared:
        trial = prepared.hypothesis(1e8, (0.4, -0.6))
        fit = FitSpec(mode="fixed_template")
        no_population = prepare_case(baseline, baseline.hypothesis(1e8, (0.4, -0.6)), baseline.observation,
                                     fit=fit, use_jax=False)
        case = prepare_case(prepared, trial, prepared.observation, fit=fit, use_jax=False)
        for role in ("smooth", "subhalo"):
            model, reference = case.model(role), no_population.model(role)
            assert model.parameter_names == reference.parameter_names
            assert not any("perturber" in name for name in model.parameter_names)
            instance = autofit_model(model).instance_from_vector(vector=model.truth.tolist())
            assert [galaxy.redshift for galaxy in instance.galaxies] == [0.2, 0.4, 0.6]
            for index, halo in enumerate(prepared.scene.perturbers):
                galaxy = instance.galaxies.lens if halo.redshift == 0.2 else instance.galaxies.perturbers_0
                profile = getattr(galaxy, f"perturber_{index}")
                np.testing.assert_array_equal(profile.centre, halo.position_yx_arcsec)
            truth = build_scene(prepared.scene.spec, prepared.scene.cosmology,
                                subhalo=None if role == "smooth" else trial,
                                perturbers=prepared.scene.perturbers, assets=prepared.renderer.assets)
            tracer = al.Tracer(galaxies=list(instance.galaxies), cosmology=prepared.scene.cosmology.autogalaxy())
            np.testing.assert_allclose(tracer.image_2d_from(grid=truth.grid).native,
                                       truth.tracer.image_2d_from(grid=truth.grid).native, rtol=1e-12, atol=1e-14)
