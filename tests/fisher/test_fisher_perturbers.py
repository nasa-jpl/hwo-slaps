"""Population realization transported through prepared reference and JAX forecasts."""

import copy

import numpy as np
import pytest

pytestmark = pytest.mark.backend


def _population(*, plane=None, count=5):
    return {"type": "NFW", "concentration": {"kind": "moline2017_eq7", "x_sub": 1.0},
            "mass_function": {"kind": "power_law", "slope": -1.9, "mass_min_msun": 5e6,
                              "mass_max_msun": 5e8, "count": count},
            "spatial": {"kind": "uniform_disk", "radius_arcsec": 0.8}, "redshift": plane}


def test_lens_plane_population_jax_matches_reference(minimal_mapping):
    from hwoslaps.fisher.api import Execution, forecast, prepare_forecast

    minimal_mapping["scene"]["perturbers"] = {"populations": [_population()]}
    with prepare_forecast(minimal_mapping) as reference, \
            prepare_forecast(minimal_mapping, execution=Execution(engine="jax")) as candidate:
        assert len(reference.scene.perturbers) == len(candidate.scene.perturbers) == 5
        assert reference.scene.perturbers == candidate.scene.perturbers
        expected = forecast(reference, masses_msun=[1e8])
        actual = forecast(candidate, masses_msun=[1e8])
    np.testing.assert_allclose(actual.fisher_raw, expected.fisher_raw, rtol=5e-6, atol=0.)
    np.testing.assert_allclose(actual.fisher_profiled, expected.fisher_profiled, rtol=5e-6, atol=0.)


def test_off_plane_population_runs_on_reference_and_jax_refuses(minimal_mapping):
    from hwoslaps.fisher.api import Execution, forecast, prepare_forecast

    population = _population(plane=0.4, count=2)
    population["concentration"] = {"kind": "fixed", "value": 10.0}
    minimal_mapping["scene"]["perturbers"] = {"populations": [population]}
    with prepare_forecast(minimal_mapping) as reference:
        result = forecast(reference, masses_msun=[1e8])
        assert all(halo.redshift == 0.4 for halo in reference.scene.perturbers)
        assert np.all(np.isfinite(result.fisher_raw)) and np.any(result.fisher_profiled > 0.)
    with pytest.raises(ValueError, match="plane"):
        prepare_forecast(minimal_mapping, execution=Execution(engine="jax"))


def test_off_plane_hypothesis_tends_to_the_lens_plane_forecast(minimal_mapping):
    from hwoslaps.fisher.api import forecast, prepare_forecast

    base = copy.deepcopy(minimal_mapping)
    base["scene"]["subhalo"]["concentration"] = {"kind": "fixed", "value": 10.0}
    positions = [[y, x] for y in (-0.4, 0.0, 0.4) for x in (-0.4, 0.0, 0.4)]
    base["forecast"]["positions"] = {"kind": "explicit", "positions_yx": positions}
    nearby = copy.deepcopy(base)
    nearby["scene"]["subhalo"]["redshift"] = base["scene"]["lens"]["redshift"] + 1e-6
    with prepare_forecast(base) as same_plane, prepare_forecast(nearby) as off_plane:
        expected = forecast(same_plane, masses_msun=[1e8])
        actual = forecast(off_plane, masses_msun=[1e8])
    np.testing.assert_allclose(actual.q_asimov, expected.q_asimov, rtol=1e-4, atol=0.)


def test_preparation_keeps_and_records_realized_populations(minimal_mapping):
    from hwoslaps.fisher.api import forecast, prepare_forecast
    from hwoslaps.scene.perturbers import realize_perturbers

    minimal_mapping["scene"]["perturbers"] = {"populations": [_population()]}
    with prepare_forecast(minimal_mapping) as prepared:
        original = tuple(halo.to_mapping() for halo in prepared.scene.perturbers)
        expected = realize_perturbers(prepared.scene.spec, prepared.scene.cosmology, seed=minimal_mapping["seed"])
        assert prepared.scene.perturbers == expected
        first = forecast(prepared, masses_msun=[1e8])
        second = forecast(prepared, masses_msun=[1e8])
        assert tuple(first.provenance["perturbers"]) == original
        assert tuple(second.provenance["perturbers"]) == original
        assert tuple(halo.to_mapping() for halo in prepared.scene.perturbers) == original
        np.testing.assert_array_equal(first.fisher_profiled, second.fisher_profiled)
