"""A+B dependency: singular brightness integrals use the same padded emitting sky."""

from copy import deepcopy

import numpy as np
import pytest

pytestmark = pytest.mark.backend


@pytest.mark.parametrize("kind", ["PointMass", "SIS"])
def test_singular_padding_common_crop_and_public_fit_share_pixel_integrals(kind, minimal_mapping, tmp_path):
    from hwoslaps.fisher.api import Execution, forecast, prepare_forecast
    from hwoslaps.inference.api import prepare_case
    from hwoslaps.inference.settings import FitSpec
    from hwoslaps.simulation import simulate

    kernel = np.arange(1.0, 16.0).reshape(3, 5)
    kernel /= kernel.sum()
    path = tmp_path / "asymmetric.npy"
    np.save(path, kernel)
    minimal_mapping["scene"]["grid"] = {"shape": [9, 11], "pixel_scale_arcsec": 0.1, "over_sample_size": 1}
    minimal_mapping["scene"]["lens"]["mass"]["mass"].update(centre=[0.007, -0.013], einstein_radius=0.18)
    minimal_mapping["scene"]["source"]["light"]["light"].update(type="Sersic", sersic_index=0.5,
                                                                          effective_radius=0.2)
    minimal_mapping["scene"]["subhalo"] = {"type": kind}
    minimal_mapping["psf"] = {"truth": {"kind": "kernel", "path": str(path), "pixel_scale_arcsec": 0.1}}
    minimal_mapping["forecast"]["positions"] = {"kind": "explicit", "positions_yx": [[0.025, -0.035]]}
    minimal_mapping["forecast"]["nuisances"] = {"fixed": ["*"], "background_offset": False}
    with prepare_forecast(minimal_mapping, execution=Execution(engine="reference")) as prepared:
        halo = prepared.hypothesis(1e8, (0.025, -0.035))
        observation = simulate(prepared, subhalo=halo, noise_seed=None)
        larger = deepcopy(minimal_mapping)
        larger["scene"]["grid"]["shape"] = [15, 17]
        big = simulate(larger, subhalo=halo, noise_seed=None)
        np.testing.assert_allclose(observation.light_rate_e_per_s, big.light_rate_e_per_s[3:12, 3:14],
                                   rtol=2e-10, atol=1e-12)
        for role, data in (("smooth", prepared.observation), ("subhalo", observation)):
            case = prepare_case(prepared, halo, data, fit=FitSpec(mode="fixed_template"), use_jax=False)
            fit = case.fit_at(role, case.truth_vector(role))
            assert np.isfinite(fit.log_likelihood) and fit.chi_squared < 1e-12
        expected = forecast(prepared, masses_msun=[1e8])
    with prepare_forecast(minimal_mapping, execution=Execution(engine="jax")) as requested_jax:
        actual = forecast(requested_jax, masses_msun=[1e8])
        np.testing.assert_array_equal(actual.q_asimov, expected.q_asimov)
        response = actual.provenance["comparison_inputs"]["response"]
        assert response == case.data.record["comparison_inputs"]["response"]
        assert response["operator"] == "native_binned_psf_padded_v1"
        assert response["light_integration"]["source"]["policy"] == "native-brightness-adaptive-v1"
        assert response["light_integration"]["source"]["max_cells"] == 1024
        assert response["light_integration"]["lens"]["policy"] == "uniform_subpixel_mean"
        route = requested_jax.engine.describe()
        assert route["kind"] == "reference" and route["requested_kind"] == "jax"
        assert route["jax_accelerated"] is False and route["device"] == "cpu"
