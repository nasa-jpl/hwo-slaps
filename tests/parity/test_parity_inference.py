"""Paper likelihood inputs and objective through the final prepared-case API, plus fit-engine anchors."""

from pathlib import Path

import numpy as np
import pytest

from hwoslaps.inference.api import prepare_case
from hwoslaps.inference.settings import FitSpec, MassSupport

pytestmark = pytest.mark.backend
FIXTURES = Path(__file__).resolve().parents[1] / "fixtures/paper_parity"
TERMS = ("figure_of_merit", "log_likelihood", "chi_squared", "noise_normalization")


def _n1(prepared, use_jax):
    from hwoslaps.simulation import simulate

    forecast = prepared("p2_delta_knowledge_error", "reference")
    trial = forecast.hypothesis(1.0e9, (0.4, -0.8))
    observation = simulate(forecast, subhalo=trial, noise_seed=11)
    return prepare_case(forecast, trial, observation, fit=FitSpec(mode="freed", mass_support=MassSupport(6.0, 9.7)),
                        use_jax=use_jax)


def _inputs(case, expected, manifest, paper_digest):
    entry = manifest["scenes"]["n1_nonlinear_likelihood"]
    lensing = case.hypothesis.lensing()
    assert lensing.parameters["kappa_s"] == entry["trial"]["kappa_s"]
    assert lensing.parameters["scale_radius"] == entry["trial"]["scale_radius_arcsec"]
    assert lensing.derived["concentration"] == entry["trial"]["concentration"]
    assert case.hypothesis.redshift == entry["trial"]["lens_redshift"]
    assert case.hypothesis.source_redshift == entry["trial"]["source_redshift"]
    imaging = case.data.imaging
    record = entry["dataset"]
    assert case.data.pixel_count == record["n_unmasked_pixels"]
    for values, key in ((imaging.data.native, "data_digest"), (imaging.noise_map.native, "noise_map_digest"),
                        (imaging.psf.kernel.native, "psf_digest"), (np.asarray(imaging.mask).astype(float), "mask_digest")):
        assert paper_digest.array(values) == record[key], key
    assert paper_digest.kernel(imaging.psf.kernel.native) == record["psf_fit_sha256"]
    for role, oracle_role in (("smooth", "smooth"), ("subhalo", "freed")):
        model = case.model(role)
        assert list(model.parameter_names) == list(expected[f"{oracle_role}_prior_paths"])
        np.testing.assert_array_equal(model.lower, expected[f"{oracle_role}_prior_lower"])
        np.testing.assert_array_equal(model.upper, expected[f"{oracle_role}_prior_upper"])


def test_nonlinear_inputs_and_likelihood_reproduce_paper(prepared, manifest, paper_digest):
    case = _n1(prepared, False)
    with np.load(FIXTURES / "n1_nonlinear_likelihood.npz", allow_pickle=False) as expected:
        _inputs(case, expected, manifest, paper_digest)
        for role, oracle_role in (("smooth", "smooth"), ("subhalo", "freed")):
            vectors = expected[f"{oracle_role}_vectors"]
            np.testing.assert_array_equal([case.log_likelihood(role, vector) for vector in vectors],
                                          expected[f"reference__{oracle_role}_log_likelihood_function"])
            for term in TERMS:
                np.testing.assert_array_equal([float(getattr(case.fit_at(role, vector), term)) for vector in vectors],
                                              expected[f"reference__{oracle_role}_{term}"])


@pytest.mark.xtx_gpu
def test_nonlinear_jax_objective_reproduces_paper(prepared, manifest, paper_digest):
    case = _n1(prepared, True)
    with np.load(FIXTURES / "n1_nonlinear_likelihood.npz", allow_pickle=False) as expected:
        _inputs(case, expected, manifest, paper_digest)
        for role, oracle_role in (("smooth", "smooth"), ("subhalo", "freed")):
            model = case.model(role)
            values = [case.objective(role).value_and_gradient((vector - model.lower) / (model.upper - model.lower))
                      for vector in expected[f"{oracle_role}_vectors"]]
            np.testing.assert_array_equal([value for value, _ in values], expected[f"jax_gpu__{oracle_role}_half_chi2"])
            np.testing.assert_array_equal([gradient for _, gradient in values], expected[f"jax_gpu__{oracle_role}_half_chi2_gradient"])


@pytest.mark.xtx_gpu
@pytest.mark.parametrize("batch_size", [8, 100])
def test_batched_fitness_equals_scalar_likelihood(batch_size, prepared):
    from autofit.non_linear.fitness import Fitness

    case = _n1(prepared, True)
    for role, oracle_role in (("smooth", "smooth"), ("subhalo", "freed")):
        model = case.model(role)
        vectors = model.lower + np.random.default_rng(20261005).random((batch_size, len(model.parameter_names))) \
            * (model.upper - model.lower)
        fitness = Fitness(model=case.autofit_models[role], analysis=case.analysis, paths=None,
                          fom_is_log_likelihood=True, resample_figure_of_merit=-1e99,
                          use_jax_vmap=True, batch_size=batch_size)
        scalar = np.array([case.log_likelihood(role, vector) for vector in vectors])
        assert np.max(np.abs(np.asarray(fitness.call_wrap(vectors)) - scalar)) <= 1e-4
