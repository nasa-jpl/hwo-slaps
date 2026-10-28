"""Current Gaussian/objective contracts on sealed historical nonlinear inputs."""

from pathlib import Path

import numpy as np
import pytest

from hwoslaps.inference.api import prepare_case
from hwoslaps.inference.settings import FitSpec, MassSupport
from current_oracles import gaussian_likelihood_terms, write_receipt

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
    # Geometry, actual kernels and prior boxes are unchanged input contracts.
    # Padded rendering deliberately changes data/noise, recorded below.
    for values, key in ((imaging.psf.kernel.native, "psf_digest"), (np.asarray(imaging.mask).astype(float), "mask_digest")):
        assert paper_digest.array(values) == record[key], key
    assert paper_digest.kernel(imaging.psf.kernel.native) == record["psf_fit_sha256"]
    for role, oracle_role in (("smooth", "smooth"), ("subhalo", "freed")):
        model = case.model(role)
        assert list(model.parameter_names) == list(expected[f"{oracle_role}_prior_paths"])
        np.testing.assert_array_equal(model.lower, expected[f"{oracle_role}_prior_lower"])
        np.testing.assert_array_equal(model.upper, expected[f"{oracle_role}_prior_upper"])


def _assert_gaussian(case, role, vector):
    independent = gaussian_likelihood_terms(case, role, vector)
    fit = case.fit_at(role, vector)
    for term, value in independent.items():
        assert float(getattr(fit, term)) == pytest.approx(value, rel=2.e-13, abs=2.e-8), term
    assert case.log_likelihood(role, vector) == pytest.approx(independent["log_likelihood"], rel=2.e-13, abs=2.e-8)
    return independent


def test_current_nonlinear_likelihood_has_independent_gaussian_normalization(prepared, manifest, paper_digest, tmp_path, record_property):
    case = _n1(prepared, False)
    changes = {}
    # Independent consumers of the same physical truth: scene/PSF renderer
    # versus nonlinear AutoLens model construction and its blurring-grid path.
    preparation = prepared("p2_delta_knowledge_error", "reference")
    prediction_errors = {}
    for role in ("smooth", "subhalo"):
        scene = preparation.renderer.scene(subhalo=case.hypothesis if role == "subhalo" else None)
        expected_model = preparation.renderer.light_rate(scene, preparation.psfs.model_kernels)[case.data.mask]
        actual_model = np.asarray(case.fit_at(role, case.truth_vector(role)).model_data)
        np.testing.assert_allclose(actual_model, expected_model, rtol=2.e-12, atol=2.e-12)
        prediction_errors[role] = float(np.max(np.abs(actual_model-expected_model)))
    with np.load(FIXTURES/"n1_nonlinear_likelihood.npz", allow_pickle=False) as expected:
        _inputs(case, expected, manifest, paper_digest)
        for role, oracle_role in (("smooth", "smooth"), ("subhalo", "freed")):
            live = [_assert_gaussian(case, role, vector) for vector in expected[oracle_role+"_vectors"]]
            changes[role] = {term: {"historical": expected["reference__"+oracle_role+"_"+term].tolist(),
                                   "current": [row[term] for row in live]} for term in TERMS}
    record = manifest["scenes"]["n1_nonlinear_likelihood"]["dataset"]
    write_receipt(tmp_path, "nonlinear_gaussian", {"terms": changes, "cross_consumer_prediction_errors": prediction_errors, "data_identity": {
        "historical_data_digest": record["data_digest"], "current_data_digest": paper_digest.array(case.data.imaging.data.native),
        "historical_noise_digest": record["noise_map_digest"], "current_noise_digest": paper_digest.array(case.data.imaging.noise_map.native)}}, record_property)


@pytest.mark.xtx_gpu
def test_current_nonlinear_jax_objective_and_gradient_have_independent_oracles(prepared, manifest, paper_digest, tmp_path, record_property):
    case, reference = _n1(prepared, True), _n1(prepared, False)
    changes = {}
    with np.load(FIXTURES/"n1_nonlinear_likelihood.npz", allow_pickle=False) as expected:
        _inputs(case, expected, manifest, paper_digest)
        for role, oracle_role in (("smooth", "smooth"), ("subhalo", "freed")):
            model, rows = case.model(role), []
            for vector in expected[oracle_role+"_vectors"]:
                z = (vector-model.lower)/(model.upper-model.lower)
                value, gradient = case.objective(role).value_and_gradient(z)
                independent = _assert_gaussian(reference, role, vector)
                assert value == pytest.approx(.5*independent["chi_squared"], rel=2.e-11, abs=2.e-8)
                def half(point):
                    return .5*gaussian_likelihood_terms(reference, role, model.lower+point*(model.upper-model.lower))["chi_squared"]
                h = 3.e-5
                finite = []
                for index in range(len(z)):
                    step = np.zeros_like(z); step[index] = h
                    finite.append((half(z-2*step)-8*half(z-step)+8*half(z+step)-half(z+2*step))/(12*h))
                np.testing.assert_allclose(gradient, finite, rtol=3.e-5, atol=2.e-4)
                rows.append({"half_chi2": float(value), "gradient_max_finite_difference_error": float(np.max(np.abs(gradient-finite)))})
            changes[role] = {"historical_half_chi2": expected["jax_gpu__"+oracle_role+"_half_chi2"].tolist(), "current": rows}
    write_receipt(tmp_path, "nonlinear_jax", changes, record_property)


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


@pytest.mark.xtx_gpu
@pytest.mark.parametrize("workload", ["B4a", "B4b"])
def test_current_fit_refinement_beats_archived_candidate_on_current_data(workload, tmp_path, record_property):
    import json
    from hwoslaps.fisher.api import Execution, prepare_forecast
    from hwoslaps.inference.api import validate_nonlinear
    from hwoslaps.inference.settings import SamplerSettings, RefineSettings
    from hwoslaps.simulation import simulate

    with (FIXTURES / "inference/fit_engine_8fa6209.json").open() as stream:
        expected = json.load(stream)["cases"][workload]
    scene = "p1_optical_matched" if workload == "B4a" else "p2_delta_knowledge_error"
    with prepare_forecast(FIXTURES / "engine" / f"{scene}.yaml", execution=Execution(engine="reference")) as prepared:
        trial = prepared.hypothesis(1e9, (0.4, -0.8))
        observation = simulate(prepared, subhalo=trial, noise_seed=None if workload == "B4a" else 11)
        fit = FitSpec(mode="fixed_template") if workload == "B4a" else FitSpec(mode="freed", mass_support=MassSupport(6, 9.7))
        sampler = SamplerSettings(n_live_smooth=50, n_live_subhalo_fixed=50, n_live_subhalo_search=80, n_eff=200,
                                  n_shell=1, f_live=0.01, n_like_max=12000, discard_exploration=False, use_jax=True,
                                  jax_n_batch=50, retain_search_internal=True)
        result = validate_nonlinear(prepared, trial, observation, fit=fit, sampler=sampler, sampler_seed=20261005,
                                    refine=RefineSettings(), output_dir=tmp_path, case_id=workload)
        current_case = prepare_case(prepared, trial, observation, fit=fit, use_jax=False)
        comparisons = {}
        for role in ("smooth", "subhalo"):
            actual, archived = result.role(role), expected["roles"][role]
            assert actual.status == "success" and actual.refinement is not None
            assert all(actual.refinement.gates.values()), actual.refinement.gates
            retained = _assert_gaussian(current_case, role, actual.refinement.best_vector)["log_likelihood"]
            archived_vector = np.asarray(archived["candidate_best_vector"])
            old_point = _assert_gaussian(current_case, role, archived_vector)["log_likelihood"]
            truth = current_case.log_likelihood(role, current_case.truth_vector(role))
            assert retained == pytest.approx(actual.log_likelihood, abs=2.e-8)
            assert retained >= max(old_point, truth, actual.sampler.log_likelihood_max)-.1
            assert actual.sampler.diagnostics.achieved["n_like"] <= 12000+sampler.jax_n_batch
            comparisons[role] = {"historical_maximum": archived["refined_log_likelihood_max"],
                                 "historical_point_on_current_data": old_point, "current_maximum": retained,
                                 "optimization_acceptance": actual.acceptance_status.value, "quality": actual.quality,
                                 "sampling_diagnostics": actual.sampler.diagnostics.to_mapping()}
        q = 2*(comparisons["subhalo"]["current_maximum"]-comparisons["smooth"]["current_maximum"])
        assert result.q_signed == pytest.approx(q, abs=5.e-8)
        if workload == "B4b" and result.recovery.pdf_usable:
            recovery = result.recovery
            for name in ("log10_mass_quantiles", "centre_y_quantiles", "centre_x_quantiles"):
                quantiles = getattr(recovery, name)
                assert quantiles is not None and np.all(np.diff(quantiles) >= 0)
            assert fit.mass_support.log10_mass_min <= recovery.log10_mass_quantiles[0]
            assert recovery.log10_mass_quantiles[-1] <= fit.mass_support.log10_mass_max
    write_receipt(tmp_path, workload, {"roles": comparisons, "historical_q": expected["q"], "current_q": q,
                                      "scope": "Bounded current fit; archived endpoint is an independently rescored candidate, not an expected optimum"}, record_property)
