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


@pytest.mark.xtx_gpu
@pytest.mark.parametrize("workload", ["B4a", "B4b"])
def test_fit_engine_reproduces_base_tree(workload, tmp_path):
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
                                  n_shell=1, f_live=0.01, discard_exploration=False, use_jax=True,
                                  jax_n_batch=50, retain_search_internal=True)
        result = validate_nonlinear(prepared, trial, observation, fit=fit, sampler=sampler, sampler_seed=20261005,
                                    refine=RefineSettings(), output_dir=tmp_path, case_id=workload)
    report = result.to_mapping()
    for role in ("smooth", "subhalo"):
        actual, oracle = result.role(role), expected["roles"][role]
        assert actual.status == oracle["status"]
        assert actual.sampler.log_likelihood_max == oracle["sampler_log_likelihood_max"]
        assert actual.log_likelihood == oracle["refined_log_likelihood_max"]
        assert actual.refinement.best_log_likelihood == oracle["candidate_best_log_likelihood"]
        assert list(actual.refinement.best_vector) == oracle["candidate_best_vector"]
        assert list(actual.refinement.record["candidate_best_z"]) == oracle["candidate_best_z"]
        assert actual.acceptance_status.value == oracle["candidate_acceptance_status"]
    assert result.q_signed == expected["q"]
    assert result.q_signed * 0.5 == expected["signed_delta_log_l"]
    if workload == "B4b":
        old, recovery = expected["recovery"], result.recovery
        assert recovery.sampler.log10_mass == old["log10_m200_ml"]
        assert recovery.sampler.centre_yx == (old["centre_ml_y"], old["centre_ml_x"])
        assert recovery.sampler.profile_scales == {"kappa_s": old["kappa_s_ml"],
                                                   "scale_radius": old["scale_radius_arcsec_ml"]}
        assert recovery.log10_mass_quantiles == tuple(old[f"log10_m200_p{p}"] for p in (16, 50, 84))
        assert recovery.centre_y_quantiles == tuple(old[f"centre_y_p{p}"] for p in (16, 50, 84))
        assert recovery.centre_x_quantiles == tuple(old[f"centre_x_p{p}"] for p in (16, 50, 84))
        assert recovery.pdf_converged == old["pdf_converged"] and recovery.sample_count == old["n_samples"]
    report["projected_gradients"] = {role: result.role(role).refinement.projected_gradient_linf
                                      for role in ("smooth", "subhalo")}
    (tmp_path / f"{workload}.report.json").write_text(json.dumps(report, indent=2) + "\n")
