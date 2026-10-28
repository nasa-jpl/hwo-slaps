"""Independent Gaussian and multimodal controls for normalized 0/1-D inference."""

import json
import math
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.optimize import brentq
from scipy.special import logsumexp
from scipy.stats import norm

from hwoslaps.inference import low_dimension, roles
from hwoslaps.inference.low_dimension import exact_likelihood, scalar_likelihood
from hwoslaps.inference.result import RetentionInventory, RoleFit, RoleStatus, SamplerRecord
from hwoslaps.inference.sampler import SearchOutcome
from hwoslaps.inference.settings import FitSpec, MassSupport, RefineSettings, SamplerSettings


def test_exact_evaluation_is_the_normalized_point_prior():
    outcome = exact_likelihood(8.25)
    assert outcome.best_vector == ()
    assert outcome.best_log_likelihood == outcome.record["log_evidence"] == 8.25
    assert outcome.record["optimization_accepted"] and outcome.record["integration_complete"]
    assert outcome.record["likelihood_calls"] == 1


@pytest.mark.parametrize("bounds", [(0., 4.), (6., 10.)])
def test_gaussian_scalar_maximum_evidence_and_quantiles(bounds):
    lower, upper = bounds
    mu, sigma, peak = lower + 1.3, .2, 5.
    outcome = scalar_likelihood(lambda x: peak - .5 * ((x[0] - mu) / sigma) ** 2,
                                lower, upper, parameter_name="declared_coordinate")
    truncated_mass = norm.cdf((upper - mu) / sigma) - norm.cdf((lower - mu) / sigma)
    expected_log_evidence = peak + math.log(sigma * math.sqrt(2 * math.pi) * truncated_mass / (upper - lower))
    probabilities = norm.cdf((lower - mu) / sigma) + np.array([.16, .5, .84]) * truncated_mass
    expected_quantiles = mu + sigma * norm.ppf(probabilities)
    assert outcome.best_vector == pytest.approx((mu,), abs=2.e-7)
    assert outcome.best_log_likelihood == pytest.approx(peak, abs=1.e-10)
    assert outcome.record["optimization_accepted"] and outcome.record["integration_complete"]
    assert outcome.record["log_evidence"] == pytest.approx(expected_log_evidence, abs=1.e-9)
    assert outcome.record["posterior_quantiles"]["declared_coordinate"] == pytest.approx(expected_quantiles, abs=1.e-9)


def test_bimodal_scalar_search_and_integral_include_the_separated_narrow_mode():
    centres, sigmas, weights = np.array([.23, .813]), np.array([.06, .009]), np.array([.3, .7])

    def likelihood(x):
        return logsumexp(np.log(weights) - np.log(sigmas * math.sqrt(2 * math.pi))
                         - .5 * ((x[0] - centres) / sigmas) ** 2)

    def cdf(x):
        return np.sum(weights * norm.cdf((x - centres) / sigmas))

    outcome = scalar_likelihood(likelihood, 0., 1., parameter_name="x")
    evidence = cdf(1.) - cdf(0.)
    expected_quantiles = [brentq(lambda x: (cdf(x) - cdf(0.)) / evidence - p, 0., 1.)
                          for p in (.16, .5, .84)]
    assert outcome.best_vector[0] == pytest.approx(centres[1], abs=2.e-7)
    assert outcome.record["log_evidence"] == pytest.approx(math.log(evidence), abs=1.e-9)
    assert outcome.record["posterior_quantiles"]["x"] == pytest.approx(expected_quantiles, abs=1.e-9)
    assert outcome.record["optimization_accepted"] and outcome.record["integration_complete"]
    assert all(level["sampled_basins"] == 2 for level in outcome.record["meshes"])


@pytest.mark.parametrize("sign", [-1., 1.])
def test_scalar_endpoint_maximum_and_prior_coordinate_scale(sign):
    first = scalar_likelihood(lambda x: sign * 7. * x[0], 0., 1., parameter_name="x")
    # An affine relabeling preserves the prior; multiplying by its width would be wrong.
    second = scalar_likelihood(lambda x: sign * 7. * ((x[0] - 6.) / 4.), 6., 10., parameter_name="log_mass")
    expected = math.log(abs(math.expm1(sign * 7.) / (sign * 7.)))
    assert first.best_vector == ((0.,) if sign < 0 else (1.,))
    assert first.record["log_evidence"] == pytest.approx(expected, abs=1.e-10)
    assert second.record["log_evidence"] == pytest.approx(first.record["log_evidence"], abs=1.e-10)
    assert second.record["posterior_quantiles"]["log_mass"] == pytest.approx(
        6. + 4. * np.asarray(first.record["posterior_quantiles"]["x"]), abs=1.e-9)


def test_scalar_optimization_is_separate_from_posterior_integration(monkeypatch):
    def refused(*args, **kwargs):
        raise ValueError("quadrature cannot establish its accuracy")

    monkeypatch.setattr(low_dimension, "quad", refused)
    outcome = scalar_likelihood(lambda x: -(x[0] - .4) ** 2, 0., 1., parameter_name="x")
    assert outcome.record["optimization_accepted"]
    assert not outcome.record["integration_complete"]
    assert outcome.record["log_evidence"] is None
    assert "cannot establish" in outcome.record["integration_error"]
    maximum_only = scalar_likelihood(lambda x: -(x[0] - .4) ** 2, 0., 1., parameter_name="x", integrate=False)
    assert maximum_only.record["optimization_accepted"]
    assert maximum_only.record["posterior_quantiles"] is None


def test_flat_scalar_likelihood_has_uniform_posterior_and_normalized_evidence():
    outcome = scalar_likelihood(lambda x: 9., -3., 7., parameter_name="x")
    assert outcome.record["log_evidence"] == pytest.approx(9., abs=1.e-12)
    assert outcome.record["posterior_quantiles"]["x"] == pytest.approx((-1.4, 2., 5.4), abs=1.e-10)
    assert outcome.record["optimization_accepted"] and outcome.record["integration_complete"]


@pytest.mark.parametrize("mu", [.0009, .9991])
@pytest.mark.parametrize("integrate", [False, True])
def test_broad_edge_gaussian_is_refined_inside_both_endpoint_intervals(mu, integrate):
    sigma = .05
    outcome = scalar_likelihood(lambda x: -.5 * ((x[0] - mu) / sigma) ** 2,
                                0., 1., parameter_name="x", integrate=integrate)
    assert outcome.best_vector[0] == pytest.approx(mu, abs=2.e-8)
    assert outcome.best_log_likelihood == pytest.approx(0., abs=1.e-10)
    assert outcome.record["optimization_accepted"]
    if integrate:
        expected = math.log(sigma * math.sqrt(2. * math.pi)
                            * (norm.cdf((1. - mu) / sigma) - norm.cdf(-mu / sigma)))
        assert outcome.record["integration_complete"]
        assert outcome.record["log_evidence"] == pytest.approx(expected, abs=1.e-9)


@pytest.mark.parametrize("integrate", [False, True])
def test_known_high_incumbent_is_preserved_and_exposes_unresolved_structure(integrate):
    centre = .371234567

    def likelihood(values):
        return float(np.logaddexp(0., 50. - .5 * ((values[0] - centre) / 1.e-7) ** 2))

    outcome = scalar_likelihood(likelihood, 0., 1., parameter_name="x", integrate=integrate,
                                initial_evaluation=(centre, likelihood((centre,))))
    assert outcome.best_vector == (centre,)
    assert outcome.best_log_likelihood == 50.
    assert not outcome.record["optimization_accepted"]
    assert not outcome.record["integration_complete"]
    assert outcome.record["known_incumbent_exceeds_resolved_mesh"]
    assert outcome.record["posterior_quantiles"] is None and outcome.record["log_evidence"] is None


@pytest.mark.parametrize("budget", [1, 10, 150])
def test_scalar_hard_cap_preserves_partial_maximum_and_counts_every_new_call(budget):
    calls = []

    def likelihood(values):
        calls.append(values[0])
        return -(values[0] - .4) ** 2

    outcome = scalar_likelihood(likelihood, 0., 1., parameter_name="x", max_likelihood_calls=budget)
    assert len(calls) == outcome.record["likelihood_calls"] == budget
    assert outcome.best_log_likelihood == max(-(x - .4) ** 2 for x in calls)
    assert not outcome.record["optimization_accepted"] and not outcome.record["integration_complete"]
    assert outcome.record["budget_exhausted"] and outcome.record["budget_exhausted_stage"] == "optimization"


def test_quadrature_budget_exhaustion_does_not_invalidate_an_accepted_maximum():
    objective = lambda values: -.5 * ((values[0] - .4) / .1) ** 2
    maximum = scalar_likelihood(objective, 0., 1., parameter_name="x", integrate=False)
    budget = maximum.record["likelihood_calls"] + 10
    calls = []

    def counted(values):
        calls.append(values[0])
        return objective(values)

    outcome = scalar_likelihood(counted, 0., 1., parameter_name="x", max_likelihood_calls=budget)
    assert len(calls) == outcome.record["likelihood_calls"] == budget
    assert outcome.record["optimization_accepted"] and not outcome.record["integration_complete"]
    assert outcome.record["budget_exhausted_stage"] == "quadrature"


@pytest.mark.parametrize("dimension", [0, 1])
def test_capped_role_counts_truth_with_search_and_retains_the_known_incumbent(dimension, tmp_path):
    calls = []
    model = SimpleNamespace(parameter_names=("x",) if dimension else (), truth=np.full(dimension, .5),
                            lower=np.zeros(dimension), upper=np.ones(dimension))

    def likelihood(role, vector):
        value = 3. - sum((x - .4) ** 2 for x in vector)
        calls.append((tuple(vector), value))
        return value

    case = SimpleNamespace(model=lambda role: model, truth_vector=lambda role: model.truth,
                           log_likelihood=likelihood)
    budget = 10 if dimension else 1
    fitted, outcome = roles.run_search_role(case, "smooth", sampler=SamplerSettings(n_like_max=budget),
                                           sampler_seed=7, refine=None, session=None, case_dir=tmp_path, case_id="case")
    assert outcome is None and len(calls) == fitted.strategy_record["likelihood_calls"] == budget
    assert fitted.log_likelihood == max(value for _, value in calls)
    assert fitted.strategy_record["precomputed_likelihood_calls"] == 1
    assert fitted.strategy_record["likelihood_function_calls"] == budget - 1
    assert fitted.strategy_record["call_count_convention"]["external_preflight_in_budget"] is False
    if dimension:
        assert fitted.acceptance_status is RoleStatus.UNRESOLVED
        assert fitted.strategy_record["stop_reason"] == "likelihood_budget_exhausted"
        assert fitted.evidence_quality != "complete" and fitted.posterior_quality != "complete"
    else:
        assert fitted.acceptance_status is RoleStatus.EXACT


@pytest.mark.parametrize("dimension", [0, 1])
def test_roles_dispatch_without_sampler_or_refinement_and_round_trip(dimension, tmp_path, monkeypatch):
    model = SimpleNamespace(parameter_names=tuple(["x"] * dimension), truth=np.full(dimension, .5),
                            lower=np.zeros(dimension), upper=np.ones(dimension))
    calls = []

    def likelihood(role, vector):
        calls.append((role, tuple(vector)))
        return 3. - sum((value - .4) ** 2 for value in vector)

    def forbidden(**kwargs):
        raise AssertionError("zero/scalar dimensions must never launch Nautilus")

    monkeypatch.setattr(roles, "run_search", forbidden)
    case = SimpleNamespace(model=lambda role: model, truth_vector=lambda role: model.truth, log_likelihood=likelihood)
    result, searched = roles.run_search_role(case, "smooth", sampler=SamplerSettings(), sampler_seed=7,
                                             refine=RefineSettings(), session=None, case_dir=tmp_path, case_id="case")
    assert searched is None and result.sampler is None and result.refinement is None
    assert result.strategy == ("exact" if dimension == 0 else "scalar")
    assert result.status == "success" and result.strategy_record["optimization_accepted"]
    if dimension == 0:
        assert len(calls) == 1
    loaded = RoleFit.from_mapping(json.loads(json.dumps(result.to_mapping())))
    assert loaded.acceptance_status is (RoleStatus.EXACT if dimension == 0 else RoleStatus.SCALAR)
    assert loaded.log_evidence == result.log_evidence
    assert loaded.strategy_record == result.strategy_record


@pytest.mark.parametrize("dimension", [0, 1])
@pytest.mark.parametrize("mode", ["local_search", "freed"])
def test_each_mixed_dimension_role_dispatches_independently(dimension, mode, tmp_path, monkeypatch):
    from hwoslaps.analysis.nonlinear import RoleAcceptance, case_status

    models = {role: SimpleNamespace(parameter_names=tuple(f"x{i}" for i in range(count)),
                                   truth=np.full(count, .5), lower=np.zeros(count), upper=np.ones(count))
              for role, count in (("smooth", dimension), ("subhalo", dimension + (2 if mode == "local_search" else 3)))}
    searched_roles = []

    def search(**kwargs):
        searched_roles.append(kwargs["role"])
        record = SamplerRecord(name="subhalo", output_path="search", identity="ab" * 32, n_live=20,
                               requested={}, effective={}, seed=1, training_workers=1, log_likelihood_max=3.,
                               log_evidence=2., likelihood_calls=10, retention=None, runtime_s=1.)
        result = SimpleNamespace(samples=SimpleNamespace(max_log_likelihood=lambda **kw: models["subhalo"].truth))
        return SearchOutcome(record, result, None)

    monkeypatch.setattr(roles, "run_search", search)
    case = SimpleNamespace(model=lambda role: models[role], truth_vector=lambda role: models[role].truth,
                           log_likelihood=lambda role, vector: 3. - sum((value - .4) ** 2 for value in vector),
                           autofit_models={"subhalo": object()}, analysis=object(), data=SimpleNamespace(record={}),
                           fit=FitSpec(mode=mode, mass_support=MassSupport(6., 10.) if mode == "freed" else None))
    fitted_roles = {}
    for role in ("smooth", "subhalo"):
        fitted, _ = roles.run_search_role(case, role, sampler=SamplerSettings(), sampler_seed=7,
                                          refine=None, session=None, case_dir=tmp_path, case_id="case")
        assert fitted.status == "success"
        assert fitted.strategy == (("exact" if dimension == 0 else "scalar") if role == "smooth" else "search")
        fitted_roles[role] = fitted
    assert searched_roles == ["subhalo"]
    acceptance = RoleAcceptance(frozenset([RoleStatus.ACCEPTED]), frozenset([RoleStatus.SAMPLER_ONLY]))
    result = SimpleNamespace(role=lambda role: fitted_roles[role])
    checked = case_status(result, acceptance=acceptance, require_retained_state=True, stationarity_tolerance=None)
    assert checked.status == "incomplete"
    assert len(checked.reasons) == 1 and "subhalo" in checked.reasons[0]
    # Only the searched H1 requires Nautilus files. H0's deterministic state is inline.
    retention = RetentionInventory("directory", {"search_internal.dill": {"bytes": 1}}, (), True)
    fitted_roles["subhalo"] = replace(fitted_roles["subhalo"], sampler=replace(
        fitted_roles["subhalo"].sampler, retention=retention))
    checked = case_status(result, acceptance=acceptance, require_retained_state=True, stationarity_tolerance=None)
    assert checked.status == "accepted"


@pytest.mark.backend
@pytest.mark.parametrize("dimension", [0, 1])
def test_public_role_budget_reports_preflight_separately(prepared_forecast_factory, dimension, tmp_path, monkeypatch):
    from hwoslaps.inference.api import PreparedCase, validate_nonlinear
    from hwoslaps.scene.parameters import scene_parameters
    from hwoslaps.simulation import simulate

    original = prepared_forecast_factory()
    fixed = [parameter.name for parameter in scene_parameters(original.scene.spec)
             if not dimension or parameter.name != "source.light.light.intensity"]
    prepared = prepared_forecast_factory({"forecast": {"nuisances": {"fixed": fixed, "background_offset": False}}})
    trial = prepared.hypothesis(1.e8, (.4, -.6))
    observation = simulate(prepared, subhalo=trial, noise_seed=None)
    calls = {"smooth": 0, "subhalo": 0}
    direct = PreparedCase.log_likelihood

    def counted(case, role, vector):
        calls[role] += 1
        return direct(case, role, vector)

    monkeypatch.setattr(PreparedCase, "log_likelihood", counted)
    budget = 10 if dimension else 1
    result = validate_nonlinear(prepared, trial, observation, fit=FitSpec(mode="fixed_template"),
                                sampler=SamplerSettings(n_like_max=budget), sampler_seed=7, output_dir=tmp_path)
    for role in ("smooth", "subhalo"):
        fitted = result.role(role)
        assert calls[role] == budget + 1  # one public preflight, then the capped role
        assert fitted.strategy_record["likelihood_calls"] == budget
        if dimension:
            assert fitted.acceptance_status is RoleStatus.UNRESOLVED
            assert fitted.evidence_quality != "complete" and fitted.posterior_quality != "complete"
        else:
            assert fitted.acceptance_status is RoleStatus.EXACT


@pytest.mark.backend
@pytest.mark.parametrize("dimension", [0, 1])
@pytest.mark.parametrize("h1", ["search", "truth_anchor"])
def test_public_fixed_morphology_validation_without_sampler(prepared_forecast_factory, dimension, h1,
                                                            tmp_path, monkeypatch):
    from hwoslaps.inference.api import prepare_case, validate_nonlinear
    from hwoslaps.scene.parameters import scene_parameters
    from hwoslaps.simulation import simulate

    with prepared_forecast_factory() as original:
        fixed = [parameter.name for parameter in scene_parameters(original.scene.spec)
                 if not dimension or parameter.name != "source.light.light.intensity"]
    prepared = prepared_forecast_factory({"forecast": {"nuisances": {"fixed": fixed, "background_offset": False}}})
    trial = prepared.hypothesis(1.e8, (.4, -.6))
    observation = simulate(prepared, subhalo=trial, noise_seed=None)
    fit = FitSpec(mode="fixed_template", h1=h1)
    case = prepare_case(prepared, trial, observation, fit=fit, use_jax=False)
    assert len(case.model("smooth").parameter_names) == dimension

    def forbidden(**kwargs):
        raise AssertionError("Nautilus invoked on a zero/one-dimensional model")

    monkeypatch.setattr(roles, "run_search", forbidden)
    result = validate_nonlinear(prepared, trial, observation, fit=fit, sampler=SamplerSettings(),
                                sampler_seed=7, output_dir=tmp_path)
    assert result.q_signed is not None and math.isfinite(result.q_signed)
    assert result.smooth.strategy == ("exact" if dimension == 0 else "scalar")
    assert result.smooth.strategy_record["optimization_accepted"]
    if h1 == "search":
        assert result.subhalo.strategy == result.smooth.strategy
        assert result.delta_log_evidence == pytest.approx(result.subhalo.log_evidence - result.smooth.log_evidence)
    else:
        assert result.subhalo.strategy == "truth_anchor"
    if dimension == 0:
        exact_q = 2. * (case.log_likelihood("subhalo", ()) - case.log_likelihood("smooth", ()))
        assert result.q_signed == pytest.approx(exact_q, abs=1.e-10)
