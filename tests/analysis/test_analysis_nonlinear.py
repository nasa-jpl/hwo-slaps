"""Paper classification/retry policy and threshold agreement on real typed case records."""

from __future__ import annotations

import dataclasses
import json
import math

import pytest

from hwoslaps.analysis.nonlinear import (
    ClassificationRule, RoleAcceptance, case_status, classify_case, detection_agreement, select_attempt,
)
from hwoslaps.config.checks import ConfigError
from hwoslaps.identity import KernelIdentity
from hwoslaps.inference.result import (
    CaseResult, ForecastReference, ObservationRecord, RefineOutcome, RetentionInventory, RoleFit, RoleStatus, SamplerRecord,
)
from hwoslaps.inference.settings import FitSpec, RefineSettings, SamplerSettings
from hwoslaps.scene.cosmology import Cosmology, CosmologySpec
from hwoslaps.scene.halos import Halo, HaloModel

SCENE_PARAMETER = "lens.mass.mass.einstein_radius"
PARAMETER_PATH = "galaxies.lens.mass.einstein_radius"
CONFIG_DIGEST, COMPARISON_DIGEST, MASK_DIGEST = "01" * 32, "02" * 32, "03" * 32
KERNEL = KernelIdentity("04" * 32, (7, 7), 0.05)
PAPER_RULE = {"q_threshold": 10.0, "marginal_half_width": 1.0,
              "acceptance": {"smooth": ["accepted_repeatable_profile"], "subhalo": ["accepted_repeatable_profile"]},
              "require_retained_state": True, "retry_log_likelihood_tolerance": 0.1, "stationarity_tolerance": None}


def _rule(**overrides):
    return ClassificationRule.from_mapping({**PAPER_RULE, **overrides})


def _role(role, likelihood, status=RoleStatus.ACCEPTED, *, gradient=1e-8):
    anchor = status is RoleStatus.ZERO_RESIDUAL_ANCHOR
    failed = status is RoleStatus.FAILED
    likelihood = None if failed else likelihood
    retention = RetentionInventory("directory", {"search_internal.dill": {
        "bytes": 1, "sha256": "05" * 32, "location": "files/search_internal/search_internal.dill"}}, (), True)
    sampler = None if anchor else SamplerRecord(
        name=role + "_0123456789abcdef", output_path=role + "_0123456789abcdef/identifier", identity="06" * 32,
        n_live=30, requested={}, effective={"n_live": 30, "n_eff": 200.0}, seed=7, training_workers=1,
        log_likelihood_max=likelihood, log_evidence=None, likelihood_calls=None if failed else 300,
        retention=retention, runtime_s=1.0)
    refinement = None if anchor or status in (RoleStatus.SAMPLER_ONLY, RoleStatus.FAILED) else RefineOutcome(
        acceptance_status=status, best_log_likelihood=likelihood,
        best_vector=None if likelihood is None else (1.0,),
        gates={"incumbent": True, "support": True, "repeat": True}, record={},
        projected_gradient_linf=gradient, repeat_converged=gradient < 1e-6 if gradient is not None else False)
    return RoleFit(role, "truth_anchor" if anchor else "search", "failed" if failed else "success", likelihood,
                   -99.0 if role == "smooth" else -92.0, (PARAMETER_PATH,), 1, sampler, refinement,
                   None if failed or anchor else {PARAMETER_PATH: 0.5}, 0.0 if anchor else None,
                   "boom" if failed else None, 1.0)


@pytest.fixture
def case_factory():
    halo = Halo(HaloModel("PointMass", None, None), 1e7, (0.0, 1.0), 0.2, 0.6,
                Cosmology(CosmologySpec("Planck15", None)))

    def build(q=12.0, *, forecast_q=None, amplitude=None):
        reference = None if forecast_q is None else ForecastReference(
            q=forecast_q, metric="q_asimov" if amplitude is None else "q_mismatch", mass_msun=halo.mass_msun,
            position_yx_arcsec=halo.position_yx_arcsec, config_digest=CONFIG_DIGEST, mask_digest=MASK_DIGEST,
            nuisance_names=(SCENE_PARAMETER,), model_kernel=KERNEL, amplitude=amplitude,
            comparison_digest=COMPARISON_DIGEST, noise_model="diagonal")
        return CaseResult("hand_case", halo, ObservationRecord("expected", None, CONFIG_DIGEST, "07" * 32, halo),
                          {"shape": (4, 4), "mask": {"digest": MASK_DIGEST}, "units": "e_per_s"},
                          FitSpec(mode="fixed_template"), SamplerSettings(use_jax=True, n_live_smooth=30,
                          n_live_subhalo_fixed=30, n_eff=200, retain_search_internal=True), 7, RefineSettings(),
                          {"smooth": "08" * 32, "subhalo": "09" * 32}, _role("smooth", -100.0),
                          _role("subhalo", -100.0 + q / 2.0), q, max(q, 0.0), None, None, reference,
                          (SCENE_PARAMETER,), COMPARISON_DIGEST, {})
    return build


def _without_retention(role, *, missing=None, bound=True):
    retention = None if missing is None else RetentionInventory("directory", {}, tuple(missing), bound)
    return dataclasses.replace(role, sampler=dataclasses.replace(role.sampler, retention=retention))


@pytest.mark.parametrize(("event", "expected", "reason"), [
    ("accepted", "accepted", None), ("unresolved", "unresolved", "subhalo: unresolved_optimization"),
    ("failed", "failed", "subhalo_failed: boom"), ("failed_incomplete", "failed", "smooth_failed: boom"),
    ("retention_none", "incomplete", "search_internal.dill"), ("retention_missing", "incomplete", "search_internal.dill"),
    ("retention_unbound", "incomplete", "not bound"), ("retention_optional", "accepted", None),
    ("anchor", "accepted", None), ("sampler_disallowed", "unresolved", "sampler_only"),
    ("sampler_allowed", "accepted", None), ("gradient_historical", "accepted", None),
    ("gradient_required", "unresolved", "not_stationary: subhalo"),
    ("gradient_missing", "unresolved", "not_stationary: subhalo"),
    ("gradient_nonfinite", "unresolved", "not_stationary: subhalo"),
    ("anchor_stationarity", "accepted", None),
])
def test_case_status_follows_the_paper_rules(event, expected, reason, case_factory):
    case, rule = case_factory(), _rule()
    if event == "unresolved":
        case = dataclasses.replace(case, subhalo=_role("subhalo", -94.0, RoleStatus.UNRESOLVED))
    elif event == "failed":
        case = dataclasses.replace(case, subhalo=_role("subhalo", None, RoleStatus.FAILED), q_signed=None, q_clipped=None)
    elif event == "failed_incomplete":
        case = dataclasses.replace(case, smooth=_role("smooth", None, RoleStatus.FAILED),
                                   subhalo=_without_retention(case.subhalo), q_signed=None, q_clipped=None)
    elif event.startswith("retention"):
        missing = ("search_internal.dill",) if event == "retention_missing" else () if event == "retention_unbound" else None
        case = dataclasses.replace(case, subhalo=_without_retention(case.subhalo, missing=missing, bound=event != "retention_unbound"))
        if event == "retention_optional":
            rule = _rule(require_retained_state=False)
    elif event.startswith("anchor"):
        case = dataclasses.replace(case, subhalo=_role("subhalo", -94.0, RoleStatus.ZERO_RESIDUAL_ANCHOR))
        rule = _rule(acceptance={"smooth": [RoleStatus.ACCEPTED], "subhalo": [RoleStatus.ZERO_RESIDUAL_ANCHOR]},
                     stationarity_tolerance=1e-6 if event == "anchor_stationarity" else None)
    elif event.startswith("sampler"):
        case = dataclasses.replace(case, subhalo=_role("subhalo", -94.0, RoleStatus.SAMPLER_ONLY))
        if event == "sampler_allowed":
            rule = _rule(acceptance={"smooth": [RoleStatus.ACCEPTED], "subhalo": [RoleStatus.SAMPLER_ONLY]},
                         stationarity_tolerance=1e-6)
    elif event.startswith("gradient"):
        gradient = None if event == "gradient_missing" else math.nan if event == "gradient_nonfinite" else 0.055
        case = dataclasses.replace(case, subhalo=_role("subhalo", -94.0, gradient=gradient))
        rule = _rule(stationarity_tolerance=None if event == "gradient_historical" else 1e-6)
    result = case_status(case, acceptance=rule.acceptance, require_retained_state=rule.require_retained_state,
                         stationarity_tolerance=rule.stationarity_tolerance)
    assert result.status == expected
    assert result.role_statuses == {"smooth": case.smooth.acceptance_status, "subhalo": case.subhalo.acceptance_status}
    assert (not result.reasons) if reason is None else any(reason in item for item in result.reasons)


@pytest.mark.parametrize(("q", "unresolved", "marginal", "detected", "clipped"), [
    (12.0, False, False, True, 12.0), (10.5, False, True, True, 10.5),
    (9.5, False, True, False, 9.5), (10.0, False, True, True, 10.0),
    (11.0, False, False, True, 11.0), (9.0, False, False, False, 9.0),
    (-2.0, False, False, False, 0.0), (10.5, True, True, None, 10.5),
])
def test_classification_reports_marginal_and_detection(q, unresolved, marginal, detected, clipped, case_factory):
    case = case_factory(q)
    if unresolved:
        case = dataclasses.replace(case, subhalo=_role("subhalo", -100.0 + q / 2.0, RoleStatus.UNRESOLVED))
    result = classify_case(case, _rule())
    assert result.q_signed == q and result.q_clipped == clipped
    assert result.marginal is marginal and result.detected is detected
    assert result.status == ("unresolved" if unresolved else "accepted")
    assert result.truth_shortfall == {"smooth": 1.0, "subhalo": -92.0 - (-100.0 + q / 2.0)}


def test_failed_case_with_no_statistic_has_no_detection_or_marginality(case_factory):
    case = dataclasses.replace(case_factory(), subhalo=_role("subhalo", None, RoleStatus.FAILED), q_signed=None, q_clipped=None)
    result = classify_case(case, _rule())
    assert (result.status, result.detected, result.marginal) == ("failed", None, None)
    assert result.truth_shortfall == {"smooth": 1.0, "subhalo": None}


@pytest.mark.parametrize(("q", "raises"), [(12.0 + 0.5e-7, False), (12.0 + 2e-7, True), (None, True), (math.nan, True)])
def test_q_must_match_the_role_likelihoods(q, raises, case_factory):
    case = dataclasses.replace(case_factory(), q_signed=q)
    if raises:
        with pytest.raises(ValueError, match="q_signed"):
            classify_case(case, _rule())
    else:
        assert classify_case(case, _rule()).q_signed == q


@pytest.mark.parametrize(("event", "selected", "label", "changed"), [
    ("first_accepted", "first", None, None), ("no_retry", "first", None, None),
    ("within_tolerance", "retry", "unresolved_to_detection", True),
    ("degraded_role", "first", None, None), ("retry_unresolved", "first", None, None),
    ("first_missing_likelihood", "retry", "unresolved_to_detection", None),
    ("failed_first", "retry", "failed_to_detection", None),
    ("incomplete_first", "retry", "incomplete_to_detection", True),
    ("non_detection", "retry", "unresolved_to_non_detection", False),
    ("new_namespace", "retry", "unresolved_to_detection", True),
    ("stored_data", "retry", "unresolved_to_detection", True),
])
def test_retry_promotion(event, selected, label, changed, case_factory):
    first = dataclasses.replace(case_factory(8.0), subhalo=_role("subhalo", -96.0, RoleStatus.UNRESOLVED))
    retry = case_factory(12.0)
    if event == "first_accepted":
        first = case_factory(8.0)
    elif event == "no_retry":
        retry = None
    elif event in ("within_tolerance", "degraded_role"):
        decrease = 0.05 if event == "within_tolerance" else 0.2
        retry = dataclasses.replace(retry, smooth=_role("smooth", -100.0 - decrease), q_signed=12.0 + 2 * decrease,
                                    q_clipped=12.0 + 2 * decrease)
    elif event == "retry_unresolved":
        retry = dataclasses.replace(retry, subhalo=_role("subhalo", -94.0, RoleStatus.UNRESOLVED))
    elif event == "first_missing_likelihood":
        first = dataclasses.replace(first, smooth=_role("smooth", None, RoleStatus.UNRESOLVED), q_signed=None, q_clipped=None)
    elif event == "failed_first":
        first = dataclasses.replace(first, smooth=_role("smooth", None, RoleStatus.FAILED), q_signed=None, q_clipped=None)
    elif event == "incomplete_first":
        first = dataclasses.replace(first, subhalo=_without_retention(_role("subhalo", -96.0)))
    elif event == "non_detection":
        retry = case_factory(8.0)
    elif event == "new_namespace":
        retry = dataclasses.replace(retry, case_id="fresh_retry_namespace", sampler_seed=8,
                                    sampler=dataclasses.replace(retry.sampler, n_eff=500), refine=RefineSettings(maxiter=500))
    elif event == "stored_data":
        retry = dataclasses.replace(retry, data=json.loads(json.dumps(retry.data)))
    result = select_attempt(first, retry, _rule())
    assert (result.selected, result.detection_changed_by_retry, result.q_classification_changed_by_retry) == (selected, label, changed)


@pytest.mark.parametrize("different", ["hypothesis", "observation", "data", "fit", "models", "comparison"])
def test_retry_refuses_a_different_scientific_case(different, case_factory):
    first, retry = case_factory(), case_factory()
    changes = {"hypothesis": {"hypothesis": dataclasses.replace(retry.hypothesis, mass_msun=2e7)},
               "observation": {"observation": dataclasses.replace(retry.observation, data_digest="10" * 32)},
               "data": {"data": {**retry.data, "mask": {"digest": "11" * 32}}},
               "fit": {"fit": dataclasses.replace(retry.fit, anchor_chi2_tolerance=1e-7)},
               "models": {"models": {**retry.models, "subhalo": "12" * 32}},
               "comparison": {"comparison_digest": "13" * 32}}
    with pytest.raises(ValueError, match="same hypothesis, observation, data, fit and model"):
        select_attempt(first, dataclasses.replace(retry, **changes[different]), _rule())


@pytest.mark.parametrize(("defect", "path"), [
    ("missing_stationarity", "classification.stationarity_tolerance"),
    ("zero_stationarity", "classification.stationarity_tolerance"),
    ("negative_stationarity", "classification.stationarity_tolerance"),
    ("bool_stationarity", "classification.stationarity_tolerance"),
    ("nan_stationarity", "classification.stationarity_tolerance"),
    ("unknown_status", "classification.acceptance.subhalo[0]"),
    ("unknown_key", "classification.typo"), ("missing_threshold", "classification.q_threshold"),
    ("zero_threshold", "classification.q_threshold"), ("bool_threshold", "classification.q_threshold"),
])
def test_classification_rule_mapping_is_strict(defect, path):
    mapping = {**PAPER_RULE}
    if defect.startswith("missing_"):
        mapping.pop("stationarity_tolerance" if defect == "missing_stationarity" else "q_threshold")
    elif defect.endswith("stationarity"):
        mapping["stationarity_tolerance"] = {"zero_stationarity": 0, "negative_stationarity": -1e-6,
                                              "bool_stationarity": True, "nan_stationarity": math.nan}[defect]
    elif defect == "unknown_status":
        mapping["acceptance"] = {"smooth": [RoleStatus.ACCEPTED], "subhalo": ["accepted_typo"]}
    elif defect == "unknown_key":
        mapping["typo"] = None
    else:
        mapping["q_threshold"] = 0 if defect == "zero_threshold" else True
    with pytest.raises(ConfigError) as error:
        ClassificationRule.from_mapping(mapping)
    assert error.value.path == path


@pytest.mark.parametrize("tolerance", [None, 1e-6])
def test_rule_round_trips_with_the_same_stationarity_domain(tolerance):
    rule = _rule(stationarity_tolerance=tolerance)
    assert rule.to_mapping() == {**PAPER_RULE, "stationarity_tolerance": tolerance}
    assert ClassificationRule.from_mapping(rule.to_mapping()) == rule
    with pytest.raises(ConfigError, match="stationarity_tolerance"):
        dataclasses.replace(rule, stationarity_tolerance=0)


@pytest.mark.parametrize(("pairs", "expected"), [
    ([(12., 12.), (8., 12.), (12., 8.), (8., 8.)], (1, 1, 1, 1)),
    ([(4., 5.), (12., 10.), (18., 20.)], (2, 0, 0, 1)),
])
def test_detection_agreement_preserves_hand_and_old_threshold_confusion(pairs, expected, case_factory):
    cases = [case_factory(q, forecast_q=forecast) for q, forecast in pairs]
    result = detection_agreement((case, classify_case(case, _rule())) for case in cases)
    assert (result.both, result.forecast_only, result.nonlinear_only, result.neither) == expected
    assert result.rule == _rule() and result.excluded == {}


@pytest.mark.parametrize(("event", "expected_counts", "exclusions"), [
    ("negative_amplitude", (0, 0, 1, 0), {}), ("unresolved", (0, 0, 0, 0), {"unresolved": 1}),
    ("failed", (0, 0, 0, 0), {"failed": 1}), ("no_reference", (0, 0, 0, 0), {"no_forecast_reference": 1}),
    ("control", (0, 0, 0, 0), {"control_observation": 1}),
    ("foreign_injection", (0, 0, 0, 0), {"observation_injection_mismatch": 1}),
])
def test_detection_agreement_counts_and_exclusions(event, expected_counts, exclusions, case_factory):
    case = case_factory(12., forecast_q=16., amplitude=-2. if event == "negative_amplitude" else 2.)
    if event == "unresolved":
        case = dataclasses.replace(case, subhalo=_role("subhalo", -94., RoleStatus.UNRESOLVED))
    elif event == "failed":
        case = dataclasses.replace(case, subhalo=_role("subhalo", None, RoleStatus.FAILED), q_signed=None, q_clipped=None)
    elif event == "no_reference":
        case = dataclasses.replace(case, forecast_reference=None)
    elif event == "control":
        case = dataclasses.replace(case, observation=dataclasses.replace(case.observation, subhalo=None))
    elif event == "foreign_injection":
        case = dataclasses.replace(case, observation=dataclasses.replace(case.observation,
                                  subhalo=dataclasses.replace(case.hypothesis, mass_msun=2e7)))
    table = detection_agreement([(case, classify_case(case, _rule()))])
    assert (table.both, table.forecast_only, table.nonlinear_only, table.neither) == expected_counts
    assert table.excluded == exclusions


@pytest.mark.parametrize(("comparison", "configuration", "expected_mismatches"), [
    ("15" * 32, "16" * 32, 2), ("15" * 32, CONFIG_DIGEST, 2), (COMPARISON_DIGEST, "16" * 32, 0),
])
@pytest.mark.parametrize("claimed_unsupported", [False, True])
def test_agreement_reports_input_differences_and_scene_namespace(comparison, configuration, expected_mismatches,
                                                                  claimed_unsupported, case_factory):

    case = case_factory(12., forecast_q=12.)
    names = (SCENE_PARAMETER, "source.light.light.intensity", "observation.background_offset_adu", "psf.zernikes[4]")
    reference = dataclasses.replace(case.forecast_reference, nuisance_names=names, mask_digest="14" * 32,
                                    comparison_digest=comparison, config_digest=configuration)
    fitted = (SCENE_PARAMETER, "observation.background_offset_adu", "psf.zernikes[4]") if claimed_unsupported else (SCENE_PARAMETER,)
    case = dataclasses.replace(case, forecast_reference=reference, fitted_parameters=fitted)
    unresolved = dataclasses.replace(case, subhalo=_role("subhalo", -94., RoleStatus.UNRESOLVED))
    table = detection_agreement([(value, classify_case(value, _rule())) for value in (case, unresolved)])
    assert (table.both, table.excluded) == (1, {"unresolved": 1})
    assert table.mask_mismatches == 2 and table.configuration_mismatches == expected_mismatches
    assert table.unfitted_forecast_nuisances == {"source.light.light.intensity": 2,
                                              "observation.background_offset_adu": 2, "psf.zernikes[4]": 2}


@pytest.mark.parametrize("invalid", ["empty", "mixed_rules", "foreign_node", "accepted_missing_statistic"])
def test_agreement_refuses_incoherent_inputs(invalid, case_factory):
    case = case_factory(12., forecast_q=12.)
    classified = classify_case(case, _rule())
    if invalid == "empty":
        pairs, message = [], "at least one case"
    elif invalid == "mixed_rules":
        pairs = [(case, classified), (case, classify_case(case, _rule(stationarity_tolerance=1e-6)))]
        message = "mixed rules"
    elif invalid == "foreign_node":
        case = dataclasses.replace(case, forecast_reference=dataclasses.replace(case.forecast_reference, mass_msun=2e7))
        pairs, message = [(case, classified)], "reference node"
    else:
        pairs, message = [(case, dataclasses.replace(classified, detected=None, q_signed=None))], "finite nonlinear statistic"
    with pytest.raises(ValueError, match=message):
        detection_agreement(pairs)
