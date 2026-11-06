"""A repeated local maximum cannot supersede an already evaluated better truth."""

import json
from types import SimpleNamespace

import numpy as np
import pytest

from hwoslaps.analysis.nonlinear import ClassificationRule, RoleAcceptance, classify_case
from hwoslaps.inference import roles
from hwoslaps.inference.low_dimension import exact_likelihood
from hwoslaps.inference.result import RoleFit, RoleStatus, SamplerRecord
from hwoslaps.inference.sampler import SearchOutcome
from hwoslaps.inference.settings import FitSpec, RefineSettings, SamplerSettings


def _half_chi2(point):
    x, y = point
    t = (x - 0.2) / 0.6
    return 11.0 - (3*t*t - 2*t*t*t) + 100*(x - 0.2)**2*(x - 0.8)**2 + (y - 0.5)**2


def _gradient(point):
    x, y = point
    t = (x - 0.2) / 0.6
    return np.array([-(6*t - 6*t*t)/0.6 + 200*(x - 0.2)*(x - 0.8)*(2*x - 1), 2*(y - 0.5)])


def _fit(tmp_path, monkeypatch, box_objective, *, role="subhalo", truth=(0.8, 0.5), refine):
    # Control only the completed sampler boundary. Start selection, all optimizer
    # calls, likelihood checks, role assembly and classification remain real.
    files = tmp_path / "search" / "files"
    files.mkdir(parents=True)
    points = [(0.2, 0.5), (0.18, 0.48), (0.22, 0.52)]
    summary = {"arguments": {"max_log_likelihood_sample": {"arguments": {
        "kwargs": {"arguments": {"a": 0.2, "b": 0.5}}, "log_likelihood": -11.0}}}}
    (files / "samples_summary.json").write_text(json.dumps(summary), encoding="utf-8")
    rows = [f"{x},{y},{-_half_chi2((x, y))},0.3333333333333333" for x, y in points]
    (files / "samples.csv").write_text("a,b,log_likelihood,weight\n" + "\n".join(rows) + "\n", encoding="utf-8")
    record = SamplerRecord(role, "search", "ab"*32, 20, {}, {}, 1, 1, -11.0, -12.0, 3, None, 0.0)
    search = SearchOutcome(record, SimpleNamespace(samples=SimpleNamespace(
        max_log_likelihood=lambda as_instance: points[0])), None)
    monkeypatch.setattr(roles, "run_search", lambda **kwargs: search)
    model = SimpleNamespace(parameter_names=("a", "b"), lower=np.zeros(2), upper=np.ones(2))
    case = SimpleNamespace(model=lambda role: model, truth_vector=lambda role: np.asarray(truth),
        log_likelihood=lambda role, point: -_half_chi2(point), autofit_models={role: object()},
        analysis=object(), data=SimpleNamespace(record={}), fit=FitSpec(mode="fixed_template"),
        objective=lambda role: box_objective([0, 0], [1, 1], _half_chi2, _gradient))
    fitted, returned_search = roles.run_search_role(case, role, sampler=SamplerSettings(), sampler_seed=1,
        refine=refine, session=None, case_dir=tmp_path, case_id="known-incumbent")
    assert returned_search is search
    assert fitted.sampler is record
    return fitted


@pytest.mark.parametrize("role", ["smooth", "subhalo"])
@pytest.mark.parametrize("start_count", [2, 8], ids=["repeatable-local-maximum", "insufficient-starts"])
def test_better_known_truth_is_retained_without_acceptance(tmp_path, monkeypatch, box_objective, role, start_count):
    fitted = _fit(tmp_path, monkeypatch, box_objective, role=role,
                  refine=RefineSettings(original_start_count=start_count))
    assert fitted.status == "success"
    assert fitted.log_likelihood == fitted.truth_log_likelihood == pytest.approx(-10.0)
    assert fitted.refinement.best_vector == pytest.approx([0.8, 0.5])
    assert fitted.acceptance_status is RoleStatus.UNRESOLVED
    assert not any(fitted.refinement.gates.values())
    assert fitted.refinement.projected_gradient_linf is None
    assert fitted.refinement.repeat_converged is None
    # Keep the sampler maximum/evidence and original refinement diagnostics as
    # provenance; truth supplies neither posterior mass nor independent starts.
    assert (fitted.sampler.log_likelihood_max, fitted.sampler.log_evidence) == (-11.0, -12.0)
    superseded = fitted.refinement.record["superseded_refinement"]
    assert superseded["best_log_likelihood"] == pytest.approx(-11.0)
    if start_count == 2:
        assert all(superseded["gates"].values())
        assert len(superseded["record"]["candidate_start_agreement"]["supporting_original_start_indices"]) == 2
    else:
        assert "start_selection_error" in superseded["record"]
    assert RoleFit.from_mapping(fitted.to_mapping()).to_mapping() == fitted.to_mapping()


@pytest.mark.parametrize("truth", [(0.02, 0.9), (0.2, 0.5)], ids=["lower-truth", "equal-truth"])
def test_equal_or_better_refinement_keeps_its_certificate(tmp_path, monkeypatch, box_objective, truth):
    fitted = _fit(tmp_path, monkeypatch, box_objective, truth=truth,
                  refine=RefineSettings(original_start_count=2))
    assert fitted.acceptance_status is RoleStatus.ACCEPTED
    assert fitted.log_likelihood == pytest.approx(-11.0)
    assert fitted.refinement.best_vector == pytest.approx([0.2, 0.5])
    assert all(fitted.refinement.gates.values())
    assert "superseded_refinement" not in fitted.refinement.record


def test_sampler_only_still_reports_the_sampler_maximum(tmp_path, monkeypatch, box_objective):
    fitted = _fit(tmp_path, monkeypatch, box_objective, refine=None)
    assert fitted.acceptance_status is RoleStatus.SAMPLER_ONLY
    assert fitted.refinement is None
    assert fitted.log_likelihood == -11.0
    assert fitted.truth_log_likelihood == -10.0


def test_known_better_truth_prevents_an_accepted_nondetection(tmp_path, monkeypatch, box_objective):
    fitted = _fit(tmp_path, monkeypatch, box_objective, refine=RefineSettings(original_start_count=2))
    smooth = RoleFit("smooth", "exact", "success", -12.0, -12.0, (), 0, None, None, {}, None, None, 0.0,
                     exact_likelihood(-12.0).record)
    q = 2*(fitted.log_likelihood - smooth.log_likelihood)
    result = SimpleNamespace(q_signed=q, q_clipped=max(q, 0.0),
                             role=lambda role: smooth if role == "smooth" else fitted)
    accepted = frozenset({RoleStatus.ACCEPTED})
    rule = ClassificationRule(3.0, 0.1, RoleAcceptance(accepted, accepted), False, 0.1, 1.0e-6)
    classified = classify_case(result, rule)
    assert classified.q_signed == pytest.approx(4.0)
    assert classified.status == "unresolved"
    assert classified.detected is None
