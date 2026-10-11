"""Completed searches retain their incumbent when refinement lacks separated starts."""

import json
from types import SimpleNamespace

import numpy as np
import pytest

from hwoslaps.inference import roles
from hwoslaps.inference.objective import BoxObjective, ScalarCheck
from hwoslaps.inference.result import RoleFit, RoleStatus, SamplerRecord
from hwoslaps.inference.sampler import SearchOutcome
from hwoslaps.inference.settings import FitSpec, RefineSettings, SamplerSettings


@pytest.mark.parametrize("problem", ["too_few_starts", "gradient_refusal", "one_sample", "identical_samples",
                                   "invalid_vector", "negative_weight", "zero_weights"])
def test_refinement_problem_keeps_completed_search_unresolved(problem, tmp_path, monkeypatch):
    files = tmp_path / "search/files"
    files.mkdir(parents=True)
    files.joinpath("samples_summary.json").write_text(json.dumps({"arguments": {
        "max_log_likelihood_sample": {"arguments": {"kwargs": {"arguments": {"a": 0.5, "b": 0.5}},
                                                        "log_likelihood": 0.0}}}}))
    rows = {
        "one_sample": "0.5,0.5,0.0,1.0\n",
        "identical_samples": "0.5,0.5,0.0,0.5\n0.5,0.5,0.0,0.5\n",
        "invalid_vector": "nan,0.4,-0.02,0.5\n0.6,0.6,-0.02,0.5\n",
        "negative_weight": "0.4,0.4,-0.02,-0.5\n0.6,0.6,-0.02,0.5\n",
        "zero_weights": "0.4,0.4,-0.02,0.0\n0.6,0.6,-0.02,0.0\n",
    }.get(problem, "0.4,0.4,-0.02,0.5\n0.6,0.6,-0.02,0.5\n")
    files.joinpath("samples.csv").write_text("a,b,log_likelihood,weight\n" + rows)
    model = SimpleNamespace(parameter_names=("a", "b"), truth=np.array([0.5, 0.5]),
                            lower=np.zeros(2), upper=np.ones(2))
    record = SamplerRecord(name="smooth", output_path="search", identity="ab" * 32, n_live=20,
                           requested={}, effective={}, seed=1, training_workers=1, log_likelihood_max=0.0,
                           log_evidence=-1.0, likelihood_calls=10, retention=None, runtime_s=1.0)
    searched = SearchOutcome(record, SimpleNamespace(), None)
    monkeypatch.setattr(roles, "run_search", lambda **kwargs: searched)

    def objective(role):
        if problem != "gradient_refusal":
            raise AssertionError("insufficient starts must not launch refinement")

        def refused(z):
            raise ValueError("undefined light-profile gradient")

        return BoxObjective(model.lower, model.upper, refused, lambda z: np.asarray(z) - 0.5,
                            lambda z, half: ScalarCheck(-half, -half, 0.0))

    case = SimpleNamespace(model=lambda role: model, truth_vector=lambda role: model.truth,
                           log_likelihood=lambda role, vector: -float(np.sum((np.asarray(vector) - 0.5) ** 2)),
                           autofit_models={"smooth": object()}, analysis=object(), data=SimpleNamespace(record={}),
                           fit=FitSpec(mode="fixed_template"), objective=objective)
    arguments = dict(sampler=SamplerSettings(), sampler_seed=1,
                     refine=RefineSettings(original_start_count=3 if problem == "too_few_starts" else 2),
                     session=None, case_dir=tmp_path, case_id="case")
    if problem in {"invalid_vector", "negative_weight", "zero_weights"}:
        message = "finite vectors" if problem == "invalid_vector" else "weights must be finite"
        with pytest.raises(ValueError, match=message):
            roles.run_search_role(case, "smooth", **arguments)
        return
    fitted, outcome = roles.run_search_role(case, "smooth", **arguments)
    assert outcome is searched and fitted.sampler is record
    assert fitted.status == "success" and fitted.error is None
    assert fitted.acceptance_status is RoleStatus.UNRESOLVED
    assert fitted.log_likelihood == 0.0 and fitted.refinement.best_vector == (0.5, 0.5)
    if problem == "too_few_starts":
        assert "supplied 2 separated samples" in fitted.refinement.record["start_selection_error"]
    elif problem in {"one_sample", "identical_samples"}:
        assert "zero spread" in fitted.refinement.record["start_selection_error"]
    else:
        assert fitted.refinement.gates["finite_gradient"] is False
        assert fitted.refinement.gates["repeat"] is False
        assert "undefined light-profile gradient" in fitted.refinement.record["runs"][0]["message"]
    assert len(fitted.refinement.record["start_provenance"]) == (1 if problem in {"one_sample", "identical_samples"} else 3)
    assert RoleFit.from_mapping(json.loads(json.dumps(fitted.to_mapping()))).acceptance_status is RoleStatus.UNRESOLVED
