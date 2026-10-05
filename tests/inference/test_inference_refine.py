"""Refinement: acceptance gates, start normalization and selection, stationarity record, base parity."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from hwoslaps.inference.refine import refine
from hwoslaps.inference.result import RoleStatus
from hwoslaps.inference.settings import RefineSettings
from hwoslaps.inference.starts import RefineStart, select_starts

FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "inference" / "refine_records_8fa6209.json"
CASES = json.loads(FIXTURE.read_text(encoding="utf-8"))["cases"]
UNIT = ([0.0, 0.0], [1.0, 1.0])


def quadratic(target, weights=None):
    target = np.asarray(target, dtype=float)
    weights = np.ones_like(target) if weights is None else np.asarray(weights, dtype=float)
    return (lambda x: 0.5 * float(np.sum(weights * (np.asarray(x) - target) ** 2)),
            lambda x: weights * (np.asarray(x) - target))


def objective_functions(spec):
    """The fixture's objective kinds (see the fixture description)."""
    if spec["kind"] == "quadratic":
        return quadratic(spec["target"], spec["weights"])
    if spec["kind"] == "two_wells_min":
        def value(x):
            return min(10.0 * (x[0] - 2.5) ** 2, 2.0 + 10.0 * (x[0] - 4.0) ** 2)

        def gradient(x):
            a, b = 10.0 * (x[0] - 2.5) ** 2, 2.0 + 10.0 * (x[0] - 4.0) ** 2
            return np.array([20.0 * (x[0] - 2.5)]) if a < b else np.array([20.0 * (x[0] - 4.0)])
        return value, gradient
    if spec["kind"] == "two_wells_split":
        return (lambda x: 5.0 * (x[0] - 0.2) ** 2 if x[0] < 0.5 else 1.0 + 50.0 * (x[0] - 0.8) ** 2,
                lambda x: np.array([10.0 * (x[0] - 0.2)]) if x[0] < 0.5 else np.array([100.0 * (x[0] - 0.8)]))
    assert spec["kind"] == "nan"
    return (lambda x: float("nan"), lambda x: np.full(len(x), np.nan))


def write_search_files(directory, names, ml, ml_log_likelihood, samples):
    """samples_summary.json and samples.csv in the layout AutoFit writes them; equal weights."""
    summary = {"arguments": {"max_log_likelihood_sample": {"arguments": {
        "kwargs": {"arguments": dict(zip(names, ml))}, "log_likelihood": ml_log_likelihood}}}}
    (directory / "samples_summary.json").write_text(json.dumps(summary), encoding="utf-8")
    lines = [",".join([*names, "log_likelihood", "weight"])]
    lines += [",".join(repr(float(v)) for v in [*vector, log_l, 1.0 / len(samples)]) for vector, log_l in samples]
    (directory / "samples.csv").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return directory


def starts_from(points, lower, upper, saved_log_likelihood):
    return tuple(RefineStart.from_physical(index=index, physical=point, lower=lower, upper=upper,
                                           source="sampler_ml" if index == 0 else "sampler_sample",
                                           original=index != 0,
                                           origin={"saved_log_likelihood": saved_log_likelihood} if index == 0
                                           else {})
                 for index, point in enumerate(points))


def test_quadratic_refinement_is_accepted_with_full_support(box_objective):
    target, incumbent = np.array([0.23, 0.71]), np.array([0.10, 0.70])
    value, gradient = quadratic(target)
    points = [incumbent, [0.05, 0.10], [0.90, 0.20], [0.80, 0.90], [0.10, 0.90], [0.50, 0.20], [0.70, 0.60],
              [0.20, 0.40], [0.40, 0.80]]
    outcome = refine(starts_from(points, *UNIT, -value(incumbent)), box_objective(*UNIT, value, gradient),
                     RefineSettings(maxiter=100, repeat_maxiter=100, start_separation_normalized_l2=0.01))
    assert outcome.acceptance_status is RoleStatus.ACCEPTED
    assert all(outcome.gates.values())
    assert outcome.best_vector == pytest.approx(target.tolist(), abs=1.0e-8)
    assert outcome.best_log_likelihood == pytest.approx(0.0, abs=1.0e-12)
    agreement = outcome.record["candidate_start_agreement"]
    assert agreement["supporting_original_start_indices"] == list(range(1, 9))
    assert outcome.record["incumbent"]["passed"] is True


@pytest.mark.parametrize("case", ["incumbent_saved_mismatch", "unsupported_incumbent", "support_below_minimum",
                                  "no_finite_evaluation"])
def test_refinement_gates_reject_each_failure(case, box_objective):
    """Each defect fails its own gate and leaves the role unresolved."""
    one_d = ([0.0], [1.0])
    if case == "incumbent_saved_mismatch":
        target = np.array([0.23, 0.71])
        value, gradient = quadratic(target)
        points = [[0.10, 0.70]] + [[0.1 * (i + 1), 0.2 + 0.05 * i] for i in range(8)]
        outcome = refine(starts_from(points, *UNIT, -5.0), box_objective(*UNIT, value, gradient),
                         RefineSettings(maxiter=100, repeat_maxiter=100))
        assert outcome.record["incumbent"]["saved_log_likelihood_error"] == pytest.approx(
            5.0 - value(np.array([0.10, 0.70])), rel=1.0e-12)
        failed = {"incumbent"}
    elif case == "unsupported_incumbent":
        value, gradient = objective_functions({"kind": "two_wells_split"})
        points = [[0.2]] + [[0.56 + 0.03 * i] for i in range(8)]
        outcome = refine(starts_from(points, *one_d, 0.0), box_objective(*one_d, value, gradient),
                         RefineSettings(maxiter=200, repeat_maxiter=200))
        assert outcome.best_vector == pytest.approx([0.2], abs=1.0e-8)
        assert all(run["observed_best_half_chi2"] == pytest.approx(1.0, abs=1.0e-8)
                   for run in outcome.record["runs"][1:])
        failed = {"support"}
    elif case == "support_below_minimum":
        value, gradient = quadratic([0.2, 0.3])
        points = [[0.2, 0.3]] + [[0.2 + 0.05 * i, 0.3] for i in range(8)]
        outcome = refine(starts_from(points, *UNIT, 0.0), box_objective(*UNIT, value, gradient),
                         RefineSettings(maxiter=20, repeat_maxiter=20, minimum_distinct_original_start_support=9))
        assert len(outcome.record["candidate_start_agreement"]["supporting_original_start_indices"]) == 8
        failed = {"support"}
    else:
        value, gradient = objective_functions({"kind": "nan"})
        points = [[0.2, 0.3]] + [[0.2 + 0.05 * i, 0.3] for i in range(8)]
        outcome = refine(starts_from(points, *UNIT, 0.0), box_objective(*UNIT, value, gradient),
                         RefineSettings(maxiter=3, repeat_maxiter=3))
        assert (outcome.best_log_likelihood, outcome.best_vector) == (None, None)
        failed = set(outcome.gates)
    assert outcome.acceptance_status is RoleStatus.UNRESOLVED
    assert {name for name, passed in outcome.gates.items() if not passed} == failed


def test_starts_are_normalized_once(tmp_path, box_objective):
    """The reviewer's counterexample: prior [2, 4], two basins, sampler maximum at 2.5."""
    write_search_files(tmp_path, ["x"], [2.5], 0.0, [([2.3], -0.4), ([2.7], -0.4)])
    settings = RefineSettings(original_start_count=2, start_separation_normalized_l2=0.05)
    starts = select_starts(tmp_path, ["x"], [2.0], [4.0], settings)
    assert [start.physical for start in starts] == [(2.5,), (2.3,), (2.7,)]
    assert [start.normalized[0] for start in starts] == pytest.approx([0.25, 0.15, 0.35], rel=1.0e-12)
    value, gradient = objective_functions({"kind": "two_wells_min"})
    outcome = refine(starts, box_objective([2.0], [4.0], value, gradient), settings)
    assert [run["start_x"] for run in outcome.record["runs"]] == [[2.5], [2.3], [2.7]]
    assert outcome.record["runs"][0]["start_half_chi2"] == 0.0
    assert outcome.best_vector == pytest.approx((2.5,), abs=1.0e-8)
    assert outcome.acceptance_status is RoleStatus.ACCEPTED


@pytest.mark.parametrize("defect", ["outside_unit_box", "other_box", "incumbent_not_first", "count"])
def test_bad_starts_fail_before_any_evaluation(defect, box_objective):
    evaluations = []

    def value(x):
        evaluations.append(list(x))
        return 0.0

    objective = box_objective([0.0], [1.0], value, lambda x: np.zeros(1))
    starts = starts_from([[0.4], [0.3]], [0.0], [1.0], 0.0)
    settings = RefineSettings(original_start_count=1)
    if defect == "outside_unit_box":
        with pytest.raises(ValueError, match="outside the unit box"):
            RefineStart(index=0, physical=(0.5,), normalized=(1.2,), source="sampler_ml", original=False, origin={})
    elif defect == "other_box":
        shifted = starts_from([[0.4], [0.3]], [0.0], [2.0], 0.0)
        with pytest.raises(ValueError, match="does not map back to its physical origin"):
            refine(shifted, objective, settings)
    elif defect == "incumbent_not_first":
        with pytest.raises(ValueError, match="maximum-likelihood incumbent"):
            refine((starts[1], starts[0]), objective, settings)
    else:
        with pytest.raises(ValueError, match="needs the incumbent and 2 starts"):
            refine(starts, objective, RefineSettings(original_start_count=2))
    assert evaluations == []


@pytest.mark.parametrize("posterior", ["sharp", "broad", "unweighted", "too_few"])
def test_start_selection_uses_prior_and_posterior_scales(posterior, tmp_path):
    names = ["a", "b"]
    if posterior == "sharp":
        radius = 0.006
        ring = [(0.5 + radius * np.cos(k * np.pi / 4), 0.5 + radius * np.sin(k * np.pi / 4)) for k in range(8)]
        samples = [(point, -100.0 * radius**2) for point in ring] + [((0.5002, 0.5), -100.0 * 0.0002**2)]
        write_search_files(tmp_path, names, [0.5, 0.5], 0.0, samples)
        starts = select_starts(tmp_path, names, *UNIT, RefineSettings())
        sigma = np.asarray(starts[0].origin["selection_rule"]["posterior_sigma_normalized"])
        assert np.all(sigma > 0.0) and np.all(sigma < 0.01)
        assert {start.physical for start in starts[1:]} == set(ring)
        assert all(start.origin["separation_prior_normalized_l2"] < 0.05
                   and start.origin["separation_posterior_sigma"] >= 1.0 for start in starts[1:])
    elif posterior == "broad":
        samples = [((0.9, 0.9), -0.1), ((0.1, 0.1), -0.2), ((0.16, 0.1), -0.3), ((0.5, 0.5005), -0.4)]
        write_search_files(tmp_path, names, [0.5, 0.5], 0.0, samples)
        starts = select_starts(tmp_path, names, *UNIT, RefineSettings(original_start_count=3))
        assert [start.physical for start in starts] == [(0.5, 0.5), (0.9, 0.9), (0.1, 0.1), (0.16, 0.1)]
        assert starts[3].origin["separation_prior_normalized_l2"] == pytest.approx(0.06, rel=1.0e-12)
        assert starts[3].origin["separation_posterior_sigma"] < 1.0
    elif posterior == "unweighted":
        write_search_files(tmp_path, ["a"], [0.5], 0.0, [((0.4,), -1.0), ((0.6,), -1.0)])
        (tmp_path / "samples.csv").write_text("a,log_likelihood\n0.4,-1.0\n0.6,-1.0\n", encoding="utf-8")
        with pytest.raises(ValueError, match="posterior weights"):
            select_starts(tmp_path, ["a"], [0.0], [1.0], RefineSettings(original_start_count=2))
    else:
        samples = [((0.9, 0.9), -0.1), ((0.1, 0.1), -0.2), ((0.16, 0.1), -0.3), ((0.5, 0.5005), -0.4)]
        write_search_files(tmp_path, names, [0.5, 0.5], 0.0, samples)
        with pytest.raises(ValueError, match="supplied 3 separated samples"):
            select_starts(tmp_path, names, *UNIT, RefineSettings(original_start_count=4))


def test_historical_gates_accept_a_non_stationary_point(box_objective):
    """SCI-14: the six gates certify repeatability, not stationarity, and the record shows it.

    f(z) = 0.04 sum (z - 0.9)^2 on [0, 1]^2 changes by less than 0.065 over the box, so every
    start supports the best and the repeat passes after one iteration each, far from z = 0.9.
    """
    target = np.array([0.9, 0.9])
    value, gradient = quadratic(target, [0.08, 0.08])
    points = [[0.11, 0.11], [0.10, 0.12], [0.12, 0.10]]
    outcome = refine(starts_from(points, *UNIT, -value(np.array(points[0]))), box_objective(*UNIT, value, gradient),
                     RefineSettings(original_start_count=2, maxiter=1, repeat_maxiter=1, maxls=1))
    assert outcome.acceptance_status is RoleStatus.ACCEPTED
    best = np.asarray(outcome.best_vector)
    assert outcome.projected_gradient_linf == pytest.approx(0.08 * float(np.max(np.abs(best - 0.9))), rel=1.0e-12)
    assert outcome.projected_gradient_linf > 0.04
    assert outcome.repeat_converged is False


@pytest.mark.parametrize("case", CASES, ids=[case["name"] for case in CASES])
def test_refinement_reproduces_the_8fa6209_records(case, tmp_path, box_objective):
    """Every record field, bit for bit, against the base tree's fresh-profile optimiser."""
    settings = RefineSettings(**case["settings"])
    value, gradient = objective_functions(case["objective"])
    if "selection" in case:
        selection = case["selection"]
        write_search_files(tmp_path, selection["names"], selection["ml"], selection["ml_log_likelihood"],
                           selection["samples"])
        starts = select_starts(tmp_path, selection["names"], case["lower"], case["upper"], settings)
    else:
        starts = starts_from(case["starts"]["points"], case["lower"], case["upper"],
                             case["starts"]["saved_log_likelihood"])
    outcome = refine(starts, box_objective(case["lower"], case["upper"], value, gradient), settings)
    expected = case["expected"]
    record = {key: value for key, value in outcome.record.items() if key != "procedure"}
    assert json.dumps(record, sort_keys=True) == json.dumps(expected["record"], sort_keys=True)
    assert outcome.acceptance_status == expected["acceptance_status"]
    assert outcome.best_log_likelihood == expected["best_log_likelihood"]
    assert outcome.best_vector == (None if expected["best_vector"] is None else tuple(expected["best_vector"]))
    assert outcome.projected_gradient_linf == expected["projected_gradient_linf"]
    assert outcome.repeat_converged == expected["repeat_converged"]
