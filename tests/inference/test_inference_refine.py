"""Refinement: acceptance gates, start normalization and selection, stationarity record, base parity."""

from __future__ import annotations

import dataclasses
import itertools
import json
from pathlib import Path

import numpy as np
import pytest

from hwoslaps.inference.objective import ScalarCheck
from hwoslaps.inference.refine import refine
from hwoslaps.inference.result import RoleStatus
from hwoslaps.inference.settings import RefineSettings
from hwoslaps.inference.starts import RefineStart, select_starts

FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "inference" / "refine_records_8fa6209.json"
CASES = json.loads(FIXTURE.read_text(encoding="utf-8"))["cases"]
UNIT = ([0.0, 0.0], [1.0, 1.0])
TARGET = (0.23, 0.71)
ACCEPT_POINTS = [[0.10, 0.70], [0.05, 0.10], [0.90, 0.20], [0.80, 0.90], [0.10, 0.90], [0.50, 0.20], [0.70, 0.60],
                 [0.20, 0.40], [0.40, 0.80]]
SLOW_TARGET, SLOW_WEIGHTS = (0.9, 0.9), (0.08, 0.08)
SLOW_POINTS = [[0.11, 0.11], [0.10, 0.12], [0.12, 0.10]]


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


def offset_direct_check(half_chi2, offset):
    """Direct check on the unit box: direct log L ``-f(z) - offset(z)`` against the implied ``-f``."""
    def direct_check(z, value):
        direct = -half_chi2(np.asarray(z)) - offset(np.asarray(z))
        return ScalarCheck(direct_log_likelihood=direct, implied_log_likelihood=-float(value),
                           direct_log_likelihood_error=abs(direct + float(value)))
    return direct_check


def test_quadratic_refinement_is_accepted_with_full_support(box_objective):
    """Every gate passes, and the role value is the direct likelihood at the best: a direct check
    3e-5 below the implied one (inside the 1e-4 tolerance) shows in ``best_log_likelihood``."""
    value, gradient = quadratic(TARGET)
    objective = dataclasses.replace(box_objective(*UNIT, value, gradient),
                                    direct_check=offset_direct_check(value, lambda z: 3.0e-5))
    outcome = refine(starts_from(ACCEPT_POINTS, *UNIT, -value(np.array(ACCEPT_POINTS[0]))), objective,
                     RefineSettings(maxiter=100, repeat_maxiter=100, start_separation_normalized_l2=0.01))
    assert outcome.acceptance_status is RoleStatus.ACCEPTED
    assert all(outcome.gates.values())
    assert outcome.best_vector == pytest.approx(TARGET, abs=1.0e-8)
    assert outcome.best_log_likelihood == pytest.approx(-3.0e-5, abs=1.0e-12)
    assert outcome.record["candidate_implied_log_likelihood"] == pytest.approx(0.0, abs=1.0e-12)
    agreement = outcome.record["candidate_start_agreement"]
    assert agreement["supporting_original_start_indices"] == list(range(1, 9))
    assert outcome.record["incumbent"]["passed"] is True


def _quadratic_setup(box_objective):
    value, gradient = quadratic(TARGET)
    starts = starts_from(ACCEPT_POINTS, *UNIT, -value(np.array(ACCEPT_POINTS[0])))
    return value, box_objective(*UNIT, value, gradient), starts, RefineSettings(maxiter=100, repeat_maxiter=100)


def _slow_setup(box_objective, gradient=None):
    """The flat objective 0.04 |z - 0.9|^2 of the SCI-14 keeper: its range over the box (0.065) is
    below the support tolerance, so every start supports any best."""
    value, slow_gradient = quadratic(SLOW_TARGET, SLOW_WEIGHTS)
    starts = starts_from(SLOW_POINTS, *UNIT, -value(np.array(SLOW_POINTS[0])))
    return value, box_objective(*UNIT, value, slow_gradient if gradient is None else gradient), starts


def _incumbent_saved_mismatch(box_objective):
    value, gradient = quadratic(TARGET)
    points = [[0.10, 0.70]] + [[0.1 * (i + 1), 0.2 + 0.05 * i] for i in range(8)]
    outcome = refine(starts_from(points, *UNIT, -5.0), box_objective(*UNIT, value, gradient),
                     RefineSettings(maxiter=100, repeat_maxiter=100))
    assert outcome.record["incumbent"]["saved_log_likelihood_error"] == pytest.approx(
        5.0 - value(np.array([0.10, 0.70])), rel=1.0e-12)
    return outcome, {"incumbent"}


def _unsupported_incumbent(box_objective):
    one_d = ([0.0], [1.0])
    value, gradient = objective_functions({"kind": "two_wells_split"})
    points = [[0.2]] + [[0.56 + 0.03 * i] for i in range(8)]
    outcome = refine(starts_from(points, *one_d, 0.0), box_objective(*one_d, value, gradient),
                     RefineSettings(maxiter=200, repeat_maxiter=200))
    assert outcome.best_vector == pytest.approx([0.2], abs=1.0e-8)
    assert all(run["observed_best_half_chi2"] == pytest.approx(1.0, abs=1.0e-8) for run in outcome.record["runs"][1:])
    return outcome, {"support"}


def _support_below_minimum(box_objective):
    value, gradient = quadratic([0.2, 0.3])
    points = [[0.2, 0.3]] + [[0.2 + 0.05 * i, 0.3] for i in range(8)]
    outcome = refine(starts_from(points, *UNIT, 0.0), box_objective(*UNIT, value, gradient),
                     RefineSettings(maxiter=20, repeat_maxiter=20, minimum_distinct_original_start_support=9))
    assert len(outcome.record["candidate_start_agreement"]["supporting_original_start_indices"]) == 8
    return outcome, {"support"}


REPEAT_SETTINGS = RefineSettings(original_start_count=2, gtol=1.0, repeat_log_likelihood_tolerance=0.01)
"""The starts stop where they begin (gtol above their gradient); the repeat (gtol 1e-12) does not."""


def _repeat_moves_the_best(box_objective):
    """The repeat lowers the best by 0.0499: above the repeat tolerance, inside the support one."""
    value, objective, starts = _slow_setup(box_objective)
    outcome = refine(starts, objective, REPEAT_SETTINGS)
    assert [run["observed_best_half_chi2"] for run in outcome.record["runs"]] == [
        pytest.approx(value(np.array(point)), rel=1.0e-12) for point in SLOW_POINTS]
    assert outcome.best_vector == pytest.approx(SLOW_TARGET, abs=1.0e-8)
    return outcome, {"repeat"}


def _repeat_raises(box_objective):
    """The objective fails from the repeat's first evaluation on (counted from a first run)."""
    value, objective, starts = _slow_setup(box_objective)
    multistart = sum(run["evaluation_count"] for run in refine(starts, objective, REPEAT_SETTINGS).record["runs"])
    calls = itertools.count(1)

    def fails_in_the_repeat(z):
        if next(calls) > multistart:
            raise FloatingPointError("the tighter repeat cannot evaluate")
        return objective.value_and_gradient(z)

    outcome = refine(starts, dataclasses.replace(objective, value_and_gradient=fails_in_the_repeat), REPEAT_SETTINGS)
    assert outcome.repeat_converged is None
    assert outcome.record["tighter_repeat"]["message"] == "FloatingPointError: the tighter repeat cannot evaluate"
    assert outcome.best_vector == tuple(SLOW_POINTS[0])
    return outcome, {"repeat"}


def _non_finite_gradient(box_objective):
    """The gradient is NaN beyond 0.15 in both coordinates, where the value keeps falling: the first
    step lands there and that point, finite in value, is retained as the best."""
    _, slow_gradient = quadratic(SLOW_TARGET, SLOW_WEIGHTS)
    _, objective, starts = _slow_setup(box_objective, lambda x: np.full(2, np.nan) if np.min(x) > 0.15
                                       else slow_gradient(x))
    outcome = refine(starts, objective, RefineSettings(original_start_count=2))
    assert min(outcome.best_vector) > 0.15
    assert outcome.record["candidate_best_gradient_valid"] is False and outcome.projected_gradient_linf is None
    return outcome, {"finite_gradient"}


def _scalar_residual_inconsistent(box_objective):
    """A residual callback of sqrt(2 f) + 1 gives |2 f - r . r| = 1 + 2 sqrt(2 f) at the best."""
    value, objective, starts, settings = _quadratic_setup(box_objective)
    outcome = refine(starts, dataclasses.replace(objective, residual=lambda z: np.sqrt([2.0 * value(z)]) + 1.0),
                     settings)
    half = outcome.record["candidate_best_half_chi2"]
    assert outcome.record["candidate_scalar_residual_error"] == pytest.approx(1.0 + 2.0 * np.sqrt(2.0 * half),
                                                                              rel=1.0e-12)
    return outcome, {"scalar_residual"}


def _direct_likelihood_inconsistent(box_objective):
    """The direct log L is 1e-3 below the implied one within 0.05 of the target and equal to it at
    the incumbent (0.13 away), so only the check at the best fails."""
    value, objective, starts, settings = _quadratic_setup(box_objective)
    offset = offset_direct_check(value, lambda z: 1.0e-3 if np.max(np.abs(z - np.asarray(TARGET))) < 0.05 else 0.0)
    outcome = refine(starts, dataclasses.replace(objective, direct_check=offset), settings)
    assert outcome.record["candidate_direct_log_likelihood_error"] == pytest.approx(1.0e-3, rel=1.0e-9)
    assert outcome.record["incumbent"]["passed"] is True
    return outcome, {"direct_log_likelihood"}


def _no_finite_evaluation(box_objective):
    value, gradient = objective_functions({"kind": "nan"})
    points = [[0.2, 0.3]] + [[0.2 + 0.05 * i, 0.3] for i in range(8)]
    outcome = refine(starts_from(points, *UNIT, 0.0), box_objective(*UNIT, value, gradient),
                     RefineSettings(maxiter=3, repeat_maxiter=3))
    assert (outcome.best_log_likelihood, outcome.best_vector) == (None, None)
    return outcome, set(outcome.gates)


GATE_FAILURES = {
    "incumbent_saved_mismatch": _incumbent_saved_mismatch, "unsupported_incumbent": _unsupported_incumbent,
    "support_below_minimum": _support_below_minimum, "repeat_moves_the_best": _repeat_moves_the_best,
    "repeat_raises": _repeat_raises, "non_finite_gradient": _non_finite_gradient,
    "scalar_residual_inconsistent": _scalar_residual_inconsistent,
    "direct_likelihood_inconsistent": _direct_likelihood_inconsistent, "no_finite_evaluation": _no_finite_evaluation,
}


@pytest.mark.parametrize("case", list(GATE_FAILURES))
def test_refinement_gates_reject_each_failure(case, box_objective):
    """Each defect fails exactly its own gate and leaves the role unresolved."""
    outcome, failed = GATE_FAILURES[case](box_objective)
    assert outcome.acceptance_status is RoleStatus.UNRESOLVED
    assert {name for name, passed in outcome.gates.items() if not passed} == failed


def test_starts_are_normalized_once(tmp_path, box_objective):
    """The reviewer's counterexample: prior [2, 4], two basins, sampler maximum at 2.5. A start whose
    unit-box vector leaves [0, 1] is refused when it is built."""
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
    with pytest.raises(ValueError, match="outside the unit box"):
        RefineStart(index=0, physical=(4.4,), normalized=(1.2,), source="sampler_ml", original=False, origin={})


@pytest.mark.parametrize("defect", ["other_box", "incumbent_not_first", "count"])
def test_bad_starts_fail_before_any_evaluation(defect, box_objective):
    evaluations = []

    def value(x):
        evaluations.append(list(x))
        return 0.0

    objective = box_objective([0.0], [1.0], value, lambda x: np.zeros(1))
    starts = starts_from([[0.4], [0.3]], [0.0], [1.0], 0.0)
    settings = RefineSettings(original_start_count=1)
    if defect == "other_box":
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
    """SCI-14: the six gates accept a maximum that repeats although it is far from stationary, and
    the record says how far.

    f(z) = 0.04 sum (z - 0.9)^2 on [0, 1]^2 changes by less than 0.065 over the box, so every
    start supports the best and the repeat passes after one iteration each, far from z = 0.9.
    """
    value, gradient = quadratic(SLOW_TARGET, SLOW_WEIGHTS)
    outcome = refine(starts_from(SLOW_POINTS, *UNIT, -value(np.array(SLOW_POINTS[0]))),
                     box_objective(*UNIT, value, gradient),
                     RefineSettings(original_start_count=2, maxiter=1, repeat_maxiter=1, maxls=1))
    assert outcome.acceptance_status is RoleStatus.ACCEPTED
    best = np.asarray(outcome.best_vector)
    assert outcome.projected_gradient_linf == pytest.approx(0.08 * float(np.max(np.abs(best - 0.9))), rel=1.0e-12)
    assert outcome.projected_gradient_linf > 0.04
    assert outcome.repeat_converged is False


def test_projected_gradient_ignores_outward_components_at_active_bounds(box_objective):
    """SCI-14 at the box edges: 0.5 |z - (1.2, -0.3)|^2 on [0, 1]^2 is lowest at the corner (1, 0),
    where the gradient (-0.2, 0.3) points out of the box in both components, so the projected
    gradient there is zero."""
    value, gradient = quadratic([1.2, -0.3])
    outcome = refine(starts_from(ACCEPT_POINTS, *UNIT, -value(np.array(ACCEPT_POINTS[0]))),
                     box_objective(*UNIT, value, gradient), RefineSettings(maxiter=100, repeat_maxiter=100))
    assert outcome.best_vector == (1.0, 0.0)
    assert outcome.record["candidate_best_gradient_z"] == pytest.approx([-0.2, 0.3], rel=1.0e-12)
    assert outcome.record["candidate_best_projected_gradient"]["values"] == [0.0, 0.0]
    assert outcome.projected_gradient_linf == 0.0


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
