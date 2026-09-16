"""Numerical and restart contracts for versioned nonlinear profiling."""
import json
from types import SimpleNamespace

import numpy as np
import pytest

from hwoslaps.modeling.nonlinear import local_profile
from hwoslaps.modeling.nonlinear.profile_replay import archive_vectors, linearized_comparator, atomic_json
from hwoslaps.modeling.nonlinear.profile_execution import BudgetLedger, memory_admissible, process_matches


def solver_result(x, success):
    return SimpleNamespace(x=np.array([x]), fun=np.array([x]), success=success,
                           status=1 if success else 0, message='test', nfev=3,
                           optimality=0., active_mask=np.array([0]))


def test_best_intermediate_survives_worse_successful_endpoint(monkeypatch):
    def solver(fun, x, **kwargs):
        fun(np.array([0.1]))
        return solver_result(1., True)
    monkeypatch.setattr(local_profile, 'least_squares', solver)
    fit = local_profile.fit_local_least_squares_profile(model_name='x', residual_fn=lambda x:x,
                                                       initial_points=[[2.]])
    assert fit.best.chi2 == pytest.approx(.01)
    assert fit.best.endpoint_chi2 == 1
    assert fit.best.residual_calls == 3


def test_no_success_preference_on_large_noisy_objective(monkeypatch):
    def residual(x):
        return np.array([1000., x[0]])
    def solver(fun, x, **kwargs):
        r = solver_result(x[0], bool(x[0]))
        r.fun = residual(x)
        return r
    monkeypatch.setattr(local_profile, 'least_squares', solver)
    fit = local_profile.fit_local_least_squares_profile(model_name='x', residual_fn=residual,
                                                       initial_points=[[0.], [.5]])
    assert not fit.best.success
    assert fit.best.chi2 == 1.e6


def test_failed_start_retains_initial_and_other_starts(monkeypatch):
    def solver(fun, x, **kwargs):
        if x[0] == 2:
            raise RuntimeError('one failed start')
        fun(np.array([0.]))
        return solver_result(0., True)
    monkeypatch.setattr(local_profile, 'least_squares', solver)
    fit = local_profile.fit_local_least_squares_profile(model_name='x', residual_fn=lambda x:x,
                                                       initial_points=[[2.], [1.]])
    assert fit.attempts[0].chi2 == 4
    assert not fit.attempts[0].success
    assert fit.chi2_min == 0


def test_no_admissible_point_is_failure():
    with pytest.raises(ValueError, match='No finite admissible'):
        local_profile.fit_local_least_squares_profile(model_name='x', residual_fn=lambda x: np.array([np.nan]),
                                                      initial_points=[[0.]])


def test_matched_comparator_profiles_common_nuisance_and_reports_bounds():
    residual = lambda x: np.array([3.-x[0], 4.])
    comparison = linearized_comparator(residual, np.array([0.]), np.zeros(2),
                                      np.array([-1.]), np.array([1.]))
    assert comparison['q'] == pytest.approx(16.)
    assert comparison['q_with_finite_prior_box'] == pytest.approx(20.)
    assert comparison['bounds_active'] == [1]


def test_background_convention_changes_comparator():
    residual = lambda x: np.array([1.-x[0], 1.+x[0], 1.])
    comparison = linearized_comparator(residual, np.array([0.]), np.zeros(3),
                                      np.array([-2.]), np.array([2.]), background_column=np.ones(3))
    assert comparison['q'] == pytest.approx(3.)
    assert comparison['q_with_free_background_only'] < 1.e-20


def test_budget_survives_restart_and_rejects_duplicate(tmp_path):
    path = tmp_path/'budget.json'
    ledger = BudgetLedger(path, 100)
    assert ledger.reserve('a', 80, {})
    assert not BudgetLedger(path, 100).reserve('b', 21, {})
    ledger.finish('a', 'TIMED_OUT', 70)
    fresh = BudgetLedger(path, 100)
    assert fresh.committed == 70
    assert fresh.reserve('b', 30, {})
    with pytest.raises(ValueError, match='already exists'):
        fresh.reserve('a', 1, {})
    with pytest.raises(ValueError, match='cap'):
        BudgetLedger(path, 101)


def test_delayed_gpu_visibility_reserves_expected_memory():
    assert not memory_admissible(0, [60], 30, 100)
    assert memory_admissible(0, [40], 30, 100)
    assert not memory_admissible(75, [], 10, 100)


def test_stale_or_missing_process_cannot_be_owned():
    assert not process_matches({'pid': 999999999, 'process_start': 0., 'spec_path': '/none'})


def test_start_selection_uses_actual_ml_and_scaled_distance(tmp_path):
    summary = tmp_path/'summary.json'
    summary.write_text(json.dumps({'arguments': {'max_log_likelihood_sample': {'arguments': {
        'log_likelihood': 10, 'kwargs': {'arguments': {'x': 1., 'y': 100.}}}}}}))
    csv = tmp_path/'samples.csv'
    csv.write_text('x,y,log_likelihood\n1,100,10\n1.001,100,9\n1,200,8\n2,100,7\n')
    starts, origins = archive_vectors(summary, csv, ['x','y'], np.array([0.,0.]),
                                     np.array([10.,1000.]), max_starts=3)
    assert len(starts) == 3
    assert origins[0]['origin'] == 'archived_ML'
    assert origins[1]['row'] == 2
    assert np.array_equal(starts[0], [1,100])


def test_atomic_json_preserves_previous_file_on_serialization_error(tmp_path):
    p = tmp_path/'state.json'
    atomic_json(p, {'state': 'old'})
    with pytest.raises(TypeError):
        atomic_json(p, {'bad': object()})
    assert json.loads(p.read_text()) == {'state': 'old'}
