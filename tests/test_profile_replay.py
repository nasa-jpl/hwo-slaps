"""Numerical and restart contracts for versioned nonlinear profiling."""

import json
from types import SimpleNamespace

import numpy as np
import pytest

from hwoslaps.modeling.nonlinear import local_profile
from hwoslaps.modeling.nonlinear.profile_replay import archive_vectors, linearized_comparator, atomic_json
from hwoslaps.modeling.nonlinear.profile_execution import BudgetLedger, memory_admissible, process_matches


def solver_result(x, success):
    return SimpleNamespace(
        x=np.array([x]),
        fun=np.array([x]),
        success=success,
        status=1 if success else 0,
        message="test",
        nfev=3,
        optimality=0.0,
        active_mask=np.array([0]),
    )


def test_best_intermediate_survives_worse_successful_endpoint(monkeypatch):
    def solver(fun, x, **kwargs):
        fun(np.array([0.1]))
        return solver_result(1.0, True)

    monkeypatch.setattr(local_profile, "least_squares", solver)
    fit = local_profile.fit_local_least_squares_profile(
        model_name="x", residual_fn=lambda x: x, initial_points=[[2.0]]
    )
    assert fit.best.chi2 == pytest.approx(0.01)
    assert fit.best.endpoint_chi2 == 1
    assert fit.best.residual_calls == 3


def test_no_success_preference_on_large_noisy_objective(monkeypatch):
    def residual(x):
        return np.array([1000.0, x[0]])

    def solver(fun, x, **kwargs):
        r = solver_result(x[0], bool(x[0]))
        r.fun = residual(x)
        return r

    monkeypatch.setattr(local_profile, "least_squares", solver)
    fit = local_profile.fit_local_least_squares_profile(
        model_name="x", residual_fn=residual, initial_points=[[0.0], [0.5]]
    )
    assert not fit.best.success
    assert fit.best.chi2 == 1.0e6


def test_failed_start_retains_initial_and_other_starts(monkeypatch):
    def solver(fun, x, **kwargs):
        if x[0] == 2:
            raise RuntimeError("one failed start")
        fun(np.array([0.0]))
        return solver_result(0.0, True)

    monkeypatch.setattr(local_profile, "least_squares", solver)
    fit = local_profile.fit_local_least_squares_profile(
        model_name="x", residual_fn=lambda x: x, initial_points=[[2.0], [1.0]]
    )
    assert fit.attempts[0].chi2 == 4
    assert not fit.attempts[0].success
    assert fit.chi2_min == 0


def test_no_admissible_point_is_failure():
    with pytest.raises(ValueError, match="No finite admissible"):
        local_profile.fit_local_least_squares_profile(
            model_name="x", residual_fn=lambda x: np.array([np.nan]), initial_points=[[0.0]]
        )


def test_matched_comparator_profiles_common_nuisance_and_reports_bounds():
    def residual(x):
        return np.array([3.0 - x[0], 4.0])

    comparison = linearized_comparator(
        residual, np.array([0.0]), np.zeros(2), np.array([-1.0]), np.array([1.0])
    )
    assert comparison["q"] == pytest.approx(16.0)
    assert comparison["q_with_finite_prior_box"] == pytest.approx(20.0)
    assert comparison["bounds_active"] == [1]


def test_background_convention_changes_comparator():
    def residual(x):
        return np.array([1.0 - x[0], 1.0 + x[0], 1.0])

    comparison = linearized_comparator(
        residual,
        np.array([0.0]),
        np.zeros(3),
        np.array([-2.0]),
        np.array([2.0]),
        background_column=np.ones(3),
    )
    assert comparison["q"] == pytest.approx(3.0)
    assert comparison["q_with_free_background_only"] < 1.0e-20


def test_budget_survives_restart_and_rejects_duplicate(tmp_path):
    path = tmp_path / "budget.json"
    ledger = BudgetLedger(path, 100)
    assert ledger.reserve("a", 80, {})
    assert not BudgetLedger(path, 100).reserve("b", 21, {})
    ledger.finish("a", "TIMED_OUT", 70)
    fresh = BudgetLedger(path, 100)
    assert fresh.committed == 70
    assert fresh.reserve("b", 30, {})
    with pytest.raises(ValueError, match="already exists"):
        fresh.reserve("a", 1, {})
    with pytest.raises(ValueError, match="cap"):
        BudgetLedger(path, 101)


def test_delayed_gpu_visibility_reserves_expected_memory():
    assert not memory_admissible(0, [60], 30, 100)
    assert memory_admissible(0, [40], 30, 100)
    assert not memory_admissible(75, [], 10, 100)


def test_stale_or_missing_process_cannot_be_owned():
    assert not process_matches({"pid": 999999999, "process_start": 0.0, "spec_path": "/none"})


def test_start_selection_uses_actual_ml_and_scaled_distance(tmp_path):
    summary = tmp_path / "summary.json"
    summary.write_text(
        json.dumps(
            {
                "arguments": {
                    "max_log_likelihood_sample": {
                        "arguments": {"log_likelihood": 10, "kwargs": {"arguments": {"x": 1.0, "y": 100.0}}}
                    }
                }
            }
        )
    )
    csv = tmp_path / "samples.csv"
    csv.write_text("x,y,log_likelihood\n1,100,10\n1.001,100,9\n1,200,8\n2,100,7\n")
    starts, origins = archive_vectors(
        summary, csv, ["x", "y"], np.array([0.0, 0.0]), np.array([10.0, 1000.0]), max_starts=3
    )
    assert len(starts) == 3
    assert origins[0]["origin"] == "archived_ML"
    assert origins[1]["row"] == 2
    assert np.array_equal(starts[0], [1, 100])


def test_atomic_json_preserves_previous_file_on_serialization_error(tmp_path):
    p = tmp_path / "state.json"
    atomic_json(p, {"state": "old"})
    with pytest.raises(TypeError):
        atomic_json(p, {"bad": object()})
    assert json.loads(p.read_text()) == {"state": "old"}


def controller_fixture(tmp_path, monkeypatch, worker_body, timeout=10):
    import sys
    import shutil
    import subprocess
    import psutil

    root = tmp_path / "task"
    root.mkdir()
    work = tmp_path / "work"
    (work / "scripts").mkdir(parents=True)
    script = work / "scripts/run_nonlinear_profile.py"
    script.write_text(
        "import sys,json,time\nfrom pathlib import Path\n"
        'spec=json.loads(Path(sys.argv[1]).read_text())\nout=Path(spec["output"])\n' + worker_body
    )
    spec = root / "job.json"
    spec.write_text(json.dumps({"output": str(root / "attempt")}))
    manifest = root / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "task_root": str(root),
                "max_workers": 1,
                "gpus": [2],
                "cap_seconds": 100,
                "python": sys.executable,
                "worktree": str(work),
                "campaign_uuid": "test",
                "jobs": [
                    {"key": "a", "spec": str(spec), "gpu": 2, "peak_mib": 100, "timeout_seconds": timeout}
                ],
            }
        )
    )
    monkeypatch.setattr(
        subprocess,
        "check_output",
        lambda cmd, **k: "" if "--query-compute-apps=pid,gpu_uuid" in cmd else "2, GPU-test, 1000, 0\n",
    )
    monkeypatch.setattr(psutil, "virtual_memory", lambda: SimpleNamespace(available=1024 * 2**30))
    monkeypatch.setattr(shutil, "disk_usage", lambda _: SimpleNamespace(free=100 * 2**30))
    return root, manifest


def test_controller_completes_and_restart_does_not_repeat(tmp_path, monkeypatch):
    from hwoslaps.modeling.nonlinear.profile_execution import supervise

    root, manifest = controller_fixture(
        tmp_path,
        monkeypatch,
        '(out/"worker_exit.json").write_text(json.dumps('
        '{"status":"COMPLETE","elapsed_s":0.2,"artifacts":{}}))\n',
    )
    supervise(manifest)
    before = (root / "state/budget.json").read_text()
    supervise(manifest)
    assert (root / "state/budget.json").read_text() == before
    assert json.loads(before)["attempts"]["a"]["status"] == "COMPLETE"


def test_controller_timeout_preserves_charge_and_stops_owned_process(tmp_path, monkeypatch):
    from hwoslaps.modeling.nonlinear.profile_execution import supervise

    root, manifest = controller_fixture(tmp_path, monkeypatch, "time.sleep(30)\n", timeout=0.1)
    with pytest.raises(RuntimeError, match="Manifest failure"):
        supervise(manifest)
    item = json.loads((root / "state/budget.json").read_text())["attempts"]["a"]
    assert item["status"] == "TIMED_OUT"
    assert item["charged_seconds"] > 0
    assert not process_matches(item)


def test_controller_exclusive_ownership(tmp_path):
    import fcntl
    from hwoslaps.modeling.nonlinear.profile_execution import supervise

    state = tmp_path / "state"
    state.mkdir()
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"task_root": str(tmp_path)}))
    with (state / "controller.lock").open("w") as owner:
        fcntl.flock(owner, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(BlockingIOError):
            supervise(manifest)


def test_compiled_residual_unwraps_autoarray_without_host_conversion():
    import jax
    import jax.numpy as jnp
    from hwoslaps.modeling.nonlinear.profile_replay import residual_array

    class Wrapper:
        def __init__(self, array):
            self.array = array

        def __array__(self, *args):
            raise AssertionError("host conversion")

    compiled = jax.jit(lambda x: residual_array(Wrapper(x * 2), jnp))
    assert np.array_equal(compiled(np.array([1.0, 2.0])), [2.0, 4.0])


def test_dataset_fingerprint_accepts_runtime_convolver():
    from hwoslaps.modeling.nonlinear.profile_replay import array_identity

    kernel = np.arange(9.0, dtype=np.float64).reshape(3, 3)
    convolver = SimpleNamespace(kernel=SimpleNamespace(native=kernel))
    assert array_identity(convolver) == array_identity(kernel)


def test_explicit_jacobian_and_disabled_relative_cost_stop():
    fit = local_profile.fit_local_least_squares_profile(
        model_name="analytic",
        residual_fn=lambda x: np.array([1000.0, x[0] - 2.0]),
        jacobian_fn=lambda x: np.array([[0.0], [1.0]]),
        initial_points=[[0.0]],
        lower_bounds=[-10.0],
        upper_bounds=[10.0],
        ftol=None,
        xtol=1e-12,
        gtol=1e-12,
    )
    assert fit.best.x == pytest.approx([2.0], abs=1e-8)
    assert fit.best.jacobian_calls > 0
    assert fit.best.residual_calls >= fit.best.jacobian_calls


def test_best_mode_stability_keeps_worse_local_optima_visible():
    from hwoslaps.modeling.nonlinear.profile_replay import profile_stability

    attempts = [SimpleNamespace(label=str(i), chi2=x) for i, x in enumerate([1.0, 1.01, 30.0])]
    report = profile_stability(attempts, 1.0, 1.0, 0.01)
    assert report["stable"]
    assert report["supporting_starts"] == ["0", "1"]
    assert report["start_best_logL_spread"] == 14.5
    assert not profile_stability(attempts, 1.0, 0.5, 0.01)["stable"]
    assert not profile_stability(attempts[:1], 1.0, 1.0, 0.01)["stable"]


def test_controller_rejects_modified_completion_artifact(tmp_path, monkeypatch):
    from hwoslaps.modeling.nonlinear.profile_execution import supervise

    body = (
        'artifact=out/"science.json"\nartifact.write_text("changed")\n'
        '(out/"worker_exit.json").write_text(json.dumps({"status":"COMPLETE",'
        '"elapsed_s":0.2,"artifacts":{str(artifact):"wrong-digest"}}))\n'
    )
    root, manifest = controller_fixture(tmp_path, monkeypatch, body)
    with pytest.raises(RuntimeError, match="Manifest failure"):
        supervise(manifest)
    ledger = json.loads((root / "state/budget.json").read_text())
    assert ledger["attempts"]["a"]["status"] == "FAILED_ARTIFACT_INTEGRITY"


def test_controller_excludes_foreign_gpu_process_with_small_allocation(tmp_path, monkeypatch):
    import subprocess
    from hwoslaps.modeling.nonlinear.profile_execution import supervise

    root, manifest = controller_fixture(tmp_path, monkeypatch, "raise RuntimeError('must not launch')\n")
    monkeypatch.setattr(
        subprocess,
        "check_output",
        lambda cmd, **kwargs: (
            "123456, GPU-test\n" if "--query-compute-apps=pid,gpu_uuid" in cmd else "2, GPU-test, 1000, 1\n"
        ),
    )
    supervise(manifest)
    assert not (root / "attempt").exists()
    assert json.loads((root / "state/budget.json").read_text())["committed_seconds"] == 0
    assert (root / "state/dispatch_blocked.json").exists()


def test_jacobian_check_refines_step_without_loosening_error_limit():
    from hwoslaps.modeling.nonlinear.profile_replay import verify_residual_jacobian

    def residual(x):
        return np.array([x[0] + 1.0e10 * x[0] ** 3])

    result = verify_residual_jacobian(residual, np.ones((1, 1)), np.zeros(1), -np.ones(1), np.ones(1), 1.0e-3)
    assert result["passed"]
    assert not result["step_trials"][0]["passed"]
    assert len(result["step_trials"]) >= 3
    bad = verify_residual_jacobian(
        residual, np.full((1, 1), 5.0), np.zeros(1), -np.ones(1), np.ones(1), 1.0e-3
    )
    assert not bad["passed"]


def test_interruption_keeps_best_earlier_start_and_its_completed_record(monkeypatch):
    from hwoslaps.modeling.nonlinear.profile_replay import update_profile_checkpoint

    record = {}
    completed = []

    def solver(fun, x, **kwargs):
        if x[0] == 3.0:
            raise KeyboardInterrupt()
        fun(np.array([0.0]))
        return solver_result(0.0, True)

    monkeypatch.setattr(local_profile, "least_squares", solver)
    with pytest.raises(KeyboardInterrupt):
        local_profile.fit_local_least_squares_profile(
            model_name="interrupt",
            residual_fn=lambda x: x,
            initial_points=[[2.0], [3.0]],
            progress_callback=lambda point: update_profile_checkpoint(record, point, "multistart"),
            attempt_callback=lambda attempt: completed.append(attempt.to_dict()),
        )
    assert record["partial_best"]["chi2"] == 0.0
    assert record["start_progress"]["multistart:start_1"]["chi2"] == 9.0
    assert len(completed) == 1 and completed[0]["chi2"] == 0.0


def test_retained_points_are_replayed_and_not_counted_as_independent(tmp_path):
    import hashlib
    from hwoslaps.modeling.nonlinear.profile_replay import verified_retained_points

    source = tmp_path / "previous.json"
    previous = {
        "smooth": {
            "identity": {"passed": True, "analysis_key": "same", "parameter_names": ["x"]},
            "best_vector": [0.1],
            "best_chi2": 0.01,
            "logL": -0.005,
        }
    }
    source.write_text(json.dumps(previous))
    reference = {"path": str(source), "sha256": hashlib.sha256(source.read_bytes()).hexdigest()}
    args = (
        [reference],
        "smooth",
        ["x"],
        np.array([-2.0]),
        np.array([2.0]),
        "same",
        lambda x: x,
        lambda x: -float(x @ x) / 2,
        1e-4,
    )
    retained = verified_retained_points(*args)
    assert retained[0]["chi2"] == pytest.approx(0.01)
    assert not retained[0]["independent_start"]
    with pytest.raises(ValueError, match="runtime objective"):
        verified_retained_points(*args[:-2], lambda x: 10.0, args[-1])
    source.write_text("modified")
    with pytest.raises(ValueError, match="hash mismatch"):
        verified_retained_points(*args)


def test_near_boundary_initialization_preserves_free_box_and_other_coordinates():
    from hwoslaps.modeling.nonlinear.profile_replay import near_boundary_starts

    lo = np.array([-2.0, 0.0, -1.0])
    hi = np.array([2.0, 10.0, 1.0])
    before_lo = lo.copy()
    before_hi = hi.copy()
    starts = [np.array([0.1, 4.0, 0.2]), np.array([0.3, 6.0, 0.4])]
    changed, indices = near_boundary_starts(starts, np.array([0.0, 9.9999999, 0.0]), lo, hi)
    assert indices == [1]
    assert np.array_equal(lo, before_lo) and np.array_equal(hi, before_hi)
    assert changed[0][0] == 0.1 and changed[1][2] == 0.4
    assert np.all(changed[0] >= lo) and np.all(changed[0] <= hi)
    assert starts[0][1] == 4.0


def test_completed_start_replay_checks_values_and_avoids_duplicate_solver_work():
    from hwoslaps.modeling.nonlinear.profile_replay import replay_completed_profile

    origins = [{"origin": "data_a"}, {"origin": "data_b"}]
    previous = {
        "stable": True,
        "starts": origins,
        "profile": {
            "attempts": [
                {"label": "start_0", "x": [0.1], "chi2": 0.01, "success": False, "status": 0},
                {"label": "start_1", "x": [0.2], "chi2": 0.04},
            ]
        },
    }
    result = replay_completed_profile(
        previous, origins, lambda x: x, lambda x: -float(x @ x) / 2, 1e-4, "smooth"
    )
    assert result.chi2_min == pytest.approx(0.01)
    assert all(a.nfev == 0 and a.residual_calls == 1 for a in result.attempts)
    assert not result.attempts[0].success
    with pytest.raises(ValueError, match="residual identity"):
        replay_completed_profile(previous, origins, lambda x: x + 1, lambda x: 0, 1e-4, "smooth")


def test_live_budget_uses_monotonic_time_despite_wall_clock_jump(monkeypatch):
    from hwoslaps.modeling.nonlinear import profile_execution as execution

    monkeypatch.setattr(execution, "clock_epoch", lambda: "same_boot")
    monkeypatch.setattr(execution.time, "monotonic", lambda: 150.0)
    monkeypatch.setattr(execution.time, "time", lambda: -999999.0)
    item = {
        "clock_epoch": "same_boot",
        "start_monotonic": 100.0,
        "start_unix": 500.0,
        "reservation_seconds": 1000.0,
    }
    assert execution.attempt_elapsed(item) == 50.0
    item["clock_epoch"] = "old_boot"
    assert execution.attempt_elapsed(item) == 1000.0
