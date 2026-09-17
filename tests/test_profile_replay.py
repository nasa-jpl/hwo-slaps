"""Numerical and restart contracts for versioned nonlinear profiling."""

import json
from types import SimpleNamespace

import numpy as np
import pytest

from hwoslaps.modeling.nonlinear import local_profile
from hwoslaps.modeling.nonlinear.profile_replay import archive_vectors, linearized_comparator, atomic_json
from hwoslaps.modeling.nonlinear.profile_execution import (
    BudgetLedger,
    memory_admissible,
    process_matches,
)


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


def test_saved_point_replay_skips_all_tangent_evaluations():
    from hwoslaps.modeling.nonlinear.profile_replay import replay_comparator

    def forbidden_residual(_):
        raise AssertionError("A saved-point check must not evaluate tangent perturbations")

    runner = SimpleNamespace(procedure={"mode": "identity", "compute_comparator": False})
    result = replay_comparator(runner, forbidden_residual, None, None, None, None, None)
    assert result["computed"] is False
    assert "q" not in result
    runner.procedure["mode"] = "profile"
    with pytest.raises(ValueError, match="identity-only"):
        replay_comparator(runner, forbidden_residual, None, None, None, None, None)


def test_default_replay_comparator_keeps_historical_quantities():
    from hwoslaps.modeling.nonlinear.profile_replay import replay_comparator

    def residual(x):
        return np.array([3.0 - x[0], 4.0])
    runner = SimpleNamespace(
        procedure={"mode": "identity", "comparator_tolerance": .001},
        replay={"case": {"rung": {"q_f_production_at_position": 20., "q_f_matched": 16.}}},
    )
    result = replay_comparator(
        runner, residual, np.zeros(1), np.zeros(2), -np.ones(1), np.ones(1), np.ones(2),
    )
    assert result["computed"] is True
    assert result["q"] == pytest.approx(16.)
    assert result["q_F_production"] == 20.
    assert result["q_F_support_matched"] == 16.


def test_direct_identity_replay_matches_archive_without_jit(tmp_path, monkeypatch):
    import jax
    import jax.numpy as jnp
    from hwoslaps.modeling.nonlinear.autolens_runner import NonlinearSearchSettings
    from hwoslaps.modeling.nonlinear.profile_replay import ProfileReplayRunner

    summary = tmp_path / "summary.json"
    value = -.5 * .75**2
    summary.write_text(json.dumps({"arguments": {"max_log_likelihood_sample": {
        "arguments": {"kwargs": {"arguments": {"x": .25}}, "log_likelihood": value}}}}))

    class Model:
        unique_prior_paths = [("x",)]
        priors_ordered_by_id = [SimpleNamespace(lower_limit=0., upper_limit=1.)]

        def instance_from_vector(self, vector, xp):
            return SimpleNamespace(x=xp.asarray(vector)[0])

    class Analysis:
        dataset = SimpleNamespace(
            data=SimpleNamespace(native=np.zeros(1)), noise_map=SimpleNamespace(native=np.ones(1)),
            mask=np.zeros(1, dtype=bool), psf=np.ones(1),
        )

        def fit_from(self, instance):
            return SimpleNamespace(normalized_residual_map=jnp.array([1. - instance.x]))

        def log_likelihood_function(self, instance):
            return -.5 * (1. - instance.x)**2

    def forbidden_jit(*args, **kwargs):
        raise AssertionError("Direct saved-point evaluation must not create a whole-model JIT")

    monkeypatch.setattr(jax, "jit", forbidden_jit)
    replay = {"case": {"case": {"smooth_fit": {"analysis_key": "same", "log_likelihood_max": value}}},
              "smooth": {"summary": str(summary)}}
    procedure = {"mode": "identity", "direct_point_evaluation": True, "jacobian": "none",
                 "max_starts": 1, "start_separation": .05, "identity_tolerance": 1e-4}
    runner = ProfileReplayRunner(NonlinearSearchSettings(), tmp_path, replay, procedure)
    result = runner.run_model(model=Model(), analysis=Analysis(), role="smooth",
                              fit_mode="smooth", case_id="toy", analysis_key="same")
    identity = runner.records["smooth"]["identity"]
    assert result.log_likelihood_max == pytest.approx(value)
    assert identity["passed"] and not identity["compiled_evaluation_performed"]
    assert identity["direct_compiled_logL_error"] is None
    assert identity["direct_compiled_squared_residual_error"] is None
    assert runner.records["smooth"]["best_chi2"] == pytest.approx(.75**2)
    runner.procedure["mode"] = "profile"
    with pytest.raises(ValueError, match="identity-only"):
        runner.run_model(model=Model(), analysis=Analysis(), role="smooth",
                         fit_mode="smooth", case_id="toy", analysis_key="same")


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


def test_packing_admission_stays_within_frozen_limits(tmp_path):
    from hwoslaps.modeling.nonlinear.profile_execution import concurrency_limits

    manifest = {"max_workers": 16, "max_workers_per_gpu": 4, "gpus": [0, 1, 2, 3]}
    assert concurrency_limits(manifest, tmp_path) == (16, 4)
    atomic_json(tmp_path / "concurrency.json", {"max_workers": 8, "max_workers_per_gpu": 2})
    assert concurrency_limits(manifest, tmp_path) == (8, 2)
    atomic_json(tmp_path / "concurrency.json", {"max_workers": 16, "max_workers_per_gpu": 4})
    assert concurrency_limits(manifest, tmp_path) == (16, 4)
    atomic_json(tmp_path / "concurrency.json", {"max_workers": 17, "max_workers_per_gpu": 4})
    with pytest.raises(ValueError, match="frozen worker limits"):
        concurrency_limits(manifest, tmp_path)


def test_stage3_v7_policy_allows_three_per_card_and_four_card_fallback():
    from hwoslaps.modeling.nonlinear.profile_execution import stage3_policy

    full = {
        "execution_policy_version": "stage3_v7",
        "authorized_gpu_limit": 8,
        "gpus": list(range(8)),
        "authorized_worker_limit": 24,
        "max_workers": 24,
        "max_workers_per_gpu": 3,
        "admission_memory_fraction": 0.85,
        "runtime_gpu_memory_fraction": 0.90,
    }
    assert stage3_policy(full)["worker_limit"] == 24
    fallback = dict(
        full,
        authorized_gpu_limit=4,
        gpus=list(range(4)),
        authorized_worker_limit=12,
        max_workers=12,
    )
    assert stage3_policy(fallback)["per_gpu_limit"] == 3
    with pytest.raises(ValueError, match="Stage 3"):
        stage3_policy(dict(full, max_workers_per_gpu=4))
    with pytest.raises(ValueError, match="Stage 3"):
        stage3_policy(dict(full, authorized_worker_limit=25, max_workers=25))
    with pytest.raises(ValueError, match="Stage 3"):
        stage3_policy(dict(full, max_workers_per_gpu=True))


def test_stage3_v7_memory_policy_requires_calibration_for_unknown_class():
    from hwoslaps.modeling.nonlinear.profile_execution import stage3_policy, validate_stage3_job

    manifest = {
        "execution_policy_version": "stage3_v7",
        "authorized_gpu_limit": 4,
        "gpus": list(range(4)),
        "authorized_worker_limit": 12,
        "max_workers": 12,
        "max_workers_per_gpu": 3,
        "admission_memory_fraction": 0.85,
        "runtime_gpu_memory_fraction": 0.90,
    }
    policy = stage3_policy(manifest)
    validate_stage3_job(
        {
            "peak_mib": 51200,
            "memory_class": "790",
            "memory_profile_id": "stage3_b200_790_v1",
            "image_shape": [790, 790],
            "kernel_shape": [51, 51],
            "batch_size": 32,
            "precision": "float64",
        },
        policy,
    )
    with pytest.raises(ValueError, match="registry"):
        validate_stage3_job(
            {
                "peak_mib": 51000,
                "memory_class": "790",
                "memory_profile_id": "stage3_b200_790_v1",
                "image_shape": [790, 790],
                "kernel_shape": [51, 51],
                "batch_size": 32,
                "precision": "float64",
            },
            policy,
        )
    with pytest.raises(ValueError, match="registry"):
        validate_stage3_job(
            {
                "peak_mib": 51200,
                "memory_class": "790",
                "image_shape": [790, 790],
                "kernel_shape": [51, 51],
                "batch_size": 32,
                "precision": "float64",
            },
            policy,
        )
    with pytest.raises(ValueError, match="registry"):
        validate_stage3_job({"peak_mib": 40000, "memory_class": "unknown"}, policy)
    with pytest.raises(ValueError, match="exclusive"):
        validate_stage3_job(
            {"peak_mib": 140000, "memory_class": "unmeasured_conservative", "exclusive_gpu": False},
            policy,
        )
    validate_stage3_job(
        {"peak_mib": 140000, "memory_class": "unmeasured_conservative", "exclusive_gpu": True},
        policy,
    )


def test_stage3_v7_memory_fraction_is_distinct_from_legacy_default():
    assert memory_admissible(0, [50000], 50000, 183359, 0.85)
    assert not memory_admissible(0, [50000], 110000, 183359, 0.85)


def test_stage3_card_memory_stop_is_local_and_persistent(tmp_path, monkeypatch):
    from hwoslaps.modeling.nonlinear import profile_execution as execution

    stopped = []
    charged = []
    monkeypatch.setattr(execution, "stop_owned", lambda item: stopped.append(item["pid"]))
    monkeypatch.setattr(execution, "attempt_elapsed", lambda item: 13.0)

    class Ledger:
        def finish(self, key, status, elapsed):
            charged.append((key, status, elapsed))

    active = {
        "a": {"gpu": 0, "pid": 1},
        "b": {"gpu": 2, "pid": 2},
        "c": {"gpu": 2, "pid": 3},
    }
    gpus = {
        0: {"used": 30, "total": 100},
        2: {"used": 91, "total": 100},
    }
    blocked = set()
    assert execution.stop_overfull_cards(active, gpus, Ledger(), blocked, tmp_path, 0.90)
    assert stopped == [2, 3] and blocked == {2}
    assert charged == [
        ("b", "STOPPED_CARD_MEMORY_LIMIT", 13.0),
        ("c", "STOPPED_CARD_MEMORY_LIMIT", 13.0),
    ]
    receipt = json.loads((tmp_path / "card_memory_events.jsonl").read_text())
    assert receipt["blocked_cards"] == [2]
    assert receipt["other_cards_preserved"]
    assert not execution.stop_overfull_cards({"a": active["a"]}, gpus, Ledger(), blocked, tmp_path, 0.90)


def test_stage3_task_disk_accounting_is_periodically_cached(tmp_path, monkeypatch):
    from hwoslaps.modeling.nonlinear import profile_execution as execution

    data = tmp_path / "artifact.bin"
    data.write_bytes(b"first")
    clock = iter((10.0, 10.5, 41.0))
    monkeypatch.setattr(execution.time, "monotonic", lambda: next(clock))
    cache = {"sample_monotonic": None, "bytes": 0}
    first, first_at = execution.cached_task_bytes(tmp_path, cache, 30)
    assert first == 5 and first_at == 10.0
    data.write_bytes(b"second-size")
    cached, cached_at = execution.cached_task_bytes(tmp_path, cache, 30)
    assert cached == first and cached_at == first_at
    refreshed, refreshed_at = execution.cached_task_bytes(tmp_path, cache, 30)
    assert refreshed == len(b"second-size") and refreshed_at == 41.0


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


def test_stage3_independent_failure_preserves_unrelated_job(tmp_path, monkeypatch):
    import shutil
    import subprocess
    import time
    from hwoslaps.modeling.nonlinear.profile_execution import supervise
    from hwoslaps.modeling.nonlinear.profile_execution import clock_epoch

    body = (
        'if out.name == "attempt0":\n'
        '    raise RuntimeError("independent failure")\n'
        'time.sleep(0.1)\n'
        '(out/"worker_exit.json").write_text(json.dumps('
        '{"status":"COMPLETE","elapsed_s":0.1,"artifacts":{}}))\n'
    )
    root, manifest = controller_fixture(tmp_path, monkeypatch, body)
    data = json.loads(manifest.read_text())
    data.update(
        execution_policy_version="stage3_v7",
        authorized_gpu_limit=4,
        authorized_worker_limit=12,
        max_workers=2,
        max_workers_per_gpu=3,
        gpus=[0, 1, 2, 3],
        cap_seconds=1000,
        admission_memory_fraction=0.85,
        runtime_gpu_memory_fraction=0.90,
    )
    first_spec = root / "job0.json"
    first_spec.write_text(json.dumps({"output": str(root / "attempt0")}))
    second_spec = root / "job1.json"
    second_spec.write_text(json.dumps({"output": str(root / "attempt1")}))
    data["jobs"] = [
        {
            "key": "fail",
            "spec": str(first_spec),
            "gpu": 0,
            "peak_mib": 51200,
            "memory_class": "790",
            "memory_profile_id": "stage3_b200_790_v1",
            "image_shape": [790, 790],
            "kernel_shape": [51, 51],
            "batch_size": 32,
            "precision": "float64",
            "timeout_seconds": 10,
        },
        {
            "key": "survive",
            "spec": str(second_spec),
            "gpu": 1,
            "peak_mib": 51200,
            "memory_class": "790",
            "memory_profile_id": "stage3_b200_790_v1",
            "image_shape": [790, 790],
            "kernel_shape": [51, 51],
            "batch_size": 32,
            "precision": "float64",
            "timeout_seconds": 10,
        },
    ]
    manifest.write_text(json.dumps(data))
    monkeypatch.setattr(
        subprocess,
        "check_output",
        lambda cmd, **kwargs: (
            ""
            if "--query-compute-apps=pid,gpu_uuid" in cmd
            else "".join(f"{g}, GPU-test{g}, 183359, 0\n" for g in range(4))
        ),
    )
    monkeypatch.setattr(shutil, "disk_usage", lambda _: SimpleNamespace(free=100 * 2**30))
    state = root / "state"
    state.mkdir()
    now = time.monotonic()
    (state / "deadline.json").write_text(
        json.dumps(
            {
                "clock_epoch": clock_epoch(),
                "captured_monotonic": now,
                "admission_stop_monotonic": now + 100,
                "hard_stop_monotonic": now + 200,
            }
        )
    )
    supervise(manifest)
    attempts = json.loads((root / "state/budget.json").read_text())["attempts"]
    assert attempts["fail"]["status"] == "INTERRUPTED_UNCERTAIN"
    assert attempts["survive"]["status"] == "COMPLETE"


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


@pytest.mark.parametrize(
    "limit,workers,gpus", [(4, 8, list(range(8))), (8, 9, list(range(8))), (8, 8, [0] * 8)]
)
def test_controller_rejects_excess_or_duplicate_allocation(tmp_path, monkeypatch, limit, workers, gpus):
    from hwoslaps.modeling.nonlinear.profile_execution import supervise

    root, manifest = controller_fixture(tmp_path, monkeypatch, "raise RuntimeError('must not launch')\n")
    data = json.loads(manifest.read_text())
    data.update(authorized_gpu_limit=limit, max_workers=workers, gpus=gpus)
    manifest.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="Invalid GPU allocation"):
        supervise(manifest)
    assert not (root / "attempt").exists()


@pytest.mark.parametrize("count,cards", [(8, 8), (16, 4), (32, 4)])
def test_controller_eight_concurrent_workers_and_restart(tmp_path, monkeypatch, count, cards):
    import subprocess
    from hwoslaps.modeling.nonlinear.profile_execution import supervise

    body = (
        '(out/"ready").write_text("ready")\n'
        "deadline=time.monotonic()+8\n"
        f'while len(list(out.parent.glob("attempt*/ready"))) < {count}:\n'
        '    assert time.monotonic() < deadline, "workers did not run concurrently"\n'
        "    time.sleep(0.05)\n"
        '(out/"worker_exit.json").write_text(json.dumps('
        '{"status":"COMPLETE","elapsed_s":0.1,"artifacts":{}}))\n'
    )
    root, manifest = controller_fixture(tmp_path, monkeypatch, body)
    data = json.loads(manifest.read_text())
    data.update(authorized_gpu_limit=cards, max_workers=count, gpus=list(range(cards)), cap_seconds=5000)
    if count > cards:
        data.update(authorized_worker_limit=count, max_workers_per_gpu=count // cards)
    data["jobs"] = []
    for index in range(count):
        spec = root / f"job{index}.json"
        spec.write_text(json.dumps({"output": str(root / f"attempt{index}")}))
        data["jobs"].append(dict(key=f"job{index}", spec=str(spec), gpu=index % cards,
                                 peak_mib=100, timeout_seconds=10))
    manifest.write_text(json.dumps(data))
    monkeypatch.setattr(
        subprocess,
        "check_output",
        lambda cmd, **k: (
            ""
            if "--query-compute-apps=pid,gpu_uuid" in cmd
            else "".join(f"{g}, GPU-test{g}, 1000, 0\n" for g in range(cards))
        ),
    )
    supervise(manifest)
    before = (root / "state/budget.json").read_text()
    assert all(x["status"] == "COMPLETE" for x in json.loads(before)["attempts"].values())
    assert len(json.loads(before)["attempts"]) == count
    assert {v["gpu"] for v in json.loads(before)["attempts"].values()} == set(range(cards))
    supervise(manifest)
    assert (root / "state/budget.json").read_text() == before


def test_controller_deadline_rejects_late_admission(tmp_path, monkeypatch):
    import time
    from hwoslaps.modeling.nonlinear.profile_execution import supervise, clock_epoch

    root, manifest = controller_fixture(tmp_path, monkeypatch, "raise RuntimeError('must not launch')\n")
    state = root / "state"
    state.mkdir()
    (state / "deadline.json").write_text(
        json.dumps(
            dict(
                clock_epoch=clock_epoch(),
                captured_monotonic=time.monotonic(),
                admission_stop_monotonic=time.monotonic() - 1,
                hard_stop_monotonic=time.monotonic() + 100,
            )
        )
    )
    supervise(manifest)
    assert not (root / "attempt").exists()
    assert (state / "dispatch_blocked.json").exists()


def test_controller_deadline_stops_owned_worker(tmp_path, monkeypatch):
    import time
    from hwoslaps.modeling.nonlinear.profile_execution import supervise, clock_epoch

    root, manifest = controller_fixture(tmp_path, monkeypatch, "time.sleep(30)\n")
    state = root / "state"
    state.mkdir()
    # Admit first, then advance the clock after the first loop sleep.
    origin = time.monotonic()
    real_sleep = time.sleep
    offset = [0.0]
    real_clock = time.monotonic
    monkeypatch.setattr(time, "monotonic", lambda: real_clock() + offset[0])

    def advance(seconds):
        if seconds == 2:
            offset[0] = 200
        else:
            real_sleep(min(seconds, 0.05))

    monkeypatch.setattr(time, "sleep", advance)
    (state / "deadline.json").write_text(
        json.dumps(
            dict(
                clock_epoch=clock_epoch(),
                captured_monotonic=origin,
                admission_stop_monotonic=origin + 80,
                hard_stop_monotonic=origin + 100,
            )
        )
    )
    with pytest.raises(RuntimeError, match="deadline"):
        supervise(manifest)
    item = json.loads((state / "budget.json").read_text())["attempts"]["a"]
    assert item["status"] == "STOPPED_AFTER_FAILURE"
    assert not process_matches(item)


def test_deadline_validation_rejects_nonfinite_and_reversed_windows():
    import math
    from hwoslaps.modeling.nonlinear.profile_execution import validate_deadline

    base = {
        "clock_epoch": "epoch",
        "captured_monotonic": 10.0,
        "admission_stop_monotonic": 20.0,
        "hard_stop_monotonic": 30.0,
    }
    for field in ("captured_monotonic", "admission_stop_monotonic", "hard_stop_monotonic"):
        for value in (math.nan, math.inf, -math.inf):
            candidate = dict(base, **{field: value})
            with pytest.raises(ValueError, match="finite"):
                validate_deadline(candidate, epoch="epoch")
    with pytest.raises(ValueError, match="later"):
        validate_deadline(dict(base, admission_stop_monotonic=31.0), epoch="epoch")
    with pytest.raises(ValueError, match="captured"):
        validate_deadline(dict(base, captured_monotonic=31.0), epoch="epoch")


def test_stop_file_blocks_controller_without_deadline(tmp_path, monkeypatch):
    from hwoslaps.modeling.nonlinear.profile_execution import supervise

    root, manifest = controller_fixture(tmp_path, monkeypatch, "raise RuntimeError('must not launch')\n")
    state = root / "state"
    state.mkdir()
    (state / "STOP").write_text("test stop\n")
    with pytest.raises(RuntimeError, match="explicit stop"):
        supervise(manifest)
    assert not (root / "attempt").exists()
    assert json.loads((state / "budget.json").read_text())["attempts"] == {}


def test_boot_mismatch_reconciles_running_attempt_before_refusing_dispatch(tmp_path, monkeypatch):
    from hwoslaps.modeling.nonlinear import profile_execution as execution
    from hwoslaps.modeling.nonlinear.profile_execution import supervise

    root, manifest = controller_fixture(tmp_path, monkeypatch, "raise RuntimeError('must not launch')\n")
    state = root / "state"
    state.mkdir()
    output = root / "attempt"
    output.mkdir()
    now = execution.time.monotonic()
    (state / "budget.json").write_text(
        json.dumps(
            {
                "cap_seconds": 100.0,
                "attempts": {
                    "a": {
                        "output": str(output),
                        "spec_path": str(root / "job.json"),
                        "gpu": 2,
                        "gpu_uuid": "GPU-test",
                        "peak_mib": 100,
                        "timeout_seconds": 10,
                        "status": "RUNNING",
                        "reservation_seconds": 70,
                        "charged_seconds": 0.0,
                        "pid": 999999999,
                        "process_start": 0.0,
                        "start_monotonic": now,
                        "clock_epoch": "old-epoch",
                    }
                },
            }
        )
    )
    (state / "deadline.json").write_text(
        json.dumps(
            {
                "clock_epoch": "old-epoch",
                "captured_monotonic": now,
                "admission_stop_monotonic": now + 10,
                "hard_stop_monotonic": now + 20,
            }
        )
    )
    stopped = []
    monkeypatch.setattr(execution, "clock_epoch", lambda: "new-epoch")
    monkeypatch.setattr(execution, "stop_owned", lambda item: stopped.append(item["output"]))
    with pytest.raises(ValueError, match="clock domain"):
        supervise(manifest)
    assert stopped == [str(output)]
    item = json.loads((state / "budget.json").read_text())["attempts"]["a"]
    assert item["status"] == "STOPPED_AFTER_FAILURE"


def test_hard_deadline_preserves_valid_exit_receipt(tmp_path, monkeypatch):
    from hwoslaps.modeling.nonlinear import profile_execution as execution
    from hwoslaps.modeling.nonlinear.profile_execution import supervise

    root, manifest = controller_fixture(tmp_path, monkeypatch, "raise RuntimeError('must not launch')\n")
    state = root / "state"
    state.mkdir()
    output = root / "attempt"
    output.mkdir()
    (output / "worker_exit.json").write_text(
        json.dumps({"status": "COMPLETE", "elapsed_s": 0.2, "artifacts": {}})
    )
    now = execution.time.monotonic()
    epoch = execution.clock_epoch()
    (state / "budget.json").write_text(
        json.dumps(
            {
                "cap_seconds": 100.0,
                "attempts": {
                    "a": {
                        "output": str(output),
                        "spec_path": str(root / "job.json"),
                        "gpu": 2,
                        "gpu_uuid": "GPU-test",
                        "peak_mib": 100,
                        "timeout_seconds": 10,
                        "status": "RUNNING",
                        "reservation_seconds": 70,
                        "charged_seconds": 0.0,
                        "pid": 999999999,
                        "process_start": 0.0,
                        "start_monotonic": now - 1,
                        "clock_epoch": epoch,
                    }
                },
            }
        )
    )
    (state / "deadline.json").write_text(
        json.dumps(
            {
                "clock_epoch": epoch,
                "captured_monotonic": now - 2,
                "admission_stop_monotonic": now - 1.5,
                "hard_stop_monotonic": now - 1,
            }
        )
    )
    monkeypatch.setattr(execution, "process_matches", lambda item: False)
    supervise(manifest)
    item = json.loads((state / "budget.json").read_text())["attempts"]["a"]
    assert item["status"] == "COMPLETE"


def test_admission_recheck_closes_reservation_before_launch(tmp_path, monkeypatch):
    import time
    from hwoslaps.modeling.nonlinear import profile_execution as execution
    from hwoslaps.modeling.nonlinear.profile_execution import supervise

    root, manifest = controller_fixture(tmp_path, monkeypatch, "raise RuntimeError('must not launch')\n")
    state = root / "state"
    state.mkdir()
    now = time.monotonic()
    epoch = execution.clock_epoch()
    (state / "deadline.json").write_text(
        json.dumps(
            {
                "clock_epoch": epoch,
                "captured_monotonic": now,
                "admission_stop_monotonic": now + 100,
                "hard_stop_monotonic": now + 200,
            }
        )
    )
    checks = iter((False, False, False, True))
    monkeypatch.setattr(execution, "deadline_reached", lambda deadline, state: next(checks, True))
    supervise(manifest)
    assert not (root / "attempt").exists()
    item = json.loads((state / "budget.json").read_text())["attempts"]["a"]
    assert item["status"] == "NOT_LAUNCHED_AFTER_DEADLINE"
    assert item["charged_seconds"] == 0.0


def test_nvml_stale_row_after_owned_exit_is_tolerated_but_pid_reuse_is_foreign():
    import psutil
    from hwoslaps.modeling.nonlinear.profile_execution import classify_compute_apps

    snapshot = {
        "a": {
            "pid": 123,
            "process_start": 10.0,
            "gpu_uuid": "GPU-owned",
            "verified": True,
        }
    }

    def exited(pid):
        raise psutil.NoSuchProcess(pid)

    assert classify_compute_apps([(123, "GPU-owned")], snapshot, exited) == []
    reused = classify_compute_apps([(123, "GPU-owned")], snapshot, lambda _pid: 11.0)
    assert reused == [{"pid": 123, "gpu_uuid": "GPU-owned", "reason": "pid_reused"}]
    foreign = classify_compute_apps([(456, "GPU-owned")], snapshot, lambda _pid: 10.0)
    assert foreign == [{"pid": 456, "gpu_uuid": "GPU-owned", "reason": "unrecognized_pid"}]


def test_unreaped_owned_zombie_nvml_row_is_tolerated(tmp_path):
    import subprocess
    import sys
    import time
    import psutil
    from hwoslaps.modeling.nonlinear.profile_execution import (
        classify_compute_apps,
        ownership_snapshot,
    )

    spec = tmp_path / "job.json"
    spec.write_text("{}")
    worker = subprocess.Popen(
        [
            sys.executable,
            "-c",
            "import os; os._exit(0)",
            str(spec),
        ],
        start_new_session=True,
    )
    try:
        process = psutil.Process(worker.pid)
        process_start = process.create_time()
        for _ in range(100):
            if process.status() == psutil.STATUS_ZOMBIE:
                break
            time.sleep(0.01)
        else:
            pytest.skip("test platform reaps children before they become observable zombies")
        active = {
            "a": {
                "pid": worker.pid,
                "process_start": process_start,
                "spec_path": str(spec),
                "gpu": 2,
                "gpu_uuid": "GPU-zombie",
            }
        }
        snapshot = ownership_snapshot(active)
        assert not snapshot["a"]["verified"]
        assert classify_compute_apps([(worker.pid, "GPU-zombie")], snapshot) == []
        reused = classify_compute_apps(
            [(worker.pid, "GPU-zombie")],
            snapshot,
            lambda _pid: process_start + 1.0,
        )
        assert reused == [{"pid": worker.pid, "gpu_uuid": "GPU-zombie", "reason": "pid_reused"}]
        foreign = classify_compute_apps([(456789, "GPU-zombie")], snapshot)
        assert foreign == [{"pid": 456789, "gpu_uuid": "GPU-zombie", "reason": "unrecognized_pid"}]
    finally:
        worker.wait(timeout=2)


def test_foreign_gpu_failure_preserves_valid_completed_receipt(tmp_path, monkeypatch):
    import subprocess
    from hwoslaps.modeling.nonlinear.profile_execution import supervise

    body = (
        '(out/"worker_exit.json").write_text(json.dumps({"status":"COMPLETE",'
        '"elapsed_s":0.2,"artifacts":{}}))\n'
        "time.sleep(30)\n"
    )
    root, manifest = controller_fixture(tmp_path, monkeypatch, body)
    app_queries = [0]

    def nvidia_query(cmd, **kwargs):
        if "--query-compute-apps=pid,gpu_uuid" in cmd:
            app_queries[0] += 1
            return "" if app_queries[0] == 1 else "123456, GPU-test\n"
        return "2, GPU-test, 1000, 0\n"

    monkeypatch.setattr(
        subprocess,
        "check_output",
        nvidia_query,
    )
    with pytest.raises(RuntimeError, match="Foreign GPU process"):
        supervise(manifest)
    state = root / "state"
    item = json.loads((state / "budget.json").read_text())["attempts"]["a"]
    assert item["status"] == "COMPLETE"
    assert not process_matches(item)
    failure = json.loads((state / "foreign_gpu_failure.json").read_text())
    assert failure["foreign_rows"] == [
        {"pid": 123456, "gpu_uuid": "GPU-test", "reason": "unrecognized_pid"}
    ]
    assert failure["owned_snapshot"]["a"]["pid"] == item["pid"]
