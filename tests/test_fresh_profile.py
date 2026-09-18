"""CPU-only contracts for the v7 fresh-search profile path."""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

try:
    from hwoslaps.modeling.nonlinear.fresh_profile import (
        AutoLensFitRunner,
        CurrentSearchStart,
        FreshProfileRunner,
        FreshProfileSettings,
        FreshProfileValidator,
        NonlinearFitSummary,
        ZeroResidualAnchorRunner,
        optimize_current_search_profile,
        select_current_search_starts,
    )
    from hwoslaps.modeling.nonlinear.output_schema import NonlinearCaseResult
except (AttributeError, ImportError) as exc:  # pragma: no cover - host-runtime guard
    pytest.skip(
        f"pinned AutoLens runtime is unavailable for package imports: {exc}",
        allow_module_level=True,
    )


def _starts(points, saved_log_likelihood=0.0, lower=None, upper=None):
    """Build unit-box starts; the incumbent carries its saved likelihood."""
    dimension = len(points[0])
    lower = [0.0] * dimension if lower is None else lower
    upper = [1.0] * dimension if upper is None else upper
    return [
        CurrentSearchStart.from_physical(
            start_index=index,
            physical_vector=point,
            lower=lower,
            upper=upper,
            source_kind="current_search_ml" if index == 0 else "current_search_sample",
            original_start=index != 0,
            origin=(
                {"origin": "current_search_ml", "saved_log_likelihood": saved_log_likelihood}
                if index == 0
                else {"origin": "current_search_sample"}
            ),
        )
        for index, point in enumerate(points)
    ]


def _physical_problem(lower, upper, half_chi2_x, gradient_x):
    """Wrap a physical half-chi2 the way ``make_jax_objective`` does."""
    lower = np.asarray(lower, dtype=float)
    upper = np.asarray(upper, dtype=float)
    widths = upper - lower

    def to_x(z):
        return lower + np.asarray(z, dtype=float) * widths

    def objective(z):
        x = to_x(z)
        return float(half_chi2_x(x)), np.asarray(gradient_x(x), dtype=float) * widths

    def residual(z):
        return np.array([np.sqrt(2.0 * float(half_chi2_x(to_x(z))))])

    def direct_check(z, value):
        direct = -float(half_chi2_x(to_x(z)))
        return {
            "physical_vector": to_x(z).tolist(),
            "direct_log_likelihood": direct,
            "implied_log_likelihood": -float(value),
            "direct_log_likelihood_error": abs(direct + float(value)),
        }

    return objective, residual, to_x, direct_check


def _write_current_search(tmp_path, names, ml, ml_log_likelihood, samples):
    """Write the AutoFit summary and samples files the selector reads."""
    summary = {
        "arguments": {
            "max_log_likelihood_sample": {
                "arguments": {
                    "kwargs": {"arguments": dict(zip(names, ml))},
                    "log_likelihood": ml_log_likelihood,
                }
            }
        }
    }
    summary_path = tmp_path / "samples_summary.json"
    samples_path = tmp_path / "samples.csv"
    summary_path.write_text(json.dumps(summary))
    lines = [",".join([*names, "log_likelihood"])]
    for vector, log_likelihood in samples:
        lines.append(",".join(str(v) for v in [*vector, log_likelihood]))
    samples_path.write_text("\n".join(lines) + "\n")
    return summary_path, samples_path


def test_complete_quadratic_profile_path_and_scalar_gate():
    settings = FreshProfileSettings(
        maxiter=100,
        repeat_maxiter=100,
        start_separation_normalized_l2=0.01,
    )
    target = np.array([0.23, 0.71])
    incumbent = np.array([0.10, 0.70])
    starts = _starts(
        [
            incumbent.tolist(),
            [0.05, 0.10],
            [0.90, 0.20],
            [0.80, 0.90],
            [0.10, 0.90],
            [0.50, 0.20],
            [0.70, 0.60],
            [0.20, 0.40],
            [0.40, 0.80],
        ],
        saved_log_likelihood=-0.5 * float((incumbent - target) @ (incumbent - target)),
    )

    def objective(z):
        delta = np.asarray(z) - target
        return float(0.5 * delta @ delta), delta

    def residual(z):
        return np.asarray(z) - target

    def to_x(z):
        return np.asarray(z)

    def direct_check(z, value):
        return {
            "direct_log_likelihood": -float(value),
            "implied_log_likelihood": -float(value),
            "direct_log_likelihood_error": 0.0,
        }

    progress = []
    result = optimize_current_search_profile(
        starts,
        objective,
        residual,
        to_x,
        direct_check,
        settings,
        progress=progress.append,
    )
    assert result["candidate_acceptance_status"] == "accepted_repeatable_profile"
    assert result["candidate_start_agreement"]["original_start_count"] == 8
    assert len(result["candidate_start_agreement"]["supporting_original_start_indices"]) == 8
    assert len(result["runs"]) == 9
    assert result["candidate_direct_log_likelihood_error"] == 0.0
    assert result["incumbent"]["passed"] is True
    assert result["candidate_not_worse_than_incumbent"] is True
    assert result["runs"][0]["start_x"] == incumbent.tolist()
    assert progress


def test_incumbent_saved_likelihood_mismatch_is_not_accepted():
    settings = FreshProfileSettings(maxiter=100, repeat_maxiter=100)
    target = np.array([0.23, 0.71])
    starts = _starts(
        [[0.10, 0.70]] + [[0.1 * (i + 1), 0.2 + 0.05 * i] for i in range(8)],
        saved_log_likelihood=-5.0,
    )
    objective, residual, to_x, direct_check = _physical_problem(
        [0.0, 0.0],
        [1.0, 1.0],
        lambda x: 0.5 * float((x - target) @ (x - target)),
        lambda x: x - target,
    )
    result = optimize_current_search_profile(
        starts, objective, residual, to_x, direct_check, settings
    )
    assert result["candidate_start_agreement"]["support_passed"] is True
    assert result["incumbent"]["matches_current_search"] is False
    assert result["incumbent"]["saved_log_likelihood_error"] == pytest.approx(
        5.0 - 0.5 * float((np.array([0.10, 0.70]) - target) @ (np.array([0.10, 0.70]) - target))
    )
    assert result["candidate_acceptance_status"] == "unresolved_optimization"


def test_physical_starts_are_normalized_once_before_the_first_evaluation(tmp_path):
    """The reviewer's counterexample: prior [2, 4], two basins, ML at 2.5."""
    names = ["x"]
    summary_path, samples_path = _write_current_search(
        tmp_path, names, [2.5], 0.0, [([2.3], -0.4), ([2.7], -0.4)]
    )
    settings = FreshProfileSettings(original_start_count=2, start_separation_normalized_l2=0.05)
    starts, lower, upper = select_current_search_starts(
        summary_path, samples_path, names, [2.0], [4.0], settings
    )
    assert [s.physical_vector for s in starts] == [(2.5,), (2.3,), (2.7,)]
    assert [s.normalized_vector[0] for s in starts] == pytest.approx([0.25, 0.15, 0.35])

    def half_chi2(x):
        a = 10.0 * (x[0] - 2.5) ** 2
        b = 2.0 + 10.0 * (x[0] - 4.0) ** 2
        return min(a, b)

    def gradient(x):
        a = 10.0 * (x[0] - 2.5) ** 2
        b = 2.0 + 10.0 * (x[0] - 4.0) ** 2
        return np.array([20.0 * (x[0] - 2.5)]) if a < b else np.array([20.0 * (x[0] - 4.0)])

    objective, residual, to_x, direct_check = _physical_problem([2.0], [4.0], half_chi2, gradient)
    result = optimize_current_search_profile(
        starts, objective, residual, to_x, direct_check, settings
    )
    assert [run["start_x"] for run in result["runs"]] == [[2.5], [2.3], [2.7]]
    assert result["runs"][0]["start_half_chi2"] == 0.0
    assert result["candidate_best_vector"] == pytest.approx([2.5], abs=1.0e-8)
    assert result["candidate_best_log_likelihood"] == pytest.approx(0.0, abs=1.0e-12)
    assert result["candidate_acceptance_status"] == "accepted_repeatable_profile"
    assert result["incumbent"]["passed"] is True


def test_non_unit_prior_box_round_trips_through_selection_and_eight_start_optimizer(tmp_path):
    names = ["a", "b", "c"]
    lower = [-3.0, 5.0, 1.0e4]
    upper = [-1.0, 5.5, 3.0e4]
    widths = np.asarray(upper) - np.asarray(lower)
    target = np.array([-2.2, 5.1, 2.4e4])

    def half_chi2(x):
        delta = (np.asarray(x) - target) / widths
        return 0.5 * float(delta @ delta)

    def gradient(x):
        return (np.asarray(x) - target) / widths ** 2

    ml = [-2.1, 5.12, 2.5e4]
    samples = [
        ([-2.9, 5.05, 1.1e4], -half_chi2([-2.9, 5.05, 1.1e4])),
        ([-1.2, 5.45, 2.9e4], -half_chi2([-1.2, 5.45, 2.9e4])),
        ([-2.5, 5.40, 1.5e4], -half_chi2([-2.5, 5.40, 1.5e4])),
        ([-1.5, 5.02, 2.0e4], -half_chi2([-1.5, 5.02, 2.0e4])),
        ([-2.8, 5.30, 2.8e4], -half_chi2([-2.8, 5.30, 2.8e4])),
        ([-1.1, 5.20, 1.2e4], -half_chi2([-1.1, 5.20, 1.2e4])),
        ([-2.0, 5.48, 2.2e4], -half_chi2([-2.0, 5.48, 2.2e4])),
        ([-1.7, 5.15, 1.7e4], -half_chi2([-1.7, 5.15, 1.7e4])),
    ]
    summary_path, samples_path = _write_current_search(
        tmp_path, names, ml, -half_chi2(ml), samples
    )
    settings = FreshProfileSettings(maxiter=200, repeat_maxiter=200)
    starts, lower_array, upper_array = select_current_search_starts(
        summary_path, samples_path, names, lower, upper, settings
    )
    assert len(starts) == 9
    ranked = [vector for vector, _ in sorted(samples, key=lambda item: -item[1])]
    for start, expected in zip(starts, [ml, *ranked]):
        assert start.physical_vector == tuple(expected)
        z = np.asarray(start.normalized_vector)
        assert np.all(z >= 0.0) and np.all(z <= 1.0)
        assert lower_array + z * (upper_array - lower_array) == pytest.approx(expected, rel=1e-12)
    objective, residual, to_x, direct_check = _physical_problem(lower, upper, half_chi2, gradient)
    result = optimize_current_search_profile(
        starts, objective, residual, to_x, direct_check, settings
    )
    assert [run["start_x"] for run in result["runs"]] == [ml, *ranked]
    assert result["candidate_best_vector"] == pytest.approx(target.tolist(), rel=1e-7)
    assert result["candidate_acceptance_status"] == "accepted_repeatable_profile"
    assert len(result["candidate_start_agreement"]["supporting_original_start_indices"]) == 8
    assert result["incumbent"]["half_chi2"] == pytest.approx(half_chi2(ml))
    assert result["incumbent"]["passed"] is True


@pytest.mark.parametrize("defect", ["outside_unit_box", "physical_mismatch"])
def test_bad_normalized_start_fails_before_any_objective_evaluation(defect):
    settings = FreshProfileSettings(original_start_count=1)
    if defect == "outside_unit_box":
        incumbent = CurrentSearchStart(
            start_index=0,
            physical_vector=(0.5,),
            normalized_vector=(1.2,),
            source_kind="current_search_ml",
            original_start=False,
            origin={"saved_log_likelihood": 0.0},
        )
    else:
        incumbent = CurrentSearchStart(
            start_index=0,
            physical_vector=(0.9,),
            normalized_vector=(0.5,),
            source_kind="current_search_ml",
            original_start=False,
            origin={"saved_log_likelihood": 0.0},
        )
    sample = CurrentSearchStart.from_physical(
        start_index=1,
        physical_vector=(0.3,),
        lower=(0.0,),
        upper=(1.0,),
        source_kind="current_search_sample",
        original_start=True,
        origin={},
    )
    evaluations = []

    def objective(z):
        evaluations.append(np.asarray(z).tolist())
        return 0.0, np.zeros(1)

    with pytest.raises(ValueError, match="not evaluated"):
        optimize_current_search_profile(
            [incumbent, sample],
            objective,
            lambda z: np.zeros(1),
            lambda z: np.asarray(z),
            None,
            settings,
        )
    assert evaluations == []


def test_selected_starts_must_lead_with_the_ml_incumbent():
    starts = _starts([[0.2], [0.4]], saved_log_likelihood=0.0)
    swapped = [starts[1], starts[0]]
    with pytest.raises(ValueError, match="ML incumbent"):
        optimize_current_search_profile(
            swapped,
            lambda z: (0.0, np.zeros(1)),
            lambda z: np.zeros(1),
            lambda z: np.asarray(z),
            None,
            FreshProfileSettings(original_start_count=1),
        )


def test_better_unsupported_incumbent_is_kept_and_leaves_profile_unresolved():
    """A repeatable worse basin never replaces the current-search maximum."""
    settings = FreshProfileSettings(maxiter=200, repeat_maxiter=200)

    def half_chi2(x):
        value = x[0]
        return 5.0 * (value - 0.2) ** 2 if value < 0.5 else 1.0 + 50.0 * (value - 0.8) ** 2

    def gradient(x):
        value = x[0]
        return np.array([10.0 * (value - 0.2)]) if value < 0.5 else np.array([100.0 * (value - 0.8)])

    starts = _starts(
        [[0.2]] + [[0.56 + 0.03 * i] for i in range(8)],
        saved_log_likelihood=0.0,
    )
    objective, residual, to_x, direct_check = _physical_problem([0.0], [1.0], half_chi2, gradient)
    result = optimize_current_search_profile(
        starts, objective, residual, to_x, direct_check, settings
    )
    assert result["candidate_best_half_chi2"] == pytest.approx(0.0, abs=1e-12)
    assert result["candidate_best_vector"] == pytest.approx([0.2], abs=1e-8)
    assert result["incumbent"]["candidate_not_worse"] is True
    assert result["candidate_start_agreement"]["support_passed"] is False
    assert result["candidate_acceptance_status"] == "unresolved_optimization"
    assert all(
        run["observed_best_half_chi2"] == pytest.approx(1.0, abs=1e-8)
        for run in result["runs"][1:]
    )


def test_current_search_start_selection_requires_ml_plus_eight_or_declared_count(tmp_path):
    summary = {
        "arguments": {
            "max_log_likelihood_sample": {
                "arguments": {
                    "kwargs": {
                        "arguments": {"a": 0.1, "b": 0.1}
                    },
                    "log_likelihood": -1.0,
                }
            }
        }
    }
    summary_path = tmp_path / "samples_summary.json"
    samples_path = tmp_path / "samples.csv"
    summary_path.write_text(json.dumps(summary))
    samples_path.write_text(
        "a,b,log_likelihood\n"
        "0.9,0.9,-0.1\n"
        "0.1,0.9,-0.2\n"
        "0.9,0.1,-0.3\n"
    )
    settings = FreshProfileSettings(original_start_count=2)
    starts, lower, upper = select_current_search_starts(
        summary_path,
        samples_path,
        ["a", "b"],
        [0.0, 0.0],
        [1.0, 1.0],
        settings,
    )
    assert len(starts) == 3
    assert starts[0].source_kind == "current_search_ml"
    assert all(start.original_start for start in starts[1:])
    assert np.array_equal(lower, [0.0, 0.0])
    assert np.array_equal(upper, [1.0, 1.0])
    assert starts[0].physical_vector == (0.1, 0.1)
    assert starts[0].normalized_vector == (0.1, 0.1)
    assert starts[0].origin["saved_log_likelihood"] == -1.0
    assert starts[0].to_dict()["physical_vector"] == [0.1, 0.1]
    assert "vector" not in starts[0].to_dict()

    offset_starts, _, _ = select_current_search_starts(
        summary_path,
        samples_path,
        ["a", "b"],
        [0.0, -1.0],
        [2.0, 1.0],
        settings,
    )
    assert offset_starts[0].physical_vector == (0.1, 0.1)
    assert offset_starts[0].normalized_vector == pytest.approx((0.05, 0.55))
    assert offset_starts[1].physical_vector == (0.9, 0.9)
    assert offset_starts[1].normalized_vector == pytest.approx((0.45, 0.95))


def test_profile_settings_reject_undeclared_flat_release_protocol():
    release = {
        "protocol": {
            "optimizer": {
                "original_start_count": 8,
                "start_separation_normalized_l2": 0.05,
                "maxiter": 500,
                "ftol": 0.0,
                "gtol": 1.0e-10,
                "maxls": 50,
                "tighter_repeat": {
                    "maxiter": 1000,
                    "ftol": 0.0,
                    "gtol": 1.0e-12,
                },
                "scalar_residual_tolerance": 1.0e-4,
            },
            "acceptance": {
                "distinct_original_starts": 2,
                "support_log_likelihood_tolerance": 0.1,
                "tighter_repeat_tolerance": 0.1,
            },
        }
    }
    with pytest.raises(ValueError, match="explicitly bind profile settings"):
        FreshProfileSettings.from_release_protocol(release)


def test_profile_settings_load_from_actual_v7_yaml_nested_schema():
    import yaml

    release_path = Path(__file__).parents[1] / "configs/design/design_freeze_v7.yaml"
    release = yaml.safe_load(release_path.read_text())
    settings = FreshProfileSettings.from_release_protocol(release)
    assert settings.original_start_count == 8
    assert settings.maxiter == 500
    assert settings.repeat_maxiter == 1000
    assert settings.scalar_residual_tolerance == pytest.approx(1.0e-4)


def test_public_nonlinear_case_result_shape_is_retained():
    fit = NonlinearFitSummary(model_role="smooth", fit_mode="freed", status="success")
    result = NonlinearCaseResult(
        case_id="case",
        trial=object(),
        dataset_metadata=object(),
        fit_mode="freed",
        psf_case="nominal",
        smooth_fit=fit,
        subhalo_fit=fit,
        metric=None,
        quality_flags=[],
    )
    assert result.smooth_fit is fit
    assert result.subhalo_fit is fit
    assert result.quality_flags == []


def test_zero_residual_anchor_has_no_evidence_claim():
    model = SimpleNamespace(
        unique_prior_paths=(("galaxies", "lens", "subhalo", "x"),),
        priors_ordered_by_id=[SimpleNamespace(lower_limit=0.0, upper_limit=1.0)],
        instance_from_vector=lambda vector: SimpleNamespace(vector=vector),
    )
    analysis = SimpleNamespace(
        fit_from=lambda instance: SimpleNamespace(
            normalized_residual_map=np.zeros(2),
        ),
        log_likelihood_function=lambda instance: -3.0,
        _use_jax=True,
    )
    fresh = SimpleNamespace(
        profile_records={},
        profile_settings=FreshProfileSettings(),
        settings=SimpleNamespace(use_jax=True),
        output_dir="/tmp/stage3-anchor",
    )
    runner = ZeroResidualAnchorRunner(
        fresh,
        {"parameter_names": ["galaxies.lens.subhalo.x"], "vector": [0.5]},
    )
    summary = runner.run_model(
        model=model,
        analysis=analysis,
        role="subhalo",
        analysis_key="anchor-key",
        fit_mode="fixed_template",
    )
    assert summary.status == "success"
    assert summary.log_evidence is None
    assert fresh.profile_records["subhalo"]["sampler_executed"] is False
    assert fresh.profile_records["subhalo"]["evidence_claim"] is False


def test_profile_records_numerically_unresolved_when_support_gate_fails():
    settings = FreshProfileSettings(
        maxiter=20,
        repeat_maxiter=20,
        minimum_distinct_original_start_support=9,
    )
    starts = _starts(
        [[0.2, 0.3]] + [[0.2 + 0.05 * i, 0.3] for i in range(8)],
        saved_log_likelihood=0.0,
    )
    target = np.array([0.2, 0.3])

    def objective(z):
        delta = np.asarray(z) - target
        return float(0.5 * delta @ delta), delta

    result = optimize_current_search_profile(
        starts,
        objective,
        lambda z: np.asarray(z) - target,
        lambda z: np.asarray(z),
        lambda z, value: {
            "direct_log_likelihood": -float(value),
            "implied_log_likelihood": -float(value),
            "direct_log_likelihood_error": 0.0,
        },
        settings,
    )
    assert result["candidate_acceptance_status"] == "unresolved_optimization"
    assert result["candidate_start_agreement"]["support_passed"] is False


def test_no_finite_profile_has_no_fallback_candidate():
    settings = FreshProfileSettings(maxiter=3, repeat_maxiter=3)
    starts = _starts([[0.2, 0.3]] + [[0.2 + 0.05 * i, 0.3] for i in range(8)])

    def objective(z):
        return float("nan"), np.full(2, np.nan)

    result = optimize_current_search_profile(
        starts,
        objective,
        lambda z: np.zeros(2),
        lambda z: np.asarray(z),
        None,
        settings,
    )
    assert result["candidate_acceptance_status"] == "unresolved_optimization"
    assert "candidate_best_log_likelihood" not in result
    assert result["old_convergence_transferred"] is False
    assert result["incumbent"]["evaluated"] is False
    assert result["incumbent"]["passed"] is False


def test_effective_sampler_contract_blocks_fit_before_search_fit(monkeypatch, tmp_path):
    import autofit as af

    calls = []

    class Search:
        n_eff = 499.0
        n_shell = 1
        discard_exploration = False

        def __init__(self, **kwargs):
            pass

        def fit(self, model, analysis):
            calls.append("fit")
            raise AssertionError("fit must not run after contract mismatch")

    monkeypatch.setattr(af, "Nautilus", Search)
    runner = AutoLensFitRunner(
        __import__(
            "hwoslaps.modeling.nonlinear.autolens_runner",
            fromlist=["NonlinearSearchSettings"],
        ).NonlinearSearchSettings(
            n_eff=500,
            n_shell=1,
            discard_exploration=False,
            sampler_contract={"n_eff": 500, "n_shell": 1, "discard_exploration": False},
        ),
        tmp_path,
    )
    summary = runner.run_model(
        model=SimpleNamespace(total_free_parameters=1),
        analysis=SimpleNamespace(),
        role="smooth",
        fit_mode="freed",
        case_id="case",
        n_live=100,
        analysis_key="key",
    )
    assert summary.status == "failed"
    assert "effective settings" in summary.error
    assert calls == []


def test_effective_live_point_contract_is_checked_before_fit(monkeypatch, tmp_path):
    import autofit as af

    calls = []

    class Search:
        n_eff = 500.0
        n_shell = 1
        discard_exploration = False
        n_live = 99

        def __init__(self, **kwargs):
            pass

        def fit(self, model, analysis):
            calls.append("fit")
            raise AssertionError("fit must not run after n_live mismatch")

    monkeypatch.setattr(af, "Nautilus", Search)
    settings_cls = __import__(
        "hwoslaps.modeling.nonlinear.autolens_runner",
        fromlist=["NonlinearSearchSettings"],
    ).NonlinearSearchSettings
    runner = AutoLensFitRunner(
        settings_cls(
            n_eff=500,
            n_shell=1,
            discard_exploration=False,
            sampler_contract={
                "n_eff": 500,
                "n_shell": 1,
                "discard_exploration": False,
                "n_live_by_fit_mode": {"freed": 100},
            },
        ),
        tmp_path,
    )
    summary = runner.run_model(
        model=SimpleNamespace(total_free_parameters=1),
        analysis=SimpleNamespace(),
        role="subhalo",
        fit_mode="freed",
        case_id="case",
        n_live=100,
        analysis_key="key",
    )
    assert summary.status == "failed"
    assert "n_live" in summary.error
    assert calls == []


def test_smooth_role_live_points_are_checked_against_the_smooth_declaration(monkeypatch, tmp_path):
    """A freed case's smooth search declares its own live points, not the subhalo's."""
    import autofit as af

    class Search:
        n_eff = 500.0
        n_shell = 1
        discard_exploration = False
        n_live = 100

        def __init__(self, **kwargs):
            pass

        def fit(self, model, analysis):
            raise RuntimeError("contract passed; stop before sampling")

    monkeypatch.setattr(af, "Nautilus", Search)
    settings_cls = __import__(
        "hwoslaps.modeling.nonlinear.autolens_runner",
        fromlist=["NonlinearSearchSettings"],
    ).NonlinearSearchSettings
    runner = AutoLensFitRunner(
        settings_cls(
            n_eff=500,
            n_shell=1,
            discard_exploration=False,
            sampler_contract={
                "n_eff": 500,
                "n_shell": 1,
                "discard_exploration": False,
                "n_live_by_fit_mode": {"smooth": 100, "freed": 200, "fixed_template": 100},
            },
        ),
        tmp_path,
    )
    summary = runner.run_model(
        model=SimpleNamespace(total_free_parameters=1),
        analysis=SimpleNamespace(),
        role="smooth",
        fit_mode="freed",
        case_id="case",
        n_live=100,
        analysis_key="key",
    )
    assert summary.status == "failed"
    assert "contract passed" in summary.error
    assert "n_live" not in summary.error


def test_result_callback_warning_blocks_profile_success(monkeypatch, tmp_path):
    base_summary = NonlinearFitSummary(
        model_role="subhalo",
        fit_mode="freed",
        status="success",
        result_path=str(tmp_path / "search"),
        warnings=["result_callback failed: synthetic recovery error"],
    )
    monkeypatch.setattr(
        __import__(
            "hwoslaps.modeling.nonlinear.autolens_runner",
            fromlist=["AutoLensFitRunner"],
        ).AutoLensFitRunner,
        "run_model",
        lambda self, **kwargs: base_summary,
    )
    runner = FreshProfileRunner(
        __import__(
            "hwoslaps.modeling.nonlinear.autolens_runner",
            fromlist=["NonlinearSearchSettings"],
        ).NonlinearSearchSettings(),
        tmp_path,
    )
    summary = runner.run_model(
        model=SimpleNamespace(total_free_parameters=1),
        analysis=SimpleNamespace(),
        role="subhalo",
        fit_mode="freed",
        case_id="case",
        n_live=200,
        analysis_key="key",
    )
    assert summary.status == "failed"
    assert "callback failure" in summary.error
    assert runner.profile_records["subhalo"]["candidate_acceptance_status"] == "unresolved_callback_error"


def test_selected_case_attaches_likelihood_matched_tangent(monkeypatch):
    import sys
    from types import ModuleType

    validator_module = ModuleType("hwoslaps.modeling.nonlinear.validator")

    class Delegate:
        def __init__(self, runner):
            self.runner = runner

        def validate_case(self, *args, **kwargs):
            return SimpleNamespace(diagnostics={})

    validator_module.NonlinearMetricValidator = Delegate
    monkeypatch.setitem(
        sys.modules,
        "hwoslaps.modeling.nonlinear.validator",
        validator_module,
    )
    runner = SimpleNamespace(profile_records={})
    validator = FreshProfileValidator(runner, compute_comparator=True)
    comparator = {"q": 12.0, "derivatives_stable": True}
    monkeypatch.setattr(
        __import__(
            "hwoslaps.modeling.nonlinear.fresh_profile",
            fromlist=["likelihood_matched_tangent"],
        ),
        "likelihood_matched_tangent",
        lambda *args, **kwargs: comparator,
    )
    result = validator.validate_case(
        object(), object(), object(), object(), fit_mode="freed"
    )
    assert result.diagnostics["likelihood_matched_tangent"] == comparator
    assert runner.profile_records["likelihood_matched_tangent"] == comparator


def test_bad_bracket_anchor_is_rejected_before_delegate_h0(monkeypatch):
    import sys
    from types import ModuleType

    calls = []
    validator_module = ModuleType("hwoslaps.modeling.nonlinear.validator")

    class Delegate:
        def __init__(self, runner):
            self.runner = runner

        def validate_case(self, *args, **kwargs):
            calls.append("delegate")
            raise AssertionError("H0 delegate must not run")

    validator_module.NonlinearMetricValidator = Delegate
    monkeypatch.setitem(
        sys.modules,
        "hwoslaps.modeling.nonlinear.validator",
        validator_module,
    )
    runner = SimpleNamespace(
        preflight_anchor=lambda *args, **kwargs: (_ for _ in ()).throw(
            ValueError("bad anchor")
        ),
        profile_records={},
    )
    validator = FreshProfileValidator(runner)
    with pytest.raises(ValueError, match="bad anchor"):
        validator.validate_case(object(), object(), object(), object())
    assert calls == []


def test_established_fisher_q_adapter_binds_999_kernel_and_mass_point(monkeypatch):
    import sys
    from types import ModuleType

    run_ladder = ModuleType("run_ladder")
    observed = {}

    def rung_config(config, ladder, aperture):
        updated = dict(config)
        updated["psf"] = {"kernel": {"shape_native": [999, 999]}}
        return updated

    class Detector:
        def _evaluate_grid_positions(self, positions):
            observed["positions"] = positions
            return [SimpleNamespace(q_asimov_local=7.25)]

    run_ladder._rung_config = rung_config
    run_ladder._build_detector = lambda config, psf: Detector()
    run_ladder._point_detector_at_rung = lambda detector, logm: observed.update(
        logm=logm
    )
    monkeypatch.setitem(sys.modules, "run_ladder", run_ladder)
    config_validation = ModuleType("hwoslaps.config.validation")
    config_validation.validate_or_raise = lambda config: observed.update(
        kernel=config["psf"]["kernel"]["shape_native"]
    )
    config_package = ModuleType("hwoslaps.config")
    config_package.__path__ = []
    monkeypatch.setitem(sys.modules, "hwoslaps.config", config_package)
    monkeypatch.setitem(sys.modules, "hwoslaps.config.validation", config_validation)
    psf_generator = ModuleType("hwoslaps.psf.generator")
    psf_generator.generate_psf_system = lambda config, full_config: object()
    psf_package = ModuleType("hwoslaps.psf")
    psf_package.__path__ = []
    monkeypatch.setitem(sys.modules, "hwoslaps.psf", psf_package)
    monkeypatch.setitem(sys.modules, "hwoslaps.psf.generator", psf_generator)

    result = __import__(
        "hwoslaps.modeling.nonlinear.fresh_profile",
        fromlist=["evaluate_established_fisher_q"],
    ).evaluate_established_fisher_q(
        config={"ladder": {"aperture": {}}, "psf": {"kernel": {"shape_native": [51, 51]}}},
        position_yx_arcsec=[0.2, -0.3],
        log10_m200=7.4,
    )
    assert result["q_f_production_at_position"] == pytest.approx(7.25)
    assert result["kernel_shape_native"] == [999, 999]
    assert observed["kernel"] == [999, 999]
    assert observed["logm"] == pytest.approx(7.4)
    assert observed["positions"] == [(0.2, -0.3)]


@pytest.mark.parametrize("retained_files", [("search_internal.dill", ".time"), (".time",)])
def test_retention_is_applied_restored_and_raw_state_is_recorded(
    monkeypatch, tmp_path, retained_files
):
    import autofit as af
    from autoconf import conf

    observed = []

    class Search:
        n_eff = 500.0
        n_shell = 1
        discard_exploration = False

        def __init__(self, **kwargs):
            self.paths = SimpleNamespace(output_path=str(tmp_path / "search"))

        def fit(self, model, analysis):
            observed.append(conf.instance["output"]["search_internal"])
            internal = Path(self.paths.output_path) / "files" / "search_internal"
            internal.mkdir(parents=True)
            for name in retained_files:
                (internal / name).write_bytes(b"sampler-state" if name.endswith(".dill") else b"0.1")
            return SimpleNamespace(
                samples=SimpleNamespace(
                    max_log_likelihood=lambda: SimpleNamespace(log_likelihood=-2.0)
                ),
                paths=self.paths,
            )

    monkeypatch.setattr(af, "Nautilus", Search)
    monkeypatch.setitem(conf.instance["output"], "search_internal", False)
    runner = AutoLensFitRunner(
        __import__(
            "hwoslaps.modeling.nonlinear.autolens_runner",
            fromlist=["NonlinearSearchSettings"],
        ).NonlinearSearchSettings(
            n_eff=500,
            n_shell=1,
            discard_exploration=False,
            retain_search_internal=True,
            sampler_contract={
                "n_eff": 500,
                "n_shell": 1,
                "discard_exploration": False,
            },
        ),
        tmp_path / "case",
    )
    summary = runner.run_model(
        model=SimpleNamespace(total_free_parameters=1),
        analysis=SimpleNamespace(),
        role="smooth",
        fit_mode="freed",
        case_id="case",
        n_live=100,
        analysis_key="key",
    )
    assert summary.status == "success"
    assert observed == [True]
    assert summary.search_internal_retention_requested is True
    assert conf.instance["output"]["search_internal"] is False
    payload = summary.search_internal_payload
    assert payload["route"] == "directory"
    assert payload["bound_to_result_path"] is True
    assert set(payload["files"]) == set(retained_files)
    if "search_internal.dill" in retained_files:
        assert summary.search_internal_retained is True
        assert payload["missing_required"] == []
        dill = Path(tmp_path / "search" / "files" / "search_internal" / "search_internal.dill")
        assert payload["files"]["search_internal.dill"]["bytes"] == dill.stat().st_size
        assert payload["files"]["search_internal.dill"]["sha256"] == __import__(
            "hashlib"
        ).sha256(dill.read_bytes()).hexdigest()
    else:
        assert summary.search_internal_retained is False
        assert payload["missing_required"] == ["search_internal.dill"]


def _bracket_inputs(tmp_path):
    """Write a ladder-shaped staged config (no subhalo) and its frozen top rung."""
    with open("configs/scenes/scene1_smooth_ring.yaml", encoding="utf-8") as stream:
        staged = yaml.safe_load(stream)
    staged["lensing"]["subhalo"] = {
        "enabled": False,
        "mass": "1.0e7",
        "model": "NFW",
        "concentration": {"model": "moline2017_eq7", "x_sub": 1.0, "h": None},
        "position": {"type": "angle", "angle": 90.0, "offset_pixels": 0},
    }
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(staged, sort_keys=False), encoding="utf-8")
    positions_path = tmp_path / "positions.json"
    positions_path.write_text(
        json.dumps(
            {
                "rungs": {
                    "top": {
                        "logm": 7.2,
                        "mass_msun": 10.0**7.2,
                        "position_yx_arcsec": [0.8, 0.05],
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    return config_path, positions_path


def _rendering_generator(observed):
    """Mimic the generator contract: a disabled subhalo renders as None."""

    def generate_lensing_system(lensing_config, full_config):
        observed.append(deepcopy(full_config))
        subhalo = lensing_config["subhalo"]
        present = bool(subhalo["enabled"])
        return SimpleNamespace(
            lens_redshift=lensing_config["lens_galaxy"]["redshift"],
            source_redshift=lensing_config["source_galaxy"]["redshift"],
            subhalo_model=subhalo["model"] if present else None,
            subhalo_mass=float(subhalo["mass"]) if present else None,
            subhalo_position=(
                tuple(subhalo["position"]["centre"]) if present else None
            ),
            subhalo_einstein_radius=None,
            subhalo_kappa_s=0.01 if present else None,
            subhalo_scale_radius_arcsec=0.2 if present else None,
            subhalo_concentration=20.0 if present else None,
            subhalo_concentration_model=(
                subhalo["concentration"]["model"] if present else None
            ),
        )

    return generate_lensing_system


def _materialize(tmp_path, target_mass_msun=10.0**7.3):
    from hwoslaps.modeling.nonlinear.fresh_profile import (
        materialize_bracket_case_from_files,
    )

    config_path, positions_path = _bracket_inputs(tmp_path)
    return materialize_bracket_case_from_files(
        config_path=config_path,
        positions_path=positions_path,
        output_dir=tmp_path / "generated",
        case_id="selected12_bracket:sys0043:plus_0.1dex",
        bracket_rung="plus_0.1dex",
        target_log10_m200=7.3,
        target_mass_msun=target_mass_msun,
        position_yx_arcsec=(0.8, 0.05),
    )


def test_bracket_materializer_renders_the_declared_target_subhalo(tmp_path, monkeypatch):
    import hwoslaps.lensing.generator as generator

    observed = []
    monkeypatch.setattr(generator, "generate_lensing_system", _rendering_generator(observed))
    target = 10.0**7.3

    generated = _materialize(tmp_path, target)

    assert len(observed) == 1
    rendered = observed[0]["lensing"]["subhalo"]
    assert rendered["enabled"] is True
    assert rendered["mass"] == pytest.approx(target)
    assert rendered["position"] == {"type": "direct", "centre": [0.8, 0.05]}
    assert generated["status"] == "MATERIALIZED_NOT_FIT"
    written = yaml.safe_load(Path(generated["config"]).read_text(encoding="utf-8"))
    assert written["lensing"]["subhalo"]["enabled"] is False
    rungs = json.loads(Path(generated["positions"]).read_text(encoding="utf-8"))["rungs"]
    assert rungs["plus_0.1dex"]["mass_msun"] == pytest.approx(target)
    assert rungs["plus_0.1dex"]["position_yx_arcsec"] == [0.8, 0.05]
    anchor = json.loads(Path(generated["h1_anchor"]).read_text(encoding="utf-8"))
    assert anchor["target_mass_msun"] == pytest.approx(target)
    assert anchor["evidence_claim"] is False and anchor["sampler_executed"] is False
    assert len(anchor["vector"]) == len(anchor["parameter_names"]) > 0
    assert np.all(np.isfinite(anchor["vector"]))


def test_bracket_materializer_rejects_a_reference_without_the_target(tmp_path, monkeypatch):
    import hwoslaps.lensing.generator as generator

    def without_subhalo(lensing_config, full_config):
        return SimpleNamespace(
            lens_redshift=0.2,
            source_redshift=0.6,
            subhalo_model=None,
            subhalo_mass=None,
            subhalo_position=None,
        )

    monkeypatch.setattr(generator, "generate_lensing_system", without_subhalo)
    with pytest.raises(RuntimeError, match="declared target subhalo"):
        _materialize(tmp_path)
    assert not (tmp_path / "generated").exists()
