"""CPU-only contracts for the v7 fresh-search profile path."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

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


def _starts(points):
    return [
        CurrentSearchStart(
            start_index=index,
            vector=tuple(point),
            source_kind="current_search_ml" if index == 0 else "current_search_sample",
            original_start=index != 0,
            origin={"origin": "fixture"},
        )
        for index, point in enumerate(points)
    ]


def test_complete_quadratic_profile_path_and_scalar_gate():
    settings = FreshProfileSettings(
        maxiter=100,
        repeat_maxiter=100,
        start_separation_normalized_l2=0.01,
    )
    target = np.array([0.23, 0.71])
    starts = _starts(
        [
            [0.10, 0.70],
            [0.05, 0.10],
            [0.90, 0.20],
            [0.80, 0.90],
            [0.10, 0.90],
            [0.50, 0.20],
            [0.70, 0.60],
            [0.20, 0.40],
            [0.40, 0.80],
        ]
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
    assert progress


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
    starts = _starts([[0.2, 0.3]] + [[0.2 + 0.05 * i, 0.3] for i in range(8)])
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


def test_retention_is_applied_restored_and_raw_state_is_recorded(monkeypatch, tmp_path):
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
            (internal / "state.bin").write_bytes(b"state")
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
    assert summary.search_internal_retained is True
    assert conf.instance["output"]["search_internal"] is False
