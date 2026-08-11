"""Tests for the in-search visualization gate on nonlinear fits."""

import os
from types import SimpleNamespace

import pytest

from hwoslaps.modeling.nonlinear.autolens_runner import (
    AutoLensFitRunner,
    NonlinearSearchSettings,
)

_ENV = "PYAUTO_SKIP_VISUALIZATION"


def _successful_result(log_likelihood=-5.0):
    """Return the smallest result accepted by runner summary extraction.

    Parameters
    ----------
    log_likelihood : `float`, optional
        Maximum log likelihood exposed by the fake samples object.

    Returns
    -------
    result : `types.SimpleNamespace`
        Minimal result object.
    """
    sample = SimpleNamespace(log_likelihood=log_likelihood)
    return SimpleNamespace(
        samples=SimpleNamespace(max_log_likelihood=lambda: sample),
    )


def _run(runner):
    """Run one minimal fake fit through ``run_model``.

    Parameters
    ----------
    runner : `AutoLensFitRunner`
        Runner under test.

    Returns
    -------
    summary : `NonlinearFitSummary`
        Fit summary.
    """
    return runner.run_model(
        model=SimpleNamespace(total_free_parameters=2),
        analysis=SimpleNamespace(),
        role="subhalo",
        fit_mode="freed",
        case_id="case",
        n_live=5,
        analysis_key="analysis",
    )


def _observing_nautilus(observed):
    """Return a fake Nautilus class that records the gate during the fit.

    Parameters
    ----------
    observed : `list`
        Receives the environment value seen inside ``fit``.

    Returns
    -------
    nautilus : `type`
        Fake search class.
    """

    class ObservingNautilus:
        """Record the visualization gate value seen inside the search."""

        def __init__(self, **kwargs):
            pass

        def fit(self, model, analysis):
            observed.append(os.environ.get(_ENV))
            return _successful_result()

    return ObservingNautilus


def test_default_disables_visualization_and_restores_absent_variable(
    monkeypatch,
    tmp_path,
):
    """Catch a fit that leaves plots on, or leaks the gate afterwards."""
    import autofit as af

    monkeypatch.delenv(_ENV, raising=False)
    observed = []
    monkeypatch.setattr(af, "Nautilus", _observing_nautilus(observed))
    runner = AutoLensFitRunner(NonlinearSearchSettings(), output_dir=tmp_path)
    summary = _run(runner)
    assert summary.status == "success"
    assert summary.visualization_disabled is True
    assert observed == ["1"]
    assert _ENV not in os.environ


def test_default_restores_preexisting_variable_value(monkeypatch, tmp_path):
    """Catch a fit that clobbers an ambient gate value after finishing."""
    import autofit as af

    monkeypatch.setenv(_ENV, "0")
    observed = []
    monkeypatch.setattr(af, "Nautilus", _observing_nautilus(observed))
    runner = AutoLensFitRunner(NonlinearSearchSettings(), output_dir=tmp_path)
    summary = _run(runner)
    assert summary.status == "success"
    assert observed == ["1"]
    assert os.environ[_ENV] == "0"


def test_opt_in_visualization_overrides_ambient_skip(monkeypatch, tmp_path):
    """Catch an opt-in plot request silently losing to the ambient gate."""
    import autofit as af

    monkeypatch.setenv(_ENV, "1")
    observed = []
    monkeypatch.setattr(af, "Nautilus", _observing_nautilus(observed))
    runner = AutoLensFitRunner(
        NonlinearSearchSettings(disable_visualization=False),
        output_dir=tmp_path,
    )
    summary = _run(runner)
    assert summary.status == "success"
    assert summary.visualization_disabled is False
    assert observed == ["0"]
    assert os.environ[_ENV] == "1"


def test_gate_is_restored_after_search_failure(monkeypatch, tmp_path):
    """Catch a failed search leaking the gate into later fits."""
    import autofit as af

    class FailingNautilus:
        """Fail inside search execution."""

        def __init__(self, **kwargs):
            pass

        def fit(self, model, analysis):
            raise RuntimeError("synthetic search failure")

    monkeypatch.delenv(_ENV, raising=False)
    monkeypatch.setattr(af, "Nautilus", FailingNautilus)
    runner = AutoLensFitRunner(NonlinearSearchSettings(), output_dir=tmp_path)
    summary = _run(runner)
    assert summary.status == "failed"
    assert "synthetic search failure" in summary.error
    assert summary.visualization_disabled is True
    assert _ENV not in os.environ


def test_gate_is_restored_before_failing_result_callback(
    monkeypatch,
    tmp_path,
):
    """Catch the gate leaking when the result callback raises."""
    import autofit as af

    seen_in_callback = []

    def failing_callback(result, model):
        seen_in_callback.append(os.environ.get(_ENV))
        raise RuntimeError("synthetic callback failure")

    monkeypatch.delenv(_ENV, raising=False)
    observed = []
    monkeypatch.setattr(af, "Nautilus", _observing_nautilus(observed))
    runner = AutoLensFitRunner(NonlinearSearchSettings(), output_dir=tmp_path)
    summary = runner.run_model(
        model=SimpleNamespace(total_free_parameters=2),
        analysis=SimpleNamespace(),
        role="subhalo",
        fit_mode="freed",
        case_id="case",
        n_live=5,
        analysis_key="analysis",
        result_callback=failing_callback,
    )
    assert summary.status == "success"
    assert any("result_callback failed" in entry for entry in summary.warnings)
    assert observed == ["1"]
    assert seen_in_callback == [None]
    assert _ENV not in os.environ


def test_settings_reject_non_boolean_disable_visualization():
    """Catch truthy non-boolean gate values passing validation."""
    with pytest.raises(ValueError) as error:
        NonlinearSearchSettings(disable_visualization=1)
    assert "disable_visualization" in str(error.value)
