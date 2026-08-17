"""Tests for PyAutoFit result extraction helpers."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from hwoslaps.modeling.nonlinear.autolens_runner import (
    AutoLensFitRunner,
    NonlinearSearchSettings,
    extract_max_log_likelihood,
    extract_max_log_likelihood_with_method,
)


def test_extract_max_log_likelihood_from_samples_callable():
    """Read the max log-likelihood from the samples accessor."""
    sample = SimpleNamespace(log_likelihood=-123.0)
    result = SimpleNamespace(
        samples=SimpleNamespace(max_log_likelihood=lambda: sample),
    )

    value, method = extract_max_log_likelihood_with_method(result)

    assert value == pytest.approx(-123.0)
    assert method == "samples.max_log_likelihood"
    assert extract_max_log_likelihood(result) == pytest.approx(-123.0)


def test_extract_max_log_likelihood_falls_back_to_fit_figure_of_merit():
    """Fall back to the fit figure of merit when samples lack it."""
    result = SimpleNamespace(
        samples=SimpleNamespace(),
        max_log_likelihood_fit=SimpleNamespace(figure_of_merit=-10.0),
    )

    value, method = extract_max_log_likelihood_with_method(result)

    assert value == pytest.approx(-10.0)
    assert method == "max_log_likelihood_fit.figure_of_merit"


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


def _capturing_nautilus(captured, fit_raises=False):
    """Return a fake Nautilus class recording constructor kwargs.

    The fake mirrors the installed backend by storing default
    convergence attributes and overriding them with passed kwargs.

    Parameters
    ----------
    captured : `list`
        Receives one kwargs dictionary per construction.
    fit_raises : `bool`, optional
        Whether ``fit`` raises instead of returning a result.

    Returns
    -------
    nautilus : `type`
        Fake search class.
    """

    class CapturingNautilus:
        """Record constructor kwargs and expose effective attributes."""

        def __init__(self, **kwargs):
            captured.append(dict(kwargs))
            self.n_eff = 500
            self.n_shell = 1
            self.discard_exploration = False
            for key, value in kwargs.items():
                setattr(self, key, value)

        def fit(self, model, analysis):
            if fit_raises:
                raise RuntimeError("synthetic search failure")
            return _successful_result()

    return CapturingNautilus


@pytest.mark.parametrize(
    "kwargs",
    [
        {"n_eff": True},
        {"n_eff": 0.0},
        {"n_eff": -5.0},
        {"n_eff": float("nan")},
        {"n_eff": float("inf")},
        {"n_shell": True},
        {"n_shell": 0},
        {"n_shell": -1},
        {"n_shell": 1.5},
        {"discard_exploration": 1},
        {"discard_exploration": "yes"},
        {"retain_search_internal": None},
        {"retain_search_internal": 1},
    ],
)
def test_settings_reject_invalid_convergence_fields(kwargs):
    """Catch bool-typed, non-positive, or non-finite sampler settings."""
    with pytest.raises(ValueError) as error:
        NonlinearSearchSettings(**kwargs)
    assert next(iter(kwargs)) in str(error.value)


def test_make_search_passes_convergence_kwargs_when_set(
    monkeypatch,
    tmp_path,
):
    """Catch requested convergence settings not reaching the backend."""
    import autofit as af

    captured = []
    monkeypatch.setattr(af, "Nautilus", _capturing_nautilus(captured))
    runner = AutoLensFitRunner(
        NonlinearSearchSettings(
            n_eff=2000.0,
            n_shell=3,
            discard_exploration=True,
        ),
        output_dir=tmp_path,
    )
    summary = _run(runner)
    assert summary.status == "success"
    assert captured[0]["n_eff"] == 2000.0
    assert captured[0]["n_shell"] == 3
    assert captured[0]["discard_exploration"] is True
    assert summary.n_eff_requested == 2000.0
    assert summary.n_eff_effective == 2000.0
    assert summary.n_shell_requested == 3
    assert summary.n_shell_effective == 3
    assert summary.discard_exploration_requested is True
    assert summary.discard_exploration_effective is True
    assert summary.search_internal_retained is False


def test_make_search_passes_false_discard_exploration(
    monkeypatch,
    tmp_path,
):
    """Catch an explicit False discard flag dropped by the None filter."""
    import autofit as af

    captured = []
    monkeypatch.setattr(af, "Nautilus", _capturing_nautilus(captured))
    runner = AutoLensFitRunner(
        NonlinearSearchSettings(discard_exploration=False),
        output_dir=tmp_path,
    )
    summary = _run(runner)
    assert summary.status == "success"
    assert captured[0]["discard_exploration"] is False
    assert summary.discard_exploration_requested is False
    assert summary.discard_exploration_effective is False


def test_make_search_omits_unset_convergence_kwargs(monkeypatch, tmp_path):
    """Catch unset settings overriding installed backend defaults."""
    import autofit as af

    captured = []
    monkeypatch.setattr(af, "Nautilus", _capturing_nautilus(captured))
    runner = AutoLensFitRunner(NonlinearSearchSettings(), output_dir=tmp_path)
    summary = _run(runner)
    assert summary.status == "success"
    assert "n_eff" not in captured[0]
    assert "n_shell" not in captured[0]
    assert "discard_exploration" not in captured[0]
    assert summary.n_eff_requested is None
    assert summary.n_eff_effective == 500.0
    assert summary.n_shell_requested is None
    assert summary.n_shell_effective == 1
    assert summary.discard_exploration_requested is None
    assert summary.discard_exploration_effective is False


def test_failed_fit_records_effective_sampler_provenance(
    monkeypatch,
    tmp_path,
):
    """Record effective settings when the fit fails after construction."""
    import autofit as af

    captured = []
    monkeypatch.setattr(
        af,
        "Nautilus",
        _capturing_nautilus(captured, fit_raises=True),
    )
    runner = AutoLensFitRunner(
        NonlinearSearchSettings(n_eff=2000.0),
        output_dir=tmp_path,
    )
    summary = _run(runner)
    assert summary.status == "failed"
    assert "synthetic search failure" in summary.error
    assert summary.n_eff_requested == 2000.0
    assert summary.n_eff_effective == 2000.0
    assert summary.n_shell_effective == 1
    assert summary.discard_exploration_effective is False
    assert summary.search_internal_retained is False


def test_failure_before_search_construction_leaves_effective_none(tmp_path):
    """Leave effective fields None when no search was constructed."""
    runner = AutoLensFitRunner(
        NonlinearSearchSettings(
            engine="Emcee",
            n_eff=2000.0,
            n_shell=3,
            discard_exploration=True,
        ),
        output_dir=tmp_path,
    )
    summary = _run(runner)
    assert summary.status == "failed"
    assert summary.n_eff_requested == 2000.0
    assert summary.n_eff_effective is None
    assert summary.n_shell_requested == 3
    assert summary.n_shell_effective is None
    assert summary.discard_exploration_requested is True
    assert summary.discard_exploration_effective is None
    assert summary.search_internal_retained is False


def test_retention_override_applied_during_fit_and_restored(
    monkeypatch,
    tmp_path,
):
    """Catch the retention override missing the fit or leaking after."""
    import autofit as af
    from autoconf import conf

    observed = []

    class ObservingNautilus:
        """Record the retention setting value seen inside the search."""

        def __init__(self, **kwargs):
            pass

        def fit(self, model, analysis):
            observed.append(conf.instance["output"]["search_internal"])
            return _successful_result()

    monkeypatch.setattr(af, "Nautilus", ObservingNautilus)
    monkeypatch.setitem(conf.instance["output"], "search_internal", False)
    runner = AutoLensFitRunner(
        NonlinearSearchSettings(retain_search_internal=True),
        output_dir=tmp_path,
    )
    summary = _run(runner)
    assert summary.status == "success"
    assert summary.search_internal_retained is True
    assert observed == [True]
    assert conf.instance["output"]["search_internal"] is False


def test_retention_override_restored_after_fit_failure(
    monkeypatch,
    tmp_path,
):
    """Catch a failed search leaking the retention override."""
    import autofit as af
    from autoconf import conf

    captured = []
    monkeypatch.setattr(
        af,
        "Nautilus",
        _capturing_nautilus(captured, fit_raises=True),
    )
    monkeypatch.setitem(conf.instance["output"], "search_internal", False)
    runner = AutoLensFitRunner(
        NonlinearSearchSettings(retain_search_internal=True),
        output_dir=tmp_path,
    )
    summary = _run(runner)
    assert summary.status == "failed"
    assert "synthetic search failure" in summary.error
    assert summary.search_internal_retained is True
    assert conf.instance["output"]["search_internal"] is False


def test_no_retention_override_when_disabled(monkeypatch, tmp_path):
    """Catch the default settings mutating the retention setting."""
    import autofit as af
    from autoconf import conf

    observed = []

    class ObservingNautilus:
        """Record the retention setting value seen inside the search."""

        def __init__(self, **kwargs):
            pass

        def fit(self, model, analysis):
            observed.append(conf.instance["output"]["search_internal"])
            return _successful_result()

    monkeypatch.setattr(af, "Nautilus", ObservingNautilus)
    monkeypatch.setitem(conf.instance["output"], "search_internal", False)
    runner = AutoLensFitRunner(NonlinearSearchSettings(), output_dir=tmp_path)
    summary = _run(runner)
    assert summary.status == "success"
    assert summary.search_internal_retained is False
    assert observed == [False]
    assert conf.instance["output"]["search_internal"] is False
