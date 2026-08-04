"""Tests for PyAutoFit result extraction helpers."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from hwoslaps.modeling.nonlinear.autolens_runner import (
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
