"""Freed-subhalo recovery: midpoint-CDF quantiles and the sampler and refined estimates."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from hwoslaps.inference.fit_model import FitArgument, FitComponent, FitGalaxy, FitModel, fixed, uniform
from hwoslaps.inference.recovery import extract_recovery, weighted_quantiles
from hwoslaps.inference.result import RefineOutcome, RoleStatus
from hwoslaps.inference.settings import MassSupport

SUPPORT = MassSupport(6.0, 9.7)


@pytest.mark.parametrize(("values", "weights", "expected"), [
    ([0.0, 10.0], [0.25, 0.75], (10.0 * (0.16 - 0.125) / 0.5, 10.0 * (0.5 - 0.125) / 0.5, 10.0)),
    ([1.0, 2.0, 3.0, 4.0], [0.01, 0.01, 0.01, 0.97], (3.0 + (0.16 - 0.025) / 0.49, 3.0 + (0.5 - 0.025) / 0.49, 4.0)),
    ([1.0, 2.0, 3.0, 4.0], [0.97, 0.01, 0.01, 0.01], (1.0, 1.0 + (0.5 - 0.485) / 0.49, 1.0 + (0.84 - 0.485) / 0.49)),
    ([3.0, 1.0, 2.0, 4.0], [0.25, 0.25, 0.25, 0.25], (1.14, 2.5, 3.86)),
    ([3.0, 1.0, 4.0, 1.5, 9.0, 2.6], None, tuple(np.quantile([3.0, 1.0, 4.0, 1.5, 9.0, 2.6], [0.16, 0.5, 0.84]))),
], ids=["asymmetric-two-point", "heavy-last-weight", "heavy-first-weight-clamps", "equal-weights",
        "no-weights"])
def test_weighted_quantiles_follow_the_midpoint_cdf(values, weights, expected):
    """Sorted samples sit at (cumsum(w) - w / 2) / sum(w); quantiles interpolate and clamp."""
    result = weighted_quantiles(np.asarray(values), None if weights is None else np.asarray(weights))
    assert result == pytest.approx(expected, rel=1.0e-12)


def test_weighted_measure_ignores_zero_probability_and_representation():
    expected = weighted_quantiles(np.array([2., 8., 10.]), np.array([1., 2., 1.]))
    # Zeros below, inside and above support include nonfinite noncontributing values.
    values = np.array([-100., 2., 5., 8., 10., 100., np.nan])
    weights = np.array([0., 1., 0., 2., 1., 0., 0.])
    for order in (np.arange(7), np.array([6, 5, 4, 3, 2, 1, 0]), np.array([2, 3, 0, 6, 4, 1, 5])):
        for scale in (1.e-200, 1., 1.e200):
            assert weighted_quantiles(values[order], weights[order] * scale) == pytest.approx(expected)
    assert weighted_quantiles(np.array([6., 8., 10.]), np.array([0., 1., 0.])) == (8., 8., 8.)
    # Splitting a tied atom cannot make its arbitrary sort order matter.
    split_values, split_weights = np.array([2., 8., 8., 10.]), np.array([1., .1, 1.9, 1.])
    assert weighted_quantiles(split_values, split_weights) == pytest.approx(expected)
    assert weighted_quantiles(split_values[::-1], split_weights[::-1]) == pytest.approx(expected)


def test_weighted_quantiles_are_continuous_at_equal_weights_and_overflow_safe():
    values = np.array([0., 10.])
    equal = weighted_quantiles(values, np.ones(2))
    assert equal == (0., 5., 10.)
    assert weighted_quantiles(values, np.array([1., 1. + 1.e-12])) == pytest.approx(equal, abs=1.e-10)
    assert weighted_quantiles(values, np.full(2, np.finfo(float).max)) == equal
    assert weighted_quantiles(values, None) == pytest.approx((1.6, 5., 8.4))


@pytest.mark.parametrize(("values", "weights", "probabilities"), [
    ([1., 2.], [0., 0.], [.5]), ([1., 2.], [-1., 2.], [.5]),
    ([1., 2.], [np.inf, 1.], [.5]), ([1., 2.], [np.nan, 1.], [.5]),
    ([np.nan, 2.], [1., 1.], [.5]), ([np.inf, 2.], None, [.5]),
    ([], None, [.5]), ([[1., 2.]], [[1., 1.]], [.5]),
    ([1., 2.], [1.], [.5]), ([1., 2.], [1., 1.], [-.1]),
    ([1., 2.], None, [1.1]), ([1., 2.], [1., 1.], [np.nan]),
])
def test_quantiles_reject_invalid_probability_measures(values, weights, probabilities):
    with pytest.raises(ValueError):
        weighted_quantiles(values, weights, probabilities)


def _subhalo_model():
    """H1 with a lens-plane freed subhalo: lens mass (2 free), subhalo centre and mass, source (1 free)."""
    def scalar(name, lower, upper, truth):
        return FitArgument(name, (uniform(lower, upper, truth=truth),), pair=False)

    def pair(name, lower, upper, truth):
        return FitArgument(name, tuple(uniform(lo, hi, truth=t) for lo, hi, t in zip(lower, upper, truth)), pair=True)

    lens = FitGalaxy("lens", 0.2, (
        ("mass", FitComponent("autogalaxy.profiles.mass.total.isothermal:Isothermal", (
            FitArgument("centre", (fixed(0.0), fixed(0.0)), pair=True), scalar("einstein_radius", 0.99, 1.01, 1.0),
            FitArgument("ell_comps", (uniform(0.08, 0.12, truth=0.1), fixed(0.0)), pair=True)))),
        ("subhalo", FitComponent("hwoslaps.inference.subhalo_classes:NFWM200SubhaloSph", (
            pair("centre", (0.25, -0.95), (0.55, -0.65), (0.4, -0.8)), scalar("log10_m200", 6.0, 9.7, 9.0)))),
    ))
    source = FitGalaxy("source", 0.6, (("light", FitComponent("autogalaxy.profiles.light.standard.exponential:"
                                                               "Exponential", (scalar("intensity", 1.0, 3.0, 2.0),))),))
    return FitModel("subhalo", (lens, source))


class _Mapping:
    support = SUPPORT

    def profile_scales(self, log10_mass):
        return {"concentration": 10.0 - log10_mass, "kappa_s": 0.01 * log10_mass}


def _result(*, pdf_converged):
    """An AutoFit result's recovery surface: the maximum-likelihood instance and the samples."""
    subhalo = SimpleNamespace(log10_m200=8.9, centre=(0.39, -0.82))
    paths = {("galaxies", "lens", "subhalo", "log10_m200"): [8.0, 8.5, 9.0],
             ("galaxies", "lens", "subhalo", "centre", "centre_0"): [0.3, 0.4, 0.5],
             ("galaxies", "lens", "subhalo", "centre", "centre_1"): [-0.9, -0.8, -0.7]}
    samples = SimpleNamespace(pdf_converged=pdf_converged, weight_list=[0.25, 0.5, 0.25],
                              values_for_path=lambda path: paths[tuple(path)])
    return SimpleNamespace(max_log_likelihood_instance=SimpleNamespace(
        galaxies=SimpleNamespace(lens=SimpleNamespace(subhalo=subhalo))), samples=samples)


def _refinement(vector):
    return RefineOutcome(acceptance_status=RoleStatus.ACCEPTED, best_log_likelihood=-1.0, best_vector=vector,
                         gates={}, record={}, projected_gradient_linf=0.0, repeat_converged=True)


def test_recovery_separates_the_sampler_and_refined_estimates():
    """SCI-15: the sampler maximum (the historical values) and the refined best that gave q."""
    model = _subhalo_model()
    assert model.parameter_names == (
        "galaxies.lens.mass.einstein_radius", "galaxies.lens.mass.ell_comps.ell_comps_0",
        "galaxies.lens.subhalo.centre.centre_0", "galaxies.lens.subhalo.centre.centre_1",
        "galaxies.lens.subhalo.log10_m200", "galaxies.source.light.intensity")
    recovery = extract_recovery(_result(pdf_converged=True), _Mapping(), model,
                                _refinement((1.0, 0.1, 0.41, -0.79, 9.2, 2.0)))
    assert (recovery.sampler.log10_mass, recovery.sampler.centre_yx) == (8.9, (0.39, -0.82))
    assert recovery.sampler.profile_scales == {"concentration": 10.0 - 8.9, "kappa_s": 0.01 * 8.9}
    assert (recovery.sampler.margin_to_lower_dex, recovery.sampler.margin_to_upper_dex) == (8.9 - 6.0, 9.7 - 8.9)
    assert (recovery.refined.log10_mass, recovery.refined.centre_yx) == (9.2, (0.41, -0.79))
    assert recovery.refined.margin_to_upper_dex == 9.7 - 9.2
    midpoints = [0.125, 0.5, 0.875]
    assert recovery.log10_mass_quantiles == pytest.approx(
        (8.0 + 0.5 * (0.16 - 0.125) / 0.375, 8.5, 8.5 + 0.5 * (0.84 - 0.5) / 0.375), rel=1.0e-12)
    assert recovery.centre_y_quantiles == pytest.approx(tuple(np.interp([0.16, 0.5, 0.84], midpoints, [0.3, 0.4, 0.5])))
    assert recovery.centre_x_quantiles == pytest.approx(tuple(np.interp([0.16, 0.5, 0.84], midpoints,
                                                                        [-0.9, -0.8, -0.7])))
    assert (recovery.support, recovery.pdf_converged, recovery.sample_count) == (SUPPORT, True, 3)
    assert extract_recovery(_result(pdf_converged=True), _Mapping(), model, None).refined is None


def test_unconverged_posterior_reports_no_quantiles():
    recovery = extract_recovery(_result(pdf_converged=False), _Mapping(), _subhalo_model(), None)
    assert (recovery.log10_mass_quantiles, recovery.centre_y_quantiles, recovery.centre_x_quantiles) == (None,) * 3
    assert (recovery.pdf_converged, recovery.sample_count, recovery.sampler.log10_mass) == (False, 3, 8.9)
