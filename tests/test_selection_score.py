"""Hand-calculable tests for the frozen D-F4 selection statistics.

Every deterministic rule of
``scratch/q1_observing_conditions/selection_rule.md`` (v2, 2026-08-23) is
pinned here: the observable units and their pixel-scale normalization,
the standardization convention, the zero and non-finite handling, the
tie handling, and the rank-stability metrics.
"""

from __future__ import annotations

import hashlib
import importlib.util
import math
from pathlib import Path
import sys

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from hwoslaps.analysis import selection_score as ss


RAMP_PIXEL_SCALE = 0.5
"""Pixel scale of the hand-calculable ramp image, arcsec per pixel."""


def _ramp_image():
    """Return the 3x3 ramp used for the gradient hand calculations.

    Pixel values equal the column index, so the image is linear in ``x``
    and flat in ``y`` and every difference stencil is exact.
    """
    return np.array([[0.0, 1.0, 2.0]] * 3)


def _zscore(values):
    """Standardize independently of the module under test."""
    array = np.asarray(values, dtype=float)
    return (array - array.mean()) / array.std()


def test_blank_variance_matches_the_declared_noise_chain():
    """``B = (sky + dark) * t + read_noise ** 2`` exactly."""
    assert ss.blank_variance_e2(1.0e-3, 1.0e-3, 2.0, 1000.0) == pytest.approx(6.0)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"sky_background_e_s": -1.0e-3},
        {"dark_current_e_s": float("nan")},
        {"read_noise_e": -0.1},
        {"exposure_time_s": 0.0},
    ],
)
def test_blank_variance_rejects_unphysical_inputs(kwargs):
    """Negative, non-finite and zero-exposure inputs raise."""
    call = {
        "sky_background_e_s": 1.0e-3,
        "dark_current_e_s": 1.0e-3,
        "read_noise_e": 2.0,
        "exposure_time_s": 1000.0,
    }
    call.update(kwargs)
    with pytest.raises(ValueError):
        ss.blank_variance_e2(**call)


def test_expected_variance_adds_source_shot_noise_to_the_floor():
    """``sigma^2 = s + B`` with no noise realization anywhere."""
    variance = ss.expected_variance_e2([[0.0, 4.0], [9.0, 1.0]], 6.0)
    assert variance.tolist() == [[6.0, 10.0], [15.0, 7.0]]


def test_expected_variance_rejects_negative_source_electrons():
    """A negative expected source map is not an electron map."""
    with pytest.raises(ValueError, match="non-negative"):
        ss.expected_variance_e2([[0.0, -1.0e-9], [1.0, 1.0]], 6.0)


def test_arc_snr_is_the_quadrature_sum_of_pixel_signal_to_noise():
    """``S = sqrt(sum s^2 / sigma^2)``, hand value ``sqrt(5)``."""
    signal = np.array([[3.0, 0.0], [0.0, 4.0]])
    variance = np.array([[9.0, 1.0], [1.0, 4.0]])
    assert ss.arc_snr(signal, variance) == pytest.approx(math.sqrt(5.0))


def test_arc_snr_matches_the_committed_derivation_convention():
    """The module and the reference derivation share one estimator."""
    pytest.importorskip("scipy")
    script_path = PROJECT_ROOT / "scripts" / "derive_hwo_eac1_hri_reference.py"
    spec = importlib.util.spec_from_file_location("derive_hwo_reference", script_path)
    derivation = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(derivation)

    rng = np.random.default_rng(20260823)
    electrons = rng.uniform(0.0, 500.0, size=(9, 9))
    blank_variance = 9.10056
    expected = derivation.integrated_source_snr(electrons, blank_variance)
    measured = ss.arc_snr(
        electrons, ss.expected_variance_e2(electrons, blank_variance)
    )
    assert measured == pytest.approx(expected, rel=0.0, abs=0.0)


def test_arc_snr_sums_only_the_aperture():
    """Masked pixels contribute nothing to the integrated statistic."""
    signal = np.array([[3.0, 100.0], [0.0, 4.0]])
    variance = np.array([[9.0, 1.0], [1.0, 4.0]])
    mask = np.array([[True, False], [True, True]])
    assert ss.arc_snr(signal, variance, mask=mask) == pytest.approx(math.sqrt(5.0))


def test_arc_snr_admits_negative_signal_on_the_noisy_path():
    """A background-subtracted realization may go negative."""
    signal = np.array([[-3.0, 0.0], [0.0, 4.0]])
    variance = np.array([[9.0, 1.0], [1.0, 4.0]])
    assert ss.arc_snr(signal, variance) == pytest.approx(math.sqrt(5.0))


def test_arc_snr_rejects_a_shape_mismatch():
    """Signal and variance must describe the same pixels."""
    with pytest.raises(ValueError, match="does not match"):
        ss.arc_snr(np.ones((2, 2)), np.ones((2, 3)))


def test_arc_snr_rejects_a_non_positive_variance():
    """A zero-variance pixel would make the statistic infinite."""
    with pytest.raises(ValueError, match="positive"):
        ss.arc_snr(np.ones((2, 2)), np.array([[1.0, 0.0], [1.0, 1.0]]))


def test_arc_snr_rejects_non_finite_signal():
    """Non-finite entries fail loudly instead of propagating."""
    with pytest.raises(ValueError, match="finite"):
        ss.arc_snr(np.array([[1.0, np.nan], [1.0, 1.0]]), np.ones((2, 2)))


@pytest.mark.parametrize(
    "mask",
    [
        np.zeros((3, 3), dtype=bool),
        np.ones((3, 3), dtype=float),
        np.ones((2, 2), dtype=bool),
    ],
)
def test_arc_snr_rejects_a_malformed_aperture(mask):
    """An empty, non-boolean or mis-shaped aperture raises."""
    with pytest.raises(ValueError):
        ss.arc_snr(np.ones((3, 3)), np.ones((3, 3)), mask=mask)


def test_gradient_power_is_an_angular_derivative():
    """A ramp of one electron per pixel at 0.5 arcsec gives ``G = 36``.

    The gradient is ``1 / 0.5 = 2`` electrons per arcsecond in ``x`` and
    zero in ``y`` at every one of the nine pixels, and the unit variance
    leaves ``G = 9 * 2 ** 2``.
    """
    image = _ramp_image()
    measured = ss.gradient_power(image, np.ones_like(image), RAMP_PIXEL_SCALE)
    assert measured == pytest.approx(36.0)


def test_gradient_power_scales_as_the_inverse_square_pixel_scale():
    """Halving the pixel scale quadruples ``G`` for fixed pixel values."""
    image = _ramp_image()
    variance = np.ones_like(image)
    coarse = ss.gradient_power(image, variance, RAMP_PIXEL_SCALE)
    fine = ss.gradient_power(image, variance, 0.5 * RAMP_PIXEL_SCALE)
    assert fine == pytest.approx(4.0 * coarse)


def test_gradient_power_is_noise_weighted_over_the_aperture():
    """Each pixel enters as ``|grad s|^2 / sigma^2`` inside the mask."""
    image = _ramp_image()
    variance = np.full_like(image, 4.0)
    mask = np.zeros_like(image, dtype=bool)
    mask[1, :] = True
    assert ss.gradient_power(image, variance, RAMP_PIXEL_SCALE, mask=mask) == pytest.approx(3.0)


def test_gradient_power_uses_both_axes():
    """The transpose of the ramp returns the same power."""
    image = _ramp_image().T
    measured = ss.gradient_power(image, np.ones_like(image), RAMP_PIXEL_SCALE)
    assert measured == pytest.approx(36.0)


def test_gradient_power_rejects_a_thin_image():
    """A central difference needs three pixels on both axes."""
    with pytest.raises(ValueError, match="three pixels"):
        ss.gradient_power(np.ones((2, 4)), np.ones((2, 4)), RAMP_PIXEL_SCALE)


def test_gradient_power_rejects_a_non_positive_pixel_scale():
    """The angular normalization must be a real pixel scale."""
    with pytest.raises(ValueError, match="pixel_scale_arcsec"):
        ss.gradient_power(np.ones((3, 3)), np.ones((3, 3)), 0.0)


def test_diffraction_scale_converts_radians_to_arcseconds():
    """``theta_res = lambda / D`` in arcseconds."""
    measured = ss.diffraction_scale_arcsec(6.0e-7, 6.0)
    assert measured == pytest.approx(1.0e-7 * 3600.0 * 180.0 / math.pi)


def test_complexity_is_the_hand_calculable_ratio():
    """``C = theta_res^2 G / S^2`` on the ramp image."""
    image = _ramp_image()
    variance = np.ones_like(image)
    power = ss.gradient_power(image, variance, RAMP_PIXEL_SCALE)
    snr = ss.arc_snr(image, variance)
    assert power == pytest.approx(36.0)
    assert snr ** 2 == pytest.approx(15.0)
    assert ss.complexity(power, snr, 0.02) == pytest.approx(0.02 ** 2 * 36.0 / 15.0)


def test_complexity_is_invariant_under_a_brightness_rescaling():
    """The statistic removes the brightness ``S`` and ``G`` both carry.

    In the background-dominated limit the variance does not follow the
    source, so scaling the arc by ``alpha`` scales ``S`` by ``alpha`` and
    ``G`` by ``alpha ** 2`` and ``C`` is unchanged. That is exactly the
    double counting the earlier ``z(log S) + z(log G)`` score carried.
    """
    image = _ramp_image()
    variance = np.full_like(image, 6.0)
    alpha = 3.7
    faint = ss.complexity(
        ss.gradient_power(image, variance, RAMP_PIXEL_SCALE),
        ss.arc_snr(image, variance),
        0.02,
    )
    bright = ss.complexity(
        ss.gradient_power(alpha * image, variance, RAMP_PIXEL_SCALE),
        ss.arc_snr(alpha * image, variance),
        0.02,
    )
    assert bright == pytest.approx(faint)


@pytest.mark.parametrize(
    "args",
    [(0.0, 10.0, 0.02), (36.0, 0.0, 0.02), (36.0, 10.0, 0.0), (float("nan"), 10.0, 0.02)],
)
def test_complexity_rejects_zero_and_non_finite_inputs(args):
    """A flat or undefined arc has no complexity."""
    with pytest.raises(ValueError):
        ss.complexity(*args)


def test_aperture_mask_is_a_closed_disc():
    """Pixels exactly on the radius are inside the aperture."""
    y_arcsec, x_arcsec = np.meshgrid([1.0, 0.0, -1.0], [-1.0, 0.0, 1.0], indexing="ij")
    assert int(np.count_nonzero(ss.aperture_mask(y_arcsec, x_arcsec, 1.0))) == 5
    assert int(np.count_nonzero(ss.aperture_mask(y_arcsec, x_arcsec, 1.4))) == 5
    assert int(np.count_nonzero(ss.aperture_mask(y_arcsec, x_arcsec, math.sqrt(2.0)))) == 9


def test_aperture_mask_honours_the_declared_centre():
    """An off-centre aperture moves with its centre."""
    y_arcsec, x_arcsec = np.meshgrid([1.0, 0.0, -1.0], [-1.0, 0.0, 1.0], indexing="ij")
    mask = ss.aperture_mask(y_arcsec, x_arcsec, 0.5, centre_arcsec=(1.0, -1.0))
    assert mask.tolist() == [[True, False, False], [False] * 3, [False] * 3]


def test_aperture_mask_rejects_an_empty_aperture():
    """An aperture holding no pixels is a configuration error."""
    y_arcsec, x_arcsec = np.meshgrid([1.0, 0.0, -1.0], [-1.0, 0.0, 1.0], indexing="ij")
    with pytest.raises(ValueError, match="no pixels"):
        ss.aperture_mask(y_arcsec, x_arcsec, 0.1, centre_arcsec=(5.0, 5.0))


def test_floor_cuts_are_strict_at_both_thresholds():
    """A member exactly on ``0.5`` arcsec or ``S = 20`` fails."""
    passed = ss.apply_floor_cuts(
        [0.5, 0.500001, 1.0, 1.0],
        [100.0, 100.0, 20.0, 20.000001],
    )
    assert passed.tolist() == [False, True, False, True]


def test_floor_cut_thresholds_are_the_collett_values():
    """The frozen floors are the cited Collett 2015 / Euclid values."""
    assert ss.FLOOR_THETA_E_ARCSEC == 0.5
    assert ss.FLOOR_ARC_SNR == 20.0


def test_floor_cuts_reject_a_length_mismatch():
    """Both statistics must describe the same pool."""
    with pytest.raises(ValueError, match="entries"):
        ss.apply_floor_cuts([1.0, 1.0], [100.0])


def test_standardize_uses_the_population_spread():
    """``z`` divides by the ``ddof = 0`` standard deviation."""
    z = ss.standardize([1.0, 2.0, 3.0])
    assert z == pytest.approx([-math.sqrt(1.5), 0.0, math.sqrt(1.5)])
    assert float(np.mean(z)) == pytest.approx(0.0)
    assert float(np.std(z)) == pytest.approx(1.0)


def test_standardize_maps_a_zero_spread_pool_to_zeros():
    """Identical members cannot rank above one another."""
    assert ss.standardize([7.0, 7.0, 7.0]).tolist() == [0.0, 0.0, 0.0]


def test_standardize_rejects_non_finite_input():
    """A non-finite statistic never reaches the score."""
    with pytest.raises(ValueError, match="finite"):
        ss.standardize([1.0, np.inf, 3.0])


def test_selection_scores_are_the_sum_of_two_standardized_logs():
    """``s_plus_c`` cancels exactly on a mirrored pool."""
    snr = np.exp([1.0, 2.0, 3.0])
    complexity_values = np.exp([3.0, 2.0, 1.0])
    s_only = ss.selection_scores(snr, complexity_values, variant="s_only")
    s_plus_c = ss.selection_scores(snr, complexity_values, variant="s_plus_c")
    assert s_only == pytest.approx([-math.sqrt(1.5), 0.0, math.sqrt(1.5)])
    assert s_plus_c == pytest.approx([0.0, 0.0, 0.0])


def test_selection_scores_standardize_over_the_pool_they_are_given():
    """Dropping a member changes the scores of the rest."""
    snr = np.exp([1.0, 2.0, 3.0, 10.0])
    complexity_values = np.exp([1.0, 2.0, 3.0, 4.0])
    full = ss.selection_scores(snr, complexity_values)
    trimmed = ss.selection_scores(snr[:3], complexity_values[:3])
    assert not np.allclose(full[:3], trimmed)


@pytest.mark.parametrize("bad", [0.0, -1.0, float("nan"), float("inf")])
def test_selection_scores_reject_an_undefined_logarithm(bad):
    """Zero or non-finite ``S`` or ``C`` rejects the pool loudly."""
    snr = np.array([10.0, 20.0, 30.0])
    complexity_values = np.array([1.0, 2.0, 3.0])
    with pytest.raises(ValueError, match="strictly positive"):
        ss.selection_scores(np.array([bad, 20.0, 30.0]), complexity_values)
    with pytest.raises(ValueError, match="strictly positive"):
        ss.selection_scores(snr, np.array([1.0, bad, 3.0]))


def test_selection_scores_reject_an_unknown_variant():
    """Only the pre-registered variants exist."""
    with pytest.raises(ValueError, match="variant"):
        ss.selection_scores([10.0, 20.0], [1.0, 2.0], variant="s_plus_g")


def test_ranking_breaks_ties_on_the_system_id_digest():
    """Equal scores order by ascending sha256 hex digest."""
    ids = ("gamma", "alpha", "beta")
    expected = tuple(
        sorted(ids, key=lambda name: hashlib.sha256(name.encode("utf-8")).hexdigest())
    )
    assert ss.rank_by_score(ids, [0.0, 0.0, 0.0]) == expected


def test_ranking_is_independent_of_the_input_order():
    """A re-ordered pool produces the identical ranking."""
    ids = ("s1", "s2", "s3", "s4")
    scores = [1.0, 3.0, 3.0, 2.0]
    forward = ss.rank_by_score(ids, scores)
    reverse = ss.rank_by_score(ids[::-1], scores[::-1])
    assert forward == reverse
    assert forward[-1] == "s1"


def test_ranking_rejects_duplicate_ids():
    """A pool cannot carry the same system twice."""
    with pytest.raises(ValueError, match="unique"):
        ss.rank_by_score(("s1", "s1"), [1.0, 2.0])


def test_ranking_rejects_a_non_finite_score():
    """A non-finite score has no place in the ordering."""
    with pytest.raises(ValueError, match="finite"):
        ss.rank_by_score(("s1", "s2"), [1.0, np.nan])


def test_sensitivity_ranking_puts_the_lowest_detectable_mass_first():
    """The oracle curve is ascending in ``log10(M_lim)``."""
    ranking = ss.rank_by_sensitivity(("s1", "s2", "s3"), [8.5, 7.1, 9.0])
    assert ranking == ("s2", "s1", "s3")


def test_sensitivity_ranking_breaks_ties_on_the_digest():
    """Equal measured masses order by ascending digest."""
    ids = ("gamma", "alpha", "beta")
    expected = tuple(
        sorted(ids, key=lambda name: hashlib.sha256(name.encode("utf-8")).hexdigest())
    )
    assert ss.rank_by_sensitivity(ids, [8.0, 8.0, 8.0]) == expected


def test_rank_pool_runs_the_whole_frozen_rule():
    """Cuts, standardization over survivors, score, ranking, tiers."""
    ids = ("m0", "m1", "m2", "m3", "m4", "m5")
    theta_e = [1.0, 0.5, 1.2, 0.9, 1.5, 0.8]
    snr = [100.0, 300.0, 20.0, 50.0, 400.0, 200.0]
    complexity_values = [2.0e-3, 9.0e-3, 9.0e-3, 5.0e-3, 1.0e-3, 4.0e-3]

    result = ss.rank_pool(
        ids, theta_e, snr, complexity_values, selected_size=3, golden_size=2
    )

    assert result.passed_floor == (True, False, False, True, True, True)
    assert result.survivor_ids == ("m0", "m3", "m4", "m5")

    survivor_snr = np.array([100.0, 50.0, 400.0, 200.0])
    survivor_complexity = np.array([2.0e-3, 5.0e-3, 1.0e-3, 4.0e-3])
    expected = _zscore(np.log(survivor_snr)) + _zscore(np.log(survivor_complexity))
    assert result.scores == pytest.approx(expected.tolist())

    order = [result.survivor_ids[index] for index in np.argsort(-expected, kind="stable")]
    assert list(result.ranking) == order
    assert result.selected_ids == result.ranking[:3]
    assert result.golden_ids == result.ranking[:2]
    assert result.variant == "s_plus_c"


def test_rank_pool_refuses_a_pool_too_small_for_the_selected_tier():
    """Fail closed rather than return a short tier."""
    with pytest.raises(ValueError, match="too few"):
        ss.rank_pool(
            ("m0", "m1"),
            [1.0, 0.4],
            [100.0, 100.0],
            [1.0e-3, 1.0e-3],
            selected_size=2,
            golden_size=1,
        )


def test_rank_pool_refuses_a_golden_tier_larger_than_the_selected_tier():
    """The goldens are drawn from the selected tier."""
    with pytest.raises(ValueError, match="golden_size"):
        ss.rank_pool(
            ("m0", "m1"),
            [1.0, 1.0],
            [100.0, 100.0],
            [1.0e-3, 2.0e-3],
            selected_size=1,
            golden_size=2,
        )


def test_rank_pool_defaults_are_the_frozen_tier_sizes():
    """Twelve selected, five golden."""
    assert ss.SELECTED_TIER_SIZE == 12
    assert ss.GOLDEN_TIER_SIZE == 5
    assert ss.APERTURE_THETA_E_MULTIPLE == 2.0


def test_ranking_positions_index_from_zero():
    """Position zero is the best-ranked member."""
    assert ss.ranking_positions(("s2", "s1", "s3")) == {"s2": 0, "s1": 1, "s3": 2}


def test_spearman_is_one_for_a_monotone_pair():
    """A perfectly concordant pair correlates at unity."""
    assert ss.spearman_rank_correlation([1.0, 2.0, 3.0], [10.0, 20.0, 30.0]) == pytest.approx(1.0)
    assert ss.spearman_rank_correlation([1.0, 2.0, 3.0], [30.0, 20.0, 10.0]) == pytest.approx(-1.0)


def test_spearman_matches_the_textbook_hand_value():
    """``rho = 1 - 6 sum d^2 / (n (n^2 - 1))`` with no ties."""
    rho = ss.spearman_rank_correlation([1.0, 2.0, 3.0, 4.0], [2.0, 1.0, 4.0, 3.0])
    assert rho == pytest.approx(1.0 - 6.0 * 4.0 / (4.0 * 15.0))


def test_spearman_shares_average_ranks_between_ties():
    """Tied entries take the mean of the ranks they span."""
    rho = ss.spearman_rank_correlation([1.0, 2.0, 2.0, 3.0], [1.0, 2.0, 3.0, 4.0])
    assert rho == pytest.approx(4.5 / math.sqrt(4.5 * 5.0))


def test_spearman_rejects_a_fully_tied_vector():
    """A constant vector leaves the correlation undefined."""
    with pytest.raises(ValueError, match="undefined"):
        ss.spearman_rank_correlation([1.0, 1.0, 1.0], [1.0, 2.0, 3.0])


def test_spearman_rejects_a_length_mismatch():
    """Paired observations must be paired."""
    with pytest.raises(ValueError, match="entries"):
        ss.spearman_rank_correlation([1.0, 2.0, 3.0], [1.0, 2.0])


def test_top_k_jaccard_is_intersection_over_union():
    """Disjoint tiers score zero, overlapping tiers score the share."""
    first = ("a", "b", "c", "d")
    second = ("c", "d", "e", "f")
    assert ss.top_k_jaccard(first, second, 2) == pytest.approx(0.0)
    assert ss.top_k_jaccard(first, second, 4) == pytest.approx(1.0 / 3.0)
    assert ss.top_k_jaccard(first, first, 4) == pytest.approx(1.0)


def test_oracle_recovered_fraction_counts_the_oracle_tier():
    """The share of the oracle top-``k`` the ranking recovers."""
    ranking = ("a", "b", "c", "d")
    oracle = ("c", "d", "e", "f")
    assert ss.oracle_recovered_fraction(ranking, oracle, 4) == pytest.approx(0.5)
    assert ss.oracle_recovered_fraction(ranking, oracle, 2) == pytest.approx(0.0)
    assert ss.oracle_recovered_fraction(oracle, oracle, 3) == pytest.approx(1.0)


@pytest.mark.parametrize("k", [0, -1, 5, 2.0, True])
def test_stability_metrics_reject_a_malformed_tier_size(k):
    """``k`` is a positive integer no larger than the rankings."""
    ranking = ("a", "b", "c", "d")
    with pytest.raises(ValueError):
        ss.top_k_jaccard(ranking, ranking, k)
    with pytest.raises(ValueError):
        ss.oracle_recovered_fraction(ranking, ranking, k)


def test_analysis_package_exports_the_selection_api():
    """The lazy package attribute resolves the new module."""
    import hwoslaps.analysis as analysis

    assert analysis.FLOOR_ARC_SNR == 20.0
    assert analysis.arc_snr is ss.arc_snr
    assert "rank_pool" in dir(analysis)
