"""Hand image statistics and preserved exact-pool ranking science from 974cee9."""

import hashlib
import math

import numpy as np
import pytest

from hwoslaps.analysis.selection import (
    Cut, RankingPolicy, ScoreTerm, aperture_mask, arc_snr, complexity, diffraction_scale_arcsec,
    electron_maps, gradient_power, rank, rank_pool, spearman_rank_correlation, standardize,
    top_k_jaccard, top_k_recovery,
)


def test_arc_snr_hand_values_and_rejections():
    variance = np.array([[9., 1.], [1., 4.]])
    for sign in (1, -1):
        assert arc_snr([[sign * 3., 0.], [0., 4.]], variance) == pytest.approx(math.sqrt(5))
    assert arc_snr([[3., 100.], [0., 4.]], variance,
                   np.array([[True, False], [True, True]])) == pytest.approx(math.sqrt(5))
    for signal, noise, mask in ((np.ones((2, 2)), np.ones((2, 3)), None),
                               ([[1, np.nan]], [[1, 1]], None), ([[1, 1]], [[1, 0]], None),
                               (np.ones((3, 3)), np.ones((3, 3)), np.zeros((3, 3), dtype=bool)),
                               (np.ones((3, 3)), np.ones((3, 3)), np.ones((3, 3))),
                               (np.ones((3, 3)), np.ones((3, 3)), np.ones((2, 2), dtype=bool))):
        with pytest.raises(ValueError):
            arc_snr(signal, noise, mask)


def test_gradient_power_hand_values_and_rejections():
    ramp = np.array([[0., 1., 2.]] * 3)
    for image in (ramp, ramp.T):
        assert gradient_power(image, np.ones((3, 3)), .5) == 36
        assert gradient_power(image, np.ones((3, 3)), .25) == 144
    both = ramp + ramp.T
    assert gradient_power(both, np.ones((3, 3)), .5) == 72
    mask = np.zeros((3, 3), dtype=bool)
    mask[1] = True
    assert gradient_power(ramp, np.full((3, 3), 4), .5, mask) == 3
    for image, noise, scale in ((np.ones((2, 4)), np.ones((2, 4)), .5),
                                (ramp, np.ones((3, 3)), 0), (ramp, np.ones((3, 3)), True),
                                (ramp, np.ones((2, 3)), .5), (ramp, np.zeros((3, 3)), .5)):
        with pytest.raises(ValueError):
            gradient_power(image, noise, scale)


def test_diffraction_scale_and_complexity():
    assert diffraction_scale_arcsec(6e-7, 6) == pytest.approx(1e-7 * 3600 * 180 / math.pi)
    ramp = np.array([[0., 1., 2.]] * 3)
    assert complexity(36, math.sqrt(15), .02) == pytest.approx(.02 ** 2 * 36 / 15)
    noise = np.full((3, 3), 6.)
    faint = complexity(gradient_power(ramp, noise, .5), arc_snr(ramp, noise), .02)
    bright = complexity(gradient_power(3.7 * ramp, noise, .5), arc_snr(3.7 * ramp, noise), .02)
    assert bright == pytest.approx(faint)
    for args in ((0., 10., .02), (36., 0., .02), (36., 10., 0.), (np.nan, 10., .02)):
        with pytest.raises(ValueError):
            complexity(*args)
    for wavelength, diameter in ((0, 6), (6e-7, -1), (np.inf, 6), (True, 6)):
        with pytest.raises(ValueError):
            diffraction_scale_arcsec(wavelength, diameter)


def test_standardize_is_exact_and_permutation_invariant():
    assert standardize([1, 2, 3]) == pytest.approx([-math.sqrt(1.5), 0, math.sqrt(1.5)])
    np.testing.assert_array_equal(standardize([7, 7, 7]), [0, 0, 0])
    for invalid in ([1, np.inf, 3], [], [[1, 2]]):
        with pytest.raises(ValueError):
            standardize(invalid)
    rng = np.random.default_rng(20260823)
    values = rng.uniform(3.5, 7., size=400)
    mean = math.fsum(float(value) for value in values) / 400
    spread = math.sqrt(math.fsum((float(value) - mean) ** 2 for value in values) / 400)
    expected = (values - mean) / spread
    np.testing.assert_array_equal(standardize(values), expected)
    for _ in range(32):
        order = rng.permutation(400)
        np.testing.assert_array_equal(standardize(values[order]), expected[order])


@pytest.mark.parametrize("operator,expected", [(">", (False, False, True)), (">=", (False, True, True)),
                                               ("<", (True, False, False)), ("<=", (True, True, False))])
def test_rank_pool_applies_each_cut_operator_at_the_threshold(operator, expected):
    policy = RankingPolicy((ScoreTerm("x", 1),), (Cut("x", operator, 2),))
    result = rank_pool(("a", "b", "c"), {"x": [1, 2, 3]}, policy)
    assert result.passed == expected and result.selected == ()
    assert result.policy == policy


def test_rank_pool_applies_the_policy():
    ids = ("a", "b", "c", "excluded")
    features = {"x": [1, 2, 3, 100], "y": [4, 1, 2, 1000], "keep": [1, 1, 1, 0]}
    policy = RankingPolicy((ScoreTerm("x", 2), ScoreTerm("y", -1, True)), (Cut("keep", ">", 0),), 2)
    result = rank_pool(ids, features, policy)
    logs = np.log([4., 1., 2.])
    expected = 2 * (np.array([1., 2., 3.]) - 2) / math.sqrt(2 / 3) - (logs - logs.mean()) / logs.std()
    assert result.ids == ids and result.survivors == ids[:3] and result.passed == (True, True, True, False)
    assert result.scores == pytest.approx(expected)
    expected_ranking = tuple(ids[i] for i in np.argsort(-expected))
    assert result.ranking == expected_ranking and result.selected == expected_ranking[:2]
    mapped = policy.to_mapping()
    assert RankingPolicy.from_mapping(mapped) == policy
    independent = rank_pool(("a", "b", "c"), {"x": [.1, .2, .3]}, RankingPolicy(
        (ScoreTerm("x", 1),), (Cut("x", ">", .1),), 2))
    assert independent.selected == ("c", "b")


@pytest.mark.parametrize("mutation", ["missing_feature", "bad_length", "nonfinite_extra", "duplicate_ids",
                                      "log_zero", "too_few", "unknown_key", "unknown_term_key", "unknown_cut_key",
                                      "empty_terms", "zero_weight", "bool_weight", "bad_cut", "bool_threshold", "bad_select"])
def test_rank_pool_refuses_invalid_policy_or_pool(mutation):
    ids, features = ("a", "b", "c"), {"x": [1, 2, 3]}
    mapping = {"terms": [{"feature": "x", "weight": 1, "log": True}],
               "cuts": [{"feature": "x", "operator": ">", "threshold": 0}], "select": 2}
    if mutation == "missing_feature":
        features = {"y": [1, 2, 3]}
    elif mutation == "bad_length":
        features["x"] = [1, 2]
    elif mutation == "nonfinite_extra":
        features["unused"] = [1, np.nan, 3]
    elif mutation == "duplicate_ids":
        ids = ("a", "a", "c")
    elif mutation == "log_zero":
        features["x"] = [0, 2, 3]
        mapping["cuts"] = []
    elif mutation == "too_few":
        mapping["cuts"][0]["threshold"] = 2
    elif mutation == "unknown_key":
        mapping["unknown"] = 1
    elif mutation == "unknown_term_key":
        mapping["terms"][0]["unknown"] = 1
    elif mutation == "unknown_cut_key":
        mapping["cuts"][0]["unknown"] = 1
    elif mutation == "empty_terms":
        mapping["terms"] = []
    elif mutation == "zero_weight":
        mapping["terms"][0]["weight"] = 0
    elif mutation == "bool_weight":
        mapping["terms"][0]["weight"] = True
    elif mutation == "bad_cut":
        mapping["cuts"][0]["operator"] = "=="
    elif mutation == "bool_threshold":
        mapping["cuts"][0]["threshold"] = True
    else:
        mapping["select"] = True
    with pytest.raises(ValueError):
        rank_pool(ids, features, RankingPolicy.from_mapping(mapping))


def test_rank_pool_reproduces_the_41621de_rule_as_data():
    ids = ("m0", "m1", "m2", "m3", "m4", "m5")
    features = {"theta_E": [1., .5, 1.2, .9, 1.5, .8], "S": [100., 300., 20., 50., 400., 200.],
                "C": [.002, .009, .009, .005, .001, .004]}
    policy = RankingPolicy((ScoreTerm("S", 1, True), ScoreTerm("C", 1, True)),
                           (Cut("theta_E", ">", .5), Cut("S", ">", 20)), 3)
    result = rank_pool(ids, features, policy)
    assert result.passed == (True, False, False, True, True, True)
    assert result.survivors == ("m0", "m3", "m4", "m5")
    expected = np.log([100., 50., 400., 200.])
    morphology = np.log([.002, .005, .001, .004])
    assert result.scores == pytest.approx((expected - expected.mean()) / expected.std() +
                                           (morphology - morphology.mean()) / morphology.std())
    assert result.ranking == ("m5", "m4", "m3", "m0") and result.selected == ("m5", "m4", "m3")


def test_ranking_is_deterministic_under_ties_and_permutations():
    ids = ("gamma", "alpha", "beta")
    expected = tuple(sorted(ids, key=lambda name: hashlib.sha256(name.encode()).hexdigest()))
    for descending in (True, False):
        assert rank(ids, [0, 0, 0], descending=descending) == expected
    assert rank(("s1", "s2", "s3"), [8.5, 7.1, 9.], descending=False) == ("s2", "s1", "s3")
    for bad_ids, keys in ((("a", "a"), [1, 2]), (("", "b"), [1, 2]), (("a", "b"), [1, np.nan])):
        with pytest.raises(ValueError):
            rank(bad_ids, keys, descending=True)
    rng = np.random.default_rng(20260823)
    logs = rng.uniform(3.5, 7., size=400)
    ids = tuple(f"id{i:04d}" for i in range(400))
    features = {"S": np.exp(logs), "C": np.exp(logs[::-1])}
    policy = RankingPolicy((ScoreTerm("S", 1, True), ScoreTerm("C", 1, True)), select=12)
    forward = rank_pool(ids, features, policy)
    assert len(set(forward.scores)) == 200
    scores = np.asarray(forward.scores)
    for _ in range(32):
        order = rng.permutation(400)
        permuted = rank_pool(tuple(ids[i] for i in order), {name: values[order] for name, values in features.items()}, policy)
        np.testing.assert_array_equal(permuted.scores, scores[order])
        assert permuted.ranking == forward.ranking and permuted.selected == forward.selected


def test_rank_agreement_metrics():
    assert spearman_rank_correlation([1, 2, 3], [10, 20, 30]) == pytest.approx(1)
    assert spearman_rank_correlation([1, 2, 3], [30, 20, 10]) == pytest.approx(-1)
    assert spearman_rank_correlation([1, 2, 3, 4], [2, 1, 4, 3]) == pytest.approx(.6)
    assert spearman_rank_correlation([1, 2, 2, 3], [1, 2, 3, 4]) == pytest.approx(4.5 / math.sqrt(4.5 * 5))
    for first, second in (([1, 1, 1], [1, 2, 3]), ([1, 2, 3], [1, 2]), ([1], [1]), ([1, np.nan], [1, 2])):
        with pytest.raises(ValueError):
            spearman_rank_correlation(first, second)
    first, second = ("a", "b", "c", "d"), ("c", "d", "e", "f")
    assert top_k_jaccard(first, second, 2) == 0
    assert top_k_jaccard(first, second, 4) == pytest.approx(1 / 3)
    assert top_k_recovery(first, second, 4) == .5
    assert top_k_recovery(first, second, 2) == 0
    assert top_k_jaccard(first, first, 4) == top_k_recovery(first, first, 3) == 1
    for k in (0, -1, 5, 2., True):
        for function in (top_k_jaccard, top_k_recovery):
            with pytest.raises(ValueError):
                function(first, second, k)


@pytest.fixture
def selection_observation(minimal_mapping):
    from hwoslaps.fisher.api import prepare_forecast

    minimal_mapping["scene"]["lens"]["light"] = {"light": {
        "type": "Exponential", "centre": [0., 0.], "ell_comps": [.1, .02],
        "intensity": .5, "effective_radius": .4}}
    minimal_mapping["instrument"]["detector"].update(gain_e_per_adu=2.5, read_noise_e=2., dark_current_e_per_s=.003)
    minimal_mapping["observation"].update(exposure_time_s=17., exposure_count=3)
    with prepare_forecast(minimal_mapping) as prepared:
        yield prepared.observation


@pytest.mark.backend
def test_electron_maps_equal_detector_moments(selection_observation):
    observation = selection_observation
    maps = electron_maps(observation)
    source = observation.light_rate_by_plane_e_per_s["source"]
    lens = observation.light_rate_by_plane_e_per_s["lens"]
    assert np.any(lens > 0) and np.any(observation.light_rate_e_per_s != source)
    assert maps.signal_e.tobytes() == (source * 17.).tobytes()
    expected = np.maximum(observation.light_rate_e_per_s * 17., 0) + (1. + .003) * 17. + 3 * 2. ** 2
    np.testing.assert_allclose(maps.variance_e2, expected, rtol=1e-12)
    assert maps.variance_e2.tobytes() == ((observation.noise_map_adu * 2.5) ** 2).tobytes()
    assert maps.pixel_scale_arcsec == .05
    noisy_maps = electron_maps(observation.draw(71))
    np.testing.assert_array_equal(noisy_maps.signal_e, maps.signal_e)
    np.testing.assert_array_equal(noisy_maps.variance_e2, maps.variance_e2)


@pytest.mark.backend
def test_aperture_mask_is_a_closed_disc_about_the_centre(selection_observation):
    observation = selection_observation
    expected = np.zeros((41, 41), dtype=bool)
    expected[20, 20] = expected[19, 20] = expected[21, 20] = expected[20, 19] = expected[20, 21] = True
    np.testing.assert_array_equal(aperture_mask(observation, centre_yx=(0, 0), radius_arcsec=.05), expected)
    np.testing.assert_array_equal(aperture_mask(observation, centre_yx=(0, 0), radius_arcsec=.07), expected)
    diagonal = np.zeros((41, 41), dtype=bool)
    diagonal[19:22, 19:22] = True
    np.testing.assert_array_equal(aperture_mask(observation, centre_yx=(0, 0),
                                               radius_arcsec=math.sqrt(2) * .05), diagonal)
    offcentre = np.zeros((41, 41), dtype=bool)
    offcentre[18, 18] = True
    np.testing.assert_array_equal(aperture_mask(observation, centre_yx=(.1, -.1), radius_arcsec=.01), offcentre)
    with pytest.raises(ValueError, match="no pixels"):
        aperture_mask(observation, centre_yx=(5, 5), radius_arcsec=.01)
    with pytest.raises(ValueError, match="radius_arcsec"):
        aperture_mask(observation, centre_yx=(0, 0), radius_arcsec=True)
