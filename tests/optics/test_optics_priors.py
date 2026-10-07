"""Mode-weight priors: packaged tables, table files, power laws, and orthonormal direction draws."""

import copy
import hashlib
from pathlib import Path

import numpy as np
import pytest
import yaml

from hwoslaps.config.checks import ConfigError
from hwoslaps.optics.mode_priors import (
    ModeWeightPrior, ModeWeightPriorSpec, draw_combined_orthonormal, draw_global_orthonormal,
    draw_segment_orthonormal, load_prior, noll_radial_order, parse_prior, power_law_prior,
)

SEGMENTS = tuple(range(19))
PACKAGED_DIGESTS = {"jwst_wss_static_v1": "27eb04495c927409ceb3a775bab94ba295b2eb96a081832b1fea0ad7c8785a58",
                    "jwst_wss_drift_v1": "bfbececdcbe5fb37a4abcb018b63544d47c4c37ecc1900a539d62755b740c488"}


def _packaged(name):
    return ModeWeightPriorSpec("packaged", name, None, None)


def test_packaged_priors_load_by_name_with_their_digests(tmp_path):
    for name, digest in PACKAGED_DIGESTS.items():
        prior, loaded_digest = load_prior(_packaged(name))
        assert loaded_digest == digest and prior.name == name
        assert prior.metadata["basis_convention"] == "sequential_orthonormal_aperture"
    copied = tmp_path / "copy.yaml"
    prior, _ = load_prior(_packaged("jwst_wss_drift_v1"))
    copied.write_bytes(yaml.safe_dump({"name": prior.name, "segment_variance_fraction": 0.5,
                                       "global_weights": {4: 1.0}}).encode())
    assert load_prior(ModeWeightPriorSpec("path", None, copied, None))[1] == hashlib.sha256(
        copied.read_bytes()).hexdigest()
    with pytest.raises(FileNotFoundError, match="no packaged prior"):
        load_prior(_packaged("jwst_wss_static_v2"))
    with pytest.raises(FileNotFoundError, match="does not exist"):
        load_prior(ModeWeightPriorSpec("path", None, tmp_path / "missing.yaml", None))


@pytest.mark.parametrize("noll, order", [(1, 0), (2, 1), (3, 1), (4, 2), (6, 2), (7, 3), (10, 3), (11, 4),
                                         (15, 4), (16, 5), (21, 5), (22, 6), (28, 6), (29, 7), (36, 7), (37, 8),
                                         (45, 8), (46, 9), (55, 9)])
def test_noll_radial_order(noll, order):
    assert noll_radial_order(noll) == order


@pytest.mark.parametrize("noll", [0, -1, True, 1.0])
def test_noll_radial_order_refuses_what_is_not_a_noll_index(noll):
    with pytest.raises(ValueError, match="a Noll index is an integer >= 1"):
        noll_radial_order(noll)


@pytest.mark.parametrize("alpha", [0.0, 1.0, 2.0])
def test_power_law_weights_follow_radial_order_with_unit_norm_per_side(alpha):
    prior = power_law_prior(alpha, global_nolls=(4, 11), segment_nolls=(1, 6), segment_variance_fraction=0.5)
    assert np.linalg.norm(list(prior.global_weights.values())) == pytest.approx(1.0, rel=1e-15)
    assert np.linalg.norm(list(prior.segment_weights.values())) == pytest.approx(1.0, rel=1e-15)
    assert prior.global_weights[4] / prior.global_weights[11] == pytest.approx((2 / 4) ** (-alpha))
    assert prior.segment_weights[1] / prior.segment_weights[4] == pytest.approx((1 / 3) ** (-alpha))
    spec = parse_prior({"power_law": {"alpha": alpha, "global_nolls": [4, 11], "segment_nolls": [1, 6],
                                      "segment_variance_fraction": 0.5}}, "prior")
    from_configuration, digest = load_prior(spec)
    assert from_configuration == prior and len(digest) == 64


@pytest.mark.parametrize("arguments, message", [
    ({"alpha": -1.0}, "alpha"),
    ({"alpha": float("nan")}, "alpha"),
    ({"alpha": float("inf")}, "alpha"),
    ({"global_nolls": (1, 55)}, "global_nolls"),
    ({"global_nolls": (6, 4)}, "global_nolls"),
    ({"segment_nolls": (3, 2)}, "segment_nolls"),
    ({"global_nolls": None, "segment_nolls": None}, "must not both be None"),
    ({"segment_variance_fraction": 1.1}, "segment_variance_fraction"),
    ({"segment_variance_fraction": -0.1}, "segment_variance_fraction"),
])
def test_power_law_prior_rejects_invalid_arguments(arguments, message):
    keywords = {"alpha": 1.0, "global_nolls": (4, 55), "segment_nolls": (1, 10), "segment_variance_fraction": 0.5,
                **arguments}
    with pytest.raises(ValueError, match=message):
        power_law_prior(keywords.pop("alpha"), **keywords)


VALID_TABLE = {"name": "flight_prior", "segment_variance_fraction": 0.4, "global_weights": {4: 3.0, 5: 4.0},
               "segment_weights": {1: 5.0, 2: 12.0}, "metadata": {"source": "offline"}}


@pytest.mark.parametrize("document, global_expected, segment_expected", [
    (VALID_TABLE, {4: .6, 5: .8}, {1: 5 / 13, 2: 12 / 13}),
    ({"name": "idempotent", "segment_variance_fraction": .5, "global_weights": {4: 2., 5: 7.}},
     {4: 2 / np.sqrt(53), 5: 7 / np.sqrt(53)}, {}),
], ids=("both-sides-and-metadata", "original-global-only"))
def test_prior_file_load_normalizes_and_preserves_normalized_reload(tmp_path, document,
                                                                  global_expected, segment_expected, monkeypatch):
    first_path = tmp_path / "raw.yaml"
    first_path.write_text(yaml.safe_dump(document), encoding="utf-8")
    original = first_path.read_bytes()
    replacement = yaml.safe_dump({"name": "published_later", "segment_variance_fraction": 0.7,
                                  "global_weights": {4: 12.0, 5: 5.0},
                                  "metadata": {"source": "later_epoch"}}).encode()
    read_actual_bytes = Path.read_bytes
    published = []

    def publish_after_actual_read(path):
        content = read_actual_bytes(path)
        if path == first_path and not published:
            first_path.write_bytes(replacement)
            published.append(True)
        return content

    try:
        with monkeypatch.context() as scope:
            scope.setattr(Path, "read_bytes", publish_after_actual_read)
            first, captured_sha = load_prior(ModeWeightPriorSpec("path", None, first_path, None))
        assert published == [True] and first_path.read_bytes() == replacement
        assert captured_sha == hashlib.sha256(original).hexdigest()
    finally:
        first_path.write_bytes(original)
    assert dict(first.global_weights) == pytest.approx(global_expected, rel=1e-15, abs=0)
    assert dict(first.segment_weights) == pytest.approx(segment_expected, rel=1e-15, abs=0)
    assert first.name == document["name"]
    assert first.segment_variance_fraction == document["segment_variance_fraction"]
    assert dict(first.metadata) == document.get("metadata", {})
    expected_modes = tuple(sorted(global_expected))
    expected_values = np.random.default_rng(11).standard_normal(len(expected_modes)) * np.array(
        [global_expected[noll] for noll in expected_modes])
    expected_values *= 17.0 / np.linalg.norm(expected_values)
    actual_draw = draw_global_orthonormal(np.random.default_rng(11), first, 17.0)
    np.testing.assert_allclose([actual_draw[noll] for noll in expected_modes], expected_values,
                               rtol=1e-15, atol=0)
    normalized = {"name": first.name, "segment_variance_fraction": first.segment_variance_fraction,
                  "global_weights": dict(first.global_weights), "segment_weights": dict(first.segment_weights),
                  "metadata": dict(first.metadata)}
    second_path = tmp_path / "normalized.yaml"
    second_path.write_text(yaml.safe_dump(normalized), encoding="utf-8")
    second, _ = load_prior(ModeWeightPriorSpec("path", None, second_path, None))
    assert dict(second.global_weights) == pytest.approx(dict(first.global_weights), rel=1e-15, abs=0)
    assert dict(second.segment_weights) == pytest.approx(dict(first.segment_weights), rel=1e-15, abs=0)
    assert second.name == first.name and second.segment_variance_fraction == first.segment_variance_fraction
    assert dict(second.metadata) == dict(first.metadata)


@pytest.mark.parametrize("edit, message", [
    ({"unknown": 1}, "unknown"),
    ({"name": None}, "name"),
    ({"global_weights": None, "segment_weights": None}, "global_weights"),
    ({"global_weights": {4: -1.0}}, "global_weights"),
    ({"segment_weights": {1: 0.0, 2: 0.0}}, "segment_weights"),
    ({"global_weights": {"4": 1.0}}, "global_weights"),
    ({"segment_weights": {True: 1.0}}, "segment_weights"),
    ({"global_weights": {3: 1.0}}, "global_weights"),
    ({"segment_variance_fraction": 1.1}, "segment_variance_fraction"),
])
def test_prior_file_refuses_malformed_tables(tmp_path, edit, message):
    path = tmp_path / "prior.yaml"
    document = copy.deepcopy(VALID_TABLE)
    document.update(edit)
    document = {key: value for key, value in document.items() if value is not None}
    path.write_text(yaml.safe_dump(document), encoding="utf-8")
    with pytest.raises(ValueError, match=message):
        load_prior(ModeWeightPriorSpec("path", None, path, None))


@pytest.mark.parametrize("fields", [
    ("packaged", None, None, None),
    ("packaged", "jwst_wss_static_v1", Path("/abs/prior.yaml"), None),
    ("path", None, None, None),
    ("power_law", None, None, None),
    ("table", None, None, None),
])
def test_prior_specification_holds_exactly_its_own_source(fields):
    with pytest.raises(ValueError):
        ModeWeightPriorSpec(*fields)


def _flat(segment_side):
    return np.array([segment_side[s][n] for s in sorted(segment_side) for n in sorted(segment_side[s])])


def test_global_draw_is_ascending_with_the_exact_norm():
    prior = ModeWeightPrior("global", {8: 2.0, 4: 1.0, 6: 3.0}, {}, 0.0)
    draw = draw_global_orthonormal(np.random.default_rng(1), prior, 17.0)
    assert tuple(draw) == (4, 6, 8)
    assert np.linalg.norm(list(draw.values())) == pytest.approx(17.0, rel=1e-12)
    assert draw_global_orthonormal(np.random.default_rng(1), prior, 0.0) == {}
    assert draw == draw_global_orthonormal(np.random.default_rng(1), prior, 17.0)
    with pytest.raises(ValueError, match="no global weights"):
        draw_global_orthonormal(np.random.default_rng(1), ModeWeightPrior("s", {}, {1: 1.0}, 1.0), 10.0)


def test_segment_draw_norm_removes_only_the_common_piston():
    prior = ModeWeightPrior("segment", {}, {1: 1.0, 2: 0.7, 3: 0.4}, 1.0)
    draw = draw_segment_orthonormal(np.random.default_rng(19), SEGMENTS, prior, 13.0)
    matrix = np.array([[draw[s][n] for n in (1, 2, 3)] for s in SEGMENTS])
    assert np.linalg.norm(matrix) == pytest.approx(13.0 * np.sqrt(19), rel=1e-12)
    assert np.mean(matrix[:, 0]) == pytest.approx(0.0, abs=1e-12)
    assert abs(np.mean(matrix[:, 1])) > 1e-3 and abs(np.mean(matrix[:, 2])) > 1e-3
    assert draw_segment_orthonormal(np.random.default_rng(19), SEGMENTS, prior, 0.0) == {}
    with pytest.raises(ValueError, match="at least one segment"):
        draw_segment_orthonormal(np.random.default_rng(1), (), prior, 10.0)
    with pytest.raises(ValueError, match="at least two segments"):
        draw_segment_orthonormal(np.random.default_rng(1), (0,), prior, 10.0)


@pytest.mark.parametrize("fraction", [0.5, 0.2])
def test_combined_draw_splits_the_budget_segments_first(fraction):
    prior = ModeWeightPrior("combined", {4: 1.0, 5: 0.5}, {1: 1.0, 2: 0.5}, fraction)
    rng, reference = np.random.default_rng(81), np.random.default_rng(81)
    segment_side, global_side = draw_combined_orthonormal(rng, SEGMENTS, prior, 30.0)
    assert np.linalg.norm(_flat(segment_side)) == pytest.approx(30.0 * np.sqrt(fraction * 19), rel=1e-12)
    assert np.linalg.norm(list(global_side.values())) == pytest.approx(30.0 * np.sqrt(1 - fraction), rel=1e-12)
    first = draw_segment_orthonormal(reference, SEGMENTS, prior, 30.0 * np.sqrt(fraction))
    second = draw_global_orthonormal(reference, prior, 30.0 * np.sqrt(1 - fraction))
    assert segment_side == first and global_side == second
    assert rng.bit_generator.state == reference.bit_generator.state


@pytest.mark.parametrize("fraction", [0.0, 1.0])
def test_combined_draw_side_without_budget_consumes_no_random_numbers(fraction):
    prior = ModeWeightPrior("edge", {4: 1.0, 5: 0.5}, {1: 1.0, 2: 0.5}, fraction)
    rng, reference = np.random.default_rng(82), np.random.default_rng(82)
    segment_side, global_side = draw_combined_orthonormal(rng, SEGMENTS, prior, 30.0)
    if fraction == 0.0:
        assert segment_side == {} and global_side == draw_global_orthonormal(reference, prior, 30.0)
    else:
        assert global_side == {} and segment_side == draw_segment_orthonormal(reference, SEGMENTS, prior, 30.0)
    assert rng.bit_generator.state == reference.bit_generator.state
    untouched = np.random.default_rng(84)
    state = copy.deepcopy(untouched.bit_generator.state)
    assert draw_combined_orthonormal(untouched, SEGMENTS, prior, 0.0) == ({}, {})
    assert untouched.bit_generator.state == state


def test_draw_sample_variances_follow_the_squared_weights():
    prior = ModeWeightPrior("three", {4: 1.0, 5: 0.95, 6: 0.9}, {}, 0.0)
    rng = np.random.default_rng(934)
    draws = np.array([list(draw_global_orthonormal(rng, prior, 1.0).values()) for _ in range(4000)])
    expected = np.array(list(prior.global_weights.values())) ** 2
    # Exact-norm conditioning moves the variances away from the squared weights by up to 15%
    # (the carried tolerance); the 4000-draw sampling error is about 1.5%.
    np.testing.assert_allclose(np.var(draws, axis=0), expected, rtol=0.15, atol=0.0)


def test_exact_norm_conditioning_moves_the_drift_prior_variance_fractions():
    prior, _ = load_prior(_packaged("jwst_wss_drift_v1"))
    rng = np.random.default_rng(107)
    draws = np.array([list(draw_global_orthonormal(rng, prior, 1.0).values()) for _ in range(50_000)])
    # Unit-norm draws and unit-norm weights: both sets of variance fractions sum to one.
    realized = np.mean(draws ** 2, axis=0)
    naive = np.array(list(prior.global_weights.values())) ** 2
    # The 974cee9 band for the summed difference; 200000 draws give 0.0753 (seed 107) and 0.0748 (seed 108).
    assert 0.05 <= np.sum(np.abs(realized - naive)) <= 0.10


def test_prior_files_never_resolve_through_the_working_directory_or_the_repository(tmp_path, monkeypatch):
    relative = Path("configs/psf_priors/jwst_wss_drift_v1.yaml")
    (tmp_path / relative).parent.mkdir(parents=True)
    (tmp_path / relative).write_bytes(b"name: decoy\nsegment_variance_fraction: 0.5\nglobal_weights: {4: 1.0}\n")
    monkeypatch.chdir(tmp_path)
    with pytest.raises(ValueError, match="absolute Path"):
        ModeWeightPriorSpec("path", None, relative, None)
    with pytest.raises(ConfigError, match="not resolved") as caught:
        parse_prior({"path": str(relative)}, "psf.model.draw.prior")
    assert caught.value.path == "psf.model.draw.prior.path"
